"""Artifact verification (the training-completion gate) and checkpoint bundle loading.

Post-training tools are unlocked only when ``verify_run`` passes every check: a
terminal training state, intact checkpoints (sha256), recorded architecture / data
provenance / history / seed, intermediate-result metadata, a strict checkpoint
reload and a forward pass whose hooked activations reproduce the persisted ones.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..config import settings
from ..core.eegnet import ArchitectureConfig, EEGNet, compute_architecture, config_fingerprint
from ..core.preprocessing import EpochSet, LearnedTransform, PreprocessingConfig
from ..core.tracing import run_trace
from ..lifecycle import State
from ..schemas import ExperimentRecord, RunRecord, now_iso
from . import repo

S = State


class ArtifactError(RuntimeError):
    pass


@dataclass
class Bundle:
    exp_id: str
    run_id: str
    fold: int
    checkpoint_id: str
    sha256: str
    model: EEGNet
    arch: ArchitectureConfig
    transform: LearnedTransform
    es: EpochSet
    class_names: list[str]
    partition: dict
    selected_epoch: int

    def model_input(self, trial_ids: list[str]) -> np.ndarray:
        idx = self.es.index_of()
        missing = [t for t in trial_ids if t not in idx]
        if missing:
            raise KeyError(f"Trial(s) not in the preprocessed data (rejected or unknown): {missing}")
        return self.transform.transform(self.es.X[[idx[t] for t in trial_ids]])

    def parent_of(self, trial_id: str) -> str:
        i = self._index.get(trial_id)
        return str(self.es.parents[i]) if i is not None else trial_id

    def partition_of(self, trial_id: str) -> str:
        parent = self.parent_of(trial_id)
        for part in ("fit", "val", "test"):
            if parent in self._sets[part]:
                return {"fit": "train", "val": "validation", "test": "test"}[part]
        return "unused"

    def rows_of(self, part: str) -> list[int]:
        return self.es.rows_for(self.partition.get(part, []))

    def __post_init__(self) -> None:
        self._sets = {k: set(v) for k, v in self.partition.items() if k in ("fit", "val", "test")}
        self._index = self.es.index_of()


_cache: "OrderedDict[tuple, Bundle]" = OrderedDict()
_lock = threading.Lock()


def _fold_dir(run: RunRecord, fold: int) -> Path:
    return settings.data_dir / "experiments" / run.experiment_id / "runs" / run.id / f"fold{fold}"


def load_epochset_for_run(run: RunRecord) -> EpochSet:
    det = run.snapshot["preprocessing_det_fingerprint"]
    path = settings.data_dir / "cache" / "preprocessed" / f"{det}.npz"
    if path.exists():
        return EpochSet.load(path)
    from .experiments import epochset

    exp = repo.get_experiment(run.experiment_id)
    es, _, fp = epochset(PreprocessingConfig(**run.snapshot["preprocessing_config"]), exp)
    if fp != det:
        raise ArtifactError("Re-computed preprocessed data do not match the run's recorded fingerprint.")
    return es


def load_bundle(run: RunRecord, fold: int = 0) -> Bundle:
    if fold < 0 or fold >= len(run.folds) or not run.folds[fold].checkpoint_id:
        raise ArtifactError(f"No checkpoint for fold {fold + 1}.")
    ckpt_id = run.folds[fold].checkpoint_id
    art = repo.get_artifact(ckpt_id)
    path = settings.data_dir / art.storage_ref
    if not path.exists():
        raise ArtifactError(f"Checkpoint file for {ckpt_id} is missing.")
    from .jobs import sha256_file

    sha = sha256_file(path)
    if sha != art.sha256:
        raise ArtifactError(f"Checkpoint {ckpt_id} failed its integrity check (sha256 mismatch).")
    key = (ckpt_id, sha)
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("checkpoint_id") != ckpt_id or payload.get("run_id") != run.id:
        raise ArtifactError("Checkpoint metadata does not match the run record.")
    arch = ArchitectureConfig(**payload["architecture"])
    model = EEGNet(arch)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    fd = _fold_dir(run, fold)
    transform = LearnedTransform.load(fd / "transform")
    partition = json.loads((fd / "partition.json").read_text())
    b = Bundle(exp_id=run.experiment_id, run_id=run.id, fold=fold, checkpoint_id=ckpt_id, sha256=sha, model=model,
               arch=arch, transform=transform, es=load_epochset_for_run(run), class_names=payload["class_names"],
               partition=partition, selected_epoch=int(payload.get("selected_epoch", 0)))
    with _lock:
        _cache[key] = b
        while len(_cache) > 6:
            _cache.popitem(last=False)
    return b


def invalidate(run_id: str) -> None:
    with _lock:
        for k in [k for k, b in _cache.items() if b.run_id == run_id]:
            _cache.pop(k, None)


def verify_run(exp: ExperimentRecord, run: RunRecord) -> dict:
    checks: list[dict] = []

    def add(cid: str, label: str, ok: bool, detail: str, recovery: str | None = None) -> bool:
        checks.append({"id": cid, "label": label, "ok": bool(ok), "detail": detail,
                       "recovery": None if ok else recovery})
        return bool(ok)

    snap = run.snapshot
    add("terminal_state", "Training reached a valid terminal state", run.status == "completed",
        f"run status: {run.status}" + (f" ({run.error})" if run.error else ""), "Retrain the model.")
    # architecture
    arch_ok = False
    try:
        arch = ArchitectureConfig(**snap["architecture_config"])
        arch_ok = compute_architecture(arch).valid and config_fingerprint(arch) == snap["architecture_fingerprint"]
        add("architecture", "Model architecture and hyperparameters recorded", arch_ok,
            f"architecture {snap['architecture_fingerprint']} (version v{snap.get('architecture_version')}), "
            f"training config recorded", "Retrain the model.")
    except Exception as exc:
        add("architecture", "Model architecture and hyperparameters recorded", False, str(exc), "Retrain the model.")
    # data provenance
    prov = []
    split_path = settings.data_dir / "experiments" / exp.id / f"split_{snap.get('split_fingerprint')}.json"
    if not snap.get("dataset_fingerprints"):
        prov.append("dataset fingerprints missing")
    if not snap.get("preprocessing_fingerprint"):
        prov.append("preprocessing configuration missing")
    if not split_path.exists():
        prov.append("split assignment file missing")
    try:
        cur = [repo.get_dataset(i).fingerprint for i in snap["dataset_ids"]]
        if cur != snap["dataset_fingerprints"]:
            prov.append("dataset content changed since training")
    except repo.NotFound:
        prov.append("dataset record deleted")
    add("provenance", "Dataset, preprocessing pipeline and split assignments recorded", not prov,
        "; ".join(prov) or f"datasets {', '.join(f[:8] for f in snap['dataset_fingerprints'])}, preprocessing "
                            f"{snap['preprocessing_fingerprint']}, split {snap['split_fingerprint']}",
        "Restore the dataset/split files or retrain.")
    hist_ok = bool(run.folds) and all(f.history for f in run.folds) and snap.get("seed") is not None
    add("history", "Training history and random seed recorded", hist_ok,
        f"{sum(len(f.history) for f in run.folds)} epoch records over {len(run.folds)} fold(s), seed {snap.get('seed')}",
        "Retrain the model.")
    # per-fold files
    from .jobs import sha256_file

    ck_problems, meta_problems = [], []
    for f in run.folds:
        if not f.checkpoint_id:
            ck_problems.append(f"fold {f.fold + 1}: no checkpoint")
            continue
        try:
            art = repo.get_artifact(f.checkpoint_id)
            p = settings.data_dir / art.storage_ref
            if not p.exists():
                ck_problems.append(f"fold {f.fold + 1}: checkpoint file missing")
            elif sha256_file(p) != art.sha256:
                ck_problems.append(f"fold {f.fold + 1}: checkpoint corrupted (sha256 mismatch)")
        except repo.NotFound:
            ck_problems.append(f"fold {f.fold + 1}: checkpoint record missing")
        fd = _fold_dir(run, f.fold)
        if not f.activations or not (fd / "activations.npz").exists():
            meta_problems.append(f"fold {f.fold + 1}: activation metadata/file missing")
        if not (fd / "transform" / "transform.json").exists() or not (fd / "partition.json").exists():
            meta_problems.append(f"fold {f.fold + 1}: learned-transform or partition file missing")
    add("checkpoint", "Valid model checkpoint saved", not ck_problems and bool(run.folds),
        "; ".join(ck_problems) or f"{len(run.folds)} checkpoint(s) with matching sha256", "Retrain the model.")
    add("intermediates", "Intermediate-result metadata available", not meta_problems,
        "; ".join(meta_problems) or "representative activations, partitions and learned transforms persisted",
        "Retrain the model.")
    # load + forward pass
    load_ok = fwd_ok = False
    if not ck_problems and arch_ok:
        try:
            invalidate(run.id)
            bundles = [load_bundle(run, f.fold) for f in run.folds]
            load_ok = True
            add("checkpoint_loads", "Checkpoint loads successfully", True,
                "state_dict loaded with strict=True into the recorded architecture")
            diffs = []
            for b in bundles:
                f = run.folds[b.fold]
                with np.load(_fold_dir(run, b.fold) / "activations.npz") as z:
                    saved_in, saved = z["input"], {k[5:]: z[k] for k in z.files if k.startswith("out__")}
                x = b.model_input(f.activations["trial_ids"])
                diffs.append(float(np.abs(x - saved_in).max()))
                tr = run_trace(b.model, x)
                spec = {s.name: s.out_shape for s in compute_architecture(b.arch).layers}
                for name, arr in tr.outputs.items():
                    if list(arr.shape[1:]) != spec[name]:
                        raise ArtifactError(f"layer {name} shape {list(arr.shape[1:])} != {spec[name]}")
                    diffs.append(float(np.abs(arr - saved[name]).max()))
            fwd_ok = max(diffs) < 1e-3
            add("forward_pass", "Forward pass and activation extraction reproduce persisted results", fwd_ok,
                f"all {len(bundles[0].model.layer_names)} layers extracted; max |difference| vs persisted "
                f"activations = {max(diffs):.2e}", "Retrain the model.")
        except Exception as exc:
            if not load_ok:
                add("checkpoint_loads", "Checkpoint loads successfully", False, str(exc), "Retrain the model.")
            add("forward_pass", "Forward pass and activation extraction reproduce persisted results", False, str(exc),
                "Retrain the model.")
    else:
        add("checkpoint_loads", "Checkpoint loads successfully", False, "skipped: checkpoint or architecture invalid",
            "Retrain the model.")
        add("forward_pass", "Forward pass and activation extraction reproduce persisted results", False,
            "skipped: checkpoint could not be loaded", "Retrain the model.")
    ready = all(c["ok"] for c in checks)
    return {"ready": ready, "checks": checks, "verified_at": now_iso(), "run_id": run.id}


def verify_and_update(exp_id: str, run_id: str) -> dict:
    exp = repo.get_experiment(exp_id)
    run = repo.get_run(run_id)
    status = verify_run(exp, run)
    run.artifact_status = status
    repo.save_run(run)
    exp = repo.get_experiment(exp_id)
    if exp.active_run_id != run_id:
        return status
    if exp.state == S.TRAINING_COMPLETED:
        repo.transition(exp, S.ARTIFACTS_READY if status["ready"] else S.ARTIFACTS_INCOMPLETE, "artifact verification")
    elif not status["ready"] and exp.state in (S.ARTIFACTS_READY, S.EVALUATION_COMPLETED):
        repo.transition(exp, S.ARTIFACTS_INCOMPLETE, "re-verification failed")
    elif status["ready"] and exp.state == S.ARTIFACTS_INCOMPLETE:
        repo.transition(exp, S.ARTIFACTS_READY, "re-verification passed")
    return status


def manifest(exp_id: str, run_id: str | None = None) -> list[dict]:
    return [{"id": a.id, "type": a.type, "experiment_id": a.experiment_id, "run_id": a.run_id,
             "checkpoint_id": a.checkpoint_id, "created_at": a.created_at, "source_ref": a.source_ref,
             "config_fingerprint": a.config_fingerprint, "storage_ref": a.storage_ref, "sha256": a.sha256,
             "size_bytes": a.size_bytes, "meta": a.meta}
            for a in repo.list_artifacts(experiment_id=exp_id, run_id=run_id)]
