"""Background training worker and in-memory progress hub (streamed to clients via SSE).

Training runs on a single worker thread so the API stays responsive. Every progress
event originates from the real training loop; nothing is simulated.
"""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
import traceback

import numpy as np
import torch

from ..config import settings
from ..core.eegnet import ArchitectureConfig, EEGNet
from ..core.preprocessing import LearnedTransform, PreprocessingConfig
from ..core.splits import validation_split
from ..core.tracing import run_trace
from ..core.training import Cancelled, FoldTensors, TrainingConfig, device_label, resolve_device, train_fold
from ..lifecycle import State
from ..schemas import ArtifactRecord, FoldRun, now_iso
from . import repo

S = State
TERMINAL = {"done", "failed", "cancelled"}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ProgressHub:
    def __init__(self) -> None:
        self._events: dict[str, list[dict]] = {}
        self._lock = threading.Lock()

    def publish(self, run_id: str, event: dict) -> None:
        with self._lock:
            lst = self._events.setdefault(run_id, [])
            event = {"seq": len(lst) + 1, "ts": time.time(), "run_id": run_id, **event}
            lst.append(event)

    def since(self, run_id: str, seq: int) -> list[dict]:
        with self._lock:
            return list(self._events.get(run_id, [])[seq:])

    def has(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._events


hub = ProgressHub()


class JobManager:
    def __init__(self) -> None:
        self._q: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self._cancel: dict[str, threading.Event] = {}
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.current: str | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, name="training-worker", daemon=True)
                self._thread.start()

    def submit(self, exp_id: str, run_id: str) -> None:
        self._cancel[run_id] = threading.Event()
        hub.publish(run_id, {"type": "status", "status": "queued", "message": "Run queued."})
        self.start()
        self._q.put((exp_id, run_id))

    def cancel(self, run_id: str) -> bool:
        ev = self._cancel.get(run_id)
        if ev is None:
            return False
        ev.set()
        return True

    def wait_idle(self, timeout: float = 600) -> bool:
        """Block until the queue is empty and no run is executing (used by tests)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._q.unfinished_tasks == 0:
                return True
            time.sleep(0.05)
        return False

    def _loop(self) -> None:
        while True:
            exp_id, run_id = self._q.get()
            try:
                self.current = run_id
                execute_run(exp_id, run_id, self._cancel.get(run_id) or threading.Event())
            except Exception:  # pragma: no cover - last-resort guard
                traceback.print_exc()
            finally:
                self.current = None
                self._q.task_done()


manager = JobManager()


def recover_orphans() -> None:
    """On start-up, runs left 'queued'/'training' by a previous process cannot resume: mark them failed."""
    for run in repo.list_runs():
        if run.status in ("queued", "training"):
            run.status, run.error, run.finished_at = "failed", "Server restarted while the run was in progress.", now_iso()
            repo.save_run(run)
            try:
                exp = repo.get_experiment(run.experiment_id)
                if exp.active_run_id == run.id and exp.state in (S.TRAINING, S.TRAINING_QUEUED):
                    repo.transition(exp, S.TRAINING_FAILED, "server restarted during training")
            except repo.NotFound:
                pass
    for exp in repo.list_experiments():
        if exp.state == S.EVALUATION_RUNNING:
            repo.transition(exp, S.ARTIFACTS_READY, "server restarted during evaluation")


def run_dir(exp_id: str, run_id: str):
    p = settings.data_dir / "experiments" / exp_id / "runs" / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def register_artifact(exp_id: str, run_id: str, typ: str, path, *, checkpoint_id: str | None = None,
                      config_fp: str | None = None, source_ref: str | None = None, meta: dict | None = None,
                      art_id: str | None = None) -> ArtifactRecord:
    rel = path.relative_to(settings.data_dir).as_posix()
    rec = ArtifactRecord(id=art_id or repo.new_id(typ.split("_")[0][:6]), type=typ, experiment_id=exp_id,
                         run_id=run_id, checkpoint_id=checkpoint_id, created_at=now_iso(), source_ref=source_ref,
                         config_fingerprint=config_fp, storage_ref=rel,
                         sha256=sha256_file(path) if path.is_file() else None,
                         size_bytes=path.stat().st_size if path.is_file() else None, meta=meta or {})
    repo.save_artifact(rec)
    return rec


def _fold_data(snap: dict, es, assignment, trials_by_id, fold_idx: int, tcfg: TrainingConfig) -> dict:
    """Partition *candidate trials* (blocks / events / epochs) of one fold; rows follow their parent trial."""
    kept = es.kept_parents()
    fold = assignment.folds[fold_idx]
    train_kept = [i for i in fold.train if i in kept]
    test_kept = [i for i in fold.test if i in kept]
    group_level = assignment.strategy == "group" or (
        assignment.strategy == "kfold" and assignment.config.kfold_mode in ("group", "stratified_group"))
    fit_ids, val_ids, val_desc = validation_split(train_kept, trials_by_id, tcfg.validation_mode,
                                                  tcfg.validation_fraction, tcfg.seed + fold_idx, group_level)
    if set(val_ids) & set(test_kept) or set(fit_ids) & set(test_kept):
        raise RuntimeError("Leakage guard: validation/training subsets intersect the test partition.")
    if group_level:
        held_out = {trials_by_id[i].group for i in test_kept}
        mixed = sorted({trials_by_id[i].group for i in fit_ids + val_ids} & held_out)
        if mixed:
            raise RuntimeError(f"Leakage guard: held-out participant(s) {mixed} appear in the training data.")
    return {"fit": fit_ids, "val": val_ids, "test": test_kept, "val_desc": val_desc,
            "rejected_train": len(fold.train) - len(train_kept), "rejected_test": len(fold.test) - len(test_kept)}


def execute_run(exp_id: str, run_id: str, cancel: threading.Event) -> None:
    from . import artifacts
    from .experiments import epochset, load_ica_reference, load_split, pool_trials

    run = repo.get_run(run_id)
    exp = repo.get_experiment(exp_id)
    log = run.log

    def emit(ev: dict) -> None:
        hub.publish(run_id, ev)

    def note(msg: str) -> None:
        log.append(f"{time.strftime('%H:%M:%S')} {msg}")
        emit({"type": "log", "message": msg})

    if cancel.is_set():
        run.status, run.finished_at, run.error = "cancelled", now_iso(), "Cancelled before start."
        repo.save_run(run)
        repo.transition(exp, S.TRAINING_CANCELLED, "cancelled while queued")
        emit({"type": "cancelled", "status": "cancelled", "message": "Cancelled before training started."})
        return
    try:
        snap = run.snapshot
        tcfg = TrainingConfig(**snap["training_config"])
        arch = ArchitectureConfig(**snap["architecture_config"])
        pcfg = PreprocessingConfig(**snap["preprocessing_config"])
        device = resolve_device(tcfg.device)
        run.status, run.started_at, run.device = "training", now_iso(), device_label(device)
        repo.save_run(run)
        repo.transition(exp, S.TRAINING, f"run {run_id} started on {run.device}")
        emit({"type": "status", "status": "training", "device": run.device, "message": f"Training on {run.device}."})

        es, _, det_fp = epochset(pcfg, exp)
        if det_fp != snap["preprocessing_det_fingerprint"]:
            raise RuntimeError("Preprocessed data no longer match the run snapshot.")
        assignment = load_split(exp, snap["split_fingerprint"])
        trials, _ = pool_trials(exp)
        by_id = {t.trial_id: t for t in trials}
        ref_ica = load_ica_reference(exp) if (pcfg.ica.enabled and pcfg.ica.exclude) else None
        class_names = snap["class_names"]
        if list(es.class_names) != list(class_names):
            raise RuntimeError("Class mapping changed since the run was configured.")
        y_all = es.y
        rd = run_dir(exp_id, run_id)
        n_folds = len(assignment.folds)
        run.folds = [FoldRun(fold=k) for k in range(n_folds)]
        repo.save_run(run)
        note(f"{n_folds} fold(s); {len(es.X)} epochs of shape [1, {es.X.shape[1]}, {es.X.shape[2]}]; "
             f"seed {tcfg.seed}.")

        for k in range(n_folds):
            fr = run.folds[k]
            fr.status = "training"
            parts = _fold_data(snap, es, assignment, by_id, k, tcfg)
            fi = es.rows_for(parts["fit"])
            vi = es.rows_for(parts["val"])
            ti = es.rows_for(parts["test"])
            fr.fit_ids_count, fr.val_ids_count, fr.test_ids_count = len(fi), len(vi), len(ti)
            fr.val_description = parts["val_desc"]
            missing_cls = sorted(set(range(len(class_names))) - set(y_all[fi].tolist()))
            if missing_cls:
                raise RuntimeError(f"Fold {k + 1}: training subset lacks class(es) "
                                   f"{[class_names[c] for c in missing_cls]}.")
            windows = len(fi) + len(vi) + len(ti) != len(parts["fit"]) + len(parts["val"]) + len(parts["test"])
            unit = f" windows from {len(parts['fit'])}/{len(parts['val'])}/{len(parts['test'])} trials" if windows else ""
            note(f"Fold {k + 1}/{n_folds}: {len(fi)} training, {len(vi)} validation, {len(ti)} test (held out)"
                 f"{unit}. {parts['val_desc']}")
            lt = LearnedTransform(pcfg, es.ch_names, es.sfreq)
            fr.transform_report = lt.fit(es.X, reference_ica=ref_ica, fit_ids=parts["fit"], rows=fi)
            data = FoldTensors(X_fit=lt.transform(es.X, fi), y_fit=y_all[fi],
                               X_val=lt.transform(es.X, vi) if vi else np.zeros((0, *es.X.shape[1:]), np.float32),
                               y_val=y_all[vi])
            repo.save_run(run)
            fd = rd / f"fold{k}"
            fd.mkdir(parents=True, exist_ok=True)
            (fd / "partition.json").write_text(json.dumps(
                {"fit": parts["fit"], "val": parts["val"], "test": parts["test"],
                 "val_description": parts["val_desc"], "unit": "candidate trials",
                 "rows": {"fit": len(fi), "val": len(vi), "test": len(ti)}}))

            def on_epoch(row: dict, k=k, fr=fr) -> None:
                fr.history.append(row)
                repo.save_run(run)
                emit({"type": "epoch", "fold": k, "n_folds": n_folds, "epochs": tcfg.epochs, **row})

            model, history, info = train_fold(arch, tcfg, data, len(class_names), device, on_epoch, cancel, note)
            fr.info = info
            ckpt_id = repo.new_id("ckpt")
            path = fd / "checkpoint.pt"
            torch.save({
                "checkpoint_id": ckpt_id, "run_id": run_id, "experiment_id": exp_id, "fold": k,
                "state_dict": {kk: v.detach().cpu() for kk, v in model.state_dict().items()},
                "architecture": arch.model_dump(), "architecture_fingerprint": snap["architecture_fingerprint"],
                "class_names": list(class_names), "ch_names": list(es.ch_names), "sfreq": float(es.sfreq),
                "tmin": float(es.tmin), "input_convention": snap["input_convention"],
                "selected_epoch": int(info["selected_epoch"]), "seed": int(tcfg.seed),
                "torch_version": str(torch.__version__),
            }, path)
            register_artifact(exp_id, run_id, "checkpoint", path, checkpoint_id=ckpt_id, art_id=ckpt_id,
                              config_fp=run.fingerprint, source_ref=f"fold {k + 1}",
                              meta={"fold": k, "selected_epoch": info["selected_epoch"]})
            fr.checkpoint_id = ckpt_id
            lt.save(fd / "transform")
            register_artifact(exp_id, run_id, "learned_transform", fd / "transform" / "transform.json",
                              checkpoint_id=ckpt_id, config_fp=run.fingerprint, source_ref=f"fold {k + 1}",
                              meta={"fit_trials": len(fi), "method": pcfg.normalization.method})
            # representative activations (hybrid strategy: a few trials persisted, the rest re-extracted on demand)
            rep_rows, seen = [], set()
            for i in es.rows_for(parts["val"] or parts["fit"]):  # one row per representative trial
                if str(es.parents[i]) not in seen:
                    seen.add(str(es.parents[i]))
                    rep_rows.append(i)
                if len(rep_rows) >= tcfg.representative_trials:
                    break
            rep_ids = [str(es.trial_ids[i]) for i in rep_rows]
            rep_x = lt.transform(es.X[rep_rows])
            tr = run_trace(model.cpu().eval(), rep_x)
            arrays = {f"out__{name}": arr for name, arr in tr.outputs.items()}
            arrays["input"] = rep_x
            arrays["logits"] = tr.logits
            np.savez_compressed(fd / "activations.npz", **arrays)
            fr.activations = {"trial_ids": rep_ids, "layers": list(tr.outputs),
                              "shapes": {n: list(a.shape[1:]) for n, a in tr.outputs.items()},
                              "checkpoint_id": ckpt_id, "mode": "eval", "dtype": "float32",
                              "extraction": "forward hooks on the saved checkpoint, eval mode"}
            register_artifact(exp_id, run_id, "activations", fd / "activations.npz", checkpoint_id=ckpt_id,
                              config_fp=run.fingerprint, source_ref=",".join(rep_ids),
                              meta=fr.activations)
            fr.status = "completed"
            repo.save_run(run)
            emit({"type": "fold_done", "fold": k, "n_folds": n_folds, "selected_epoch": info["selected_epoch"],
                  "checkpoint_id": ckpt_id})

        run.status, run.finished_at = "completed", now_iso()
        (rd / "manifest.json").write_text(run.model_dump_json(indent=1))
        repo.save_run(run)
        exp = repo.get_experiment(exp_id)
        repo.transition(exp, S.TRAINING_COMPLETED, f"run {run_id} completed")
        emit({"type": "status", "status": "verifying", "message": "Training finished; verifying artifacts."})
        status = artifacts.verify_and_update(exp_id, run_id)
        emit({"type": "done", "status": "completed", "artifacts_ready": status["ready"],
              "message": "Artifacts verified." if status["ready"] else "Artifacts incomplete."})
    except Cancelled:
        run.status, run.finished_at, run.error = "cancelled", now_iso(), "Cancelled by user."
        for f in run.folds:
            if f.status == "training":
                f.status = "cancelled"
        repo.save_run(run)
        exp = repo.get_experiment(exp_id)
        repo.transition(exp, S.TRAINING_CANCELLED, "cancelled by user")
        emit({"type": "cancelled", "status": "cancelled", "message": "Training cancelled; no post-training tools "
                                                                     "were unlocked."})
    except Exception as exc:
        run.status, run.finished_at = "failed", now_iso()
        run.error = f"{type(exc).__name__}: {exc}"
        for f in run.folds:
            if f.status == "training":
                f.status = "failed"
        repo.save_run(run)
        exp = repo.get_experiment(exp_id)
        try:
            repo.transition(exp, S.TRAINING_FAILED, run.error)
        except Exception:
            traceback.print_exc()
        emit({"type": "failed", "status": "failed", "message": run.error})
