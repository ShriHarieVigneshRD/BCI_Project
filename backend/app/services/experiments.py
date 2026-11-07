"""Experiment orchestration: configuration stages, lifecycle transitions, invalidation and capabilities."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

from ..config import settings
from ..core.eegnet import ArchitectureConfig, config_fingerprint, preset_config, validate_architecture
from ..core.preprocessing import (EpochSet, LearnedTransform, PreprocessingConfig, PreprocessingError,
                                  deterministic_fingerprint, fit_ica, full_fingerprint, ica_diagnostics,
                                  ica_reference_fingerprint, iter_rows, preview_subset, run_deterministic,
                                  validate_config)
from ..core.splits import SplitAssignment, SplitConfig, SplitError, integrity_report, make_split
from ..core.training import TrainingConfig, validate_training_config
from ..data.store import CandidateTrial, Dataset, candidate_trials
from ..lifecycle import BUSY, LEARNED_READY, RANK, State, TransitionError
from ..schemas import ArchitectureVersion, ExperimentRecord, RunRecord, now_iso
from . import datasets as dsvc
from . import repo

S = State


class WorkflowError(Exception):
    """A request that conflicts with the experiment's lifecycle state (HTTP 409)."""


class ConfigError(ValueError):
    """An invalid configuration (HTTP 422)."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


_det_lock = threading.Lock()


def exp_dir(exp_id: str) -> Path:
    p = settings.data_dir / "experiments" / exp_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Data pool
# ---------------------------------------------------------------------------


def sources_for(exp: ExperimentRecord) -> list[tuple[str, Dataset]]:
    if not exp.dataset_id:
        return []
    if exp.workflow == "separate":
        if not exp.test_dataset_id:
            return []
        return [("train/", dsvc.load(exp.dataset_id)), ("test/", dsvc.load(exp.test_dataset_id))]
    return [("", dsvc.load(exp.dataset_id))]


def dataset_fps(exp: ExperimentRecord) -> list[str]:
    ids = [exp.dataset_id] + ([exp.test_dataset_id] if exp.workflow == "separate" else [])
    return [repo.get_dataset(i).fingerprint for i in ids if i]


def pool_trials(exp: ExperimentRecord) -> tuple[list[CandidateTrial], list[str] | None]:
    trials: list[CandidateTrial] = []
    sources: list[str] = []
    for prefix, ds in sources_for(exp):
        for t in candidate_trials(ds):
            t = CandidateTrial(**{**t.__dict__, "trial_id": prefix + t.trial_id})
            trials.append(t)
            sources.append("test" if prefix == "test/" else "train")
    return trials, (sources if exp.workflow == "separate" else None)


def load_split(exp: ExperimentRecord, fingerprint: str | None = None) -> SplitAssignment:
    fp = fingerprint or exp.split_fingerprint
    if not fp:
        raise WorkflowError("No split has been configured.")
    path = exp_dir(exp.id) / f"split_{fp}.json"
    if not path.exists():
        raise WorkflowError(f"Split assignment file for {fp} is missing.")
    return SplitAssignment.model_validate_json(path.read_text())


# Live previews of large continuous datasets process this many recordings (spread over participants).
PREVIEW_MAX_RECORDINGS = 40


def epochset(cfg: PreprocessingConfig, exp: ExperimentRecord, preview: bool = False) -> tuple[EpochSet, dict, str]:
    """Deterministic preprocessing output, cached on disk by content fingerprint.

    Epochs are streamed to a raw float32 file and memory-mapped, so the dataset never has to fit in RAM.
    ``preview``: the full result if it is already cached, else a participant-balanced subset of recordings
    (fast, labelled in ``report['scope']``); Run & validate always processes everything."""
    det_fp = deterministic_fingerprint(cfg, dataset_fps(exp))
    cache = settings.data_dir / "cache" / "preprocessed"
    cache.mkdir(parents=True, exist_ok=True)
    full = (cache / f"{det_fp}.npz", cache / f"{det_fp}.json")
    sub = (cache / f"{det_fp}.preview.npz", cache / f"{det_fp}.preview.json")
    with _det_lock:
        for npz, rep in ((full, sub) if preview else (full,)):
            if npz.exists() and rep.exists():
                return EpochSet.load(npz), json.loads(rep.read_text()), det_fp
        srcs = sources_for(exp)
        npz, rep = full
        limit = None
        if preview and preview_subset(srcs, PREVIEW_MAX_RECORDINGS) is not None:
            npz, rep, limit = *sub, PREVIEW_MAX_RECORDINGS
        es, report = run_deterministic(cfg, srcs, out=EpochSet.x_path(npz), max_recordings=limit)
        es.save(npz)
        rep.write_text(json.dumps(report, default=str))
    return es, report, det_fp


# ---------------------------------------------------------------------------
# Capabilities (the frontend derives every available action from this)
# ---------------------------------------------------------------------------

NEXT_STEP = {
    S.CREATED: ("dataset", "Attach or upload a dataset."),
    S.DATASET_REQUIRED: ("dataset", "Attach or upload a dataset."),
    S.DATASET_READY: ("dataset", "Configure and confirm the evaluation split."),
    S.SPLIT_CONFIGURED: ("preprocessing", "Configure the preprocessing pipeline."),
    S.PREPROCESSING_CONFIGURED: ("preprocessing", "Run and validate the preprocessing pipeline."),
    S.PREPROCESSING_COMPLETED: ("architecture/config", "Configure and validate the EEGNet architecture."),
    S.ARCHITECTURE_CONFIGURED: ("architecture/training", "Review training settings and start training."),
    S.TRAINING_QUEUED: ("architecture/training", "Training is queued."),
    S.TRAINING: ("architecture/training", "Training is running."),
    S.TRAINING_FAILED: ("architecture/training", "Training failed. Review the error, adjust settings and retry."),
    S.TRAINING_CANCELLED: ("architecture/training", "Training was cancelled. Start a new run when ready."),
    S.TRAINING_COMPLETED: ("architecture/training", "Verifying persisted artifacts."),
    S.ARTIFACTS_INCOMPLETE: ("architecture/training", "Artifacts are incomplete. Re-verify or retrain."),
    S.ARTIFACTS_READY: ("architecture/results", "Evaluate the trained model on the test partition."),
    S.EVALUATION_RUNNING: ("architecture/results", "Evaluation is running."),
    S.EVALUATION_COMPLETED: ("architecture/results", "Explore results, kernels and comparisons."),
}


def active_run(exp: ExperimentRecord) -> RunRecord | None:
    if not exp.active_run_id:
        return None
    try:
        return repo.get_run(exp.active_run_id)
    except repo.NotFound:
        return None


def capabilities(exp: ExperimentRecord, run: RunRecord | None = None) -> dict:
    run = run if run is not None else active_run(exp)
    st = exp.state
    rank = RANK[st]
    busy = st in BUSY
    trainable = True
    if exp.dataset_id:
        try:
            trainable = repo.get_dataset(exp.dataset_id).trainable
        except repo.NotFound:
            trainable = False
    artifacts_ok = bool(run and run.artifact_status and run.artifact_status.get("ready") and not run.stale)
    learned = st in LEARNED_READY and artifacts_ok
    page, msg = NEXT_STEP[st]
    if st == S.TRAINING_FAILED and run and run.error:
        msg = f"Training failed: {run.error}"
    if st == S.ARTIFACTS_INCOMPLETE and run and run.artifact_status:
        missing = [c["label"] for c in run.artifact_status.get("checks", []) if not c["ok"]]
        msg = "Artifacts incomplete: " + "; ".join(missing)
    if st == S.DATASET_READY and not trainable:
        msg = "The attached dataset has no usable labelled trials/events; attach another dataset."

    def stage(sid: str, label: str, done: bool, available: bool) -> dict:
        return {"id": sid, "label": label, "status": "complete" if done else ("available" if available else "locked")}

    stages = [
        stage("dataset", "Dataset & Splits", rank >= RANK[S.SPLIT_CONFIGURED], True),
        stage("preprocessing", "Preprocessing Studio", rank >= RANK[S.PREPROCESSING_COMPLETED],
              rank >= RANK[S.SPLIT_CONFIGURED]),
        stage("architecture", "Architecture Explorer", rank >= RANK[S.ARCHITECTURE_CONFIGURED],
              rank >= RANK[S.PREPROCESSING_COMPLETED]),
        stage("training", "Training", st in LEARNED_READY or st in (S.TRAINING_COMPLETED, S.ARTIFACTS_INCOMPLETE),
              rank >= RANK[S.ARCHITECTURE_CONFIGURED]),
        stage("artifacts", "Verified artifacts", artifacts_ok, st in (S.TRAINING_COMPLETED, S.ARTIFACTS_INCOMPLETE)
              or st in LEARNED_READY),
        stage("evaluation", "Evaluation", st == S.EVALUATION_COMPLETED, learned),
    ]
    return {
        "state": st.value,
        "busy": busy,
        "can_attach_dataset": not busy,
        "can_configure_split": rank >= RANK[S.DATASET_READY] and not busy and trainable,
        "can_configure_preprocessing": rank >= RANK[S.SPLIT_CONFIGURED] and not busy,
        "can_run_preprocessing": rank >= RANK[S.PREPROCESSING_CONFIGURED] and not busy,
        "can_configure_architecture": rank >= RANK[S.PREPROCESSING_COMPLETED] and not busy,
        "can_train": rank >= RANK[S.ARCHITECTURE_CONFIGURED] and not busy,
        "can_cancel": st in (S.TRAINING_QUEUED, S.TRAINING),
        "can_inspect_untrained": rank >= RANK[S.PREPROCESSING_COMPLETED] and exp.architecture is not None,
        "can_inspect_learned": learned,
        "can_use_breaker": learned,
        "can_evaluate": learned and st != S.EVALUATION_RUNNING,
        "has_results": st == S.EVALUATION_COMPLETED and artifacts_ok and bool(run and run.evaluation),
        "can_reverify": bool(run and run.status == "completed") and not busy,
        "run_stale": bool(run and run.stale),
        "next_page": page,
        "next_step": msg,
        "stages": stages,
    }


def detail(exp: ExperimentRecord) -> dict:
    run = active_run(exp)
    d = exp.model_dump(mode="json")
    d["capabilities"] = capabilities(exp, run)
    d["active_run"] = run.model_dump(mode="json") if run else None
    d["runs"] = [{"id": r.id, "status": r.status, "created_at": r.created_at, "stale": r.stale,
                  "stale_reason": r.stale_reason, "fingerprint": r.fingerprint} for r in repo.list_runs(exp.id)]
    d["datasets"] = []
    for did in (exp.dataset_id, exp.test_dataset_id if exp.workflow == "separate" else None):
        if did:
            try:
                r = repo.get_dataset(did)
                d["datasets"].append({"id": r.id, "name": r.name, "fingerprint": r.fingerprint,
                                      "synthetic": r.synthetic, "structure": r.structure})
            except repo.NotFound:
                pass
    d["synthetic"] = any(x["synthetic"] for x in d["datasets"])
    return d


def summary(exp: ExperimentRecord) -> dict:
    run = active_run(exp)
    ev = (run.evaluation or {}) if run else {}
    pooled = (ev.get("test") or {}).get("pooled") or {}
    best_val = None
    if run and run.folds and run.folds[0].history:
        vals = [h.get("val_balanced_accuracy") for f in run.folds for h in f.history if "val_balanced_accuracy" in h]
        best_val = max(vals) if vals else None
    ds_name = None
    if exp.dataset_id:
        try:
            ds_name = repo.get_dataset(exp.dataset_id).name
        except repo.NotFound:
            ds_name = "(deleted dataset)"
    return {
        "id": exp.id, "name": exp.name, "created_at": exp.created_at, "updated_at": exp.updated_at,
        "state": exp.state.value, "parent_id": exp.parent_id, "dataset_name": ds_name,
        "strategy": exp.split_config.strategy if exp.split_config else None,
        "architecture_preset": exp.architecture.preset if exp.architecture else None,
        "params": None, "run_status": run.status if run else None, "stale": bool(run and run.stale),
        "test_accuracy": pooled.get("accuracy"), "test_balanced_accuracy": pooled.get("balanced_accuracy"),
        "best_val_balanced_accuracy": best_val, "synthetic": _is_synthetic(exp),
    }


def _is_synthetic(exp: ExperimentRecord) -> bool:
    for did in (exp.dataset_id, exp.test_dataset_id):
        if did:
            try:
                if repo.get_dataset(did).synthetic:
                    return True
            except repo.NotFound:
                pass
    return False


# ---------------------------------------------------------------------------
# Creation, duplication, deletion
# ---------------------------------------------------------------------------


def create_experiment(name: str | None = None, dataset_id: str | None = None, test_dataset_id: str | None = None,
                      workflow: str = "single", source_id: str | None = None,
                      inherit: list[str] | None = None) -> ExperimentRecord:
    inherit = inherit or []
    ts = now_iso()
    exp = ExperimentRecord(id=repo.new_id("exp"), name=name or f"Experiment {datetime.now():%d %b %Y %H:%M}",
                           created_at=ts, updated_at=ts, state=S.CREATED)
    repo.save_experiment(exp)
    repo.log_event(exp.id, None, S.CREATED.value, "created")
    if source_id:
        src = repo.get_experiment(source_id)
        exp.parent_id = src.id
        if "dataset" in inherit and src.dataset_id:
            dataset_id, test_dataset_id, workflow = src.dataset_id, src.test_dataset_id, src.workflow
            exp.inherited.append(f"Dataset from '{src.name}'")
        if "split" in inherit and src.split_config:
            exp.split_config = src.split_config
            exp.inherited.append(f"Split settings from '{src.name}' (prefilled - confirm in Dataset & Splits)")
        if "preprocessing" in inherit and src.preprocessing_config:
            exp.preprocessing_config = src.preprocessing_config
            exp.inherited.append(f"Preprocessing settings from '{src.name}' (prefilled - run again to confirm)")
        if "architecture" in inherit and src.architecture:
            exp.architecture = src.architecture
            exp.inherited.append(f"Architecture from '{src.name}' (prefilled - validate to confirm)")
        if "training" in inherit:
            exp.training_config = src.training_config
            exp.inherited.append(f"Training settings from '{src.name}'")
    repo.transition(exp, S.DATASET_REQUIRED, "new experiment starts at Dataset & Splits")
    if dataset_id:
        attach_dataset(exp, dataset_id, test_dataset_id, workflow)
    return repo.get_experiment(exp.id)


def derive_experiment(src: ExperimentRecord, name: str, architecture: ArchitectureConfig | None = None,
                      training: TrainingConfig | None = None, reason: str = "derived") -> ExperimentRecord:
    """Create a separate experiment that inherits the *confirmed* data protocol of ``src``.

    Used for Architecture Breaker variants, Battle Arena variants and registry duplicates. Results, runs and
    checkpoints are never copied; the source experiment is not modified.
    """
    ts = now_iso()
    exp = ExperimentRecord(id=repo.new_id("exp"), name=name, created_at=ts, updated_at=ts, state=S.CREATED,
                           parent_id=src.id, workflow=src.workflow, dataset_id=src.dataset_id,
                           test_dataset_id=src.test_dataset_id, training_config=training or src.training_config)
    repo.save_experiment(exp)
    repo.log_event(exp.id, None, S.CREATED.value, "created", f"{reason} from {src.id}")
    target = S.DATASET_REQUIRED
    if src.dataset_id:
        target = S.DATASET_READY
        exp.inherited.append(f"Dataset from '{src.name}'")
    if RANK[src.state] >= RANK[S.SPLIT_CONFIGURED] and src.split_fingerprint:
        exp.split_config, exp.split_fingerprint = src.split_config, src.split_fingerprint
        shutil.copy(exp_dir(src.id) / f"split_{src.split_fingerprint}.json",
                    exp_dir(exp.id) / f"split_{src.split_fingerprint}.json")
        target = S.SPLIT_CONFIGURED
        exp.inherited.append(f"Split assignment {src.split_fingerprint} (identical trials, matched protocol)")
    if src.preprocessing_config:
        exp.preprocessing_config = src.preprocessing_config
        exp.preprocessing_fingerprint = src.preprocessing_fingerprint
        exp.preprocessing_det_fingerprint = src.preprocessing_det_fingerprint
        if src.ica_reference:
            exp.ica_reference = src.ica_reference
            ref_src = exp_dir(src.id) / "ica"
            if ref_src.exists():
                shutil.copytree(ref_src, exp_dir(exp.id) / "ica", dirs_exist_ok=True)
        if target == S.SPLIT_CONFIGURED and RANK[src.state] >= RANK[S.PREPROCESSING_CONFIGURED]:
            target = S.PREPROCESSING_CONFIGURED
        if RANK[src.state] >= RANK[S.PREPROCESSING_COMPLETED] and src.preprocessing_summary:
            exp.preprocessing_summary = src.preprocessing_summary
            pv = exp_dir(src.id) / "preview_transform"
            if pv.exists():
                shutil.copytree(pv, exp_dir(exp.id) / "preview_transform", dirs_exist_ok=True)
            target = S.PREPROCESSING_COMPLETED
        exp.inherited.append(f"Preprocessing {src.preprocessing_fingerprint} from '{src.name}'")
    arch = architecture or src.architecture
    exp.state = target
    repo.save_experiment(exp)
    repo.log_event(exp.id, S.CREATED.value, target.value, "derived",
                   f"inherited confirmed settings from {src.id} ({src.name})")
    if arch is not None and target == S.PREPROCESSING_COMPLETED:
        try:
            save_architecture(exp, arch, name=f"{name} architecture")
            exp = repo.get_experiment(exp.id)
            exp.inherited.append("Architecture " + ("variant (modified)" if architecture else f"from '{src.name}'"))
        except (ConfigError, WorkflowError) as exc:
            exp = repo.get_experiment(exp.id)
            exp.architecture = arch
            exp.warnings.append(f"Inherited architecture needs attention: {exc}")
    elif arch is not None:
        exp.architecture = arch
    repo.save_experiment(exp)
    return exp


def delete_experiment(exp: ExperimentRecord) -> None:
    if exp.state in BUSY:
        raise WorkflowError("Cancel training/evaluation before deleting this experiment.")
    repo.delete_experiment(exp.id)
    shutil.rmtree(settings.data_dir / "experiments" / exp.id, ignore_errors=True)


# ---------------------------------------------------------------------------
# Stage 1: dataset + split
# ---------------------------------------------------------------------------


def _ensure_not_busy(exp: ExperimentRecord) -> None:
    if exp.state in BUSY:
        raise WorkflowError(f"The experiment is '{exp.state.value}'. Wait for it to finish or cancel it first.")


def attach_dataset(exp: ExperimentRecord, dataset_id: str, test_dataset_id: str | None = None,
                   workflow: str = "single") -> dict:
    _ensure_not_busy(exp)
    train = repo.get_dataset(dataset_id)
    compat = None
    if workflow == "separate":
        if not test_dataset_id:
            raise ConfigError("The separate-dataset workflow needs a test dataset.")
        if test_dataset_id == dataset_id:
            raise ConfigError("Training and test datasets must be different uploads.")
        test = repo.get_dataset(test_dataset_id)
        compat = dsvc.compatibility(train, test)
        if not compat["ok"]:
            raise ConfigError("Training and test datasets are incompatible.", {"compatibility": compat})
    else:
        test_dataset_id = None
    unchanged = (exp.dataset_id == dataset_id and exp.test_dataset_id == test_dataset_id
                 and exp.workflow == workflow and RANK[exp.state] >= RANK[S.DATASET_READY])
    if unchanged:
        return {"changed": False, "compatibility": compat}
    if RANK[exp.state] >= RANK[S.DATASET_READY]:
        repo.rollback(exp, S.DATASET_READY, "dataset changed")
        exp = repo.get_experiment(exp.id)
    exp.dataset_id, exp.test_dataset_id, exp.workflow = dataset_id, test_dataset_id, workflow
    exp.split_fingerprint = None
    exp.preprocessing_summary = None
    exp.preprocessing_fingerprint = None
    exp.ica_reference = None
    if exp.split_config is not None and workflow == "separate":
        exp.split_config = exp.split_config.model_copy(update={"workflow": "separate", "strategy": "separate"})
    repo.save_experiment(exp)
    if exp.state in (S.CREATED, S.DATASET_REQUIRED):
        repo.transition(exp, S.DATASET_READY, f"dataset {dataset_id} attached")
    return {"changed": True, "compatibility": compat}


def preview_split(exp: ExperimentRecord, cfg: SplitConfig) -> dict:
    if not exp.dataset_id:
        raise WorkflowError("Attach a dataset first.")
    trials, sources = pool_trials(exp)
    if exp.workflow == "separate":
        cfg = cfg.model_copy(update={"workflow": "separate", "strategy": "separate"})
    try:
        assignment = make_split(cfg, trials, sources)
    except SplitError as exc:
        raise ConfigError(str(exc)) from exc
    return {"assignment": _assignment_brief(assignment), "integrity": integrity_report(assignment, trials)}


def _assignment_brief(a: SplitAssignment) -> dict:
    return {"strategy": a.strategy, "protocol": a.protocol, "question": a.question, "seed": a.seed,
            "fingerprint": a.fingerprint, "n_candidates": a.n_candidates, "n_folds": len(a.folds),
            "warnings": a.warnings, "config": a.config.model_dump(),
            "folds": [{"index": f.index, "n_train": len(f.train), "n_test": len(f.test)} for f in a.folds]}


def save_split(exp: ExperimentRecord, cfg: SplitConfig) -> dict:
    _ensure_not_busy(exp)
    if RANK[exp.state] < RANK[S.DATASET_READY]:
        raise WorkflowError("Attach a dataset before configuring the split.")
    if exp.dataset_id and not repo.get_dataset(exp.dataset_id).trainable:
        raise WorkflowError("The dataset has no usable labelled trials or event markers.")
    trials, sources = pool_trials(exp)
    if exp.workflow == "separate":
        cfg = cfg.model_copy(update={"workflow": "separate", "strategy": "separate"})
    try:
        assignment = make_split(cfg, trials, sources)
    except SplitError as exc:
        raise ConfigError(str(exc)) from exc
    integrity = integrity_report(assignment, trials)
    (exp_dir(exp.id) / f"split_{assignment.fingerprint}.json").write_text(assignment.model_dump_json())
    same = exp.split_fingerprint == assignment.fingerprint and RANK[exp.state] >= RANK[S.SPLIT_CONFIGURED]
    if not same:
        if RANK[exp.state] > RANK[S.SPLIT_CONFIGURED]:
            repo.rollback(exp, S.SPLIT_CONFIGURED, "split assignment changed")
            exp = repo.get_experiment(exp.id)
            exp.preprocessing_summary = None
            exp.ica_reference = None
        exp.split_config, exp.split_fingerprint = cfg, assignment.fingerprint
        repo.save_experiment(exp)
        if exp.state == S.DATASET_READY:
            repo.transition(exp, S.SPLIT_CONFIGURED, f"split {assignment.fingerprint} confirmed")
            # an inherited preprocessing prefill becomes an unconfirmed configuration
    return {"assignment": _assignment_brief(assignment), "integrity": integrity, "changed": not same}


# ---------------------------------------------------------------------------
# Stage 2: preprocessing
# ---------------------------------------------------------------------------


def validate_preprocessing(exp: ExperimentRecord, cfg: PreprocessingConfig) -> dict:
    v = validate_config(cfg, sources_for(exp))
    if cfg.ica.enabled and cfg.ica.exclude:
        ref_fp = ica_reference_fingerprint(cfg, dataset_fps(exp), exp.split_fingerprint or "")
        if not exp.ica_reference or exp.ica_reference.get("fingerprint") != ref_fp:
            v["errors"].append("ICA components were selected against a reference fit that no longer matches the "
                               "pipeline or split. Fit ICA again and re-select components.")
        elif any(k >= exp.ica_reference.get("n_components", 0) for k in cfg.ica.exclude):
            v["errors"].append("Selected ICA component index exceeds the number of fitted components.")
    return v


def save_preprocessing(exp: ExperimentRecord, cfg: PreprocessingConfig) -> dict:
    _ensure_not_busy(exp)
    if RANK[exp.state] < RANK[S.SPLIT_CONFIGURED]:
        raise WorkflowError("Confirm the dataset and split before configuring preprocessing.")
    v = validate_preprocessing(exp, cfg)
    if v["errors"]:
        raise ConfigError("Invalid preprocessing configuration.", {"validation": v})
    fps = dataset_fps(exp)
    fp = full_fingerprint(cfg, fps)
    same = exp.preprocessing_fingerprint == fp and RANK[exp.state] >= RANK[S.PREPROCESSING_CONFIGURED]
    if not same:
        if RANK[exp.state] > RANK[S.PREPROCESSING_CONFIGURED]:
            repo.rollback(exp, S.PREPROCESSING_CONFIGURED, "preprocessing configuration changed")
            exp = repo.get_experiment(exp.id)
        exp.preprocessing_config = cfg
        exp.preprocessing_fingerprint = fp
        exp.preprocessing_det_fingerprint = deterministic_fingerprint(cfg, fps)
        exp.preprocessing_summary = None
        repo.save_experiment(exp)
        if exp.state == S.SPLIT_CONFIGURED:
            repo.transition(exp, S.PREPROCESSING_CONFIGURED, f"preprocessing {fp} saved")
    return {"validation": v, "fingerprint": fp, "changed": not same}


def _fold0_train_idx(exp: ExperimentRecord, es: EpochSet) -> list[int]:
    """Rows (epochs or windows) whose candidate trial is in the training partition of fold 1."""
    a = load_split(exp)
    return es.rows_for(a.folds[0].train)


def fit_ica_reference(exp: ExperimentRecord, cfg: PreprocessingConfig) -> dict:
    _ensure_not_busy(exp)
    if RANK[exp.state] < RANK[S.SPLIT_CONFIGURED]:
        raise WorkflowError("ICA is fitted on the training partition; confirm the split first.")
    if not cfg.ica.enabled:
        raise ConfigError("Enable ICA first.")
    v = validate_config(cfg, sources_for(exp))
    if v["errors"]:
        raise ConfigError("Invalid preprocessing configuration.", {"validation": v})
    es, _, _ = epochset(cfg, exp)
    tr = _fold0_train_idx(exp, es)
    if len(tr) < 2:
        raise ConfigError("Not enough training epochs to fit ICA.")
    ica, used = fit_ica(es.X, es.ch_names, es.sfreq, cfg.ica, rows=tr)
    ref_fp = ica_reference_fingerprint(cfg, dataset_fps(exp), exp.split_fingerprint or "")
    d = exp_dir(exp.id) / "ica"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    ica.save(d / "reference-ica.fif", overwrite=True)
    (d / "config.json").write_text(cfg.model_dump_json())
    diag = ica_diagnostics(ica, es.X[used], es.ch_names, es.sfreq)
    (d / "diagnostics.json").write_text(json.dumps(diag))
    exp.ica_reference = {"fingerprint": ref_fp, "n_components": diag["n_components"], "n_fit_trials": len(used),
                         "n_training_trials": len(tr),
                         "fit_scope": "training partition of fold 1 (test trials excluded)",
                         "created_at": now_iso(), "method": cfg.ica.method,
                         "fit_trial_ids_sha": _hash(sorted(map(str, es.trial_ids[tr])))}
    repo.save_experiment(exp)
    return {"reference": exp.ica_reference, "diagnostics": diag}


def load_ica_reference(exp: ExperimentRecord):
    from ..core.preprocessing import _mne

    p = exp_dir(exp.id) / "ica" / "reference-ica.fif"
    if not p.exists():
        return None
    return _mne().preprocessing.read_ica(p)


def run_preprocessing(exp: ExperimentRecord) -> dict:
    _ensure_not_busy(exp)
    if RANK[exp.state] < RANK[S.PREPROCESSING_CONFIGURED] or exp.preprocessing_config is None:
        raise WorkflowError("Save a valid preprocessing configuration first.")
    cfg = exp.preprocessing_config
    v = validate_preprocessing(exp, cfg)
    if v["errors"]:
        raise ConfigError("Invalid preprocessing configuration.", {"validation": v})
    try:
        es, report, det_fp = epochset(cfg, exp)
    except PreprocessingError as exc:
        raise ConfigError(str(exc)) from exc
    trials, _ = pool_trials(exp)
    assignment = load_split(exp)
    integrity = integrity_report(assignment, trials, kept_ids=es.kept_parents())
    # final tensor validation
    checks = []
    import numpy as np

    finite, s1, s2 = True, np.zeros(es.X.shape[1]), np.zeros(es.X.shape[1])
    for _i, blk in iter_rows(es.X):  # chunked: es.X is memory-mapped and may exceed RAM
        b = blk.astype(np.float64)
        finite = finite and bool(np.isfinite(b).all())
        s1 += b.sum(axis=(0, 2))
        s2 += (b * b).sum(axis=(0, 2))
    n_vals = len(es.X) * es.X.shape[2]
    ch_sd = np.sqrt(np.maximum(s2 / n_vals - (s1 / n_vals) ** 2, 0))
    checks.append({"id": "finite", "label": "All values finite", "ok": finite,
                   "detail": "no NaN/Inf" if finite else "non-finite values present"})
    checks.append({"id": "classes", "label": "At least two classes", "ok": len(es.class_names) >= 2,
                   "detail": ", ".join(es.class_names)})
    flat_ch = [es.ch_names[c] for c in range(es.X.shape[1]) if finite and float(ch_sd[c]) < 1e-9]
    checks.append({"id": "variance", "label": "No constant channels", "ok": not flat_ch,
                   "detail": ", ".join(flat_ch) if flat_ch else "every channel varies"})
    checks.append({"id": "integrity", "label": "Split integrity after rejection", "ok": integrity["ok"],
                   "detail": "; ".join(c["detail"] for c in integrity["checks"] if c["status"] == "fail") or "ok"})
    checks.append({"id": "shape", "label": "Model input tensor", "ok": es.X.shape[2] >= 8,
                   "detail": f"[N, 1, C, T] = [{len(es.X)}, 1, {es.X.shape[1]}, {es.X.shape[2]}] "
                             f"({es.X.shape[2] / es.sfreq:.3g} s at {es.sfreq:g} Hz)"})
    tensor_ok = all(c["ok"] for c in checks)
    # learned stages: preview fit on the training partition of fold 1 (shown in the studio, refitted per fold)
    preview = None
    if tensor_ok:
        tr = _fold0_train_idx(exp, es)
        ref = load_ica_reference(exp) if (cfg.ica.enabled and cfg.ica.exclude) else None
        lt = LearnedTransform(cfg, es.ch_names, es.sfreq)
        preview = lt.fit(es.X, reference_ica=ref, fit_ids=[str(es.trial_ids[i]) for i in tr], rows=tr)
        pv = exp_dir(exp.id) / "preview_transform"
        if pv.exists():
            shutil.rmtree(pv)
        lt.save(pv)
    result = {"report": report, "integrity": integrity, "tensor_checks": checks, "valid": tensor_ok,
              "learned_preview": preview, "det_fingerprint": det_fp, "fingerprint": exp.preprocessing_fingerprint,
              "computed_at": now_iso(),
              "learned_note": "ICA and normalisation shown here were fitted on the training partition of fold 1. "
                              "During training they are refitted on each fold's training subset only."}
    if not tensor_ok:
        exp.preprocessing_summary = None
        repo.save_experiment(exp)
        raise ConfigError("Preprocessing output failed validation.", {"result": result})
    exp.preprocessing_summary = result
    exp.preprocessing_det_fingerprint = det_fp
    # keep any architecture prefill consistent with the data, without confirming it
    shape = report["output"]["shape"]
    if exp.architecture is not None and exp.state == S.PREPROCESSING_CONFIGURED:
        exp.architecture = exp.architecture.model_copy(update={
            "n_channels": shape[2], "n_times": shape[3], "n_classes": len(es.class_names), "sfreq": es.sfreq})
    repo.save_experiment(exp)
    if exp.state == S.PREPROCESSING_CONFIGURED:
        repo.transition(exp, S.PREPROCESSING_COMPLETED, "preprocessing output validated")
    return result


def data_dims(exp: ExperimentRecord) -> dict:
    """Model input dimensions derived from the validated, processed dataset (never typed in by hand)."""
    if not exp.preprocessing_summary:
        raise WorkflowError("Run preprocessing first.")
    rep = exp.preprocessing_summary["report"]
    out = rep["output"]
    ep = rep.get("epoching", {})
    return {"n_channels": out["shape"][2], "n_times": out["shape"][3], "n_classes": len(out["classes"]),
            "sfreq": out["sfreq"], "classes": out["classes"], "ch_names": rep["channels"]["output_names"],
            "epoch_s": out["duration_s"], "n_epochs": out["n_epochs"], "epoching_mode": ep.get("mode"),
            "windows": ep.get("windows"), "datasets": [repo.get_dataset(i).name for i in
                                                      (exp.dataset_id, exp.test_dataset_id) if i]}


def candidates_report(exp: ExperimentRecord) -> dict:
    """What the split controls work with: candidate trials, participants, recordings, runs and labels."""
    trials, _ = pool_trials(exp)
    groups = sorted({t.group for t in trials if t.group})
    parts: dict[str, dict] = {}
    for t in trials:
        if not t.group:
            continue
        row = parts.setdefault(t.group, {"participant": t.group, "n_trials": 0, "recordings": set(), "sessions": set(),
                                         "runs": set(), "class_counts": {}})
        row["n_trials"] += 1
        if t.recording_id:
            row["recordings"].add(t.recording_id)
        if t.session:
            row["sessions"].add(t.session)
        if t.run:
            row["runs"].add(t.run)
        row["class_counts"][t.label] = row["class_counts"].get(t.label, 0) + 1
    segs = [t for t in trials if t.duration_s > 0]
    return {"n": len(trials), "groups": groups,
            "group_counts": {g: parts[g]["n_trials"] for g in groups},
            "participants": [{**{k: v for k, v in r.items() if k not in ("recordings", "sessions", "runs")},
                              "n_recordings": len(r["recordings"]), "sessions": sorted(r["sessions"]),
                              "runs": sorted(r["runs"])} for r in parts.values()],
            "classes": sorted({t.label for t in trials}),
            "class_counts": {c: sum(1 for t in trials if t.label == c) for c in sorted({t.label for t in trials})},
            "n_recordings": len({t.recording_id or t.recording for t in trials}),
            "n_segments": len(segs),
            "trial_unit": "segments (start/end blocks)" if segs and len(segs) == len(trials) else
            "events / epochs" if not segs else "events and segments",
            "has_groups": bool(groups) and all(t.group for t in trials)}


def suggest_preprocessing(exp: ExperimentRecord) -> dict:
    """Defaults derived from the dataset manifest: EEG-only channels, line noise, rate, epoching for the event type."""
    srcs = sources_for(exp)
    if not srcs:
        raise WorkflowError("Attach a dataset first.")
    ds = srcs[0][1]
    rec = repo.get_dataset(exp.dataset_id)
    summ = rec.summary
    types = summ.get("ch_types") or ["eeg"] * len(ds.ch_names)
    eeg = [c for c, t in zip(ds.ch_names, types) if t == "eeg"]
    meta = (rec.manifest or {}).get("metadata") or {}
    bad = sorted({c for v in (meta.get("bad_channels") or {}).values() for c in v if c in eeg})
    line = meta.get("power_line_hz")
    nyq = ds.sfreq / 2
    trials, _ = pool_trials(exp)
    segs = [t for t in trials if t.duration_s > 0]
    cfg = PreprocessingConfig()
    cfg.channels.selected = eeg if len(eeg) != len(ds.ch_names) and len(eeg) >= 2 else None
    cfg.channels.bad_channels = bad
    cfg.filter.h_freq = float(min(40.0, max(5.0, round(nyq - 5))))
    cfg.filter.notch_enabled = bool(line and line < nyq)
    cfg.filter.notch_freqs = [float(line)] if line else [50.0]
    if ds.sfreq > 256:
        cfg.resample.enabled, cfg.resample.target_sfreq = True, 128.0
    notes = []
    if ds.kind == "continuous":
        med = float(np.median([t.duration_s for t in segs])) if segs else 0.0
        if segs and len(segs) >= len(trials) / 2 and med >= 6.0:
            # long blocks (block designs): cut each block into windows
            shortest = min(t.duration_s for t in segs)
            win = 2.0 if shortest >= 2.0 else max(0.5, round(shortest, 2))
            cfg.epoching = cfg.epoching.model_copy(update={"mode": "windows", "window_s": win, "step_s": win / 2,
                                                           "tmin": 0.0, "tmax": win})
            notes.append(f"Epoching: SLIDING WINDOWS. Trials are long segments ({len(segs)} blocks, median "
                         f"{med:.1f} s): {win:g} s windows with {win / 2:g} s step. Windows of one block stay in "
                         "one partition.")
        else:
            # short trials (e.g. 4 s motor imagery cues): one epoch per event covering the trial
            tmax = min(4.0, max(1.0, np.floor(med * 2) / 2)) if segs else 2.0
            cfg.epoching = cfg.epoching.model_copy(update={"mode": "events", "tmin": 0.0, "tmax": tmax})
            notes.append(f"Epoching: FIXED WINDOW at each event, 0 to {tmax:g} s"
                         + (f" (events last {med:.1f} s on median; sliding windows are suggested only for blocks "
                            "of 6 s or more)." if segs else "."))
    else:
        t0 = ds.tmin
        cfg.epoching = cfg.epoching.model_copy(update={"mode": "keep", "tmin": t0,
                                                       "tmax": t0 + ds.X.shape[2] / ds.sfreq})
    if cfg.channels.selected is not None:
        dropped = [c for c, t in zip(ds.ch_names, types) if t != "eeg"]
        notes.append(f"Non-EEG channels left out of the model input: {', '.join(dropped)} (change under Channels).")
    if bad:
        notes.append(f"Channels marked bad in the dataset metadata: {', '.join(bad)}.")
    if cfg.resample.enabled:
        notes.append(f"Data are sampled at {ds.sfreq:g} Hz; resampling to 128 Hz matches EEGNet's original design.")
    if line:
        notes.append(f"Power-line frequency {line:g} Hz from the dataset sidecar: notch filter enabled.")
    return {"config": cfg.model_dump(), "notes": notes}


def pretraining_checks(exp: ExperimentRecord) -> dict:
    """Compatibility checks run before every training run (channels, rates, classes, events, recordings)."""
    checks: list[dict] = []

    def add(cid: str, label: str, ok: bool, detail: str) -> None:
        checks.append({"id": cid, "label": label, "ok": bool(ok), "detail": detail})

    ids = [i for i in (exp.dataset_id, exp.test_dataset_id if exp.workflow == "separate" else None) if i]
    recs = []
    for i in ids:
        try:
            recs.append(repo.get_dataset(i))
        except repo.NotFound:
            add("dataset", "Datasets available", False, f"dataset {i} was deleted")
    if recs and len(recs) == len(ids):
        add("dataset", "Datasets available", True, ", ".join(r.name for r in recs))
    for r in recs:
        m = r.manifest or {}
        inc = [x for x in m.get("recordings", []) if x.get("included", True)]
        errs = [x["message"] for x in m.get("issues", []) if x.get("level") == "error"]
        errs += [f"{x['file']}: {i['message']}" for x in inc for i in x.get("issues", []) if i.get("level") == "error"]
        dropped = (m.get("summary") or {}).get("dropped_channels") or []
        add(f"manifest_{r.id}", f"{r.name}: recordings compatible", not errs,
            "; ".join(errs[:3]) or f"{len(inc)} recording(s), {len(r.summary.get('ch_names', []))} common channels at "
                                    f"{r.summary.get('sfreq')} Hz" + (f"; {len(dropped)} non-shared channel(s) "
                                                                      "dropped at import" if dropped else ""))
        no_events = [x["file"] for x in inc if (x.get("events") or {}).get("kind") == "none"]
        add(f"events_{r.id}", f"{r.name}: every recording has events", not no_events,
            f"missing for {', '.join(no_events[:3])}" if no_events else "event markers / event files present")
    if len(recs) == 2:
        compat = dsvc.compatibility(recs[0], recs[1])
        add("separate", "Training and test datasets compatible", compat["ok"],
            "; ".join(c["detail"] for c in compat["checks"] if c["status"] != "pass") or "compatible")
    if exp.preprocessing_summary:
        out = exp.preprocessing_summary["report"]["output"]
        integ = exp.preprocessing_summary.get("integrity", {})
        add("classes", "Class mapping usable", len(out["classes"]) >= 2,
            f"{len(out['classes'])} classes: {', '.join(out['classes'])}")
        add("split", "Split integrity after preprocessing", bool(integ.get("ok", False)),
            "; ".join(c["detail"] for c in integ.get("checks", []) if c["status"] == "fail") or
            f"{integ.get('protocol', '')}")
        cur = deterministic_fingerprint(exp.preprocessing_config, dataset_fps(exp)) if exp.preprocessing_config else None
        add("fresh", "Processed data match the attached datasets", cur == exp.preprocessing_det_fingerprint,
            "preprocessing output is current" if cur == exp.preprocessing_det_fingerprint else
            "dataset content changed after preprocessing; run preprocessing again")
    else:
        add("preprocessing", "Preprocessing completed", False, "run and validate the preprocessing pipeline")
    if exp.architecture and exp.preprocessing_summary:
        d = data_dims(exp)
        a = exp.architecture
        ok = (a.n_channels, a.n_times, a.n_classes) == (d["n_channels"], d["n_times"], d["n_classes"])
        add("architecture", "Model input matches the processed data", ok,
            f"model [1, {a.n_channels}, {a.n_times}] → {a.n_classes} classes; data [1, {d['n_channels']}, "
            f"{d['n_times']}] → {d['n_classes']} classes")
    return {"ok": all(c["ok"] for c in checks), "checks": checks}


# ---------------------------------------------------------------------------
# Stage 3: architecture + training configuration
# ---------------------------------------------------------------------------


def default_architecture(exp: ExperimentRecord, preset: str = "eegnet-8,2") -> ArchitectureConfig:
    d = data_dims(exp)
    return preset_config(preset, d["n_channels"], d["n_times"], d["n_classes"], d["sfreq"])


def save_architecture(exp: ExperimentRecord, cfg: ArchitectureConfig, name: str | None = None) -> dict:
    _ensure_not_busy(exp)
    if RANK[exp.state] < RANK[S.PREPROCESSING_COMPLETED]:
        raise WorkflowError("Run and validate preprocessing before configuring the architecture.")
    d = data_dims(exp)
    mism = []
    if cfg.n_channels != d["n_channels"]:
        mism.append(f"input channels {cfg.n_channels} != {d['n_channels']} in the preprocessed data")
    if cfg.n_times != d["n_times"]:
        mism.append(f"time samples {cfg.n_times} != {d['n_times']} in the preprocessed data")
    if cfg.n_classes != d["n_classes"]:
        mism.append(f"classes {cfg.n_classes} != {d['n_classes']} selected classes")
    report = validate_architecture(cfg)
    if mism or not report.valid:
        errors = mism + [i.message for i in report.issues if i.level == "error"]
        raise ConfigError("Invalid architecture: " + "; ".join(errors), {"report": report.to_dict()})
    fp = config_fingerprint(cfg)
    current = None
    if exp.architecture_id:
        try:
            current = repo.get_arch(exp.architecture_id)
        except repo.NotFound:
            current = None
    same = current is not None and current.fingerprint == fp and RANK[exp.state] >= RANK[S.ARCHITECTURE_CONFIGURED]
    if same:
        if exp.architecture != cfg:
            exp.architecture = cfg  # preset label / sfreq only
            repo.save_experiment(exp)
        return {"report": report.to_dict(), "version": current.model_dump(mode="json"), "changed": False}
    if current is not None and current.locked:
        version = ArchitectureVersion(id=repo.new_id("arch"), name=name or current.name, created_at=now_iso(),
                                      experiment_id=exp.id, version=current.version + 1, parent_id=current.id,
                                      config=cfg, fingerprint=fp)
    elif current is not None:
        version = current.model_copy(update={"config": cfg, "fingerprint": fp, "name": name or current.name})
    else:
        version = ArchitectureVersion(id=repo.new_id("arch"), name=name or "Architecture", created_at=now_iso(),
                                      experiment_id=exp.id, version=1, config=cfg, fingerprint=fp)
    repo.save_arch(version)
    if RANK[exp.state] > RANK[S.ARCHITECTURE_CONFIGURED]:
        repo.rollback(exp, S.ARCHITECTURE_CONFIGURED, f"architecture changed (new version v{version.version})")
        exp = repo.get_experiment(exp.id)
    exp.architecture_id, exp.architecture = version.id, cfg
    repo.save_experiment(exp)
    if exp.state == S.PREPROCESSING_COMPLETED:
        repo.transition(exp, S.ARCHITECTURE_CONFIGURED, f"architecture {fp} validated")
    return {"report": report.to_dict(), "version": version.model_dump(mode="json"), "changed": True}


def save_training_config(exp: ExperimentRecord, cfg: TrainingConfig) -> dict:
    _ensure_not_busy(exp)
    errors = validate_training_config(cfg)
    if errors:
        raise ConfigError("Invalid training configuration: " + "; ".join(errors), {"errors": errors})
    changed = cfg != exp.training_config
    if changed:
        if RANK[exp.state] > RANK[S.ARCHITECTURE_CONFIGURED]:
            repo.rollback(exp, S.ARCHITECTURE_CONFIGURED, "training configuration changed")
            exp = repo.get_experiment(exp.id)
        exp.training_config = cfg
        repo.save_experiment(exp)
    return {"changed": changed, "config": cfg.model_dump()}


def run_snapshot(exp: ExperimentRecord) -> dict:
    d = data_dims(exp)
    arch = repo.get_arch(exp.architecture_id)
    return {
        "dataset_ids": [i for i in (exp.dataset_id, exp.test_dataset_id) if i],
        "dataset_fingerprints": dataset_fps(exp),
        "workflow": exp.workflow,
        "split_config": exp.split_config.model_dump() if exp.split_config else None,
        "split_fingerprint": exp.split_fingerprint,
        "preprocessing_config": exp.preprocessing_config.model_dump(),
        "preprocessing_fingerprint": exp.preprocessing_fingerprint,
        "preprocessing_det_fingerprint": exp.preprocessing_det_fingerprint,
        "ica_reference_fingerprint": (exp.ica_reference or {}).get("fingerprint"),
        "architecture_config": arch.config.model_dump(),
        "architecture_fingerprint": arch.fingerprint,
        "architecture_version_id": arch.id,
        "architecture_version": arch.version,
        "training_config": exp.training_config.model_dump(),
        "seed": exp.training_config.seed,
        "class_names": d["classes"],
        "sfreq": d["sfreq"],
        "input_convention": "[batch, 1, channels, time_samples]",
    }


def snapshot_fingerprint(snap: dict) -> str:
    keys = ["dataset_fingerprints", "split_fingerprint", "preprocessing_fingerprint", "ica_reference_fingerprint",
            "architecture_fingerprint", "training_config"]
    return _hash({k: snap.get(k) for k in keys})


def start_training(exp: ExperimentRecord, cfg: TrainingConfig | None = None) -> RunRecord:
    from . import jobs

    _ensure_not_busy(exp)
    if RANK[exp.state] < RANK[S.ARCHITECTURE_CONFIGURED]:
        missing = []
        if RANK[exp.state] < RANK[S.SPLIT_CONFIGURED]:
            missing.append("a confirmed dataset split")
        if RANK[exp.state] < RANK[S.PREPROCESSING_COMPLETED]:
            missing.append("a validated preprocessing run")
        missing.append("a validated architecture")
        raise WorkflowError("Training needs " + ", ".join(missing) + f" (current state: {exp.state.value}).")
    if cfg is not None:
        save_training_config(exp, cfg)
        exp = repo.get_experiment(exp.id)
    errors = validate_training_config(exp.training_config)
    if errors:
        raise ConfigError("; ".join(errors))
    pre = pretraining_checks(exp)
    if not pre["ok"]:
        failed = [f"{c['label']}: {c['detail']}" for c in pre["checks"] if not c["ok"]]
        raise ConfigError("Pre-training checks failed: " + " · ".join(failed), {"pretraining": pre})
    arch = repo.get_arch(exp.architecture_id)
    if not arch.locked:
        arch.locked = True
        repo.save_arch(arch)
    snap = run_snapshot(exp)
    from .system import software_versions

    run = RunRecord(id=repo.new_id("run"), experiment_id=exp.id, created_at=now_iso(), status="queued",
                    fingerprint=snapshot_fingerprint(snap), snapshot=snap, software=software_versions())
    repo.save_run(run)
    exp.active_run_id = run.id
    repo.save_experiment(exp)
    repo.transition(exp, S.TRAINING_QUEUED, f"run {run.id} queued")
    jobs.manager.submit(exp.id, run.id)
    return run


def current_fingerprint(exp: ExperimentRecord) -> str | None:
    try:
        return snapshot_fingerprint(run_snapshot(exp))
    except Exception:
        return None


def ensure_learned_ready(exp: ExperimentRecord) -> RunRecord:
    """Gate for every learned-model inspection endpoint."""
    run = active_run(exp)
    if exp.state not in LEARNED_READY or run is None:
        raise WorkflowError(f"Learned-model inspection needs a completed training run with verified artifacts "
                            f"(current state: {exp.state.value}).")
    if run.stale:
        raise WorkflowError(f"The last run is stale ({run.stale_reason}); retrain to inspect this configuration.")
    if not (run.artifact_status and run.artifact_status.get("ready")):
        raise WorkflowError("Artifacts have not been verified for this run.")
    return run


__all__ = ["WorkflowError", "ConfigError", "TransitionError"]
