import asyncio
import json
import time
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..config import settings
from ..core.eegnet import ArchitectureConfig, validate_architecture
from ..core.preprocessing import PreprocessingConfig
from ..core.splits import SplitConfig
from ..core.training import TrainingConfig
from ..lifecycle import BUSY, State
from ..services import artifacts as asvc
from ..services import compare as csvc
from ..services import evaluation as evsvc
from ..services import experiments as xsvc
from ..services import exports as exsvc
from ..services import inspection as isvc
from ..services import preview as pvsvc
from ..services import repo
from ..services.jobs import TERMINAL, hub, manager

router = APIRouter(prefix="/experiments", tags=["experiments"])


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class CreateRequest(BaseModel):
    name: str | None = None
    dataset_id: str | None = None
    test_dataset_id: str | None = None
    workflow: Literal["single", "separate"] = "single"
    source_experiment_id: str | None = None
    inherit: list[Literal["dataset", "split", "preprocessing", "architecture", "training"]] = []


class RenameRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class DuplicateRequest(BaseModel):
    name: str | None = None


class AttachRequest(BaseModel):
    dataset_id: str
    test_dataset_id: str | None = None
    workflow: Literal["single", "separate"] = "single"


class PreviewRequest(BaseModel):
    config: PreprocessingConfig
    trial_id: str | None = None
    channels: list[str] | None = None


class ArchitectureRequest(BaseModel):
    config: ArchitectureConfig
    name: str | None = None


class TrainRequest(BaseModel):
    config: TrainingConfig | None = None


class ConvolutionRequest(BaseModel):
    layer: str
    trial_id: str
    channel: int = 0
    row: int = 0
    position: int = 0
    fold: int = 0
    compare_trial_id: str | None = None


class ArenaRequest(BaseModel):
    trial_id: str
    fold: int = 0
    map_index: int = 0
    demo_stride: int = Field(1, ge=1, le=16)
    demo_padding: Literal["same", "valid"] = "same"


class BreakerRequest(BaseModel):
    changes: dict[str, Any] = {}
    trial_id: str | None = None
    fold: int = 0


class SpawnRequest(BaseModel):
    changes: dict[str, Any]
    name: str | None = None


class VariantRequest(BaseModel):
    name: str | None = None
    architecture: ArchitectureConfig | None = None
    training: TrainingConfig | None = None


class CompareRequest(BaseModel):
    ids: list[str]


class ExportRequest(BaseModel):
    items: list[str]


# ---------------------------------------------------------------------------
# Registry / lifecycle
# ---------------------------------------------------------------------------


@router.get("")
def list_experiments(search: str | None = None, state: str | None = None) -> list[dict]:
    out = []
    for e in repo.list_experiments():
        if search and search.lower() not in e.name.lower() and search.lower() not in e.id.lower():
            continue
        if state and e.state.value != state:
            continue
        out.append(xsvc.summary(e))
    return out


@router.post("")
def create(req: CreateRequest) -> dict:
    exp = xsvc.create_experiment(req.name, req.dataset_id, req.test_dataset_id, req.workflow,
                                 req.source_experiment_id, list(req.inherit))
    return xsvc.detail(exp)


@router.post("/compare")
def compare(req: CompareRequest) -> dict:
    return csvc.compare(req.ids)


@router.get("/{exp_id}")
def get(exp_id: str) -> dict:
    return xsvc.detail(repo.get_experiment(exp_id))


@router.patch("/{exp_id}")
def rename(exp_id: str, req: RenameRequest) -> dict:
    exp = repo.get_experiment(exp_id)
    exp.name = req.name
    repo.save_experiment(exp)
    return xsvc.detail(exp)


@router.post("/{exp_id}/duplicate")
def duplicate(exp_id: str, req: DuplicateRequest) -> dict:
    src = repo.get_experiment(exp_id)
    exp = xsvc.derive_experiment(src, req.name or f"{src.name} (copy)", reason="duplicated")
    return xsvc.detail(exp)


@router.post("/{exp_id}/variants")
def variant(exp_id: str, req: VariantRequest) -> dict:
    """Battle Arena: new experiment with the same confirmed data protocol and modified hyperparameters."""
    src = repo.get_experiment(exp_id)
    exp = xsvc.derive_experiment(src, req.name or f"{src.name} - variant", architecture=req.architecture,
                                 training=req.training, reason="battle arena variant")
    return xsvc.detail(exp)


@router.delete("/{exp_id}")
def delete(exp_id: str, confirm: bool = False) -> dict:
    if not confirm:
        raise HTTPException(400, "Deleting an experiment is irreversible; repeat the request with confirm=true.")
    xsvc.delete_experiment(repo.get_experiment(exp_id))
    return {"deleted": exp_id}


@router.get("/{exp_id}/events")
def events(exp_id: str) -> list[dict]:
    repo.get_experiment(exp_id)
    return repo.events(exp_id)


# ---------------------------------------------------------------------------
# Stage 1: dataset & split
# ---------------------------------------------------------------------------


@router.put("/{exp_id}/dataset")
def attach(exp_id: str, req: AttachRequest) -> dict:
    exp = repo.get_experiment(exp_id)
    res = xsvc.attach_dataset(exp, req.dataset_id, req.test_dataset_id, req.workflow)
    return {**res, "experiment": xsvc.detail(repo.get_experiment(exp_id))}


@router.get("/{exp_id}/candidates")
def candidates(exp_id: str) -> dict:
    return xsvc.candidates_report(repo.get_experiment(exp_id))


@router.get("/{exp_id}/preprocessing/defaults")
def prep_defaults(exp_id: str) -> dict:
    """Suggested preprocessing derived from the dataset manifest (channel types, line noise, event structure)."""
    return xsvc.suggest_preprocessing(repo.get_experiment(exp_id))


@router.get("/{exp_id}/pretraining-checks")
def pretraining_checks(exp_id: str) -> dict:
    return xsvc.pretraining_checks(repo.get_experiment(exp_id))


@router.post("/{exp_id}/split/preview")
def split_preview(exp_id: str, cfg: SplitConfig) -> dict:
    return xsvc.preview_split(repo.get_experiment(exp_id), cfg)


@router.put("/{exp_id}/split")
def split_save(exp_id: str, cfg: SplitConfig) -> dict:
    res = xsvc.save_split(repo.get_experiment(exp_id), cfg)
    return {**res, "experiment": xsvc.detail(repo.get_experiment(exp_id))}


@router.get("/{exp_id}/split")
def split_get(exp_id: str) -> dict:
    exp = repo.get_experiment(exp_id)
    a = xsvc.load_split(exp)
    trials, _ = xsvc.pool_trials(exp)
    kept = None
    if exp.preprocessing_summary:
        es, _, _ = xsvc.epochset(exp.preprocessing_config, exp)
        kept = es.kept_parents()
    from ..core.splits import integrity_report

    return {"assignment": xsvc._assignment_brief(a), "integrity": integrity_report(a, trials, kept_ids=kept),
            "membership": [{"index": f.index, "train": f.train[:5000], "test": f.test[:5000]} for f in a.folds]}


# ---------------------------------------------------------------------------
# Stage 2: preprocessing
# ---------------------------------------------------------------------------


@router.post("/{exp_id}/preprocessing/validate")
def prep_validate(exp_id: str, cfg: PreprocessingConfig) -> dict:
    return xsvc.validate_preprocessing(repo.get_experiment(exp_id), cfg)


@router.post("/{exp_id}/preprocessing/preview")
def prep_preview(exp_id: str, req: PreviewRequest) -> dict:
    return pvsvc.preview(repo.get_experiment(exp_id), req.config, req.trial_id, req.channels)


@router.put("/{exp_id}/preprocessing")
def prep_save(exp_id: str, cfg: PreprocessingConfig) -> dict:
    res = xsvc.save_preprocessing(repo.get_experiment(exp_id), cfg)
    return {**res, "experiment": xsvc.detail(repo.get_experiment(exp_id))}


@router.post("/{exp_id}/preprocessing/run")
def prep_run(exp_id: str) -> dict:
    res = xsvc.run_preprocessing(repo.get_experiment(exp_id))
    return {"result": res, "experiment": xsvc.detail(repo.get_experiment(exp_id))}


@router.post("/{exp_id}/preprocessing/ica")
def prep_ica_fit(exp_id: str, cfg: PreprocessingConfig) -> dict:
    exp = repo.get_experiment(exp_id)
    xsvc.fit_ica_reference(exp, cfg)
    return pvsvc.ica_view(repo.get_experiment(exp_id))


@router.get("/{exp_id}/preprocessing/ica")
def prep_ica_get(exp_id: str, trial_id: str | None = None) -> dict:
    return pvsvc.ica_view(repo.get_experiment(exp_id), trial_id)


# ---------------------------------------------------------------------------
# Stage 3: architecture & training configuration
# ---------------------------------------------------------------------------


@router.get("/{exp_id}/architecture")
def arch_get(exp_id: str) -> dict:
    exp = repo.get_experiment(exp_id)
    dims = None
    default = None
    if exp.preprocessing_summary:
        dims = xsvc.data_dims(exp)
        default = xsvc.default_architecture(exp).model_dump()
    cfg = exp.architecture.model_dump() if exp.architecture else default
    report = validate_architecture(ArchitectureConfig(**cfg)).to_dict() if cfg else None
    return {"config": cfg, "default": default, "data_dims": dims, "report": report,
            "confirmed": exp.state.value not in ("created", "dataset_required", "dataset_ready", "split_configured",
                                                 "preprocessing_configured", "preprocessing_completed"),
            "versions": [v.model_dump(mode="json") for v in repo.list_arch(exp.id)],
            "current_version_id": exp.architecture_id}


@router.put("/{exp_id}/architecture")
def arch_save(exp_id: str, req: ArchitectureRequest) -> dict:
    res = xsvc.save_architecture(repo.get_experiment(exp_id), req.config, req.name)
    return {**res, "experiment": xsvc.detail(repo.get_experiment(exp_id))}


@router.put("/{exp_id}/training-config")
def training_config(exp_id: str, cfg: TrainingConfig) -> dict:
    res = xsvc.save_training_config(repo.get_experiment(exp_id), cfg)
    return {**res, "experiment": xsvc.detail(repo.get_experiment(exp_id))}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@router.post("/{exp_id}/train")
def train(exp_id: str, req: TrainRequest | None = None) -> dict:
    run = xsvc.start_training(repo.get_experiment(exp_id), req.config if req else None)
    return {"run": run.model_dump(mode="json"), "experiment": xsvc.detail(repo.get_experiment(exp_id))}


@router.post("/{exp_id}/cancel")
def cancel(exp_id: str) -> dict:
    exp = repo.get_experiment(exp_id)
    if exp.state not in (State.TRAINING_QUEUED, State.TRAINING) or not exp.active_run_id:
        raise xsvc.WorkflowError("No training run is in progress.")
    manager.cancel(exp.active_run_id)
    return {"cancelling": exp.active_run_id}


def _sse(ev: dict, event: str | None = None) -> str:
    return (f"id: {ev.get('seq', 0)}\n" if "seq" in ev else "") + f"event: {event or ev.get('type', 'message')}\n" \
        f"data: {json.dumps(ev, default=str)}\n\n"


@router.get("/{exp_id}/progress")
async def progress(exp_id: str, request: Request, run_id: str | None = None, since: int = 0):
    exp = repo.get_experiment(exp_id)
    run_id = run_id or exp.active_run_id
    if not run_id:
        raise HTTPException(404, "This experiment has no training run.")
    run = repo.get_run(run_id)

    async def gen():
        snap = {"type": "snapshot", "run_id": run_id, "status": run.status, "device": run.device,
                "error": run.error, "log": run.log[-80:],
                "folds": [{"fold": f.fold, "status": f.status, "history": f.history} for f in run.folds]}
        yield _sse(snap)
        seq = since
        beat = time.time()
        while True:
            if await request.is_disconnected():
                break
            evs = hub.since(run_id, seq)
            for ev in evs:
                seq = ev["seq"]
                yield _sse(ev)
            if any(ev["type"] in TERMINAL for ev in evs):
                break
            if not evs:
                r = repo.get_run(run_id)
                if r.status in ("completed", "failed", "cancelled") and not hub.has(run_id):
                    yield _sse({"type": "done", "status": r.status, "message": r.error or "Run finished."})
                    break
            if time.time() - beat > 10:
                beat = time.time()
                yield ": keep-alive\n\n"
            await asyncio.sleep(0.25)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/{exp_id}/runs")
def runs(exp_id: str) -> list[dict]:
    repo.get_experiment(exp_id)
    return [r.model_dump(mode="json") for r in repo.list_runs(exp_id)]


@router.get("/{exp_id}/runs/{run_id}")
def run_detail(exp_id: str, run_id: str) -> dict:
    r = repo.get_run(run_id)
    if r.experiment_id != exp_id:
        raise repo.NotFound(run_id)
    return r.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


@router.get("/{exp_id}/artifacts")
def artifacts(exp_id: str) -> dict:
    exp = repo.get_experiment(exp_id)
    run = xsvc.active_run(exp)
    return {"state": exp.state.value, "run_id": run.id if run else None,
            "status": run.artifact_status if run else None, "stale": bool(run and run.stale),
            "stale_reason": run.stale_reason if run else None,
            "manifest": asvc.manifest(exp_id, run.id if run else None),
            "capabilities": xsvc.capabilities(exp, run)}


@router.post("/{exp_id}/artifacts/verify")
def verify(exp_id: str) -> dict:
    exp = repo.get_experiment(exp_id)
    if exp.state in BUSY:
        raise xsvc.WorkflowError("Wait for the current job to finish.")
    run = xsvc.active_run(exp)
    if run is None or run.status != "completed":
        raise xsvc.WorkflowError("There is no completed training run to verify.")
    status = asvc.verify_and_update(exp_id, run.id)
    return {"status": status, "experiment": xsvc.detail(repo.get_experiment(exp_id))}


@router.get("/{exp_id}/checkpoints/{ckpt_id}/download")
def download_checkpoint(exp_id: str, ckpt_id: str) -> FileResponse:
    art = repo.get_artifact(ckpt_id)
    if art.experiment_id != exp_id or art.type != "checkpoint":
        raise repo.NotFound(ckpt_id)
    return FileResponse(settings.data_dir / art.storage_ref, filename=f"{exp_id}_{ckpt_id}.pt",
                        media_type="application/octet-stream")


# ---------------------------------------------------------------------------
# Inspection (journey works untrained with explicit labelling; the rest needs verified artifacts)
# ---------------------------------------------------------------------------


@router.get("/{exp_id}/trials")
def trials(exp_id: str, fold: int = 0) -> dict:
    return isvc.trial_list(repo.get_experiment(exp_id), fold)


@router.get("/{exp_id}/journey")
def journey(exp_id: str, trial_id: str, fold: int = 0, mode: Literal["trained", "untrained"] = "trained") -> dict:
    return isvc.journey(repo.get_experiment(exp_id), trial_id, fold, mode)


@router.get("/{exp_id}/kernels")
def kernels(exp_id: str, fold: int = 0) -> dict:
    return isvc.kernels(repo.get_experiment(exp_id), fold)


@router.post("/{exp_id}/convolution")
def convolution(exp_id: str, req: ConvolutionRequest) -> dict:
    return isvc.convolution(repo.get_experiment(exp_id), req.layer, req.trial_id, req.channel, req.row, req.position,
                            req.fold, req.compare_trial_id)


@router.post("/{exp_id}/arena")
def arena(exp_id: str, req: ArenaRequest) -> dict:
    return isvc.arena(repo.get_experiment(exp_id), req.trial_id, req.fold, req.map_index, req.demo_stride,
                      req.demo_padding)


@router.post("/{exp_id}/breaker")
def breaker(exp_id: str, req: BreakerRequest) -> dict:
    return isvc.breaker(repo.get_experiment(exp_id), req.changes, req.trial_id, req.fold)


@router.post("/{exp_id}/breaker/spawn")
def breaker_spawn(exp_id: str, req: SpawnRequest) -> dict:
    exp = isvc.spawn_variant(repo.get_experiment(exp_id), req.changes, req.name)
    return xsvc.detail(exp)


# ---------------------------------------------------------------------------
# Evaluation, results, export
# ---------------------------------------------------------------------------


@router.post("/{exp_id}/evaluate")
def evaluate(exp_id: str) -> dict:
    res = evsvc.evaluate(repo.get_experiment(exp_id))
    return {"evaluation": res, "experiment": xsvc.detail(repo.get_experiment(exp_id))}


@router.get("/{exp_id}/results")
def results(exp_id: str) -> dict:
    return evsvc.results(repo.get_experiment(exp_id))


@router.get("/{exp_id}/predictions")
def predictions(exp_id: str) -> list[dict]:
    return evsvc.predictions(repo.get_experiment(exp_id))


@router.get("/{exp_id}/export/items")
def export_items(exp_id: str) -> dict:
    repo.get_experiment(exp_id)
    return exsvc.EXPORT_ITEMS


@router.post("/{exp_id}/export")
def export(exp_id: str, req: ExportRequest) -> dict:
    return exsvc.build_export(repo.get_experiment(exp_id), req.items)
