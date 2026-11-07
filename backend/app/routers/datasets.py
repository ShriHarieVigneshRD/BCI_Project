import json
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..data.loaders import UploadOptions
from ..services import datasets as dsvc
from ..services import repo
from ..services.experiments import WorkflowError

router = APIRouter(prefix="/datasets", tags=["datasets"])


class SyntheticRequest(BaseModel):
    kind: Literal["epoched", "continuous"] = "epoched"
    n_participants: int = Field(6, ge=1, le=40)
    n_classes: int = Field(3, ge=2, le=4)
    trials_per_class: int = Field(40, ge=4, le=400)
    n_channels: Literal[8, 16, 22] = 16
    include_frontal: bool = True
    sfreq: float = Field(128.0, ge=64, le=1024)
    n_sessions: int = Field(1, ge=1, le=4)
    seed: int = 7
    name: str | None = None


@router.get("")
def list_datasets() -> list[dict]:
    return [d.model_dump() for d in repo.list_datasets()]


@router.post("/upload")
async def upload(files: list[UploadFile] = File(...), name: str | None = Form(None), sfreq: float | None = Form(None),
                 unit: str = Form("uV"), ch_names: str | None = Form(None), tmin: float = Form(0.0),
                 file_meta: str | None = Form(None)) -> dict:
    if unit not in ("uV", "mV", "V", "nV", "unknown"):
        raise HTTPException(422, "unit must be one of uV, mV, V, nV, unknown")
    try:
        meta = json.loads(file_meta) if file_meta else {}
    except json.JSONDecodeError:
        raise HTTPException(422, "file_meta must be JSON")
    chn = [c.strip() for c in ch_names.split(",") if c.strip()] if ch_names else None
    payload = [(f.filename or "upload", await f.read()) for f in files]
    opts = UploadOptions(name=name or None, sfreq=sfreq, unit=unit, ch_names=chn, tmin=tmin, file_meta=meta)
    rec = dsvc.create_from_upload(payload, opts)
    return rec.model_dump()


@router.post("/synthetic")
def synthetic(req: SyntheticRequest) -> dict:
    return dsvc.create_synthetic(**req.model_dump()).model_dump()


@router.get("/{ds_id}")
def get_dataset(ds_id: str) -> dict:
    return repo.get_dataset(ds_id).model_dump()


@router.get("/{ds_id}/trials")
def trials(ds_id: str) -> list[dict]:
    repo.get_dataset(ds_id)
    return dsvc.trials_table(ds_id)


@router.get("/{ds_id}/signal")
def signal(ds_id: str, trial_id: str) -> dict:
    repo.get_dataset(ds_id)
    return dsvc.trial_signal(ds_id, trial_id)


@router.post("/{ds_id}/revise")
def revise(ds_id: str) -> dict:
    """Open the dataset review again; importing the revision creates a new dataset (experiments are unchanged)."""
    from ..services import imports as isvc

    s = isvc.revise(ds_id)
    return isvc.session_payload(s.id)


@router.get("/{ds_id}/manifest")
def manifest(ds_id: str) -> dict:
    rec = repo.get_dataset(ds_id)
    return {"dataset_id": ds_id, "name": rec.name, "fingerprint": rec.fingerprint, "source": rec.source,
            "manifest": rec.manifest}


@router.get("/{ds_id}/compatibility/{other_id}")
def compatibility(ds_id: str, other_id: str) -> dict:
    return dsvc.compatibility(repo.get_dataset(ds_id), repo.get_dataset(other_id))


@router.get("/{ds_id}/export")
def export_dataset(ds_id: str) -> Response:
    """Explicit dataset export (never included automatically in experiment bundles)."""
    rec = repo.get_dataset(ds_id)
    path = dsvc.settings.data_dir / "datasets" / ds_id / "arrays.npz"
    meta = (dsvc.settings.data_dir / "datasets" / ds_id / "meta.json").read_text()
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(path, "arrays.npz")
        z.writestr("meta.json", meta)
        z.writestr("dataset_record.json", rec.model_dump_json(indent=1))
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{ds_id}_dataset.zip"'})


@router.delete("/{ds_id}")
def delete_dataset(ds_id: str) -> dict:
    repo.get_dataset(ds_id)
    users = [e.name for e in repo.list_experiments() if ds_id in (e.dataset_id, e.test_dataset_id)]
    if users:
        raise WorkflowError(f"Dataset is used by experiment(s): {', '.join(users[:5])}. Delete those first.")
    repo.delete_dataset(ds_id)
    import shutil

    shutil.rmtree(dsvc.settings.data_dir / "datasets" / ds_id, ignore_errors=True)
    shutil.rmtree(dsvc.settings.data_dir / "uploads" / ds_id, ignore_errors=True)
    return {"deleted": ds_id}
