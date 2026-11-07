import json
from typing import Any, Literal

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from ..data.manifest import ArrayOptions
from ..services import imports as isvc

router = APIRouter(prefix="/imports", tags=["imports"])

LOOPBACK = {"127.0.0.1", "::1", "localhost", "testclient"}


class CreateRequest(BaseModel):
    mode: Literal["folder", "files", "local_path"]
    name: str | None = None
    local_path: str | None = None
    attach_experiment_id: str | None = None


class PlanRequest(BaseModel):
    files: list[dict[str, Any]]


class ScanRequest(BaseModel):
    array_options: ArrayOptions | None = None


class ManifestPatch(BaseModel):
    name: str | None = None
    recordings: list[dict[str, Any]] | None = None
    mapping: list[dict[str, Any]] | None = None
    rules: list[dict[str, Any]] | None = None
    channel_policy: Literal["intersection", "strict"] | None = None
    sfreq_policy: Literal["resample", "strict"] | None = None
    target_sfreq: float | None = None
    array_options: ArrayOptions | None = None


@router.post("")
def create(req: CreateRequest, request: Request, wait: bool = False) -> dict:
    if req.mode == "local_path":
        host = request.client.host if request.client else ""
        if host not in LOOPBACK:
            raise HTTPException(403, "Local folder paths can only be imported from this machine.")
    s = isvc.create_session(req.mode, req.name, req.local_path, req.attach_experiment_id)
    if req.mode == "local_path":
        isvc.start_scan(s.id, wait=wait)
    return isvc.session_payload(s.id)


@router.get("/{sid}")
def get(sid: str) -> dict:
    return isvc.session_payload(sid)


@router.post("/{sid}/plan")
def plan(sid: str, req: PlanRequest) -> dict:
    return isvc.plan(sid, req.files)


@router.post("/{sid}/files")
async def files(sid: str, paths: str = Form(...), files: list[UploadFile] = File(...)) -> dict:
    try:
        rels = json.loads(paths)
    except json.JSONDecodeError:
        raise HTTPException(422, "paths must be a JSON array of relative paths")
    return await isvc.stage_files(sid, list(rels), files)


@router.post("/{sid}/scan")
def scan(sid: str, req: ScanRequest | None = None, wait: bool = False) -> dict:
    isvc.start_scan(sid, req.array_options if req else None, wait=wait)
    return isvc.session_payload(sid)


@router.put("/{sid}/manifest")
def update_manifest(sid: str, patch: ManifestPatch) -> dict:
    m = isvc.update_manifest(sid, patch.model_dump(exclude_unset=True))
    return {"manifest": m.model_dump(mode="json")}


@router.post("/{sid}/import")
def run_import(sid: str, wait: bool = False) -> dict:
    isvc.start_import(sid, wait=wait)
    return isvc.session_payload(sid)


@router.delete("/{sid}")
def cancel(sid: str) -> dict:
    isvc.cancel(sid)
    return {"cancelled": sid}
