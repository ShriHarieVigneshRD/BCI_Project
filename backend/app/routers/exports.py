from fastapi import APIRouter
from fastapi.responses import FileResponse

from ..services import exports as exsvc
from ..services import repo

router = APIRouter(prefix="/exports", tags=["exports"])


@router.get("/{export_id}")
def get_export(export_id: str) -> dict:
    return repo.get_export(export_id)


@router.get("/{export_id}/download")
def download(export_id: str) -> FileResponse:
    rec = repo.get_export(export_id)
    return FileResponse(exsvc.export_path(export_id), filename=rec["filename"], media_type="application/zip")
