from fastapi import APIRouter

from ..config import APP_VERSION
from ..data.loaders import SUPPORTED_FORMATS
from ..lifecycle import describe
from ..services.jobs import manager
from ..services.system import device_info, software_versions

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "app_version": APP_VERSION, "software": software_versions(), "device": device_info(),
            "training_worker": {"running_run_id": manager.current}, "binding": "localhost only by default",
            "privacy": "EEG data are processed locally and never sent to external services."}


@router.get("/meta/formats")
def formats() -> list[dict]:
    return SUPPORTED_FORMATS


@router.get("/meta/lifecycle")
def lifecycle() -> dict:
    return describe()
