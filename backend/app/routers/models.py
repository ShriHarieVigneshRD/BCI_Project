from fastapi import APIRouter
from pydantic import BaseModel

from ..core.eegnet import PRESETS, ArchitectureConfig, config_fingerprint, map_coordinate, preset_config, validate_architecture
from ..schemas import ArchitectureVersion, now_iso
from ..services import repo

router = APIRouter(prefix="/models", tags=["models"])


class ValidateRequest(BaseModel):
    config: ArchitectureConfig


class CoordinateRequest(BaseModel):
    config: ArchitectureConfig
    layer: str
    coord: list[int]


class PresetRequest(BaseModel):
    preset: str
    n_channels: int
    n_times: int
    n_classes: int
    sfreq: float | None = None


class SaveConfigRequest(BaseModel):
    name: str
    config: ArchitectureConfig


@router.get("/presets")
def presets() -> list[dict]:
    return PRESETS


@router.post("/presets/build")
def build_preset(req: PresetRequest) -> dict:
    return preset_config(req.preset, req.n_channels, req.n_times, req.n_classes, req.sfreq).model_dump()


@router.post("/validate")
def validate(req: ValidateRequest) -> dict:
    """Shape propagation + validity for any configuration (pre-training validation; no data needed)."""
    return validate_architecture(req.config).to_dict()


@router.post("/coordinate")
def coordinate(req: CoordinateRequest) -> dict:
    rep = validate_architecture(req.config)
    if not rep.valid:
        raise ValueError("Architecture is invalid.")
    return map_coordinate(rep, req.layer, req.coord)


@router.get("/configs")
def library() -> list[dict]:
    return [v.model_dump(mode="json") for v in repo.list_arch(library=True)]


@router.post("/configs")
def save_config(req: SaveConfigRequest) -> dict:
    v = ArchitectureVersion(id=repo.new_id("arch"), name=req.name, created_at=now_iso(), config=req.config,
                            fingerprint=config_fingerprint(req.config))
    repo.save_arch(v)
    return v.model_dump(mode="json")


@router.post("/configs/{arch_id}/duplicate")
def duplicate_config(arch_id: str) -> dict:
    src = repo.get_arch(arch_id)
    v = ArchitectureVersion(id=repo.new_id("arch"), name=f"{src.name} (copy)", created_at=now_iso(),
                            config=src.config, fingerprint=src.fingerprint, parent_id=src.id)
    repo.save_arch(v)
    return v.model_dump(mode="json")


@router.get("/configs/{arch_id}")
def get_config(arch_id: str) -> dict:
    return repo.get_arch(arch_id).model_dump(mode="json")


@router.delete("/configs/{arch_id}")
def delete_config(arch_id: str) -> dict:
    v = repo.get_arch(arch_id)
    if v.locked:
        raise ValueError("This configuration was used by a training run and is immutable.")
    repo.delete_arch(arch_id)
    return {"deleted": arch_id}
