"""Typed records persisted in the metadata store and returned by the API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from .core.eegnet import ArchitectureConfig
from .core.preprocessing import PreprocessingConfig
from .core.splits import SplitConfig
from .core.training import TrainingConfig
from .lifecycle import State


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DatasetRecord(BaseModel):
    id: str
    name: str
    created_at: str
    format: str
    structure: str
    fingerprint: str
    synthetic: bool = False
    files: list[str] = []
    summary: dict[str, Any] = {}
    issues: list[dict[str, Any]] = []
    trainable: bool = True
    notes: list[str] = []
    manifest: dict[str, Any] | None = None      # dataset manifest (recordings, identities, mapping, file references)
    source: dict[str, Any] = {}                  # how the data were imported (mode, root label, local path, import id)


class ImportSession(BaseModel):
    id: str
    created_at: str
    updated_at: str
    mode: Literal["folder", "files", "local_path", "revise"]
    status: Literal["created", "uploading", "scanning", "review", "importing", "done", "failed", "cancelled"] = "created"
    name: str | None = None
    root_label: str | None = None
    local_path: str | None = None
    source_dir: str | None = None                # data-store relative directory holding staged files
    staged_files: int = 0
    staged_bytes: int = 0
    progress: dict[str, Any] = {}
    error: str | None = None
    dataset_id: str | None = None
    revises: str | None = None
    attach_experiment_id: str | None = None


class ArchitectureVersion(BaseModel):
    id: str
    name: str
    created_at: str
    experiment_id: str | None = None
    version: int = 1
    parent_id: str | None = None
    locked: bool = False          # used by a training run -> immutable
    config: ArchitectureConfig
    fingerprint: str


class ExperimentRecord(BaseModel):
    id: str
    name: str
    created_at: str
    updated_at: str
    state: State
    parent_id: str | None = None
    inherited: list[str] = []
    description: str | None = None
    workflow: Literal["single", "separate"] = "single"
    dataset_id: str | None = None
    test_dataset_id: str | None = None
    split_config: SplitConfig | None = None
    split_fingerprint: str | None = None
    preprocessing_config: PreprocessingConfig | None = None
    preprocessing_fingerprint: str | None = None
    preprocessing_det_fingerprint: str | None = None
    preprocessing_summary: dict[str, Any] | None = None
    ica_reference: dict[str, Any] | None = None
    architecture_id: str | None = None
    architecture: ArchitectureConfig | None = None
    training_config: TrainingConfig = Field(default_factory=TrainingConfig)
    active_run_id: str | None = None
    warnings: list[str] = []
    stale_notes: list[str] = []


class FoldRun(BaseModel):
    fold: int
    status: Literal["pending", "training", "completed", "failed", "cancelled"] = "pending"
    checkpoint_id: str | None = None
    fit_ids_count: int = 0
    val_ids_count: int = 0
    test_ids_count: int = 0
    val_description: str | None = None
    history: list[dict[str, Any]] = []
    info: dict[str, Any] = {}
    transform_report: dict[str, Any] = {}
    activations: dict[str, Any] | None = None


class RunRecord(BaseModel):
    id: str
    experiment_id: str
    created_at: str
    status: Literal["queued", "training", "completed", "failed", "cancelled"] = "queued"
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    device: str | None = None
    fingerprint: str
    snapshot: dict[str, Any]
    folds: list[FoldRun] = []
    artifact_status: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    stale: bool = False
    stale_reason: str | None = None
    software: dict[str, str] = {}
    log: list[str] = []


class ArtifactRecord(BaseModel):
    id: str
    type: str
    experiment_id: str | None
    run_id: str | None = None
    checkpoint_id: str | None = None
    created_at: str
    source_ref: str | None = None
    config_fingerprint: str | None = None
    storage_ref: str
    sha256: str | None = None
    size_bytes: int | None = None
    meta: dict[str, Any] = {}
