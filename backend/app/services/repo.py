"""Repository: typed access to the metadata store."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from ..config import settings
from ..db import Database
from ..lifecycle import State, TransitionError, can_rollback, can_transition
from ..schemas import (ArchitectureVersion, ArtifactRecord, DatasetRecord, ExperimentRecord, RunRecord, now_iso)

_db: Database | None = None
_db_path: Path | None = None


def db() -> Database:
    global _db, _db_path
    path = settings.data_dir / "app.db"
    if _db is None or _db_path != path:
        if _db is not None:
            _db.close()
        _db, _db_path = Database(path), path
    return _db


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class NotFound(KeyError):
    pass


# ---- datasets -------------------------------------------------------------------------


def save_dataset(rec: DatasetRecord) -> None:
    db().execute("INSERT OR REPLACE INTO datasets (id, created_at, data) VALUES (?, ?, ?)",
                 (rec.id, rec.created_at, rec.model_dump_json()))


def get_dataset(ds_id: str) -> DatasetRecord:
    rows = db().query("SELECT data FROM datasets WHERE id = ?", (ds_id,))
    if not rows:
        raise NotFound(f"Dataset {ds_id} not found")
    return DatasetRecord.model_validate_json(rows[0]["data"])


def list_datasets() -> list[DatasetRecord]:
    return [DatasetRecord.model_validate_json(r["data"])
            for r in db().query("SELECT data FROM datasets ORDER BY created_at DESC")]


def delete_dataset(ds_id: str) -> None:
    db().execute("DELETE FROM datasets WHERE id = ?", (ds_id,))


# ---- experiments ----------------------------------------------------------------------


def save_experiment(exp: ExperimentRecord) -> ExperimentRecord:
    exp.updated_at = now_iso()
    db().execute("INSERT OR REPLACE INTO experiments (id, created_at, updated_at, state, name, data) "
                 "VALUES (?, ?, ?, ?, ?, ?)",
                 (exp.id, exp.created_at, exp.updated_at, exp.state.value, exp.name, exp.model_dump_json()))
    return exp


def get_experiment(exp_id: str) -> ExperimentRecord:
    rows = db().query("SELECT data FROM experiments WHERE id = ?", (exp_id,))
    if not rows:
        raise NotFound(f"Experiment {exp_id} not found")
    return ExperimentRecord.model_validate_json(rows[0]["data"])


def list_experiments() -> list[ExperimentRecord]:
    return [ExperimentRecord.model_validate_json(r["data"])
            for r in db().query("SELECT data FROM experiments ORDER BY created_at DESC")]


def delete_experiment(exp_id: str) -> None:
    for table, col in (("runs", "experiment_id"), ("artifacts", "experiment_id"), ("lifecycle_events", "experiment_id"),
                       ("arch_configs", "experiment_id"), ("exports", "experiment_id"), ("experiments", "id")):
        db().execute(f"DELETE FROM {table} WHERE {col} = ?", (exp_id,))


def log_event(exp_id: str, from_state: str | None, to_state: str, kind: str, reason: str | None = None) -> None:
    db().execute("INSERT INTO lifecycle_events (experiment_id, ts, from_state, to_state, kind, reason) "
                 "VALUES (?, ?, ?, ?, ?, ?)", (exp_id, now_iso(), from_state, to_state, kind, reason))


def events(exp_id: str) -> list[dict]:
    return [dict(r) for r in db().query("SELECT ts, from_state, to_state, kind, reason FROM lifecycle_events "
                                        "WHERE experiment_id = ? ORDER BY id", (exp_id,))]


def transition(exp: ExperimentRecord, target: State, reason: str | None = None) -> ExperimentRecord:
    """Apply a forward lifecycle transition; raises ``TransitionError`` when not allowed."""
    current = exp.state
    if current == target:
        return exp
    if not can_transition(current, target):
        raise TransitionError(current, target)
    exp.state = target
    save_experiment(exp)
    log_event(exp.id, current.value, target.value, "transition", reason)
    return exp


def rollback(exp: ExperimentRecord, target: State, reason: str) -> ExperimentRecord:
    """Invalidate downstream stages after an upstream change; marks the active run stale."""
    current = exp.state
    if not can_rollback(current, target):
        raise TransitionError(current, target, f"Cannot change configuration while the experiment is "
                                               f"'{current.value}'. Cancel or wait for it to finish first.")
    if exp.active_run_id:
        try:
            run = get_run(exp.active_run_id)
            if not run.stale:
                run.stale, run.stale_reason = True, reason
                save_run(run)
                exp.stale_notes.append(f"Run {run.id} is stale: {reason}")
        except NotFound:
            pass
    exp.state = target
    save_experiment(exp)
    log_event(exp.id, current.value, target.value, "rollback", reason)
    return exp


# ---- runs -------------------------------------------------------------------------------


def save_run(run: RunRecord) -> None:
    db().execute("INSERT OR REPLACE INTO runs (id, experiment_id, created_at, status, data) VALUES (?, ?, ?, ?, ?)",
                 (run.id, run.experiment_id, run.created_at, run.status, run.model_dump_json()))


def get_run(run_id: str) -> RunRecord:
    rows = db().query("SELECT data FROM runs WHERE id = ?", (run_id,))
    if not rows:
        raise NotFound(f"Run {run_id} not found")
    return RunRecord.model_validate_json(rows[0]["data"])


def list_runs(exp_id: str | None = None, status: str | None = None) -> list[RunRecord]:
    sql, params = "SELECT data FROM runs WHERE 1=1", []
    if exp_id:
        sql += " AND experiment_id = ?"
        params.append(exp_id)
    if status:
        sql += " AND status = ?"
        params.append(status)
    sql += " ORDER BY created_at DESC"
    return [RunRecord.model_validate_json(r["data"]) for r in db().query(sql, tuple(params))]


# ---- architecture versions ------------------------------------------------------------


def save_arch(v: ArchitectureVersion) -> None:
    db().execute("INSERT OR REPLACE INTO arch_configs (id, experiment_id, created_at, data) VALUES (?, ?, ?, ?)",
                 (v.id, v.experiment_id, v.created_at, v.model_dump_json()))


def get_arch(arch_id: str) -> ArchitectureVersion:
    rows = db().query("SELECT data FROM arch_configs WHERE id = ?", (arch_id,))
    if not rows:
        raise NotFound(f"Architecture configuration {arch_id} not found")
    return ArchitectureVersion.model_validate_json(rows[0]["data"])


def list_arch(experiment_id: str | None = None, library: bool = False) -> list[ArchitectureVersion]:
    if library:
        rows = db().query("SELECT data FROM arch_configs WHERE experiment_id IS NULL ORDER BY created_at DESC")
    else:
        rows = db().query("SELECT data FROM arch_configs WHERE experiment_id = ? ORDER BY created_at DESC",
                          (experiment_id,))
    return [ArchitectureVersion.model_validate_json(r["data"]) for r in rows]


def delete_arch(arch_id: str) -> None:
    db().execute("DELETE FROM arch_configs WHERE id = ?", (arch_id,))


# ---- artifacts ------------------------------------------------------------------------


def save_artifact(a: ArtifactRecord) -> None:
    db().execute("INSERT OR REPLACE INTO artifacts (id, experiment_id, run_id, created_at, type, data) "
                 "VALUES (?, ?, ?, ?, ?, ?)", (a.id, a.experiment_id, a.run_id, a.created_at, a.type,
                                               a.model_dump_json()))


def get_artifact(art_id: str) -> ArtifactRecord:
    rows = db().query("SELECT data FROM artifacts WHERE id = ?", (art_id,))
    if not rows:
        raise NotFound(f"Artifact {art_id} not found")
    return ArtifactRecord.model_validate_json(rows[0]["data"])


def list_artifacts(experiment_id: str | None = None, run_id: str | None = None) -> list[ArtifactRecord]:
    sql, params = "SELECT data FROM artifacts WHERE 1=1", []
    if experiment_id:
        sql += " AND experiment_id = ?"
        params.append(experiment_id)
    if run_id:
        sql += " AND run_id = ?"
        params.append(run_id)
    return [ArtifactRecord.model_validate_json(r["data"]) for r in db().query(sql + " ORDER BY created_at",
                                                                               tuple(params))]


# ---- exports ----------------------------------------------------------------------------


def save_export(exp_id: str, export_id: str, data: dict) -> None:
    db().execute("INSERT OR REPLACE INTO exports (id, experiment_id, created_at, data) VALUES (?, ?, ?, ?)",
                 (export_id, exp_id, now_iso(), json.dumps(data)))


def get_export(export_id: str) -> dict:
    rows = db().query("SELECT data FROM exports WHERE id = ?", (export_id,))
    if not rows:
        raise NotFound(f"Export {export_id} not found")
    return json.loads(rows[0]["data"])
