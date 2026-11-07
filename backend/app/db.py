"""SQLite metadata store (thread-safe, JSON payload columns)."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, state TEXT NOT NULL,
    name TEXT NOT NULL, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, created_at TEXT NOT NULL, status TEXT NOT NULL,
    data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS arch_configs (
    id TEXT PRIMARY KEY, experiment_id TEXT, created_at TEXT NOT NULL, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY, experiment_id TEXT, run_id TEXT, created_at TEXT NOT NULL, type TEXT NOT NULL,
    data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS lifecycle_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id TEXT NOT NULL, ts TEXT NOT NULL, from_state TEXT,
    to_state TEXT NOT NULL, kind TEXT NOT NULL, reason TEXT);
CREATE TABLE IF NOT EXISTS import_sessions (
    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, status TEXT NOT NULL, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS exports (
    id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, created_at TEXT NOT NULL, data TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_runs_exp ON runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_art_run ON artifacts(run_id);
CREATE INDEX IF NOT EXISTS idx_events_exp ON lifecycle_events(experiment_id);
"""


class Database:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def execute(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            self._conn.execute(sql, params)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
