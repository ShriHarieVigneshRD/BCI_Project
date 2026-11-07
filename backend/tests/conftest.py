"""Shared fixtures. All tests use small, explicitly SYNTHETIC EEG data and an isolated data directory."""

from __future__ import annotations

import warnings

import pytest

warnings.filterwarnings("ignore")

SMALL_SYNTH = {"kind": "epoched", "n_participants": 4, "n_classes": 3, "trials_per_class": 12, "n_channels": 8,
               "include_frontal": False, "sfreq": 128.0, "seed": 3}
FAST_TRAIN = {"epochs": 3, "batch_size": 16, "device": "cpu", "patience": 5, "seed": 1}


def ok(r):
    assert r.status_code < 400, f"{r.status_code}: {r.text[:2000]}"
    return r.json()


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    from app.config import configure

    configure(tmp_path_factory.mktemp("eegnet-data"))
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app()) as c:
        yield c


@pytest.fixture(scope="session")
def synth_epoched(client):
    return ok(client.post("/api/datasets/synthetic", json=SMALL_SYNTH))


@pytest.fixture(scope="session")
def synth_continuous(client):
    return ok(client.post("/api/datasets/synthetic", json={**SMALL_SYNTH, "kind": "continuous"}))


def prepare(client, dataset_id: str, split: dict | None = None, prep: dict | None = None, name: str = "test",
            arch_overrides: dict | None = None) -> str:
    """Walk an experiment to ``architecture_configured``."""
    eid = ok(client.post("/api/experiments", json={"name": name, "dataset_id": dataset_id}))["id"]
    ok(client.put(f"/api/experiments/{eid}/split", json=split or {"strategy": "random", "test_ratio": 0.25}))
    ok(client.put(f"/api/experiments/{eid}/preprocessing",
                  json=prep or {"epoching": {"mode": "crop", "tmin": 0.0, "tmax": 1.0}}))
    ok(client.post(f"/api/experiments/{eid}/preprocessing/run"))
    cfg = ok(client.get(f"/api/experiments/{eid}/architecture"))["config"]
    cfg.update(arch_overrides or {})
    ok(client.put(f"/api/experiments/{eid}/architecture", json={"config": cfg}))
    return eid


def train(client, eid: str, **overrides) -> dict:
    from app.services.jobs import manager

    ok(client.post(f"/api/experiments/{eid}/train", json={"config": {**FAST_TRAIN, **overrides}}))
    assert manager.wait_idle(600)
    return ok(client.get(f"/api/experiments/{eid}"))


@pytest.fixture(scope="session")
def trained(client, synth_epoched):
    """A trained + verified experiment shared by read-only tests (do not mutate)."""
    eid = prepare(client, synth_epoched["id"], name="shared trained")
    exp = train(client, eid)
    assert exp["state"] == "artifacts_ready", exp["active_run"]
    return exp
