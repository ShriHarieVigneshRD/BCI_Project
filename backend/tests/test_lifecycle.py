"""Experiment lifecycle as a state machine (requirements 1, 3, 16, 17)."""

import pytest

from app.lifecycle import BUSY, TRANSITIONS, State, can_rollback, can_transition
from conftest import ok, prepare, train


def test_transition_table_is_explicit():
    assert can_transition(State.ARCHITECTURE_CONFIGURED, State.TRAINING_QUEUED)
    assert not can_transition(State.DATASET_READY, State.TRAINING_QUEUED)
    assert not can_transition(State.TRAINING_FAILED, State.ARTIFACTS_READY)
    assert not can_transition(State.TRAINING_CANCELLED, State.ARTIFACTS_READY)
    assert not can_transition(State.TRAINING, State.ARTIFACTS_READY)  # must pass through verification
    assert can_transition(State.TRAINING_COMPLETED, State.ARTIFACTS_INCOMPLETE)
    for busy in BUSY:
        assert not can_rollback(busy, State.DATASET_READY)
    assert can_rollback(State.EVALUATION_COMPLETED, State.PREPROCESSING_CONFIGURED)
    assert all(isinstance(k, State) for k in TRANSITIONS)


def test_new_experiment_begins_at_dataset_and_splits(client):
    """Requirement 1."""
    exp = ok(client.post("/api/experiments", json={"name": "fresh"}))
    assert exp["state"] == "dataset_required"
    assert exp["capabilities"]["next_page"] == "dataset"
    assert exp["capabilities"]["stages"][0]["id"] == "dataset"
    events = ok(client.get(f"/api/experiments/{exp['id']}/events"))
    assert [e["to_state"] for e in events][:2] == ["created", "dataset_required"]


def test_training_blocked_before_data_and_preprocessing(client, synth_epoched):
    """Requirement 3."""
    eid = ok(client.post("/api/experiments", json={"dataset_id": synth_epoched["id"]}))["id"]
    r = client.post(f"/api/experiments/{eid}/train")
    assert r.status_code == 409 and "split" in r.json()["detail"]
    ok(client.put(f"/api/experiments/{eid}/split", json={"strategy": "random"}))
    r = client.put(f"/api/experiments/{eid}/architecture",
                   json={"config": {"n_channels": 8, "n_times": 128, "n_classes": 3}})
    assert r.status_code == 409  # architecture needs completed preprocessing
    assert client.post(f"/api/experiments/{eid}/train").status_code == 409
    exp = ok(client.get(f"/api/experiments/{eid}"))
    assert not exp["capabilities"]["can_train"]


def test_architecture_must_match_preprocessed_data(client, synth_epoched):
    eid = ok(client.post("/api/experiments", json={"dataset_id": synth_epoched["id"]}))["id"]
    ok(client.put(f"/api/experiments/{eid}/split", json={"strategy": "random"}))
    ok(client.put(f"/api/experiments/{eid}/preprocessing", json={"epoching": {"mode": "crop", "tmin": 0, "tmax": 1}}))
    ok(client.post(f"/api/experiments/{eid}/preprocessing/run"))
    cfg = ok(client.get(f"/api/experiments/{eid}/architecture"))["config"]
    r = client.put(f"/api/experiments/{eid}/architecture", json={"config": {**cfg, "n_channels": cfg["n_channels"] + 1}})
    assert r.status_code == 422
    r = client.put(f"/api/experiments/{eid}/architecture", json={"config": {**cfg, "pool2_kernel": 999}})
    assert r.status_code == 422  # requirement 4 at the API level


def test_new_experiment_preserves_previous_records(client, trained):
    """Requirement 16."""
    before = ok(client.get(f"/api/experiments/{trained['id']}"))
    new = ok(client.post("/api/experiments", json={"source_experiment_id": trained["id"],
                                                   "inherit": ["dataset", "preprocessing", "architecture"]}))
    assert new["id"] != trained["id"] and new["state"] == "dataset_ready"
    assert any("prefilled" in i for i in new["inherited"]) and new["parent_id"] == trained["id"]
    after = ok(client.get(f"/api/experiments/{trained['id']}"))
    assert after["state"] == before["state"] and after["active_run_id"] == before["active_run_id"]
    assert after["active_run"]["artifact_status"] == before["active_run"]["artifact_status"]
    ids = {e["id"] for e in ok(client.get("/api/experiments"))}
    assert {trained["id"], new["id"]} <= ids


def test_changes_after_training_mark_results_stale(client, synth_epoched):
    """Requirement 17."""
    eid = prepare(client, synth_epoched["id"], name="stale test")
    exp = train(client, eid)
    assert exp["capabilities"]["can_inspect_learned"]
    run_id = exp["active_run_id"]
    # preprocessing change
    ok(client.put(f"/api/experiments/{eid}/preprocessing",
                  json={"epoching": {"mode": "crop", "tmin": 0, "tmax": 1.0}, "filter": {"l_freq": 8, "h_freq": 30}}))
    exp = ok(client.get(f"/api/experiments/{eid}"))
    assert exp["state"] == "preprocessing_configured"
    assert exp["capabilities"]["run_stale"] and not exp["capabilities"]["can_inspect_learned"]
    run = ok(client.get(f"/api/experiments/{eid}/runs/{run_id}"))
    assert run["stale"] and "preprocessing" in run["stale_reason"]
    assert client.get(f"/api/experiments/{eid}/kernels").status_code == 409
    assert client.get(f"/api/experiments/{eid}/results").status_code == 409
    # dataset change rolls back further
    other = ok(client.post("/api/datasets/synthetic", json={"kind": "epoched", "n_participants": 3, "n_channels": 8,
                                                             "trials_per_class": 6, "include_frontal": False}))
    ok(client.put(f"/api/experiments/{eid}/dataset", json={"dataset_id": other["id"]}))
    exp = ok(client.get(f"/api/experiments/{eid}"))
    assert exp["state"] == "dataset_ready" and exp["split_fingerprint"] is None
    kinds = [e["kind"] for e in ok(client.get(f"/api/experiments/{eid}/events"))]
    assert kinds.count("rollback") >= 2


def test_architecture_change_after_training_creates_new_version(client, synth_epoched):
    eid = prepare(client, synth_epoched["id"], name="versioning")
    exp = train(client, eid)
    v1 = exp["architecture_id"]
    cfg = ok(client.get(f"/api/experiments/{eid}/architecture"))["config"]
    ok(client.put(f"/api/experiments/{eid}/architecture", json={"config": {**cfg, "F1": 4, "F2": 8}}))
    arch = ok(client.get(f"/api/experiments/{eid}/architecture"))
    assert arch["current_version_id"] != v1
    versions = {v["id"]: v for v in arch["versions"]}
    assert versions[v1]["locked"] and versions[v1]["config"]["F1"] == cfg["F1"]  # trained version untouched
    assert versions[arch["current_version_id"]]["version"] == 2
    run = ok(client.get(f"/api/experiments/{eid}/runs/{exp['active_run_id']}"))
    assert run["stale"] and run["snapshot"]["architecture_config"]["F1"] == cfg["F1"]


def test_delete_requires_confirmation(client):
    eid = ok(client.post("/api/experiments", json={"name": "to delete"}))["id"]
    assert client.delete(f"/api/experiments/{eid}").status_code == 400
    ok(client.delete(f"/api/experiments/{eid}", params={"confirm": "true"}))
    assert client.get(f"/api/experiments/{eid}").status_code == 404
