"""Training, artifact gate, learned-model inspection, breaker, results and export (requirements 5-13, 18)."""

import io
import json
import time
import zipfile

import numpy as np
import pytest
import torch

from app.config import settings
from app.core.eegnet import ArchitectureConfig, EEGNet
from app.core.preprocessing import EpochSet, LearnedTransform
from app.core.tracing import run_trace
from app.services import repo
from app.services.jobs import hub, manager
from conftest import FAST_TRAIN, ok, prepare, train


def _ckpt(exp, fold=0):
    run = exp["active_run"]
    art = repo.get_artifact(run["folds"][fold]["checkpoint_id"])
    return settings.data_dir / art.storage_ref


def test_progress_comes_from_real_training(client, trained):
    """Requirement 5: streamed metrics are the backend's real training metrics."""
    run = trained["active_run"]
    fold = run["folds"][0]
    events = [e for e in hub.since(run["id"], 0) if e["type"] == "epoch"]
    assert len(events) == len(fold["history"]) == FAST_TRAIN["epochs"]
    assert [e["train_loss"] for e in events] == [h["train_loss"] for h in fold["history"]]
    # recompute the validation loss of the restored (selected) model from the checkpoint
    payload = torch.load(_ckpt(trained), map_location="cpu", weights_only=True)
    model = EEGNet(ArchitectureConfig(**payload["architecture"]))
    model.load_state_dict(payload["state_dict"])
    fd = _ckpt(trained).parent
    part = json.loads((fd / "partition.json").read_text())
    es = EpochSet.load(settings.data_dir / "cache" / "preprocessed" /
                       f"{run['snapshot']['preprocessing_det_fingerprint']}.npz")
    idx = es.index_of()
    x = LearnedTransform.load(fd / "transform").transform(es.X[[idx[i] for i in part["val"]]])
    logits = torch.from_numpy(run_trace(model, x).logits)
    loss = torch.nn.functional.cross_entropy(logits, torch.from_numpy(es.y[[idx[i] for i in part["val"]]])).item()
    sel = fold["info"]["selected_epoch"]
    assert abs(loss - fold["history"][sel - 1]["val_loss"]) < 1e-4
    # training changed the weights relative to the seeded initialisation
    torch.manual_seed(FAST_TRAIN["seed"])
    init = EEGNet(ArchitectureConfig(**payload["architecture"]))
    assert not torch.allclose(init.temporal_conv.conv.weight, model.temporal_conv.conv.weight)


def test_successful_training_unlocks_post_training_tools(client, trained):
    """Requirement 7."""
    caps = trained["capabilities"]
    assert trained["state"] == "artifacts_ready"
    assert caps["can_inspect_learned"] and caps["can_use_breaker"] and caps["can_evaluate"]
    checks = {c["id"]: c["ok"] for c in trained["active_run"]["artifact_status"]["checks"]}
    assert all(checks.values()) and {"checkpoint", "checkpoint_loads", "forward_pass", "provenance"} <= set(checks)
    eid = trained["id"]
    assert client.get(f"/api/experiments/{eid}/kernels").status_code == 200


def test_failed_training_does_not_unlock(client, synth_epoched, monkeypatch):
    """Requirement 6 (failure)."""
    import app.services.jobs as jobs

    def boom(*a, **k):
        raise RuntimeError("simulated optimiser failure")

    monkeypatch.setattr(jobs, "train_fold", boom)
    eid = prepare(client, synth_epoched["id"], name="fails")
    exp = train(client, eid)
    assert exp["state"] == "training_failed"
    assert "simulated optimiser failure" in exp["active_run"]["error"]
    caps = exp["capabilities"]
    assert not caps["can_inspect_learned"] and not caps["can_evaluate"] and caps["can_train"]
    for path in ("kernels", "results"):
        assert client.get(f"/api/experiments/{eid}/{path}").status_code == 409
    assert client.post(f"/api/experiments/{eid}/breaker", json={"changes": {"F1": 4}}).status_code == 409
    assert client.post(f"/api/experiments/{eid}/evaluate").status_code == 409


def test_cancelled_training_does_not_unlock(client, synth_epoched):
    """Requirement 6 (cancellation of a real run)."""
    eid = prepare(client, synth_epoched["id"], name="cancel")
    out = ok(client.post(f"/api/experiments/{eid}/train", json={"config": {**FAST_TRAIN, "epochs": 2000,
                                                                           "early_stopping": False}}))
    run_id = out["run"]["id"]
    t0 = time.time()
    while not any(e["type"] == "epoch" for e in hub.since(run_id, 0)):
        assert time.time() - t0 < 120
        time.sleep(0.05)
    ok(client.post(f"/api/experiments/{eid}/cancel"))
    assert manager.wait_idle(120)
    exp = ok(client.get(f"/api/experiments/{eid}"))
    assert exp["state"] == "training_cancelled"
    assert exp["active_run"]["status"] == "cancelled"
    assert not exp["capabilities"]["can_inspect_learned"]
    assert client.get(f"/api/experiments/{eid}/kernels").status_code == 409


def test_corrupt_or_missing_checkpoint_blocks_readiness(client, synth_epoched):
    """Requirement 8."""
    eid = prepare(client, synth_epoched["id"], name="corrupt")
    exp = train(client, eid)
    assert exp["state"] == "artifacts_ready"
    path = _ckpt(exp)
    original = path.read_bytes()
    path.write_bytes(original[:-64] + b"\0" * 64)
    out = ok(client.post(f"/api/experiments/{eid}/artifacts/verify"))
    assert out["experiment"]["state"] == "artifacts_incomplete"
    failed = [c for c in out["status"]["checks"] if not c["ok"]]
    assert any("corrupted" in c["detail"] for c in failed)
    assert client.get(f"/api/experiments/{eid}/kernels").status_code == 409
    path.unlink()
    out = ok(client.post(f"/api/experiments/{eid}/artifacts/verify"))
    assert any("missing" in c["detail"] for c in out["status"]["checks"] if not c["ok"])
    path.write_bytes(original)  # recovery: restored file re-verifies
    out = ok(client.post(f"/api/experiments/{eid}/artifacts/verify"))
    assert out["experiment"]["state"] == "artifacts_ready"


def test_activation_extraction_uses_checkpoint_and_input(client, trained):
    """Requirement 9."""
    eid = trained["id"]
    trials = ok(client.get(f"/api/experiments/{eid}/trials"))["trials"]
    tid = next(t["trial_id"] for t in trials if t["partition"] == "test")
    j = ok(client.get(f"/api/experiments/{eid}/journey", params={"trial_id": tid}))
    assert j["mode"] == "trained" and j["provenance"]["checkpoint_id"] == trained["active_run"]["folds"][0]["checkpoint_id"]
    assert j["provenance"]["partition"] == "test"
    payload = torch.load(_ckpt(trained), map_location="cpu", weights_only=True)
    model = EEGNet(ArchitectureConfig(**payload["architecture"]))
    model.load_state_dict(payload["state_dict"])
    run = trained["active_run"]
    es = EpochSet.load(settings.data_dir / "cache" / "preprocessed" /
                       f"{run['snapshot']['preprocessing_det_fingerprint']}.npz")
    x = LearnedTransform.load(_ckpt(trained).parent / "transform").transform(es.X[[es.index_of()[tid]]])
    tr = run_trace(model, x)
    stages = {s["id"]: s for s in j["stages"]}
    np.testing.assert_allclose(np.array(stages["model_input"]["data"]), x[0], atol=1e-4)
    for layer in ("temporal_conv", "depthwise_conv", "separable_pointwise", "classifier"):
        np.testing.assert_allclose(np.array(stages[layer]["data"]), tr.outputs[layer][0], atol=1e-4)
    np.testing.assert_allclose(j["probabilities"], tr.probabilities[0], atol=1e-4)
    # a different trial gives different activations
    other = next(t["trial_id"] for t in trials if t["trial_id"] != tid)
    j2 = ok(client.get(f"/api/experiments/{eid}/journey", params={"trial_id": other}))
    assert j2["stages"][3]["data"] != stages["temporal_conv"]["data"]


def test_kernel_visualisation_uses_real_weights(client, trained):
    """Requirement 10."""
    eid = trained["id"]
    k = ok(client.get(f"/api/experiments/{eid}/kernels"))
    sd = torch.load(_ckpt(trained), map_location="cpu", weights_only=True)["state_dict"]
    np.testing.assert_allclose(k["temporal"]["weights"], sd["temporal_conv.conv.weight"][:, 0, 0].numpy(), atol=1e-6)
    np.testing.assert_allclose(k["depthwise"]["weights"], sd["depthwise_conv.conv.weight"][:, 0, :, 0].numpy(),
                               atol=1e-6)
    np.testing.assert_allclose(k["pointwise"]["weights"], sd["separable_pointwise.conv.weight"][:, :, 0, 0].numpy(),
                               atol=1e-6)
    np.testing.assert_allclose(k["classifier"]["weights"], sd["classifier.weight"].numpy(), atol=1e-6)
    tid = ok(client.get(f"/api/experiments/{eid}/trials"))["trials"][0]["trial_id"]
    for layer in ("temporal_conv", "depthwise_conv", "separable_depthwise", "separable_pointwise", "pool1", "bn2",
                  "act1", "classifier"):
        c = ok(client.post(f"/api/experiments/{eid}/convolution",
                           json={"layer": layer, "trial_id": tid, "channel": 1, "row": 0, "position": 3}))
        assert c["verified"], (layer, c["abs_diff"])
    c = ok(client.post(f"/api/experiments/{eid}/convolution",
                       json={"layer": "temporal_conv", "trial_id": tid, "channel": 2, "row": 1, "position": 0}))
    np.testing.assert_allclose(c["weights"], sd["temporal_conv.conv.weight"][2, 0, 0].numpy(), atol=1e-6)
    assert any(c["window"]["padded"])  # 'same' padding at position 0 includes padded zeros


def test_breaker_variants_leave_original_untouched(client, trained):
    """Requirements 11 and 12."""
    eid = trained["id"]
    before = ok(client.get(f"/api/experiments/{eid}"))
    ckpt_bytes = _ckpt(trained).read_bytes()
    tid = ok(client.get(f"/api/experiments/{eid}/trials"))["trials"][0]["trial_id"]
    structural = ok(client.post(f"/api/experiments/{eid}/breaker", json={"changes": {"F1": 4, "F2": 8},
                                                                         "trial_id": tid}))
    assert structural["valid"] and not structural["checkpoint_valid"]
    assert structural["changed"]["F1"]["kind"] == "structural"
    assert any(not t["compatible"] for t in structural["tensors"])
    assert structural["learned_outputs"] is None  # no invented learned activations
    comp = ok(client.post(f"/api/experiments/{eid}/breaker", json={"changes": {"activation": "relu"}}))
    assert comp["changed"]["activation"]["kind"] == "computation" and not comp["checkpoint_valid"]
    neutral = ok(client.post(f"/api/experiments/{eid}/breaker", json={"changes": {"dropout": 0.1}, "trial_id": tid}))
    assert neutral["checkpoint_valid"] and neutral["learned_outputs"]["max_abs_difference"] < 1e-6
    invalid = ok(client.post(f"/api/experiments/{eid}/breaker", json={"changes": {"pool2_kernel": 500}}))
    assert not invalid["valid"]
    variant = ok(client.post(f"/api/experiments/{eid}/breaker/spawn", json={"changes": {"F1": 4, "F2": 8}}))
    assert variant["id"] != eid and variant["architecture"]["F1"] == 4
    assert variant["state"] == "architecture_configured" and variant["active_run_id"] is None
    after = ok(client.get(f"/api/experiments/{eid}"))
    for key in ("state", "architecture", "architecture_id", "active_run_id", "preprocessing_fingerprint",
                "split_fingerprint"):
        assert after[key] == before[key]
    assert _ckpt(trained).read_bytes() == ckpt_bytes


def test_results_only_after_evaluation(client, synth_epoched):
    """Requirement 13."""
    eid = prepare(client, synth_epoched["id"], name="results gate")
    exp = train(client, eid)
    r = client.get(f"/api/experiments/{eid}/results")
    assert r.status_code == 409 and "never substituted" in r.json()["detail"]
    assert client.get(f"/api/experiments/{eid}/predictions").status_code == 409
    assert not exp["capabilities"]["has_results"]
    ev = ok(client.post(f"/api/experiments/{eid}/evaluate"))
    assert ev["experiment"]["state"] == "evaluation_completed"
    res = ok(client.get(f"/api/experiments/{eid}/results"))
    pooled = res["evaluation"]["test"]["pooled"]
    preds = ok(client.get(f"/api/experiments/{eid}/predictions"))
    assert pooled["n"] == len(preds) == sum(map(sum, pooled["confusion_matrix"]))
    assert {"training", "validation", "test"} <= set(res["evaluation"]["labels"])
    split = ok(client.get(f"/api/experiments/{eid}/split"))
    test_ids = set(split["membership"][0]["test"])
    assert {p["trial_id"] for p in preds} <= test_ids
    for p in preds:
        assert abs(sum(p["probabilities"]) - 1) < 1e-4


def test_export_metadata_matches_experiment(client, synth_epoched):
    """Requirement 18."""
    eid = prepare(client, synth_epoched["id"], name="export")
    exp = train(client, eid)
    ok(client.post(f"/api/experiments/{eid}/evaluate"))
    items = ["model_config", "training_config", "preprocessing_config", "split_manifest", "metrics", "predictions",
             "history", "confusion", "plots", "checkpoints", "report"]
    rec = ok(client.post(f"/api/experiments/{eid}/export", json={"items": items}))
    blob = client.get(f"/api/exports/{rec['id']}/download").content
    z = zipfile.ZipFile(io.BytesIO(blob))
    manifest = json.loads(z.read("manifest.json"))
    run = exp["active_run"]
    assert manifest["experiment_id"] == eid and manifest["run_id"] == run["id"]
    assert manifest["run_fingerprint"] == run["fingerprint"] and manifest["dataset_included"] is False
    assert manifest["synthetic_data"] is True
    model_cfg = json.loads(z.read("config/model_config.json"))
    assert model_cfg["architecture"] == run["snapshot"]["architecture_config"]
    split = json.loads(z.read("splits/split_manifest.json"))
    assert split["fingerprint"] == run["snapshot"]["split_fingerprint"]
    import hashlib

    for f in manifest["files"]:
        assert hashlib.sha256(z.read(f["path"])).hexdigest() == f["sha256"]
    assert not any(n.endswith("arrays.npz") for n in z.namelist())
    assert b"SYNTHETIC" in z.read("report.html")


def test_export_needs_evaluation_for_metrics(client, trained):
    r = client.post(f"/api/experiments/{trained['id']}/export", json={"items": ["metrics"]})
    assert r.status_code == 409
