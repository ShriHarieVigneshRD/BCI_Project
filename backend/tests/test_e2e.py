"""Requirement 20: complete workflow end-to-end on a small, explicitly SYNTHETIC continuous dataset.

Synthetic data are used for software testing only; the metrics below are not evidence of real-world
EEG decoding performance.
"""

import json

from conftest import ok


def test_complete_workflow(client):
    ds = ok(client.post("/api/datasets/synthetic", json={"kind": "continuous", "n_participants": 5, "n_classes": 2,
                                                         "trials_per_class": 16, "n_channels": 8,
                                                         "include_frontal": True, "seed": 11}))
    assert ds["synthetic"] and ds["structure"] == "continuous"

    # 1. Dataset & Splits
    exp = ok(client.post("/api/experiments", json={"name": "e2e"}))
    eid = exp["id"]
    assert exp["state"] == "dataset_required"
    ok(client.put(f"/api/experiments/{eid}/dataset", json={"dataset_id": ds["id"]}))
    cand = ok(client.get(f"/api/experiments/{eid}/candidates"))
    assert cand["has_groups"] and len(cand["groups"]) == 5
    prev = ok(client.post(f"/api/experiments/{eid}/split/preview",
                          json={"strategy": "group", "test_groups": cand["groups"][:1]}))
    assert prev["integrity"]["ok"] and prev["integrity"]["group_level"]
    sp = ok(client.put(f"/api/experiments/{eid}/split", json={"strategy": "group", "test_groups": cand["groups"][:1]}))
    assert sp["experiment"]["state"] == "split_configured"

    # 2. Preprocessing Studio
    cfg = {"filter": {"l_freq": 4, "h_freq": 40, "notch_enabled": True, "notch_freqs": [50]},
           "epoching": {"mode": "events", "tmin": -0.25, "tmax": 1.5},
           "baseline": {"enabled": True, "tmin": -0.25, "tmax": 0.0},
           "artifacts": {"amplitude_enabled": True, "amplitude_threshold_uv": 250},
           "resample": {"enabled": True, "target_sfreq": 64},
           "normalization": {"method": "channel_zscore"}}
    pv = ok(client.post(f"/api/experiments/{eid}/preprocessing/preview", json={"config": cfg}))
    assert pv["ok"] and pv["psd_raw"]["freqs"] and pv["processed"]["sfreq"] == 64
    ok(client.put(f"/api/experiments/{eid}/preprocessing", json=cfg))
    run = ok(client.post(f"/api/experiments/{eid}/preprocessing/run"))
    assert run["experiment"]["state"] == "preprocessing_completed"
    shape = run["result"]["report"]["output"]["shape"]
    assert shape[1:] == [1, 10, 112]  # [N, 1, C, T]: 10 channels, 1.75 s at 64 Hz

    # 3. Architecture Explorer: configuration, untrained journey, training
    arch = ok(client.get(f"/api/experiments/{eid}/architecture"))
    assert arch["config"]["n_channels"] == 10 and arch["config"]["n_times"] == 112
    ok(client.put(f"/api/experiments/{eid}/architecture", json={"config": arch["config"]}))
    trials = ok(client.get(f"/api/experiments/{eid}/trials"))["trials"]
    j = ok(client.get(f"/api/experiments/{eid}/journey", params={"trial_id": trials[0]["trial_id"],
                                                                 "mode": "untrained"}))
    assert j["banner"].startswith("UNTRAINED")
    assert client.get(f"/api/experiments/{eid}/kernels").status_code == 409

    ok(client.post(f"/api/experiments/{eid}/train", json={"config": {"epochs": 6, "batch_size": 16, "device": "cpu",
                                                                     "patience": 10}}))
    events = []
    with client.stream("GET", f"/api/experiments/{eid}/progress") as s:
        for line in s.iter_lines():
            if line.startswith("data:"):
                ev = json.loads(line[5:])
                events.append(ev)
                if ev["type"] in ("done", "failed", "cancelled"):
                    break
    assert events[0]["type"] == "snapshot"
    epochs = [e for e in events if e["type"] == "epoch"]
    assert len(epochs) == 6 and all("val_loss" in e for e in epochs)
    assert events[-1]["type"] == "done" and events[-1]["artifacts_ready"] is True

    exp = ok(client.get(f"/api/experiments/{eid}"))
    assert exp["state"] == "artifacts_ready" and exp["capabilities"]["can_inspect_learned"]

    # post-training tools
    tid = next(t["trial_id"] for t in ok(client.get(f"/api/experiments/{eid}/trials"))["trials"]
               if t["partition"] == "test")
    j = ok(client.get(f"/api/experiments/{eid}/journey", params={"trial_id": tid}))
    assert j["mode"] == "trained" and j["provenance"]["partition"] == "test"
    conv = ok(client.post(f"/api/experiments/{eid}/convolution",
                          json={"layer": "separable_pointwise", "trial_id": tid, "channel": 0, "position": 2,
                                "compare_trial_id": trials[1]["trial_id"]}))
    assert conv["verified"] and "comparison" in conv
    arena = ok(client.post(f"/api/experiments/{eid}/arena", json={"trial_id": tid, "demo_stride": 2}))
    assert arena["demo"]["output_length"] > 0
    br = ok(client.post(f"/api/experiments/{eid}/breaker", json={"changes": {"D": 1}}))
    assert br["valid"] and not br["checkpoint_valid"]

    # Results & Predictions
    assert client.get(f"/api/experiments/{eid}/results").status_code == 409
    ev = ok(client.post(f"/api/experiments/{eid}/evaluate"))
    pooled = ev["evaluation"]["test"]["pooled"]
    assert pooled["n"] > 0 and len(pooled["confusion_matrix"]) == 2

    # Battle Arena: matched variant, trained, compared
    var = ok(client.post(f"/api/experiments/{eid}/variants", json={
        "name": "e2e variant", "architecture": {**arch["config"], "F1": 4, "F2": 8, "preset": "custom"}}))
    assert var["state"] == "architecture_configured" and var["split_fingerprint"] == exp["split_fingerprint"]
    ok(client.post(f"/api/experiments/{var['id']}/train", json={"config": {"epochs": 3, "device": "cpu"}}))
    from app.services.jobs import manager

    assert manager.wait_idle(300)
    ok(client.post(f"/api/experiments/{var['id']}/evaluate"))
    cmp = ok(client.post("/api/experiments/compare", json={"ids": [eid, var["id"]]}))
    assert cmp["matched_protocol"] and cmp["disagreement"] is not None

    # Tensor Detective (pre-training validation path, no data needed)
    rep = ok(client.post("/api/models/validate", json={"config": arch["config"]}))
    assert rep["valid"] and rep["torch_verified"]
    m = ok(client.post("/api/models/coordinate", json={"config": arch["config"], "layer": "pool2",
                                                       "coord": [0, 0, 0]}))
    assert m["receptive_field_samples"] == rep["receptive_field_samples"]

    # Experiments & Exports
    reg = ok(client.get("/api/experiments", params={"search": "e2e"}))
    assert {r["id"] for r in reg} >= {eid, var["id"]}
    ex = ok(client.post(f"/api/experiments/{eid}/export", json={"items": ["report", "metrics", "predictions"]}))
    assert {f["path"] for f in ex["manifest"]["files"]} >= {"report.html", "results/predictions.csv"}
