"""Preprocessing validation, epoching and training-only / fold-specific fitting (requirements 2, 15)."""

import json

import numpy as np
import pytest

from app.config import settings
from app.core.preprocessing import EpochSet, LearnedTransform, PreprocessingConfig, run_deterministic, validate_config
from app.data.synthetic import generate
from conftest import ok, prepare, train


def test_nyquist_and_cutoff_validation():
    ds = generate("epoched", n_participants=2, trials_per_class=5, n_channels=8, include_frontal=False, sfreq=128)
    cfg = PreprocessingConfig(**{"filter": {"l_freq": 4, "h_freq": 70}})
    v = validate_config(cfg, [("", ds)])
    assert any("Nyquist" in e for e in v["errors"])
    cfg = PreprocessingConfig(**{"filter": {"l_freq": 30, "h_freq": 10}})
    assert any("lower than" in e for e in validate_config(cfg, [("", ds)])["errors"])
    cfg = PreprocessingConfig(**{"filter": {"method": "fir", "phase": "causal"}})
    assert validate_config(cfg, [("", ds)])["errors"]


def test_epoching_from_events_and_rejection():
    ds = generate("continuous", n_participants=2, trials_per_class=6, n_channels=8, include_frontal=True, sfreq=128)
    cfg = PreprocessingConfig(**{"epoching": {"mode": "events", "tmin": -0.2, "tmax": 1.0},
                                 "baseline": {"enabled": True, "tmin": -0.2, "tmax": 0.0},
                                 "artifacts": {"amplitude_enabled": True, "amplitude_threshold_uv": 120}})
    es, rep = run_deterministic(cfg, [("", ds)])
    assert es.X.shape[2] == round(1.2 * 128)
    assert rep["rejection"]["amplitude"] > 0  # synthetic blinks/spikes exceed 120 uV
    assert len(es.X) + rep["rejection"]["total"] == rep["input"]["n_candidates"]
    pre = es.times < 0
    np.testing.assert_allclose(es.X[:, :, pre].mean(axis=2), 0, atol=1e-3)  # baseline corrected


def test_epoched_dataset_cannot_rebuild_epochs():
    ds = generate("epoched", n_participants=2, trials_per_class=5, n_channels=8, include_frontal=False)
    v = validate_config(PreprocessingConfig(**{"epoching": {"mode": "events"}}), [("", ds)])
    assert any("already epoched" in e for e in v["errors"])


def test_preprocessing_blocked_for_dataset_without_events(client):
    """Requirement 2: preprocessing cannot proceed with invalid dataset metadata."""
    import io

    buf = io.BytesIO()
    np.savez(buf, X=np.random.randn(3, 2000).astype(np.float32), sfreq=np.array(128.0))
    ds = ok(client.post("/api/datasets/upload", files=[("files", ("noevents.npz", buf.getvalue()))]))
    eid = ok(client.post("/api/experiments", json={"dataset_id": ds["id"]}))["id"]
    r = client.put(f"/api/experiments/{eid}/split", json={"strategy": "random"})
    assert r.status_code == 409  # no usable trials: split (and so preprocessing) blocked
    r = client.put(f"/api/experiments/{eid}/preprocessing", json={"epoching": {"mode": "events"}})
    assert r.status_code == 409
    r = client.post(f"/api/experiments/{eid}/preprocessing/run")
    assert r.status_code == 409
    exp = ok(client.get(f"/api/experiments/{eid}"))
    assert exp["state"] == "dataset_ready" and not exp["capabilities"]["can_configure_preprocessing"]


def test_invalid_preprocessing_config_rejected(client, synth_epoched):
    eid = ok(client.post("/api/experiments", json={"dataset_id": synth_epoched["id"]}))["id"]
    ok(client.put(f"/api/experiments/{eid}/split", json={"strategy": "random"}))
    r = client.put(f"/api/experiments/{eid}/preprocessing", json={"filter": {"h_freq": 100}})
    assert r.status_code == 422 and "Nyquist" in json.dumps(r.json())


def test_learned_transforms_fitted_on_training_subset_per_fold(client, synth_epoched):
    """Requirement 15: normalisation fitted only on each fold's training subset."""
    eid = prepare(client, synth_epoched["id"], split={"strategy": "kfold", "kfold_mode": "stratified_trial",
                                                      "n_folds": 3},
                  prep={"epoching": {"mode": "crop", "tmin": 0, "tmax": 1.0},
                        "normalization": {"method": "channel_zscore"}}, name="per-fold")
    exp = train(client, eid, epochs=1)
    assert exp["state"] == "artifacts_ready"
    run = exp["active_run"]
    es = EpochSet.load(settings.data_dir / "cache" / "preprocessed" /
                       f"{run['snapshot']['preprocessing_det_fingerprint']}.npz")
    idx = es.index_of()
    means = []
    for f in run["folds"]:
        fd = settings.data_dir / "experiments" / eid / "runs" / run["id"] / f"fold{f['fold']}"
        part = json.loads((fd / "partition.json").read_text())
        assert not set(part["fit"]) & set(part["test"]) and not set(part["val"]) & set(part["test"])
        lt = LearnedTransform.load(fd / "transform")
        fit_X = es.X[[idx[i] for i in part["fit"]]]
        np.testing.assert_allclose(lt.stats["mean"], fit_X.mean(axis=(0, 2)), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(lt.stats["scale"], fit_X.std(axis=(0, 2)) + 1e-8, rtol=1e-5)
        all_mean = es.X.mean(axis=(0, 2))
        assert not np.allclose(lt.stats["mean"], all_mean, rtol=1e-6, atol=1e-7)  # test data excluded
        assert f["transform_report"]["n_fit_trials"] == len(part["fit"])
        means.append(lt.stats["mean"])
    assert not np.allclose(means[0], means[1])  # fitted independently per fold


def test_ica_fitted_on_training_partition_only(client, synth_continuous, monkeypatch):
    from app.services import experiments as xsvc

    eid = ok(client.post("/api/experiments", json={"dataset_id": synth_continuous["id"]}))["id"]
    ok(client.put(f"/api/experiments/{eid}/split", json={"strategy": "group", "n_test_groups": 1}))
    cfg = {"filter": {"l_freq": 1.0, "h_freq": 40}, "epoching": {"mode": "events", "tmin": 0, "tmax": 1.0},
           "ica": {"enabled": True, "n_components": 6}}
    seen = {}
    real = xsvc.fit_ica

    def spy(X, ch, sf, st, rows=None):
        ica, used = real(X, ch, sf, st, rows=rows)
        seen.update(rows=set(rows), used=set(used))
        return ica, used

    monkeypatch.setattr(xsvc, "fit_ica", spy)
    res = ok(client.post(f"/api/experiments/{eid}/preprocessing/ica", json=cfg))
    split = ok(client.get(f"/api/experiments/{eid}/split"))
    assert seen["used"] <= seen["rows"]  # only training trials
    assert len(seen["used"]) == res["reference"]["n_fit_trials"] <= split["assignment"]["folds"][0]["n_train"]
    assert res["diagnostics"]["n_components"] == 6
    assert any("Explained variance alone" in n for n in res["diagnostics"]["notes"])
    # selecting a component, then saving, requires the matching reference fit
    cfg["ica"]["exclude"] = [0]
    ok(client.put(f"/api/experiments/{eid}/preprocessing", json=cfg))
    out = ok(client.post(f"/api/experiments/{eid}/preprocessing/run"))
    assert out["result"]["learned_preview"]["ica"]["excluded"]


def test_ica_fits_a_bounded_subset_and_applies_in_batches(monkeypatch):
    """Large training sets: ICA is fitted on a seeded subset of trials and applied batch-wise (OOM fix)."""
    import numpy as np

    from app.core import preprocessing as pp

    X = (np.random.default_rng(0).standard_normal((40, 4, 128)) * 10).astype(np.float32)
    cfg = pp.PreprocessingConfig(ica={"enabled": True, "n_components": 4, "exclude": [0]},
                                 normalization={"method": "none"})
    chans = ["C3", "Cz", "C4", "Pz"]
    full = pp.LearnedTransform(cfg, chans, 128.0)
    assert full.fit(X)["ica"]["n_fit_trials"] == 40  # small data: every training trial, as before
    one_shot = full.transform(X)
    monkeypatch.setattr(pp, "MAX_CHUNK_VALUES", 4 * 128 * 10)  # budget of 10 trials per MNE call
    small = pp.LearnedTransform(cfg, chans, 128.0)
    assert small.fit(X)["ica"]["n_fit_trials"] == 10
    small.ica = full.ica  # same unmixing: batched apply must equal the one-shot apply
    np.testing.assert_allclose(small.transform(X), one_shot, rtol=1e-5, atol=1e-4)


def test_epochs_streamed_to_disk_match_in_memory(tmp_path):
    """Large datasets: epochs are written per recording to a memory-mapped file with identical content."""
    ds = generate("continuous", n_participants=3, trials_per_class=6, n_channels=8, include_frontal=True, sfreq=128)
    cfg = PreprocessingConfig(**{"epoching": {"mode": "events", "tmin": -0.2, "tmax": 1.0},
                                 "baseline": {"enabled": True, "tmin": -0.2, "tmax": 0.0},
                                 "artifacts": {"amplitude_enabled": True, "amplitude_threshold_uv": 120}})
    mem, rep_mem = run_deterministic(cfg, [("", ds)])
    disk, rep_disk = run_deterministic(cfg, [("", ds)], out=tmp_path / "x.X.f32")
    assert isinstance(disk.X, np.memmap) and not isinstance(mem.X, np.memmap)
    np.testing.assert_array_equal(np.asarray(disk.X), mem.X)
    assert list(disk.trial_ids) == list(mem.trial_ids) and rep_disk["rejection"] == rep_mem["rejection"]
    disk.save(tmp_path / "x.npz")  # X already in place: only metadata is written
    back = EpochSet.load(tmp_path / "x.npz")
    assert isinstance(back.X, np.memmap)
    np.testing.assert_array_equal(np.asarray(back.X), mem.X)
    mem.save(tmp_path / "m.npz")  # in-memory X is written next to the metadata
    np.testing.assert_array_equal(np.asarray(EpochSet.load(tmp_path / "m.npz").X), mem.X)


def test_learned_stages_in_chunks_match_one_shot(monkeypatch):
    from app.core import preprocessing as pp

    X = (np.random.default_rng(1).standard_normal((30, 4, 64)) * 20 + 5).astype(np.float32)
    for method in ("channel_zscore", "global_zscore", "trial_zscore"):  # exact streamed statistics
        cfg = PreprocessingConfig(normalization={"method": method})
        one = LearnedTransform(cfg, ["a", "b", "c", "d"], 128.0)
        one.fit(X, rows=list(range(20)))
        want = one.transform(X)
        with monkeypatch.context() as m:
            m.setattr(pp, "MAX_CHUNK_VALUES", 4 * 64 * 3)  # 3 trials per chunk
            many = LearnedTransform(cfg, ["a", "b", "c", "d"], 128.0)
            many.fit(X, rows=list(range(20)))
            got = many.transform(X, rows=list(range(30)))
        for k in one.stats:
            np.testing.assert_allclose(many.stats[k], one.stats[k], rtol=1e-6)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
    # robust quantiles over the budget use a reported, seeded subset of the fit rows
    with monkeypatch.context() as m:
        m.setattr(pp, "MAX_CHUNK_VALUES", 4 * 64 * 3)
        rob = LearnedTransform(PreprocessingConfig(normalization={"method": "robust_channel"}), list("abcd"), 128.0)
        assert rob.fit(X, rows=list(range(20)))["normalization_fit_trials"] == 3
    # statistics come from the fit rows only
    lt = LearnedTransform(PreprocessingConfig(normalization={"method": "channel_zscore"}), list("abcd"), 128.0)
    lt.fit(X, rows=list(range(20)))
    np.testing.assert_allclose(lt.stats["mean"], X[:20].astype(np.float64).mean(axis=(0, 2)), rtol=1e-9)


def test_preview_uses_recording_subset_and_run_uses_everything(client, synth_continuous, monkeypatch):
    from app.services import experiments as xsvc

    monkeypatch.setattr(xsvc, "PREVIEW_MAX_RECORDINGS", 2)
    eid = ok(client.post("/api/experiments", json={"dataset_id": synth_continuous["id"]}))["id"]
    ok(client.put(f"/api/experiments/{eid}/split", json={"strategy": "random"}))
    cfg = {"epoching": {"mode": "events", "tmin": 0, "tmax": 1.0}}
    pv = ok(client.post(f"/api/experiments/{eid}/preprocessing/preview", json={"config": cfg}))
    scope = pv["report"]["scope"]
    n_rec = synth_continuous["summary"]["n_recordings"]
    assert scope == {"subset": True, "recordings": 2, "of": n_rec, "participants": 2}
    ok(client.put(f"/api/experiments/{eid}/preprocessing", json=cfg))
    res = ok(client.post(f"/api/experiments/{eid}/preprocessing/run"))["result"]
    assert res["report"]["scope"]["subset"] is False
    assert res["report"]["input"]["n_candidates"] == synth_continuous["summary"]["n_trials"]
    # once the full result exists, the preview shows it (exact counts)
    pv = ok(client.post(f"/api/experiments/{eid}/preprocessing/preview", json={"config": cfg}))
    assert pv["report"]["scope"]["subset"] is False
