"""Phase 2: dataset upload, format validation and metadata checks (requirement 2 partially)."""

import io
import json

import numpy as np
import pandas as pd
import pytest

from conftest import ok


def _npz(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def _npy(a) -> bytes:
    buf = io.BytesIO()
    np.save(buf, a)
    return buf.getvalue()


def test_formats_listed(client):
    fmts = ok(client.get("/api/meta/formats"))
    names = " ".join(f["format"] for f in fmts)
    for n in ("EDF", "FIF", "EEGLAB", ".npz", ".npy", "CSV"):
        assert n in names


def test_npz_epoched_upload(client):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4, 64)).astype(np.float32) * 10
    y = np.array([0, 1, 2] * 10)
    data = _npz(X=X, y=y, sfreq=np.array(128.0), ch_names=np.array(["C3", "Cz", "C4", "Pz"]),
                groups=np.array(["A", "B", "C"] * 10), class_names=np.array(["l", "r", "f"]))
    rec = ok(client.post("/api/datasets/upload", files=[("files", ("demo.npz", data))]))
    s = rec["summary"]
    assert rec["structure"] == "epoched" and s["n_trials"] == 30 and s["n_channels"] == 4
    assert s["class_distribution"] == {"f": 10, "l": 10, "r": 10}
    assert s["n_participants"] == 3 and rec["synthetic"] is False


def test_npz_missing_sfreq_rejected(client):
    data = _npz(X=np.zeros((10, 2, 32), np.float32) + 1, y=np.arange(10) % 2)
    r = client.post("/api/datasets/upload", files=[("files", ("x.npz", data))])
    assert r.status_code == 422
    assert any(i["code"] == "sfreq_missing" for i in r.json()["issues"])


def test_nan_rejected(client):
    X = np.ones((10, 2, 32), np.float32)
    X[0, 0, 0] = np.nan
    r = client.post("/api/datasets/upload", files=[("files", ("x.npz", _npz(X=X, y=np.arange(10) % 2,
                                                                             sfreq=np.array(100.0))))])
    assert r.status_code == 422 and any(i["code"] == "non_finite" for i in r.json()["issues"])


def test_npy_with_companion_labels(client):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((12, 3, 40)).astype(np.float32)
    r = client.post("/api/datasets/upload", data={"sfreq": "100", "ch_names": "a,b,c"},
                    files=[("files", ("eeg.npy", _npy(X))), ("files", ("labels.npy", _npy(np.arange(12) % 2)))])
    rec = ok(r)
    assert rec["summary"]["ch_names"] == ["a", "b", "c"] and rec["summary"]["n_classes"] == 2


def test_csv_long_format_epoched(client):
    rows = []
    for trial in range(8):
        for i in range(32):
            rows.append({"trial": trial, "label": "L" if trial % 2 else "R", "participant": f"p{trial % 2}",
                         "time": i / 64, "C3": np.sin(i + trial), "C4": np.cos(i)})
    data = pd.DataFrame(rows).to_csv(index=False).encode()
    rec = ok(client.post("/api/datasets/upload", files=[("files", ("eeg.csv", data))]))
    assert rec["summary"]["sfreq"] == 64 and rec["summary"]["n_trials"] == 8
    assert rec["summary"]["epoch_samples"] == 32


def test_csv_continuous_with_markers(client):
    n = 640
    df = pd.DataFrame({"time": np.arange(n) / 64, "C3": np.random.randn(n), "C4": np.random.randn(n),
                       "marker": [""] * n})
    df.loc[[100, 300, 500], "marker"] = ["left", "right", "left"]
    rec = ok(client.post("/api/datasets/upload", files=[("files", ("raw.csv", df.to_csv(index=False).encode()))]))
    assert rec["structure"] == "continuous" and rec["summary"]["n_trials"] == 3


def test_fif_raw_and_epochs(client, tmp_path):
    import mne

    info = mne.create_info(["C3", "Cz", "C4"], 100.0, "eeg")
    raw = mne.io.RawArray(np.random.randn(3, 2000) * 1e-5, info, verbose=False)
    raw.set_annotations(mne.Annotations([2, 6, 10, 14], [0.1] * 4, ["left", "right", "left", "right"]))
    p = tmp_path / "sub-01_ses-01_raw.fif"
    raw.save(p, verbose=False)
    rec = ok(client.post("/api/datasets/upload", files=[("files", (p.name, p.read_bytes()))]))
    s = rec["summary"]
    assert rec["structure"] == "continuous" and s["n_trials"] == 4 and s["participants"] == ["01"]
    first = ok(client.get(f"/api/datasets/{rec['id']}/trials"))[0]["trial_id"]
    assert first.startswith("sub-01_ses-01:")  # trial IDs carry recording provenance
    assert 1 < np.abs(np.array(ok(client.get(f"/api/datasets/{rec['id']}/signal",
                                              params={"trial_id": first}))["data"])).mean() < 100  # uV
    events = np.array([[200 * i, 0, 1 + i % 2] for i in range(1, 9)])
    ep = mne.Epochs(raw, events, {"a": 1, "b": 2}, tmin=0, tmax=0.5, baseline=None, verbose=False, preload=True)
    p2 = tmp_path / "s-epo.fif"
    ep.save(p2, verbose=False)
    rec2 = ok(client.post("/api/datasets/upload", files=[("files", (p2.name, p2.read_bytes()))]))
    assert rec2["structure"] == "epoched" and rec2["summary"]["class_distribution"] == {"a": 4, "b": 4}


def test_edf_upload(client, tmp_path):
    mne = pytest.importorskip("mne")
    pytest.importorskip("edfio")
    info = mne.create_info(["C3", "C4"], 128.0, "eeg")
    raw = mne.io.RawArray(np.random.randn(2, 128 * 20) * 1e-5, info, verbose=False)
    raw.set_annotations(mne.Annotations([2, 5, 8], [0, 0, 0], ["T1", "T2", "T1"]))
    p = tmp_path / "S001R04.edf"
    raw.export(p, verbose=False)
    rec = ok(client.post("/api/datasets/upload", files=[("files", (p.name, p.read_bytes()))]))
    assert rec["summary"]["n_trials"] == 3 and rec["summary"]["participants"] == ["001"]


def test_eeglab_raw_and_epochs(client, tmp_path):
    mne = pytest.importorskip("mne")
    pytest.importorskip("eeglabio")
    info = mne.create_info(["C3", "Cz", "C4"], 100.0, "eeg")
    raw = mne.io.RawArray(np.random.randn(3, 3000) * 1e-5, info, verbose=False)
    raw.set_annotations(mne.Annotations([2, 6, 10, 14], [0.1] * 4, ["left", "right", "left", "right"]))
    p = tmp_path / "sub-02_raw.set"
    raw.export(p, verbose=False)
    rec = ok(client.post("/api/datasets/upload", files=[("files", (p.name, p.read_bytes()))]))
    assert rec["format"] == "eeglab" and rec["structure"] == "continuous"
    assert rec["summary"]["n_trials"] == 4 and rec["summary"]["participants"] == ["02"]
    events = np.array([[200 * i, 0, 1 + i % 2] for i in range(1, 9)])
    ep = mne.Epochs(raw, events, {"a": 1, "b": 2}, tmin=0, tmax=0.5, baseline=None, preload=True, verbose=False)
    p2 = tmp_path / "sub-03_epo.set"
    ep.export(p2, verbose=False)
    rec2 = ok(client.post("/api/datasets/upload", files=[("files", (p2.name, p2.read_bytes()))]))
    assert rec2["structure"] == "epoched" and rec2["summary"]["n_epochs"] == 8


def test_continuous_without_events_is_flagged(client):
    data = _npz(X=np.random.randn(2, 1000).astype(np.float32), sfreq=np.array(100.0))
    rec = ok(client.post("/api/datasets/upload", files=[("files", ("cont.npz", data))]))
    assert rec["trainable"] is False
    assert any(i["code"] == "no_events" for i in rec["issues"])


def test_unsafe_filename_and_extension_rejected(client):
    r = client.post("/api/datasets/upload", files=[("files", ("../../evil.exe", b"MZ"))])
    assert r.status_code == 422


def test_synthetic_is_labelled(synth_epoched):
    assert synth_epoched["synthetic"] is True
    assert synth_epoched["name"].startswith("SYNTHETIC")
    assert any("not evidence" in n for n in synth_epoched["notes"])


def test_api_never_exposes_filesystem_paths(client, synth_epoched):
    text = json.dumps(ok(client.get(f"/api/datasets/{synth_epoched['id']}")))
    from app.config import settings

    assert str(settings.data_dir) not in text and ":\\" not in text
