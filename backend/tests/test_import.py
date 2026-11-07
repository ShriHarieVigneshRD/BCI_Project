"""Multi-participant dataset import (folder / files / local path) and end-to-end compatibility.

Acceptance: a nested dataset with several participants, several runs and associated event files is
imported, the three existing split methods work on it, and every downstream step (preprocessing,
training, Signal Journey, kernels, results, export) uses the right dataset, trial identities, labels
and participant IDs. Signals are explicitly SYNTHETIC.
"""

import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from app.config import settings
from app.core.preprocessing import EpochSet
from app.data.entities import infer_entities
from app.data.manifest import block_key, build_trials, lookup, suggest_mapping
from app.data.scanner import classify_name, parse_event_table
from app.services.imports import safe_relpath
from conftest import ok
from nested_fixture import build_bids_like, build_block_design

FAST = {"epochs": 2, "batch_size": 16, "device": "cpu", "patience": 5, "seed": 3}


@pytest.fixture(scope="module")
def nested(tmp_path_factory):
    root = tmp_path_factory.mktemp("nested")
    return {"bids": root / "bids", "bids_expected": build_bids_like(root / "bids"),
            "blocks": root / "blocks", "blocks_expected": build_block_design(root / "blocks")}


def import_local(client, path: Path, edits: dict | None = None) -> dict:
    s = ok(client.post("/api/imports", params={"wait": "true"}, json={"mode": "local_path", "local_path": str(path)}))
    assert s["status"] == "review", s
    if edits:
        ok(client.put(f"/api/imports/{s['id']}/manifest", json=edits))
    done = ok(client.post(f"/api/imports/{s['id']}/import", params={"wait": "true"}))
    assert done["status"] == "done", done
    return ok(client.get(f"/api/datasets/{done['dataset_id']}"))


def train_wait(client, eid: str, **over) -> dict:
    from app.services.jobs import manager

    ok(client.post(f"/api/experiments/{eid}/train", json={"config": {**FAST, **over}}))
    assert manager.wait_idle(600)
    return ok(client.get(f"/api/experiments/{eid}"))


# ---------------------------------------------------------------------------
# Unit level: classification, IDs, event files, mapping
# ---------------------------------------------------------------------------


def test_file_classification_never_counts_event_files_as_recordings():
    assert classify_name("sub-01/ses-01/eeg/sub-01_ses-01_task-mi_run-01_eeg.edf")[0] == "eeg"
    assert classify_name("sub-01/ses-01/eeg/sub-01_ses-01_task-mi_run-01_events.tsv")[0] == "events"
    assert classify_name("S01/S01R01_events.csv")[0] == "events"
    assert classify_name("participants.tsv")[0] == "metadata"
    assert classify_name("sub-01/eeg/sub-01_channels.tsv")[0] == "metadata"
    assert classify_name("sub-01/eeg/sub-01_task-mi_eeg.json")[0] == "metadata"
    assert classify_name("data/rec.set")[0] == "eeg" and classify_name("data/rec.fdt")[0] == "companion"
    assert classify_name("data/rec.vhdr")[0] == "eeg" and classify_name("data/rec.vmrk")[0] == "companion"
    assert classify_name("x/labels.npy")[0] == "companion"
    assert classify_name("stimuli/arrow.png")[0] == "ignored"
    assert classify_name(".git/config")[0] == "ignored"
    assert "MATLAB" in (classify_name("old/data.mat")[1] or "")


@pytest.mark.parametrize("path,expected", [
    ("sub-03/ses-02/eeg/sub-03_ses-02_task-mi_run-04_eeg.edf", ("03", "02", "04", "mi")),
    ("S007/S007R12.edf", ("007", None, "12", None)),
    ("A05E.gdf", ("A05", "E", None, None)),
    ("subject_2/session1/run-3.bdf", ("2", "1", "3", None)),
    ("Lab2.edf", ("Lab2", None, None, None)),
])
def test_entity_inference(path, expected):
    e = infer_entities(path)
    assert (e.participant, e.session, e.run, e.task) == expected


def test_event_table_parsing(tmp_path):
    p = tmp_path / "a_events.tsv"
    p.write_text("onset\tduration\ttrial_type\tvalue\n1.5\t2\tleft\t1\n3.0\tn/a\tright\t2\n4.0\t0\tn/a\t3\n")
    unit, evs, notes = parse_event_table(p)
    assert unit == "seconds" and evs == [(1.5, "left", 2.0), (3.0, "right", 0.0)] and "skipped" in notes[0]
    q = tmp_path / "b_events.csv"
    q.write_text("sample,label\n100,7\n250,8.0\n")
    unit, evs, _ = parse_event_table(q)
    assert unit == "samples" and evs == [(100.0, "7", 0.0), (250.0, "8", 0.0)]
    r = tmp_path / "c_events.csv"
    r.write_text("onset_ms,event\n1500,go\n")
    assert parse_event_table(r)[1] == [(1.5, "go", 0.0)]


def test_mapping_suggestions_pair_start_end_blocks():
    from collections import Counter

    descs = ["Recording starts", "Date: 2022-5-25  TimeStart- 11:37:7", "2022_05_25_11_30_58", "NSR-start",
             "NSR end"] + [f"FC-{i} {w}" for i in range(1, 5) for w in ("start", "end")] + \
            [f"V{i} {w}" for i in range(1, 5) for w in ("start", "end")]
    mapping = suggest_mapping(Counter({(None, d): 1 for d in descs}))
    by = {m.event: m for m in mapping}
    assert by["Recording starts"].action == "ignore" and by["2022_05_25_11_30_58"].action == "ignore"
    assert by["Date: 2022-5-25  TimeStart- 11:37:7"].action == "ignore"
    assert (by["FC-3 start"].action, by["FC-3 start"].label) == ("segment_start", "FC")
    assert (by["V2 end"].action, by["V2 end"].label) == ("segment_end", "V")
    assert (by["NSR-start"].action, by["NSR-start"].label) == ("segment_start", "NSR")
    assert block_key("FC-1 start") == block_key("FC-1 end") != block_key("FC-10 end")
    events = [(0, "Recording starts", 0), (1280, "NSR-start", 0), (16640, "NSR end", 0), (19584, "FC-1 start", 0),
              (20224, "FC-1 end", 0), (20480, "V1 start", 0), (28544, "V1 end", 0), (30000, "V2 start", 0)]
    trials, stats = build_trials(events, None, lookup(mapping))
    assert [(s, lbl, d) for s, lbl, d in trials] == [(1280, "NSR", 15360), (19584, "FC", 640), (20480, "V", 8064)]
    assert stats["unpaired_starts"] == 1  # V2 never ends -> dropped and reported


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------


def test_local_path_validation(client, tmp_path, nested, monkeypatch):
    def create(path):
        return client.post("/api/imports", json={"mode": "local_path", "local_path": path})

    assert create("relative/folder").status_code == 422
    assert create(str(tmp_path / "missing")).status_code == 422
    f = tmp_path / "file.edf"
    f.write_bytes(b"0")
    assert create(str(f)).status_code == 422
    assert create(r"\\server\share\eeg").status_code == 422
    assert create(str(Path(tmp_path.anchor))).status_code == 422
    assert create(str(settings.data_dir)).status_code == 422
    monkeypatch.setattr(settings, "import_roots", [str(tmp_path / "allowed")])
    r = create(str(nested["bids"]))
    assert r.status_code == 422 and "allowed import locations" in r.json()["detail"]
    # a request from another machine can never make the server read its local folders
    from fastapi.testclient import TestClient

    remote = TestClient(client.app, client=("192.168.1.20", 50000))
    r = remote.post("/api/imports", json={"mode": "local_path", "local_path": str(nested["bids"])})
    assert r.status_code == 403 and "this machine" in r.json()["detail"]


def test_upload_paths_are_sanitised(client):
    assert safe_relpath("study/sub-01/a.edf") == "study/sub-01/a.edf"
    for bad in ("../secret.edf", "a/../../b.edf", "C:/Windows/x.edf", ""):
        with pytest.raises(ValueError):
            safe_relpath(bad)
    s = ok(client.post("/api/imports", json={"mode": "files"}))
    r = client.post(f"/api/imports/{s['id']}/files", data={"paths": json.dumps(["../evil.edf"])},
                    files=[("files", ("evil.edf", b"0"))])
    assert r.status_code == 422
    r = client.post(f"/api/imports/{s['id']}/files", data={"paths": json.dumps(["notes/readme.pdf"])},
                    files=[("files", ("readme.pdf", b"0"))])
    assert ok(r)["skipped"] == ["notes/readme.pdf"]  # unrelated files are never stored


# ---------------------------------------------------------------------------
# Scan -> review -> import
# ---------------------------------------------------------------------------


def test_review_counts_are_dynamic_and_edits_apply(client, nested):
    exp = nested["bids_expected"]
    s = ok(client.post("/api/imports", params={"wait": "true"},
                       json={"mode": "local_path", "local_path": str(nested["bids"])}))
    m = s["manifest"]
    summ = m["summary"]
    assert s["status"] == "review" and m["importable"]
    assert summ["n_recordings"] == exp["recordings"] + 1          # +1 resting recording (no events)
    assert summ["n_included"] == exp["recordings"]                 # ... excluded automatically
    assert summ["n_participants"] == exp["participants"] and summ["n_trials"] == exp["events"]
    rest = next(r for r in m["recordings"] if r["task"] == "rest")
    assert not rest["included"] and rest["exclude_reason"] == "no events"
    roles = {f["role"] for f in m["files"]}
    assert {"eeg", "events", "metadata", "ignored"} <= roles
    assert sum(1 for f in m["files"] if f["role"] == "eeg") == exp["recordings"] + 1  # event files are not recordings
    assert any(i["code"] == "orphan_event_files" for i in m["issues"])
    assert all(r["events"]["kind"] == "events_file" for r in m["recordings"] if r["included"])
    p01 = next(p for p in summ["participants"] if p["participant"] == "01")
    assert p01["n_recordings"] == 2 and p01["n_trials"] == exp["per_participant"]["01"] and p01["attributes"]["age"] == 20
    assert m["metadata"]["power_line_hz"] == 50 and m["metadata"]["bad_channels"]["0"] == ["FCz"]
    # edits: rename a participant, relabel a class, exclude one recording -> counts recomputed
    first = m["recordings"][0]
    edited = ok(client.put(f"/api/imports/{s['id']}/manifest", json={
        "recordings": [{"index": first["index"], "participant": "01x"},
                       {"index": m["recordings"][1]["index"], "included": False}],
        "mapping": [{"task": "mi", "event": "right_hand", "action": "class", "label": "right"}]}))["manifest"]
    es = edited["summary"]
    assert es["n_included"] == exp["recordings"] - 1
    assert es["n_trials"] == exp["events"] - m["recordings"][1]["n_trials"]
    assert set(es["classes"]) == {"left_hand", "right"}
    assert edited["recordings"][0]["recording_id"].startswith("sub-01x_")
    assert {p["participant"] for p in es["participants"]} >= {"01x", "02"}
    # including a recording without events is an error until excluded again
    bad = ok(client.put(f"/api/imports/{s['id']}/manifest",
                        json={"recordings": [{"index": rest["index"], "included": True}]}))["manifest"]
    assert not bad["importable"]
    r = client.post(f"/api/imports/{s['id']}/import", params={"wait": "true"})
    assert r.status_code == 422 and "cannot provide trials" in r.json()["detail"]


def test_folder_upload_matches_local_import(client, nested):
    root = nested["bids"]
    files = [p for p in root.rglob("*") if p.is_file()]
    listing = [{"path": f"bids/{p.relative_to(root).as_posix()}", "size": p.stat().st_size} for p in files]
    s = ok(client.post("/api/imports", json={"mode": "folder", "name": "uploaded nested"}))
    plan = ok(client.post(f"/api/imports/{s['id']}/plan", json={"files": listing}))
    assert plan["root_label"] == "bids" and plan["counts"]["ignored"] >= 2
    to_send = [f for f in plan["files"] if f["role"] != "ignored"]
    for i in range(0, len(to_send), 7):  # several batches, relative folder structure preserved
        batch = to_send[i:i + 7]
        r = client.post(f"/api/imports/{s['id']}/files",
                        data={"paths": json.dumps([f["path"] for f in batch])},
                        files=[("files", (Path(f["path"]).name,
                                          (root / Path(f["path"]).relative_to("bids")).read_bytes())) for f in batch])
        ok(r)
    scanned = ok(client.post(f"/api/imports/{s['id']}/scan", params={"wait": "true"}))
    m = scanned["manifest"]
    assert m["summary"]["n_participants"] == 4 and m["summary"]["n_trials"] == nested["bids_expected"]["events"]
    assert not any(f["path"].startswith("bids/") for f in m["files"])  # root folder stripped
    done = ok(client.post(f"/api/imports/{s['id']}/import", params={"wait": "true"}))
    ds = ok(client.get(f"/api/datasets/{done['dataset_id']}"))
    assert ds["source"]["mode"] == "folder" and ds["summary"]["n_trials"] == nested["bids_expected"]["events"]


# ---------------------------------------------------------------------------
# Acceptance: nested dataset through every screen with all split methods
# ---------------------------------------------------------------------------


def test_acceptance_nested_dataset_all_split_methods(client, nested):
    ds = import_local(client, nested["bids"])
    exp = nested["bids_expected"]
    summ = ds["summary"]
    assert summ["n_recordings"] == exp["recordings"] and summ["n_participants"] == 4
    assert summ["n_trials"] == exp["events"] and summ["n_runs"] == 2
    assert ds["manifest"]["summary"]["n_trials"] == summ["n_trials"]
    trials = ok(client.get(f"/api/datasets/{ds['id']}/trials"))
    assert all(":" in t["trial_id"] and t["group"] in {"01", "02", "03", "04"} for t in trials)

    previous = None
    for split in ({"strategy": "random", "test_ratio": 0.25},
                  {"strategy": "group", "test_groups": ["04"]},
                  {"strategy": "kfold", "kfold_mode": "stratified_group", "n_folds": 2}):
        e = ok(client.post("/api/experiments", json={"name": f"nested {split['strategy']}", "dataset_id": ds["id"]}))
        eid = e["id"]
        cand = ok(client.get(f"/api/experiments/{eid}/candidates"))
        assert cand["n"] == exp["events"] and cand["has_groups"]
        assert {p["participant"]: p["n_recordings"] for p in cand["participants"]} == {f"0{i}": 2 for i in range(1, 5)}
        sp = ok(client.put(f"/api/experiments/{eid}/split", json=split))
        assert sp["integrity"]["ok"]
        defaults = ok(client.get(f"/api/experiments/{eid}/preprocessing/defaults"))
        cfg = defaults["config"]
        assert cfg["filter"]["notch_enabled"] and cfg["channels"]["bad_channels"] == ["FCz"]
        cfg["epoching"].update({"mode": "events", "tmin": 0.0, "tmax": 1.0})
        ok(client.put(f"/api/experiments/{eid}/preprocessing", json=cfg))
        res = ok(client.post(f"/api/experiments/{eid}/preprocessing/run"))["result"]
        assert {r["participant"] for r in res["report"]["by_participant"]} == {"01", "02", "03", "04"}
        arch = ok(client.get(f"/api/experiments/{eid}/architecture"))
        assert arch["data_dims"]["n_channels"] == 9 and arch["data_dims"]["datasets"] == [ds["name"]]
        ok(client.put(f"/api/experiments/{eid}/architecture", json={"config": arch["config"]}))
        pre = ok(client.get(f"/api/experiments/{eid}/pretraining-checks"))
        assert pre["ok"], pre
        done = train_wait(client, eid)
        assert done["state"] == "artifacts_ready", done["active_run"]
        run = done["active_run"]
        # training never sees held-out participants (group-based splits)
        for f in run["folds"]:
            part = json.loads((settings.data_dir / "experiments" / eid / "runs" / run["id"] / f"fold{f['fold']}" /
                               "partition.json").read_text())
            by_id = {t["trial_id"]: t["group"] for t in trials}
            if split["strategy"] != "random":
                assert not {by_id[t] for t in part["fit"] + part["val"]} & {by_id[t] for t in part["test"]}
        # Signal Journey and Kernel Playground use the right trial, participant and checkpoint
        tl = ok(client.get(f"/api/experiments/{eid}/trials"))["trials"]
        test_trial = next(t for t in tl if t["partition"] == "test")
        j = ok(client.get(f"/api/experiments/{eid}/journey", params={"trial_id": test_trial["trial_id"]}))
        assert j["trial"]["group"] == test_trial["group"] and j["trial"]["run"] == test_trial["run"]
        assert j["trial"]["recording_id"].startswith(f"sub-{test_trial['group']}_")
        assert j["stages"][0]["id"] == "uploaded"
        conv = ok(client.post(f"/api/experiments/{eid}/convolution",
                              json={"layer": "temporal_conv", "trial_id": test_trial["trial_id"], "channel": 0,
                                    "row": 1, "position": 5}))
        assert conv["verified"] and conv["checkpoint_id"] == run["folds"][0]["checkpoint_id"]
        # Results: predictions only for held-out samples, with participant / run provenance
        ev = ok(client.post(f"/api/experiments/{eid}/evaluate"))["evaluation"]
        preds = ok(client.get(f"/api/experiments/{eid}/predictions"))
        split_info = ok(client.get(f"/api/experiments/{eid}/split"))
        test_ids = {t for f in split_info["membership"] for t in f["test"]}
        assert {p["parent_trial"] for p in preds} <= test_ids
        assert all(p["recording_id"] and p["run"] for p in preds)
        if split["strategy"] == "group":
            assert {p["group"] for p in preds} == {"04"} and ev["test"]["participants"] == ["04"]
        assert {r["group"] for r in ev["test"]["by_participant"]} == {p["group"] for p in preds}
        # Breaker remains compatible with the dataset-derived model
        br = ok(client.post(f"/api/experiments/{eid}/breaker", json={"changes": {"dropout": 0.2},
                                                                     "trial_id": test_trial["trial_id"]}))
        assert br["checkpoint_valid"]
        # Export: dataset manifest, file references and trial provenance
        rec = ok(client.post(f"/api/experiments/{eid}/export",
                             json={"items": ["dataset_manifest", "trial_provenance", "predictions"]}))
        z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/exports/{rec['id']}/download").content))
        man = json.loads(z.read(f"dataset/{ds['id']}_manifest.json"))
        assert man["fingerprint"] == ds["fingerprint"] and man["source"]["mode"] == "local_path"
        assert len([r for r in man["manifest"]["recordings"] if r["included"]]) == exp["recordings"]
        rows = list(csv.DictReader(io.StringIO(z.read("splits/trial_provenance.csv").decode())))
        assert len(rows) == exp["events"] and all(r["source_file"].endswith("_eeg.edf") for r in rows)
        assert {r["participant"] for r in rows} == {"01", "02", "03", "04"}
        assert any(r["fold1"] == "test" for r in rows)
        if previous:  # starting a new experiment never overwrote the previous one
            assert ok(client.get(f"/api/experiments/{previous}"))["state"] == "evaluation_completed"
        previous = eid


def test_block_design_windows_keep_blocks_together(client, nested):
    ds = import_local(client, nested["blocks"])
    assert ds["summary"]["class_distribution"] == nested["blocks_expected"]["blocks"]
    assert ds["summary"]["n_segments"] == sum(nested["blocks_expected"]["blocks"].values())
    assert "ECG1" in ds["summary"]["non_eeg_channels"]
    e = ok(client.post("/api/experiments", json={"name": "blocks", "dataset_id": ds["id"]}))
    eid = e["id"]
    ok(client.put(f"/api/experiments/{eid}/split", json={"strategy": "group", "test_groups": ["03"]}))
    d = ok(client.get(f"/api/experiments/{eid}/preprocessing/defaults"))
    cfg = d["config"]
    assert cfg["epoching"]["mode"] == "windows" and "ECG1" not in cfg["channels"]["selected"]
    cfg["epoching"]["classes"] = ["FC", "V"]
    cfg["epoching"].update({"window_s": 2.0, "step_s": 2.0, "max_windows": 6})
    ok(client.put(f"/api/experiments/{eid}/preprocessing", json=cfg))
    res = ok(client.post(f"/api/experiments/{eid}/preprocessing/run"))["result"]
    w = res["report"]["epoching"]["windows"]
    assert w["n_windows"] > w["n_parent_trials"] == 48  # 2 windows per 5 s FC block, 6 per V block
    exp_ = ok(client.get(f"/api/experiments/{eid}"))
    es = EpochSet.load(settings.data_dir / "cache" / "preprocessed" / f"{exp_['preprocessing_det_fingerprint']}.npz")
    assert all("#w" in str(t) for t in es.trial_ids) and es.X.shape[1] == 11  # EEG channels only
    arch = ok(client.get(f"/api/experiments/{eid}/architecture"))
    ok(client.put(f"/api/experiments/{eid}/architecture", json={"config": arch["config"]}))
    done = train_wait(client, eid, validation_mode="group")
    run = done["active_run"]
    part = json.loads((settings.data_dir / "experiments" / eid / "runs" / run["id"] / "fold0" /
                       "partition.json").read_text())
    parents_fit, parents_test = set(part["fit"] + part["val"]), set(part["test"])
    assert not parents_fit & parents_test
    window_parents = {str(p) for p in es.parents}
    assert parents_test <= window_parents
    tl = ok(client.get(f"/api/experiments/{eid}/trials"))
    assert tl["windows"]
    for t in tl["trials"]:  # every window inherits its block's partition
        assert t["partition"] == ("test" if t["parent"] in parents_test else
                                  "train" if t["parent"] in part["fit"] else "validation")
    win = next(t for t in tl["trials"] if t["partition"] == "test" and t["trial_id"].endswith("#w001"))
    j = ok(client.get(f"/api/experiments/{eid}/journey", params={"trial_id": win["trial_id"]}))
    up = j["stages"][0]
    assert up["id"] == "uploaded" and abs(up["times"][0] - win["offset_s"]) < 0.02
    ev = ok(client.post(f"/api/experiments/{eid}/evaluate"))["evaluation"]
    assert ev["test"]["participants"] == ["03"]
    assert ev["test"]["by_trial"]["n_trials"] == len(parents_test) < ev["test"]["by_trial"]["n_windows"]


def test_revised_dataset_marks_dependent_results_stale(client, nested):
    ds = import_local(client, nested["bids"])
    e = ok(client.post("/api/experiments", json={"name": "revise", "dataset_id": ds["id"]}))
    eid = e["id"]
    ok(client.put(f"/api/experiments/{eid}/split", json={"strategy": "random"}))
    ok(client.put(f"/api/experiments/{eid}/preprocessing",
                  json={"epoching": {"mode": "events", "tmin": 0, "tmax": 1.0}}))
    ok(client.post(f"/api/experiments/{eid}/preprocessing/run"))
    arch = ok(client.get(f"/api/experiments/{eid}/architecture"))
    ok(client.put(f"/api/experiments/{eid}/architecture", json={"config": arch["config"]}))
    done = train_wait(client, eid)
    assert done["state"] == "artifacts_ready"
    rev = ok(client.post(f"/api/datasets/{ds['id']}/revise"))
    assert rev["status"] == "review" and rev["revises"] == ds["id"]
    ok(client.put(f"/api/imports/{rev['id']}/manifest",
                  json={"recordings": [{"index": rev["manifest"]["recordings"][0]["index"], "included": False}]}))
    new = ok(client.post(f"/api/imports/{rev['id']}/import", params={"wait": "true"}))
    assert new["dataset_id"] != ds["id"]
    assert ok(client.get(f"/api/datasets/{ds['id']}"))["fingerprint"] == ds["fingerprint"]  # original untouched
    assert ok(client.get(f"/api/experiments/{eid}"))["state"] == "artifacts_ready"     # no silent change
    ok(client.put(f"/api/experiments/{eid}/dataset", json={"dataset_id": new["dataset_id"]}))
    after = ok(client.get(f"/api/experiments/{eid}"))
    assert after["state"] == "dataset_ready" and after["capabilities"]["run_stale"]


def test_mixed_rates_and_channels_are_harmonised(client, tmp_path):
    import mne

    root = tmp_path / "mixed"
    for i, (sf, chans) in enumerate([(128.0, ["C3", "Cz", "C4", "Pz"]), (256.0, ["Cz", "C3", "C4", "Oz"])]):
        n = int(sf * 40)
        raw = mne.io.RawArray(np.random.default_rng(i).standard_normal((len(chans), n)) * 1e-5,
                              mne.create_info(chans, sf, "eeg"), verbose=False)
        raw.set_annotations(mne.Annotations([2, 6, 10, 14, 18, 22], [0] * 6, ["a", "b"] * 3))
        (root / f"sub-0{i + 1}").mkdir(parents=True)
        raw.export(root / f"sub-0{i + 1}" / f"sub-0{i + 1}_eeg.edf", fmt="edf", verbose=False)
    s = ok(client.post("/api/imports", params={"wait": "true"}, json={"mode": "local_path", "local_path": str(root)}))
    m = s["manifest"]
    codes = {i["code"] for i in m["issues"]}
    assert {"sfreq_resample", "channels_dropped", "channels_reordered"} <= codes and m["importable"]
    assert m["summary"]["common_channels"] == ["C3", "Cz", "C4"]
    strict = ok(client.put(f"/api/imports/{s['id']}/manifest", json={"channel_policy": "strict"}))["manifest"]
    assert not strict["importable"]
    ok(client.put(f"/api/imports/{s['id']}/manifest", json={"channel_policy": "intersection", "target_sfreq": 128}))
    done = ok(client.post(f"/api/imports/{s['id']}/import", params={"wait": "true"}))
    ds = ok(client.get(f"/api/datasets/{done['dataset_id']}"))
    assert ds["summary"]["sfreq"] == 128 and ds["summary"]["ch_names"] == ["C3", "Cz", "C4"]
    sig = ok(client.get(f"/api/datasets/{ds['id']}/signal",
                        params={"trial_id": ok(client.get(f"/api/datasets/{ds['id']}/trials"))[-1]["trial_id"]}))
    assert sig["sfreq"] == 128
