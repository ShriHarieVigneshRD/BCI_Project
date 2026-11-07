"""Dataset upload, validation, storage and signal access."""

from __future__ import annotations

import re
import shutil
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ..config import settings
from ..data import synthetic
from ..data.loaders import ALLOWED_EXTENSIONS, LoadResult, UploadOptions, load_files
from ..data.manifest import manifest_from_dataset
from ..data.store import (ContinuousData, Dataset, EpochedData, candidate_trials, load_dataset, make_recording_id,
                          save_dataset, summarize)
from ..schemas import DatasetRecord, now_iso
from . import repo

_cache: "OrderedDict[str, Dataset]" = OrderedDict()
_cache_lock = threading.Lock()
SYNTHETIC_NOTE = ("SYNTHETIC data generated for software testing. Metrics obtained on it are not evidence of "
                  "real-world EEG decoding performance.")


class UploadRejected(ValueError):
    def __init__(self, issues: list[dict], fmt: str):
        self.issues, self.format = issues, fmt
        super().__init__("; ".join(i["message"] for i in issues if i["level"] == "error"))


def safe_filename(name: str) -> str:
    base = Path(name.replace("\\", "/")).name
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)[:120].lstrip(".")
    if not base:
        raise ValueError("Invalid file name.")
    lower = base.lower()
    ext = ".fif.gz" if lower.endswith(".fif.gz") else Path(lower).suffix
    if ext not in ALLOWED_EXTENSIONS and ext != ".fif.gz":
        raise ValueError(f"File type '{ext or base}' is not supported.")
    if ext == ".gz" and not lower.endswith(".fif.gz"):
        raise ValueError("Only .fif.gz compressed files are supported.")
    return base


def load(ds_id: str) -> Dataset:
    with _cache_lock:
        if ds_id in _cache:
            _cache.move_to_end(ds_id)
            return _cache[ds_id]
    ds = load_dataset(settings.data_dir / "datasets" / ds_id)
    with _cache_lock:
        _cache[ds_id] = ds
        while len(_cache) > 4:
            _cache.popitem(last=False)
    return ds


def assign_recording_ids(ds: Dataset) -> None:
    """Give every recording / epoch block a stable provenance ID (participant / session / run / task)."""
    if isinstance(ds, ContinuousData):
        used: dict[str, int] = {}
        for i, r in enumerate(ds.recordings):
            if not r.recording_id:
                rid = make_recording_id(r.group, r.session, r.run, r.task, i)
                used[rid] = used.get(rid, 0) + 1
                r.recording_id = rid if used[rid] == 1 else f"{rid}_{used[rid]}"
    elif ds.recording_ids is None and (ds.groups is not None or ds.sessions is not None):
        ids = []
        for i in range(len(ds.X)):
            g = str(ds.groups[i]) if ds.groups is not None and str(ds.groups[i]) else None
            se = str(ds.sessions[i]) if ds.sessions is not None and str(ds.sessions[i]) else None
            ids.append(make_recording_id(g, se, None, None, 0))
        ds.recording_ids = np.array(ids)


def _register(ds: Dataset, name: str, fmt: str, files: list[str], issues: list[dict], synthetic_flag: bool,
              ds_id: str | None = None, source: dict | None = None,
              sizes: dict[str, int] | None = None) -> DatasetRecord:
    ds_id = ds_id or repo.new_id("ds")
    assign_recording_ids(ds)
    source = source or {"mode": "synthetic" if synthetic_flag else "api_upload"}
    manifest = manifest_from_dataset(ds, name, source, [(f, (sizes or {}).get(f, 0)) for f in files], fmt=fmt)
    fp = save_dataset(ds, settings.data_dir / "datasets" / ds_id, {"manifest": manifest.model_dump(mode="json")})
    summary = summarize(ds)
    trainable = len({t.label for t in candidate_trials(ds)}) >= 2
    notes = [SYNTHETIC_NOTE] if synthetic_flag else []
    rec = DatasetRecord(id=ds_id, name=name, created_at=now_iso(), format=fmt, structure=ds.kind, fingerprint=fp,
                        synthetic=synthetic_flag, files=files, summary=summary, issues=issues, trainable=trainable,
                        notes=notes, manifest=manifest.model_dump(mode="json"), source=source)
    repo.save_dataset(rec)
    with _cache_lock:
        _cache.pop(ds_id, None)
    return rec


def create_from_upload(files: list[tuple[str, bytes]], opts: UploadOptions) -> DatasetRecord:
    ds_id = repo.new_id("ds")
    up_dir = settings.data_dir / "uploads" / ds_id
    up_dir.mkdir(parents=True, exist_ok=True)
    limit = settings.max_upload_mb * 1024 * 1024
    total = 0
    names = []
    try:
        for raw_name, content in files:
            name = safe_filename(raw_name)
            total += len(content)
            if total > limit:
                raise ValueError(f"Upload exceeds the {settings.max_upload_mb} MB limit.")
            (up_dir / name).write_bytes(content)
            names.append(name)
        opts.file_meta = {safe_filename(k): v for k, v in opts.file_meta.items()} if opts.file_meta else {}
        res: LoadResult = load_files([up_dir / n for n in names], opts)
    except ValueError as exc:
        shutil.rmtree(up_dir, ignore_errors=True)
        raise UploadRejected([{"level": "error", "code": "invalid_upload", "message": str(exc)}], "unknown")
    if not res.ok:
        shutil.rmtree(up_dir, ignore_errors=True)
        raise UploadRejected(res.issues or [{"level": "error", "code": "unreadable",
                                             "message": "No data could be read."}], res.format)
    name = opts.name or (names[0] if len(names) == 1 else f"{names[0]} (+{len(names) - 1} files)")
    sizes = {n: (up_dir / n).stat().st_size for n in names}
    return _register(res.dataset, name, res.format, names, res.issues, False, ds_id=ds_id,
                     source={"mode": "api_upload", "source_dir": f"uploads/{ds_id}"}, sizes=sizes)


def create_synthetic(kind: str = "epoched", n_participants: int = 6, n_classes: int = 3, trials_per_class: int = 40,
                     n_channels: int = 16, include_frontal: bool = True, sfreq: float = 128.0, seed: int = 7,
                     n_sessions: int = 1, name: str | None = None) -> DatasetRecord:
    ds = synthetic.generate(kind=kind, n_participants=n_participants, n_classes=n_classes,
                            trials_per_class=trials_per_class, n_channels=n_channels, include_frontal=include_frontal,
                            sfreq=sfreq, seed=seed, n_sessions=n_sessions)
    nm = name or (f"SYNTHETIC motor imagery - {n_participants} participants, {n_classes} classes, "
                  f"{'continuous' if kind == 'continuous' else 'epoched'}")
    if not nm.upper().startswith("SYNTHETIC"):
        nm = "SYNTHETIC - " + nm
    issues = [{"level": "info", "code": "synthetic", "message": SYNTHETIC_NOTE}]
    return _register(ds, nm, "synthetic", [], issues, True, source={"mode": "synthetic", "seed": seed})


def trial_signal(ds_id: str, trial_id: str, max_channels: int | None = None) -> dict:
    ds = load(ds_id)
    trials = {t.trial_id: t for t in candidate_trials(ds)}
    if trial_id not in trials:
        raise KeyError(f"Unknown trial {trial_id}")
    t = trials[trial_id]
    if isinstance(ds, EpochedData):
        i = int(np.flatnonzero(ds.trial_ids == trial_id)[0])
        x = ds.X[i]
        t0 = ds.tmin
    else:
        rec = ds.recordings[t.recording]
        a = max(0, t.sample - int(0.5 * ds.sfreq))
        b = min(rec.data.shape[1], t.sample + int(2.5 * ds.sfreq))
        x = rec.data[:, a:b]
        t0 = (a - t.sample) / ds.sfreq
    chans = ds.ch_names if max_channels is None else ds.ch_names[:max_channels]
    times = t0 + np.arange(x.shape[1]) / ds.sfreq
    return {"trial_id": trial_id, "label": t.label, "group": t.group, "session": t.session, "sfreq": ds.sfreq,
            "unit": ds.unit, "ch_names": chans, "times": times.round(5).tolist(),
            "data": x[:len(chans)].round(3).tolist(), "event_time": 0.0,
            "label_note": "Raw uploaded signal (before preprocessing)."}


def trials_table(ds_id: str, limit: int = 5000) -> list[dict]:
    ds = load(ds_id)
    return [{"trial_id": t.trial_id, "label": t.label, "group": t.group, "session": t.session}
            for t in candidate_trials(ds)[:limit]]


def compatibility(train: DatasetRecord, test: DatasetRecord) -> dict:
    """Checks for the separate train/test dataset workflow."""
    a, b = train.summary, test.summary
    checks = []

    def add(cid, label, status, detail):
        checks.append({"id": cid, "label": label, "status": status, "detail": detail})

    add("structure", "Same data structure", "pass" if a["structure"] == b["structure"] else "fail",
        f"{a['structure']} vs {b['structure']}")
    if a["ch_names"] == b["ch_names"]:
        add("channels", "Channel identities and order", "pass", f"{a['n_channels']} identical channels")
    elif sorted(a["ch_names"]) == sorted(b["ch_names"]):
        add("channels", "Channel identities and order", "warn", "Same channels in a different order; channels are "
                                                                "selected by name during preprocessing.")
    else:
        common = sorted(set(a["ch_names"]) & set(b["ch_names"]))
        add("channels", "Channel identities and order", "warn" if common else "fail",
            f"Channel sets differ; {len(common)} shared channel(s) can be selected in preprocessing.")
    add("sfreq", "Sampling rate", "pass" if abs(a["sfreq"] - b["sfreq"]) < 1e-6 else "warn",
        f"{a['sfreq']:g} vs {b['sfreq']:g} Hz" + ("" if abs(a["sfreq"] - b["sfreq"]) < 1e-6 else
                                                 " - enable resampling in preprocessing"))
    add("unit", "Signal units", "pass" if a["unit"] == b["unit"] else "warn", f"{a['unit']} vs {b['unit']}")
    if a.get("epoch_samples") and b.get("epoch_samples"):
        same = a["epoch_samples"] == b["epoch_samples"] and a["tmin"] == b["tmin"]
        add("epoch_length", "Epoch length", "pass" if same else "warn",
            f"{a['epoch_samples']} vs {b['epoch_samples']} samples" + ("" if same else " - crop to a common window"))
    ca, cb = set(a["classes"]), set(b["classes"])
    add("classes", "Class / event mapping", "pass" if ca == cb else ("warn" if ca & cb else "fail"),
        f"train: {sorted(ca)}; test: {sorted(cb)}")
    ga, gb = set(a.get("participants") or []), set(b.get("participants") or [])
    if ga and gb:
        ov = sorted(ga & gb)
        add("participant_overlap", "Participant overlap", "warn" if ov else "pass",
            f"{len(ov)} participant(s) appear in both datasets ({', '.join(ov[:8])}). The test set then measures "
            "within-participant (e.g. cross-session) generalisation, not cross-participant." if ov
            else "No shared participant IDs.")
    else:
        add("participant_overlap", "Participant overlap", "warn",
            "Participant IDs unavailable in at least one dataset; overlap cannot be checked.")
    sa, sb = set(a.get("sessions") or []), set(b.get("sessions") or [])
    if sa and sb and ga & gb:
        add("session_overlap", "Session overlap", "warn" if sa & sb else "pass",
            "Session labels overlap between datasets." if sa & sb else "Different session labels.")
    return {"ok": not any(c["status"] == "fail" for c in checks), "checks": checks}
