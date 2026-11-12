"""Canonical in-memory and on-disk representation of EEG datasets.

All signal arrays are stored in microvolts (uV) when the unit is known. Datasets are
either *continuous* (one or more recordings with event markers) or *epoched*
(trials x channels x samples with labels). Every candidate trial has a stable ID that
encodes its recording (participant / session / run), so split assignments survive
preprocessing and provenance is never lost.

On-disk format v2 stores one ``.npy`` file per recording (memory-mapped on load), so
large multi-participant datasets are read lazily, one recording at a time. The legacy
v1 format (a single ``arrays.npz``) is still readable.

Events are stored as ``[sample, code, duration_samples]``. A non-zero duration marks a
*segment* (e.g. a block delimited by start/end markers); zero marks a point event.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FORMAT_VERSION = 2


@dataclass
class Recording:
    name: str
    data: np.ndarray                 # [C, S] float32 (may be a read-only memmap)
    events: np.ndarray               # [n, 3] int64 -> (sample, code, duration_samples); legacy [n, 2]
    event_id: dict[str, int]         # description / class label -> code
    group: str | None = None         # participant ID
    session: str | None = None
    bad_segments: list[tuple[float, float]] = field(default_factory=list)  # (onset_s, duration_s)
    run: str | None = None
    task: str | None = None
    recording_id: str | None = None
    source_file: str | None = None
    event_source: str | None = None

    def __post_init__(self) -> None:
        ev = np.asarray(self.events, dtype=np.int64)
        if ev.size == 0:
            ev = np.zeros((0, 3), dtype=np.int64)
        elif ev.ndim != 2:
            ev = ev.reshape(-1, 2)
        if ev.shape[1] == 2:
            ev = np.c_[ev, np.zeros(len(ev), dtype=np.int64)]
        self.events = ev


@dataclass
class ContinuousData:
    sfreq: float
    ch_names: list[str]
    unit: str
    recordings: list[Recording]
    ch_types: list[str] | None = None
    highpass: float | None = None
    lowpass: float | None = None
    kind: str = "continuous"


@dataclass
class EpochedData:
    sfreq: float
    ch_names: list[str]
    unit: str
    X: np.ndarray                    # [n, C, T] float32 (may be a read-only memmap)
    labels: np.ndarray               # [n] str
    tmin: float = 0.0
    groups: np.ndarray | None = None     # [n] str ('' = unknown)
    sessions: np.ndarray | None = None
    trial_ids: np.ndarray | None = None
    ch_types: list[str] | None = None
    highpass: float | None = None
    lowpass: float | None = None
    runs: np.ndarray | None = None
    recording_ids: np.ndarray | None = None
    source_files: dict[str, str] | None = None   # recording_id -> source file
    kind: str = "epoched"

    def __post_init__(self) -> None:
        if self.trial_ids is None:
            self.trial_ids = np.array([f"t{i:05d}" for i in range(len(self.X))])


Dataset = ContinuousData | EpochedData


@dataclass
class CandidateTrial:
    trial_id: str
    label: str
    group: str | None
    session: str | None
    recording: int | None = None
    sample: int | None = None
    run: str | None = None
    recording_id: str | None = None
    duration_s: float = 0.0


def _is_bad(desc: str) -> bool:
    d = desc.lower()
    return d.startswith("bad") or d.startswith("edge")


def _opt(arr: np.ndarray | None, i: int) -> str | None:
    if arr is None:
        return None
    return str(arr[i]) or None


def candidate_trials(ds: Dataset) -> list[CandidateTrial]:
    """Units that splits operate on: epochs, or event markers / segments of continuous recordings."""
    out: list[CandidateTrial] = []
    if isinstance(ds, EpochedData):
        for i in range(len(ds.X)):
            out.append(CandidateTrial(
                trial_id=str(ds.trial_ids[i]), label=str(ds.labels[i]), group=_opt(ds.groups, i),
                session=_opt(ds.sessions, i), run=_opt(ds.runs, i), recording_id=_opt(ds.recording_ids, i),
            ))
        return out
    for r, rec in enumerate(ds.recordings):
        inv = {v: k for k, v in rec.event_id.items()}
        for e, (sample, code, dur) in enumerate(rec.events):
            desc = inv.get(int(code), str(int(code)))
            if _is_bad(desc):
                continue
            tid = f"{rec.recording_id}:e{e:04d}" if rec.recording_id else f"r{r}e{e:05d}"
            out.append(CandidateTrial(trial_id=tid, label=desc, group=rec.group, session=rec.session, recording=r,
                                      sample=int(sample), run=rec.run, recording_id=rec.recording_id,
                                      duration_s=float(dur) / ds.sfreq))
    return out


# ---------------------------------------------------------------------------
# Channel types and IDs
# ---------------------------------------------------------------------------

_TYPE_RULES = [
    (re.compile(r"^(ecg|ekg)", re.I), "ecg"),
    (re.compile(r"^emg", re.I), "emg"),
    (re.compile(r"^(eog|heog|veog|leog|reog)", re.I), "eog"),
    (re.compile(r"^(a1|a2|m1|m2|ref|gnd|ground)$", re.I), "misc"),
    (re.compile(r"^(sti|trig|status|event|marker)", re.I), "stim"),
    (re.compile(r"^(resp|gsr|eda|temp|spo2|pulse|acc|gyro)", re.I), "misc"),
]


def infer_channel_type(name: str, declared: str | None = None) -> str:
    """Channel type from the file, corrected by name (many EDF files label every channel 'eeg')."""
    if declared and declared not in ("eeg", "misc"):
        return declared
    clean = re.sub(r"^(eeg|ch)[\s_-]*", "", name.strip(), flags=re.I)
    for rx, typ in _TYPE_RULES:
        if rx.match(clean) or rx.match(name.strip()):
            return typ
    return declared or "eeg"


def safe_id(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(s))[:32] or "x"


def make_recording_id(participant: str | None, session: str | None, run: str | None, task: str | None,
                      index: int) -> str:
    parts = []
    if participant:
        parts.append(f"sub-{safe_id(participant)}")
    if session:
        parts.append(f"ses-{safe_id(session)}")
    if task:
        parts.append(f"task-{safe_id(task)}")
    if run:
        parts.append(f"run-{safe_id(run)}")
    return "_".join(parts) if parts else f"rec-{index + 1:03d}"


# ---------------------------------------------------------------------------
# Persistence (format v2: per-recording memory-mapped arrays)
# ---------------------------------------------------------------------------


def _hash_array(h, arr: np.ndarray, chunk_rows: int = 4) -> None:
    a = np.asarray(arr)
    if a.ndim == 0 or a.shape[0] == 0:
        h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
        return
    for i in range(0, a.shape[0], chunk_rows):
        h.update(np.ascontiguousarray(a[i:i + chunk_rows], dtype=np.float32).tobytes())


class DatasetWriter:
    """Write a canonical dataset incrementally (one recording at a time) and fingerprint it."""

    def __init__(self, directory: Path, kind: str, sfreq: float, ch_names: list[str], unit: str,
                 ch_types: list[str] | None = None, highpass: float | None = None, lowpass: float | None = None,
                 tmin: float = 0.0):
        if directory.exists():
            shutil.rmtree(directory)
        (directory / "recordings").mkdir(parents=True)
        self.dir = directory
        self.meta: dict = {"format_version": FORMAT_VERSION, "kind": kind, "sfreq": float(sfreq),
                           "ch_names": list(ch_names), "unit": unit, "ch_types": ch_types, "highpass": highpass,
                           "lowpass": lowpass, "tmin": tmin, "recordings": []}
        self.h = hashlib.sha256()
        self.h.update(json.dumps({"kind": kind, "sfreq": float(sfreq), "ch": list(ch_names), "unit": unit}).encode())
        self._events: dict[str, np.ndarray] = {}
        self._blocks: list[Path] = []
        self._trials: dict[str, list] = {"labels": [], "trial_ids": [], "groups": [], "sessions": [], "runs": [],
                                         "recording_ids": []}

    def add_recording(self, rec: Recording) -> None:
        i = len(self.meta["recordings"])
        data = np.ascontiguousarray(rec.data, dtype=np.float32)
        np.save(self.dir / "recordings" / f"r{i:04d}.npy", data)
        _hash_array(self.h, data)
        self.h.update(np.ascontiguousarray(rec.events, dtype=np.int64).tobytes())
        self.h.update(json.dumps([rec.event_id, rec.group, rec.session, rec.run, rec.task], sort_keys=True).encode())
        self._events[f"e{i}"] = np.asarray(rec.events, dtype=np.int64).reshape(-1, 3)
        self.meta["recordings"].append({
            "name": rec.name, "event_id": rec.event_id, "group": rec.group, "session": rec.session, "run": rec.run,
            "task": rec.task, "recording_id": rec.recording_id, "source_file": rec.source_file,
            "event_source": rec.event_source, "bad_segments": [list(b) for b in rec.bad_segments],
            "n_samples": int(data.shape[1]),
        })

    def add_epochs(self, X: np.ndarray, labels, trial_ids, group: str | None, session: str | None, run: str | None,
                   recording_id: str | None, source_file: str | None = None) -> None:
        i = len(self._blocks)
        path = self.dir / "recordings" / f"block{i:04d}.npy"
        X = np.ascontiguousarray(X, dtype=np.float32)
        np.save(path, X)
        self._blocks.append(path)
        _hash_array(self.h, X)
        n = len(X)
        self._trials["labels"] += [str(v) for v in labels]
        self._trials["trial_ids"] += [str(v) for v in trial_ids]
        self._trials["groups"] += [group or ""] * n
        self._trials["sessions"] += [session or ""] * n
        self._trials["runs"] += [run or ""] * n
        self._trials["recording_ids"] += [recording_id or ""] * n
        self.h.update("|".join(map(str, labels)).encode())
        self.h.update(json.dumps([group, session, run, recording_id]).encode())
        if recording_id:
            self.meta.setdefault("source_files", {})[recording_id] = source_file

    def finish(self, extra_meta: dict | None = None) -> str:
        if self.meta["kind"] == "epoched":
            shapes = [np.load(p, mmap_mode="r").shape for p in self._blocks]
            n = sum(s[0] for s in shapes)
            if n == 0:
                raise ValueError("No epochs to write.")
            out = np.lib.format.open_memmap(self.dir / "X.npy", mode="w+", dtype=np.float32,
                                            shape=(n, shapes[0][1], shapes[0][2]))
            pos = 0
            for p, s in zip(self._blocks, shapes):
                out[pos:pos + s[0]] = np.load(p, mmap_mode="r")
                pos += s[0]
            out.flush()
            del out
            for p in self._blocks:
                p.unlink()
            np.savez(self.dir / "trials.npz", **{k: np.array(v, dtype=str) for k, v in self._trials.items()})
        else:
            np.savez(self.dir / "events.npz", **self._events)
        fp = self.h.hexdigest()
        self.meta["fingerprint"] = fp
        if extra_meta:
            self.meta.update(extra_meta)
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=1, default=str))
        return fp


def save_dataset(ds: Dataset, directory: Path, extra_meta: dict | None = None) -> str:
    """Write an in-memory dataset in the canonical format and return its fingerprint (sha256)."""
    w = DatasetWriter(directory, ds.kind, ds.sfreq, ds.ch_names, ds.unit, ds.ch_types, ds.highpass, ds.lowpass,
                      tmin=getattr(ds, "tmin", 0.0))
    if isinstance(ds, EpochedData):
        rec_ids = ds.recording_ids if ds.recording_ids is not None else np.array([""] * len(ds.X))
        # write contiguous blocks that share recording provenance
        order = list(range(len(ds.X)))
        start = 0
        while start < len(order):
            end = start + 1
            key = lambda i: (str(rec_ids[i]), _opt(ds.groups, i), _opt(ds.sessions, i), _opt(ds.runs, i))  # noqa: E731
            while end < len(order) and key(end) == key(start):
                end += 1
            sl = slice(start, end)
            w.add_epochs(ds.X[sl], ds.labels[sl], ds.trial_ids[sl], _opt(ds.groups, start), _opt(ds.sessions, start),
                         _opt(ds.runs, start), str(rec_ids[start]) or None,
                         (ds.source_files or {}).get(str(rec_ids[start])))
            start = end
    else:
        for rec in ds.recordings:
            w.add_recording(rec)
    return w.finish(extra_meta)


def _arr_or_none(z, key: str) -> np.ndarray | None:
    if key not in z.files:
        return None
    a = z[key]
    return a if any(str(v) for v in a) else None


def load_dataset(directory: Path) -> Dataset:
    meta = json.loads((directory / "meta.json").read_text())
    common = dict(sfreq=float(meta["sfreq"]), ch_names=list(meta["ch_names"]), unit=meta["unit"],
                  ch_types=meta.get("ch_types"), highpass=meta.get("highpass"), lowpass=meta.get("lowpass"))
    if meta.get("format_version", 1) >= 2:
        if meta["kind"] == "epoched":
            with np.load(directory / "trials.npz", allow_pickle=False) as z:
                return EpochedData(X=np.load(directory / "X.npy", mmap_mode="r"), labels=z["labels"],
                                   trial_ids=z["trial_ids"], groups=_arr_or_none(z, "groups"),
                                   sessions=_arr_or_none(z, "sessions"), runs=_arr_or_none(z, "runs"),
                                   recording_ids=_arr_or_none(z, "recording_ids"),
                                   source_files=meta.get("source_files"), tmin=float(meta.get("tmin", 0.0)), **common)
        recs = []
        with np.load(directory / "events.npz", allow_pickle=False) as z:
            for i, rm in enumerate(meta["recordings"]):
                recs.append(Recording(
                    name=rm["name"], data=np.load(directory / "recordings" / f"r{i:04d}.npy", mmap_mode="r"),
                    events=z[f"e{i}"], event_id={k: int(v) for k, v in rm["event_id"].items()}, group=rm.get("group"),
                    session=rm.get("session"), run=rm.get("run"), task=rm.get("task"),
                    recording_id=rm.get("recording_id"), source_file=rm.get("source_file"),
                    event_source=rm.get("event_source"),
                    bad_segments=[tuple(b) for b in rm.get("bad_segments", [])]))
        return ContinuousData(recordings=recs, **common)
    # legacy v1: single arrays.npz
    with np.load(directory / "arrays.npz", allow_pickle=False) as z:
        arrs = {k: z[k] for k in z.files}
    if meta["kind"] == "epoched":
        return EpochedData(X=arrs["X"], labels=arrs["labels"], trial_ids=arrs["trial_ids"],
                           groups=arrs.get("groups"), sessions=arrs.get("sessions"),
                           tmin=float(meta.get("tmin", 0.0)), **common)
    recs = []
    for i, rm in enumerate(meta["recordings"]):
        recs.append(Recording(name=rm["name"], data=arrs[f"rec{i}_data"], events=arrs[f"rec{i}_events"],
                              event_id={k: int(v) for k, v in rm["event_id"].items()}, group=rm.get("group"),
                              session=rm.get("session"),
                              bad_segments=[tuple(b) for b in rm.get("bad_segments", [])]))
    return ContinuousData(recordings=recs, **common)


def fingerprint_dataset(ds: Dataset) -> str:
    """Fingerprint of an in-memory dataset (same algorithm as ``DatasetWriter``)."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        return save_dataset(ds, Path(d) / "ds")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarize(ds: Dataset) -> dict:
    trials = candidate_trials(ds)
    labels = [t.label for t in trials]
    classes = sorted(set(labels))
    dist = {c: labels.count(c) for c in classes}
    groups = sorted({t.group for t in trials if t.group})
    sessions = sorted({t.session for t in trials if t.session})
    runs = sorted({t.run for t in trials if t.run})
    missing: list[str] = []
    if not groups:
        missing.append("No participant/group IDs: group-level splits and overlap reports are unavailable.")
    elif any(t.group is None for t in trials):
        missing.append(f"{sum(t.group is None for t in trials)} trial(s) have no participant/group ID.")
    if not sessions:
        missing.append("No session IDs: session-overlap reporting is unavailable.")
    if ds.unit == "unknown":
        missing.append("Signal unit unknown: amplitude thresholds are interpreted in raw units.")
    ch_types = ds.ch_types or ["eeg"] * len(ds.ch_names)
    ch_types = [infer_channel_type(n, t) for n, t in zip(ds.ch_names, ch_types)]
    non_eeg = [n for n, t in zip(ds.ch_names, ch_types) if t != "eeg"]
    segs = [t for t in trials if t.duration_s > 0]
    summary: dict = {
        "structure": ds.kind,
        "n_channels": len(ds.ch_names),
        "ch_names": ds.ch_names,
        "ch_types": ch_types,
        "non_eeg_channels": non_eeg,
        "sfreq": ds.sfreq,
        "unit": ds.unit,
        "n_trials": len(trials),
        "n_segments": len(segs),
        "segment_seconds": {c: round(sum(t.duration_s for t in segs if t.label == c), 2) for c in classes} if segs
        else None,
        "n_classes": len(classes),
        "classes": classes,
        "class_distribution": dist,
        "n_participants": len(groups),
        "participants": groups,
        "n_sessions": len(sessions),
        "sessions": sessions,
        "n_runs": len(runs),
        "missing_metadata": missing,
        "highpass": ds.highpass,
        "lowpass": ds.lowpass,
    }
    by_p: dict[str, dict] = {}
    for t in trials:
        p = by_p.setdefault(t.group or "(none)", {"participant": t.group, "n_trials": 0, "recordings": set(),
                                                  "sessions": set(), "runs": set(), "class_counts": {}})
        p["n_trials"] += 1
        if t.recording_id or t.recording is not None:
            p["recordings"].add(t.recording_id or str(t.recording))
        if t.session:
            p["sessions"].add(t.session)
        if t.run:
            p["runs"].add(t.run)
        p["class_counts"][t.label] = p["class_counts"].get(t.label, 0) + 1
    summary["participants_table"] = [
        {**{k: v for k, v in p.items() if k not in ("recordings", "sessions", "runs")},
         "n_recordings": len(p["recordings"]) or 1, "sessions": sorted(p["sessions"]), "runs": sorted(p["runs"])}
        for p in by_p.values()]
    if isinstance(ds, EpochedData):
        n, C, T = ds.X.shape
        rec_tab: dict[str, dict] = {}
        for t in trials:
            if not t.recording_id:
                continue
            r = rec_tab.setdefault(t.recording_id, {"name": t.recording_id, "recording_id": t.recording_id,
                                                    "group": t.group, "session": t.session, "run": t.run,
                                                    "n_events": 0, "n_trials": 0, "duration_s": None,
                                                    "event_types": set(), "bad_segments": 0,
                                                    "source_file": (ds.source_files or {}).get(t.recording_id),
                                                    "event_source": "epochs"})
            r["n_trials"] += 1
            r["n_events"] += 1
            r["event_types"].add(t.label)
        summary.update({
            "n_recordings": len(rec_tab) or None, "n_epochs": n, "n_samples": int(n * T), "epoch_samples": T,
            "epoch_duration_s": T / ds.sfreq, "tmin": ds.tmin,
            "recordings": [{**r, "event_types": sorted(r["event_types"])} for r in rec_tab.values()],
        })
    else:
        total = sum(r.data.shape[1] for r in ds.recordings)
        per_rec = {}
        for t in trials:
            per_rec[t.recording] = per_rec.get(t.recording, 0) + 1
        summary.update({
            "n_recordings": len(ds.recordings), "n_epochs": None, "n_samples": int(total),
            "epoch_samples": None, "epoch_duration_s": None, "tmin": None,
            "recordings": [{"name": r.name, "recording_id": r.recording_id, "duration_s": r.data.shape[1] / ds.sfreq,
                            "n_events": int(len(r.events)), "n_trials": per_rec.get(i, 0), "group": r.group,
                            "session": r.session, "run": r.run, "task": r.task, "source_file": r.source_file,
                            "event_source": r.event_source, "event_types": sorted(r.event_id),
                            "bad_segments": len(r.bad_segments)} for i, r in enumerate(ds.recordings)],
        })
    if ds.kind == "continuous":
        hint = "Continuous recording(s); trials are defined by event markers or start/end segments during epoching."
    else:
        hint = "Already epoched; epochs can be kept or cropped (rebuilding needs continuous data)."
    if ds.highpass is not None and ds.lowpass is not None:
        if ds.highpass <= 0 and ds.lowpass >= ds.sfreq / 2 - 1e-6:
            hint += " Header filter settings indicate unfiltered (raw) data."
        else:
            hint += f" Header filter settings: {ds.highpass:g}-{ds.lowpass:g} Hz (previously filtered)."
    else:
        hint += " Processing history unknown."
    summary["processing_hint"] = hint
    return summary
