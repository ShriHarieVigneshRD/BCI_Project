"""Scan a dataset directory (uploaded folder, selected files, or a local folder path).

Scanning reads *headers only* where the format allows it (MNE ``preload=False``,
``.npy`` memory maps), one file at a time, so large datasets are never loaded into
memory. It classifies every file (recording / event file / companion / metadata /
ignored), infers participant / session / run / task identifiers, associates event
files with their recordings, and produces a draft :class:`DatasetManifest` plus the
raw events needed to compute trial counts.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .entities import infer_entities, match_key
from .loaders import (UploadOptions, _ext, _mne, annotation_events, data_picks, load_files, read_raw_any,
                      stim_events)
from .manifest import (ArrayOptions, DatasetManifest, EventsCache, EventSourceInfo, FileEntry, Issue,
                       ScannedRecording, suggest_mapping, validate_manifest)
from .store import ContinuousData, EpochedData, infer_channel_type

MAX_FILES = 50_000
MAX_DEPTH = 12
JUNK_DIRS = {"__macosx", ".git", ".svn", "__pycache__", "node_modules", ".ipynb_checkpoints", "derivatives",
             "sourcedata", "code"}
RECORDING_EXTS = {".edf", ".bdf", ".gdf", ".fif", ".fif.gz", ".set", ".vhdr", ".npz", ".npy", ".csv", ".tsv"}
COMPANION_EXTS = {".fdt", ".vmrk", ".eeg"}
EVENT_NAME_RX = re.compile(r"(^|[_\-.])(events?|eve|markers?|triggers?|annotations?|stim)($|[_\-.])", re.I)
METADATA_TSV_RX = re.compile(r"(^participants|_channels|_electrodes|_scans|^sessions|_sessions|_coordsystem)", re.I)
NPY_ROLE_RX = {
    "labels": re.compile(r"(^|[_\-])(labels?|y|targets?|classes)$", re.I),
    "groups": re.compile(r"(^|[_\-])(groups?|subjects?|participants?)$", re.I),
    "sessions": re.compile(r"(^|[_\-])(sessions?)$", re.I),
    "events": re.compile(r"(^|[_\-])(events?)$", re.I),
}
UNSUPPORTED_NOTES = {
    ".mat": "MATLAB .mat files are not supported (only EEGLAB .set).",
    ".cnt": "Neuroscan .cnt is not supported; convert to EDF or FIF.",
    ".xdf": "XDF is not supported; convert to EDF or FIF.",
}

TIME_COLS = [("sample", "samples"), ("samples", "samples"), ("latency", "samples"), ("onset", "seconds"),
             ("onset_s", "seconds"), ("time", "seconds"), ("time_s", "seconds"), ("timestamp", "seconds"),
             ("onset_ms", "ms"), ("time_ms", "ms"), ("latency_ms", "ms")]
DESC_COLS = ["trial_type", "label", "class", "condition", "event", "event_type", "type", "description", "marker",
             "annotation", "value", "code", "trigger", "stim"]


def _rel(p: Path, root: Path) -> str:
    return p.relative_to(root).as_posix()


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_name(relpath: str) -> tuple[str, str | None]:
    """Role from the path alone (used to plan uploads before any content is read)."""
    parts = relpath.replace("\\", "/").split("/")
    name = parts[-1]
    low = name.lower()
    if any(p.startswith(".") for p in parts) or any(p.lower() in JUNK_DIRS for p in parts[:-1]):
        return "ignored", "hidden or auxiliary folder"
    ext = _ext(Path(name))
    stem = low[: -len(ext)] if ext and low.endswith(ext) else low
    if ext in COMPANION_EXTS:
        return "companion", None
    if ext == ".json":
        return "metadata", None
    if ext == ".eve" or low.endswith(("-eve.fif", "_eve.fif", "-eve.txt")):
        return "events", None
    if ext in (".tsv", ".csv"):
        if METADATA_TSV_RX.search(stem):
            return "metadata", None
        if EVENT_NAME_RX.search(stem):
            return "events", None
        return "eeg", "table: content is checked during the scan"
    if ext == ".npy":
        for role_rx in NPY_ROLE_RX.values():
            if role_rx.search(stem):
                return "companion", "NumPy companion array"
        return "eeg", None
    if ext in RECORDING_EXTS:
        return "eeg", None
    if (low.startswith(("readme", "changes", "license")) and ext in ("", ".md", ".txt", ".rst")) or ext == ".md":
        return "metadata", None
    return "ignored", UNSUPPORTED_NOTES.get(ext, "not an EEG, event or metadata file")


def plan_files(files: list[dict]) -> list[FileEntry]:
    out = []
    for f in files[:MAX_FILES]:
        role, note = classify_name(f["path"])
        out.append(FileEntry(path=f["path"], size=int(f.get("size") or 0), role=role, note=note))
    return out


def list_directory(root: Path) -> list[FileEntry]:
    """Walk a directory without following symlinks; never leaves ``root``."""
    root_res = root.resolve()
    out: list[FileEntry] = []
    for dirpath, dirnames, filenames in os.walk(root_res, followlinks=False):
        depth = len(Path(dirpath).relative_to(root_res).parts)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d.lower() not in JUNK_DIRS
                             and depth < MAX_DEPTH)
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            if p.is_symlink():
                continue
            rel = _rel(p, root_res)
            role, note = classify_name(rel)
            try:
                size = p.stat().st_size
            except OSError:
                continue
            out.append(FileEntry(path=rel, size=size, role=role, note=note))
            if len(out) > MAX_FILES:
                raise ValueError(f"The folder contains more than {MAX_FILES} files; choose the dataset folder itself.")
    return out


def sniff_table(path: Path) -> str:
    """'events', 'metadata' or 'eeg' for a CSV/TSV whose name is not conclusive."""
    try:
        df = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",", nrows=50)
    except Exception:
        return "ignored"
    cols = [str(c).lower().strip() for c in df.columns]
    if "participant_id" in cols:
        return "metadata"
    has_time = any(c in cols for c, _ in TIME_COLS)
    has_desc = any(c in cols for c in DESC_COLS)
    reserved = {"trial", "label", "participant", "subject", "group", "session", "time", "sample", "marker", "event"}
    numeric = [c for c, col in zip(cols, df.columns) if c not in reserved and pd.api.types.is_numeric_dtype(df[col])]
    if has_time and has_desc and len(numeric) <= 3:
        return "events"
    return "eeg" if numeric else "ignored"


# ---------------------------------------------------------------------------
# Event tables
# ---------------------------------------------------------------------------


def parse_event_table(path: Path) -> tuple[str, list[tuple[float, str, float]], list[str]]:
    """Return (onset unit, [(onset, description, duration_seconds)], notes); onset unit is 'seconds' or 'samples'."""
    notes: list[str] = []
    ext = _ext(path)
    if ext == ".eve" or path.name.lower().endswith(("-eve.fif", "_eve.fif", "-eve.txt")):
        ev = _mne().read_events(path)
        return "samples", [(float(e[0]), str(int(e[2])), 0.0) for e in ev], ["MNE event file (absolute samples)"]
    df = pd.read_csv(path, sep="\t" if ext == ".tsv" else ",", na_values=["n/a", "N/A", "NaN", ""],
                     keep_default_na=True)
    cols = {str(c).lower().strip(): c for c in df.columns}
    tcol = next(((cols[c], u) for c, u in TIME_COLS if c in cols), None)
    if tcol is None:
        raise ValueError("Event file needs an 'onset' (seconds), 'sample' or 'latency' column.")
    dcol = None
    for c in DESC_COLS:
        if c in cols and df[cols[c]].notna().any():
            dcol = cols[c]
            break
    if dcol is None:
        raise ValueError("Event file needs a description column such as 'trial_type', 'label', 'event' or 'value'.")
    durcol = cols.get("duration") or cols.get("duration_s") or cols.get("duration_ms")
    dur_scale = 1e-3 if (durcol is not None and durcol == cols.get("duration_ms")) else 1.0
    col, unit = tcol
    out = []
    skipped = 0
    for _, row in df.iterrows():
        t, d = row[col], row[dcol]
        if pd.isna(t) or pd.isna(d) or str(d).strip() == "":
            skipped += 1
            continue
        desc = str(d).strip()
        if re.fullmatch(r"-?\d+\.0+", desc):
            desc = desc.split(".")[0]
        # durations are in seconds (BIDS), whatever unit the onsets use
        dur = float(row[durcol]) * dur_scale if durcol is not None and not pd.isna(row[durcol]) else 0.0
        out.append((float(t) / 1000.0 if unit == "ms" else float(t), desc, dur))
    if skipped:
        notes.append(f"{skipped} row(s) without onset or description were skipped")
    return ("seconds" if unit in ("seconds", "ms") else "samples"), out, notes


# ---------------------------------------------------------------------------
# Recording headers
# ---------------------------------------------------------------------------


def _types(names: list[str], declared: list[str]) -> list[str]:
    return [infer_channel_type(n, t) for n, t in zip(names, declared)]


def _scan_mne(path: Path, rec: ScannedRecording) -> tuple[dict, dict]:
    """Header-only scan of MNE-readable files. Returns (events cache entry, scan extras)."""
    mne = _mne()
    ext = _ext(path)
    low = path.name.lower()
    epochs = None
    raw = None
    if ext in (".fif", ".fif.gz") and re.search(r"[-_]epo\.fif(\.gz)?$", low):
        epochs = mne.read_epochs(path, preload=False)
    else:
        try:
            raw = read_raw_any(path, preload=False)
        except Exception as exc:
            msg = str(exc).lower()
            if ext == ".set" and ("trial" in msg or "epoch" in msg):
                epochs = mne.read_epochs_eeglab(path)
            elif ext in (".fif", ".fif.gz"):
                epochs = mne.read_epochs(path, preload=False)
            else:
                raise
    if raw is not None:
        picks = data_picks(raw)
        names = [raw.ch_names[i] for i in picks]
        rec.ch_names, rec.ch_types = names, _types(names, raw.get_channel_types(picks=picks))
        rec.n_channels, rec.sfreq = len(names), float(raw.info["sfreq"])
        rec.n_samples = int(raw.n_times)
        rec.duration_s = raw.n_times / raw.info["sfreq"]
        rec.structure = "continuous"
        events, bad = annotation_events(raw) if len(raw.annotations) else ([], [])
        kind = "annotations"
        if not events:
            events = stim_events(raw)
            kind = "stim channel"
        rec.events = EventSourceInfo(kind=kind if events else "none", n_events=len(events))
        rec.event_counts = dict(Counter(d for _, d, _ in events))
        if bad:
            rec.issues.append(Issue(level="info", code="scan_bad_segments", recording=rec.index,
                                    message=f"{len(bad)} BAD-annotated segment(s) will be excluded at epoching."))
        return ({"unit": "samples", "events": [(float(s), d, float(du)) for s, d, du in events]},
                {"first_samp": int(raw.first_samp)})
    picks = data_picks(epochs)
    names = [epochs.ch_names[i] for i in picks]
    rec.ch_names, rec.ch_types = names, _types(names, epochs.get_channel_types(picks=picks))
    rec.n_channels, rec.sfreq = len(names), float(epochs.info["sfreq"])
    rec.structure = "epoched"
    rec.n_epochs = len(epochs.events)
    rec.epoch_samples, rec.epoch_tmin = len(epochs.times), float(epochs.tmin)
    inv = {v: k for k, v in epochs.event_id.items()}
    labels = [inv.get(int(c), str(int(c))) for c in epochs.events[:, 2]]
    rec.events = EventSourceInfo(kind="epochs", n_events=len(labels))
    rec.event_counts = dict(Counter(labels))
    md = epochs.metadata
    if md is not None:
        for col in ("participant", "subject", "subject_id", "group"):
            if col in md.columns:
                g = Counter(md[col].astype(str))
                if len(g) > 1:
                    rec.inner_groups = dict(g)
                else:
                    rec.participant = next(iter(g))
                    rec.id_source = "epochs metadata"
                break
    return {"unit": "samples", "events": []}, {"first_samp": 0}


def _scan_arrays(paths: list[Path], rec: ScannedRecording, opts: ArrayOptions) -> tuple[dict, dict]:
    """NumPy / CSV recordings: read with the validated array readers (these files are small)."""
    def load(sf):
        return load_files(paths, UploadOptions(sfreq=sf, unit=opts.unit, ch_names=opts.ch_names, tmin=opts.tmin))

    res = load(opts.sfreq)
    sfreq_known = True
    if not res.ok and any(i["code"] == "sfreq_missing" for i in res.issues):
        res = load(1.0)
        sfreq_known = False
    if not res.ok:
        raise ValueError("; ".join(i["message"] for i in res.issues if i["level"] == "error"))
    ds = res.dataset
    rec.ch_names = list(ds.ch_names)
    rec.ch_types = _types(rec.ch_names, ["eeg"] * len(rec.ch_names))
    rec.n_channels = len(rec.ch_names)
    rec.sfreq = float(ds.sfreq) if sfreq_known else None
    rec.unit = ds.unit
    for i in res.issues:
        if i["level"] == "warning":
            rec.issues.append(Issue(level="warning", code="scan_" + i["code"], recording=rec.index, message=i["message"]))
    if isinstance(ds, EpochedData):
        rec.structure = "epoched"
        rec.n_epochs, rec.epoch_samples, rec.epoch_tmin = int(ds.X.shape[0]), int(ds.X.shape[2]), float(ds.tmin)
        rec.events = EventSourceInfo(kind="labels", n_events=int(ds.X.shape[0]))
        rec.event_counts = dict(Counter(map(str, ds.labels)))
        if ds.groups is not None:
            g = Counter(str(v) for v in ds.groups if str(v))
            if len(g) > 1:
                rec.inner_groups = dict(g)
            elif g:
                rec.participant, rec.id_source = next(iter(g)), "file contents"
        if ds.sessions is not None:
            s = {str(v) for v in ds.sessions if str(v)}
            if len(s) == 1:
                rec.session = next(iter(s))
        return {"unit": "samples", "events": []}, {"first_samp": 0}
    assert isinstance(ds, ContinuousData)
    r0 = ds.recordings[0]
    rec.structure = "continuous"
    rec.n_samples = int(r0.data.shape[1])
    rec.duration_s = rec.n_samples / rec.sfreq if rec.sfreq else None
    inv = {v: k for k, v in r0.event_id.items()}
    events = [(float(s), inv.get(int(c), str(int(c))), float(du)) for s, c, du in r0.events]
    rec.events = EventSourceInfo(kind="embedded" if events else "none", n_events=len(events))
    rec.event_counts = dict(Counter(d for _, d, _ in events))
    if r0.group and not rec.participant:
        rec.participant, rec.id_source = r0.group, "file contents"
    return {"unit": "samples", "events": events}, {"first_samp": 0}


# ---------------------------------------------------------------------------
# Dataset scan
# ---------------------------------------------------------------------------


def _companions(rec_file: str, all_files: dict[str, FileEntry]) -> list[str]:
    p = Path(rec_file)
    stem, parent = p.name[: -len(_ext(p))], p.parent.as_posix()
    out = []
    ext = _ext(p)
    for f in all_files.values():
        fp = Path(f.path)
        if fp.parent.as_posix() != parent or f.path == rec_file:
            continue
        fext = _ext(fp)
        fstem = fp.name[: -len(fext)] if fext else fp.name
        if ext == ".set" and fext == ".fdt" and fstem == stem:
            out.append(f.path)
        elif ext == ".vhdr" and fext in (".vmrk", ".eeg") and fstem == stem:
            out.append(f.path)
        elif ext == ".npy" and fext == ".npy" and f.role == "companion":
            out.append(f.path)
    return sorted(out)


def _npy_key(path: str) -> str:
    stem = Path(path).stem.lower()
    for rx in NPY_ROLE_RX.values():
        stem = rx.sub("", stem)
    return re.sub(r"(^|[_\-])(x|data|eeg|signals?)$", "", stem).strip("_-")


def scan_dataset(root: Path, name: str, source: dict, files: list[FileEntry] | None = None,
                 array_options: ArrayOptions | None = None,
                 progress: Callable[[int, int, str], None] = lambda *_: None) -> tuple[DatasetManifest, EventsCache]:
    opts = array_options or ArrayOptions()
    entries = files if files is not None else list_directory(root)
    by_path = {f.path: f for f in entries}
    # refine ambiguous tables by content
    for f in entries:
        if f.role == "eeg" and _ext(Path(f.path)) in (".csv", ".tsv"):
            f.role = sniff_table(root / f.path)
            f.note = None if f.role in ("eeg", "events", "metadata") else "table without channel columns"
    # recordings: main data files (NumPy: one data array per folder/key, companions attach to it)
    rec_files = [f for f in entries if f.role == "eeg"]
    npy_data = [f for f in rec_files if _ext(Path(f.path)) == ".npy"]
    npy_comp = [f for f in entries if f.role == "companion" and _ext(Path(f.path)) == ".npy"]
    recs: list[ScannedRecording] = []
    cache: EventsCache = {}
    extras: dict[int, dict] = {}
    total = len(rec_files)
    for i, f in enumerate(rec_files):
        progress(i, total, f"Reading header {i + 1}/{total}: {f.path}")
        ent = infer_entities(f.path)
        ext = _ext(Path(f.path))
        rec = ScannedRecording(index=len(recs), file=f.path, format=ext.lstrip(".").replace("fif.gz", "fif"),
                               participant=ent.participant, session=ent.session, run=ent.run, task=ent.task,
                               id_source=ent.participant_source)
        if ext == ".npy":
            key, folder = _npy_key(f.path), Path(f.path).parent.as_posix()
            same_dir_data = [d for d in npy_data if Path(d.path).parent.as_posix() == folder]
            rec.companions = sorted(c.path for c in npy_comp if Path(c.path).parent.as_posix() == folder
                                    and (len(same_dir_data) == 1 or _npy_key(c.path) == key))
        else:
            rec.companions = _companions(f.path, by_path)
        for c in rec.companions:
            by_path[c].recording = rec.index
        f.recording = rec.index
        try:
            if ext in (".npz", ".npy", ".csv", ".tsv"):
                paths = [root / f.path] + [root / c for c in rec.companions]
                entry, extra = _scan_arrays(paths, rec, opts)
            else:
                if ext == ".vhdr":
                    need = {".vmrk", ".eeg"} - {_ext(Path(c)) for c in rec.companions}
                    if need:
                        raise FileNotFoundError(f"BrainVision companion file(s) missing: {', '.join(sorted(need))}")
                entry, extra = _scan_mne(root / f.path, rec)
            cache[rec.index] = entry
            extras[rec.index] = extra
        except Exception as exc:  # unreadable recordings are reported, never fatal for the scan
            rec.readable, rec.included = False, False
            rec.exclude_reason = "unreadable"
            msg = str(exc).split("\n")[0][:300]
            if ext == ".set" and ("fdt" in msg.lower() or isinstance(exc, FileNotFoundError)):
                msg = f"companion .fdt data file missing or unreadable ({msg})"
            rec.issues.append(Issue(level="error", code="scan_unreadable", recording=rec.index,
                                    message=f"Could not read this file: {msg}"))
        recs.append(rec)
    progress(total, total, "Matching event files")

    # event files -> recordings
    key_index: dict[str, list[ScannedRecording]] = defaultdict(list)
    for r in recs:
        key_index[match_key(r.file)].append(r)
    event_files = [f for f in entries if f.role == "events"]
    assigned: dict[int, str] = {}
    for f in event_files:
        key, folder = match_key(f.path), Path(f.path).parent.as_posix()
        cands = key_index.get(key, [])
        same = [r for r in cands if Path(r.file).parent.as_posix() == folder]
        target = (same or cands)[0] if (len(same) == 1 or (not same and len(cands) == 1)) else None
        how = "matching name"
        if target is None:
            in_dir = [r for r in recs if Path(r.file).parent.as_posix() == folder]
            if len(in_dir) == 1:
                target, how = in_dir[0], "only recording in its folder"
        if target is None:
            f.note = "no recording with a matching name"
            continue
        if target.structure == "epoched":
            f.note = "ignored: the matching recording is already epoched"
            target.issues.append(Issue(level="info", code="scan_events_ignored", recording=target.index,
                                       message=f"{f.path} ignored (labels come from the epochs)."))
            f.recording = target.index
            continue
        if target.index in assigned:
            f.note = f"ignored: {assigned[target.index]} is already used for this recording"
            target.issues.append(Issue(level="warning", code="scan_multiple_event_files", recording=target.index,
                                       message=f"More than one event file matches; using {assigned[target.index]}."))
            continue
        try:
            unit, evs, notes = parse_event_table(root / f.path)
        except Exception as exc:
            f.note = f"unreadable: {exc}"
            target.issues.append(Issue(level="error", code="scan_bad_event_file", recording=target.index,
                                       message=f"Event file {f.path} could not be parsed: {exc}"))
            continue
        if unit == "samples" and f.path.lower().endswith((".eve", "-eve.fif", "_eve.fif", "-eve.txt")):
            fs = extras.get(target.index, {}).get("first_samp", 0)
            if fs and evs and min(t for t, _, _ in evs) >= fs:
                evs = [(t - fs, d, du) for t, d, du in evs]
        embedded = target.events.n_events
        assigned[target.index] = f.path
        f.recording = target.index
        f.note = f"events for {target.file} ({how})"
        cache[target.index] = {"unit": unit, "dur_unit": "seconds", "events": evs}
        target.events = EventSourceInfo(kind="events_file", file=f.path, n_events=len(evs),
                                        note=("; ".join(notes + ([f"{embedded} embedded marker(s) superseded"]
                                                                  if embedded else [])) or None))
        target.event_counts = dict(Counter(d for _, d, _ in evs))
        if not target.included and target.exclude_reason == "no events" and target.readable:
            target.included, target.exclude_reason = True, None

    # metadata (BIDS)
    metadata: dict = {"participants": {}, "bad_channels": {}, "power_line_hz": None, "sidecars": []}
    for f in entries:
        if f.role != "metadata":
            continue
        low = Path(f.path).name.lower()
        try:
            if low == "participants.tsv":
                df = pd.read_csv(root / f.path, sep="\t", na_values=["n/a"])
                if "participant_id" in df.columns:
                    for _, row in df.iterrows():
                        pid = re.sub(r"^sub-", "", str(row["participant_id"]))
                        metadata["participants"][pid] = {k: (None if pd.isna(v) else v)
                                                         for k, v in row.items() if k != "participant_id"}
            elif low.endswith("_channels.tsv"):
                df = pd.read_csv(root / f.path, sep="\t", na_values=["n/a"])
                if "status" in df.columns and "name" in df.columns:
                    bad = [str(n) for n, s in zip(df["name"], df["status"]) if str(s).lower() == "bad"]
                    for r in key_index.get(match_key(f.path), []):
                        if bad:
                            metadata["bad_channels"][str(r.index)] = bad
                            r.issues.append(Issue(level="info", code="scan_bids_bad_channels", recording=r.index,
                                                  message=f"channels.tsv marks {len(bad)} channel(s) bad: "
                                                          f"{', '.join(bad[:8])}."))
            elif low.endswith("_eeg.json"):
                side = json.loads((root / f.path).read_text(encoding="utf-8"))
                if side.get("PowerLineFrequency"):
                    metadata["power_line_hz"] = side.get("PowerLineFrequency")
                metadata["sidecars"].append(f.path)
            elif low == "dataset_description.json":
                dd = json.loads((root / f.path).read_text(encoding="utf-8"))
                metadata["dataset_description"] = {k: dd.get(k) for k in ("Name", "BIDSVersion", "License")}
                if dd.get("Name") and not name:
                    name = str(dd["Name"])
        except Exception as exc:
            f.note = f"metadata not parsed: {exc}"

    # participant IDs that came only from folder or file names: use the first folder level that varies
    fallback = [r for r in recs if r.id_source in ("top-level folder", "file name")]
    if fallback:
        parts = [Path(r.file).parts[:-1] for r in fallback]
        depth = next((d for d in range(max((len(p) for p in parts), default=0))
                      if len({p[d] if len(p) > d else "" for p in parts}) > 1), None)
        if depth is not None:
            for r, p in zip(fallback, parts):
                if len(p) > depth:
                    r.participant = re.sub(r"[^A-Za-z0-9_-]+", "", p[depth])[:32] or r.participant
                    r.id_source = f"folder level {depth + 1}"

    for r in recs:
        if r.readable and r.events.kind == "none" and not r.event_counts:
            r.included, r.exclude_reason = False, "no events"
    pairs: Counter = Counter()
    for r in recs:
        for ev, n in r.event_counts.items():
            pairs[(r.task, ev)] += n
    manifest = DatasetManifest(name=name or source.get("root_label") or "Imported dataset", source=source,
                               recordings=recs, mapping=suggest_mapping(pairs), array_options=opts,
                               files=entries, metadata=metadata)
    progress(total, total, "Validating")
    return validate_manifest(manifest, cache), cache
