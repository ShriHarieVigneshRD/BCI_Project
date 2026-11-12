"""Readers for supported EEG formats, producing canonical ``ContinuousData``/``EpochedData``.

Every reader returns structured issues (error / warning / info). A dataset with any
error is rejected before it can be attached to an experiment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .entities import infer_entities
from .store import ContinuousData, Dataset, EpochedData, Recording, candidate_trials, infer_channel_type

SUPPORTED_FORMATS = [
    {"format": "EDF / EDF+ / BDF / GDF", "extensions": [".edf", ".bdf", ".gdf"], "structure": "continuous",
     "requirements": "One file per recording. Event markers are read from EDF+ annotations (or a BDF 'Status' "
                     "channel). Annotations starting with 'BAD' mark noisy segments.",
     "metadata": "Participant / session IDs can be entered per file; otherwise they are inferred from "
                 "patterns such as 'sub-01', 'S01' or 'ses-02' in the file name."},
    {"format": "FIF (MNE)", "extensions": [".fif", ".fif.gz"], "structure": "continuous or epoched",
     "requirements": "Raw FIF files use annotations or a stim channel for events. Epochs FIF files (*-epo.fif) "
                     "use their event codes as labels; epochs.metadata may provide 'participant'/'session'.",
     "metadata": "Same per-file participant/session fields as EDF."},
    {"format": "BrainVision", "extensions": [".vhdr", ".vmrk", ".eeg"], "structure": "continuous",
     "requirements": "Select the .vhdr header together with its .vmrk marker and .eeg data files. Markers become "
                     "events.",
     "metadata": "Participant / session / run IDs from BIDS-style names or folders."},
    {"format": "Event files", "extensions": ["_events.tsv", ".csv", ".tsv", ".eve"], "structure": "events",
     "requirements": "BIDS *_events.tsv or CSV/TSV with an onset (s) or sample column and a trial_type / label / "
                     "event / value column (optional duration). Matched to recordings by shared file-name stem, "
                     "e.g. sub-01_run-1_eeg.edf + sub-01_run-1_events.tsv. MNE *-eve.fif / .eve also supported.",
     "metadata": "Event descriptions are mapped to classes (or start/end segments) in the dataset review."},
    {"format": "EEGLAB", "extensions": [".set", ".fdt"], "structure": "continuous or epoched",
     "requirements": "Upload the .set file together with its companion .fdt file when the data are stored "
                     "separately. Event types become labels.",
     "metadata": "Per-file participant/session fields."},
    {"format": "NumPy .npz", "extensions": [".npz"], "structure": "epoched or continuous",
     "requirements": "Keys: 'X' [trials, channels, samples] + 'y' [trials] (epoched), or 'X' [channels, samples] "
                     "+ optional 'events' [n, 2] = (sample, code) (continuous). Optional: 'sfreq', 'ch_names', "
                     "'groups', 'sessions', 'class_names', 'tmin', 'unit'.",
     "metadata": "Sampling rate must be in the file or entered on upload."},
    {"format": "NumPy .npy", "extensions": [".npy"], "structure": "epoched or continuous",
     "requirements": "A data array ([trials, channels, samples] or [channels, samples]) plus companion arrays "
                     "identified by file name: '*label*.npy' or 'y.npy' (required for epoched data), "
                     "'*group*.npy'/'*subject*.npy', '*session*.npy', '*event*.npy' ([n, 2] or MNE [n, 3]).",
     "metadata": "Sampling rate and (optionally) channel names are entered on upload."},
    {"format": "CSV / TSV", "extensions": [".csv", ".tsv"], "structure": "epoched or continuous",
     "requirements": "Epoched (long format): columns 'trial', 'label', optional 'participant', 'session', "
                     "'time' (s), then one numeric column per channel; every trial must have the same number of "
                     "rows. Continuous: one numeric column per channel, optional 'time' (s) and 'marker' "
                     "(non-empty / non-zero = event label at that sample).",
     "metadata": "Sampling rate from the 'time' column or entered on upload."},
]

ALLOWED_EXTENSIONS = {".edf", ".bdf", ".gdf", ".fif", ".gz", ".set", ".fdt", ".vhdr", ".vmrk", ".eeg", ".npz",
                      ".npy", ".csv", ".tsv", ".eve"}

UNIT_SCALE = {"uV": 1.0, "mV": 1e3, "V": 1e6, "nV": 1e-3, "unknown": 1.0}


@dataclass
class UploadOptions:
    name: str | None = None
    sfreq: float | None = None
    unit: str = "uV"
    ch_names: list[str] | None = None
    tmin: float = 0.0
    file_meta: dict[str, dict] = field(default_factory=dict)  # filename -> {"group": .., "session": ..}


@dataclass
class LoadResult:
    dataset: Dataset | None
    format: str
    issues: list[dict] = field(default_factory=list)

    def error(self, code: str, msg: str) -> None:
        self.issues.append({"level": "error", "code": code, "message": msg})

    def warn(self, code: str, msg: str) -> None:
        self.issues.append({"level": "warning", "code": code, "message": msg})

    def info(self, code: str, msg: str) -> None:
        self.issues.append({"level": "info", "code": code, "message": msg})

    @property
    def ok(self) -> bool:
        return self.dataset is not None and not any(i["level"] == "error" for i in self.issues)


def detect_format(paths: list[Path]) -> str:
    exts = {_ext(p) for p in paths}
    if exts <= {".edf", ".bdf", ".gdf"}:
        return "edf"
    if ".vhdr" in exts and exts <= {".vhdr", ".vmrk", ".eeg"}:
        return "brainvision"
    if exts <= {".fif", ".fif.gz"}:
        return "fif"
    if ".set" in exts and exts <= {".set", ".fdt"}:
        return "eeglab"
    if exts == {".npz"}:
        return "npz"
    if exts == {".npy"}:
        return "npy"
    if exts <= {".csv"} or exts <= {".tsv"}:
        return "csv"
    return "unknown"


def _ext(p: Path) -> str:
    name = p.name.lower()
    if name.endswith(".fif.gz"):
        return ".fif.gz"
    return p.suffix.lower()


def infer_ids(filename: str) -> tuple[str | None, str | None]:
    """Participant and session from a file name (pattern-based only; no fallback to the name itself)."""
    ent = infer_entities(Path(filename).name, flat_fallback_to_stem=False)
    return ent.participant, ent.session


def load_files(paths: list[Path], opts: UploadOptions) -> LoadResult:
    fmt = detect_format(paths)
    if fmt == "unknown":
        res = LoadResult(None, fmt)
        res.error("unsupported_combination",
                  "Unsupported file type or mixed formats. Upload one format at a time (EEGLAB .set + .fdt "
                  "and NumPy data + label arrays are the only multi-file combinations).")
        return res
    try:
        res = {"edf": _load_mne_raw_files, "brainvision": _load_mne_raw_files, "fif": _load_fif,
               "eeglab": _load_eeglab, "npz": _load_npz, "npy": _load_npy, "csv": _load_csv}[fmt](paths, opts)
    except Exception as exc:  # reader errors become validation errors, never 500s
        res = LoadResult(None, fmt)
        res.error("read_failed", f"Could not read the file(s): {exc}")
        return res
    if res.dataset is not None:
        validate_dataset(res)
    return res


# ---------------------------------------------------------------------------
# MNE-based readers
# ---------------------------------------------------------------------------


def _mne():
    import mne

    mne.set_log_level("ERROR")
    return mne


def read_raw_any(path: Path, preload: bool = True):
    """Open any supported continuous MNE format."""
    mne = _mne()
    ext = _ext(path)
    if ext == ".edf":
        return mne.io.read_raw_edf(path, preload=preload)
    if ext == ".bdf":
        return mne.io.read_raw_bdf(path, preload=preload)
    if ext == ".gdf":
        return mne.io.read_raw_gdf(path, preload=preload)
    if ext == ".vhdr":
        return mne.io.read_raw_brainvision(path, preload=preload)
    if ext in (".fif", ".fif.gz"):
        return mne.io.read_raw_fif(path, preload=preload)
    if ext == ".set":
        return mne.io.read_raw_eeglab(path, preload=preload)
    raise ValueError(f"Unsupported continuous format: {path.name}")


def data_picks(raw_or_epochs) -> list[int]:
    mne = _mne()
    info = raw_or_epochs.info
    picks = mne.pick_types(info, eeg=True, eog=True, emg=True, ecg=True, seeg=True, ecog=True, misc=False, stim=False,
                           exclude=[])
    if len(picks) == 0:
        picks = mne.pick_types(info, misc=True, exclude=[])
    return [int(i) for i in picks if infer_channel_type(info["ch_names"][i]) != "stim"]


def annotation_events(raw) -> tuple[list[tuple[int, str, int]], list[tuple[float, float]]]:
    """(sample, description, duration_samples) for every non-BAD annotation, plus BAD segments (onset_s, dur_s).

    Samples are relative to the first sample of the data. Durations are kept so annotated
    segments are preserved.
    """
    sf = raw.info["sfreq"]
    ann = raw.annotations
    events, bad = [], []
    for a in ann:
        samp = int(round(float(a["onset"]) * sf))
        if ann.orig_time is not None:
            samp -= int(raw.first_samp)
        desc = str(a["description"]).strip()
        dur = float(a["duration"])
        if desc.lower().startswith("bad") or desc.lower().startswith("edge"):
            bad.append((samp / sf, dur))
            continue
        events.append((samp, desc, int(round(dur * sf))))
    return events, bad


def stim_events(raw) -> list[tuple[int, str, int]]:
    mne = _mne()
    stim = mne.pick_types(raw.info, stim=True, exclude=[])
    if not len(stim):
        return []
    ev = mne.find_events(raw, stim_channel=raw.ch_names[stim[0]], shortest_event=1)
    return [(int(e[0] - raw.first_samp), str(int(e[2])), 0) for e in ev]


def events_to_arrays(events: list[tuple[int, str, int]]) -> tuple[np.ndarray, dict[str, int]]:
    labels = sorted({d for _, d, _ in events})
    event_id = {d: i + 1 for i, d in enumerate(labels)}
    arr = np.array([[s, event_id[d], du] for s, d, du in sorted(events)], dtype=np.int64).reshape(-1, 3)
    return arr, event_id


def _raw_to_recording(raw, name: str, opts: UploadOptions, res: LoadResult) -> tuple[Recording, list[str], list[str]]:
    picks = data_picks(raw)
    ch_names = [raw.ch_names[i] for i in picks]
    ch_types = [infer_channel_type(n, t) for n, t in zip(ch_names, raw.get_channel_types(picks=picks))]
    data = raw.get_data(picks=picks).astype(np.float32) * 1e6  # V -> uV
    ev_list, bad = annotation_events(raw) if len(raw.annotations) else ([], [])
    source = "annotations"
    if not ev_list:
        ev_list = stim_events(raw)
        source = "stim channel"
        if ev_list:
            res.info("events_from_stim", f"{name}: events read from the stim channel.")
    events, event_id = events_to_arrays(ev_list)
    meta = opts.file_meta.get(name, {})
    g, s = infer_ids(name)
    group = meta.get("group") or g  # header identity fields (names, birth dates) are never used
    session = meta.get("session") or s
    rec = Recording(name=name, data=data, events=events, event_id=event_id, group=group or None,
                    session=session or None, bad_segments=bad, source_file=name,
                    event_source=source if len(events) else "none")
    return rec, ch_names, ch_types


def _combine_recordings(recs: list[tuple[Recording, list[str], list[str], float, object]], res: LoadResult,
                        fmt_unit: str = "uV") -> ContinuousData | None:
    ref_rec, ref_ch, ref_types, ref_sf, ref_info = recs[0]
    out = [ref_rec]
    for rec, ch, _types, sf, _info in recs[1:]:
        if abs(sf - ref_sf) > 1e-6:
            res.error("sfreq_mismatch", f"{rec.name} has sampling rate {sf:g} Hz but {ref_rec.name} has {ref_sf:g} Hz.")
            return None
        if ch != ref_ch:
            if sorted(ch) == sorted(ref_ch):
                order = [ch.index(c) for c in ref_ch]
                rec.data = rec.data[order]
                res.info("channel_reordered", f"{rec.name}: channels re-ordered to match {ref_rec.name}.")
            else:
                diff = sorted(set(ch) ^ set(ref_ch))
                res.error("channel_mismatch", f"{rec.name} has different channels from {ref_rec.name} "
                                              f"(differences: {', '.join(diff[:10])}).")
                return None
        out.append(rec)
    hp = getattr(ref_info, "get", lambda *_: None)("highpass")
    lp = getattr(ref_info, "get", lambda *_: None)("lowpass")
    return ContinuousData(sfreq=float(ref_sf), ch_names=list(ref_ch), unit=fmt_unit, recordings=out,
                          ch_types=list(ref_types), highpass=hp, lowpass=lp)


def _load_mne_raw_files(paths: list[Path], opts: UploadOptions) -> LoadResult:
    res = LoadResult(None, "brainvision" if any(_ext(p) == ".vhdr" for p in paths) else "edf")
    recs = []
    for p in sorted(paths):
        if _ext(p) in (".vmrk", ".eeg"):
            continue  # BrainVision companions are opened through the .vhdr header
        raw = read_raw_any(p, preload=True)
        rec, ch, types = _raw_to_recording(raw, p.name, opts, res)
        recs.append((rec, ch, types, raw.info["sfreq"], raw.info))
    res.dataset = _combine_recordings(recs, res)
    return res


def _epochs_to_dataset(epochs, name: str, opts: UploadOptions, res: LoadResult) -> EpochedData:
    mne = _mne()
    picks = mne.pick_types(epochs.info, eeg=True, eog=True, emg=True, ecg=True, seeg=True, ecog=True, stim=False,
                           exclude=[])
    X = epochs.get_data(picks=picks).astype(np.float32) * 1e6
    inv = {v: k for k, v in epochs.event_id.items()}
    labels = np.array([inv.get(int(c), str(int(c))) for c in epochs.events[:, 2]])
    groups = sessions = None
    md = epochs.metadata
    if md is not None:
        for col in ("participant", "subject", "subject_id", "group"):
            if col in md.columns:
                groups = md[col].astype(str).to_numpy()
                break
        for col in ("session", "run"):
            if col in md.columns:
                sessions = md[col].astype(str).to_numpy()
                break
    meta = opts.file_meta.get(name, {})
    g, s = infer_ids(name)
    if groups is None and (meta.get("group") or g):
        groups = np.array([meta.get("group") or g] * len(X))
    if sessions is None and (meta.get("session") or s):
        sessions = np.array([meta.get("session") or s] * len(X))
    ch_names = [epochs.ch_names[i] for i in picks]
    return EpochedData(sfreq=float(epochs.info["sfreq"]), ch_names=ch_names, unit="uV", X=X, labels=labels,
                       tmin=float(epochs.tmin), groups=groups, sessions=sessions,
                       ch_types=[infer_channel_type(n, t) for n, t in zip(ch_names, epochs.get_channel_types(picks=picks))],
                       highpass=epochs.info.get("highpass"),
                       lowpass=epochs.info.get("lowpass"))


def _concat_epoched(parts: list[EpochedData], names: list[str], res: LoadResult) -> EpochedData | None:
    ref = parts[0]
    for p, n in zip(parts[1:], names[1:]):
        if p.ch_names != ref.ch_names or abs(p.sfreq - ref.sfreq) > 1e-6 or p.X.shape[2] != ref.X.shape[2]:
            res.error("epoch_mismatch", f"{n} differs from {names[0]} in channels, sampling rate or epoch length.")
            return None
    n_total = sum(len(p.X) for p in parts)

    def cat(attr):
        vals = [getattr(p, attr) for p in parts]
        if all(v is None for v in vals):
            return None
        return np.concatenate([v if v is not None else np.array([""] * len(p.X)) for v, p in zip(vals, parts)])

    return EpochedData(sfreq=ref.sfreq, ch_names=ref.ch_names, unit=ref.unit,
                       X=np.concatenate([p.X for p in parts]), labels=np.concatenate([p.labels for p in parts]),
                       tmin=ref.tmin, groups=cat("groups"), sessions=cat("sessions"),
                       trial_ids=np.array([f"t{i:05d}" for i in range(n_total)]), ch_types=ref.ch_types,
                       highpass=ref.highpass, lowpass=ref.lowpass)


def _load_fif(paths: list[Path], opts: UploadOptions) -> LoadResult:
    mne = _mne()
    res = LoadResult(None, "fif")
    raws, epochs_parts = [], []
    for p in sorted(paths):
        is_epo = re.search(r"[-_]epo\.fif(\.gz)?$", p.name.lower()) is not None
        obj = None
        if not is_epo:
            try:
                obj = mne.io.read_raw_fif(p, preload=True)
            except Exception:
                is_epo = True
        if is_epo:
            obj = mne.read_epochs(p, preload=True)
            epochs_parts.append((_epochs_to_dataset(obj, p.name, opts, res), p.name))
        else:
            rec, ch, types = _raw_to_recording(obj, p.name, opts, res)
            raws.append((rec, ch, types, obj.info["sfreq"], obj.info))
    if raws and epochs_parts:
        res.error("mixed_structure", "Upload either raw FIF files or epochs FIF files, not both.")
        return res
    if raws:
        res.dataset = _combine_recordings(raws, res)
    else:
        res.dataset = _concat_epoched([e for e, _ in epochs_parts], [n for _, n in epochs_parts], res)
    return res


def _load_eeglab(paths: list[Path], opts: UploadOptions) -> LoadResult:
    mne = _mne()
    res = LoadResult(None, "eeglab")
    sets = [p for p in paths if _ext(p) == ".set"]
    raws, epochs_parts = [], []
    for p in sorted(sets):
        try:
            raw = mne.io.read_raw_eeglab(p, preload=True)
            rec, ch, types = _raw_to_recording(raw, p.name, opts, res)
            raws.append((rec, ch, types, raw.info["sfreq"], raw.info))
        except (FileNotFoundError, OSError) as exc:
            res.error("companion_missing", f"{p.name}: the companion data file (.fdt) referenced by the .set file "
                                           f"was not uploaded or cannot be read ({exc}). Upload both files together.")
            return res
        except Exception as exc:
            if "trial" in str(exc).lower() or "epoch" in str(exc).lower():
                try:
                    ep = mne.read_epochs_eeglab(p)
                except (FileNotFoundError, OSError) as exc2:
                    res.error("companion_missing", f"{p.name}: companion .fdt file missing ({exc2}).")
                    return res
                epochs_parts.append((_epochs_to_dataset(ep, p.name, opts, res), p.name))
            else:
                raise
    if raws and epochs_parts:
        res.error("mixed_structure", "Upload either continuous or epoched EEGLAB files, not both.")
        return res
    if raws:
        res.dataset = _combine_recordings(raws, res)
    elif epochs_parts:
        res.dataset = _concat_epoched([e for e, _ in epochs_parts], [n for _, n in epochs_parts], res)
    return res


# ---------------------------------------------------------------------------
# NumPy / CSV readers
# ---------------------------------------------------------------------------


def _str_array(a: np.ndarray, class_names: list[str] | None = None) -> np.ndarray:
    a = np.asarray(a).reshape(-1)
    if class_names is not None and np.issubdtype(a.dtype, np.integer):
        return np.array([class_names[int(v)] if 0 <= int(v) < len(class_names) else str(v) for v in a])
    if np.issubdtype(a.dtype, np.floating) and np.all(np.mod(a, 1) == 0):
        a = a.astype(np.int64)
    return a.astype(str)


def _array_dataset(X: np.ndarray, y, groups, sessions, events, sfreq: float | None, ch_names, unit: str,
                   tmin: float, class_names, res: LoadResult) -> Dataset | None:
    if sfreq is None or not np.isfinite(sfreq) or sfreq <= 0:
        res.error("sfreq_missing", "Sampling rate is required (include 'sfreq' in the file or enter it on upload).")
        return None
    scale = UNIT_SCALE.get(unit, 1.0)
    X = np.asarray(X, dtype=np.float32) * scale
    unit_out = "unknown" if unit == "unknown" else "uV"
    if X.ndim == 3:
        n, C, T = X.shape
        names = list(ch_names) if ch_names is not None else [f"Ch{i + 1}" for i in range(C)]
        if len(names) != C:
            res.error("ch_names_mismatch", f"{len(names)} channel names given for {C} channels.")
            return None
        if y is None:
            res.error("labels_missing", "Epoched arrays need labels ('y' key or a '*label*.npy' file).")
            return None
        labels = _str_array(y, class_names)
        if len(labels) != n:
            res.error("labels_length", f"{len(labels)} labels for {n} trials.")
            return None
        g = _str_array(groups) if groups is not None else None
        s = _str_array(sessions) if sessions is not None else None
        for nm, arr in (("groups", g), ("sessions", s)):
            if arr is not None and len(arr) != n:
                res.error(f"{nm}_length", f"{len(arr)} {nm} entries for {n} trials.")
                return None
        return EpochedData(sfreq=float(sfreq), ch_names=names, unit=unit_out, X=X, labels=labels, tmin=tmin,
                           groups=g, sessions=s)
    if X.ndim == 2:
        C, S = X.shape
        if C > S:
            res.warn("orientation", f"Array has more rows ({C}) than columns ({S}); it is interpreted as "
                                    "[channels, samples]. Transpose it if rows are samples.")
        names = list(ch_names) if ch_names is not None else [f"Ch{i + 1}" for i in range(C)]
        if len(names) != C:
            res.error("ch_names_mismatch", f"{len(names)} channel names given for {C} channels.")
            return None
        ev = np.zeros((0, 2), dtype=np.int64)
        event_id: dict[str, int] = {}
        if events is not None:
            events = np.asarray(events)
            if events.ndim != 2 or events.shape[1] not in (2, 3):
                res.error("events_shape", "Events must be [n, 2] (sample, code) or MNE-style [n, 3].")
                return None
            codes = events[:, -1].astype(np.int64)
            ev = np.c_[events[:, 0].astype(np.int64), codes]
            names_map = class_names or None
            event_id = {(names_map[c] if names_map and 0 <= c < len(names_map) else str(c)): int(c)
                        for c in np.unique(codes)}
        rec = Recording(name="recording", data=X, events=ev, event_id=event_id,
                        group=str(np.asarray(groups).reshape(-1)[0]) if groups is not None else None,
                        session=str(np.asarray(sessions).reshape(-1)[0]) if sessions is not None else None)
        return ContinuousData(sfreq=float(sfreq), ch_names=names, unit=unit_out, recordings=[rec])
    res.error("ndim", f"Data array must be 2-D or 3-D, got {X.ndim}-D.")
    return None


def _load_npz(paths: list[Path], opts: UploadOptions) -> LoadResult:
    res = LoadResult(None, "npz")
    if len(paths) != 1:
        res.error("multi_npz", "Upload a single .npz file.")
        return res
    with np.load(paths[0], allow_pickle=False) as z:
        keys = {k.lower(): k for k in z.files}

        def get(*names):
            for n in names:
                if n in keys:
                    return z[keys[n]]
            return None

        X = get("x", "data", "eeg")
        if X is None:
            res.error("missing_x", f"No data array found (expected key 'X'); keys present: {', '.join(z.files)}.")
            return res
        sf = get("sfreq", "fs", "srate")
        sfreq = float(np.asarray(sf).reshape(-1)[0]) if sf is not None else opts.sfreq
        chn = get("ch_names", "channels")
        cls = get("class_names")
        unit_arr = get("unit")
        unit = str(np.asarray(unit_arr).reshape(-1)[0]) if unit_arr is not None else opts.unit
        tmin_arr = get("tmin")
        res.dataset = _array_dataset(
            X, get("y", "labels"), get("groups", "subjects", "participants", "subject"), get("sessions", "session"),
            get("events"), sfreq, list(map(str, chn)) if chn is not None else opts.ch_names, unit,
            float(np.asarray(tmin_arr).reshape(-1)[0]) if tmin_arr is not None else opts.tmin,
            list(map(str, cls)) if cls is not None else None, res)
    return res


def _load_npy(paths: list[Path], opts: UploadOptions) -> LoadResult:
    res = LoadResult(None, "npy")
    roles: dict[str, np.ndarray] = {}
    data_files = []
    for p in paths:
        stem = p.stem.lower()
        arr = np.load(p, allow_pickle=False)
        if "label" in stem or stem in ("y", "targets"):
            roles["y"] = arr
        elif "group" in stem or "subject" in stem or "participant" in stem:
            roles["groups"] = arr
        elif "session" in stem:
            roles["sessions"] = arr
        elif "event" in stem:
            roles["events"] = arr
        else:
            data_files.append((p.name, arr))
    if len(data_files) != 1:
        res.error("npy_roles", f"Expected exactly one data array, found {len(data_files)} "
                               f"({', '.join(n for n, _ in data_files) or 'none'}). Name companion arrays "
                               "'labels.npy', 'groups.npy', 'sessions.npy' or 'events.npy'.")
        return res
    res.dataset = _array_dataset(data_files[0][1], roles.get("y"), roles.get("groups"), roles.get("sessions"),
                                 roles.get("events"), opts.sfreq, opts.ch_names, opts.unit, opts.tmin, None, res)
    return res


RESERVED = {"trial", "label", "participant", "subject", "group", "session", "time", "sample", "marker", "event"}


def _load_csv(paths: list[Path], opts: UploadOptions) -> LoadResult:
    res = LoadResult(None, "csv")
    if len(paths) != 1:
        res.error("multi_csv", "Upload a single CSV/TSV file.")
        return res
    p = paths[0]
    df = pd.read_csv(p, sep="\t" if _ext(p) == ".tsv" else ",")
    cols = {c.lower().strip(): c for c in df.columns}
    ch_cols = [c for c in df.columns if c.lower().strip() not in RESERVED]
    non_numeric = [c for c in ch_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        res.error("non_numeric", f"Channel columns must be numeric: {', '.join(map(str, non_numeric[:8]))}.")
        return res
    if not ch_cols:
        res.error("no_channels", "No channel columns found.")
        return res
    sfreq = opts.sfreq
    tcol = cols.get("time")
    if sfreq is None and tcol is not None:
        t = df[tcol].to_numpy(dtype=float)
        dt = np.diff(t)
        dt = dt[dt > 0]
        if len(dt):
            sfreq = float(round(1.0 / np.median(dt), 6))
            res.info("sfreq_inferred", f"Sampling rate inferred from the 'time' column: {sfreq:g} Hz.")
    if "trial" in cols:
        tr = df[cols["trial"]]
        if "label" not in cols:
            res.error("labels_missing", "Epoched CSV needs a 'label' column.")
            return res
        order = pd.unique(tr)
        sizes = tr.value_counts()
        if sizes.nunique() != 1:
            res.error("ragged_trials", "Every trial must have the same number of rows (samples); found sizes "
                                       f"{sorted(sizes.unique().tolist())[:6]}.")
            return res
        T = int(sizes.iloc[0])
        X = np.stack([df.loc[tr == k, ch_cols].to_numpy(dtype=np.float32).T for k in order])

        def per_trial(colname):
            if colname not in cols:
                return None
            vals = []
            for k in order:
                v = df.loc[tr == k, cols[colname]].unique()
                if len(v) != 1:
                    res.error("inconsistent", f"Column '{colname}' changes within trial {k}.")
                    return None
                vals.append(str(v[0]))
            return np.array(vals)

        y = per_trial("label")
        g = per_trial("participant") if "participant" in cols else per_trial("subject") if "subject" in cols else \
            per_trial("group")
        s = per_trial("session")
        if any(i["level"] == "error" for i in res.issues):
            return res
        tmin = opts.tmin
        if tcol is not None:
            tmin = float(df.loc[tr == order[0], tcol].iloc[0])
        res.dataset = _array_dataset(X, y, g, s, None, sfreq, [str(c) for c in ch_cols], opts.unit, tmin, None, res)
        if res.dataset is not None:
            res.info("csv_epoched", f"Read {len(order)} trials x {T} samples from long-format CSV.")
        return res
    X = df[ch_cols].to_numpy(dtype=np.float32).T
    events = None
    mcol = cols.get("marker") or cols.get("event")
    event_names = None
    if mcol is not None:
        m = df[mcol]
        mask = m.notna() & (m.astype(str).str.strip() != "") & (m.astype(str) != "0") & (m.astype(str) != "0.0")
        labels = m[mask].astype(str).to_numpy()
        event_names = sorted(set(labels))
        codes = np.array([event_names.index(v) for v in labels], dtype=np.int64)
        events = np.c_[np.flatnonzero(mask.to_numpy()), codes]
    g = df[cols["participant"]].iloc[:1].to_numpy() if "participant" in cols else None
    s = df[cols["session"]].iloc[:1].to_numpy() if "session" in cols else None
    res.dataset = _array_dataset(X, None, g, s, events, sfreq, [str(c) for c in ch_cols], opts.unit, opts.tmin,
                                 event_names, res)
    return res


# ---------------------------------------------------------------------------
# Common validation
# ---------------------------------------------------------------------------


def validate_dataset(res: LoadResult) -> None:
    ds = res.dataset
    if ds is None:
        return
    if not ds.ch_names:
        res.error("no_channels", "The dataset has no channels.")
    dup = sorted({c for c in ds.ch_names if ds.ch_names.count(c) > 1})
    if dup:
        res.error("duplicate_channels", f"Duplicate channel names: {', '.join(dup)}.")
    if not ds.sfreq or ds.sfreq <= 0:
        res.error("sfreq_invalid", "Sampling rate must be positive.")
    arrays = [ds.X] if isinstance(ds, EpochedData) else [r.data for r in ds.recordings]
    bad = sum(int((~np.isfinite(a)).sum()) for a in arrays)
    if bad:
        res.error("non_finite", f"The signal contains {bad} NaN/Inf values. Clean or interpolate them first.")
    med = float(np.median(np.abs(np.concatenate([a.reshape(-1)[:200_000] for a in arrays])))) if arrays else 0.0
    if ds.unit == "uV" and med > 0:
        if med > 5e4:
            res.warn("unit_suspect", f"Median |amplitude| is {med:.3g} uV - unusually large; check the unit setting.")
        elif med < 1e-3:
            res.warn("unit_suspect", f"Median |amplitude| is {med:.3g} uV - values look like volts; "
                                     "set the unit to V.")
    trials = candidate_trials(ds)
    if isinstance(ds, EpochedData):
        classes = sorted(set(map(str, ds.labels)))
        if len(classes) < 2:
            res.error("single_class", "At least two classes are required for classification.")
        if ds.X.shape[2] < 8:
            res.error("short_epochs", f"Epochs have only {ds.X.shape[2]} samples.")
    else:
        if not trials:
            res.warn("no_events", "No event markers were found. The dataset can be inspected, but epoching (and "
                                  "therefore training) cannot proceed; trial boundaries will not be invented.")
        elif len({t.label for t in trials}) < 2:
            res.warn("single_event_type", "Only one event type found; classification needs at least two classes.")
    counts: dict[str, int] = {}
    for t in trials:
        counts[t.label] = counts.get(t.label, 0) + 1
    few = [f"{k} ({v})" for k, v in counts.items() if v < 5]
    if few:
        res.warn("few_trials", f"Classes with fewer than 5 trials: {', '.join(few)}.")
    if trials and not any(t.group for t in trials):
        res.info("no_groups", "No participant IDs found; only trial-level splits are available.")
