"""Materialise a reviewed :class:`DatasetManifest` into the canonical dataset store.

Recordings are read and written one at a time (never the whole dataset in memory):
channels are harmonised to the common set (re-ordered by name), sampling rates are
unified by polyphase resampling when allowed, events come from the source chosen
during scanning (associated event file, annotations, stim channel or labels), and
raw event descriptions are converted to classes or start/end segments with the
reviewed mapping. Every recording keeps its participant / session / run / task
identity and source file reference.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.signal import resample_poly

from .loaders import UploadOptions, _ext, _mne, annotation_events, data_picks, load_files, read_raw_any
from .manifest import DatasetManifest, EventsCache, build_trials, events_in_samples, lookup, validate_manifest
from .store import ContinuousData, DatasetWriter, EpochedData, Recording


class DatasetImportError(ValueError):
    pass


def _resample(x: np.ndarray, sf: float, target: float, axis: int = -1) -> tuple[np.ndarray, float]:
    frac = Fraction(target / sf).limit_denominator(1000)
    if frac.numerator == frac.denominator:
        return x, 1.0
    y = resample_poly(np.asarray(x, dtype=np.float64), frac.numerator, frac.denominator, axis=axis)
    return y.astype(np.float32), frac.numerator / frac.denominator


def _pick(names: list[str], wanted: list[str], label: str) -> list[int]:
    missing = [c for c in wanted if c not in names]
    if missing:
        raise DatasetImportError(f"{label}: channel(s) missing: {', '.join(missing[:8])}")
    return [names.index(c) for c in wanted]


def import_manifest(m: DatasetManifest, events: EventsCache, root: Path, out_dir: Path,
                    progress: Callable[[int, int, str], None] = lambda *_: None) -> str:
    m = validate_manifest(m, events)
    if not m.importable:
        errs = [i.message for i in m.issues if i.level == "error"]
        errs += [f"{r.file}: {i.message}" for r in m.recordings if r.included for i in r.issues if i.level == "error"]
        raise DatasetImportError("The dataset cannot be imported: " + "; ".join(errs[:5]))
    s = m.summary
    inc = [r for r in m.recordings if r.included]
    common, target = s["common_channels"], float(s["target_sfreq"])
    table = lookup(m.mapping)
    classes = sorted(s["classes"])
    code = {c: i + 1 for i, c in enumerate(classes)}
    units = {r.unit for r in inc}
    unit = "unknown" if "unknown" in units else "uV"
    writer = DatasetWriter(out_dir, s["structure"], target, common, unit, ch_types=s["common_channel_types"],
                           tmin=float(inc[0].epoch_tmin or 0.0) if s["structure"] == "epoched" else 0.0)
    opts = m.array_options
    totals: Counter = Counter()
    for k, r in enumerate(inc):
        progress(k, len(inc), f"Importing {k + 1}/{len(inc)}: {r.file}")
        path = root / r.file
        ext = _ext(path)
        is_array = ext in (".npz", ".npy", ".csv", ".tsv")
        if s["structure"] == "continuous":
            bad: list[tuple[float, float]] = []
            if is_array:
                res = load_files([path] + [root / c for c in r.companions],
                                 UploadOptions(sfreq=r.sfreq or opts.sfreq, unit=opts.unit, ch_names=opts.ch_names,
                                               tmin=opts.tmin))
                if not res.ok:
                    raise DatasetImportError(f"{r.file}: " + "; ".join(i["message"] for i in res.issues
                                                                    if i["level"] == "error"))
                ds = res.dataset
                assert isinstance(ds, ContinuousData)
                data = np.asarray(ds.recordings[0].data)[_pick(list(ds.ch_names), common, r.file)]
                sf = float(ds.sfreq)
            else:
                raw = read_raw_any(path, preload=True)
                names = [raw.ch_names[i] for i in data_picks(raw)]
                idx = _pick(list(raw.ch_names), common, r.file)
                data = (raw.get_data(picks=idx) * 1e6).astype(np.float32)
                sf = float(raw.info["sfreq"])
                if len(raw.annotations):
                    bad = annotation_events(raw)[1]
                del raw, names
            trials, _ = build_trials(events_in_samples(events, r.index, sf), r.task, table)
            factor = 1.0
            if abs(sf - target) > 1e-6:
                data, factor = _resample(data, sf, target)
            ev = np.array([[int(round(smp * factor)), code[lbl], int(round(dur * factor))] for smp, lbl, dur in trials],
                          dtype=np.int64).reshape(-1, 3)
            totals.update(lbl for _, lbl, _ in trials)
            src = f"events file {r.events.file}" if r.events.kind == "events_file" else r.events.kind
            writer.add_recording(Recording(
                name=Path(r.file).name, data=data, events=ev, event_id=dict(code), group=r.participant,
                session=r.session, bad_segments=bad, run=r.run, task=r.task, recording_id=r.recording_id,
                source_file=r.file, event_source=src))
            del data
        else:
            if is_array:
                res = load_files([path] + [root / c for c in r.companions],
                                 UploadOptions(sfreq=r.sfreq or opts.sfreq, unit=opts.unit, ch_names=opts.ch_names,
                                               tmin=opts.tmin))
                if not res.ok:
                    raise DatasetImportError(f"{r.file}: " + "; ".join(i["message"] for i in res.issues
                                                                    if i["level"] == "error"))
                ds = res.dataset
                assert isinstance(ds, EpochedData)
                X = np.asarray(ds.X)[:, _pick(list(ds.ch_names), common, r.file)]
                labels = [str(v) for v in ds.labels]
                groups = [str(v) for v in ds.groups] if ds.groups is not None else None
            else:
                mne = _mne()
                if ext == ".set":
                    ep = mne.read_epochs_eeglab(path)
                else:
                    ep = mne.read_epochs(path, preload=True)
                X = (ep.get_data(picks=_pick(list(ep.ch_names), common, r.file)) * 1e6).astype(np.float32)
                inv = {v: k_ for k_, v in ep.event_id.items()}
                labels = [inv.get(int(c), str(int(c))) for c in ep.events[:, 2]]
                groups = None
                md = ep.metadata
                if md is not None and r.inner_groups:
                    for col in ("participant", "subject", "subject_id", "group"):
                        if col in md.columns:
                            groups = [str(v) for v in md[col]]
                            break
            keep, mapped = [], []
            for j, lbl in enumerate(labels):
                e = table.get((r.task, lbl))
                if e is not None and e.action == "class" and e.label:
                    keep.append(j)
                    mapped.append(e.label)
            if not keep:
                continue
            ids = [f"{r.recording_id}:t{j:04d}" for j in keep]
            Xk = X[keep]
            if groups and r.inner_groups:
                gk = [groups[j] for j in keep]
                for g in sorted(set(gk)):
                    sel = [i for i, v in enumerate(gk) if v == g]
                    writer.add_epochs(Xk[sel], [mapped[i] for i in sel], [ids[i] for i in sel], g, r.session, r.run,
                                      r.recording_id, r.file)
            else:
                writer.add_epochs(Xk, mapped, ids, r.participant, r.session, r.run, r.recording_id, r.file)
            totals.update(mapped)
            del X, Xk
    progress(len(inc), len(inc), "Writing dataset")
    if not totals:
        raise DatasetImportError("No trials were produced; check the event mapping.")
    return writer.finish({"manifest": m.model_dump(mode="json"), "imported_class_counts": dict(totals)})
