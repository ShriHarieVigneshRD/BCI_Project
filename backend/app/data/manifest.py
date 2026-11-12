"""Dataset manifest: the reviewable description of a multi-recording dataset before (and after) import.

The manifest is the single source of truth for what an imported dataset contains:
which files are recordings, their participant / session / run / task identities,
where their events come from, how raw event descriptions map to classes (or to
start/end *segments*), and which recordings are included. Every count shown to the
user (trials, classes, participants, seconds) is recomputed here from the scanned
events and the current mapping; nothing is a placeholder.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

from .store import infer_channel_type, make_recording_id

Level = Literal["error", "warning", "info"]
Action = Literal["class", "segment_start", "segment_end", "ignore"]


class Issue(BaseModel):
    level: Level
    code: str
    message: str
    recording: int | None = None


class EventSourceInfo(BaseModel):
    kind: Literal["events_file", "annotations", "stim channel", "epochs", "labels", "embedded", "none"] = "none"
    file: str | None = None
    n_events: int = 0
    note: str | None = None


class ScannedRecording(BaseModel):
    index: int
    file: str
    format: str
    companions: list[str] = []
    structure: Literal["continuous", "epoched"] = "continuous"
    participant: str | None = None
    session: str | None = None
    run: str | None = None
    task: str | None = None
    id_source: str | None = None
    sfreq: float | None = None
    n_channels: int = 0
    ch_names: list[str] = []
    ch_types: list[str] = []
    n_samples: int | None = None
    duration_s: float | None = None
    n_epochs: int | None = None
    epoch_samples: int | None = None
    epoch_tmin: float | None = None
    inner_groups: dict[str, int] | None = None      # epoched files that already carry per-epoch participant IDs
    unit: str = "uV"
    events: EventSourceInfo = EventSourceInfo()
    event_counts: dict[str, int] = {}
    included: bool = True
    exclude_reason: str | None = None
    readable: bool = True
    issues: list[Issue] = []
    recording_id: str | None = None                 # computed
    n_trials: int = 0                               # computed (after mapping / pairing)
    class_counts: dict[str, int] = {}               # computed


class MappingEntry(BaseModel):
    task: str | None = None
    event: str
    action: Action = "class"
    label: str | None = None
    count: int = 0                                  # computed


class ArrayOptions(BaseModel):
    sfreq: float | None = Field(None, gt=0)
    unit: Literal["uV", "mV", "V", "nV", "unknown"] = "uV"
    ch_names: list[str] | None = None
    tmin: float = 0.0


class FileEntry(BaseModel):
    path: str
    size: int = 0
    role: Literal["eeg", "events", "companion", "metadata", "ignored"]
    recording: int | None = None
    note: str | None = None


class DatasetManifest(BaseModel):
    version: int = 1
    name: str
    source: dict = {}
    recordings: list[ScannedRecording] = []
    mapping: list[MappingEntry] = []
    channel_policy: Literal["intersection", "strict"] = "intersection"
    sfreq_policy: Literal["resample", "strict"] = "resample"
    target_sfreq: float | None = Field(None, gt=0)
    array_options: ArrayOptions = ArrayOptions()
    files: list[FileEntry] = []
    metadata: dict = {}
    issues: list[Issue] = []                        # computed
    summary: dict = {}                              # computed
    importable: bool = False                        # computed


# Raw events kept next to the manifest (not sent to the browser):
# index -> {"unit": "samples" | "seconds", "dur_unit": "samples" | "seconds", "events": [(time, desc, duration)]}
# Onsets and durations carry separate units: BIDS event files give durations in seconds even with sample onsets.
EventsCache = dict[int, dict]


def events_in_samples(cache: EventsCache, index: int, sfreq: float | None) -> list[tuple[float, str, float]]:
    entry = cache.get(index) or {"unit": "samples", "events": []}
    f = sfreq or 1.0
    tf = f if entry.get("unit") == "seconds" else 1.0
    df = f if entry.get("dur_unit", entry.get("unit")) == "seconds" else 1.0
    return [(float(t) * tf, d, float(du) * df) for t, d, du in entry.get("events", [])]


# ---------------------------------------------------------------------------
# Event description heuristics
# ---------------------------------------------------------------------------

START_RX = re.compile(r"(?:^|[^a-z])(start|begin|onset|on)(?:$|[^a-z])", re.I)
END_RX = re.compile(r"(?:^|[^a-z])(end|stop|finish|offset|off)(?:$|[^a-z])", re.I)
NOISE_RX = re.compile(
    r"(^\d{4}[-_/]\d{1,2}[-_/]\d{1,2})|(\bdate\b)|(timestart)|(^recording\s*start)|(^boundary$)|(new\s*segment)|"
    r"(impedance)|(^pause)|(^comment)|(^\d{1,2}:\d{2})", re.I)


def block_key(desc: str) -> str:
    """'FC-1 start' and 'FC-1 end' share the block key 'fc1'."""
    d = START_RX.sub(" ", desc)
    d = END_RX.sub(" ", d)
    return re.sub(r"[^a-z0-9]+", "", d.lower())


def class_stem(desc: str) -> str:
    """'FC-1 start' -> 'FC'; 'V12 end' -> 'V'; 'NSR-start' -> 'NSR'."""
    d = START_RX.sub(" ", desc)
    d = END_RX.sub(" ", d)
    d = re.sub(r"\d+", " ", d)
    d = re.sub(r"[^A-Za-z]+", " ", d).strip()
    return re.sub(r"\s+", "_", d) or desc.strip()


def suggest_mapping(pairs: Counter) -> list[MappingEntry]:
    """Default mapping for (task, description) -> action/label. The user reviews and edits it."""
    stems = Counter(re.sub(r"\d+", "#", e).strip().lower() for (_t, e) in pairs)
    out = []
    for (task, ev), n in sorted(pairs.items(), key=lambda kv: (kv[0][0] or "", kv[0][1])):
        if NOISE_RX.search(ev):
            out.append(MappingEntry(task=task, event=ev, action="ignore", label=None, count=n))
        elif START_RX.search(ev):
            out.append(MappingEntry(task=task, event=ev, action="segment_start", label=class_stem(ev), count=n))
        elif END_RX.search(ev):
            out.append(MappingEntry(task=task, event=ev, action="segment_end", label=class_stem(ev), count=n))
        else:
            stem = re.sub(r"\d+", "#", ev).strip().lower()
            label = class_stem(ev) if (stems[stem] >= 3 and "#" in stem and class_stem(ev) != ev) else ev
            out.append(MappingEntry(task=task, event=ev, action="class", label=label, count=n))
    return out


def lookup(mapping: list[MappingEntry]) -> dict[tuple[str | None, str], MappingEntry]:
    return {(m.task, m.event): m for m in mapping}


def build_trials(events: list[tuple[float, str, float]], task: str | None,
                 table: dict[tuple[str | None, str], MappingEntry]) -> tuple[list[tuple[int, str, int]], dict]:
    """Apply the mapping to a recording's raw events.

    Returns trials as (sample, class label, duration_samples) and pairing statistics. A
    ``segment_start`` opens a block that the next ``segment_end`` with the same block key
    closes (falling back to the only open block); unmatched markers are reported.
    """
    trials: list[tuple[int, str, int]] = []
    open_blocks: list[tuple[int, str, str]] = []  # (sample, label, key)
    stats = {"unpaired_starts": 0, "orphan_ends": 0, "ignored": 0, "unmapped": 0}
    for sample, desc, dur in sorted(events, key=lambda e: (e[0], e[1])):
        m = table.get((task, desc))
        if m is None:
            stats["unmapped"] += 1
            continue
        if m.action == "ignore" or (m.action in ("class", "segment_start") and not m.label):
            stats["ignored"] += 1
        elif m.action == "class":
            trials.append((int(round(sample)), m.label, int(round(dur))))
        elif m.action == "segment_start":
            open_blocks.append((int(round(sample)), m.label, block_key(desc)))
        else:  # segment_end
            key = block_key(desc)
            idx = next((i for i in range(len(open_blocks) - 1, -1, -1) if open_blocks[i][2] == key), None)
            if idx is None and len(open_blocks) == 1:
                idx = 0
            if idx is None:
                stats["orphan_ends"] += 1
                continue
            s0, label, _ = open_blocks.pop(idx)
            if sample > s0:
                trials.append((s0, label, int(round(sample - s0))))
    stats["unpaired_starts"] = len(open_blocks)
    trials.sort()
    return trials, stats


# ---------------------------------------------------------------------------
# Validation and summary (recomputed after every edit)
# ---------------------------------------------------------------------------


def _mode(values: list[float]) -> float:
    c = Counter(round(v, 6) for v in values)
    return max(c.items(), key=lambda kv: (kv[1], kv[0]))[0]


def validate_manifest(m: DatasetManifest, events: EventsCache) -> DatasetManifest:
    issues: list[Issue] = []
    recs = m.recordings
    inc = [r for r in recs if r.included]
    table = lookup(m.mapping)
    # mapping rows must cover every (task, event) of every readable recording
    seen: Counter = Counter()
    for r in recs:
        if r.readable:
            for ev, n in r.event_counts.items():
                seen[(r.task, ev)] += n
    known = {(e.task, e.event) for e in m.mapping}
    missing = Counter({k: v for k, v in seen.items() if k not in known})
    if missing:
        m.mapping = m.mapping + suggest_mapping(missing)
        table = lookup(m.mapping)
    inc_counts: Counter = Counter()
    for r in inc:
        for ev, n in r.event_counts.items():
            inc_counts[(r.task, ev)] += n
    for e in m.mapping:
        e.count = inc_counts.get((e.task, e.event), 0)
    m.mapping = [e for e in m.mapping if (e.task, e.event) in seen]

    # recording IDs (unique, derived from the editable identity fields)
    used: Counter = Counter()
    for r in recs:
        rid = make_recording_id(r.participant, r.session, r.run, r.task, r.index)
        used[rid] += 1
        r.recording_id = rid if used[rid] == 1 else f"{rid}_{used[rid]}"
    dup = [k for k, v in used.items() if v > 1]
    if dup:
        issues.append(Issue(level="info", code="duplicate_ids",
                            message=f"{len(dup)} recording ID(s) occur more than once (e.g. {dup[0]}); a suffix keeps "
                                    "them unique. Check the run / session fields."))

    # per-recording trials
    class_counts: Counter = Counter()
    class_seconds: Counter = Counter()
    for r in recs:
        r.issues = [i for i in r.issues if i.code.startswith("scan_")]
        r.n_trials, r.class_counts = 0, {}
        sf = r.sfreq or m.array_options.sfreq
        if not r.readable:
            continue
        if r.structure == "epoched":
            counts: Counter = Counter()
            for ev, n in r.event_counts.items():
                e = table.get((r.task, ev))
                if e and e.action == "class" and e.label:
                    counts[e.label] += n
                elif e and e.action in ("segment_start", "segment_end"):
                    r.issues.append(Issue(level="warning", code="segment_on_epochs", recording=r.index,
                                          message=f"'{ev}': start/end segments do not apply to already-epoched "
                                                  "data; these epochs are ignored."))
            r.n_trials, r.class_counts = sum(counts.values()), dict(counts)
        else:
            trials, stats = build_trials(events_in_samples(events, r.index, sf), r.task, table)
            counts = Counter(lbl for _, lbl, _ in trials)
            r.n_trials, r.class_counts = len(trials), dict(counts)
            if sf:
                for _, lbl, dur in trials:
                    class_seconds[lbl] += dur / (r.sfreq or sf) if dur else 0
            if stats["unpaired_starts"]:
                r.issues.append(Issue(level="warning", code="unpaired_start", recording=r.index,
                                      message=f"{stats['unpaired_starts']} segment start marker(s) without a matching "
                                              "end marker were dropped."))
            if stats["orphan_ends"]:
                r.issues.append(Issue(level="info", code="orphan_end", recording=r.index,
                                      message=f"{stats['orphan_ends']} end marker(s) without an open segment."))
            if r.n_samples and trials:
                late = sum(1 for s, _, _ in trials if s >= r.n_samples)
                if late:
                    r.issues.append(Issue(level="warning", code="events_beyond_end", recording=r.index,
                                          message=f"{late} event(s) start after the end of the recording."))
        if r.included:
            class_counts.update(r.class_counts)
            if r.events.kind == "none" or not r.event_counts:
                r.issues.append(Issue(level="error", code="no_events", recording=r.index,
                                      message="No event markers, labels or associated event file: this recording "
                                              "cannot provide trials. Exclude it or add an event file."))
            elif r.n_trials == 0:
                r.issues.append(Issue(level="warning", code="no_trials", recording=r.index,
                                      message="The current event mapping yields no trials for this recording."))
            if r.structure == "continuous" and not (r.sfreq or m.array_options.sfreq):
                r.issues.append(Issue(level="error", code="sfreq_missing", recording=r.index,
                                      message="Sampling rate unknown; enter it under 'Array file options'."))
            if not r.participant and not r.inner_groups:
                r.issues.append(Issue(level="warning", code="no_participant", recording=r.index,
                                      message="No participant ID: participant-level splits will not be possible."))

    # dataset-level checks
    if not inc:
        issues.append(Issue(level="error", code="nothing_included", message="No recordings are included."))
    structures = {r.structure for r in inc}
    if len(structures) > 1:
        issues.append(Issue(level="error", code="mixed_structure",
                            message="Included recordings mix continuous and already-epoched data; exclude one kind."))
    structure = next(iter(structures)) if len(structures) == 1 else None
    sfreqs = sorted({float(r.sfreq or m.array_options.sfreq) for r in inc if (r.sfreq or m.array_options.sfreq)})
    target = None
    if sfreqs:
        target = m.target_sfreq or _mode([float(r.sfreq or m.array_options.sfreq) for r in inc
                                          if (r.sfreq or m.array_options.sfreq)])
        if len(sfreqs) > 1:
            if m.sfreq_policy == "strict" or structure == "epoched":
                issues.append(Issue(level="error", code="sfreq_mismatch",
                                    message=f"Recordings have different sampling rates ({', '.join(f'{s:g}' for s in sfreqs)} "
                                            "Hz). Exclude the odd ones or allow resampling at import."))
            else:
                issues.append(Issue(level="warning", code="sfreq_resample",
                                    message=f"Sampling rates differ ({', '.join(f'{s:g}' for s in sfreqs)} Hz); all "
                                            f"recordings will be resampled to {target:g} Hz at import (polyphase, "
                                            "anti-aliased)."))
        elif m.target_sfreq and abs(m.target_sfreq - sfreqs[0]) > 1e-6:
            issues.append(Issue(level="info", code="sfreq_resample",
                                message=f"Recordings will be resampled from {sfreqs[0]:g} to {target:g} Hz at import."))
    common: list[str] = []
    dropped: list[str] = []
    if inc:
        first = inc[0]
        names = [set(r.ch_names) for r in inc if r.ch_names]
        inter = set.intersection(*names) if names else set()
        common = [c for c in first.ch_names if c in inter]
        union = set().union(*names) if names else set()
        dropped = sorted(union - inter)
        if not common:
            issues.append(Issue(level="error", code="no_common_channels",
                                message="The included recordings share no channel names."))
        elif dropped:
            if m.channel_policy == "strict":
                issues.append(Issue(level="error", code="channel_mismatch",
                                    message=f"Channel sets differ ({len(dropped)} channel(s) not in every recording: "
                                            f"{', '.join(dropped[:10])}). Exclude recordings or allow the common "
                                            "channel set."))
            else:
                issues.append(Issue(level="warning", code="channels_dropped",
                                    message=f"{len(dropped)} channel(s) are not present in every recording and will be "
                                            f"dropped: {', '.join(dropped[:12])}{'…' if len(dropped) > 12 else ''}."))
        reordered = sum(1 for r in inc if [c for c in r.ch_names if c in inter] != common)
        if reordered:
            issues.append(Issue(level="info", code="channels_reordered",
                                message=f"{reordered} recording(s) list channels in a different order; they are "
                                        "re-ordered by name at import."))
    if structure == "epoched":
        lens = {(r.epoch_samples, round(r.epoch_tmin or 0.0, 6)) for r in inc}
        if len(lens) > 1:
            issues.append(Issue(level="error", code="epoch_mismatch",
                                message="Epoched files differ in epoch length or start time."))
    classes = sorted(class_counts)
    if inc and len(classes) < 2:
        issues.append(Issue(level="error", code="too_few_classes",
                            message=f"The mapping yields {len(classes)} class(es); at least two are needed. Map event "
                                    "descriptions to class labels (or start/end segments) in the event mapping."))
    small = [f"{c} ({n})" for c, n in class_counts.items() if n < 2]
    if small:
        issues.append(Issue(level="warning", code="tiny_classes",
                            message=f"Classes with fewer than 2 trials cannot be split or stratified: "
                                    f"{', '.join(small)}."))
    if len(class_counts) > 20:
        issues.append(Issue(level="warning", code="many_classes",
                            message=f"{len(class_counts)} classes: event descriptions may need grouping (e.g. "
                                    "'FC-1 start' … 'FC-16 start' → one class 'FC')."))
    orphans = [f.path for f in m.files if f.role == "events" and f.recording is None]
    if orphans:
        issues.append(Issue(level="warning", code="orphan_event_files",
                            message=f"{len(orphans)} event file(s) could not be matched to a recording (e.g. "
                                    f"{orphans[0]}); name them like their recording (…_events.tsv)."))
    ignored = [f for f in m.files if f.role == "ignored"]
    if ignored:
        issues.append(Issue(level="info", code="ignored_files",
                            message=f"{len(ignored)} file(s) were not used (unsupported or non-EEG files)."))

    # participants table
    parts: dict[str, dict] = {}
    for r in inc:
        groups = r.inner_groups or ({r.participant: r.n_trials} if r.participant else {"(none)": r.n_trials})
        for p, n in groups.items():
            row = parts.setdefault(p, {"participant": None if p == "(none)" else p, "n_recordings": 0,
                                       "sessions": set(), "runs": set(), "n_trials": 0, "class_counts": Counter()})
            row["n_recordings"] += 1
            if r.session:
                row["sessions"].add(r.session)
            if r.run:
                row["runs"].add(r.run)
            row["n_trials"] += n if r.inner_groups else r.n_trials
            if not r.inner_groups:
                row["class_counts"].update(r.class_counts)
    attrs = (m.metadata or {}).get("participants", {})
    participants = [{"participant": v["participant"], "n_recordings": v["n_recordings"],
                     "sessions": sorted(v["sessions"]), "runs": sorted(v["runs"]), "n_trials": v["n_trials"],
                     "class_counts": dict(v["class_counts"]), "attributes": attrs.get(v["participant"] or "", {})}
                    for v in sorted(parts.values(), key=lambda x: str(x["participant"]))]
    ch_types = [infer_channel_type(c, t) for c, t in zip(inc[0].ch_names, inc[0].ch_types or ["eeg"] * len(inc[0].ch_names))
                ] if inc else []
    type_of = dict(zip(inc[0].ch_names, ch_types)) if inc else {}
    m.summary = {
        "n_files": len(m.files),
        "n_recordings": len(recs),
        "n_included": len(inc),
        "n_unreadable": sum(1 for r in recs if not r.readable),
        "structure": structure,
        "n_participants": len([p for p in participants if p["participant"]]),
        "participants": participants,
        "n_trials": sum(class_counts.values()),
        "classes": classes,
        "class_counts": dict(class_counts),
        "class_seconds": {k: round(v, 2) for k, v in class_seconds.items()} if class_seconds else None,
        "sfreqs": sfreqs,
        "target_sfreq": target,
        "common_channels": common,
        "common_channel_types": [type_of.get(c, "eeg") for c in common],
        "dropped_channels": dropped,
        "total_duration_s": round(sum(r.duration_s or 0 for r in inc), 1),
        "sessions": sorted({r.session for r in inc if r.session}),
        "runs": sorted({r.run for r in inc if r.run}),
        "tasks": sorted({r.task for r in inc if r.task}),
    }
    rec_errors = [i for r in inc for i in r.issues if i.level == "error"]
    m.issues = issues
    m.importable = not any(i.level == "error" for i in issues) and not rec_errors
    return m


def manifest_from_dataset(ds, name: str, source: dict, files: list[tuple[str, int]] | None = None,
                          fmt: str = "upload") -> DatasetManifest:
    """Describe an already-loaded dataset (API upload, synthetic generator) with the same manifest model."""
    from .store import ContinuousData, EpochedData, _is_bad

    recs: list[ScannedRecording] = []
    cache: EventsCache = {}
    ch_types = [infer_channel_type(n, t) for n, t in zip(ds.ch_names, ds.ch_types or ["eeg"] * len(ds.ch_names))]
    common = dict(sfreq=ds.sfreq, n_channels=len(ds.ch_names), ch_names=list(ds.ch_names), ch_types=ch_types,
                  unit=ds.unit, id_source="file contents")
    if isinstance(ds, ContinuousData):
        for i, r in enumerate(ds.recordings):
            inv = {v: k for k, v in r.event_id.items()}
            evs = [(float(s), inv.get(int(c), str(int(c))), float(d)) for s, c, d in r.events]
            evs = [e for e in evs if not _is_bad(e[1])]
            kind = r.event_source if r.event_source in ("annotations", "stim channel", "embedded") else (
                "embedded" if evs else "none")
            recs.append(ScannedRecording(
                index=i, file=r.source_file or r.name, format=fmt, structure="continuous", participant=r.group,
                session=r.session, run=r.run, task=r.task, n_samples=int(r.data.shape[1]),
                duration_s=r.data.shape[1] / ds.sfreq, events=EventSourceInfo(kind=kind, n_events=len(evs)),
                event_counts=dict(Counter(d for _, d, _ in evs)), **common))
            cache[i] = {"unit": "samples", "events": evs}
    else:
        assert isinstance(ds, EpochedData)
        rec_ids = [str(v) for v in ds.recording_ids] if ds.recording_ids is not None else [""] * len(ds.X)
        order = list(dict.fromkeys(rec_ids))
        for i, rid in enumerate(order):
            sel = [j for j, v in enumerate(rec_ids) if v == rid]
            groups = Counter(str(ds.groups[j]) for j in sel if ds.groups is not None and str(ds.groups[j]))
            sess = {str(ds.sessions[j]) for j in sel if ds.sessions is not None and str(ds.sessions[j])}
            runs = {str(ds.runs[j]) for j in sel if ds.runs is not None and str(ds.runs[j])}
            recs.append(ScannedRecording(
                index=i, file=(ds.source_files or {}).get(rid) or rid or name, format=fmt, structure="epoched",
                participant=next(iter(groups)) if len(groups) == 1 else None,
                inner_groups=dict(groups) if len(groups) > 1 else None,
                session=next(iter(sess)) if len(sess) == 1 else None, run=next(iter(runs)) if len(runs) == 1 else None,
                n_epochs=len(sel), epoch_samples=int(ds.X.shape[2]), epoch_tmin=float(ds.tmin),
                events=EventSourceInfo(kind="labels", n_events=len(sel)),
                event_counts=dict(Counter(str(ds.labels[j]) for j in sel)), **common))
            cache[i] = {"unit": "samples", "events": []}
    pairs: Counter = Counter()
    for r in recs:
        for ev, n in r.event_counts.items():
            pairs[(r.task, ev)] += n
    mapping = [MappingEntry(task=t, event=e, action="class", label=e, count=n) for (t, e), n in sorted(
        pairs.items(), key=lambda kv: (kv[0][0] or "", kv[0][1]))]
    m = DatasetManifest(name=name, source=source, recordings=recs, mapping=mapping,
                        files=[FileEntry(path=p, size=s, role="eeg") for p, s in (files or [])])
    return validate_manifest(m, cache)
