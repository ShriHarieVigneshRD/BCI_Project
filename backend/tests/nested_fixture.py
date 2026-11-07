"""Builders for nested, multi-participant, multi-run test datasets (EXPLICITLY SYNTHETIC signals).

* ``build_bids_like``: ``sub-XX/ses-01/eeg/sub-XX_ses-01_task-mi_run-0R_eeg.edf`` recordings *without*
  embedded annotations, each with an associated ``_events.tsv`` (onset / duration / trial_type), BIDS
  ``participants.tsv``, ``_channels.tsv`` (one channel marked bad), an ``_eeg.json`` sidecar, an orphan
  events file without a recording, a resting recording without any events, and unrelated files.
* ``build_block_design``: ``SXX/SXX_RYY.edf`` recordings whose EDF+ annotations delimit blocks with
  start/end markers (``FC-1 start`` / ``FC-1 end``, ``V1 start`` / ``V1 end``, ``NSR-start`` / ``NSR end``)
  plus timestamp notes, mirroring real clinical/lab exports.
"""

from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np

from app.data.synthetic import generate_recordings

mne.set_log_level("ERROR")


def _export_edf(path: Path, data_uv: np.ndarray, ch_names: list[str], sfreq: float,
                annotations: mne.Annotations | None = None) -> None:
    info = mne.create_info(ch_names, sfreq, "eeg")
    raw = mne.io.RawArray(data_uv.astype(np.float64) * 1e-6, info, verbose=False)
    if annotations is not None:
        raw.set_annotations(annotations)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.export(path, fmt="edf", overwrite=True, verbose=False)


def build_bids_like(root: Path, n_participants: int = 4, n_runs: int = 2, trials_per_class: int = 8,
                    seed: int = 21) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    cont = generate_recordings(n_participants=n_participants, n_classes=2, trials_per_class=trials_per_class * n_runs,
                               n_channels=8, include_frontal=True, sfreq=128.0, n_sessions=n_runs, seed=seed)
    (root / "dataset_description.json").write_text(json.dumps({"Name": "Nested MI test (SYNTHETIC)",
                                                               "BIDSVersion": "1.8.0"}))
    (root / "README").write_text("Synthetic nested dataset for software tests.")
    rows = ["participant_id\tage\tsex"] + [f"sub-{p + 1:02d}\t{20 + p}\t{'F' if p % 2 else 'M'}"
                                          for p in range(n_participants)]
    (root / "participants.tsv").write_text("\n".join(rows) + "\n")
    expected = {"recordings": 0, "events": 0, "participants": n_participants, "per_participant": {}}
    for i, rec in enumerate(cont.recordings):
        p, r = i // n_runs + 1, i % n_runs + 1
        d = root / f"sub-{p:02d}" / "ses-01" / "eeg"
        stem = f"sub-{p:02d}_ses-01_task-mi_run-{r:02d}"
        _export_edf(d / f"{stem}_eeg.edf", rec.data, cont.ch_names, cont.sfreq)
        inv = {v: k for k, v in rec.event_id.items()}
        lines = ["onset\tduration\ttrial_type\tvalue"]
        for s, c, _ in rec.events:
            lines.append(f"{s / cont.sfreq:.4f}\t2.0\t{inv[int(c)]}\t{int(c)}")
        (d / f"{stem}_events.tsv").write_text("\n".join(lines) + "\n")
        status = ["good"] * len(cont.ch_names)
        if i == 0:
            status[3] = "bad"
        ch = ["name\ttype\tunits\tstatus"] + [f"{n}\tEEG\tuV\t{st}" for n, st in zip(cont.ch_names, status)]
        (d / f"{stem}_channels.tsv").write_text("\n".join(ch) + "\n")
        (d / f"sub-{p:02d}_ses-01_task-mi_eeg.json").write_text(json.dumps({"SamplingFrequency": 128,
                                                                            "PowerLineFrequency": 50}))
        expected["recordings"] += 1
        expected["events"] += len(rec.events)
        expected["per_participant"][f"{p:02d}"] = expected["per_participant"].get(f"{p:02d}", 0) + len(rec.events)
    # resting recording with no events at all (must be excluded, not counted as a trial)
    rest = root / f"sub-{n_participants:02d}" / "ses-01" / "eeg" / f"sub-{n_participants:02d}_ses-01_task-rest_eeg.edf"
    _export_edf(rest, cont.recordings[0].data[:, : 128 * 30], cont.ch_names, cont.sfreq)
    # orphan event file and unrelated files
    (root / f"sub-{n_participants:02d}" / "ses-01" / "eeg" /
     f"sub-{n_participants:02d}_ses-01_task-mi_run-09_events.tsv").write_text("onset\tduration\ttrial_type\n1.0\t0\tx\n")
    (root / "stimuli").mkdir(exist_ok=True)
    (root / "stimuli" / "arrow.png").write_bytes(b"\x89PNG\r\n")
    (root / ".DS_Store").write_bytes(b"\x00")
    expected["classes"] = sorted(cont.recordings[0].event_id)
    expected["ch_names"] = cont.ch_names
    return expected


def build_block_design(root: Path, n_participants: int = 3, n_runs: int = 2, n_blocks: int = 4,
                       seed: int = 5) -> dict:
    """Recordings with NSR (rest), FC (5 s) and V (40 s) blocks delimited by start/end annotations."""
    rng = np.random.default_rng(seed)
    sf = 128.0
    ch = ["Fp1", "Fp2", "F3", "F4", "C3", "Cz", "C4", "P3", "P4", "O1", "O2", "ECG1", "EMG1"]
    expected = {"recordings": 0, "blocks": {"FC": 0, "V": 0, "NSR": 0}}
    for p in range(1, n_participants + 1):
        for r in range(1, n_runs + 1):
            onsets, descs = [0.0, 0.0, 0.0], ["Recording starts", "Date: 2022-5-25  TimeStart- 11:37:7",
                                              "2022_05_25_11_30_58"]
            t = 5.0
            onsets += [t, t + 20.0]
            descs += ["NSR-start", "NSR end"]
            expected["blocks"]["NSR"] += 1
            t += 25.0
            segments = []
            for b in range(1, n_blocks + 1):
                for name, dur in (("FC-", 5.0), ("V", 40.0)):
                    label = f"{name}{b}"
                    onsets += [t, t + dur]
                    descs += [f"{label} start", f"{label} end"]
                    segments.append((t, dur, name.rstrip("-")))
                    expected["blocks"][name.rstrip("-")] += 1
                    t += dur + 3.0
            n = int((t + 5) * sf)
            x = rng.standard_normal((len(ch), n)) * 8.0
            tt = np.arange(n) / sf
            for s0, dur, kind in segments:  # class-specific rhythm inside blocks
                m = (tt >= s0) & (tt < s0 + dur)
                f = 10.0 if kind == "V" else 20.0
                x[4:7, m] += 12.0 * np.sin(2 * np.pi * f * tt[m])
            ann = mne.Annotations(onsets, [0.0] * len(onsets), descs)
            _export_edf(root / f"S{p:02d}" / f"S{p:02d}_R{r:02d}.edf", x, ch, sf, ann)
            expected["recordings"] += 1
    return expected
