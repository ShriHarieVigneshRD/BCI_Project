"""Infer participant / session / run / task identifiers from dataset paths.

Patterns are tried on the file name first, then on parent folders (nearest first).
Supported conventions include BIDS (``sub-01/ses-01/eeg/sub-01_ses-01_task-mi_run-02_eeg.edf``),
PhysioNet (``S001R04.edf``), BCI Competition IV (``A01T.gdf``), and loose forms such as
``subject_3``, ``P12``, ``session2`` and ``run-1``. Header fields that may contain
personal information (patient names, birth dates) are never used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

SEP = r"(?:^|[^A-Za-z0-9])"
END = r"(?=$|[^A-Za-z0-9])"

PATTERNS: dict[str, list[re.Pattern]] = {
    "participant": [
        re.compile(SEP + r"sub-([A-Za-z0-9]+)" + END, re.I),
        re.compile(SEP + r"(?:subject|subj|sub|participant|part|patient|pat)[-_ ]?(\d{1,5})" + END, re.I),
        re.compile(r"(?:^|[^A-Za-z0-9])S(\d{1,4})(?=R\d)"),                     # PhysioNet S001R04
        re.compile(SEP + r"[Ss](\d{1,4})" + END),                                # S01, s3
        re.compile(SEP + r"[Pp](?!300" + END + r")(\d{1,4})" + END),             # P12 (not the P300 ERP)
        re.compile(r"^([A-Z]\d{2})[TE]$"),                                       # BCI IV 2a/2b A01T
    ],
    "session": [
        re.compile(SEP + r"ses-([A-Za-z0-9]+)" + END, re.I),
        re.compile(SEP + r"(?:session|sess|ses|day|visit)[-_ ]?(\d{1,3})" + END, re.I),
        re.compile(r"^[A-Z]\d{2}([TE])$"),
    ],
    "run": [
        re.compile(SEP + r"run-([A-Za-z0-9]+)" + END, re.I),
        re.compile(SEP + r"(?:run|block|rec|recording)[-_ ]?(\d{1,3})" + END, re.I),
        re.compile(r"(?<=\d)R(\d{1,3})" + END),                                  # PhysioNet S001R04
        re.compile(SEP + r"[Rr](\d{1,3})" + END),
    ],
    "task": [
        re.compile(SEP + r"task-([A-Za-z0-9]+)" + END, re.I),
    ],
}


@dataclass
class Entities:
    participant: str | None = None
    session: str | None = None
    run: str | None = None
    task: str | None = None
    participant_source: str | None = None


def _strip_ext(name: str) -> str:
    low = name.lower()
    for ext in (".fif.gz", ".nii.gz"):
        if low.endswith(ext):
            return name[: -len(ext)]
    return str(PurePosixPath(name).with_suffix("")) if "." in name else name


def _match(kind: str, text: str) -> str | None:
    for rx in PATTERNS[kind]:
        m = rx.search(text)
        if m:
            return m.group(1)
    return None


def infer_entities(relpath: str, flat_fallback_to_stem: bool = True) -> Entities:
    """Infer IDs from a relative path such as ``sub-01/ses-02/eeg/sub-01_ses-02_run-3_eeg.edf``."""
    parts = [p for p in relpath.replace("\\", "/").split("/") if p]
    stem = _strip_ext(parts[-1]) if parts else ""
    folders = list(reversed(parts[:-1]))
    ent = Entities()
    for kind in ("participant", "session", "run", "task"):
        for i, text in enumerate([stem, *folders]):
            v = _match(kind, text)
            if v:
                setattr(ent, kind, v)
                if kind == "participant":
                    ent.participant_source = "file name pattern" if i == 0 else "folder name pattern"
                break
    if ent.participant is None:
        if len(parts) >= 2:
            ent.participant = re.sub(r"[^A-Za-z0-9_-]+", "", parts[0])[:32] or None
            ent.participant_source = "top-level folder"
        elif flat_fallback_to_stem and stem:
            ent.participant = re.sub(r"[^A-Za-z0-9_-]+", "", stem)[:32] or None
            ent.participant_source = "file name"
    return ent


EVENT_SUFFIXES = ("_eeg", "_ieeg", "_meg", "_events", "_event", "-eve", "_eve", "_raw", "-raw", "_epo", "-epo",
                  "_markers", "_marker", "_triggers", "_trigger", "_labels", "_label", "_annotations",
                  "_annot", "_channels", "_electrodes", "_coordsystem")


def match_key(relpath: str) -> str:
    """Key shared by a recording and its sidecars: ``sub-01_run-1_eeg.edf`` and ``sub-01_run-1_events.tsv``."""
    name = relpath.replace("\\", "/").split("/")[-1]
    stem = _strip_ext(name).lower()
    changed = True
    while changed:
        changed = False
        for suf in EVENT_SUFFIXES:
            if stem.endswith(suf) and len(stem) > len(suf):
                stem = stem[: -len(suf)]
                changed = True
    return stem.rstrip("_- .")
