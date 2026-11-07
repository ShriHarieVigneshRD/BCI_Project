"""Preprocessing pipeline.

The pipeline is split in two parts:

* **Deterministic stages** (channel selection/mapping, notch + band filtering,
  resampling, epoching, baseline correction, rule-based rejection). They have no
  fitted parameters, so they can be applied to every trial without leakage.
* **Learned stages** (ICA, normalisation). Their parameters are fitted on the
  training subset of each fold only (``LearnedTransform.fit``) and then applied to
  validation/test data of that fold.

Order (continuous data): channels -> notch -> filter -> resample -> epoch -> baseline
-> rejection -> [ICA -> normalisation].
Order (epoched data):    channels -> notch -> filter -> resample -> crop  -> baseline
-> rejection -> [ICA -> normalisation].
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import warnings
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from scipy.signal import resample_poly, welch
from scipy.stats import kurtosis

from ..data.store import ContinuousData, Dataset, EpochedData, candidate_trials

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ChannelStage(BaseModel):
    selected: list[str] | None = None          # None = every channel, dataset order
    bad_channels: list[str] = []               # annotated bad: excluded from the model input
    rename: dict[str, str] = {}                # channel mapping, e.g. {"EEG C3-REF": "C3"}


class FilterStage(BaseModel):
    enabled: bool = True
    kind: Literal["bandpass", "highpass", "lowpass"] = "bandpass"
    method: Literal["fir", "iir"] = "fir"
    l_freq: float | None = Field(4.0, gt=0)
    h_freq: float | None = Field(40.0, gt=0)
    iir_order: int = Field(4, ge=1, le=10)
    phase: Literal["zero", "minimum", "causal"] = "zero"
    notch_enabled: bool = False
    notch_freqs: list[float] = [50.0]


class ArtifactStage(BaseModel):
    reject_bad_segments: bool = True
    amplitude_enabled: bool = False
    amplitude_threshold_uv: float = Field(150.0, gt=0)
    flat_enabled: bool = False
    flat_threshold_uv: float = Field(0.5, gt=0)


class ICAStage(BaseModel):
    enabled: bool = False
    n_components: float | None = Field(None, gt=0)  # >= 1: count; (0, 1): cumulative variance fraction
    method: Literal["fastica", "infomax"] = "fastica"
    random_state: int = 97
    exclude: list[int] = []
    match_threshold: float = Field(0.8, gt=0, le=1)


class ResampleStage(BaseModel):
    enabled: bool = False
    target_sfreq: float = Field(128.0, gt=0)


class EpochStage(BaseModel):
    # events: fixed window [tmin, tmax] around each event onset (or segment start)
    # windows: sliding windows inside each start/end segment (block designs); point events get one window
    mode: Literal["keep", "crop", "events", "windows"] = "keep"
    classes: list[str] | None = None
    tmin: float = 0.0
    tmax: float = 2.0
    window_s: float = Field(2.0, gt=0)
    step_s: float = Field(1.0, gt=0)
    segment_offset_s: float = Field(0.0, ge=0)
    max_windows: int | None = Field(None, ge=1)


class BaselineStage(BaseModel):
    enabled: bool = False
    tmin: float = -0.2
    tmax: float = 0.0


class NormalizationStage(BaseModel):
    method: Literal["none", "channel_zscore", "global_zscore", "robust_channel", "trial_zscore"] = "channel_zscore"


class PreprocessingConfig(BaseModel):
    channels: ChannelStage = ChannelStage()
    filter: FilterStage = FilterStage()
    artifacts: ArtifactStage = ArtifactStage()
    ica: ICAStage = ICAStage()
    resample: ResampleStage = ResampleStage()
    epoching: EpochStage = EpochStage()
    baseline: BaselineStage = BaselineStage()
    normalization: NormalizationStage = NormalizationStage()


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def deterministic_fingerprint(cfg: PreprocessingConfig, dataset_fps: list[str]) -> str:
    return _hash({"ds": dataset_fps, **cfg.model_dump(include={"channels", "filter", "artifacts", "resample",
                                                                "epoching", "baseline"})})


def full_fingerprint(cfg: PreprocessingConfig, dataset_fps: list[str]) -> str:
    return _hash({"ds": dataset_fps, **cfg.model_dump()})


def ica_reference_fingerprint(cfg: PreprocessingConfig, dataset_fps: list[str], split_fp: str) -> str:
    return _hash({"det": deterministic_fingerprint(cfg, dataset_fps), "split": split_fp,
                  "ica": cfg.ica.model_dump(exclude={"exclude", "match_threshold"})})


NORMALIZATION_DOC = {
    "none": "No normalisation.",
    "channel_zscore": "Per-channel z-score; mean and SD fitted on the training subset of each fold.",
    "global_zscore": "Single mean/SD over all channels; fitted on the training subset of each fold.",
    "robust_channel": "Per-channel median / (IQR / 1.349); fitted on the training subset of each fold.",
    "trial_zscore": "Each trial and channel standardised by its own mean/SD (no fitted parameters).",
}


# ---------------------------------------------------------------------------
# Epoch set container
# ---------------------------------------------------------------------------


@dataclass
class EpochSet:
    """Model-ready epochs with full provenance.

    ``trial_ids`` identify rows (a window ID such as ``sub-01_run-02:e0007#w003`` in windows mode);
    ``parents`` are the candidate trials the split operates on, so every window of a block stays in the
    same partition. ``offsets`` give each row's start (s) relative to its event onset.
    """

    X: np.ndarray                 # [n, C, T] float32
    labels: np.ndarray            # [n] str
    trial_ids: np.ndarray         # [n] str
    groups: np.ndarray            # [n] str ('' = unknown)
    sessions: np.ndarray          # [n] str
    sfreq: float
    ch_names: list[str]
    tmin: float
    class_names: list[str]
    parents: np.ndarray | None = None      # [n] str  candidate trial (block) ID
    runs: np.ndarray | None = None         # [n] str
    recordings: np.ndarray | None = None   # [n] str  recording ID
    offsets: np.ndarray | None = None      # [n] float seconds from event onset to the row start

    def __post_init__(self) -> None:
        n = len(self.trial_ids)
        if self.parents is None:
            self.parents = np.array([str(t) for t in self.trial_ids])
        if self.runs is None:
            self.runs = np.array([""] * n)
        if self.recordings is None:
            self.recordings = np.array([""] * n)
        if self.offsets is None:
            self.offsets = np.full(n, self.tmin, dtype=np.float64)

    def rows_for(self, parent_ids) -> list[int]:
        """Rows (epochs / windows) belonging to the given candidate trials, in row order."""
        want = set(map(str, parent_ids))
        return [i for i, p in enumerate(self.parents) if str(p) in want]

    def kept_parents(self) -> set[str]:
        return set(map(str, self.parents))

    @property
    def y(self) -> np.ndarray:
        lut = {c: i for i, c in enumerate(self.class_names)}
        return np.array([lut[str(v)] for v in self.labels], dtype=np.int64)

    @property
    def times(self) -> np.ndarray:
        return self.tmin + np.arange(self.X.shape[2]) / self.sfreq

    def index_of(self) -> dict[str, int]:
        return {str(t): i for i, t in enumerate(self.trial_ids)}

    @staticmethod
    def x_path(path: Path) -> Path:
        return path.with_name(path.stem + ".X.f32")

    def save(self, path: Path) -> None:
        """Metadata in ``path`` (.npz); X as raw float32 next to it, memory-mapped on load."""
        path.parent.mkdir(parents=True, exist_ok=True)
        xp = self.x_path(path)
        if not (isinstance(self.X, np.memmap) and Path(self.X.filename).resolve() == xp.resolve()):
            tmp = xp.with_name(xp.name + ".tmp")
            with open(tmp, "wb") as f:
                for i in range(0, len(self.X), 4096):
                    f.write(np.ascontiguousarray(self.X[i:i + 4096], dtype=np.float32).tobytes())
            os.replace(tmp, xp)
        np.savez(path, X_shape=np.array(self.X.shape, dtype=np.int64), labels=self.labels.astype(str), trial_ids=self.trial_ids.astype(str),
                 groups=self.groups.astype(str), sessions=self.sessions.astype(str),
                 ch_names=np.array(self.ch_names, dtype=str), class_names=np.array(self.class_names, dtype=str),
                 sfreq=np.array(self.sfreq), tmin=np.array(self.tmin), parents=self.parents.astype(str),
                 runs=self.runs.astype(str), recordings=self.recordings.astype(str),
                 offsets=np.asarray(self.offsets, dtype=np.float64))

    @classmethod
    def load(cls, path: Path) -> "EpochSet":
        with np.load(path, allow_pickle=False) as z:
            opt = {k: z[k] for k in ("parents", "runs", "recordings", "offsets") if k in z.files}
            X = z["X"] if "X" in z.files else np.memmap(cls.x_path(path), dtype=np.float32, mode="r",
                                                         shape=tuple(int(v) for v in z["X_shape"]))
            return cls(X=X, labels=z["labels"], trial_ids=z["trial_ids"], groups=z["groups"],
                       sessions=z["sessions"], sfreq=float(z["sfreq"]), ch_names=[str(c) for c in z["ch_names"]],
                       tmin=float(z["tmin"]), class_names=[str(c) for c in z["class_names"]], **opt)


class PreprocessingError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _mne():
    import mne

    mne.set_log_level("ERROR")
    return mne


def validate_config(cfg: PreprocessingConfig, sources: list[tuple[str, Dataset]]) -> dict:
    """Check the configuration against the data; returns {'errors': [...], 'warnings': [...], 'info': [...]}."""
    errors: list[str] = []
    warns: list[str] = []
    info: list[str] = []
    if not sources:
        return {"errors": ["No dataset attached."], "warnings": [], "info": []}
    ds0 = sources[0][1]
    sf = ds0.sfreq
    nyq = sf / 2
    kinds = {ds.kind for _, ds in sources}
    if len(kinds) > 1:
        errors.append("Training and test datasets must both be continuous or both be epoched.")
    for _, ds in sources[1:]:
        if abs(ds.sfreq - sf) > 1e-6 and not cfg.resample.enabled:
            errors.append(f"Datasets have different sampling rates ({sf:g} vs {ds.sfreq:g} Hz); enable resampling.")
    # channels
    sel = cfg.channels.selected if cfg.channels.selected is not None else list(ds0.ch_names)
    for name, ds in sources:
        missing = [c for c in sel if c not in ds.ch_names]
        if missing:
            errors.append(f"{'Test dataset' if name == 'test/' else 'Dataset'} lacks channel(s): {', '.join(missing[:8])}.")
    unknown_bad = [c for c in cfg.channels.bad_channels if c not in sel]
    if unknown_bad:
        warns.append(f"Bad-channel annotation ignored for unselected channel(s): {', '.join(unknown_bad)}.")
    final_ch = [c for c in sel if c not in cfg.channels.bad_channels]
    if not final_ch:
        errors.append("No channels remain after selection and bad-channel exclusion.")
    renamed = [cfg.channels.rename.get(c, c) for c in final_ch]
    if len(set(renamed)) != len(renamed):
        errors.append("Channel mapping produces duplicate names.")
    # filter
    f = cfg.filter
    eff_sf = cfg.resample.target_sfreq if cfg.resample.enabled else sf
    if f.enabled:
        l = f.l_freq if f.kind in ("bandpass", "highpass") else None
        h = f.h_freq if f.kind in ("bandpass", "lowpass") else None
        if f.kind in ("bandpass", "highpass") and l is None:
            errors.append("High-pass cutoff is required.")
        if f.kind in ("bandpass", "lowpass") and h is None:
            errors.append("Low-pass cutoff is required.")
        if l is not None and l >= nyq:
            errors.append(f"High-pass cutoff {l:g} Hz must be below the Nyquist frequency ({nyq:g} Hz).")
        if h is not None and h >= nyq:
            errors.append(f"Low-pass cutoff {h:g} Hz must be below the Nyquist frequency ({nyq:g} Hz at "
                          f"{sf:g} Hz sampling).")
        if l is not None and h is not None and l >= h:
            errors.append(f"High-pass cutoff ({l:g} Hz) must be lower than the low-pass cutoff ({h:g} Hz).")
        if f.method == "fir" and f.phase == "causal":
            errors.append("'causal' phase applies to IIR filters; use 'minimum' phase for a causal FIR filter.")
        if f.method == "iir" and f.phase == "minimum":
            errors.append("'minimum' phase applies to FIR filters; use 'causal' (forward-only) for IIR.")
        if h is not None and cfg.resample.enabled and h >= cfg.resample.target_sfreq / 2:
            warns.append(f"Low-pass cutoff {h:g} Hz is above the post-resampling Nyquist "
                         f"({cfg.resample.target_sfreq / 2:g} Hz); the resampler's anti-aliasing filter will "
                         "remove those frequencies.")
        if ds0.kind == "epoched":
            warns.append("Filtering already-epoched data can introduce edge artefacts; filtering continuous data "
                         "is preferable when available.")
        if f.phase != "zero":
            info.append("Non-zero-phase filtering shifts signal timing (latencies) relative to events.")
    if f.notch_enabled:
        bad_n = [fr for fr in f.notch_freqs if fr <= 0 or fr >= nyq]
        if bad_n:
            errors.append(f"Notch frequencies must be within (0, {nyq:g}) Hz: {bad_n}.")
    # resample
    if cfg.resample.enabled:
        if cfg.resample.target_sfreq > sf:
            warns.append("Target rate is above the original rate (upsampling adds no information).")
        if cfg.resample.target_sfreq < 16:
            errors.append("Target sampling rate must be at least 16 Hz.")
    # epoching
    e = cfg.epoching
    all_labels = sorted({t.label for _, ds in sources for t in candidate_trials(ds)})
    if ds0.kind == "continuous":
        if e.mode not in ("events", "windows"):
            errors.append("Continuous data must be epoched from event markers ('events') or segmented into "
                          "sliding windows ('windows').")
        if e.mode == "windows":
            segs = [t for _, ds in sources for t in candidate_trials(ds) if t.duration_s > 0]
            if not segs:
                warns.append("No segments (events with a duration or start/end blocks) were found; every event "
                             "yields one window starting at the epoch start.")
            elif min(t.duration_s for t in segs) < e.window_s:
                n_short = sum(1 for t in segs if t.duration_s < e.window_s + e.segment_offset_s)
                info.append(f"{n_short} segment(s) are shorter than one window; each yields a single window from "
                            "its start.")
            if round(e.window_s * eff_sf) < 8:
                errors.append("Window is shorter than 8 samples.")
            if e.step_s > e.window_s:
                info.append("Step is longer than the window: parts of each segment are skipped.")
            info.append("Windows from the same segment always stay in the same partition (splits operate on "
                        "segments), so overlapping windows cannot leak between training and test.")
        n_events = sum(len(candidate_trials(ds)) for _, ds in sources)
        if n_events == 0:
            errors.append("Epoching cannot proceed: the dataset has no event markers. Trial boundaries will not be "
                          "invented.")
    else:
        if e.mode == "events":
            errors.append("Rebuilding epochs needs continuous recordings with event markers; this dataset is "
                          "already epoched. Keep or crop the existing epochs.")
        if e.mode == "crop":
            ds_t0 = ds0.tmin
            ds_t1 = ds0.tmin + ds0.X.shape[2] / ds0.sfreq
            if e.tmin < ds_t0 - 1e-9 or e.tmax > ds_t1 + 1e-9:
                errors.append(f"Crop window [{e.tmin:g}, {e.tmax:g}] s lies outside the epochs "
                              f"[{ds_t0:g}, {ds_t1:g}] s.")
    if e.mode in ("events", "crop"):
        if e.tmax <= e.tmin:
            errors.append("Epoch end must be after epoch start.")
        elif round((e.tmax - e.tmin) * eff_sf) < 8:
            errors.append("Epoch window is shorter than 8 samples.")
    if e.classes is not None:
        unknown = [c for c in e.classes if c not in all_labels]
        if unknown:
            errors.append(f"Unknown event/class label(s): {', '.join(unknown)}.")
        if len(set(e.classes) - set(unknown)) < 2:
            errors.append("Select at least two classes.")
    elif len(all_labels) < 2:
        errors.append("At least two event/class labels are needed.")
    # baseline
    b = cfg.baseline
    if b.enabled:
        t0, t1 = _epoch_window(cfg, ds0)
        if b.tmin >= b.tmax:
            errors.append("Baseline start must be before baseline end.")
        elif b.tmin < t0 - 1e-9 or b.tmax > t1 + 1e-9:
            errors.append(f"Baseline [{b.tmin:g}, {b.tmax:g}] s must lie within the epoch [{t0:g}, {t1:g}] s.")
    # ICA
    ica = cfg.ica
    if ica.enabled:
        n_ch = len(final_ch)
        if ica.n_components is not None and ica.n_components >= 1 and int(ica.n_components) > n_ch:
            errors.append(f"ICA components ({int(ica.n_components)}) exceed the number of channels ({n_ch}).")
        if ica.n_components is not None and ica.n_components >= 1 and ica.n_components != int(ica.n_components):
            errors.append("ICA component count must be an integer (or a fraction between 0 and 1).")
        if n_ch < 2:
            errors.append("ICA needs at least two channels.")
        if not f.enabled or f.kind == "lowpass" or (f.l_freq or 0) < 1:
            warns.append("ICA is sensitive to slow drifts; a high-pass of at least 1 Hz is recommended before ICA.")
    return {"errors": errors, "warnings": warns, "info": info}


def _epoch_window(cfg: PreprocessingConfig, ds: Dataset) -> tuple[float, float]:
    e = cfg.epoching
    if ds.kind == "continuous" and e.mode == "windows":
        return 0.0, e.window_s
    if ds.kind == "continuous" or e.mode == "crop":
        return e.tmin, e.tmax
    return ds.tmin, ds.tmin + ds.X.shape[2] / ds.sfreq


# ---------------------------------------------------------------------------
# Deterministic stages
# ---------------------------------------------------------------------------


def apply_filters(x: np.ndarray, sfreq: float, f: FilterStage, log: list[str]) -> np.ndarray:
    """Notch + band filtering along the last axis using MNE's filter implementations."""
    mne = _mne()
    if x.ndim == 2 and x.size > 20_000_000:
        # long multi-channel recordings: filter blocks of channels to bound peak memory (identical result)
        step = max(1, int(20_000_000 // x.shape[1]))
        return np.concatenate([apply_filters(np.asarray(x[i:i + step]), sfreq, f, log)
                               for i in range(0, x.shape[0], step)])
    out = np.asarray(x, dtype=np.float64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if f.notch_enabled and f.notch_freqs:
            freqs = [fr for fr in f.notch_freqs if 0 < fr < sfreq / 2]
            if freqs:
                out = mne.filter.notch_filter(out, sfreq, freqs, method="fir" if f.method == "fir" else "iir",
                                              verbose=False)
        if f.enabled:
            l = f.l_freq if f.kind in ("bandpass", "highpass") else None
            h = f.h_freq if f.kind in ("bandpass", "lowpass") else None
            if f.method == "fir":
                out = mne.filter.filter_data(out, sfreq, l, h, method="fir", fir_design="firwin",
                                             phase="zero" if f.phase == "zero" else "minimum", verbose=False)
            else:
                out = mne.filter.filter_data(out, sfreq, l, h, method="iir",
                                             iir_params={"order": f.iir_order, "ftype": "butter", "output": "sos"},
                                             phase="zero" if f.phase == "zero" else "forward", verbose=False)
    for w in caught:
        msg = str(w.message).split("\n")[0]
        if msg not in log:
            log.append(msg)
    return out.astype(np.float32)


def _resample(x: np.ndarray, sfreq: float, target: float) -> tuple[np.ndarray, int, int]:
    frac = Fraction(target / sfreq).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    if up == down:
        return x, 1, 1
    return resample_poly(x, up, down, axis=-1).astype(np.float32), up, down


def filter_description(cfg: PreprocessingConfig) -> str:
    f = cfg.filter
    parts = []
    if f.notch_enabled and f.notch_freqs:
        parts.append(f"notch at {', '.join(f'{v:g}' for v in f.notch_freqs)} Hz")
    if f.enabled:
        band = {"bandpass": f"{f.l_freq:g}-{f.h_freq:g} Hz band-pass", "highpass": f"{f.l_freq:g} Hz high-pass",
                "lowpass": f"{f.h_freq:g} Hz low-pass"}[f.kind]
        if f.method == "fir":
            parts.append(f"{band} (FIR, firwin, {'zero' if f.phase == 'zero' else 'minimum'}-phase)")
        else:
            parts.append(f"{band} (IIR Butterworth order {f.iir_order}, "
                         f"{'zero-phase forward-backward' if f.phase == 'zero' else 'causal forward-only'})")
    return "; ".join(parts) or "no filtering"


def preview_subset(sources: list[tuple[str, Dataset]], max_recordings: int) -> dict[int, set[int]] | None:
    """Recordings used by the live preview of a large continuous dataset: round-robin over participants
    (first recording of every participant, then the second, ...) up to ``max_recordings``. None = use everything."""
    recs = [(si, r, rec.group or "") for si, (_p, ds) in enumerate(sources) if isinstance(ds, ContinuousData)
            for r, rec in enumerate(ds.recordings)]
    if len(recs) <= max_recordings:
        return None
    by_p: dict[str, list[tuple[int, int]]] = {}
    for si, r, g in recs:
        by_p.setdefault(g, []).append((si, r))
    chosen: dict[int, set[int]] = {}
    k, n = 0, 0
    while n < max_recordings:
        added = False
        for lst in by_p.values():
            if k < len(lst) and n < max_recordings:
                si, r = lst[k]
                chosen.setdefault(si, set()).add(r)
                n, added = n + 1, True
        if not added:
            break
        k += 1
    return chosen


class _EpochSink:
    """Collects epoch blocks in memory, or streams them to a raw float32 file when ``path`` is given."""

    def __init__(self, path: Path | None):
        self.path = path
        self.tmp = path.with_name(path.name + ".tmp") if path else None
        self.f = open(self.tmp, "wb") if self.tmp else None  # noqa: SIM115 (closed in finish/abort)
        self.parts: list[np.ndarray] = []
        self.shape: tuple[int, int] | None = None
        self.n = 0

    def add(self, X: np.ndarray) -> None:
        if self.shape is None:
            self.shape = X.shape[1:]
        elif X.shape[1:] != self.shape:
            raise PreprocessingError(f"Epoch lengths differ between datasets ({sorted({self.shape[1], X.shape[2]})} "
                                     "samples).")
        if not len(X):
            return
        if self.f:
            self.f.write(np.ascontiguousarray(X, dtype=np.float32).tobytes())
        else:
            self.parts.append(X)
        self.n += len(X)

    def finish(self) -> np.ndarray:
        if not self.f:
            return np.concatenate(self.parts) if self.parts else np.zeros((0, 1, 1), np.float32)
        self.f.close()
        os.replace(self.tmp, self.path)
        return np.memmap(self.path, dtype=np.float32, mode="r", shape=(self.n, *self.shape))

    def abort(self) -> None:
        if self.f:
            self.f.close()
            self.tmp.unlink(missing_ok=True)


def run_deterministic(cfg: PreprocessingConfig, sources: list[tuple[str, Dataset]], out: Path | None = None,
                      max_recordings: int | None = None) -> tuple[EpochSet, dict]:
    """Deterministic stages, one recording at a time (bounded memory).

    ``out``: stream the epochs to this raw float32 file and return them memory-mapped (large datasets).
    ``max_recordings``: preview only; process a participant-balanced subset of the recordings."""
    sink = _EpochSink(out)
    try:
        return _run_deterministic(cfg, sources, sink, max_recordings)
    except BaseException:
        sink.abort()
        raise


def _run_deterministic(cfg: PreprocessingConfig, sources: list[tuple[str, Dataset]], sink: _EpochSink,
                       max_recordings: int | None) -> tuple[EpochSet, dict]:
    v = validate_config(cfg, sources)
    if v["errors"]:
        raise PreprocessingError("; ".join(v["errors"]))
    ds0 = sources[0][1]
    sel = cfg.channels.selected if cfg.channels.selected is not None else list(ds0.ch_names)
    final_ch = [c for c in sel if c not in cfg.channels.bad_channels]
    out_names = [cfg.channels.rename.get(c, c) for c in final_ch]
    filter_log: list[str] = []
    classes_sel = set(cfg.epoching.classes) if cfg.epoching.classes is not None else None
    target_sf = cfg.resample.target_sfreq if cfg.resample.enabled else ds0.sfreq

    labels, ids, groups, sessions = [], [], [], []
    parents, runs, recs_out, offsets = [], [], [], []
    subset = preview_subset(sources, max_recordings) if max_recordings else None
    n_recordings_total = sum(len(ds.recordings) for _p, ds in sources if isinstance(ds, ContinuousData))
    windows_mode = ds0.kind == "continuous" and cfg.epoching.mode == "windows"
    n_built = 0
    n_segments = 0
    rejected: list[dict] = []
    counts = {"excluded_class": 0, "out_of_bounds": 0, "bad_segment": 0, "amplitude": 0, "flat": 0}
    before: dict[str, int] = {}
    n_candidates = 0
    resample_info = {"applied": False}
    tmin_out = 0.0
    used_trials: list[list] = []

    def flush(block: list[np.ndarray], meta: list[tuple], t0: float, sf_blk: float) -> None:
        """Baseline + rule-based rejection for one recording's epochs (both are per epoch), then to the sink."""
        nonlocal n_built
        if not block:
            return
        Xb = np.stack(block).astype(np.float32)
        n_built += len(Xb)
        if cfg.baseline.enabled:
            tt = t0 + np.arange(Xb.shape[2]) / sf_blk
            m = (tt >= cfg.baseline.tmin - 1e-9) & (tt <= cfg.baseline.tmax + 1e-9)
            if not m.any():
                raise PreprocessingError("Baseline window contains no samples.")
            Xb -= Xb[:, :, m].mean(axis=2, keepdims=True)
        keep = np.ones(len(Xb), dtype=bool)
        ptp = Xb.max(axis=2) - Xb.min(axis=2)  # [n, C]
        if cfg.artifacts.amplitude_enabled:
            bad = ptp.max(axis=1) > cfg.artifacts.amplitude_threshold_uv
            for i in np.flatnonzero(bad):
                c = int(ptp[i].argmax())
                rejected.append({"trial_id": meta[i][0], "reason": f"peak-to-peak {ptp[i, c]:.0f} uV on "
                                                                   f"{out_names[c]} > "
                                                                   f"{cfg.artifacts.amplitude_threshold_uv:g} uV"})
            counts["amplitude"] += int(bad.sum())
            keep &= ~bad
        if cfg.artifacts.flat_enabled:
            flat = ptp.min(axis=1) < cfg.artifacts.flat_threshold_uv
            for i in np.flatnonzero(flat & keep):
                c = int(ptp[i].argmin())
                rejected.append({"trial_id": meta[i][0], "reason": f"flat signal on {out_names[c]} "
                                                                   f"(peak-to-peak {ptp[i, c]:.2f} uV)"})
            counts["flat"] += int((flat & keep).sum())
            keep &= ~flat
        sink.add(Xb[keep])
        for i in np.flatnonzero(keep):
            rid, lbl, grp, ses, par, run, rec_id, off = meta[i]
            ids.append(rid)
            labels.append(lbl)
            groups.append(grp)
            sessions.append(ses)
            parents.append(par)
            runs.append(run)
            recs_out.append(rec_id)
            offsets.append(off)
        block.clear()
        meta.clear()

    for si, (prefix, ds) in enumerate(sources):
        ch_idx = [ds.ch_names.index(c) for c in final_ch]
        trials = candidate_trials(ds)
        if subset is not None:
            trials = [t for t in trials if t.recording in subset.get(si, set())]
        used_trials.append(trials)
        n_candidates += len(trials)
        for t in trials:
            before[str(t.label)] = before.get(str(t.label), 0) + 1
        if isinstance(ds, ContinuousData):
            e = cfg.epoching
            n_t = int(round((e.window_s if windows_mode else e.tmax - e.tmin) * target_sf))
            tmin_out = 0.0 if windows_mode else e.tmin
            by_rec: dict[int, list] = {}
            for t in trials:
                by_rec.setdefault(t.recording, []).append(t)
            for r, rec in enumerate(ds.recordings):
                rec_trials = by_rec.get(r, [])
                if not rec_trials:
                    continue
                block: list[np.ndarray] = []
                meta: list[tuple] = []
                x = rec.data[ch_idx]
                x = apply_filters(x, ds.sfreq, cfg.filter, filter_log)
                sf = ds.sfreq
                if cfg.resample.enabled:
                    x, up, down = _resample(x, ds.sfreq, target_sf)
                    sf = ds.sfreq * up / down
                    resample_info = {"applied": True, "from": ds.sfreq, "to": sf, "up": up, "down": down,
                                     "method": "polyphase (scipy.signal.resample_poly) with anti-aliasing FIR"}
                for t in rec_trials:
                    tid = prefix + t.trial_id
                    if classes_sel is not None and t.label not in classes_sel:
                        counts["excluded_class"] += 1
                        continue
                    onset = int(round(t.sample * sf / ds.sfreq))
                    if windows_mode:
                        span = t.duration_s - e.segment_offset_s
                        if t.duration_s > 0:
                            n_segments += 1
                        if t.duration_s > 0 and span >= e.window_s:
                            n_win = int(np.floor((span - e.window_s) / e.step_s + 1e-9)) + 1
                            offs = [e.segment_offset_s + k * e.step_s for k in range(n_win)]
                        else:
                            offs = [e.segment_offset_s if t.duration_s > 0 else e.tmin]
                        if e.max_windows:
                            offs = offs[: e.max_windows]
                        rows = [(f"{tid}#w{k:03d}", off) for k, off in enumerate(offs)]
                    else:
                        rows = [(tid, e.tmin)]
                    for rid, off in rows:
                        start = onset + int(round(off * sf))
                        if start < 0 or start + n_t > x.shape[1]:
                            counts["out_of_bounds"] += 1
                            rejected.append({"trial_id": rid, "reason": "epoch window outside the recording"})
                            continue
                        if cfg.artifacts.reject_bad_segments and rec.bad_segments:
                            a0, a1 = start / sf, (start + n_t) / sf
                            if any(on < a1 and on + du > a0 for on, du in rec.bad_segments):
                                counts["bad_segment"] += 1
                                rejected.append({"trial_id": rid, "reason": "overlaps a BAD annotation"})
                                continue
                        block.append(x[:, start:start + n_t])
                        meta.append((rid, t.label, t.group or "", t.session or "", tid, t.run or "",
                                     t.recording_id or rec.recording_id or rec.name, float(off)))
                flush(block, meta, tmin_out, target_sf)
                del x
        else:
            assert isinstance(ds, EpochedData)
            keep = [i for i, t in enumerate(trials) if classes_sel is None or t.label in classes_sel]
            counts["excluded_class"] += len(trials) - len(keep)
            if max_recordings and len(keep) > max_recordings * 50:  # preview: evenly spaced epochs
                keep = keep[:: -(-len(keep) // (max_recordings * 50))]
            # filtering and resampling act on each epoch alone, so chunks of epochs give identical results
            for c0 in range(0, len(keep), 512):
                chunk = keep[c0:c0 + 512]
                x = ds.X[chunk][:, ch_idx]
                x = apply_filters(x, ds.sfreq, cfg.filter, filter_log)
                sf = ds.sfreq
                if cfg.resample.enabled:
                    x, up, down = _resample(x, ds.sfreq, target_sf)
                    sf = ds.sfreq * up / down
                    resample_info = {"applied": True, "from": ds.sfreq, "to": sf, "up": up, "down": down,
                                     "method": "polyphase (scipy.signal.resample_poly) with anti-aliasing FIR"}
                tmin_out = ds.tmin
                if cfg.epoching.mode == "crop":
                    i0 = int(round((cfg.epoching.tmin - ds.tmin) * sf))
                    n_t = int(round((cfg.epoching.tmax - cfg.epoching.tmin) * sf))
                    x = x[:, :, i0:i0 + n_t]
                    tmin_out = ds.tmin + i0 / sf
                flush(list(x), [(prefix + trials[i].trial_id, trials[i].label, trials[i].group or "",
                                 trials[i].session or "", prefix + trials[i].trial_id, trials[i].run or "",
                                 trials[i].recording_id or "", float(tmin_out)) for i in chunk], tmin_out, sf)

    if n_built == 0:
        raise PreprocessingError("No epochs remain after epoching.")
    if sink.n == 0:
        raise PreprocessingError("All epochs were rejected; relax the rejection thresholds.")
    X = sink.finish()
    sf_out = target_sf
    times = tmin_out + np.arange(X.shape[2]) / sf_out
    labels_a = np.array(labels)
    classes = sorted({str(c) for c in labels_a}) if classes_sel is None else sorted(classes_sel)
    es = EpochSet(X=X, labels=labels_a, trial_ids=np.array(ids), groups=np.array(groups),
                  sessions=np.array(sessions), sfreq=float(sf_out), ch_names=out_names, tmin=float(tmin_out),
                  class_names=classes, parents=np.array(parents), runs=np.array(runs), recordings=np.array(recs_out),
                  offsets=np.array(offsets, dtype=np.float64))
    after = {c: int((es.labels == c).sum()) for c in classes}
    # provenance summaries: candidates / kept per participant and per recording
    cand_by_p: dict[str, dict] = {}
    for trials in used_trials:
        for t in trials:
            if classes_sel is not None and t.label not in classes_sel:
                continue
            row = cand_by_p.setdefault(t.group or "(none)", {"candidates": 0, "kept_trials": 0, "kept_rows": 0,
                                                             "class_rows": {}})
            row["candidates"] += 1
    kept_parent_by_p: dict[str, set] = {}
    for g, par, lbl in zip(es.groups, es.parents, es.labels):
        key = str(g) or "(none)"
        row = cand_by_p.setdefault(key, {"candidates": 0, "kept_trials": 0, "kept_rows": 0, "class_rows": {}})
        row["kept_rows"] += 1
        row["class_rows"][str(lbl)] = row["class_rows"].get(str(lbl), 0) + 1
        kept_parent_by_p.setdefault(key, set()).add(str(par))
    for key, row in cand_by_p.items():
        row["kept_trials"] = len(kept_parent_by_p.get(key, set()))
    by_rec: dict[str, dict] = {}
    for g, run, rid, par in zip(es.groups, es.runs, es.recordings, es.parents):
        row = by_rec.setdefault(str(rid), {"recording_id": str(rid), "participant": str(g) or None,
                                           "run": str(run) or None, "rows": 0, "trials": set()})
        row["rows"] += 1
        row["trials"].add(str(par))
    report = {
        "pipeline_order": ["channels", "notch", "filter", "resample",
                           "epoching" if ds0.kind == "continuous" else "crop", "baseline", "rejection",
                           "ica (fit on training data)", "normalization (fit on training data)"],
        "input": {"structure": ds0.kind, "n_channels": len(ds0.ch_names), "sfreq": ds0.sfreq,
                  "n_candidates": n_candidates, "class_distribution": before},
        "channels": {"selected": final_ch, "output_names": out_names, "excluded_bad": cfg.channels.bad_channels,
                     "n": len(final_ch)},
        "filter": {"description": filter_description(cfg), "messages": filter_log[:10]},
        "resample": resample_info,
        "epoching": {"mode": cfg.epoching.mode, "tmin": float(tmin_out), "tmax": float(times[-1] + 1 / sf_out),
                     "n_times": int(X.shape[2]), "excluded_class": counts["excluded_class"],
                     "out_of_bounds": counts["out_of_bounds"],
                     "windows": ({"window_s": cfg.epoching.window_s, "step_s": cfg.epoching.step_s,
                                  "segment_offset_s": cfg.epoching.segment_offset_s,
                                  "max_windows": cfg.epoching.max_windows, "n_segments": n_segments,
                                  "n_parent_trials": len(es.kept_parents()), "n_windows": int(len(X))}
                                 if windows_mode else None)},
        "by_participant": [{"participant": None if k == "(none)" else k, **v}
                           for k, v in sorted(cand_by_p.items())],
        "by_recording": [{**{k: v for k, v in r.items() if k != "trials"}, "trials": len(r["trials"])}
                         for r in list(by_rec.values())[:2000]],
        "baseline": cfg.baseline.model_dump(),
        "scope": {"subset": subset is not None,
                  "recordings": sum(len(v) for v in subset.values()) if subset is not None else n_recordings_total,
                  "of": n_recordings_total, "participants": len(cand_by_p)},
        "rejection": {"bad_segment": counts["bad_segment"], "amplitude": counts["amplitude"],
                      "flat": counts["flat"], "total": len(rejected), "rejected": rejected[:1000]},
        "output": {"n_epochs": int(len(X)), "class_distribution": after, "shape": [int(len(X)), 1, X.shape[1],
                                                                                    X.shape[2]],
                   "sfreq": float(sf_out), "tmin": float(tmin_out), "duration_s": X.shape[2] / sf_out,
                   "classes": classes},
        "normalization": {"method": cfg.normalization.method, "doc": NORMALIZATION_DOC[cfg.normalization.method]},
        "validation": v,
    }
    return es, report


# ---------------------------------------------------------------------------
# Learned stages (fitted on training data only)
# ---------------------------------------------------------------------------


def _epochs_array(X: np.ndarray, ch_names: list[str], sfreq: float):
    mne = _mne()
    info = mne.create_info(list(ch_names), sfreq, "eeg")
    return mne.EpochsArray(X.astype(np.float64) * 1e-6, info, verbose=False)


# Channel x sample values processed at once by ICA and the learned stages (~200 MB as float64; MNE makes a few
# working copies). Keeps memory bounded however many epochs the (memory-mapped) dataset has.
MAX_CHUNK_VALUES = 25_000_000


def iter_rows(X: np.ndarray, rows=None):
    """Yield (start, float32 block) over X[rows] (default: all rows) in chunks of about MAX_CHUNK_VALUES values."""
    n = len(X) if rows is None else len(rows)
    step = max(1, MAX_CHUNK_VALUES // max(1, X.shape[1] * X.shape[2]))
    for i in range(0, n, step):
        blk = X[i:i + step] if rows is None else X[np.asarray(rows[i:i + step])]
        yield i, np.asarray(blk, dtype=np.float32)


def ica_fit_rows(rows: np.ndarray, n_channels: int, n_times: int, seed: int) -> np.ndarray:
    """Trials used to fit ICA: all of them if they fit the memory budget, else a seeded random subset.

    ICA unmixing is a spatial transform estimated from the distribution of samples, so a random subset of trials
    (spread over all participants) estimates it without loading the whole training set as float64."""
    k = max(2, MAX_CHUNK_VALUES // max(1, n_channels * n_times))
    if len(rows) <= k:
        return rows
    return np.sort(np.random.default_rng(seed).choice(rows, k, replace=False))


def fit_ica(X: np.ndarray, ch_names: list[str], sfreq: float, st: ICAStage, rows=None):
    """Fit ICA on X[rows] (default: all trials). Returns (ica, rows actually used)."""
    mne = _mne()
    n_comp = st.n_components
    if n_comp is not None and n_comp >= 1:
        n_comp = int(n_comp)
    used = ica_fit_rows(np.arange(len(X)) if rows is None else np.asarray(rows), X.shape[1], X.shape[2],
                        st.random_state)
    ica = mne.preprocessing.ICA(n_components=n_comp, method=st.method, random_state=st.random_state,
                                max_iter="auto")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica.fit(_epochs_array(X[used], ch_names, sfreq))
    return ica, used


def match_components(ref_ica, ica, exclude: list[int], threshold: float) -> tuple[list[int], list[dict]]:
    """Map reference components (selected by the user) onto a new fold's ICA by spatial-pattern correlation."""
    ref = ref_ica.get_components()
    cur = ica.get_components()
    chosen, matches = [], []
    for k in exclude:
        if k >= ref.shape[1]:
            matches.append({"reference": k, "matched": None, "corr": None, "note": "reference index out of range"})
            continue
        corr = [abs(float(np.corrcoef(ref[:, k], cur[:, j])[0, 1])) for j in range(cur.shape[1])]
        j = int(np.argmax(corr))
        if corr[j] >= threshold and j not in chosen:
            chosen.append(j)
            matches.append({"reference": k, "matched": j, "corr": round(corr[j], 3)})
        else:
            matches.append({"reference": k, "matched": None, "corr": round(corr[j], 3),
                            "note": f"no component above |r| >= {threshold}; not removed in this fold"})
    return chosen, matches


class LearnedTransform:
    """ICA + normalisation, fitted on the training subset of a single fold."""

    def __init__(self, cfg: PreprocessingConfig, ch_names: list[str], sfreq: float):
        self.cfg = cfg
        self.ch_names = ch_names
        self.sfreq = sfreq
        self.ica = None
        self.ica_exclude: list[int] = []
        self.ica_matches: list[dict] = []
        self.stats: dict[str, np.ndarray] = {}
        self.fit_report: dict = {}

    def fit(self, X: np.ndarray, reference_ica=None, fit_ids: list[str] | None = None, rows=None) -> dict:
        """Fit on X[rows] (default: all rows), chunk by chunk so a memory-mapped X is never fully loaded."""
        rows = np.arange(len(X)) if rows is None else np.asarray(rows)
        rep: dict = {"n_fit_trials": int(len(rows)), "fit_scope": "training subset of this fold only"}
        if fit_ids is not None:
            rep["fit_trial_ids_sha"] = hashlib.sha256("|".join(sorted(fit_ids)).encode()).hexdigest()[:16]
        st = self.cfg.ica
        if st.enabled and st.exclude:
            self.ica, used = fit_ica(X, self.ch_names, self.sfreq, st, rows=rows)
            if reference_ica is not None:
                self.ica_exclude, self.ica_matches = match_components(reference_ica, self.ica, st.exclude,
                                                                      st.match_threshold)
            else:
                self.ica_exclude = list(st.exclude)
            rep["ica"] = {"n_components": int(self.ica.n_components_), "excluded": self.ica_exclude,
                          "matches": self.ica_matches, "n_fit_trials": int(len(used))}
        elif st.enabled:
            rep["ica"] = {"note": "ICA enabled but no components selected for removal; ICA is a no-op."}
        m = self.cfg.normalization.method
        if m in ("channel_zscore", "global_zscore"):
            s1 = np.zeros(X.shape[1])
            s2 = np.zeros(X.shape[1])
            for _i, blk in iter_rows(X, rows):
                b = self._apply_ica(blk).astype(np.float64)
                s1 += b.sum(axis=(0, 2))
                s2 += (b * b).sum(axis=(0, 2))
            n = len(rows) * X.shape[2]
            if m == "channel_zscore":
                mean = s1 / n
                self.stats = {"mean": mean, "scale": np.sqrt(np.maximum(s2 / n - mean ** 2, 0)) + 1e-8}
            else:
                mean = s1.sum() / (n * X.shape[1])
                sd = np.sqrt(max(s2.sum() / (n * X.shape[1]) - mean ** 2, 0.0))
                self.stats = {"mean": np.array([mean]), "scale": np.array([sd + 1e-8])}
        elif m == "robust_channel":
            # shortcut: quantiles need every value at once, so large sets use the same seeded trial subset
            # as the ICA fit (exact when the training set fits MAX_CHUNK_VALUES); upgrade to a streaming
            # quantile sketch if exact robust statistics on huge sets are needed.
            sub = self._apply_ica(np.asarray(X[ica_fit_rows(rows, X.shape[1], X.shape[2], 0)], np.float32))
            med = np.median(sub, axis=(0, 2))
            q75, q25 = np.percentile(sub, [75, 25], axis=(0, 2))
            self.stats = {"mean": med, "scale": (q75 - q25) / 1.349 + 1e-8}
            rep["normalization_fit_trials"] = int(len(sub))
        rep["normalization"] = {"method": m, "doc": NORMALIZATION_DOC[m],
                                "stats": {k: [round(float(x), 6) for x in v] for k, v in self.stats.items()}}
        self.fit_report = rep
        return rep

    def _apply_ica(self, X: np.ndarray) -> np.ndarray:
        if self.ica is None or not self.ica_exclude:
            return X
        # in batches of trials: the projection is per sample, so batching gives the same result in bounded memory
        out = np.empty(X.shape, np.float32)
        step = max(1, MAX_CHUNK_VALUES // max(1, X.shape[1] * X.shape[2]))
        for i in range(0, len(X), step):
            ep = _epochs_array(X[i:i + step], self.ch_names, self.sfreq)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.ica.apply(ep, exclude=self.ica_exclude)
            out[i:i + step] = ep.get_data() * 1e6
        return out

    def transform(self, X: np.ndarray, rows=None) -> np.ndarray:
        """Learned stages applied to X[rows] (default: all rows), chunk by chunk into one float32 array."""
        out = np.empty((len(X) if rows is None else len(rows), *X.shape[1:]), np.float32)
        m = self.cfg.normalization.method
        for i, blk in iter_rows(X, rows):
            b = self._apply_ica(blk)
            if m in ("channel_zscore", "robust_channel"):
                b = (b - self.stats["mean"][None, :, None]) / self.stats["scale"][None, :, None]
            elif m == "global_zscore":
                b = (b - self.stats["mean"][0]) / self.stats["scale"][0]
            elif m == "trial_zscore":
                b = (b - b.mean(axis=2, keepdims=True)) / (b.std(axis=2, keepdims=True) + 1e-8)
            out[i:i + len(b)] = b
        return out

    def save(self, directory: Path) -> list[str]:
        directory.mkdir(parents=True, exist_ok=True)
        files = []
        if self.stats:
            np.savez(directory / "normalization.npz", **self.stats)
            files.append("normalization.npz")
        if self.ica is not None:
            self.ica.save(directory / "transform-ica.fif", overwrite=True)
            files.append("transform-ica.fif")
        (directory / "transform.json").write_text(json.dumps({
            "config": self.cfg.model_dump(), "ch_names": self.ch_names, "sfreq": self.sfreq,
            "ica_exclude": self.ica_exclude, "ica_matches": self.ica_matches, "fit_report": self.fit_report}))
        files.append("transform.json")
        return files

    @classmethod
    def load(cls, directory: Path) -> "LearnedTransform":
        meta = json.loads((directory / "transform.json").read_text())
        obj = cls(PreprocessingConfig(**meta["config"]), meta["ch_names"], meta["sfreq"])
        obj.ica_exclude = meta["ica_exclude"]
        obj.ica_matches = meta["ica_matches"]
        obj.fit_report = meta["fit_report"]
        if (directory / "normalization.npz").exists():
            with np.load(directory / "normalization.npz") as z:
                obj.stats = {k: z[k] for k in z.files}
        if (directory / "transform-ica.fif").exists():
            obj.ica = _mne().preprocessing.read_ica(directory / "transform-ica.fif")
        return obj


# ---------------------------------------------------------------------------
# Diagnostics and previews
# ---------------------------------------------------------------------------

FRONTAL = re.compile(r"^(fp|af|f7|f8|eog|veog|heog)", re.I)


def ica_diagnostics(ica, X: np.ndarray, ch_names: list[str], sfreq: float) -> dict:
    ep = _epochs_array(X, ch_names, sfreq)
    src = ica.get_sources(ep).get_data()  # [n, K, T]
    K = src.shape[1]
    flat_src = src.transpose(1, 0, 2).reshape(K, -1)
    frontal = [i for i, c in enumerate(ch_names) if FRONTAL.match(c)]
    flat_x = X.transpose(1, 0, 2).reshape(len(ch_names), -1)
    nper = min(256, X.shape[2])
    freqs, psd = welch(src, fs=sfreq, nperseg=nper, axis=-1)
    psd_m = psd.mean(axis=0)  # [K, F]
    band = (freqs >= 1)
    patterns = ica.get_components()
    comps = []
    for k in range(K):
        try:
            evr = float(ica.get_explained_variance_ratio(ep, components=[k], ch_type="eeg")["eeg"])
        except Exception:
            evr = None
        tot = psd_m[k, band].sum() + 1e-30
        hf = float(psd_m[k, freqs >= 20].sum() / tot)
        lf = float(psd_m[k, (freqs >= 1) & (freqs < 4)].sum() / tot)
        fcorr = None
        if frontal:
            fcorr = max(abs(float(np.corrcoef(flat_src[k], flat_x[c])[0, 1])) for c in frontal)
        comps.append({
            "index": k, "explained_variance": evr, "kurtosis": float(kurtosis(flat_src[k])),
            "high_freq_ratio": hf, "low_freq_ratio": lf, "frontal_corr": fcorr,
            "pattern": [float(v) for v in patterns[:, k]],
            "psd_db": [float(v) for v in 10 * np.log10(psd_m[k, freqs <= min(60, sfreq / 2)] + 1e-30)],
        })
    return {
        "n_components": K, "ch_names": ch_names, "freqs": [float(f) for f in freqs[freqs <= min(60, sfreq / 2)]],
        "components": comps, "method": ica.method,
        "notes": [
            "Explained variance alone does not identify artefacts: large components can be brain activity and small "
            "ones can be artefacts. Inspect the spatial pattern, time course and spectrum together.",
            "High frontal correlation with a 1-4 Hz dominated spectrum suggests eye blinks; a flat, high-frequency "
            "dominated spectrum suggests muscle activity.",
            "Fitted on the training partition only. Components selected here are matched to each fold's own ICA "
            "by spatial-pattern correlation during training.",
        ],
    }


def ica_sources(ica, x_trial: np.ndarray, ch_names: list[str], sfreq: float) -> np.ndarray:
    ep = _epochs_array(x_trial[None], ch_names, sfreq)
    return ica.get_sources(ep).get_data()[0]


def mean_psd(X: np.ndarray, sfreq: float, fmax: float | None = None) -> tuple[list[float], list[float]]:
    if len(X) == 0:
        return [], []
    nper = min(256, X.shape[-1])
    f, p = welch(X, fs=sfreq, nperseg=nper, axis=-1)
    p = p.reshape(-1, p.shape[-1]).mean(axis=0)
    m = f <= (fmax or sfreq / 2)
    return [float(v) for v in f[m]], [float(v) for v in 10 * np.log10(p[m] + 1e-30)]
