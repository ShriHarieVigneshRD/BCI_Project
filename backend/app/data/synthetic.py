"""Explicitly SYNTHETIC motor-imagery-like EEG for software testing and demonstrations.

The generator mixes pink background noise, participant-specific mu (8-13 Hz) and
beta (~20 Hz) rhythms with class-dependent event-related desynchronisation over
C3 / C4 / Cz, line noise, blink artefacts on frontal channels and occasional
amplitude spikes. It exists only to exercise the software. Metrics obtained on it
are NOT evidence of real-world EEG decoding performance.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .store import ContinuousData, EpochedData, Recording

POSITIONS = {
    "Fp1": (-1, 4), "Fp2": (1, 4), "Fz": (0, 2), "FC3": (-2, 1), "FC1": (-1, 1), "FCz": (0, 1), "FC2": (1, 1),
    "FC4": (2, 1), "C5": (-3, 0), "C3": (-2, 0), "C1": (-1, 0), "Cz": (0, 0), "C2": (1, 0), "C4": (2, 0),
    "C6": (3, 0), "CP3": (-2, -1), "CP1": (-1, -1), "CPz": (0, -1), "CP2": (1, -1), "CP4": (2, -1),
    "P1": (-1, -2), "Pz": (0, -2), "P2": (1, -2), "POz": (0, -3),
}
MONTAGES = {
    8: ["FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CP4"],
    16: ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C3", "C1", "Cz", "C2", "C4", "CP3", "CP1", "CPz", "CP2", "CP4"],
    22: ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz",
         "CP2", "CP4", "P1", "Pz", "P2", "POz"],
}
CLASS_NAMES = ["left_hand", "right_hand", "feet", "tongue"]
# class -> (source site whose mu rhythm desynchronises, depth)
ERD_SITE = {"left_hand": "C4", "right_hand": "C3", "feet": "Cz", "tongue": "FCz"}


def _pink(rng: np.random.Generator, n_ch: int, n: int) -> np.ndarray:
    white = rng.standard_normal((n_ch, n))
    spec = np.fft.rfft(white, axis=1)
    f = np.arange(spec.shape[1])
    f[0] = 1
    spec /= np.sqrt(f)
    x = np.fft.irfft(spec, n=n, axis=1)
    return x / x.std(axis=1, keepdims=True)


def _narrowband(rng: np.random.Generator, n: int, sfreq: float, lo: float, hi: float) -> np.ndarray:
    sos = butter(4, [lo, hi], btype="band", fs=sfreq, output="sos")
    x = sosfiltfilt(sos, rng.standard_normal(n))
    return x / (x.std() + 1e-12)


def _spatial(ch_names: list[str], site: str, sigma: float = 0.9) -> np.ndarray:
    sx, sy = POSITIONS[site]
    w = np.array([np.exp(-((POSITIONS[c][0] - sx) ** 2 + (POSITIONS[c][1] - sy) ** 2) / (2 * sigma**2))
                  for c in ch_names])
    return w


def generate_recordings(n_participants: int = 6, n_classes: int = 3, trials_per_class: int = 40,
                        n_channels: int = 16, include_frontal: bool = True, sfreq: float = 128.0,
                        trial_s: float = 2.0, n_sessions: int = 1, seed: int = 7, artifacts: bool = True,
                        line_noise_hz: float = 50.0, effect: float = 0.55) -> ContinuousData:
    if n_channels not in MONTAGES:
        raise ValueError(f"n_channels must be one of {sorted(MONTAGES)}")
    if not 2 <= n_classes <= 4:
        raise ValueError("n_classes must be between 2 and 4")
    rng = np.random.default_rng(seed)
    ch = (["Fp1", "Fp2"] if include_frontal else []) + MONTAGES[n_channels]
    C = len(ch)
    classes = CLASS_NAMES[:n_classes]
    rest_s, jitter_s = 1.5, 0.5
    recs: list[Recording] = []
    for p in range(n_participants):
        mu_f = rng.uniform(9.0, 12.0)
        gain = rng.uniform(0.75, 1.25)
        p_effect = effect * rng.uniform(0.7, 1.15)
        for s in range(n_sessions):
            n_trials = trials_per_class * n_classes // n_sessions
            labels = np.array([classes[i % n_classes] for i in range(n_trials)])
            rng.shuffle(labels)
            onsets, t = [], int(1.0 * sfreq)
            for _ in range(n_trials):
                t += int((rest_s + rng.uniform(0, jitter_s)) * sfreq)
                onsets.append(t)
                t += int(trial_s * sfreq)
            n = t + int(1.5 * sfreq)
            x = 9.0 * gain * _pink(rng, C, n)
            # mu and beta sources with class-dependent desynchronisation envelopes
            sites = sorted({ERD_SITE[c] for c in classes} | {"C3", "C4"})
            for site in sites:
                env = np.ones(n)
                for onset, lab in zip(onsets, labels):
                    if ERD_SITE[lab] == site:
                        a, b = onset + int(0.25 * sfreq), onset + int(trial_s * sfreq)
                        ramp = np.clip(np.minimum(np.arange(b - a) / (0.25 * sfreq), 1), 0, 1)
                        env[a:b] = 1.0 - p_effect * ramp
                    elif lab == "feet" and site in ("C3", "C4"):
                        a, b = onset + int(0.25 * sfreq), onset + int(trial_s * sfreq)
                        env[a:b] = np.minimum(env[a:b], 1.0 + 0.25 * p_effect)  # mild ERS
                mu = 7.0 * gain * _narrowband(rng, n, sfreq, mu_f - 1.5, mu_f + 1.5) * env
                beta = 3.0 * gain * _narrowband(rng, n, sfreq, 18, 24) * (0.6 + 0.4 * env)
                x += np.outer(_spatial(ch, site), mu + beta)
            if line_noise_hz and line_noise_hz < sfreq / 2:
                tt = np.arange(n) / sfreq
                x += 2.0 * np.sin(2 * np.pi * line_noise_hz * tt + rng.uniform(0, np.pi))[None, :]
            if artifacts:
                if include_frontal:
                    for _ in range(int(n / sfreq / 6)):  # a blink every ~6 s
                        c = rng.integers(0, n - int(0.4 * sfreq))
                        w = np.hanning(int(0.4 * sfreq)) * rng.uniform(80, 160)
                        # blinks dominate Fp1/Fp2 and decay towards posterior sites
                        prop = np.array([np.exp(-(4 - POSITIONS[nm][1]) * 0.9) for nm in ch])
                        x[:, c:c + len(w)] += np.outer(prop, w)
                for _ in range(max(1, n_trials // 30)):  # rare large spikes for rejection demos
                    c = rng.integers(0, n - 10)
                    x[rng.integers(0, C), c:c + 6] += rng.choice([-1, 1]) * rng.uniform(250, 400)
            event_id = {name: i + 1 for i, name in enumerate(classes)}
            events = np.array([[o, event_id[lab]] for o, lab in zip(onsets, labels)], dtype=np.int64)
            recs.append(Recording(name=f"sub-{p + 1:02d}_ses-{s + 1:02d}", data=x.astype(np.float32), events=events,
                                  event_id=event_id, group=f"P{p + 1:02d}", session=f"S{s + 1}",
                                  recording_id=f"sub-P{p + 1:02d}_ses-S{s + 1}", event_source="embedded",
                                  source_file=f"synthetic/sub-{p + 1:02d}_ses-{s + 1:02d}"))
    return ContinuousData(sfreq=float(sfreq), ch_names=ch, unit="uV", recordings=recs, ch_types=["eeg"] * C,
                          highpass=0.0, lowpass=sfreq / 2)


def epoch_continuous(data: ContinuousData, tmin: float = -0.5, tmax: float = 2.0) -> EpochedData:
    """Cut synthetic continuous recordings into an already-epoched dataset (for the epoched variant)."""
    s0, n_t = int(round(tmin * data.sfreq)), int(round((tmax - tmin) * data.sfreq))
    X, y, g, s, rid = [], [], [], [], []
    for rec in data.recordings:
        inv = {v: k for k, v in rec.event_id.items()}
        for sample, code, _dur in rec.events:
            a = int(sample) + s0
            if a < 0 or a + n_t > rec.data.shape[1]:
                continue
            X.append(rec.data[:, a:a + n_t])
            y.append(inv[int(code)])
            g.append(rec.group or "")
            s.append(rec.session or "")
            rid.append(rec.recording_id or "")
    return EpochedData(sfreq=data.sfreq, ch_names=data.ch_names, unit=data.unit, X=np.stack(X).astype(np.float32),
                       labels=np.array(y), tmin=tmin, groups=np.array(g), sessions=np.array(s),
                       recording_ids=np.array(rid), ch_types=data.ch_types, highpass=data.highpass,
                       lowpass=data.lowpass)


def generate(kind: str = "epoched", **kwargs) -> ContinuousData | EpochedData:
    cont = generate_recordings(**kwargs)
    if kind == "continuous":
        return cont
    return epoch_continuous(cont, tmin=-0.5, tmax=kwargs.get("trial_s", 2.0))
