"""Preprocessing Studio previews: raw vs processed signals, spectra and counts (deterministic stages)."""

from __future__ import annotations

import json

import numpy as np

from ..core.preprocessing import PreprocessingConfig, PreprocessingError, ica_sources, mean_psd
from ..schemas import ExperimentRecord
from .experiments import ConfigError, epochset, exp_dir, load_ica_reference, validate_preprocessing
from .inspection import _arr, _raw_epoch


def preview(exp: ExperimentRecord, cfg: PreprocessingConfig, trial_id: str | None = None,
            channels: list[str] | None = None) -> dict:
    v = validate_preprocessing(exp, cfg)
    if v["errors"] and not any("ICA components were selected" in e for e in v["errors"]):
        return {"validation": v, "ok": False}
    try:
        es, report, det_fp = epochset(cfg, exp, preview=True)
    except PreprocessingError as exc:
        return {"validation": {**v, "errors": v["errors"] + [str(exc)]}, "ok": False}
    idx = es.index_of()
    rejected = {r["trial_id"]: r["reason"] for r in report["rejection"]["rejected"]}
    if trial_id is None or (trial_id not in idx and trial_id not in rejected):
        trial_id = str(es.trial_ids[0])
    chans = channels or es.ch_names[: min(6, len(es.ch_names))]
    ci = [es.ch_names.index(c) for c in chans if c in es.ch_names]
    out: dict = {"validation": v, "ok": True, "det_fingerprint": det_fp, "trial_id": trial_id,
                 "report": {**report, "rejection": {**report["rejection"], "rejected": report["rejection"]["rejected"][:300]}},
                 "channels": [es.ch_names[i] for i in ci], "all_channels": es.ch_names,
                 "trials": [{"trial_id": str(es.trial_ids[i]), "label": str(es.labels[i]),
                             "group": str(es.groups[i]) or None, "run": str(es.runs[i]) or None,
                             "recording_id": str(es.recordings[i]) or None}
                            for i in range(0, len(es.trial_ids), max(1, len(es.trial_ids) // 400))][:400]}
    inv = {cfg.channels.rename.get(c, c): c for c in (cfg.channels.selected or [])}
    raw = _raw_epoch(exp, trial_id, cfg)
    if raw is not None:
        rci = [raw["ch_names"].index(inv.get(es.ch_names[i], es.ch_names[i])) for i in ci
               if inv.get(es.ch_names[i], es.ch_names[i]) in raw["ch_names"]]
        out["raw"] = {"times": _arr(raw["times"], 5), "data": _arr(raw["data"][rci], 3), "sfreq": raw["sfreq"],
                      "unit": raw["unit"]}
    if trial_id in idx:
        i = idx[trial_id]
        out["processed"] = {"times": _arr(es.times, 5), "data": _arr(es.X[i][ci], 3), "sfreq": es.sfreq,
                            "label": str(es.labels[i]), "unit": "uV"}
    else:
        out["processed"] = None
        out["rejected_reason"] = rejected.get(trial_id)
    rng = np.random.default_rng(0)
    sample = rng.choice(len(es.X), size=min(60, len(es.X)), replace=False)
    raws = []
    for j in sample:
        r = _raw_epoch(exp, str(es.trial_ids[j]), cfg)
        if r is not None and r["data"].shape[1] >= 16:
            raws.append(r)
    if raws:
        n = min(r["data"].shape[1] for r in raws)
        rf, rp = mean_psd(np.stack([r["data"][:, :n] for r in raws]), raws[0]["sfreq"])
        out["psd_raw"] = {"freqs": rf, "power_db": rp}
    pf, pp = mean_psd(es.X[sample], es.sfreq)
    out["psd_processed"] = {"freqs": pf, "power_db": pp}
    out["note"] = ("Preview shows deterministic stages. ICA and normalisation are fitted on training data only when "
                   "the pipeline is run (and refitted per fold during training).")
    return out


def ica_view(exp: ExperimentRecord, trial_id: str | None = None) -> dict:
    d = exp_dir(exp.id) / "ica"
    if not exp.ica_reference or not (d / "diagnostics.json").exists():
        raise ConfigError("No ICA reference fit exists yet.")
    diag = json.loads((d / "diagnostics.json").read_text())
    cfg_path = d / "config.json"
    out = {"reference": exp.ica_reference, "diagnostics": diag}
    if cfg_path.exists():
        cfg = PreprocessingConfig(**json.loads(cfg_path.read_text()))
        es, _, _ = epochset(cfg, exp)
        ica = load_ica_reference(exp)
        idx = es.index_of()
        if trial_id is None or trial_id not in idx:
            from .experiments import load_split

            a = load_split(exp)
            rows = es.rows_for(a.folds[0].train)
            trial_id = str(es.trial_ids[rows[0]]) if rows else str(es.trial_ids[0])
        src = ica_sources(ica, es.X[idx[trial_id]], es.ch_names, es.sfreq)
        out["sources"] = {"trial_id": trial_id, "times": _arr(es.times, 5), "data": _arr(src, 4)}
    return out
