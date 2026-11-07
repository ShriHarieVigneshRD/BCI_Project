"""Learned-model inspection: Signal Journey, kernels, convolution arithmetic, Convolution Arena, Architecture Breaker.

Every value is computed from a saved checkpoint (or an explicitly labelled untrained
initialisation) and real input data. Convolution arithmetic is recomputed by hand
and cross-checked against the activation captured by the forward hook.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import ValidationError

from ..core.eegnet import (COMPUTATION_PARAMS, DISPLAY_ONLY_PARAMS, TRAINING_ONLY_PARAMS, ArchitectureConfig,
                           EEGNet, compute_architecture, parameter_shapes, validate_architecture)
from ..core.preprocessing import EpochSet, LearnedTransform, PreprocessingConfig
from ..core.tracing import run_trace
from ..data.store import EpochedData, candidate_trials
from ..lifecycle import RANK, State
from ..schemas import ExperimentRecord
from .artifacts import load_bundle
from .experiments import WorkflowError, derive_experiment, ensure_learned_ready, epochset, exp_dir, sources_for

MAX_ELEMS = 250_000


def _arr(x: np.ndarray, decimals: int = 5):
    return np.round(np.asarray(x, dtype=np.float64), decimals).tolist()


def _decimate(x: np.ndarray) -> tuple[np.ndarray, int]:
    """Decimate the last axis for display only when a tensor is very large (never for computation)."""
    if x.size <= MAX_ELEMS:
        return x, 1
    step = int(math.ceil(x.size / MAX_ELEMS))
    return x[..., ::step], step


# ---------------------------------------------------------------------------
# Trial listing and model access
# ---------------------------------------------------------------------------


def trial_list(exp: ExperimentRecord, fold: int = 0) -> dict:
    run = None
    try:
        run = ensure_learned_ready(exp)
    except WorkflowError:
        pass
    if run is not None:
        b = load_bundle(run, fold)
        es, part = b.es, b.partition
    else:
        if RANK[exp.state] < RANK[State.PREPROCESSING_COMPLETED]:
            raise WorkflowError("Run preprocessing first.")
        es, _, _ = epochset(exp.preprocessing_config, exp)
        part = None
    sets = {k: set(v) for k, v in (part or {}).items() if k in ("fit", "val", "test")}
    rows = []
    for i, tid in enumerate(es.trial_ids):
        tid, par = str(tid), str(es.parents[i])
        p = "unknown"
        if sets:
            p = "train" if par in sets["fit"] else "validation" if par in sets["val"] else \
                "test" if par in sets["test"] else "unused"
        rows.append({"trial_id": tid, "parent": par, "label": str(es.labels[i]), "group": str(es.groups[i]) or None,
                     "session": str(es.sessions[i]) or None, "run": str(es.runs[i]) or None,
                     "recording_id": str(es.recordings[i]) or None, "offset_s": round(float(es.offsets[i]), 4),
                     "partition": p})
    return {"trials": rows, "class_names": es.class_names, "fold": fold, "trained": run is not None,
            "windows": any(r["trial_id"] != r["parent"] for r in rows)}


def _untrained(exp: ExperimentRecord) -> tuple[EEGNet, LearnedTransform, EpochSet]:
    if exp.architecture is None or RANK[exp.state] < RANK[State.PREPROCESSING_COMPLETED]:
        raise WorkflowError("Configure the architecture after preprocessing to inspect the untrained model.")
    rep = compute_architecture(exp.architecture)
    if not rep.valid:
        raise WorkflowError("The current architecture is invalid.")
    torch.manual_seed(exp.training_config.seed)
    model = EEGNet(exp.architecture).eval()
    pv = exp_dir(exp.id) / "preview_transform"
    if not (pv / "transform.json").exists():
        raise WorkflowError("Run preprocessing to create the training-fitted preview transform.")
    es, _, _ = epochset(exp.preprocessing_config, exp)
    return model, LearnedTransform.load(pv), es


def _raw_epoch(exp: ExperimentRecord, trial_id: str, cfg: PreprocessingConfig) -> dict | None:
    """The uploaded signal for the same trial window, before any preprocessing (selected channels)."""
    window = None
    if "#w" in trial_id:
        trial_id, k = trial_id.rsplit("#w", 1)
        window = int(k)
    prefix, local = "", trial_id
    if "/" in trial_id:
        prefix, local = trial_id.split("/", 1)
        prefix += "/"
    src = dict(sources_for(exp))
    ds = src.get(prefix)
    if ds is None:
        return None
    sel = cfg.channels.selected if cfg.channels.selected is not None else list(ds.ch_names)
    chans = [c for c in sel if c not in cfg.channels.bad_channels]
    ci = [ds.ch_names.index(c) for c in chans]
    if isinstance(ds, EpochedData):
        hits = np.flatnonzero(ds.trial_ids == local)
        if not len(hits):
            return None
        x = ds.X[int(hits[0])][ci]
        t0 = ds.tmin
    else:
        t = next((t for t in candidate_trials(ds) if t.trial_id == local), None)
        if t is None:
            return None
        rec = ds.recordings[t.recording]
        e = cfg.epoching
        if window is not None:  # same window placement as the preprocessing (see run_deterministic)
            off = (e.segment_offset_s + window * e.step_s) if t.duration_s > 0 else e.tmin
            a = t.sample + int(round(off * ds.sfreq))
            n = int(round(e.window_s * ds.sfreq))
        else:
            a = t.sample + int(round(e.tmin * ds.sfreq))
            n = int(round((e.tmax - e.tmin) * ds.sfreq))
        a0, a1 = max(a, 0), min(a + n, rec.data.shape[1])
        x = rec.data[ci, a0:a1]
        t0 = (a0 - t.sample) / ds.sfreq
    return {"data": x, "times": t0 + np.arange(x.shape[1]) / ds.sfreq, "sfreq": ds.sfreq, "ch_names": chans,
            "unit": ds.unit}


# ---------------------------------------------------------------------------
# Signal Journey
# ---------------------------------------------------------------------------


def journey(exp: ExperimentRecord, trial_id: str, fold: int = 0, mode: str = "trained") -> dict:
    if mode == "trained":
        run = ensure_learned_ready(exp)
        b = load_bundle(run, fold)
        model, transform, es = b.model, b.transform, b.es
        banner = (f"Trained checkpoint {b.checkpoint_id} (fold {fold + 1}, epoch {b.selected_epoch}); "
                  "eval mode, forward hooks.")
        provenance = {"checkpoint_id": b.checkpoint_id, "sha256": b.sha256, "run_id": run.id, "fold": fold,
                      "partition": b.partition_of(trial_id)}
        pcfg = PreprocessingConfig(**run.snapshot["preprocessing_config"])
    else:
        model, transform, es = _untrained(exp)
        banner = (f"UNTRAINED model - random initialisation (seed {exp.training_config.seed}). These activations show "
                  "shapes and data flow only; they are not learned features.")
        provenance = {"checkpoint_id": None, "partition": "n/a (untrained)"}
        pcfg = exp.preprocessing_config
    idx = es.index_of()
    if trial_id not in idx:
        raise KeyError(f"Trial {trial_id} is not in the preprocessed data (rejected or unknown).")
    i = idx[trial_id]
    x_proc = es.X[i]
    x_in = transform.transform(x_proc[None])[0]
    tr = run_trace(model, x_in)
    rep = compute_architecture(model.cfg)
    specs = {s.name: s for s in rep.layers}
    stages = []
    raw = _raw_epoch(exp, trial_id, pcfg)
    if raw is not None:
        d, step = _decimate(raw["data"])
        stages.append({"id": "uploaded", "label": "Uploaded signal", "kind": "signal", "unit": raw["unit"],
                       "shape": list(raw["data"].shape), "data": _arr(d, 3), "times": _arr(raw["times"][::step], 5),
                       "ch_names": raw["ch_names"], "sfreq": raw["sfreq"],
                       "description": "Original recording for this trial window, before any preprocessing."})
    times = es.times
    stages.append({"id": "processed", "label": "Processed epoch", "kind": "signal", "unit": "uV",
                   "shape": list(x_proc.shape), "data": _arr(x_proc, 3), "times": _arr(times, 5),
                   "ch_names": es.ch_names, "sfreq": es.sfreq,
                   "description": "After the deterministic preprocessing stages (filtering, resampling, epoching, "
                                  "baseline)."})
    stages.append({"id": "model_input", "label": "Model input", "kind": "signal", "unit": "normalised",
                   "shape": [1, *x_in.shape], "data": _arr(x_in, 4), "times": _arr(times, 5),
                   "ch_names": es.ch_names, "sfreq": es.sfreq,
                   "description": f"After learned stages fitted on training data ({es.X.shape[1]} channels x "
                                  f"{es.X.shape[2]} samples, convention [batch, 1, channels, time])."})
    for name in model.layer_names:
        out = tr.outputs[name][0]
        sp = specs[name]
        d, step = _decimate(out)
        stages.append({
            "id": name, "label": sp.label, "kind": "activation" if out.ndim == 3 else "vector", "unit": "activation",
            "type": sp.type, "block": sp.block, "purpose": sp.purpose, "in_shape": sp.in_shape,
            "shape": list(out.shape), "data": _arr(d, 4), "decimation": step,
            "stats": {"min": float(out.min()), "max": float(out.max()), "mean": float(out.mean()),
                      "std": float(out.std())},
            "time_jump": sp.jump_time, "receptive_field": sp.rf_time,
            "ch_names": es.ch_names if (out.ndim == 3 and out.shape[1] == len(es.ch_names)
                                        and sp.rf_channels == 1) else None,
        })
    probs = tr.probabilities[0]
    return {
        "mode": mode, "banner": banner, "provenance": provenance,
        "trial": {"trial_id": trial_id, "label": str(es.labels[i]), "group": str(es.groups[i]) or None,
                  "session": str(es.sessions[i]) or None, "run": str(es.runs[i]) or None,
                  "recording_id": str(es.recordings[i]) or None, "parent": str(es.parents[i]),
                  "offset_s": round(float(es.offsets[i]), 4)},
        "class_names": es.class_names, "stages": stages,
        "logits": _arr(tr.logits[0], 5), "probabilities": _arr(probs, 5),
        "predicted": es.class_names[int(probs.argmax())],
        "note": "Intermediate feature maps are activations, not EEG voltages.",
    }


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def kernels(exp: ExperimentRecord, fold: int = 0) -> dict:
    run = ensure_learned_ready(exp)
    b = load_bundle(run, fold)
    m, cfg = b.model, b.arch
    sd = {k: v.detach().cpu().numpy() for k, v in m.state_dict().items()}
    sf = b.es.sfreq
    wt = sd["temporal_conv.conv.weight"][:, 0, 0, :]
    nfft = max(512, 1 << int(math.ceil(math.log2(wt.shape[1]))))
    H = np.abs(np.fft.rfft(wt, n=nfft, axis=1))
    freqs = np.fft.rfftfreq(nfft, 1 / sf)
    resp_db = 20 * np.log10(H / (H.max(axis=1, keepdims=True) + 1e-12) + 1e-12)
    peak = freqs[H.argmax(axis=1)]
    fd = cfg.F1 * cfg.D
    wd = sd["depthwise_conv.conv.weight"][:, 0, :, 0]
    rows = b.es.ch_names[: cfg.spatial_k] if cfg.spatial_k == cfg.n_channels else \
        [f"rows h..h+{cfg.spatial_k - 1}" for _ in range(cfg.spatial_k)]
    out = {
        "checkpoint_id": b.checkpoint_id, "sha256": b.sha256, "fold": fold, "run_id": run.id,
        "source": f"state_dict of checkpoint {b.checkpoint_id} (strict load, eval mode)",
        "temporal": {"weights": _arr(wt, 6), "shape": list(sd["temporal_conv.conv.weight"].shape),
                     "freqs": _arr(freqs[freqs <= sf / 2], 3), "response_db": _arr(resp_db, 3),
                     "peak_hz": _arr(peak, 2), "sfreq": sf, "kernel_ms": wt.shape[1] / sf * 1000},
        "depthwise": {"weights": _arr(wd, 6), "shape": list(sd["depthwise_conv.conv.weight"].shape),
                      "groups": cfg.F1, "D": cfg.D, "rows": list(rows),
                      "group_of_output": [o // cfg.D for o in range(fd)],
                      "max_norm": cfg.depthwise_max_norm,
                      "norms": _arr(np.linalg.norm(wd, axis=1), 4)},
        "separable_depthwise": {"weights": _arr(sd["separable_depthwise.conv.weight"][:, 0, 0, :], 6),
                                "shape": list(sd["separable_depthwise.conv.weight"].shape), "groups": fd},
        "pointwise": {"weights": _arr(sd["separable_pointwise.conv.weight"][:, :, 0, 0], 6),
                      "shape": list(sd["separable_pointwise.conv.weight"].shape)},
        "classifier": {"weights": _arr(sd["classifier.weight"], 6), "bias": _arr(sd["classifier.bias"], 6),
                       "shape": list(sd["classifier.weight"].shape),
                       "feature_shape": compute_architecture(cfg).layers[-2].in_shape, "class_names": b.class_names},
        "batchnorm": {},
    }
    for bn in ("bn1", "bn2", "bn3"):
        if f"{bn}.weight" in sd:
            out["batchnorm"][bn] = {k: _arr(sd[f"{bn}.{k}"], 6) for k in ("weight", "bias", "running_mean",
                                                                          "running_var")}
            out["batchnorm"][bn]["eps"] = cfg.bn_eps
    return out


# ---------------------------------------------------------------------------
# Convolution arithmetic
# ---------------------------------------------------------------------------

ACT = {"elu": F.elu, "relu": F.relu, "leaky_relu": lambda t: F.leaky_relu(t, 0.01), "gelu": F.gelu,
       "silu": F.silu}


def _check(computed: float, actual: float) -> dict:
    diff = abs(computed - actual)
    return {"computed": computed, "actual": actual, "abs_diff": diff,
            "verified": diff <= 1e-4 * max(1.0, abs(actual))}


def _window(row: np.ndarray, start: int, k: int) -> tuple[list[int], list[float], list[bool]]:
    idx = list(range(start, start + k))
    vals = [float(row[t]) if 0 <= t < len(row) else 0.0 for t in idx]
    pad = [not (0 <= t < len(row)) for t in idx]
    return idx, vals, pad


def convolution(exp: ExperimentRecord, layer: str, trial_id: str, channel: int = 0, row: int = 0,
                position: int = 0, fold: int = 0, compare_trial_id: str | None = None) -> dict:
    run = ensure_learned_ready(exp)
    b = load_bundle(run, fold)
    m, cfg = b.model, b.arch
    if layer not in m.layer_names:
        raise KeyError(f"Unknown layer {layer}")
    x = b.model_input([trial_id])
    tr = run_trace(m, x)
    inp, out = tr.inputs[layer][0], tr.outputs[layer][0]
    spec = {s.name: s for s in compute_architecture(cfg).layers}[layer]
    res: dict = {"layer": layer, "type": spec.type, "label": spec.label, "purpose": spec.purpose,
                 "in_shape": list(inp.shape), "out_shape": list(out.shape), "checkpoint_id": b.checkpoint_id,
                 "trial_id": trial_id, "partition": b.partition_of(trial_id), "kernel": spec.kernel,
                 "stride": spec.stride, "padding": spec.padding, "groups": spec.groups}

    def clamp(v, n):
        if not 0 <= v < n:
            raise ValueError(f"index {v} outside [0, {n - 1}]")
        return v

    mod = getattr(m, layer)
    if layer in ("temporal_conv", "separable_depthwise"):
        W = mod.conv.weight.detach().numpy()
        f = clamp(channel, out.shape[0])
        h = clamp(row, out.shape[1])
        t = clamp(position, out.shape[2])
        k = W.shape[-1]
        s = mod.conv.stride[1]
        pl = mod.pad_time[0]
        in_ch = 0 if layer == "temporal_conv" else f
        in_row = inp[in_ch, h]
        start = t * s - pl
        idx, vals, pad = _window(in_row, start, k)
        w = W[f, 0, 0, :]
        prod = np.array(vals) * w
        res.update({
            "operation": "temporal convolution (cross-correlation): y[f,h,t] = sum_k x[c,h,t*s - pad_left + k] * w[f,k]",
            "selected": {"out_channel": f, "row": h, "position": t, "in_channel": in_ch,
                         "row_name": b.es.ch_names[h] if layer == "temporal_conv" else None},
            "input_row": _arr(in_row), "window": {"indices": idx, "values": _arr(vals), "padded": pad},
            "weights": _arr(w, 6), "products": _arr(prod, 6), "output_row": _arr(out[f, h]),
            "output_map": _arr(out[f]) if out[f].size <= MAX_ELEMS else None,
            **_check(float(prod.sum()), float(out[f, h, t])),
        })
    elif layer == "depthwise_conv":
        W = mod.conv.weight.detach().numpy()
        o = clamp(channel, out.shape[0])
        h = clamp(row, out.shape[1])
        t = clamp(position, out.shape[2])
        g = o // cfg.D
        ks = W.shape[2]
        vals = inp[g, h:h + ks, t]
        w = W[o, 0, :, 0]
        prod = vals * w
        res.update({
            "operation": "depthwise spatial convolution: y[o,h,t] = sum_c x[o // D, h + c, t] * w[o,c] "
                         f"(groups = F1 = {cfg.F1}; output map {o} reads only input map {g})",
            "selected": {"out_channel": o, "row": h, "position": t, "in_channel": g},
            "input_map": _arr(inp[g]), "window": {"channels": b.es.ch_names[h:h + ks], "values": _arr(vals)},
            "weights": _arr(w, 6), "products": _arr(prod, 6), "output_row": _arr(out[o, h]),
            **_check(float(prod.sum()), float(out[o, h, t])),
        })
    elif layer == "separable_pointwise":
        W = mod.conv.weight.detach().numpy()
        o = clamp(channel, out.shape[0])
        h = clamp(row, out.shape[1])
        t = clamp(position, out.shape[2])
        vals = inp[:, h, t]
        w = W[o, :, 0, 0]
        prod = vals * w
        res.update({
            "operation": "pointwise (1x1) convolution: y[o,h,t] = sum_c x[c,h,t] * w[o,c] - mixes all feature maps",
            "selected": {"out_channel": o, "row": h, "position": t},
            "input_column": _arr(vals), "weights": _arr(w, 6), "products": _arr(prod, 6),
            "input_map": _arr(inp[:, h, :]), "output_row": _arr(out[o, h]),
            **_check(float(prod.sum()), float(out[o, h, t])),
        })
    elif layer in ("pool1", "pool2"):
        c = clamp(channel, out.shape[0])
        h = clamp(row, out.shape[1])
        t = clamp(position, out.shape[2])
        k, s = spec.kernel[1], spec.stride[1]
        win = inp[c, h, t * s:t * s + k]
        comp = float(win.mean() if cfg.pool_type == "avg" else win.max())
        res.update({
            "operation": f"{'average' if cfg.pool_type == 'avg' else 'max'} pooling over {k} samples, stride {s}",
            "selected": {"out_channel": c, "row": h, "position": t},
            "input_row": _arr(inp[c, h]), "window": {"indices": list(range(t * s, t * s + k)), "values": _arr(win)},
            "output_row": _arr(out[c, h]), **_check(comp, float(out[c, h, t])),
        })
    elif layer.startswith("bn"):
        c = clamp(channel, out.shape[0])
        h = clamp(row, out.shape[1])
        t = clamp(position, out.shape[2])
        g, be = mod.weight.detach().numpy()[c], mod.bias.detach().numpy()[c]
        mu, var = mod.running_mean.numpy()[c], mod.running_var.numpy()[c]
        v = float(inp[c, h, t])
        comp = (v - mu) / math.sqrt(var + mod.eps) * g + be
        res.update({
            "operation": "batch normalisation (eval mode, running statistics): y = (x - mean) / sqrt(var + eps) * "
                         "gamma + beta",
            "selected": {"out_channel": c, "row": h, "position": t},
            "params": {"gamma": float(g), "beta": float(be), "running_mean": float(mu), "running_var": float(var),
                       "eps": float(mod.eps)},
            "input_value": v, "input_row": _arr(inp[c, h]), "output_row": _arr(out[c, h]),
            **_check(float(comp), float(out[c, h, t])),
        })
    elif layer.startswith("act"):
        c = clamp(channel, out.shape[0])
        h = clamp(row, out.shape[1])
        t = clamp(position, out.shape[2])
        fn = ACT[cfg.activation]
        v = float(inp[c, h, t])
        xs = np.linspace(float(inp.min()), float(inp.max()), 120)
        ys = fn(torch.from_numpy(xs)).numpy()
        res.update({
            "operation": f"element-wise {cfg.activation.upper()}", "selected": {"out_channel": c, "row": h,
                                                                               "position": t},
            "input_value": v, "curve": {"x": _arr(xs), "y": _arr(ys)}, "input_row": _arr(inp[c, h]),
            "output_row": _arr(out[c, h]),
            **_check(float(fn(torch.tensor(v, dtype=torch.float64)).item()), float(out[c, h, t])),
        })
    elif layer.startswith("drop"):
        c = clamp(channel, out.shape[0])
        h = clamp(row, out.shape[1])
        t = clamp(position, out.shape[2])
        res.update({
            "operation": f"dropout (p = {cfg.dropout}): identity in evaluation mode. During training each "
                         f"{'activation' if cfg.dropout_type == 'dropout' else 'feature map'} is zeroed with "
                         f"probability p and survivors are scaled by 1/(1-p).",
            "selected": {"out_channel": c, "row": h, "position": t}, "input_row": _arr(inp[c, h]),
            "output_row": _arr(out[c, h]), **_check(float(inp[c, h, t]), float(out[c, h, t])),
        })
    elif layer == "flatten":
        Fm, H, Wd = inp.shape
        c = clamp(channel, Fm)
        h = clamp(row, H)
        t = clamp(position, Wd)
        flat = c * H * Wd + h * Wd + t
        res.update({"operation": "flatten [F2, H, W] -> vector (row-major): index = f*H*W + h*W + w",
                    "selected": {"out_channel": c, "row": h, "position": t, "flat_index": flat},
                    "feature_shape": [Fm, H, Wd], **_check(float(inp[c, h, t]), float(out[flat]))})
    elif layer == "classifier":
        k = clamp(channel, out.shape[0])
        W = mod.weight.detach().numpy()
        bias = mod.bias.detach().numpy()
        prod = inp * W[k]
        fshape = spec.in_shape
        flat_spec = {s.name: s for s in compute_architecture(cfg).layers}["flatten"].in_shape
        top = np.argsort(-np.abs(prod))[:16]
        res.update({
            "operation": "dense layer: logit[k] = sum_i feature[i] * W[k,i] + b[k]",
            "selected": {"class_index": k, "class_name": b.class_names[k]},
            "features": _arr(inp), "weights": _arr(W[k], 6), "bias": float(bias[k]), "products": _arr(prod, 6),
            "products_map": _arr(prod.reshape(flat_spec)), "feature_shape": flat_spec, "in_shape": fshape,
            "top_contributions": [{"index": int(i), "fhw": [int(v) for v in np.unravel_index(int(i), flat_spec)],
                                   "feature": float(inp[i]), "weight": float(W[k, i]), "product": float(prod[i])}
                                  for i in top],
            "probabilities": _arr(tr.probabilities[0]),
            **_check(float(prod.sum() + bias[k]), float(out[k])),
        })
    res["prediction"] = {"probabilities": _arr(tr.probabilities[0]),
                         "predicted": b.class_names[int(tr.probabilities[0].argmax())],
                         "true": str(b.es.labels[b.es.index_of()[trial_id]]), "class_names": b.class_names}
    if compare_trial_id:
        x2 = b.model_input([compare_trial_id])
        tr2 = run_trace(m, x2)
        o1, o2 = tr.outputs[layer][0], tr2.outputs[layer][0]
        sel = res["selected"].get("out_channel", res["selected"].get("class_index", 0))
        a1 = o1[sel] if o1.ndim == 3 else o1
        a2 = o2[sel] if o2.ndim == 3 else o2
        res["comparison"] = {
            "trial_a": trial_id, "trial_b": compare_trial_id,
            "label_b": str(b.es.labels[b.es.index_of()[compare_trial_id]]),
            "output_a": _arr(a1), "output_b": _arr(a2), "difference": _arr(a2 - a1),
            "input_difference_rms": float(np.sqrt(((x2 - x) ** 2).mean())),
            "activation_difference_rms": float(np.sqrt(((o2 - o1) ** 2).mean())),
            "probabilities_a": _arr(tr.probabilities[0]), "probabilities_b": _arr(tr2.probabilities[0]),
            "predicted_b": b.class_names[int(tr2.probabilities[0].argmax())],
        }
    return res


# ---------------------------------------------------------------------------
# Convolution Arena: depthwise vs separable (actual weights and activations)
# ---------------------------------------------------------------------------


def arena(exp: ExperimentRecord, trial_id: str, fold: int = 0, map_index: int = 0, demo_stride: int = 1,
          demo_padding: str = "same") -> dict:
    run = ensure_learned_ready(exp)
    b = load_bundle(run, fold)
    m, cfg = b.model, b.arch
    tr = run_trace(m, b.model_input([trial_id]))
    fd = cfg.F1 * cfg.D
    j = max(0, min(map_index, fd - 1))
    dw_in = tr.inputs["depthwise_conv"][0]
    dw_out = tr.outputs["depthwise_conv"][0]
    wd = m.depthwise_conv.conv.weight.detach().numpy()[:, 0, :, 0]
    sd_in = tr.inputs["separable_depthwise"][0]
    sd_out = tr.outputs["separable_depthwise"][0]
    ws = m.separable_depthwise.conv.weight.detach().numpy()[:, 0, 0, :]
    pw_out = tr.outputs["separable_pointwise"][0]
    wp = m.separable_pointwise.conv.weight.detach().numpy()[:, :, 0, 0]
    rep = {s.name: s for s in compute_architecture(cfg).layers}
    # isolated demonstration: the learned separable-depthwise kernel j applied with another stride / padding
    k2 = ws.shape[1]
    xin = torch.from_numpy(sd_in[j:j + 1][None]).float()  # [1,1,1,W]
    if demo_padding == "same":
        total = max((math.ceil(xin.shape[-1] / demo_stride) - 1) * demo_stride + k2 - xin.shape[-1], 0)
        pads = (total // 2, total - total // 2)
    else:
        pads = (0, 0)
    xin_p = F.pad(xin, (pads[0], pads[1], 0, 0))
    demo = None
    if xin_p.shape[-1] >= k2:
        y = F.conv2d(xin_p, torch.from_numpy(ws[j]).float().view(1, 1, 1, k2), stride=(1, demo_stride))
        demo = {"stride": demo_stride, "padding": demo_padding, "pad": list(pads), "input_length": int(xin.shape[-1]),
                "output_length": int(y.shape[-1]), "output": _arr(y[0, 0, 0].numpy()),
                "formula": f"floor(({xin.shape[-1]} + {pads[0]} + {pads[1]} - {k2}) / {demo_stride}) + 1 = "
                           f"{int(y.shape[-1])}",
                "label": "Isolated educational demonstration using the learned kernel - not part of the trained "
                         "model's computation."}
    return {
        "checkpoint_id": b.checkpoint_id, "trial_id": trial_id, "map_index": j,
        "depthwise": {
            "layer": "depthwise_conv", "groups": cfg.F1, "D": cfg.D, "kernel": rep["depthwise_conv"].kernel,
            "in_shape": rep["depthwise_conv"].in_shape, "out_shape": rep["depthwise_conv"].out_shape,
            "params": rep["depthwise_conv"].params, "weights": _arr(wd, 6), "ch_names": b.es.ch_names,
            "mapping": [{"out": o, "in": o // cfg.D} for o in range(fd)],
            "selected_input_map": _arr(dw_in[j // cfg.D]), "selected_output": _arr(dw_out[j]),
            "explanation": f"Each of the F1 = {cfg.F1} temporal feature maps gets its own D = {cfg.D} spatial filters "
                           f"(groups = F1). Output map {j} reads only input map {j // cfg.D}; no mixing across "
                           "temporal filters happens here.",
        },
        "separable": {
            "depthwise": {"layer": "separable_depthwise", "groups": fd, "kernel": rep["separable_depthwise"].kernel,
                          "padding": rep["separable_depthwise"].padding, "params": rep["separable_depthwise"].params,
                          "in_shape": rep["separable_depthwise"].in_shape,
                          "out_shape": rep["separable_depthwise"].out_shape, "weights": _arr(ws, 6),
                          "selected_input": _arr(sd_in[j, 0]), "selected_output": _arr(sd_out[j, 0])},
            "pointwise": {"layer": "separable_pointwise", "params": rep["separable_pointwise"].params,
                          "in_shape": rep["separable_pointwise"].in_shape,
                          "out_shape": rep["separable_pointwise"].out_shape, "weights": _arr(wp, 6),
                          "outputs": _arr(pw_out[:, 0, :])},
            "total_params": rep["separable_depthwise"].params + rep["separable_pointwise"].params,
            "explanation": f"Step 1 filters each of the {fd} maps over time on its own (groups = {fd}); step 2 mixes "
                           f"the {fd} maps into F2 = {cfg.F2} maps with a {cfg.F2}x{fd} matrix at every time step.",
        },
        "demo": demo,
    }


# ---------------------------------------------------------------------------
# Architecture Breaker
# ---------------------------------------------------------------------------


def _field_kind(base: ArchitectureConfig, field: str, value) -> str:
    if field in DISPLAY_ONLY_PARAMS:
        return "display_only"
    try:
        single = ArchitectureConfig(**{**base.model_dump(), field: value})
    except ValidationError:
        return "invalid"
    rep = compute_architecture(single)
    if not rep.valid:
        return "structural"
    if parameter_shapes(EEGNet(single)) != parameter_shapes(EEGNet(base)):
        return "structural"
    if field in TRAINING_ONLY_PARAMS:
        return "training_only"
    return "computation"


def breaker(exp: ExperimentRecord, changes: dict, trial_id: str | None = None, fold: int = 0) -> dict:
    run = ensure_learned_ready(exp)
    b = load_bundle(run, fold)
    base = ArchitectureConfig(**run.snapshot["architecture_config"])
    allowed = set(ArchitectureConfig.model_fields)
    unknown = sorted(set(changes) - allowed)
    if unknown:
        raise ValueError(f"Unknown architecture parameter(s): {', '.join(unknown)}")
    try:
        variant = ArchitectureConfig(**{**base.model_dump(), **changes})
    except ValidationError as exc:
        return {"valid": False, "field_errors": [{"field": ".".join(map(str, e["loc"])), "message": e["msg"]}
                                                 for e in exc.errors()], "base": None, "variant": None}
    base_rep, var_rep = validate_architecture(base), validate_architecture(variant)
    changed = {k: {"before": getattr(base, k), "after": getattr(variant, k)} for k in changes
               if getattr(base, k) != getattr(variant, k)}
    for k, v in changed.items():
        kind = _field_kind(base, k, v["after"])
        v["kind"] = kind
        v["note"] = (TRAINING_ONLY_PARAMS.get(k) or DISPLAY_ONLY_PARAMS.get(k) or COMPUTATION_PARAMS.get(k)
                     or ("Changes the shape of learned weights." if kind == "structural" else ""))
    base_shapes = parameter_shapes(b.model)
    var_shapes = parameter_shapes(EEGNet(variant)) if var_rep.valid else {
        f"{s.name}.{'conv.' if s.name.endswith('conv') or s.name.startswith('separable') else ''}{p}": shp
        for s in var_rep.layers for p, shp in s.param_shapes.items()}
    tensors = []
    for key in sorted(set(base_shapes) | set(var_shapes)):
        bs, vs = base_shapes.get(key), var_shapes.get(key)
        tensors.append({"tensor": key, "base": bs, "variant": vs, "compatible": bs == vs})
    shapes_match = all(t["compatible"] for t in tensors)
    kinds = {v["kind"] for v in changed.values()}
    checkpoint_valid = var_rep.valid and shapes_match and kinds <= {"training_only", "display_only"}
    if not changed:
        verdict = "No change: the trained checkpoint is the model shown."
    elif not var_rep.valid:
        verdict = "The variant is invalid; fix the reported errors."
    elif checkpoint_valid:
        verdict = ("The saved checkpoint remains valid for inference with this variant (only training-time or display "
                   "settings changed). Training with these settings would still produce a different model.")
    elif shapes_match:
        verdict = ("Learned weights have compatible shapes but were learned for a different computation; they are "
                   "marked incompatible and are not reused. Train the variant to obtain learned results.")
    else:
        verdict = ("Learned weight shapes no longer match; the checkpoint cannot be used. Train the variant as a new "
                   "experiment to obtain learned results.")

    def layers(rep):
        return {s.name: {"type": s.type, "label": s.label, "in_shape": s.in_shape, "out_shape": s.out_shape,
                         "params": s.params, "macs": s.macs, "rf_time": s.rf_time, "valid": s.valid,
                         "message": s.message} for s in rep.layers}

    lb, lv = layers(base_rep), layers(var_rep)
    order = [s.name for s in var_rep.layers] + [n for n in lb if n not in lv]
    diff = [{"layer": n, "base": lb.get(n), "variant": lv.get(n),
             "changed": (lb.get(n) or {}).get("out_shape") != (lv.get(n) or {}).get("out_shape")
             or (lb.get(n) or {}).get("params") != (lv.get(n) or {}).get("params")} for n in order]

    def totals(rep):
        return {"params": rep.total_params, "trainable": rep.trainable_params, "macs": rep.total_macs,
                "activation_kb_per_sample": round(rep.activation_bytes_per_sample / 1024, 1),
                "param_kb": round(rep.param_bytes / 1024, 2), "receptive_field_samples": rep.receptive_field_samples,
                "receptive_field_s": rep.receptive_field_seconds, "classifier_in": rep.classifier_in_features,
                "valid": rep.valid}

    learned = None
    if checkpoint_valid and trial_id:
        vm = EEGNet(variant)
        vm.load_state_dict(b.model.state_dict(), strict=True)
        x = b.model_input([trial_id])
        p_var = run_trace(vm, x).probabilities[0]
        p_base = run_trace(b.model, x).probabilities[0]
        learned = {"trial_id": trial_id, "probabilities_base": _arr(p_base), "probabilities_variant": _arr(p_var),
                   "max_abs_difference": float(np.abs(p_var - p_base).max()), "class_names": b.class_names,
                   "note": "Computed with the saved checkpoint loaded into the variant (eval mode)."}
    return {
        "valid": var_rep.valid, "base_checkpoint_id": b.checkpoint_id, "changed": changed,
        "base": {"config": base.model_dump(), "totals": totals(base_rep),
                 "issues": [i.__dict__ for i in base_rep.issues]},
        "variant": {"config": variant.model_dump(), "totals": totals(var_rep),
                    "issues": [i.__dict__ for i in var_rep.issues], "torch_verified": var_rep.torch_verified},
        "layers": diff, "tensors": tensors, "checkpoint_valid": checkpoint_valid, "verdict": verdict,
        "learned_outputs": learned,
        "immutability": f"Experiment {exp.id} and checkpoint {b.checkpoint_id} are unchanged; variants are analysed "
                        "separately and can only be trained as new experiments.",
    }


def spawn_variant(exp: ExperimentRecord, changes: dict, name: str | None = None) -> ExperimentRecord:
    run = ensure_learned_ready(exp)
    base = ArchitectureConfig(**run.snapshot["architecture_config"])
    variant = ArchitectureConfig(**{**base.model_dump(), **changes, "preset": "custom"})
    rep = compute_architecture(variant)
    if not rep.valid:
        raise ValueError("; ".join(i.message for i in rep.issues if i.level == "error"))
    label = ", ".join(f"{k}={v}" for k, v in changes.items())
    return derive_experiment(exp, name or f"{exp.name} - variant ({label})", architecture=variant,
                             reason="architecture breaker variant")
