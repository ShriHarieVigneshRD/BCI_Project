"""Experiment export bundles (ZIP) with a provenance manifest and an HTML report."""

from __future__ import annotations

import base64
import csv
import hashlib
import html
import io
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..config import settings  # noqa: E402
from ..core.eegnet import ArchitectureConfig, compute_architecture  # noqa: E402
from ..schemas import ExperimentRecord, now_iso  # noqa: E402
from . import repo  # noqa: E402
from .experiments import WorkflowError, active_run, capabilities, exp_dir, load_split  # noqa: E402
from .jobs import run_dir  # noqa: E402

EXPORT_ITEMS = {
    "model_config": "Architecture configuration (JSON)",
    "training_config": "Training configuration (JSON)",
    "preprocessing_config": "Preprocessing configuration and processing metadata (JSON)",
    "dataset_manifest": "Dataset manifest: recordings, participant/session/run IDs, event mapping, file references",
    "trial_provenance": "Trial provenance: dataset, participant, session, run, recording, source file, partition (CSV)",
    "split_manifest": "Split manifest: protocol, seed and trial assignments (JSON)",
    "metrics": "Metrics (JSON + CSV)",
    "predictions": "Predictions (CSV)",
    "history": "Training history (CSV + JSON)",
    "confusion": "Confusion matrix (CSV)",
    "plots": "Plots (PNG + SVG)",
    "checkpoints": "Trained PyTorch checkpoints (.pt)",
    "report": "Experiment report (HTML)",
}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _csv(rows: list[list]) -> bytes:
    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    return buf.getvalue().encode()


def _plots(run, evaluation) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for f in run.folds:
        ep = [h["epoch"] for h in f.history]
        axes[0].plot(ep, [h["train_loss"] for h in f.history], color="#1f2b6c", alpha=0.8,
                     label="train" if f.fold == 0 else None)
        if f.history and "val_loss" in f.history[0]:
            axes[0].plot(ep, [h["val_loss"] for h in f.history], color="#b45309", alpha=0.8,
                         label="validation" if f.fold == 0 else None)
            axes[1].plot(ep, [h["val_balanced_accuracy"] for h in f.history], color="#b45309", alpha=0.8,
                         label="validation bal. acc." if f.fold == 0 else None)
        axes[1].plot(ep, [h["train_acc"] for h in f.history], color="#1f2b6c", alpha=0.8,
                     label="train acc." if f.fold == 0 else None)
    axes[0].set(title="Loss", xlabel="epoch")
    axes[1].set(title="Accuracy", xlabel="epoch", ylim=(0, 1))
    for a in axes:
        a.legend(frameon=False)
    fig.tight_layout()
    for fmt in ("png", "svg"):
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, dpi=150)
        out[f"plots/training_curves.{fmt}"] = buf.getvalue()
    plt.close(fig)
    if evaluation:
        pooled = evaluation["test"]["pooled"]
        cm = pooled["confusion_matrix"]
        names = pooled["class_names"]
        fig, ax = plt.subplots(figsize=(1.2 + 0.9 * len(names), 1 + 0.8 * len(names)))
        ax.imshow(cm, cmap="Blues")
        for i, row in enumerate(cm):
            for j, v in enumerate(row):
                ax.text(j, i, str(v), ha="center", va="center", fontsize=9)
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set(xlabel="predicted", ylabel="true", title="Test confusion matrix")
        fig.tight_layout()
        for fmt in ("png", "svg"):
            buf = io.BytesIO()
            fig.savefig(buf, format=fmt, dpi=150)
            out[f"plots/confusion_matrix.{fmt}"] = buf.getvalue()
        plt.close(fig)
    return out


def _report(exp: ExperimentRecord, run, plots: dict[str, bytes], synthetic: bool) -> bytes:
    e = html.escape
    arch = ArchitectureConfig(**run.snapshot["architecture_config"])
    rep = compute_architecture(arch)
    ev = run.evaluation
    img = lambda key: (f'<img alt="{e(key)}" src="data:image/png;base64,{base64.b64encode(plots[key]).decode()}">'  # noqa: E731
                       if key in plots else "")
    rows = "".join(f"<tr><td>{e(s.label)}</td><td>{e(s.type)}</td><td>{s.in_shape}</td><td>{s.out_shape}</td>"
                   f"<td>{s.params}</td></tr>" for s in rep.layers)
    metrics_html = "<p><em>The model has not been evaluated on the test partition.</em></p>"
    if ev:
        p = ev["test"]["pooled"]
        fmt = lambda v: "undefined" if v is None else f"{v:.3f}"  # noqa: E731
        per = "".join(f"<tr><td>{e(c['class'])}</td><td>{fmt(c['precision'])}</td><td>{fmt(c['recall'])}</td>"
                      f"<td>{fmt(c['f1'])}</td><td>{c['support']}</td></tr>" for c in p["per_class"])
        metrics_html = (f"<p>Test trials: {p['n']} &middot; accuracy {fmt(p['accuracy'])} &middot; balanced accuracy "
                        f"{fmt(p['balanced_accuracy'])} &middot; macro F1 {fmt(p['averages']['macro']['f1'])} &middot; "
                        f"ROC-AUC (OvR macro) {fmt(p['roc_auc_ovr_macro'])} &middot; chance "
                        f"{p['chance_level']:.3f}</p><table><tr><th>Class</th><th>Precision</th><th>Recall</th>"
                        f"<th>F1</th><th>Support</th></tr>{per}</table>{img('plots/confusion_matrix.png')}")
    split = load_split(exp, run.snapshot["split_fingerprint"])
    warn = ('<p class="warn">SYNTHETIC DATA: generated for software testing. These metrics are not evidence of '
            'real-world EEG decoding performance.</p>' if synthetic else "")
    sw = ", ".join(f"{k} {v}" for k, v in run.software.items())
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>{e(exp.name)} - EEGNet report</title>
<style>body{{font-family:Georgia,serif;max-width:900px;margin:40px auto;color:#1b1b1f;line-height:1.5;padding:0 20px}}
h1,h2{{font-weight:400}}table{{border-collapse:collapse;width:100%;font:13px system-ui,sans-serif}}
td,th{{border-bottom:1px solid #e4e1da;padding:4px 8px;text-align:left}}th{{background:#eeebe5}}
.warn{{background:#fdecea;color:#b42318;padding:8px 12px;border-radius:8px;font-family:system-ui}}
code{{font-size:12px}}img{{max-width:100%}}</style></head><body>
<h1>{e(exp.name)}</h1>{warn}
<p>Experiment <code>{exp.id}</code> &middot; run <code>{run.id}</code> &middot; generated {now_iso()}</p>
<h2>Evaluation protocol</h2><p>{e(split.protocol)}</p><p>{e(split.question)}</p>
<p>Split fingerprint <code>{split.fingerprint}</code>, seed {split.seed}. Validation: {e(run.folds[0].val_description or '')}</p>
<h2>Preprocessing</h2><pre>{e(json.dumps(run.snapshot['preprocessing_config'], indent=1))}</pre>
<h2>Architecture</h2><p>{rep.total_params} parameters, receptive field {rep.receptive_field_samples} samples.</p>
<table><tr><th>Layer</th><th>Type</th><th>Input</th><th>Output</th><th>Params</th></tr>{rows}</table>
<h2>Training</h2><pre>{e(json.dumps(run.snapshot['training_config'], indent=1))}</pre>
<p>Device: {e(run.device or '')}. Selected epochs: {', '.join(str(f.info.get('selected_epoch')) for f in run.folds)}.</p>
{img('plots/training_curves.png')}
<h2>Test results</h2>{metrics_html}
<h2>Software</h2><p>{e(sw)}</p></body></html>"""
    return doc.encode()


def _trial_provenance(exp: ExperimentRecord, run) -> bytes:
    """One row per candidate trial: identity, source, preprocessing outcome and partition in every fold."""
    from ..data.store import ContinuousData
    from .experiments import epochset, pool_trials, sources_for

    trials, _ = pool_trials(exp)
    src_files: dict[str, str] = {}
    for prefix, ds in sources_for(exp):
        if isinstance(ds, ContinuousData):
            for r in ds.recordings:
                if r.recording_id:
                    src_files[prefix + r.recording_id] = r.source_file or r.name
        else:
            for rid, f in (ds.source_files or {}).items():
                src_files[prefix + rid] = f
    rows_per: dict[str, int] = {}
    rejected: dict[str, str] = {}
    if exp.preprocessing_summary and exp.preprocessing_config:
        es, report, _ = epochset(exp.preprocessing_config, exp)
        for p in es.parents:
            rows_per[str(p)] = rows_per.get(str(p), 0) + 1
        for r in report["rejection"]["rejected"]:
            rejected.setdefault(r["trial_id"].split("#w")[0], r["reason"])
    folds: list[dict[str, str]] = []
    if run is not None:
        for f in run.folds:
            pf = run_dir(exp.id, run.id) / f"fold{f.fold}" / "partition.json"
            if pf.exists():
                part = json.loads(pf.read_text())
                folds.append({**{t: "train" for t in part["fit"]}, **{t: "validation" for t in part["val"]},
                              **{t: "test" for t in part["test"]}})
    elif exp.split_fingerprint:
        sp = load_split(exp)
        for f in sp.folds:
            folds.append({**{t: "train" for t in f.train}, **{t: "test" for t in f.test}})
    sfreq = {prefix: ds.sfreq for prefix, ds in sources_for(exp)}
    header = ["trial_id", "dataset", "participant", "session", "run", "recording_id", "source_file", "label",
              "onset_s", "duration_s", "preprocessing", "model_rows", *[f"fold{k + 1}" for k in range(len(folds))]]
    out = [header]
    for t in trials:
        prefix = t.trial_id.split("/", 1)[0] + "/" if "/" in t.trial_id else ""
        rid = prefix + (t.recording_id or "")
        status = ("kept" if t.trial_id in rows_per else (f"rejected: {rejected[t.trial_id]}" if t.trial_id in rejected
                                                         else "excluded (class not selected)")) \
            if exp.preprocessing_summary else "not preprocessed"
        onset = "" if t.sample is None else round(t.sample / sfreq.get(prefix, 1.0), 4)
        out.append([t.trial_id, prefix.rstrip("/") or "dataset", t.group or "", t.session or "", t.run or "",
                    t.recording_id or "", src_files.get(rid, ""), t.label, onset, round(t.duration_s, 4), status,
                    rows_per.get(t.trial_id, 0), *[f.get(t.trial_id, "") for f in folds]])
    return _csv(out)


def build_export(exp: ExperimentRecord, items: list[str]) -> dict:
    unknown = sorted(set(items) - set(EXPORT_ITEMS))
    if unknown:
        raise ValueError(f"Unknown export item(s): {', '.join(unknown)}")
    run = active_run(exp)
    caps = capabilities(exp, run)
    files: dict[str, bytes] = {}
    provenance: dict[str, str] = {}
    synthetic = any(repo.get_dataset(i).synthetic for i in (exp.dataset_id, exp.test_dataset_id) if i)
    snap = run.snapshot if run else None
    needs_run = {"metrics", "predictions", "history", "confusion", "plots", "checkpoints", "report"}
    if set(items) & needs_run and (run is None or run.status != "completed"):
        raise WorkflowError("These exports need a completed training run.")
    if set(items) & {"metrics", "predictions", "confusion"} and not caps["has_results"]:
        raise WorkflowError("Metrics, predictions and confusion matrices need a completed evaluation.")
    if "model_config" in items:
        cfg = snap["architecture_config"] if snap else (exp.architecture.model_dump() if exp.architecture else None)
        if cfg is None:
            raise WorkflowError("No architecture configured.")
        files["config/model_config.json"] = json.dumps({
            "architecture": cfg, "fingerprint": snap["architecture_fingerprint"] if snap else None,
            "input_convention": "[batch, 1, channels, time_samples]"}, indent=1).encode()
        provenance["config/model_config.json"] = "run snapshot" if snap else "current configuration"
    if "training_config" in items:
        files["config/training_config.json"] = json.dumps(snap["training_config"] if snap else
                                                          exp.training_config.model_dump(), indent=1).encode()
        provenance["config/training_config.json"] = "run snapshot" if snap else "current configuration"
    if "preprocessing_config" in items:
        if not exp.preprocessing_config:
            raise WorkflowError("No preprocessing configured.")
        files["config/preprocessing_config.json"] = json.dumps({
            "config": snap["preprocessing_config"] if snap else exp.preprocessing_config.model_dump(),
            "fingerprint": snap["preprocessing_fingerprint"] if snap else exp.preprocessing_fingerprint,
            "processing_report": (exp.preprocessing_summary or {}).get("report"),
            "learned_stage_fit_reports": [f.transform_report for f in run.folds] if run else None},
            indent=1, default=str).encode()
        provenance["config/preprocessing_config.json"] = "experiment preprocessing record"
    if "split_manifest" in items:
        sp = load_split(exp, snap["split_fingerprint"] if snap else None)
        fold_parts = []
        if run:
            for f in run.folds:
                pf = run_dir(exp.id, run.id) / f"fold{f.fold}" / "partition.json"
                if pf.exists():
                    fold_parts.append({"fold": f.fold, **json.loads(pf.read_text())})
        files["splits/split_manifest.json"] = json.dumps({
            "protocol": sp.protocol, "question": sp.question, "strategy": sp.strategy, "seed": sp.seed,
            "fingerprint": sp.fingerprint, "config": sp.config.model_dump(),
            "folds": [f.model_dump() for f in sp.folds], "train_validation_test_per_fold": fold_parts}, indent=1).encode()
        provenance["splits/split_manifest.json"] = f"split {sp.fingerprint}"
    if "dataset_manifest" in items:
        for did in [i for i in (exp.dataset_id, exp.test_dataset_id) if i]:
            rec = repo.get_dataset(did)
            files[f"dataset/{did}_manifest.json"] = json.dumps({
                "dataset_id": did, "name": rec.name, "fingerprint": rec.fingerprint, "synthetic": rec.synthetic,
                "format": rec.format, "structure": rec.structure, "source": rec.source, "files": rec.files,
                "summary": {k: rec.summary.get(k) for k in ("n_recordings", "n_trials", "n_participants",
                                                           "participants", "classes", "class_distribution", "sfreq",
                                                           "ch_names", "ch_types", "unit")},
                "manifest": rec.manifest}, indent=1, default=str).encode()
            provenance[f"dataset/{did}_manifest.json"] = f"dataset record {did} (fingerprint {rec.fingerprint[:12]})"
    if "trial_provenance" in items:
        files["splits/trial_provenance.csv"] = _trial_provenance(exp, run)
        provenance["splits/trial_provenance.csv"] = (f"split {exp.split_fingerprint}, preprocessing "
                                                     f"{exp.preprocessing_det_fingerprint}" + (f", run {run.id}" if run
                                                                                               else ""))
    ev = run.evaluation if run else None
    if "metrics" in items:
        files["results/metrics.json"] = json.dumps(ev, indent=1).encode()
        p = ev["test"]["pooled"]
        rows = [["scope", "metric", "value"]]
        for k in ("accuracy", "balanced_accuracy", "cohen_kappa", "roc_auc_ovr_macro", "average_precision_macro"):
            rows.append(["test_pooled", k, "" if p.get(k) is None else p[k]])
        for avg, vals in p["averages"].items():
            for k, v in vals.items():
                rows.append(["test_pooled", f"{k}_{avg}", "" if v is None else v])
        for c in p["per_class"]:
            for k in ("precision", "recall", "f1", "support"):
                rows.append([f"test_class:{c['class']}", k, "" if c[k] is None else c[k]])
        for vf in ev["validation"]:
            if vf["metrics"]:
                rows.append([f"validation_fold{vf['fold'] + 1}", "balanced_accuracy",
                             vf["metrics"]["balanced_accuracy"]])
        files["results/metrics.csv"] = _csv(rows)
        provenance["results/metrics.json"] = provenance["results/metrics.csv"] = f"evaluation of run {run.id}"
    if "predictions" in items:
        src = run_dir(exp.id, run.id) / "evaluation" / "predictions.csv"
        files["results/predictions.csv"] = src.read_bytes()
        provenance["results/predictions.csv"] = f"evaluation of run {run.id}"
    if "confusion" in items:
        files["results/confusion_matrix.csv"] = (run_dir(exp.id, run.id) / "evaluation" /
                                                 "confusion_matrix.csv").read_bytes()
        provenance["results/confusion_matrix.csv"] = f"evaluation of run {run.id}"
    if "history" in items:
        rows = [["fold", "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_balanced_accuracy", "lr",
                 "elapsed_s"]]
        for f in run.folds:
            for h in f.history:
                rows.append([f.fold + 1, h["epoch"], h["train_loss"], h["train_acc"], h.get("val_loss", ""),
                             h.get("val_acc", ""), h.get("val_balanced_accuracy", ""), h["lr"], h["elapsed_s"]])
        files["results/training_history.csv"] = _csv(rows)
        files["results/training_history.json"] = json.dumps(
            [{"fold": f.fold, "history": f.history, "info": f.info} for f in run.folds], indent=1).encode()
        provenance["results/training_history.csv"] = provenance["results/training_history.json"] = f"run {run.id}"
    plots = _plots(run, ev) if (run and ({"plots", "report"} & set(items))) else {}
    if "plots" in items:
        for k, v in plots.items():
            files[k] = v
            provenance[k] = f"rendered from run {run.id}"
    if "checkpoints" in items:
        for f in run.folds:
            art = repo.get_artifact(f.checkpoint_id)
            files[f"checkpoints/fold{f.fold + 1}_{f.checkpoint_id}.pt"] = (settings.data_dir / art.storage_ref).read_bytes()
            provenance[f"checkpoints/fold{f.fold + 1}_{f.checkpoint_id}.pt"] = f"checkpoint {f.checkpoint_id}"
    if "report" in items:
        files["report.html"] = _report(exp, run, plots, synthetic)
        provenance["report.html"] = f"generated from run {run.id}"
    manifest = {
        "export_id": None, "created_at": now_iso(), "experiment_id": exp.id, "experiment_name": exp.name,
        "run_id": run.id if run else None, "run_fingerprint": run.fingerprint if run else None,
        "state": exp.state.value, "synthetic_data": synthetic, "dataset_ids": [i for i in (exp.dataset_id,
                                                                                         exp.test_dataset_id) if i],
        "dataset_included": False,
        "note": "Raw EEG data are not included in experiment bundles; export datasets explicitly.",
        "files": [{"path": p, "sha256": _sha(b), "bytes": len(b), "provenance": provenance.get(p, "")}
                  for p, b in sorted(files.items())],
    }
    export_id = repo.new_id("export")
    manifest["export_id"] = export_id
    out = settings.data_dir / "exports" / f"{export_id}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p, b in files.items():
            z.writestr(p, b)
        z.writestr("manifest.json", json.dumps(manifest, indent=1))
    rec = {"id": export_id, "experiment_id": exp.id, "items": items, "manifest": manifest,
           "size_bytes": out.stat().st_size, "filename": f"{exp.id}_{export_id}.zip"}
    repo.save_export(exp.id, export_id, rec)
    return rec


def export_path(export_id: str) -> Path:
    p = settings.data_dir / "exports" / f"{export_id}.zip"
    if not p.exists():
        raise FileNotFoundError(export_id)
    return p
