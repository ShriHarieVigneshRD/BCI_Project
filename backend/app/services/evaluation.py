"""Final evaluation on the held-out test partition(s), persisted as artifacts."""

from __future__ import annotations

import csv
import json

import numpy as np

from ..core.metrics import classification_metrics
from ..core.tracing import run_trace
from ..lifecycle import State
from ..schemas import ExperimentRecord, now_iso
from . import repo
from .artifacts import load_bundle
from .experiments import WorkflowError, ensure_learned_ready
from .jobs import register_artifact, run_dir

S = State


def _predict(b, parent_ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Predict every row (epoch or window) whose candidate trial is in ``parent_ids``."""
    rows = b.es.rows_for(parent_ids)
    if not rows:
        return np.zeros((0,), int), np.zeros((0,), int), np.zeros((0, len(b.class_names))), []
    x = b.transform.transform(b.es.X, rows)
    probs = np.concatenate([run_trace(b.model, x[i:i + 256], layers=[]).probabilities for i in range(0, len(x), 256)])
    return b.es.y[rows], probs.argmax(1), probs, rows


def _group_metrics(rows: list[dict], key: str, class_names: list[str]) -> list[dict]:
    """Test metrics per participant (or per other provenance key)."""
    lut = {c: i for i, c in enumerate(class_names)}
    out = []
    for g in sorted({r[key] or "(none)" for r in rows}):
        sel = [r for r in rows if (r[key] or "(none)") == g]
        y = np.array([lut[r["true"]] for r in sel])
        p = np.array([lut[r["predicted"]] for r in sel])
        m = classification_metrics(y, p, None, class_names)
        out.append({key: None if g == "(none)" else g, "n": len(sel), "accuracy": m["accuracy"],
                    "balanced_accuracy": m["balanced_accuracy"], "class_counts": {c: int((y == lut[c]).sum())
                                                                                  for c in class_names}})
    return out


def _aggregate_by_trial(rows: list[dict], class_names: list[str]) -> dict | None:
    """When trials were cut into windows: average window probabilities per trial (block)."""
    if len({r["parent_trial"] for r in rows}) == len(rows):
        return None
    lut = {c: i for i, c in enumerate(class_names)}
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["parent_trial"], []).append(r)
    y, p, pr = [], [], []
    for _t, sel in by.items():
        prob = np.mean([s["probabilities"] for s in sel], axis=0)
        y.append(lut[sel[0]["true"]])
        p.append(int(prob.argmax()))
        pr.append(prob)
    metrics = classification_metrics(np.array(y), np.array(p), np.array(pr), class_names)
    return {"metrics": metrics, "n_trials": len(by), "n_windows": len(rows),
            "note": "Each trial (segment) is classified by averaging the probabilities of its windows."}


def evaluate(exp: ExperimentRecord) -> dict:
    run = ensure_learned_ready(exp)
    if exp.state == S.EVALUATION_RUNNING:
        raise WorkflowError("Evaluation is already running.")
    repo.transition(exp, S.EVALUATION_RUNNING, f"evaluating run {run.id}")
    try:
        rows, per_fold = [], []
        y_all, p_all, pr_all = [], [], []
        class_names = None
        for f in run.folds:
            b = load_bundle(run, f.fold)
            class_names = b.class_names
            part = b.partition
            fold_res = {"fold": f.fold, "checkpoint_id": b.checkpoint_id, "selected_epoch": b.selected_epoch}
            for key, label in (("fit", "training"), ("val", "validation"), ("test", "test")):
                y, p, pr, idx_rows = _predict(b, part[key])
                fold_res[label] = classification_metrics(y, p, pr, class_names) if len(y) else None
                if key == "test":
                    y_all.append(y), p_all.append(p), pr_all.append(pr)
                    es = b.es
                    held_out = set(part["test"])
                    for i, yt, yp, prob in zip(idx_rows, y, p, pr):
                        if str(es.parents[i]) not in held_out:
                            raise RuntimeError("Prediction provenance error: row outside the test partition.")
                        rows.append({"trial_id": str(es.trial_ids[i]), "parent_trial": str(es.parents[i]),
                                     "fold": f.fold, "group": str(es.groups[i]) or None,
                                     "session": str(es.sessions[i]) or None, "run": str(es.runs[i]) or None,
                                     "recording_id": str(es.recordings[i]) or None,
                                     "true": class_names[int(yt)], "predicted": class_names[int(yp)],
                                     "correct": bool(yt == yp), "probabilities": [round(float(v), 6) for v in prob],
                                     "checkpoint_id": b.checkpoint_id})
            per_fold.append(fold_res)
        yc, pc, prc = np.concatenate(y_all), np.concatenate(p_all), np.concatenate(pr_all)
        pooled = classification_metrics(yc, pc, prc, class_names)
        fold_summary = None
        if len(per_fold) > 1:
            fold_summary = {}
            for m in ("accuracy", "balanced_accuracy"):
                vals = [pf["test"][m] for pf in per_fold if pf["test"] and pf["test"].get(m) is not None]
                fold_summary[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))
                                   if len(vals) > 1 else 0.0, "values": vals}
            vals = [pf["test"]["averages"]["macro"]["f1"] for pf in per_fold
                    if pf["test"] and pf["test"]["averages"]["macro"]["f1"] is not None]
            fold_summary["macro_f1"] = {"mean": float(np.mean(vals)) if vals else None,
                                        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                                        "values": vals}
        rd = run_dir(exp.id, run.id) / "evaluation"
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "predictions.json").write_text(json.dumps(rows))
        with open(rd / "predictions.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["trial_id", "parent_trial", "fold", "participant", "session", "run", "recording_id",
                        "true_label", "predicted_label", "correct", *[f"p_{c}" for c in class_names],
                        "checkpoint_id"])
            for r in rows:
                w.writerow([r["trial_id"], r["parent_trial"], r["fold"] + 1, r["group"] or "", r["session"] or "",
                            r["run"] or "", r["recording_id"] or "", r["true"], r["predicted"], int(r["correct"]),
                            *r["probabilities"], r["checkpoint_id"]])
        result = {
            "created_at": now_iso(), "run_id": run.id, "n_test": int(len(yc)), "class_names": class_names,
            "protocol": exp.split_config.strategy if exp.split_config else None,
            "test": {"pooled": pooled, "per_fold": [{"fold": pf["fold"], "metrics": pf["test"]} for pf in per_fold],
                     "fold_summary": fold_summary,
                     "by_participant": _group_metrics(rows, "group", class_names),
                     "by_trial": _aggregate_by_trial(rows, class_names),
                     "participants": sorted({r["group"] for r in rows if r["group"]}),
                     "note": "Out-of-fold predictions pooled over all test folds." if len(per_fold) > 1 else
                     "Held-out test partition."},
            "validation": [{"fold": pf["fold"], "metrics": pf["validation"]} for pf in per_fold],
            "training": [{"fold": pf["fold"], "metrics": pf["training"]} for pf in per_fold],
            "checkpoints": [{"fold": pf["fold"], "checkpoint_id": pf["checkpoint_id"],
                             "selected_epoch": pf["selected_epoch"]} for pf in per_fold],
            "labels": {"training": "Training subset (fitted data; optimistic)",
                       "validation": "Validation subset carved from training data (used for model selection)",
                       "test": "Held-out test data (never used for training, early stopping or selection)"},
        }
        (rd / "metrics.json").write_text(json.dumps(result))
        with open(rd / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["true \\ predicted", *class_names])
            for c, row in zip(class_names, pooled["confusion_matrix"]):
                w.writerow([c, *row])
        ids = []
        for name, typ in (("predictions.csv", "predictions"), ("metrics.json", "metrics"),
                          ("confusion_matrix.csv", "confusion_matrix")):
            a = register_artifact(exp.id, run.id, typ, rd / name, config_fp=run.fingerprint,
                                  checkpoint_id=",".join(pf["checkpoint_id"] for pf in per_fold),
                                  source_ref=f"{len(yc)} test trials")
            ids.append(a.id)
        result["artifact_ids"] = ids
        run = repo.get_run(run.id)
        run.evaluation = {k: v for k, v in result.items()}
        repo.save_run(run)
        exp = repo.get_experiment(exp.id)
        repo.transition(exp, S.EVALUATION_COMPLETED, f"{len(yc)} test trials evaluated")
        return result
    except Exception:
        exp = repo.get_experiment(exp.id)
        if exp.state == S.EVALUATION_RUNNING:
            repo.transition(exp, S.ARTIFACTS_READY, "evaluation failed")
        raise


def results(exp: ExperimentRecord) -> dict:
    run = ensure_learned_ready(exp)
    if exp.state != S.EVALUATION_COMPLETED or not run.evaluation:
        raise WorkflowError("Evaluation has not been run for this model. Run 'Evaluate model' first; validation "
                            "metrics are never substituted for test metrics.")
    return {
        "evaluation": run.evaluation,
        "history": [{"fold": f.fold, "history": f.history, "info": f.info, "val_description": f.val_description}
                    for f in run.folds],
        "metadata": {"run_id": run.id, "device": run.device, "started_at": run.started_at,
                     "finished_at": run.finished_at, "seed": run.snapshot.get("seed"),
                     "split_fingerprint": run.snapshot.get("split_fingerprint"),
                     "preprocessing_fingerprint": run.snapshot.get("preprocessing_fingerprint"),
                     "architecture_fingerprint": run.snapshot.get("architecture_fingerprint"),
                     "software": run.software, "training_config": run.snapshot.get("training_config")},
    }


def predictions(exp: ExperimentRecord) -> list[dict]:
    run = ensure_learned_ready(exp)
    if exp.state != S.EVALUATION_COMPLETED or not run.evaluation:
        raise WorkflowError("No prediction artifacts exist yet; evaluate the model first.")
    path = run_dir(exp.id, run.id) / "evaluation" / "predictions.json"
    if not path.exists():
        raise WorkflowError("Prediction artifact is missing; re-run the evaluation.")
    return json.loads(path.read_text())
