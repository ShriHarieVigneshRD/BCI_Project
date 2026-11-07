"""Experiment comparison (Hyperparameter Battle Arena) with protocol-compatibility warnings."""

from __future__ import annotations

import json

from ..core.eegnet import ArchitectureConfig, compute_architecture
from . import repo
from .experiments import active_run, capabilities
from .jobs import run_dir


def _ds_name(ds_id: str) -> str:
    try:
        return repo.get_dataset(ds_id).name
    except repo.NotFound:
        return "(deleted dataset)"


def _entry(exp_id: str) -> dict:
    exp = repo.get_experiment(exp_id)
    run = active_run(exp)
    caps = capabilities(exp, run)
    arch_cfg = None
    if run is not None:
        arch_cfg = ArchitectureConfig(**run.snapshot["architecture_config"])
    elif exp.architecture is not None:
        arch_cfg = exp.architecture
    arch = None
    if arch_cfg is not None:
        rep = compute_architecture(arch_cfg)
        arch = {"config": arch_cfg.model_dump(), "valid": rep.valid, "params": rep.total_params,
                "macs": rep.total_macs, "receptive_field_samples": rep.receptive_field_samples,
                "layers": [{"name": s.name, "label": s.label, "type": s.type, "out_shape": s.out_shape,
                            "params": s.params} for s in rep.layers],
                "source": "trained run snapshot" if run is not None else "current (untrained) configuration"}
    measured = caps["can_inspect_learned"] and run is not None
    ev = run.evaluation if (run is not None and caps["has_results"]) else None
    histories = [{"fold": f.fold, "history": f.history} for f in run.folds] if measured else []
    ds_fps = run.snapshot["dataset_fingerprints"] if run else None
    return {
        "id": exp.id, "name": exp.name, "state": exp.state.value, "trained": measured,
        "evaluated": ev is not None, "stale": bool(run and run.stale), "architecture": arch,
        "training_config": (run.snapshot["training_config"] if run else exp.training_config.model_dump()),
        "preprocessing_config": (run.snapshot["preprocessing_config"] if run else
                                 (exp.preprocessing_config.model_dump() if exp.preprocessing_config else None)),
        "preprocessing_fingerprint": run.snapshot["preprocessing_fingerprint"] if run else exp.preprocessing_fingerprint,
        "split": {"strategy": exp.split_config.strategy if exp.split_config else None,
                  "config": exp.split_config.model_dump() if exp.split_config else None,
                  "fingerprint": run.snapshot["split_fingerprint"] if run else exp.split_fingerprint,
                  "seed": exp.split_config.seed if exp.split_config else None},
        "dataset_fingerprints": ds_fps, "class_names": run.snapshot["class_names"] if run else None,
        "datasets": [{"id": i, "name": _ds_name(i)} for i in (exp.dataset_id, exp.test_dataset_id) if i],
        "trial_unit": ((exp.preprocessing_summary or {}).get("report", {}).get("epoching", {}) or {}).get("mode"),
        "histories": histories,
        "train_time_s": sum(f.info.get("train_time_s", 0) for f in run.folds) if measured else None,
        "device": run.device if run else None,
        "evaluation": ({"pooled": ev["test"]["pooled"], "fold_summary": ev["test"]["fold_summary"],
                        "validation": ev["validation"]} if ev else None),
        "best_validation": max((h.get("val_balanced_accuracy") for fh in histories for h in fh["history"]
                                if h.get("val_balanced_accuracy") is not None), default=None) if measured else None,
        "artifact_status": run.artifact_status if run else None,
        "_run": run,
    }


def compare(ids: list[str]) -> dict:
    if len(ids) < 2:
        raise ValueError("Select at least two experiments.")
    entries = [_entry(i) for i in ids]
    warnings: list[str] = []

    def distinct(key):
        return {json.dumps(e[key], sort_keys=True) for e in entries if e[key] is not None}

    if len(distinct("dataset_fingerprints")) > 1:
        warnings.append("Experiments use different datasets; their metrics are not directly comparable.")
    if len(distinct("class_names")) > 1:
        warnings.append("Class mappings differ between experiments.")
    if len(distinct("preprocessing_fingerprint")) > 1:
        warnings.append("Preprocessing pipelines differ; differences in performance may come from preprocessing "
                        "rather than the architecture.")
    splits = {e["split"]["fingerprint"] for e in entries if e["split"]["fingerprint"]}
    if len(splits) > 1:
        warnings.append("Split assignments differ (not a matched comparison). Derive variants from one experiment to "
                        "reuse identical splits.")
    if len({json.dumps(e["trial_unit"]) for e in entries if e["trial_unit"]}) > 1:
        warnings.append("Epoching differs (fixed windows around events vs sliding windows within segments); "
                        "metrics refer to different units of analysis.")
    if len({e["split"]["strategy"] for e in entries}) > 1:
        warnings.append("Evaluation protocols differ (e.g. trial-wise vs participant-independent); they answer "
                        "different scientific questions.")
    untrained = [e["name"] for e in entries if not e["trained"]]
    if untrained:
        warnings.append(f"No measured results for: {', '.join(untrained)} (theoretical shapes and parameter counts "
                        "only).")
    stale = [e["name"] for e in entries if e["stale"]]
    if stale:
        warnings.append(f"Stale runs (configuration changed after training): {', '.join(stale)}.")
    compatible = not any(w.startswith(("Experiments use different datasets", "Class mappings", "Split assignments",
                                       "Evaluation protocols")) for w in warnings)
    disagreement = None
    evaluated = [e for e in entries if e["evaluated"]]
    if compatible and len(evaluated) >= 2:
        preds = {}
        for e in evaluated:
            run = e["_run"]
            p = run_dir(e["id"], run.id) / "evaluation" / "predictions.json"
            if p.exists():
                preds[e["id"]] = {r["trial_id"]: r["predicted"] for r in json.loads(p.read_text())}
        trial_sets = {frozenset(v) for v in preds.values()}
        if len(preds) >= 2 and len(trial_sets) == 1:
            ids_e = list(preds)
            pairs = []
            for i in range(len(ids_e)):
                for j in range(i + 1, len(ids_e)):
                    a, b = preds[ids_e[i]], preds[ids_e[j]]
                    diff = [t for t in a if a[t] != b[t]]
                    pairs.append({"a": ids_e[i], "b": ids_e[j], "n": len(a), "disagreements": len(diff),
                                  "rate": len(diff) / max(len(a), 1), "trials": diff[:200]})
            disagreement = {"pairs": pairs, "note": "Computed on identical test trials under matched protocols."}
        else:
            warnings.append("Prediction disagreement not computed: evaluated test trial sets differ.")
    for e in entries:
        e.pop("_run", None)
    return {"experiments": entries, "warnings": warnings, "matched_protocol": compatible,
            "disagreement": disagreement,
            "selection_rule": "Rank configurations by validation metrics. Test metrics are reported for the final "
                              "model-selection protocol only and must not be used to pick hyperparameters."}
