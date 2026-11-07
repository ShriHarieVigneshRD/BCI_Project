"""Classification metrics with explicit handling of undefined values and absent classes."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, precision_recall_fscore_support, roc_auc_score)
from sklearn.preprocessing import label_binarize

AVERAGING_DOC = {
    "macro": "Unweighted mean over classes (every class counts equally).",
    "weighted": "Mean over classes weighted by the number of true trials per class (support).",
    "micro": "Computed from global true/false positive counts; equals accuracy for single-label multiclass tasks.",
    "balanced_accuracy": "Mean per-class recall over the classes present in the evaluated trials.",
    "roc_auc": "One-vs-rest ROC-AUC, macro-averaged (binary: positive class = second class). Only defined when "
               "every class is present among the evaluated trials.",
    "average_precision": "One-vs-rest average precision (area under the precision-recall curve), macro-averaged.",
}


def _clean(v) -> float | None:
    if v is None:
        return None
    v = float(v)
    return None if math.isnan(v) or math.isinf(v) else v


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None,
                           class_names: list[str]) -> dict:
    labels = list(range(len(class_names)))
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    present = sorted(set(y_true.tolist()))
    absent = [class_names[i] for i in labels if i not in present]
    never_predicted = [class_names[i] for i in labels if i not in set(y_pred.tolist())]
    undefined: list[str] = []
    if len(y_true) == 0:
        return {"n": 0, "undefined": ["No trials were evaluated."]}

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=np.nan)
    per_class = []
    for i, c in enumerate(class_names):
        notes = []
        if i not in present:
            notes.append("absent from evaluated trials: recall/F1 undefined")
        if c in never_predicted:
            notes.append("never predicted: precision undefined")
        per_class.append({"class": c, "precision": _clean(p[i]), "recall": _clean(r[i]), "f1": _clean(f[i]),
                          "support": int(s[i]), "notes": notes})
    avgs = {}
    for avg in ("macro", "weighted", "micro"):
        ap, ar, af, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=avg,
                                                        zero_division=np.nan)
        avgs[avg] = {"precision": _clean(ap), "recall": _clean(ar), "f1": _clean(af)}
    if absent:
        undefined.append(f"Classes absent from the evaluated trials: {', '.join(absent)}. Their recall/F1 are "
                         "undefined and excluded (NaN) from macro averages.")
    if never_predicted:
        undefined.append(f"Classes never predicted: {', '.join(never_predicted)}. Their precision is undefined.")

    roc = ap_score = None
    if proba is not None and len(present) == len(labels):
        try:
            if len(labels) == 2:
                roc = roc_auc_score(y_true, proba[:, 1])
                ap_score = average_precision_score(y_true, proba[:, 1])
            else:
                roc = roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=labels)
                ap_score = average_precision_score(label_binarize(y_true, classes=labels), proba, average="macro")
        except ValueError as exc:
            undefined.append(f"ROC-AUC undefined: {exc}")
    elif proba is not None:
        undefined.append("ROC-AUC and average precision are undefined because not every class is present.")

    with np.errstate(all="ignore"):
        bacc = balanced_accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred, labels=labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "n": int(len(y_true)),
        "accuracy": _clean(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": _clean(bacc),
        "cohen_kappa": _clean(kappa),
        "chance_level": 1.0 / len(labels),
        "averages": avgs,
        "per_class": per_class,
        "roc_auc_ovr_macro": _clean(roc),
        "average_precision_macro": _clean(ap_score),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "undefined": undefined,
        "averaging_doc": AVERAGING_DOC,
    }
