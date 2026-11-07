"""Evaluation protocols: hold-out (trial- or group-level), separate datasets and K-fold CV.

Splits are computed over *candidate trial IDs* (epochs or event markers) so that the
assignment is fixed before preprocessing and survives epoch rejection. Group-level
strategies are verified in the backend to have zero participant overlap.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold, StratifiedGroupKFold, StratifiedKFold,
                                     train_test_split)

from ..data.store import CandidateTrial


class SplitError(ValueError):
    """Raised when a split configuration cannot be satisfied safely."""


class SplitConfig(BaseModel):
    workflow: Literal["single", "separate"] = "single"
    strategy: Literal["random", "group", "kfold", "separate"] = "random"
    test_ratio: float = Field(0.2, gt=0.0, lt=1.0)
    stratify: bool = True
    seed: int = 42
    test_groups: list[str] | None = None
    n_test_groups: int | None = Field(None, ge=1)
    kfold_mode: Literal["trial", "stratified_trial", "group", "stratified_group"] = "stratified_trial"
    n_folds: int = Field(5, ge=2, le=20)


class Fold(BaseModel):
    index: int
    train: list[str]
    test: list[str]


class SplitAssignment(BaseModel):
    strategy: str
    protocol: str
    question: str
    seed: int
    config: SplitConfig
    folds: list[Fold]
    n_candidates: int
    fingerprint: str
    warnings: list[str] = []


QUESTION = {
    "random": "Trial-wise hold-out: how well does the model classify new trials from the same participants and "
              "sessions it was trained on? Trials of one participant can appear in both partitions.",
    "group": "Participant-independent hold-out: how well does the model generalise to participants it has never "
             "seen? No participant contributes trials to both partitions.",
    "kfold_trial": "Trial-wise K-fold cross-validation: average within-participant performance over K disjoint "
                   "test folds. Participants appear in training and test folds.",
    "kfold_group": "Group-wise K-fold cross-validation: average cross-participant performance; each participant's "
                   "trials are in exactly one test fold.",
    "separate": "Separate datasets: the test dataset is evaluated as provided. Whether this measures within- or "
                "cross-participant generalisation depends on whether participants/sessions overlap.",
}


def _stratify_ok(labels: list[str], n_test: int, n_train: int) -> tuple[bool, str]:
    c = Counter(labels)
    if len(c) < 2:
        return False, "only one class present"
    if min(c.values()) < 2:
        return False, f"class '{min(c, key=c.get)}' has fewer than 2 trials"
    if n_test < len(c) or n_train < len(c):
        return False, "a partition would be smaller than the number of classes"
    return True, ""


def _fingerprint(folds: list[Fold]) -> str:
    payload = json.dumps([[sorted(f.train), sorted(f.test)] for f in folds])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def make_split(cfg: SplitConfig, trials: list[CandidateTrial], sources: list[str] | None = None) -> SplitAssignment:
    """Create a split assignment. ``sources`` marks trials as 'train'/'test' for the separate-dataset workflow."""
    if not trials:
        raise SplitError("The dataset has no candidate trials (no epochs or event markers).")
    ids = [t.trial_id for t in trials]
    labels = [t.label for t in trials]
    groups = [t.group for t in trials]
    idx = np.arange(len(trials))
    warnings: list[str] = []
    folds: list[Fold] = []
    strategy = cfg.strategy

    def need_groups() -> None:
        missing = sum(g is None for g in groups)
        if missing:
            raise SplitError(f"{missing} trial(s) have no participant/group ID; group-level splits need an ID for "
                             "every trial.")
        if len(set(groups)) < 2:
            raise SplitError("Group-level splits need at least two participants/groups.")

    if cfg.workflow == "separate" or strategy == "separate":
        strategy = "separate"
        if sources is None:
            raise SplitError("The separate-dataset workflow needs a training and a test dataset.")
        tr = [i for i, s in zip(ids, sources) if s == "train"]
        te = [i for i, s in zip(ids, sources) if s == "test"]
        if not tr or not te:
            raise SplitError("Both the training and the test dataset must contain trials.")
        folds = [Fold(index=0, train=tr, test=te)]
        protocol = "Separate training and test datasets"
        question = QUESTION["separate"]
    elif strategy == "random":
        n_test = int(round(len(ids) * cfg.test_ratio))
        if n_test < 1 or n_test >= len(ids):
            raise SplitError("Test ratio leaves an empty partition.")
        strat = None
        if cfg.stratify:
            ok, why = _stratify_ok(labels, n_test, len(ids) - n_test)
            if ok:
                strat = labels
            else:
                warnings.append(f"Stratification disabled: {why}.")
        tr_idx, te_idx = train_test_split(idx, test_size=n_test, random_state=cfg.seed, shuffle=True, stratify=strat)
        folds = [Fold(index=0, train=[ids[i] for i in sorted(tr_idx)], test=[ids[i] for i in sorted(te_idx)])]
        protocol = f"Random trial-level hold-out ({1 - cfg.test_ratio:.0%}/{cfg.test_ratio:.0%}" \
                   f"{', stratified' if strat else ''}, seed {cfg.seed})"
        question = QUESTION["random"]
    elif strategy == "group":
        need_groups()
        uniq = sorted(set(groups))
        if cfg.test_groups:
            unknown = sorted(set(cfg.test_groups) - set(uniq))
            if unknown:
                raise SplitError(f"Unknown participant/group IDs: {', '.join(unknown)}.")
            test_groups = sorted(set(cfg.test_groups))
        else:
            n_test_g = cfg.n_test_groups or max(1, int(round(len(uniq) * cfg.test_ratio)))
            if n_test_g >= len(uniq):
                raise SplitError(f"Cannot hold out {n_test_g} of {len(uniq)} groups and keep any for training.")
            rng = np.random.default_rng(cfg.seed)
            test_groups = sorted(rng.choice(uniq, size=n_test_g, replace=False).tolist())
        if set(test_groups) == set(uniq):
            raise SplitError("All participants are in the test set; at least one must remain for training.")
        tr = [i for i, g in zip(ids, groups) if g not in test_groups]
        te = [i for i, g in zip(ids, groups) if g in test_groups]
        folds = [Fold(index=0, train=tr, test=te)]
        protocol = f"Participant-level hold-out: test = {', '.join(test_groups)} " \
                   f"({len(uniq) - len(test_groups)} train / {len(test_groups)} test participants)"
        question = QUESTION["group"]
    elif strategy == "kfold":
        mode = cfg.kfold_mode
        k = cfg.n_folds
        if mode in ("group", "stratified_group"):
            need_groups()
            n_g = len(set(groups))
            if n_g < k:
                raise SplitError(f"Group-wise {k}-fold needs at least {k} participants; the dataset has {n_g}.")
            if mode == "stratified_group":
                splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=cfg.seed)
                it = splitter.split(idx, labels, groups)
            else:
                try:
                    splitter = GroupKFold(n_splits=k, shuffle=True, random_state=cfg.seed)
                except TypeError:  # older scikit-learn without shuffle support
                    splitter = GroupKFold(n_splits=k)
                it = splitter.split(idx, labels, groups)
            question = QUESTION["kfold_group"]
        else:
            if len(ids) < k:
                raise SplitError(f"{k}-fold needs at least {k} trials.")
            if mode == "stratified_trial":
                c = Counter(labels)
                if min(c.values()) < k:
                    raise SplitError(f"Stratified {k}-fold needs at least {k} trials per class; "
                                     f"'{min(c, key=c.get)}' has {min(c.values())}. Use fewer folds or "
                                     "non-stratified K-fold.")
                it = StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg.seed).split(idx, labels)
            else:
                it = KFold(n_splits=k, shuffle=True, random_state=cfg.seed).split(idx)
            question = QUESTION["kfold_trial"]
        for f, (tr_idx, te_idx) in enumerate(it):
            folds.append(Fold(index=f, train=[ids[i] for i in sorted(tr_idx)], test=[ids[i] for i in sorted(te_idx)]))
        names = {"trial": "Trial-wise", "stratified_trial": "Stratified trial-wise", "group": "Group-wise",
                 "stratified_group": "Stratified group-wise"}
        protocol = f"{names[mode]} {k}-fold cross-validation (seed {cfg.seed})"
    else:
        raise SplitError(f"Unknown strategy {strategy}")

    assignment = SplitAssignment(strategy=strategy, protocol=protocol, question=question, seed=cfg.seed, config=cfg,
                                 folds=folds, n_candidates=len(ids), fingerprint=_fingerprint(folds),
                                 warnings=warnings)
    report = integrity_report(assignment, trials)
    if not report["ok"]:
        failing = [c["detail"] for c in report["checks"] if c["status"] == "fail"]
        raise SplitError("Split integrity check failed: " + "; ".join(failing))
    return assignment


def integrity_report(assignment: SplitAssignment, trials: list[CandidateTrial],
                     kept_ids: set[str] | None = None) -> dict:
    """Backend integrity checks. ``kept_ids`` restricts the report to trials that survived preprocessing."""
    by_id = {t.trial_id: t for t in trials}
    group_level = assignment.strategy == "group" or (
        assignment.strategy == "kfold" and assignment.config.kfold_mode in ("group", "stratified_group"))
    classes = sorted({t.label for t in trials if kept_ids is None or t.trial_id in kept_ids})
    checks: list[dict] = []
    folds_out = []
    for fold in assignment.folds:
        tr = [i for i in fold.train if kept_ids is None or i in kept_ids]
        te = [i for i in fold.test if kept_ids is None or i in kept_ids]
        unknown = [i for i in fold.train + fold.test if i not in by_id]
        dup = set(tr) & set(te)
        g_tr = {by_id[i].group for i in tr if i in by_id and by_id[i].group}
        g_te = {by_id[i].group for i in te if i in by_id and by_id[i].group}
        s_tr = {(by_id[i].group, by_id[i].session) for i in tr if i in by_id and by_id[i].session}
        s_te = {(by_id[i].group, by_id[i].session) for i in te if i in by_id and by_id[i].session}
        g_overlap = sorted(g_tr & g_te)
        s_overlap = sorted(f"{g or '?'}/{s}" for g, s in (s_tr & s_te))
        cls_tr = Counter(by_id[i].label for i in tr if i in by_id)
        cls_te = Counter(by_id[i].label for i in te if i in by_id)
        folds_out.append({
            "index": fold.index, "n_train": len(tr), "n_test": len(te),
            "train_groups": sorted(g_tr), "test_groups": sorted(g_te),
            "group_overlap": g_overlap, "session_overlap": s_overlap,
            "class_counts": {"train": {c: cls_tr.get(c, 0) for c in classes},
                             "test": {c: cls_te.get(c, 0) for c in classes}},
        })
        tag = f"Fold {fold.index + 1}: " if len(assignment.folds) > 1 else ""
        checks.append({"id": f"disjoint_{fold.index}", "label": f"{tag}train/test trials disjoint",
                       "status": "fail" if dup else "pass",
                       "detail": f"{tag}{len(dup)} trial(s) in both partitions" if dup else f"{tag}no shared trials"})
        if unknown:
            checks.append({"id": f"unknown_{fold.index}", "label": f"{tag}trial IDs exist", "status": "fail",
                           "detail": f"{tag}{len(unknown)} unknown trial ID(s)"})
        if group_level:
            checks.append({
                "id": f"group_overlap_{fold.index}", "label": f"{tag}zero participant overlap",
                "status": "fail" if g_overlap else "pass",
                "detail": f"{tag}participants in both partitions: {', '.join(g_overlap)}" if g_overlap
                else f"{tag}{len(g_tr)} train vs {len(g_te)} test participants, no overlap"})
        elif g_tr or g_te:
            checks.append({
                "id": f"group_overlap_{fold.index}", "label": f"{tag}participant overlap (expected for trial-wise)",
                "status": "warn" if g_overlap else "pass",
                "detail": f"{tag}{len(g_overlap)} participant(s) contribute to train and test - this estimates "
                          "within-participant, not cross-participant, performance" if g_overlap
                else f"{tag}no participant overlap"})
        if s_overlap:
            checks.append({"id": f"session_overlap_{fold.index}", "label": f"{tag}session overlap", "status": "warn",
                           "detail": f"{tag}{len(s_overlap)} recording session(s) appear in both partitions"})
        missing_te = [c for c in classes if cls_te.get(c, 0) == 0]
        missing_tr = [c for c in classes if cls_tr.get(c, 0) == 0]
        if missing_tr:
            checks.append({"id": f"train_classes_{fold.index}", "label": f"{tag}all classes in training",
                           "status": "fail", "detail": f"{tag}training set lacks {', '.join(missing_tr)}"})
        if missing_te:
            checks.append({"id": f"test_classes_{fold.index}", "label": f"{tag}all classes in test", "status": "warn",
                           "detail": f"{tag}test set lacks {', '.join(missing_te)}; per-class metrics for those "
                                     "classes will be undefined"})
        if not tr or not te:
            checks.append({"id": f"empty_{fold.index}", "label": f"{tag}non-empty partitions", "status": "fail",
                           "detail": f"{tag}a partition is empty"})
    if len(assignment.folds) > 1:
        all_test = [i for f in assignment.folds for i in f.test]
        dup_test = len(all_test) - len(set(all_test))
        checks.append({"id": "folds_disjoint", "label": "test folds are disjoint",
                       "status": "fail" if dup_test else "pass",
                       "detail": f"{dup_test} trial(s) appear in more than one test fold" if dup_test
                       else "each trial is tested exactly once"})
    checks.append({"id": "test_isolation", "label": "test data excluded from model selection", "status": "pass",
                   "detail": "Early stopping and model selection use a validation subset carved from the training "
                             "partition only; learned preprocessing (ICA, normalisation) is fitted on training data "
                             "only, separately per fold."})
    return {"ok": not any(c["status"] == "fail" for c in checks), "checks": checks, "folds": folds_out,
            "classes": classes, "group_level": group_level, "protocol": assignment.protocol,
            "question": assignment.question, "seed": assignment.seed, "fingerprint": assignment.fingerprint}


def check_group_isolation(assignment: SplitAssignment, trials: list[CandidateTrial]) -> None:
    """Raise if any fold of a group-level split shares participants between train and test."""
    report = integrity_report(assignment, trials)
    if report["group_level"]:
        bad = [f for f in report["folds"] if f["group_overlap"]]
        if bad:
            raise SplitError(f"Participant overlap in fold(s) {[f['index'] + 1 for f in bad]}: "
                             f"{bad[0]['group_overlap']}")


def validation_split(train_ids: list[str], trials_by_id: dict[str, CandidateTrial], mode: str, fraction: float,
                     seed: int, group_level: bool) -> tuple[list[str], list[str], str]:
    """Carve a validation subset out of the *training* partition (never the test partition)."""
    if mode == "none":
        return list(train_ids), [], "No validation set (early stopping and model selection disabled)."
    labels = [trials_by_id[i].label for i in train_ids]
    groups = [trials_by_id[i].group for i in train_ids]
    idx = np.arange(len(train_ids))
    if mode == "auto":
        mode = "group" if group_level and None not in groups and len(set(groups)) >= 3 else "stratified"
    if mode == "group":
        if None in groups or len(set(groups)) < 2:
            raise SplitError("Group-wise validation needs participant IDs and at least two training participants.")
        gss = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
        fit_idx, val_idx = next(gss.split(idx, labels, groups))
        desc = f"Group-wise validation: {len(set(groups[i] for i in val_idx))} training participant(s) held out " \
               f"({fraction:.0%} of training groups)."
    else:
        n_val = max(1, int(round(len(train_ids) * fraction)))
        strat = None
        if mode == "stratified":
            ok, _ = _stratify_ok(labels, n_val, len(train_ids) - n_val)
            strat = labels if ok else None
        fit_idx, val_idx = train_test_split(idx, test_size=n_val, random_state=seed, shuffle=True, stratify=strat)
        desc = f"{'Stratified r' if strat else 'R'}andom validation subset: {fraction:.0%} of the training partition."
    return [train_ids[i] for i in sorted(fit_idx)], [train_ids[i] for i in sorted(val_idx)], desc
