"""Split strategies, leakage safeguards and integrity checks (requirement 14)."""

import pytest

from app.core.splits import (Fold, SplitAssignment, SplitConfig, SplitError, check_group_isolation, integrity_report,
                             make_split, validation_split)
from app.data.store import CandidateTrial


def trials(n_groups=5, per_group=12, classes=("a", "b", "c")):
    out = []
    for g in range(n_groups):
        for i in range(per_group):
            out.append(CandidateTrial(trial_id=f"g{g}t{i}", label=classes[i % len(classes)], group=f"P{g}",
                                      session="S1"))
    return out


def test_group_split_has_zero_overlap():
    tr = trials()
    a = make_split(SplitConfig(strategy="group", test_groups=["P1", "P3"]), tr)
    by = {t.trial_id: t.group for t in tr}
    train_g = {by[i] for i in a.folds[0].train}
    test_g = {by[i] for i in a.folds[0].test}
    assert test_g == {"P1", "P3"} and not (train_g & test_g)
    rep = integrity_report(a, tr)
    assert rep["ok"] and rep["group_level"]


def test_overlapping_group_assignment_is_rejected():
    tr = trials()
    good = make_split(SplitConfig(strategy="group", test_groups=["P0"]), tr)
    # tamper: move one P0 trial into training -> participant overlap
    leaked = good.folds[0].test[0]
    bad = SplitAssignment(**{**good.model_dump(), "folds": [Fold(index=0, train=good.folds[0].train + [leaked],
                                                                  test=good.folds[0].test[1:])]})
    rep = integrity_report(bad, tr)
    assert not rep["ok"]
    assert any(c["status"] == "fail" and "participant" in c["label"] for c in rep["checks"])
    with pytest.raises(SplitError):
        check_group_isolation(bad, tr)


def test_group_split_api_rejects_unknown_or_all_groups(client, synth_epoched):
    eid = client.post("/api/experiments", json={"dataset_id": synth_epoched["id"]}).json()["id"]
    r = client.put(f"/api/experiments/{eid}/split", json={"strategy": "group", "test_groups": ["nobody"]})
    assert r.status_code == 422
    groups = client.get(f"/api/experiments/{eid}/candidates").json()["groups"]
    r = client.put(f"/api/experiments/{eid}/split", json={"strategy": "group", "test_groups": groups})
    assert r.status_code == 422


@pytest.mark.parametrize("mode", ["trial", "stratified_trial", "group", "stratified_group"])
def test_kfold_variants(mode):
    tr = trials()
    a = make_split(SplitConfig(strategy="kfold", kfold_mode=mode, n_folds=5, seed=3), tr)
    assert len(a.folds) == 5
    all_test = [i for f in a.folds for i in f.test]
    assert sorted(all_test) == sorted(t.trial_id for t in tr)  # every trial tested exactly once
    rep = integrity_report(a, tr)
    assert rep["ok"]
    if "group" in mode:
        assert all(not f["group_overlap"] for f in rep["folds"])


def test_group_kfold_needs_enough_groups():
    with pytest.raises(SplitError):
        make_split(SplitConfig(strategy="kfold", kfold_mode="group", n_folds=8), trials(n_groups=5))


def test_random_split_reports_participant_overlap_and_is_reproducible():
    tr = trials()
    a1 = make_split(SplitConfig(strategy="random", test_ratio=0.25, seed=11), tr)
    a2 = make_split(SplitConfig(strategy="random", test_ratio=0.25, seed=11), tr)
    assert a1.fingerprint == a2.fingerprint
    rep = integrity_report(a1, tr)
    assert any(c["status"] == "warn" and "participant overlap" in c["label"] for c in rep["checks"])
    assert "same participants" in a1.question


def test_stratification_falls_back_with_warning():
    # 1 test trial cannot represent 2 classes -> stratification is disabled with an explicit warning
    tr = [CandidateTrial(trial_id=f"t{i}", label="b" if i < 2 else "a", group=None, session=None) for i in range(20)]
    a = make_split(SplitConfig(strategy="random", test_ratio=0.05, stratify=True), tr)
    assert any("Stratification disabled" in w for w in a.warnings)


def test_split_leaving_a_class_out_of_training_is_refused():
    tr = [CandidateTrial(trial_id=f"t{i}", label="rare" if i == 0 else "common", group=None, session=None)
          for i in range(20)]
    outcomes = []
    for seed in range(10):
        try:
            make_split(SplitConfig(strategy="random", test_ratio=0.2, seed=seed), tr)
            outcomes.append("ok")
        except SplitError as exc:
            assert "training set lacks rare" in str(exc)
            outcomes.append("refused")
    assert "refused" in outcomes  # whenever the only 'rare' trial lands in test, the split is rejected


def test_validation_split_never_touches_test():
    tr = trials()
    a = make_split(SplitConfig(strategy="group", test_groups=["P4"]), tr)
    by = {t.trial_id: t for t in tr}
    fit, val, desc = validation_split(a.folds[0].train, by, "auto", 0.25, 0, group_level=True)
    assert not (set(fit) | set(val)) & set(a.folds[0].test)
    assert "Group-wise" in desc
    assert not {by[i].group for i in fit} & {by[i].group for i in val}
