"""Backend-owned experiment lifecycle (explicit, testable state machine)."""

from __future__ import annotations

from enum import Enum


class State(str, Enum):
    CREATED = "created"
    DATASET_REQUIRED = "dataset_required"
    DATASET_READY = "dataset_ready"
    SPLIT_CONFIGURED = "split_configured"
    PREPROCESSING_CONFIGURED = "preprocessing_configured"
    PREPROCESSING_COMPLETED = "preprocessing_completed"
    ARCHITECTURE_CONFIGURED = "architecture_configured"
    TRAINING_QUEUED = "training_queued"
    TRAINING = "training"
    TRAINING_FAILED = "training_failed"
    TRAINING_CANCELLED = "training_cancelled"
    TRAINING_COMPLETED = "training_completed"
    EVALUATION_RUNNING = "evaluation_running"
    EVALUATION_COMPLETED = "evaluation_completed"
    ARTIFACTS_INCOMPLETE = "artifacts_incomplete"
    ARTIFACTS_READY = "artifacts_ready"


S = State

#: Forward transitions. Rollbacks (invalidation after an upstream change) are a separate,
#: explicit operation - see ``can_rollback``.
TRANSITIONS: dict[State, set[State]] = {
    S.CREATED: {S.DATASET_REQUIRED, S.DATASET_READY},
    S.DATASET_REQUIRED: {S.DATASET_READY},
    S.DATASET_READY: {S.SPLIT_CONFIGURED},
    S.SPLIT_CONFIGURED: {S.PREPROCESSING_CONFIGURED},
    S.PREPROCESSING_CONFIGURED: {S.PREPROCESSING_COMPLETED},
    S.PREPROCESSING_COMPLETED: {S.ARCHITECTURE_CONFIGURED},
    S.ARCHITECTURE_CONFIGURED: {S.TRAINING_QUEUED},
    S.TRAINING_QUEUED: {S.TRAINING, S.TRAINING_CANCELLED, S.TRAINING_FAILED},
    S.TRAINING: {S.TRAINING_COMPLETED, S.TRAINING_FAILED, S.TRAINING_CANCELLED},
    S.TRAINING_FAILED: {S.TRAINING_QUEUED},
    S.TRAINING_CANCELLED: {S.TRAINING_QUEUED},
    S.TRAINING_COMPLETED: {S.ARTIFACTS_READY, S.ARTIFACTS_INCOMPLETE},
    S.ARTIFACTS_INCOMPLETE: {S.ARTIFACTS_READY, S.TRAINING_QUEUED},
    S.ARTIFACTS_READY: {S.EVALUATION_RUNNING, S.ARTIFACTS_INCOMPLETE, S.TRAINING_QUEUED},
    S.EVALUATION_RUNNING: {S.EVALUATION_COMPLETED, S.ARTIFACTS_READY},
    S.EVALUATION_COMPLETED: {S.EVALUATION_RUNNING, S.ARTIFACTS_INCOMPLETE, S.TRAINING_QUEUED},
}

#: Ordering of configuration stages; every training/post-training state ranks above them.
RANK: dict[State, int] = {
    S.CREATED: 0, S.DATASET_REQUIRED: 0, S.DATASET_READY: 1, S.SPLIT_CONFIGURED: 2,
    S.PREPROCESSING_CONFIGURED: 3, S.PREPROCESSING_COMPLETED: 4, S.ARCHITECTURE_CONFIGURED: 5,
}
for _s in State:
    RANK.setdefault(_s, 6)

BUSY = {S.TRAINING_QUEUED, S.TRAINING, S.EVALUATION_RUNNING}
LEARNED_READY = {S.ARTIFACTS_READY, S.EVALUATION_RUNNING, S.EVALUATION_COMPLETED}
ROLLBACK_TARGETS = {S.DATASET_REQUIRED, S.DATASET_READY, S.SPLIT_CONFIGURED, S.PREPROCESSING_CONFIGURED,
                    S.PREPROCESSING_COMPLETED, S.ARCHITECTURE_CONFIGURED}


class TransitionError(Exception):
    def __init__(self, current: State, target: State, reason: str | None = None):
        self.current, self.target = current, target
        super().__init__(reason or f"Transition {current.value} -> {target.value} is not allowed.")


def can_transition(current: State, target: State) -> bool:
    return target in TRANSITIONS.get(current, set())


def can_rollback(current: State, target: State) -> bool:
    return target in ROLLBACK_TARGETS and current not in BUSY and RANK[current] >= RANK[target]


def describe() -> dict:
    return {
        "states": [s.value for s in State],
        "transitions": {k.value: sorted(v.value for v in vs) for k, vs in TRANSITIONS.items()},
        "rollback_targets": sorted(s.value for s in ROLLBACK_TARGETS),
        "busy": sorted(s.value for s in BUSY),
        "learned_ready": sorted(s.value for s in LEARNED_READY),
    }
