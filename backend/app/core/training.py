"""PyTorch training loop for a single fold.

Model selection (early stopping / best-epoch restore) uses the validation subset
carved from the training partition. Test data never enters this module.
"""

from __future__ import annotations

import copy
import random
import threading
import time
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, Field
from sklearn.metrics import balanced_accuracy_score

from .eegnet import ArchitectureConfig, EEGNet


class TrainingConfig(BaseModel):
    epochs: int = Field(60, ge=1, le=2000)
    batch_size: int = Field(32, ge=2, le=4096)
    learning_rate: float = Field(1e-3, gt=0, le=10)
    optimizer: Literal["adam", "adamw", "sgd"] = "adam"
    momentum: float = Field(0.9, ge=0, lt=1)
    weight_decay: float = Field(0.0, ge=0, le=1)
    loss: Literal["cross_entropy"] = "cross_entropy"
    label_smoothing: float = Field(0.0, ge=0, lt=1)
    seed: int = 42
    device: Literal["auto", "cpu", "cuda"] = "auto"
    validation_mode: Literal["auto", "stratified", "random", "group", "none"] = "auto"
    validation_fraction: float = Field(0.2, gt=0, lt=0.9)
    early_stopping: bool = True
    patience: int = Field(15, ge=1, le=1000)
    monitor: Literal["val_loss", "val_balanced_accuracy"] = "val_loss"
    restore_best: bool = True
    scheduler: Literal["none", "step", "cosine", "plateau"] = "none"
    step_size: int = Field(20, ge=1)
    gamma: float = Field(0.5, gt=0, lt=1)
    class_weighting: Literal["none", "balanced"] = "none"
    representative_trials: int = Field(4, ge=1, le=32)


def validate_training_config(cfg: TrainingConfig) -> list[str]:
    errors = []
    if cfg.validation_mode == "none":
        if cfg.early_stopping:
            errors.append("Early stopping needs a validation set; choose a validation strategy or disable it.")
        if cfg.scheduler == "plateau":
            errors.append("The plateau scheduler monitors validation loss; choose a validation strategy.")
        if cfg.restore_best:
            errors.append("Restoring the best epoch needs a validation set; disable it or add validation.")
    if cfg.device == "cuda" and not torch.cuda.is_available():
        errors.append("CUDA was requested but no CUDA device is available to PyTorch.")
    return errors


def resolve_device(pref: str) -> torch.device:
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def device_label(dev: torch.device) -> str:
    if dev.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Cancelled(Exception):
    pass


@dataclass
class FoldTensors:
    X_fit: np.ndarray
    y_fit: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray


def _evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor, loss_fn, batch: int = 256) -> tuple[float, np.ndarray]:
    model.eval()
    dev = next(model.parameters()).device
    losses, preds = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out = model(X[i:i + batch].to(dev))
            losses.append(loss_fn(out, y[i:i + batch].to(dev)).item() * len(out))
            preds.append(out.argmax(1).cpu().numpy())
    return float(sum(losses) / max(len(X), 1)), np.concatenate(preds) if preds else np.array([])


def train_fold(arch: ArchitectureConfig, cfg: TrainingConfig, data: FoldTensors, n_classes: int,
               device: torch.device, on_epoch: Callable[[dict], None], cancel: threading.Event,
               log: Callable[[str], None] = lambda _m: None) -> tuple[EEGNet, list[dict], dict]:
    set_seed(cfg.seed)
    model = EEGNet(arch).to(device)
    # data stay in host memory; each mini-batch is moved to the device (large datasets exceed GPU memory)
    Xf = torch.from_numpy(data.X_fit[:, None]).float()
    yf = torch.from_numpy(data.y_fit).long()
    has_val = len(data.X_val) > 0
    Xv = torch.from_numpy(data.X_val[:, None]).float() if has_val else None
    yv = torch.from_numpy(data.y_val).long() if has_val else None

    weight = None
    if cfg.class_weighting == "balanced":
        counts = np.bincount(data.y_fit, minlength=n_classes).astype(float)
        w = np.where(counts > 0, len(data.y_fit) / (n_classes * np.maximum(counts, 1)), 0.0)
        weight = torch.tensor(w, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.label_smoothing)
    eval_loss_fn = nn.CrossEntropyLoss(weight=weight)

    params = model.parameters()
    if cfg.optimizer == "adam":
        opt = torch.optim.Adam(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.SGD(params, lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    sched = None
    if cfg.scheduler == "step":
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma)
    elif cfg.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    elif cfg.scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=cfg.gamma,
                                                           patience=max(2, cfg.patience // 3))

    gen = torch.Generator()
    gen.manual_seed(cfg.seed)
    history: list[dict] = []
    best_state, best_epoch, best_score = None, None, None
    stale = 0
    t0 = time.perf_counter()
    stopped_early = False
    n = len(Xf)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        tot_loss, correct, seen = 0.0, 0, 0
        for i in range(0, n, cfg.batch_size):
            if cancel.is_set():
                raise Cancelled()
            idx = perm[i:i + cfg.batch_size]
            if len(idx) < 2 and n > 1:  # BatchNorm needs > 1 sample in training mode
                continue
            opt.zero_grad(set_to_none=True)
            xb, yb = Xf[idx].to(device, non_blocking=True), yf[idx].to(device, non_blocking=True)
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            model.apply_max_norm()
            tot_loss += loss.item() * len(idx)
            correct += (out.argmax(1) == yb).sum().item()
            seen += len(idx)
        if not np.isfinite(tot_loss):
            raise FloatingPointError(f"Training loss became non-finite at epoch {epoch}; lower the learning rate.")
        row = {"epoch": epoch, "train_loss": tot_loss / max(seen, 1), "train_acc": correct / max(seen, 1),
               "lr": opt.param_groups[0]["lr"]}
        if has_val:
            vl, vp = _evaluate(model, Xv, yv, eval_loss_fn)
            yv_np = data.y_val
            row.update({"val_loss": vl, "val_acc": float((vp == yv_np).mean()),
                        "val_balanced_accuracy": float(balanced_accuracy_score(yv_np, vp))})
            score = vl if cfg.monitor == "val_loss" else -row["val_balanced_accuracy"]
            if best_score is None or score < best_score - 1e-6:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            else:
                stale += 1
        row["elapsed_s"] = time.perf_counter() - t0
        history.append(row)
        on_epoch(row)
        if sched is not None:
            if cfg.scheduler == "plateau":
                sched.step(row["val_loss"])
            else:
                sched.step()
        if has_val and cfg.early_stopping and stale >= cfg.patience:
            stopped_early = True
            log(f"Early stopping at epoch {epoch}: no validation improvement for {cfg.patience} epochs.")
            break
    selected_epoch = history[-1]["epoch"]
    if has_val and cfg.restore_best and best_state is not None:
        model.load_state_dict(best_state)
        selected_epoch = best_epoch
        log(f"Restored weights from epoch {best_epoch} (best validation {cfg.monitor}).")
    info = {"selected_epoch": selected_epoch, "stopped_early": stopped_early, "epochs_run": len(history),
            "train_time_s": time.perf_counter() - t0, "monitor": cfg.monitor if has_val else None,
            "selection_basis": "validation subset of the training partition" if has_val else "last epoch"}
    model.eval()
    return model, history, info
