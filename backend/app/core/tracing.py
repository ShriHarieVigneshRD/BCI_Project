"""Forward-hook based activation extraction.

Every activation returned by the API is produced here by running the real model
(trained checkpoint or an explicitly labelled untrained initialisation) on a real
input tensor. Nothing is synthesised to fill gaps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .eegnet import EEGNet


@dataclass
class Trace:
    inputs: dict[str, np.ndarray]   # layer name -> input tensor (batch removed)
    outputs: dict[str, np.ndarray]  # layer name -> output tensor (batch removed)
    logits: np.ndarray              # [batch, n_classes]
    probabilities: np.ndarray       # [batch, n_classes]


def run_trace(model: EEGNet, x: np.ndarray | torch.Tensor, layers: list[str] | None = None,
              device: str = "cpu") -> Trace:
    """Run ``model`` in eval mode on ``x`` ([N, 1, C, T] or [C, T]) and record layer I/O via hooks."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    if x.ndim == 2:
        x = x[None, None]
    elif x.ndim == 3:
        x = x[:, None]
    wanted = set(layers) if layers else set(model.layer_names)
    ins: dict[str, np.ndarray] = {}
    outs: dict[str, np.ndarray] = {}

    def hook(name):
        def fn(_module, inp, out):
            ins[name] = inp[0].detach().cpu().numpy()
            outs[name] = out.detach().cpu().numpy()
        return fn

    handles = [getattr(model, n).register_forward_hook(hook(n)) for n in model.layer_names if n in wanted]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = model(x.to(device)).detach().cpu()
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    probs = torch.softmax(logits, dim=1).numpy()
    return Trace(inputs=ins, outputs=outs, logits=logits.numpy(), probabilities=probs)
