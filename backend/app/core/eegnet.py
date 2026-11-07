"""Configurable EEGNet (Lawhern et al., 2018) with explicit, inspectable layers.

The architecture specification (``compute_architecture``) and the PyTorch module
(``EEGNet``) are generated from the same ``ArchitectureConfig``. Analytic shape
propagation is cross-checked against a real forward pass (``verify_with_torch``),
so every shape shown in the UI is either computed by the model itself or by
logic that has been verified against it.

Input convention: ``[batch, 1, channels, time_samples]``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field

INPUT_CONVENTION = "[batch, 1, channels, time_samples]"

Activation = Literal["elu", "relu", "leaky_relu", "gelu", "silu"]


class ArchitectureConfig(BaseModel):
    """Every parameter that defines the EEGNet computation graph."""

    preset: str = "eegnet-8,2"
    n_channels: int = Field(22, ge=1, le=1024, description="Input EEG channels (C)")
    n_times: int = Field(256, ge=1, le=200_000, description="Input time samples (T)")
    n_classes: int = Field(2, ge=2, le=256)
    sfreq: float | None = Field(None, gt=0, description="Sampling rate (guidance only)")
    F1: int = Field(8, ge=1, le=512, description="Temporal filters")
    D: int = Field(2, ge=1, le=32, description="Depth multiplier (spatial filters per temporal filter)")
    F2: int = Field(16, ge=1, le=2048, description="Pointwise filters")
    temporal_kernel: int = Field(64, ge=1, le=4096)
    temporal_stride: int = Field(1, ge=1, le=128)
    spatial_kernel: int | None = Field(None, ge=1, le=1024, description="None = all channels")
    pool1_kernel: int = Field(4, ge=1, le=1024)
    pool1_stride: int | None = Field(None, ge=1, le=1024, description="None = kernel size")
    separable_kernel: int = Field(16, ge=1, le=4096)
    pool2_kernel: int = Field(8, ge=1, le=1024)
    pool2_stride: int | None = Field(None, ge=1, le=1024, description="None = kernel size")
    pool_type: Literal["avg", "max"] = "avg"
    padding: Literal["same", "valid"] = "same"
    dropout: float = Field(0.5, ge=0.0, lt=1.0)
    dropout_type: Literal["dropout", "spatial"] = "dropout"
    activation: Activation = "elu"
    batch_norm: bool = True
    bn_momentum: float = Field(0.01, gt=0.0, le=1.0)
    bn_eps: float = Field(1e-3, gt=0.0, le=1.0)
    depthwise_max_norm: float | None = Field(1.0, gt=0.0)
    classifier_max_norm: float | None = Field(0.25, gt=0.0)

    @property
    def spatial_k(self) -> int:
        return self.spatial_kernel if self.spatial_kernel is not None else self.n_channels

    @property
    def p1_stride(self) -> int:
        return self.pool1_stride if self.pool1_stride is not None else self.pool1_kernel

    @property
    def p2_stride(self) -> int:
        return self.pool2_stride if self.pool2_stride is not None else self.pool2_kernel


def config_fingerprint(cfg: ArchitectureConfig) -> str:
    """Hash of every field that affects computation (preset label excluded)."""
    payload = cfg.model_dump(exclude={"preset", "sfreq"})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Parameter classification (used by Architecture Breaker)
# ---------------------------------------------------------------------------

#: Parameters that only change optimisation behaviour; eval-mode outputs of an
#: existing checkpoint are unchanged.
TRAINING_ONLY_PARAMS = {
    "dropout": "Dropout is disabled in evaluation mode; inference outputs are unchanged, but training would differ.",
    "dropout_type": "Dropout variant only acts during training.",
    "bn_momentum": "BatchNorm momentum only updates running statistics during training.",
    "depthwise_max_norm": "Max-norm constraints are applied after optimiser steps during training only.",
    "classifier_max_norm": "Max-norm constraints are applied after optimiser steps during training only.",
}
#: Parameters that are pure labels.
DISPLAY_ONLY_PARAMS = {"preset": "Preset label only.", "sfreq": "Used for guidance text only."}
#: Parameters that keep weight shapes but change the learned computation.
COMPUTATION_PARAMS = {
    "activation": "The weights were learned with a different non-linearity.",
    "pool_type": "The weights were learned with a different pooling operator.",
    "bn_eps": "Changes BatchNorm normalisation at inference.",
    "batch_norm": "Removes/adds learned BatchNorm layers.",
    "padding": "Changes which samples each kernel sees.",
    "temporal_stride": "Changes where the temporal kernel is applied.",
    "pool1_stride": "Changes the pooled time grid seen by later layers.",
    "pool2_stride": "Changes the pooled time grid seen by the classifier.",
    "pool1_kernel": "Changes the pooled time grid seen by later layers.",
    "pool2_kernel": "Changes the pooled time grid seen by the classifier.",
    "n_times": "Changes the epoch length.",
}


# ---------------------------------------------------------------------------
# Layer specification and analytic shape propagation
# ---------------------------------------------------------------------------


@dataclass
class LayerSpec:
    name: str
    type: str
    label: str
    block: str
    purpose: str
    in_shape: list[int]
    out_shape: list[int]
    kernel: list[int] | None = None
    stride: list[int] | None = None
    padding: list[int] | None = None  # [left, right] along time (or [top, bottom] for spatial)
    groups: int | None = None
    dilation: list[int] | None = None
    params: int = 0
    trainable: int = 0
    buffers: int = 0
    param_shapes: dict[str, list[int]] = field(default_factory=dict)
    macs: int = 0
    rf_time: int = 1
    rf_channels: int = 1
    jump_time: int = 1
    start_time: float = 0.0
    valid: bool = True
    message: str | None = None
    learnable: bool = False

    @property
    def out_elements(self) -> int:
        return math.prod(self.out_shape) if self.out_shape else 0


@dataclass
class Issue:
    level: Literal["error", "warning", "info"]
    code: str
    message: str
    layer: str | None = None
    fix: str | None = None


@dataclass
class ArchitectureReport:
    config: dict
    fingerprint: str
    input_shape: list[int]
    input_convention: str
    layers: list[LayerSpec]
    issues: list[Issue]
    valid: bool
    total_params: int
    trainable_params: int
    buffers: int
    total_macs: int
    activation_bytes_per_sample: int
    param_bytes: int
    receptive_field_samples: int | None
    receptive_field_seconds: float | None
    classifier_in_features: int | None
    torch_verified: bool = False
    torch_message: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _same_pad(length: int, kernel: int, stride: int) -> tuple[int, int, int]:
    """TensorFlow-style 'same' padding. Returns (left, right, out_len)."""
    out = math.ceil(length / stride)
    total = max((out - 1) * stride + kernel - length, 0)
    left = total // 2
    return left, total - left, out


def _conv_out(length: int, kernel: int, stride: int, pad_l: int, pad_r: int) -> int:
    return (length + pad_l + pad_r - kernel) // stride + 1


def compute_architecture(cfg: ArchitectureConfig) -> ArchitectureReport:
    """Analytic shape propagation with explicit validity checks for every layer."""
    C, T = cfg.n_channels, cfg.n_times
    issues: list[Issue] = []
    layers: list[LayerSpec] = []

    # Receptive-field tracking along time: r (size), j (jump), start (centre offset of first output)
    r, j, start = 1, 1, 0.0
    shape = [1, C, T]

    def add(spec: LayerSpec) -> None:
        layers.append(spec)

    def elementwise(name: str, typ: str, label: str, block: str, purpose: str, **extra) -> None:
        add(
            LayerSpec(
                name=name, type=typ, label=label, block=block, purpose=purpose,
                in_shape=list(shape), out_shape=list(shape), rf_time=r, jump_time=j,
                start_time=start, rf_channels=rf_ch, **extra,
            )
        )

    rf_ch = 1
    broken = False

    # ---- Block 1: temporal convolution --------------------------------------------------
    K1, s1 = cfg.temporal_kernel, cfg.temporal_stride
    if cfg.padding == "same":
        pl, pr, T1 = _same_pad(T, K1, s1)
    else:
        pl = pr = 0
        T1 = _conv_out(T, K1, s1, 0, 0)
    spec = LayerSpec(
        name="temporal_conv", type="Conv2d", label="Temporal convolution", block="Block 1",
        purpose="Learns F1 band-pass-like temporal filters applied identically to every channel.",
        in_shape=list(shape), out_shape=[cfg.F1, C, max(T1, 0)], kernel=[1, K1], stride=[1, s1],
        padding=[pl, pr], groups=1, dilation=[1, 1], params=cfg.F1 * K1, trainable=cfg.F1 * K1,
        param_shapes={"weight": [cfg.F1, 1, 1, K1]}, learnable=True,
    )
    if T1 < 1:
        spec.valid, broken = False, True
        spec.message = f"Temporal kernel ({K1}) is longer than the epoch ({T} samples) with valid padding."
        issues.append(Issue("error", "temporal_kernel_too_long", spec.message, "temporal_conv",
                            "Use 'same' padding, a shorter temporal kernel, or a longer epoch."))
    r, j, start = r + (K1 - 1) * j, j * s1, start - pl * j + (K1 - 1) * j / 2
    spec.rf_time, spec.jump_time, spec.start_time, spec.rf_channels = r, j, start, rf_ch
    spec.macs = max(T1, 0) * C * cfg.F1 * K1
    add(spec)
    shape = spec.out_shape

    if cfg.batch_norm:
        elementwise("bn1", "BatchNorm2d", "Batch norm 1", "Block 1",
                    "Normalises each temporal filter's output using learned scale/shift and running statistics.",
                    params=2 * cfg.F1, trainable=2 * cfg.F1, buffers=2 * cfg.F1,
                    param_shapes={"weight": [cfg.F1], "bias": [cfg.F1]}, learnable=True)

    # ---- Block 1: depthwise spatial convolution ----------------------------------------
    Ks = cfg.spatial_k
    H1 = C - Ks + 1
    FD = cfg.F1 * cfg.D
    spec = LayerSpec(
        name="depthwise_conv", type="Conv2d (depthwise)", label="Depthwise spatial convolution",
        block="Block 1",
        purpose="Learns D spatial filters per temporal filter (groups = F1): frequency-specific spatial patterns.",
        in_shape=list(shape), out_shape=[FD, max(H1, 0), shape[2]], kernel=[Ks, 1], stride=[1, 1],
        padding=[0, 0], groups=cfg.F1, dilation=[1, 1], params=FD * Ks, trainable=FD * Ks,
        param_shapes={"weight": [FD, 1, Ks, 1]}, learnable=True,
    )
    if Ks > C:
        spec.valid, broken = False, True
        spec.message = f"Spatial kernel height ({Ks}) exceeds the number of input channels ({C})."
        issues.append(Issue("error", "spatial_kernel_too_tall", spec.message, "depthwise_conv",
                            f"Set the spatial kernel to {C} (full spatial filter, original EEGNet) or fewer."))
    rf_ch = Ks
    spec.rf_time, spec.jump_time, spec.start_time, spec.rf_channels = r, j, start, rf_ch
    spec.macs = max(H1, 0) * shape[2] * FD * Ks
    add(spec)
    shape = spec.out_shape

    if cfg.batch_norm:
        elementwise("bn2", "BatchNorm2d", "Batch norm 2", "Block 1",
                    "Normalises each spatial filter output.", params=2 * FD, trainable=2 * FD, buffers=2 * FD,
                    param_shapes={"weight": [FD], "bias": [FD]}, learnable=True)
    elementwise("act1", cfg.activation.upper(), f"{cfg.activation.upper()} activation", "Block 1",
                "Element-wise non-linearity.")

    # ---- Block 1: pooling --------------------------------------------------------------
    p1, sp1 = cfg.pool1_kernel, cfg.p1_stride
    W_in = shape[2]
    W1 = (W_in - p1) // sp1 + 1 if W_in >= p1 else 0
    spec = LayerSpec(
        name="pool1", type="AvgPool2d" if cfg.pool_type == "avg" else "MaxPool2d", label="Pooling 1",
        block="Block 1", purpose="Reduces the temporal sampling rate of the feature maps.",
        in_shape=list(shape), out_shape=[shape[0], shape[1], W1], kernel=[1, p1], stride=[1, sp1],
        padding=[0, 0],
    )
    if W1 < 1 and not broken:
        spec.valid, broken = False, True
        spec.message = f"Pool 1 window ({p1}) is larger than the time dimension entering it ({W_in})."
        issues.append(Issue("error", "pool1_too_large", spec.message, "pool1",
                            "Reduce the pool 1 size/stride, use a longer epoch, or a shorter temporal stride."))
    r, j, start = r + (p1 - 1) * j, j * sp1, start + (p1 - 1) * j / 2
    spec.rf_time, spec.jump_time, spec.start_time, spec.rf_channels = r, j, start, rf_ch
    add(spec)
    shape = spec.out_shape

    elementwise("drop1", "Dropout" if cfg.dropout_type == "dropout" else "Dropout2d", "Dropout 1", "Block 1",
                f"Randomly zeroes {'activations' if cfg.dropout_type == 'dropout' else 'whole feature maps'} "
                f"with p = {cfg.dropout} during training; identity at inference.")

    # ---- Block 2: separable convolution ------------------------------------------------
    K2 = cfg.separable_kernel
    W_in = shape[2]
    if cfg.padding == "same":
        ql, qr, W2 = _same_pad(W_in, K2, 1)
    else:
        ql = qr = 0
        W2 = W_in - K2 + 1
    spec = LayerSpec(
        name="separable_depthwise", type="Conv2d (depthwise)", label="Separable conv: depthwise", block="Block 2",
        purpose="Separable step 1 - one temporal kernel per feature map (groups = F1*D) summarises each map over time.",
        in_shape=list(shape), out_shape=[FD, shape[1], max(W2, 0)], kernel=[1, K2], stride=[1, 1],
        padding=[ql, qr], groups=FD, dilation=[1, 1], params=FD * K2, trainable=FD * K2,
        param_shapes={"weight": [FD, 1, 1, K2]}, learnable=True,
    )
    if W2 < 1 and not broken:
        spec.valid, broken = False, True
        spec.message = f"Separable kernel ({K2}) is longer than the pooled time dimension ({W_in}) with valid padding."
        issues.append(Issue("error", "separable_kernel_too_long", spec.message, "separable_depthwise",
                            "Use 'same' padding, a shorter separable kernel or a smaller pool 1."))
    r, start = r + (K2 - 1) * j, start - ql * j + (K2 - 1) * j / 2
    spec.rf_time, spec.jump_time, spec.start_time, spec.rf_channels = r, j, start, rf_ch
    spec.macs = max(W2, 0) * shape[1] * FD * K2
    add(spec)
    shape = spec.out_shape

    spec = LayerSpec(
        name="separable_pointwise", type="Conv2d (1x1)", label="Separable conv: pointwise", block="Block 2",
        purpose="Separable step 2 - 1x1 convolution that mixes the F1*D feature maps into F2 maps.",
        in_shape=list(shape), out_shape=[cfg.F2, shape[1], shape[2]], kernel=[1, 1], stride=[1, 1],
        padding=[0, 0], groups=1, dilation=[1, 1], params=cfg.F2 * FD, trainable=cfg.F2 * FD,
        param_shapes={"weight": [cfg.F2, FD, 1, 1]}, learnable=True,
        rf_time=r, jump_time=j, start_time=start, rf_channels=rf_ch,
    )
    spec.macs = shape[1] * shape[2] * cfg.F2 * FD
    add(spec)
    shape = spec.out_shape

    if cfg.batch_norm:
        elementwise("bn3", "BatchNorm2d", "Batch norm 3", "Block 2", "Normalises each pointwise feature map.",
                    params=2 * cfg.F2, trainable=2 * cfg.F2, buffers=2 * cfg.F2,
                    param_shapes={"weight": [cfg.F2], "bias": [cfg.F2]}, learnable=True)
    elementwise("act2", cfg.activation.upper(), f"{cfg.activation.upper()} activation", "Block 2",
                "Element-wise non-linearity.")

    p2, sp2 = cfg.pool2_kernel, cfg.p2_stride
    W_in = shape[2]
    W3 = (W_in - p2) // sp2 + 1 if W_in >= p2 else 0
    spec = LayerSpec(
        name="pool2", type="AvgPool2d" if cfg.pool_type == "avg" else "MaxPool2d", label="Pooling 2",
        block="Block 2", purpose="Further temporal down-sampling before classification.",
        in_shape=list(shape), out_shape=[shape[0], shape[1], W3], kernel=[1, p2], stride=[1, sp2], padding=[0, 0],
    )
    if W3 < 1 and not broken:
        spec.valid, broken = False, True
        spec.message = f"Pool 2 window ({p2}) is larger than the time dimension entering it ({W_in})."
        issues.append(Issue("error", "pool2_too_large", spec.message, "pool2",
                            "Reduce pool 1/pool 2 sizes or strides, or use a longer epoch."))
    r, j, start = r + (p2 - 1) * j, j * sp2, start + (p2 - 1) * j / 2
    spec.rf_time, spec.jump_time, spec.start_time, spec.rf_channels = r, j, start, rf_ch
    add(spec)
    shape = spec.out_shape

    elementwise("drop2", "Dropout" if cfg.dropout_type == "dropout" else "Dropout2d", "Dropout 2", "Block 2",
                f"Dropout with p = {cfg.dropout} during training; identity at inference.")

    n_flat = math.prod(shape) if not broken else 0
    add(LayerSpec(
        name="flatten", type="Flatten", label="Flatten", block="Classifier",
        purpose="Reshapes [F2, H, W] into a feature vector (row-major: index = f*H*W + h*W + w).",
        in_shape=list(shape), out_shape=[n_flat], rf_time=r, jump_time=j, start_time=start, rf_channels=rf_ch,
    ))
    add(LayerSpec(
        name="classifier", type="Linear", label="Dense classifier", block="Classifier",
        purpose="Linear map from the feature vector to one logit per class (softmax gives probabilities).",
        in_shape=[n_flat], out_shape=[cfg.n_classes], params=n_flat * cfg.n_classes + cfg.n_classes,
        trainable=n_flat * cfg.n_classes + cfg.n_classes,
        param_shapes={"weight": [cfg.n_classes, n_flat], "bias": [cfg.n_classes]}, learnable=True,
        macs=n_flat * cfg.n_classes, rf_time=T, jump_time=j, start_time=start, rf_channels=C,
    ))

    # ---- Advisory issues -----------------------------------------------------------------
    if not broken:
        if cfg.F2 != cfg.F1 * cfg.D:
            issues.append(Issue("info", "f2_differs", f"Original EEGNet uses F2 = F1 x D = {cfg.F1 * cfg.D}; "
                                f"this configuration uses F2 = {cfg.F2}.", "separable_pointwise"))
        if cfg.padding == "same" and K1 % 2 == 0 and s1 == 1:
            issues.append(Issue("info", "asymmetric_padding",
                                f"Even temporal kernel with 'same' padding pads {pl} sample(s) on the left and "
                                f"{pr} on the right.", "temporal_conv"))
        if cfg.sfreq:
            half = int(round(cfg.sfreq / 2))
            if abs(K1 - half) > max(2, half * 0.25):
                issues.append(Issue("info", "temporal_kernel_guidance",
                                    f"The original EEGNet uses a temporal kernel of about half the sampling rate "
                                    f"({half} samples at {cfg.sfreq:g} Hz), covering 500 ms. Current kernel: "
                                    f"{K1} samples ({K1 / cfg.sfreq * 1000:.0f} ms).", "temporal_conv"))
        if shape[2] == 1 and T > 1:
            issues.append(Issue("warning", "time_collapsed",
                                "The time dimension is reduced to a single sample before the classifier; "
                                "temporal structure is fully pooled away.", "pool2"))
        if Ks < C:
            issues.append(Issue("info", "partial_spatial_kernel",
                                f"Spatial kernel covers {Ks} of {C} channels; the depthwise output keeps "
                                f"{H1} rows (sliding over the channel order).", "depthwise_conv"))
        if cfg.n_channels > 1 and cfg.n_times < cfg.n_channels:
            issues.append(Issue("warning", "short_epoch", "Fewer time samples than channels; check the input "
                                "orientation is [channels, time].", None))

    total_params = sum(sp.params for sp in layers)
    trainable = sum(sp.trainable for sp in layers)
    buffers = sum(sp.buffers for sp in layers)
    macs = sum(sp.macs for sp in layers)
    act_elems = C * T + sum(sp.out_elements for sp in layers if sp.name != "flatten")
    rf = layers[-3].rf_time if not broken else None  # pool2/drop2 receptive field
    report = ArchitectureReport(
        config=cfg.model_dump(), fingerprint=config_fingerprint(cfg), input_shape=[1, C, T],
        input_convention=INPUT_CONVENTION, layers=layers, issues=issues,
        valid=not any(i.level == "error" for i in issues),
        total_params=total_params, trainable_params=trainable, buffers=buffers, total_macs=macs,
        activation_bytes_per_sample=act_elems * 4, param_bytes=(total_params + buffers) * 4,
        receptive_field_samples=rf, receptive_field_seconds=(rf / cfg.sfreq) if (rf and cfg.sfreq) else None,
        classifier_in_features=n_flat if not broken else None,
    )
    return report


# ---------------------------------------------------------------------------
# PyTorch module
# ---------------------------------------------------------------------------


class PaddedConv2d(nn.Module):
    """Conv2d with explicit (possibly asymmetric) zero padding along time.

    Keeping padding explicit means the Kernel Playground can reproduce exactly which
    input samples (including padded zeros) contribute to each output element.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: tuple[int, int], stride: tuple[int, int] = (1, 1),
                 groups: int = 1, pad_time: tuple[int, int] = (0, 0)):
        super().__init__()
        self.pad_time = (int(pad_time[0]), int(pad_time[1]))
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, groups=groups, bias=False)

    @property
    def weight(self) -> torch.Tensor:
        return self.conv.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_time[0] or self.pad_time[1]:
            x = F.pad(x, (self.pad_time[0], self.pad_time[1], 0, 0))
        return self.conv(x)


def make_activation(name: str) -> nn.Module:
    return {
        "elu": nn.ELU(),
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(0.01),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
    }[name]


class EEGNet(nn.Module):
    """EEGNet built from an ``ArchitectureConfig``; layer names match ``compute_architecture``."""

    def __init__(self, cfg: ArchitectureConfig):
        super().__init__()
        report = compute_architecture(cfg)
        if not report.valid:
            errs = "; ".join(i.message for i in report.issues if i.level == "error")
            raise ValueError(f"Invalid architecture: {errs}")
        self.cfg = cfg
        specs = {s.name: s for s in report.layers}
        FD = cfg.F1 * cfg.D
        drop = nn.Dropout if cfg.dropout_type == "dropout" else nn.Dropout2d
        pool = nn.AvgPool2d if cfg.pool_type == "avg" else nn.MaxPool2d
        bn = lambda n: nn.BatchNorm2d(n, momentum=cfg.bn_momentum, eps=cfg.bn_eps)  # noqa: E731

        self.temporal_conv = PaddedConv2d(1, cfg.F1, (1, cfg.temporal_kernel), (1, cfg.temporal_stride),
                                          pad_time=tuple(specs["temporal_conv"].padding))
        if cfg.batch_norm:
            self.bn1 = bn(cfg.F1)
        self.depthwise_conv = PaddedConv2d(cfg.F1, FD, (cfg.spatial_k, 1), groups=cfg.F1)
        if cfg.batch_norm:
            self.bn2 = bn(FD)
        self.act1 = make_activation(cfg.activation)
        self.pool1 = pool((1, cfg.pool1_kernel), stride=(1, cfg.p1_stride))
        self.drop1 = drop(cfg.dropout)
        self.separable_depthwise = PaddedConv2d(FD, FD, (1, cfg.separable_kernel), groups=FD,
                                                pad_time=tuple(specs["separable_depthwise"].padding))
        self.separable_pointwise = PaddedConv2d(FD, cfg.F2, (1, 1))
        if cfg.batch_norm:
            self.bn3 = bn(cfg.F2)
        self.act2 = make_activation(cfg.activation)
        self.pool2 = pool((1, cfg.pool2_kernel), stride=(1, cfg.p2_stride))
        self.drop2 = drop(cfg.dropout)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(report.classifier_in_features or 1, cfg.n_classes)
        self.layer_names = [s.name for s in report.layers]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for name in self.layer_names:
            x = getattr(self, name)(x)
        return x

    @torch.no_grad()
    def apply_max_norm(self) -> None:
        """Keras-style max-norm constraints (renorm each output unit's weight vector)."""
        if self.cfg.depthwise_max_norm:
            w = self.depthwise_conv.conv.weight
            w.copy_(torch.renorm(w, p=2, dim=0, maxnorm=self.cfg.depthwise_max_norm))
        if self.cfg.classifier_max_norm:
            w = self.classifier.weight
            w.copy_(torch.renorm(w, p=2, dim=0, maxnorm=self.cfg.classifier_max_norm))


def parameter_shapes(model: nn.Module) -> dict[str, list[int]]:
    return {k: list(v.shape) for k, v in model.state_dict().items()}


def verify_with_torch(cfg: ArchitectureConfig, report: ArchitectureReport | None = None) -> tuple[bool, str]:
    """Run a real forward pass on zeros and compare every layer's output shape and parameter count."""
    report = report or compute_architecture(cfg)
    if not report.valid:
        return False, "Skipped: analytic validation failed."
    if cfg.n_channels * cfg.n_times * cfg.F1 * cfg.D > 400_000_000:
        return False, "Skipped: input too large for an on-the-fly verification pass."
    model = EEGNet(cfg).eval()
    shapes: dict[str, list[int]] = {}
    hooks = [
        getattr(model, name).register_forward_hook(
            lambda _m, _i, o, name=name: shapes.__setitem__(name, list(o.shape[1:]))
        )
        for name in model.layer_names
    ]
    try:
        with torch.no_grad():
            model(torch.zeros(1, 1, cfg.n_channels, cfg.n_times))
    finally:
        for h in hooks:
            h.remove()
    for spec in report.layers:
        if shapes.get(spec.name) != spec.out_shape:
            return False, f"Shape mismatch at {spec.name}: analytic {spec.out_shape}, torch {shapes.get(spec.name)}"
        module = getattr(model, spec.name)
        n_params = sum(p.numel() for p in module.parameters())
        if n_params != spec.params:
            return False, f"Parameter mismatch at {spec.name}: analytic {spec.params}, torch {n_params}"
    return True, "Every layer shape and parameter count matches a real PyTorch forward pass."


def validate_architecture(cfg: ArchitectureConfig) -> ArchitectureReport:
    report = compute_architecture(cfg)
    ok, msg = verify_with_torch(cfg, report)
    report.torch_verified, report.torch_message = ok, msg
    if report.valid and not ok and msg.startswith("Shape mismatch"):
        report.valid = False
        report.issues.append(Issue("error", "torch_mismatch", msg))
    return report


# ---------------------------------------------------------------------------
# Presets and coordinate mapping
# ---------------------------------------------------------------------------


def preset_config(name: str, n_channels: int, n_times: int, n_classes: int, sfreq: float | None) -> ArchitectureConfig:
    k1 = 64
    if sfreq:
        k1 = max(2, int(round(sfreq / 2)))
    base = dict(n_channels=n_channels, n_times=n_times, n_classes=n_classes, sfreq=sfreq, temporal_kernel=k1)
    if name == "eegnet-8,2":
        return ArchitectureConfig(preset=name, F1=8, D=2, F2=16, **base)
    if name == "eegnet-4,2":
        return ArchitectureConfig(preset=name, F1=4, D=2, F2=8, **base)
    if name == "eegnet-16,2":
        return ArchitectureConfig(preset=name, F1=16, D=2, F2=32, **base)
    raise KeyError(name)


PRESETS = [
    {"id": "eegnet-8,2", "label": "EEGNet-8,2 (original baseline)",
     "description": "F1 = 8, D = 2, F2 = 16, temporal kernel = sfreq/2, pools 4 and 8, dropout 0.5, ELU, max-norm "
                    "1.0 / 0.25 (Lawhern et al., 2018)."},
    {"id": "eegnet-4,2", "label": "EEGNet-4,2 (compact)", "description": "F1 = 4, D = 2, F2 = 8."},
    {"id": "eegnet-16,2", "label": "EEGNet-16,2 (wider)", "description": "F1 = 16, D = 2, F2 = 32."},
]


def map_coordinate(report: ArchitectureReport, layer: str, coord: list[int]) -> dict:
    """Map an output coordinate of ``layer`` back to the input region it depends on."""
    spec = next((s for s in report.layers if s.name == layer), None)
    if spec is None:
        raise KeyError(layer)
    C, T = report.input_shape[1], report.input_shape[2]
    out = spec.out_shape
    result: dict = {"layer": layer, "out_shape": out}
    if layer == "classifier":
        k = coord[0] if coord else 0
        if not 0 <= k < out[0]:
            raise ValueError("class index out of range")
        result.update({"depends_on": "all input samples and channels", "time_range": [0, T - 1],
                       "channel_range": [0, C - 1], "feature_index": k})
        return result
    flat_spec = next(s for s in report.layers if s.name == "flatten")
    if layer == "flatten":
        idx = coord[0] if coord else 0
        Fm, H, W = flat_spec.in_shape
        if not 0 <= idx < Fm * H * W:
            raise ValueError("flat index out of range")
        f, rem = divmod(idx, H * W)
        h, w = divmod(rem, W)
        coord = [f, h, w]
        result["flat_index"] = idx
        result["unflattened"] = coord
        spec = next(s for s in report.layers if s.name == "drop2")
        out = spec.out_shape
    if len(coord) != 3 or any(not 0 <= c < n for c, n in zip(coord, out)):
        raise ValueError(f"coordinate {coord} outside output shape {out}")
    f, h, w = coord
    centre = spec.start_time + w * spec.jump_time
    t0 = int(math.floor(centre - (spec.rf_time - 1) / 2))
    t1 = t0 + spec.rf_time - 1
    ch0, ch1 = (h, h + spec.rf_channels - 1) if spec.rf_channels > 1 else (h, h)
    result.update({
        "coordinate": coord,
        "time_range_unclipped": [t0, t1],
        "time_range": [max(t0, 0), min(t1, T - 1)],
        "padded_samples": max(0, -t0) + max(0, t1 - (T - 1)),
        "channel_range": [ch0, min(ch1, C - 1)],
        "receptive_field_samples": spec.rf_time,
        "jump": spec.jump_time,
        "flat_index": result.get("flat_index", (f * flat_spec.in_shape[1] * flat_spec.in_shape[2]
                                               + h * flat_spec.in_shape[2] + w) if layer in ("pool2", "drop2") else None),
    })
    return result
