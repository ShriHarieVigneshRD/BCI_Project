"""Phase 1: EEGNet model, architecture validation, tensor tracing (requirement 4 + scientific unit tests)."""

import numpy as np
import pytest
import torch

from app.core.eegnet import (ArchitectureConfig, EEGNet, compute_architecture, map_coordinate, preset_config,
                             validate_architecture)
from app.core.tracing import run_trace

CONFIGS = [
    dict(n_channels=22, n_times=256, n_classes=4),
    dict(n_channels=8, n_times=128, n_classes=2, temporal_kernel=32, F1=4, D=1, F2=8, padding="valid",
         separable_kernel=8),
    dict(n_channels=16, n_times=500, n_classes=3, temporal_kernel=125, pool1_kernel=5, pool1_stride=3,
         pool2_kernel=4, pool2_stride=2, pool_type="max", activation="gelu", temporal_stride=2),
    dict(n_channels=12, n_times=300, n_classes=2, spatial_kernel=5, batch_norm=False, dropout_type="spatial",
         F2=10, activation="relu"),
]


@pytest.mark.parametrize("kw", CONFIGS)
def test_analytic_shapes_match_torch(kw):
    cfg = ArchitectureConfig(**kw)
    rep = validate_architecture(cfg)
    assert rep.valid, rep.issues
    assert rep.torch_verified, rep.torch_message
    model = EEGNet(cfg)
    assert sum(p.numel() for p in model.parameters()) == rep.total_params
    out = model.eval()(torch.zeros(2, 1, cfg.n_channels, cfg.n_times))
    assert out.shape == (2, cfg.n_classes)


def test_eegnet_8_2_baseline_parameter_count():
    # EEGNet-8,2 on 22 channels x 256 samples, 4 classes; values follow from the published architecture.
    cfg = preset_config("eegnet-8,2", 22, 256, 4, 128.0)
    rep = compute_architecture(cfg)
    layers = {s.name: s for s in rep.layers}
    assert layers["temporal_conv"].params == 8 * 64
    assert layers["depthwise_conv"].params == 16 * 22
    assert layers["separable_depthwise"].params == 16 * 16
    assert layers["separable_pointwise"].params == 16 * 16
    assert layers["flatten"].out_shape == [16 * 8]
    assert layers["classifier"].params == 128 * 4 + 4


@pytest.mark.parametrize("change,code", [
    (dict(pool2_kernel=200), "pool2_too_large"),
    (dict(spatial_kernel=40), "spatial_kernel_too_tall"),
    (dict(padding="valid", temporal_kernel=400), "temporal_kernel_too_long"),
    (dict(pool1_kernel=300), "pool1_too_large"),
])
def test_invalid_architectures_rejected(change, code):
    cfg = ArchitectureConfig(n_channels=22, n_times=256, n_classes=2, **change)
    rep = validate_architecture(cfg)
    assert not rep.valid
    assert any(i.code == code and i.level == "error" for i in rep.issues)
    with pytest.raises(ValueError):
        EEGNet(cfg)


def test_field_constraints_rejected_by_schema():
    with pytest.raises(Exception):
        ArchitectureConfig(n_channels=22, n_times=256, n_classes=2, F1=0)
    with pytest.raises(Exception):
        ArchitectureConfig(n_channels=22, n_times=256, n_classes=1)
    with pytest.raises(Exception):
        ArchitectureConfig(n_channels=22, n_times=256, n_classes=2, dropout=1.0)


def test_validate_endpoint_rejects_invalid(client):
    r = client.post("/api/models/validate", json={"config": {"n_channels": 8, "n_times": 64, "n_classes": 2,
                                                             "pool1_kernel": 32, "pool2_kernel": 32}})
    assert r.status_code == 200 and r.json()["valid"] is False
    r = client.post("/api/models/validate", json={"config": {"n_channels": 8, "n_times": 64, "n_classes": 2,
                                                             "F1": -1}})
    assert r.status_code == 422


@pytest.mark.parametrize("kw", CONFIGS[:3])
def test_receptive_field_matches_gradients(kw):
    """The analytic receptive field of a pool2 output equals the support of its input gradient."""
    # average pooling: every sample in the window receives gradient (max pooling routes it to one sample)
    cfg = ArchitectureConfig(**{**kw, "batch_norm": False, "dropout": 0.0, "pool_type": "avg"})
    rep = compute_architecture(cfg)
    model = EEGNet(cfg).eval()
    torch.manual_seed(0)
    for p in model.parameters():
        torch.nn.init.uniform_(p, 0.5, 1.5)  # strictly positive weights: no cancellations
    model.act1 = model.act2 = torch.nn.Identity()
    x = torch.randn(1, 1, cfg.n_channels, cfg.n_times).abs().requires_grad_(True)
    acts = {}
    model.pool2.register_forward_hook(lambda m, i, o: acts.__setitem__("p2", o))
    model(x)
    W = acts["p2"].shape[-1]
    w = W // 2
    acts["p2"][0, 0, 0, w].backward()
    support = torch.nonzero(x.grad[0, 0].abs().sum(0) > 0).flatten()
    m = map_coordinate(rep, "pool2", [0, 0, w])
    assert [int(support.min()), int(support.max())] == m["time_range"]


def test_flatten_index_mapping_roundtrip():
    rep = compute_architecture(ArchitectureConfig(n_channels=8, n_times=256, n_classes=2))
    Fm, H, W = rep.layers[-2].in_shape
    for idx in (0, 7, Fm * H * W - 1):
        m = map_coordinate(rep, "flatten", [idx])
        f, h, w = m["unflattened"]
        assert f * H * W + h * W + w == idx


def test_trace_hooks_capture_every_layer():
    cfg = ArchitectureConfig(n_channels=8, n_times=128, n_classes=3, temporal_kernel=32)
    model = EEGNet(cfg)
    tr = run_trace(model, np.random.randn(2, 8, 128).astype(np.float32))
    assert list(tr.outputs) == model.layer_names
    spec = {s.name: s.out_shape for s in compute_architecture(cfg).layers}
    for name, arr in tr.outputs.items():
        assert list(arr.shape[1:]) == spec[name]
    np.testing.assert_allclose(tr.probabilities.sum(1), 1.0, rtol=1e-5)


def test_max_norm_constraint():
    cfg = ArchitectureConfig(n_channels=8, n_times=128, n_classes=2)
    m = EEGNet(cfg)
    with torch.no_grad():
        m.depthwise_conv.conv.weight.mul_(100)
        m.classifier.weight.mul_(100)
    m.apply_max_norm()
    assert m.depthwise_conv.conv.weight.flatten(1).norm(dim=1).max() <= 1.0 + 1e-5
    assert m.classifier.weight.norm(dim=1).max() <= 0.25 + 1e-5
