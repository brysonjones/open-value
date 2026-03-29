"""Tests for value_estimator.py."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import open_value_estimator.utils as utils_module
from open_value_estimator.value_estimator import MLPHead, OpenValueEstimator


def make_state_stub() -> SimpleNamespace:
    return SimpleNamespace(_state_bins=torch.linspace(-1, 1, 257)[:-1])


def make_value_stub(num_bins: int = 201, v_min: float = -1.0, v_max: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        bin_centers=torch.linspace(v_min, v_max, num_bins),
        v_min=v_min,
        v_max=v_max,
        num_bins=num_bins,
    )


@pytest.mark.parametrize("depth", [1, 3])
def test_mlp_head_output_shape(depth: int) -> None:
    head = MLPHead(in_dim=5, hidden_dim=7, out_dim=3, depth=depth)
    x = torch.randn(4, 5)

    y = head(x)

    assert y.shape == (4, 3)


def test_mlp_head_rejects_depth_zero() -> None:
    with pytest.raises(ValueError, match="depth must be >= 1"):
        MLPHead(in_dim=4, hidden_dim=8, out_dim=2, depth=0)


def test_mlp_head_gradients_reach_all_parameters() -> None:
    head = MLPHead(in_dim=6, hidden_dim=8, out_dim=2, depth=2)
    x = torch.randn(3, 6, requires_grad=True)

    loss = head(x).sum()
    loss.backward()

    assert x.grad is not None
    for param in head.parameters():
        assert param.grad is not None


def test_discretize_state_maps_zero_to_middle_bin() -> None:
    stub = make_state_stub()

    state_strings = OpenValueEstimator.discretize_state(stub, torch.tensor([[0.0, 0.001]]))

    assert state_strings == ["127 128"]


def test_discretize_state_maps_boundary_values() -> None:
    stub = make_state_stub()

    state_strings = OpenValueEstimator.discretize_state(stub, torch.tensor([[-1.0, 1.0]]))

    assert state_strings == ["0 255"]


def test_discretize_state_clamps_out_of_range_values() -> None:
    stub = make_state_stub()

    state_strings = OpenValueEstimator.discretize_state(stub, torch.tensor([[-5.0, 7.0]]))

    assert state_strings == ["0 255"]


def test_get_expected_value_uniform_distribution_gives_support_mean() -> None:
    stub = make_value_stub(num_bins=5, v_min=-1.0, v_max=1.0)
    logits = torch.zeros(2, 5)

    values = OpenValueEstimator.get_expected_value(stub, logits)

    assert torch.allclose(values, torch.zeros(2))


def test_get_expected_value_all_mass_on_one_bin_gives_that_bin_center() -> None:
    stub = make_value_stub(num_bins=5, v_min=-1.0, v_max=1.0)
    logits = torch.full((1, 5), -20.0)
    logits[0, 3] = 20.0

    values = OpenValueEstimator.get_expected_value(stub, logits)

    assert torch.allclose(values, torch.tensor([stub.bin_centers[3].item()]), atol=1e-6)


def test_expected_value_round_trips_through_target_bins() -> None:
    stub = make_value_stub(num_bins=201, v_min=-1.0, v_max=0.0)
    values = torch.tensor([-1.0, -0.5, -0.123, 0.2])

    target_bins = OpenValueEstimator.get_target_bin_indices(stub, values)
    logits = torch.full((values.shape[0], stub.num_bins), -20.0)
    logits[torch.arange(values.shape[0]), target_bins] = 20.0
    round_tripped = OpenValueEstimator.get_expected_value(stub, logits)

    bin_width = (stub.v_max - stub.v_min) / (stub.num_bins - 1)
    assert torch.allclose(round_tripped, values.clamp(stub.v_min, stub.v_max), atol=bin_width / 2 + 1e-6)


def test_get_target_bin_indices_maps_boundaries_to_boundary_bins() -> None:
    stub = make_value_stub(num_bins=11, v_min=-1.0, v_max=0.0)

    indices = OpenValueEstimator.get_target_bin_indices(stub, torch.tensor([-1.0, 0.0]))

    assert torch.equal(indices, torch.tensor([0, 10]))


def test_get_target_bin_indices_clamps_out_of_range_values() -> None:
    stub = make_value_stub(num_bins=11, v_min=-1.0, v_max=0.0)

    indices = OpenValueEstimator.get_target_bin_indices(stub, torch.tensor([-5.0, 2.0]))

    assert torch.equal(indices, torch.tensor([0, 10]))


def test_from_pretrained_uses_json_sidecar_value_head_depth_and_prefers_ema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_config_dict: dict,
) -> None:
    checkpoint_path = tmp_path / "checkpoint_100.safetensors"
    ema_path = tmp_path / "checkpoint_100_ema.safetensors"
    config_path = tmp_path / "checkpoint_100.json"
    checkpoint_path.write_bytes(b"regular")
    ema_path.write_bytes(b"ema")

    model_config = dict(model_config_dict)
    model_config["value_head_depth"] = 5
    config_path.write_text(json.dumps(model_config), encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_init(self, *args, **kwargs):
        nn.Module.__init__(self)
        self.dummy = nn.Parameter(torch.ones(1))
        captured["init_kwargs"] = kwargs

    def fake_load_model_weights(path, model, device=None, strict=False):
        captured["weights_path"] = str(path)
        captured["device"] = device
        captured["strict"] = strict

    monkeypatch.setattr(OpenValueEstimator, "__init__", fake_init)
    monkeypatch.setattr(utils_module, "load_model_weights", fake_load_model_weights)

    model = OpenValueEstimator.from_pretrained(
        str(checkpoint_path),
        device="cpu",
        freeze=True,
        use_ema=True,
    )

    assert captured["init_kwargs"]["value_head_depth"] == 5
    assert captured["weights_path"] == str(ema_path)
    assert captured["device"] == "cpu"
    assert all(not param.requires_grad for param in model.parameters())
    assert model.training is False


def test_from_pretrained_missing_json_sidecar_raises(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint_100.safetensors"
    checkpoint_path.write_bytes(b"regular")

    with pytest.raises(ValueError, match="Config file not found"):
        OpenValueEstimator.from_pretrained(str(checkpoint_path), device="cpu", use_ema=False)

