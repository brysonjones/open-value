"""Tests for training utilities."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from open_value_estimator.training import flatten_optimizer_state, lr_schedule_multiplier, update_ema


def test_update_ema_known_decay() -> None:
    ema_model = nn.Linear(1, 1, bias=False)
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        ema_model.weight.fill_(0.0)
        model.weight.fill_(10.0)

    update_ema(ema_model, model, decay=0.9)

    assert ema_model.weight.item() == pytest.approx(1.0)


def test_update_ema_decay_one_preserves_ema() -> None:
    ema_model = nn.Linear(1, 1, bias=False)
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        ema_model.weight.fill_(2.0)
        model.weight.fill_(10.0)

    update_ema(ema_model, model, decay=1.0)

    assert ema_model.weight.item() == pytest.approx(2.0)


def test_update_ema_decay_zero_copies_model() -> None:
    ema_model = nn.Linear(1, 1, bias=False)
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        ema_model.weight.fill_(2.0)
        model.weight.fill_(10.0)

    update_ema(ema_model, model, decay=0.0)

    assert ema_model.weight.item() == pytest.approx(10.0)


def test_flatten_optimizer_state_produces_tensors_and_metadata() -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()

    tensors, metadata = flatten_optimizer_state(optimizer)

    assert tensors
    assert any(key.startswith("optimizer.") for key in tensors)
    assert "param_groups" in metadata


def test_lr_schedule_multiplier_starts_at_zero() -> None:
    assert lr_schedule_multiplier(0, warmup_steps=10, total_steps=100) == pytest.approx(0.0)


def test_lr_schedule_multiplier_is_one_at_warmup_boundary() -> None:
    assert lr_schedule_multiplier(10, warmup_steps=10, total_steps=100) == pytest.approx(1.0)


def test_lr_schedule_multiplier_decays_to_zero_at_final_step() -> None:
    assert lr_schedule_multiplier(100, warmup_steps=10, total_steps=100) == pytest.approx(0.0)
