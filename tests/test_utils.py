"""Tests for utility helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from open_value_estimator.training import flatten_optimizer_state
from open_value_estimator.utils import (
    _normalize_to_chw,
    _rescale_and_normalize,
    augment_images,
    compute_explained_variance,
    compute_td_error_magnitude,
    load_checkpoint,
    load_checkpoint_metadata,
    load_model_weights,
    preprocess_batch,
    siglip_preprocess,
)


def build_optimizer_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler) -> None:
    tensors = {f"model.{k}": v.clone() for k, v in model.state_dict().items()}
    optimizer_tensors, optimizer_metadata = flatten_optimizer_state(optimizer)
    tensors.update({k: v.clone() for k, v in optimizer_tensors.items()})
    metadata = {
        "step": "7",
        "scheduler_state_dict": json.dumps(scheduler.state_dict()),
    }
    metadata.update(optimizer_metadata)
    save_file(tensors, path, metadata=metadata)


def test_load_checkpoint_metadata_reads_safetensors_metadata(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.safetensors"
    save_file({"tensor": torch.tensor([1.0])}, path, metadata={"step": "3", "name": "test"})

    metadata = load_checkpoint_metadata(path)

    assert metadata == {"step": "3", "name": "test"}


def test_load_checkpoint_metadata_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_checkpoint_metadata(tmp_path / "missing.safetensors")


@pytest.mark.parametrize("use_model_prefix", [True, False])
def test_load_model_weights_supports_prefixed_and_plain_checkpoints(tmp_path: Path, use_model_prefix: bool) -> None:
    source = nn.Linear(2, 1)
    with torch.no_grad():
        source.weight.fill_(3.0)
        source.bias.fill_(4.0)

    tensors = {}
    for key, value in source.state_dict().items():
        tensors[f"model.{key}" if use_model_prefix else key] = value.clone()
    tensors["optimizer.0.exp_avg"] = torch.ones(1)

    checkpoint_path = tmp_path / f"weights_{use_model_prefix}.safetensors"
    save_file(tensors, checkpoint_path)

    target = nn.Linear(2, 1)
    with torch.no_grad():
        target.weight.zero_()
        target.bias.zero_()

    load_model_weights(checkpoint_path, target, device="cpu")

    assert torch.allclose(target.weight, source.weight)
    assert torch.allclose(target.bias, source.bias)


def test_load_checkpoint_restores_model_optimizer_scheduler_and_step(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5 if step > 0 else 1.0)

    x = torch.tensor([[1.0, 2.0]])
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    checkpoint_path = tmp_path / "checkpoint.safetensors"
    build_optimizer_checkpoint(checkpoint_path, model, optimizer, scheduler)

    restored_model = nn.Linear(2, 1)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.1)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer, lambda step: 0.5 if step > 0 else 1.0
    )

    result = load_checkpoint(
        checkpoint_path,
        restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        device=torch.device("cpu"),
    )

    assert result["step"] == 7
    assert torch.allclose(restored_model.weight, model.weight)
    assert torch.allclose(restored_model.bias, model.bias)
    assert restored_optimizer.state_dict()["state"]
    assert restored_scheduler.state_dict()["last_epoch"] == scheduler.state_dict()["last_epoch"]


def test_normalize_to_chw_converts_hwc() -> None:
    image = torch.arange(2 * 3 * 3, dtype=torch.float32).reshape(2, 3, 3)

    normalized = _normalize_to_chw(image)

    assert normalized.shape == (3, 2, 3)


def test_normalize_to_chw_leaves_chw_unchanged() -> None:
    image = torch.randn(3, 4, 5)

    normalized = _normalize_to_chw(image)

    assert torch.equal(normalized, image)


def test_normalize_to_chw_expands_grayscale_to_rgb() -> None:
    image = torch.ones(1, 2, 2)

    normalized = _normalize_to_chw(image)

    assert normalized.shape == (3, 2, 2)
    assert torch.equal(normalized[0], normalized[1])
    assert torch.equal(normalized[1], normalized[2])


def test_normalize_to_chw_rejects_wrong_dimensions() -> None:
    with pytest.raises(ValueError, match="Expected 3D tensor"):
        _normalize_to_chw(torch.ones(2, 2))


def test_rescale_and_normalize_uint8_maps_zero_and_max() -> None:
    tensor = torch.tensor([0, 255], dtype=torch.uint8)

    normalized = _rescale_and_normalize(tensor)

    assert normalized.dtype == torch.float32
    assert torch.allclose(normalized, torch.tensor([-1.0, 1.0]))


def test_rescale_and_normalize_float_uses_siglip_formula() -> None:
    tensor = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)

    normalized = _rescale_and_normalize(tensor)

    assert torch.allclose(normalized, torch.tensor([-1.0, 0.0, 1.0]))


def test_siglip_preprocess_batched_tensor_contract() -> None:
    images = torch.tensor(
        [[[[0, 255], [255, 0]]]],
        dtype=torch.uint8,
    )

    processed = siglip_preprocess(images, size=4, device="cpu", dtype=torch.float32)

    assert processed.shape == (1, 3, 4, 4)
    assert processed.dtype == torch.float32
    assert processed.min().item() >= -1.0
    assert processed.max().item() <= 1.0


def test_siglip_preprocess_list_path_batches_images() -> None:
    image = torch.ones(3, 2, 2, dtype=torch.float32)

    processed = siglip_preprocess([image, image * 0.5], size=4, device="cpu", dtype=torch.float32)

    assert processed.shape == (2, 3, 4, 4)


def test_preprocess_batch_assembles_camera_keys_in_requested_order() -> None:
    batch = {
        "observation.images.top": torch.ones(2, 3, 4, 4),
        "observation.images.wrist": torch.full((2, 3, 4, 4), 2.0),
        "task": ["a", "b"],
    }

    processed = preprocess_batch(
        batch,
        device=torch.device("cpu"),
        image_keys=["observation.images.wrist", "observation.images.top"],
    )

    assert processed["observation.images"].shape == (2, 2, 3, 4, 4)
    assert torch.equal(processed["observation.images"][:, 0], torch.full((2, 3, 4, 4), 2.0))
    assert torch.equal(processed["observation.images"][:, 1], torch.ones(2, 3, 4, 4))
    assert "observation.images.top" not in processed
    assert "observation.images.wrist" not in processed


def test_preprocess_batch_uses_last_timestep_for_5d_images() -> None:
    batch = {
        "observation.images.top": torch.stack(
            [torch.zeros(2, 3, 4, 4), torch.ones(2, 3, 4, 4)],
            dim=1,
        ),
    }

    processed = preprocess_batch(batch, device=torch.device("cpu"))

    assert torch.equal(processed["observation.images"][:, 0], torch.ones(2, 3, 4, 4))


def test_preprocess_batch_missing_image_key_raises() -> None:
    batch = {"observation.images.top": torch.ones(2, 3, 4, 4)}

    with pytest.raises(ValueError, match="Requested camera view keys not found"):
        preprocess_batch(
            batch,
            device=torch.device("cpu"),
            image_keys=["observation.images.wrist"],
        )


def test_preprocess_batch_quantile_normalization_maps_expected_range() -> None:
    batch = {
        "observation.state": torch.tensor([[0.0, 5.0, 10.0]]),
    }
    state_stats = {
        "q01": torch.tensor([0.0, 0.0, 0.0]),
        "q99": torch.tensor([10.0, 10.0, 10.0]),
    }

    processed = preprocess_batch(batch, device=torch.device("cpu"), state_stats=state_stats)

    assert torch.allclose(processed["observation.state"], torch.tensor([[-1.0, 0.0, 1.0]]))


def test_preprocess_batch_zero_range_quantiles_do_not_create_nan() -> None:
    batch = {
        "observation.state": torch.tensor([[1.0, 2.0]]),
    }
    state_stats = {
        "q01": torch.tensor([1.0, 2.0]),
        "q99": torch.tensor([1.0, 2.0]),
    }

    processed = preprocess_batch(batch, device=torch.device("cpu"), state_stats=state_stats)

    assert not torch.isnan(processed["observation.state"]).any()


def test_augment_images_preserves_shape_and_float_range() -> None:
    torch.manual_seed(0)
    images = torch.rand(2, 3, 3, 16, 16)

    augmented = augment_images(images)

    assert augmented.shape == images.shape
    assert augmented.dtype == torch.float32
    assert augmented.min().item() >= 0.0
    assert augmented.max().item() <= 1.0


def test_augment_images_preserves_uint8_dtype_and_range() -> None:
    torch.manual_seed(0)
    images = torch.randint(0, 256, (1, 2, 3, 16, 16), dtype=torch.uint8)

    augmented = augment_images(images)

    assert augmented.shape == images.shape
    assert augmented.dtype == torch.uint8
    assert augmented.min().item() >= 0
    assert augmented.max().item() <= 255


def test_compute_td_error_magnitude_returns_rmse() -> None:
    value = compute_td_error_magnitude(torch.tensor([1.0, 3.0]), torch.tensor([1.0, 1.0]))

    assert value == pytest.approx(2.0**0.5)


def test_compute_explained_variance_handles_perfect_predictions() -> None:
    value = compute_explained_variance(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0]))

    assert value == pytest.approx(1.0)


def test_compute_explained_variance_returns_zero_for_zero_variance_target() -> None:
    value = compute_explained_variance(torch.tensor([1.0, 1.0]), torch.tensor([2.0, 2.0]))

    assert value == pytest.approx(0.0)

