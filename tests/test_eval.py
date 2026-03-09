"""Tests for evaluation helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from open_value_estimator.eval import (
    compute_grid_layout,
    get_available_camera_view_keys,
    hex_to_bgr,
    resolve_camera_view_keys,
)


class DummyHFDataset(dict):
    @property
    def column_names(self) -> list[str]:
        return list(self.keys())


@pytest.mark.parametrize(
    ("n_cameras", "expected"),
    [
        (1, (1, 1)),
        (2, (2, 1)),
        (3, (2, 2)),
        (4, (2, 2)),
        (5, (3, 2)),
        (6, (3, 2)),
        (7, (4, 2)),
        (8, (4, 2)),
        (9, (4, 3)),
    ],
)
def test_compute_grid_layout_exact_values(n_cameras: int, expected: tuple[int, int]) -> None:
    assert compute_grid_layout(n_cameras) == expected


@pytest.mark.parametrize(
    ("hex_color", "expected"),
    [
        ("#ff0000", (0, 0, 255)),
        ("#00ff00", (0, 255, 0)),
        ("#0000ff", (255, 0, 0)),
    ],
)
def test_hex_to_bgr_known_colors(hex_color: str, expected: tuple[int, int, int]) -> None:
    assert hex_to_bgr(hex_color) == expected


def test_get_available_camera_view_keys_prefers_meta_features() -> None:
    dataset = SimpleNamespace(
        meta=SimpleNamespace(
            features={
                "observation.images.top": {},
                "observation.images.wrist": {},
                "reward": {},
            }
        )
    )

    keys = get_available_camera_view_keys(dataset)

    assert keys == ["observation.images.top", "observation.images.wrist"]


def test_get_available_camera_view_keys_falls_back_to_hf_dataset_columns() -> None:
    dataset = SimpleNamespace(
        meta=SimpleNamespace(features=None),
        hf_dataset=DummyHFDataset(
            {
                "observation.images.front": [1],
                "observation.images.side": [1],
                "task": [1],
            }
        ),
    )

    keys = get_available_camera_view_keys(dataset)

    assert keys == ["observation.images.front", "observation.images.side"]


def test_resolve_camera_view_keys_supports_suffixes_and_dedupes() -> None:
    available = ["observation.images.top", "observation.images.wrist"]

    resolved = resolve_camera_view_keys(["top", "observation.images.wrist", "top"], available)

    assert resolved == ["observation.images.top", "observation.images.wrist"]


def test_resolve_camera_view_keys_unknown_view_raises() -> None:
    with pytest.raises(ValueError, match="Unknown camera views"):
        resolve_camera_view_keys(["missing"], ["observation.images.top"])


def test_resolve_camera_view_keys_ambiguous_suffix_raises() -> None:
    available = ["observation.images.front", "observation.images.front"]

    with pytest.raises(ValueError, match="Ambiguous camera views"):
        resolve_camera_view_keys(["front"], available)
