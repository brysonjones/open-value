from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from open_value_estimator.advantage import (
    ADVANTAGE_NEGATIVE,
    ADVANTAGE_POSITIVE,
    build_advantage_feature_info,
    binarize_advantages,
    compute_advantage_stats,
    compute_full_episode_advantages,
    compute_n_step_advantages,
    compute_task_advantage_thresholds,
    compute_task_advantage_thresholds_from_arrays,
    format_quantile_key,
)
from open_value_estimator.config import AdvantageMode, Config
from open_value_estimator.dataset import (
    compute_normalized_step_rewards,
    compute_task_max_lengths,
    compute_value_targets_from_step_rewards,
)


def make_meta(episodes: list[dict]) -> SimpleNamespace:
    total_frames = sum(ep["dataset_to_index"] - ep["dataset_from_index"] for ep in episodes)
    return SimpleNamespace(
        episodes=episodes,
        total_episodes=len(episodes),
        total_frames=total_frames,
    )


class DummyHFDataset(dict):
    @property
    def column_names(self) -> list[str]:
        return list(self.keys())


def test_config_from_dict_parses_advantage_mode() -> None:
    cfg = Config.from_dict(
        {
            "advantage": {
                "mode": "full_episode",
                "checkpoint": "checkpoint.safetensors",
            }
        }
    )

    assert cfg.advantage.mode == AdvantageMode.FULL_EPISODE
    assert cfg.advantage.checkpoint == "checkpoint.safetensors"


def test_compute_task_max_lengths_collects_all_episode_tasks() -> None:
    meta = make_meta(
        [
            {
                "dataset_from_index": 0,
                "dataset_to_index": 3,
                "length": 3,
                "tasks": ["task_a"],
            },
            {
                "dataset_from_index": 3,
                "dataset_to_index": 8,
                "length": 5,
                "tasks": ["task_a", "task_b"],
            },
        ]
    )

    task_max_lengths, episode_tasks = compute_task_max_lengths(meta)

    assert task_max_lengths == {"task_a": 5, "task_b": 5}
    assert episode_tasks == [["task_a"], ["task_a", "task_b"]]


def test_compute_normalized_step_rewards_terminal_mode_matches_training_scale() -> None:
    meta = make_meta(
        [
            {
                "dataset_from_index": 0,
                "dataset_to_index": 3,
                "length": 3,
                "tasks": ["task_a"],
            },
            {
                "dataset_from_index": 3,
                "dataset_to_index": 5,
                "length": 2,
                "tasks": ["task_b"],
            },
        ]
    )
    rewards_all = torch.tensor([0.0, 0.0, 1.0, 0.0, -1.0])
    episode_tasks = [["task_a"], ["task_b"]]
    task_max_lengths = {"task_a": 3, "task_b": 2}

    step_rewards = compute_normalized_step_rewards(
        meta=meta,
        rewards_all=rewards_all,
        episode_tasks=episode_tasks,
        task_max_lengths=task_max_lengths,
        fail_penalty=3.0,
        precomputed_rewards=False,
    )

    expected = torch.tensor([-0.5, -0.5, 0.0, -1.0, -3.0])
    assert torch.allclose(step_rewards, expected)


def test_compute_value_targets_from_step_rewards_clamps_returns() -> None:
    meta = make_meta(
        [
            {
                "dataset_from_index": 0,
                "dataset_to_index": 3,
                "length": 3,
                "tasks": ["task_a"],
            },
            {
                "dataset_from_index": 3,
                "dataset_to_index": 5,
                "length": 2,
                "tasks": ["task_b"],
            },
        ]
    )
    step_rewards = torch.tensor([-0.5, -0.5, 0.0, -1.0, -3.0])

    value_targets = compute_value_targets_from_step_rewards(meta, step_rewards)

    expected = torch.tensor([-1.0, -0.5, 0.0, -1.0, -1.0])
    assert torch.allclose(value_targets, expected)


def test_compute_full_episode_advantages_subtracts_predicted_values() -> None:
    predicted_values = torch.tensor([-0.7, -0.2, -0.1])
    value_targets = torch.tensor([-1.0, -0.5, 0.0])

    advantages = compute_full_episode_advantages(predicted_values, value_targets)

    expected = torch.tensor([-0.3, -0.3, 0.1])
    assert torch.allclose(advantages, expected)


def test_compute_n_step_advantages_respects_episode_boundaries() -> None:
    meta = make_meta(
        [
            {
                "dataset_from_index": 0,
                "dataset_to_index": 4,
                "length": 4,
                "tasks": ["task_a"],
            },
            {
                "dataset_from_index": 4,
                "dataset_to_index": 7,
                "length": 3,
                "tasks": ["task_b"],
            },
        ]
    )
    predicted_values = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.9, 0.8, 0.7])
    step_rewards = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.05, 0.05, 0.05])

    advantages = compute_n_step_advantages(predicted_values, step_rewards, meta, n_step=2)

    expected = torch.tensor([0.1, 0.3, 0.4, 0.2, -0.1, -0.7, -0.65])
    assert torch.allclose(advantages, expected)


def test_compute_task_advantage_thresholds_from_arrays_groups_by_task() -> None:
    thresholds = compute_task_advantage_thresholds_from_arrays(
        advantages=[0.1, 0.2, 0.9, -0.1],
        tasks=["task_a", "task_a", "task_b", "task_b"],
        percentile=50.0,
    )

    assert thresholds["task_a"] == pytest.approx(0.15)
    assert thresholds["task_b"] == pytest.approx(0.4)


def test_compute_task_advantage_thresholds_reads_saved_dataset_columns() -> None:
    dataset = SimpleNamespace(
        hf_dataset=DummyHFDataset(
            {
                "advantage": [[0.1], [0.2], [0.9], [-0.1]],
                "task_index": [0, 0, 1, 1],
            }
        ),
        meta=SimpleNamespace(
            tasks=pd.DataFrame({"task_index": [0, 1]}, index=["task_a", "task_b"])
        ),
    )

    thresholds = compute_task_advantage_thresholds(dataset, percentile=50.0)

    assert thresholds["task_a"] == pytest.approx(0.15)
    assert thresholds["task_b"] == pytest.approx(0.4)


def test_binarize_advantages_uses_per_task_thresholds() -> None:
    labels = binarize_advantages(
        advantages=[0.1, 0.2, 0.9, -0.1],
        task_thresholds={"task_a": 0.15, "task_b": 0.4},
        tasks=["task_a", "task_a", "task_b", "task_b"],
    )

    assert labels == [
        ADVANTAGE_NEGATIVE,
        ADVANTAGE_POSITIVE,
        ADVANTAGE_POSITIVE,
        ADVANTAGE_NEGATIVE,
    ]


def test_binarize_advantages_requires_threshold_for_each_task() -> None:
    with pytest.raises(ValueError, match="Missing task thresholds"):
        binarize_advantages(
            advantages=[0.1, 0.2],
            task_thresholds={"task_a": 0.0},
            tasks=["task_a", "task_b"],
        )


def test_compute_advantage_stats_includes_quantiles() -> None:
    stats = compute_advantage_stats(
        advantages=torch.tensor([0.0, 1.0, 2.0, 3.0]),
        quantiles=[0.25, 0.50],
    )

    assert stats["count"].tolist() == [4]
    assert np.isclose(stats["min"][0], 0.0)
    assert np.isclose(stats["max"][0], 3.0)
    assert np.isclose(stats["mean"][0], 1.5)
    assert np.isclose(stats["std"][0], np.sqrt(1.25))
    assert np.isclose(stats["q25"][0], 0.75)
    assert np.isclose(stats["q50"][0], 1.5)


def test_build_advantage_feature_info_describes_scalar_float_feature() -> None:
    info = build_advantage_feature_info()

    assert info == {"dtype": "float32", "shape": (1,), "names": None}


@pytest.mark.parametrize(
    ("quantile", "expected"),
    [
        (0.0, "q00"),
        (0.1, "q10"),
        (0.99, "q99"),
        (1.0, "q100"),
    ],
)
def test_format_quantile_key_formats_supported_quantiles(quantile: float, expected: str) -> None:
    assert format_quantile_key(quantile) == expected


def test_format_quantile_key_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="Quantile must be in \\[0, 1\\]"):
        format_quantile_key(1.1)


def test_format_quantile_key_rejects_non_integer_percent_keys() -> None:
    with pytest.raises(ValueError, match="whole-number percent keys"):
        format_quantile_key(0.125)


def test_compute_n_step_advantages_rejects_non_positive_horizon() -> None:
    meta = make_meta(
        [{"dataset_from_index": 0, "dataset_to_index": 1, "length": 1, "tasks": ["task_a"]}]
    )

    with pytest.raises(ValueError, match="n_step must be >= 1"):
        compute_n_step_advantages(torch.tensor([0.0]), torch.tensor([0.0]), meta, n_step=0)


def test_compute_task_advantage_thresholds_from_arrays_rejects_invalid_percentile() -> None:
    with pytest.raises(ValueError, match="Percentile must be in \\[0, 100\\]"):
        compute_task_advantage_thresholds_from_arrays([0.1], ["task_a"], percentile=101.0)


def test_compute_task_advantage_thresholds_from_arrays_requires_matching_lengths() -> None:
    with pytest.raises(ValueError, match="Expected one task per advantage value"):
        compute_task_advantage_thresholds_from_arrays([0.1, 0.2], ["task_a"], percentile=50.0)


def test_binarize_advantages_requires_matching_lengths() -> None:
    with pytest.raises(ValueError, match="Expected one task per advantage value"):
        binarize_advantages([0.1, 0.2], {"task_a": 0.0}, ["task_a"])


def test_compute_task_advantage_thresholds_requires_advantage_column() -> None:
    dataset = SimpleNamespace(
        hf_dataset=DummyHFDataset({"reward": [0.1, 0.2]}),
        meta=SimpleNamespace(tasks=pd.DataFrame({"task_index": [0]}, index=["task_a"])),
    )

    with pytest.raises(ValueError, match="Dataset does not contain 'advantage'"):
        compute_task_advantage_thresholds(dataset, percentile=50.0)
