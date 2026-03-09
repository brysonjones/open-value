"""Tests for dataset reward shaping helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from open_value_estimator.dataset import (
    ValueDataset,
    build_terminal_reward_sequence,
    compute_normalized_step_rewards,
    compute_task_max_lengths,
    compute_value_targets_from_step_rewards,
    get_scalar_feature_tensor,
    get_episode_boundaries,
    normalize_step_rewards,
)


class DummyHFDataset(dict):
    @property
    def column_names(self) -> list[str]:
        return list(self.keys())


def make_meta(episodes: list[dict], stats: dict | None = None) -> SimpleNamespace:
    total_frames = sum(ep["dataset_to_index"] - ep["dataset_from_index"] for ep in episodes)
    return SimpleNamespace(
        episodes=episodes,
        total_episodes=len(episodes),
        total_frames=total_frames,
        stats=stats or {},
    )


def test_get_scalar_feature_tensor_accepts_scalar() -> None:
    dataset = SimpleNamespace(hf_dataset={"reward": 3.0})

    values = get_scalar_feature_tensor(dataset, "reward")

    assert values.dtype == torch.float32
    assert torch.equal(values, torch.tensor([3.0]))


def test_get_scalar_feature_tensor_accepts_column_vector() -> None:
    dataset = SimpleNamespace(hf_dataset={"reward": [[1.0], [2.0]]})

    values = get_scalar_feature_tensor(dataset, "reward")

    assert torch.equal(values, torch.tensor([1.0, 2.0]))


def test_get_scalar_feature_tensor_rejects_non_scalar_feature() -> None:
    dataset = SimpleNamespace(hf_dataset={"reward": [[1.0, 2.0], [3.0, 4.0]]})

    with pytest.raises(ValueError, match="must be scalar per frame"):
        get_scalar_feature_tensor(dataset, "reward")


def test_get_episode_boundaries_returns_start_end_and_lengths() -> None:
    meta = make_meta(
        [
            {"dataset_from_index": 0, "dataset_to_index": 3, "length": 3, "tasks": ["task_a"]},
            {"dataset_from_index": 3, "dataset_to_index": 7, "length": 4, "tasks": ["task_b"]},
        ]
    )

    starts, ends, lengths = get_episode_boundaries(meta)

    assert torch.equal(starts, torch.tensor([0, 3]))
    assert torch.equal(ends, torch.tensor([3, 7]))
    assert torch.equal(lengths, torch.tensor([3, 4]))


def test_build_terminal_reward_sequence_sets_success_terminal_to_zero() -> None:
    rewards = build_terminal_reward_sequence(episode_length=4, is_success=True, fail_penalty=10.0)

    assert torch.equal(rewards, torch.tensor([-1.0, -1.0, -1.0, 0.0]))


def test_build_terminal_reward_sequence_sets_failure_terminal_to_penalty() -> None:
    rewards = build_terminal_reward_sequence(episode_length=4, is_success=False, fail_penalty=10.0)

    assert torch.equal(rewards, torch.tensor([-1.0, -1.0, -1.0, -10.0]))


def test_normalize_step_rewards_avoids_division_by_zero() -> None:
    rewards = normalize_step_rewards(torch.tensor([-1.0]), max_length_for_task=1)

    assert torch.equal(rewards, torch.tensor([-1.0]))


def test_compute_normalized_step_rewards_terminal_mode_matches_contract() -> None:
    meta = make_meta(
        [
            {"dataset_from_index": 0, "dataset_to_index": 3, "length": 3, "tasks": ["task_a"]},
            {"dataset_from_index": 3, "dataset_to_index": 5, "length": 2, "tasks": ["task_b"]},
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

    assert torch.allclose(step_rewards, torch.tensor([-0.5, -0.5, 0.0, -1.0, -3.0]))


def test_compute_normalized_step_rewards_dense_mode_uses_dataset_rewards() -> None:
    meta = make_meta(
        [{"dataset_from_index": 0, "dataset_to_index": 4, "length": 4, "tasks": ["task_a"]}]
    )
    rewards_all = torch.tensor([0.0, -1.0, -2.0, -3.0])

    step_rewards = compute_normalized_step_rewards(
        meta=meta,
        rewards_all=rewards_all,
        episode_tasks=[["task_a"]],
        task_max_lengths={"task_a": 4},
        fail_penalty=10.0,
        precomputed_rewards=True,
    )

    assert torch.allclose(step_rewards, torch.tensor([0.0, -1.0 / 3.0, -2.0 / 3.0, -1.0]))


def test_compute_value_targets_from_step_rewards_backward_cumsum_and_clamp() -> None:
    meta = make_meta(
        [
            {"dataset_from_index": 0, "dataset_to_index": 3, "length": 3, "tasks": ["task_a"]},
            {"dataset_from_index": 3, "dataset_to_index": 5, "length": 2, "tasks": ["task_b"]},
        ]
    )
    step_rewards = torch.tensor([-0.5, -0.5, 0.0, -1.0, -3.0])

    targets = compute_value_targets_from_step_rewards(meta, step_rewards)

    assert torch.allclose(targets, torch.tensor([-1.0, -0.5, 0.0, -1.0, -1.0]))


def test_compute_task_max_lengths_collects_all_episode_tasks() -> None:
    meta = make_meta(
        [
            {"dataset_from_index": 0, "dataset_to_index": 3, "length": 3, "tasks": ["task_a"]},
            {"dataset_from_index": 3, "dataset_to_index": 8, "length": 5, "tasks": ["task_a", "task_b"]},
        ]
    )

    task_max_lengths, episode_tasks = compute_task_max_lengths(meta)

    assert task_max_lengths == {"task_a": 5, "task_b": 5}
    assert episode_tasks == [["task_a"], ["task_a", "task_b"]]


def test_get_state_stats_reads_quantiles_from_metadata() -> None:
    dataset = ValueDataset.__new__(ValueDataset)
    dataset.meta = SimpleNamespace(
        stats={"observation.state": {"q01": [1.0, 2.0], "q99": [3.0, 4.0]}}
    )
    dataset.hf_dataset = DummyHFDataset()

    stats = ValueDataset._get_state_stats(dataset)

    assert torch.equal(stats["q01"], torch.tensor([1.0, 2.0]))
    assert torch.equal(stats["q99"], torch.tensor([3.0, 4.0]))


def test_get_state_stats_computes_quantiles_when_metadata_missing() -> None:
    dataset = ValueDataset.__new__(ValueDataset)
    dataset.meta = SimpleNamespace(stats={})
    dataset.hf_dataset = DummyHFDataset(
        {
            "observation.state": [
                [0.0, 10.0],
                [1.0, 11.0],
                [2.0, 12.0],
                [3.0, 13.0],
            ]
        }
    )

    stats = ValueDataset._get_state_stats(dataset)

    expected = torch.tensor([0.03, 10.03])
    assert torch.allclose(stats["q01"], expected, atol=1e-5)


def test_get_state_stats_returns_none_without_state_feature() -> None:
    dataset = ValueDataset.__new__(ValueDataset)
    dataset.meta = SimpleNamespace(stats={})
    dataset.hf_dataset = DummyHFDataset({"reward": [0.0, 1.0]})

    assert ValueDataset._get_state_stats(dataset) is None
