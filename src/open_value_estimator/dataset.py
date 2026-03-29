"""Dataset wrapper with precomputed value targets."""

import logging

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler

from open_value_estimator.config import DataConfig


def get_scalar_feature_tensor(
    dataset: LeRobotDataset,
    feature_name: str,
) -> torch.Tensor:
    """Load a scalar per-frame dataset feature as a flat float tensor."""

    tensor = torch.as_tensor(dataset.hf_dataset[feature_name], dtype=torch.float32)
    if tensor.ndim == 0:
        return tensor.unsqueeze(0)
    if tensor.ndim == 1:
        return tensor
    if any(dim != 1 for dim in tensor.shape[1:]):
        raise ValueError(
            f"Feature '{feature_name}' must be scalar per frame, got shape {tuple(tensor.shape)}"
        )
    return tensor.reshape(tensor.shape[0])


def get_episode_boundaries(meta) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get episode start, end, and length tensors from LeRobot metadata."""

    total_episodes = meta.total_episodes
    ep_starts = torch.tensor(
        [meta.episodes[i]["dataset_from_index"] for i in range(total_episodes)],
        dtype=torch.long,
    )
    ep_ends = torch.tensor(
        [meta.episodes[i]["dataset_to_index"] for i in range(total_episodes)],
        dtype=torch.long,
    )
    ep_lengths = ep_ends - ep_starts
    return ep_starts, ep_ends, ep_lengths


def compute_task_max_lengths(meta) -> tuple[dict[str, int], list[list[str]]]:
    """Compute max episode length per task from dataset metadata."""

    task_max_lengths: dict[str, int] = {}
    episode_tasks: list[list[str]] = []

    for ep_idx in range(meta.total_episodes):
        ep_meta = meta.episodes[ep_idx]
        ep_len = ep_meta["length"]
        tasks = ep_meta["tasks"]
        episode_tasks.append(tasks)

        for task in tasks:
            task_max_lengths[task] = max(task_max_lengths.get(task, ep_len), ep_len)

    return task_max_lengths, episode_tasks


def build_terminal_reward_sequence(
    episode_length: int,
    is_success: bool,
    fail_penalty: float,
) -> torch.Tensor:
    """Construct the shaped reward sequence used by terminal reward mode."""

    rewards = torch.full((episode_length,), -1.0)
    rewards[-1] = 0.0 if is_success else -fail_penalty
    return rewards


def normalize_step_rewards(rewards: torch.Tensor, max_length_for_task: int) -> torch.Tensor:
    """Normalize per-step rewards using the task-specific episode-length scale."""

    normalizer = max(max_length_for_task - 1, 1)
    return rewards.float() / normalizer


def compute_normalized_step_rewards(
    meta,
    rewards_all: torch.Tensor,
    episode_tasks: list[list[str]],
    task_max_lengths: dict[str, int],
    fail_penalty: float,
    precomputed_rewards: bool,
) -> torch.Tensor:
    """Precompute normalized per-step rewards for all frames."""

    total_frames = meta.total_frames
    total_episodes = meta.total_episodes
    step_rewards = torch.zeros(total_frames, dtype=torch.float32)
    ep_starts, ep_ends, ep_lengths = get_episode_boundaries(meta)

    if precomputed_rewards:
        logging.info("Using precomputed per-step rewards (dense reward mode)")
        for ep_idx in range(total_episodes):
            task = episode_tasks[ep_idx][0]
            task_max_len = task_max_lengths[task]
            start_idx = ep_starts[ep_idx].item()
            end_idx = ep_ends[ep_idx].item()
            ep_rewards = rewards_all[start_idx:end_idx]
            step_rewards[start_idx:end_idx] = normalize_step_rewards(ep_rewards, task_max_len)
        return step_rewards

    last_frame_indices = ep_ends - 1
    last_frame_rewards = rewards_all[last_frame_indices]
    episode_success = last_frame_rewards > 0

    logging.info(f"Last frame rewards sample (first 10): {last_frame_rewards[:10].tolist()}")
    logging.info(f"Last frame rewards unique values: {last_frame_rewards.unique().tolist()}")

    n_success = episode_success.sum().item()
    n_failure = total_episodes - n_success
    success_lengths = ep_lengths[episode_success].tolist()
    failure_lengths = ep_lengths[~episode_success].tolist()

    for ep_idx in range(total_episodes):
        task = episode_tasks[ep_idx][0]
        task_max_len = task_max_lengths[task]
        ep_rewards = build_terminal_reward_sequence(
            episode_length=ep_lengths[ep_idx].item(),
            is_success=bool(episode_success[ep_idx].item()),
            fail_penalty=fail_penalty,
        )
        start_idx = ep_starts[ep_idx].item()
        end_idx = ep_ends[ep_idx].item()
        step_rewards[start_idx:end_idx] = normalize_step_rewards(ep_rewards, task_max_len)

    logging.info(f"Value targets: {n_success}/{total_episodes} success, {n_failure}/{total_episodes} failure")
    if success_lengths:
        avg_success_len = sum(success_lengths) / len(success_lengths)
        logging.info(
            f"  Success episodes: avg_len={avg_success_len:.1f}, "
            f"range=[{min(success_lengths)}, {max(success_lengths)}]"
        )
    if failure_lengths:
        avg_failure_len = sum(failure_lengths) / len(failure_lengths)
        logging.info(
            f"  Failure episodes: avg_len={avg_failure_len:.1f}, "
            f"range=[{min(failure_lengths)}, {max(failure_lengths)}]"
        )

    return step_rewards


def compute_value_targets_from_step_rewards(meta, step_rewards: torch.Tensor) -> torch.Tensor:
    """Compute clamped return targets from normalized step rewards."""

    targets = torch.zeros(meta.total_frames, dtype=torch.float32)
    ep_starts, ep_ends, _ = get_episode_boundaries(meta)

    for start_idx, end_idx in zip(ep_starts.tolist(), ep_ends.tolist(), strict=False):
        returns = step_rewards[start_idx:end_idx].flip(0).cumsum(0).flip(0)
        targets[start_idx:end_idx] = returns.clamp(-1.0, 0.0)

    return targets


class ValueDataset(LeRobotDataset):
    """LeRobotDataset subclass that precomputes value targets.

    Inherits all LeRobotDataset functionality and adds value target computation.

    Success/Failure is determined by the last frame reward:
        - Success: reward > 0 at last frame (+1)
        - Failure: reward < 0 at last frame (-1)

    Value targets follow:
        - Success (t=T): 0
        - Failure (t=T): -fail_penalty
        - Intermediate steps: -1

    Returns (R_t) are computed as cumulative reward from t to T,
    then normalized to (-1, 0) based on task-specific max episode length.
    Each task has its own max length used for normalization.
    """

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.fail_penalty = cfg.fail_penalty

        # Initialize the parent LeRobotDataset
        logging.info(f"Loading dataset: {cfg.repo_id}")
        super().__init__(
            repo_id=cfg.repo_id,
            root=cfg.root,
        )

        logging.info(f"Total episodes: {self.meta.total_episodes}")
        logging.info(f"Total frames: {self.meta.total_frames}")

        # Compute per-task max episode lengths for normalization
        self.task_max_lengths, self.episode_tasks = compute_task_max_lengths(self.meta)
        self._log_task_stats()

        rewards_all = get_scalar_feature_tensor(self, "reward")
        self.normalized_step_rewards = compute_normalized_step_rewards(
            meta=self.meta,
            rewards_all=rewards_all,
            episode_tasks=self.episode_tasks,
            task_max_lengths=self.task_max_lengths,
            fail_penalty=self.fail_penalty,
            precomputed_rewards=cfg.precomputed_rewards,
        )
        self.value_targets = compute_value_targets_from_step_rewards(self.meta, self.normalized_step_rewards)
        logging.info("Precomputed value targets for all frames")
        logging.info(f"  Target range: [{self.value_targets.min():.3f}, {self.value_targets.max():.3f}]")

        # Extract observation.state normalization stats from metadata
        self.state_stats = self._get_state_stats()

    def _log_task_stats(self) -> None:
        """Log per-task statistics."""
        logging.info(f"Found {len(self.task_max_lengths)} unique tasks:")
        for task, max_len in sorted(self.task_max_lengths.items()):
            task_eps = sum(1 for tasks in self.episode_tasks if task in tasks)
            logging.info(f"  '{task}': max_length={max_len}, episodes={task_eps}")

    def _get_state_stats(self) -> dict[str, torch.Tensor] | None:
        """Get observation.state quantile normalization stats.

        Tries to read q01/q99 from dataset metadata first. If unavailable,
        computes them from the full dataset (one-time cost at init).

        Returns:
            Dict with 'q01' and 'q99' tensors, or None if state is unavailable.
        """
        # Try to read precomputed quantiles from dataset metadata
        if hasattr(self.meta, "stats") and "observation.state" in self.meta.stats:
            state_stats = self.meta.stats["observation.state"]
            if "q01" in state_stats and "q99" in state_stats:
                q01 = state_stats["q01"]
                q99 = state_stats["q99"]
                if not isinstance(q01, torch.Tensor):
                    q01 = torch.tensor(q01)
                if not isinstance(q99, torch.Tensor):
                    q99 = torch.tensor(q99)
                logging.info(f"Loaded observation.state quantiles from metadata: dim={len(q01)}")
                logging.info(f"  q01={q01.tolist()}")
                logging.info(f"  q99={q99.tolist()}")
                return {"q01": q01, "q99": q99}

        # Fallback: compute quantiles from full dataset
        if "observation.state" not in self.hf_dataset.column_names:
            logging.info("No observation.state in dataset")
            return None

        logging.info("Computing observation.state quantiles from data (q01/q99)...")
        all_states = torch.tensor(self.hf_dataset["observation.state"])  # [N, state_dim]
        q01 = torch.quantile(all_states.float(), 0.01, dim=0)
        q99 = torch.quantile(all_states.float(), 0.99, dim=0)
        logging.info(f"Computed observation.state quantiles: dim={len(q01)}")
        logging.info(f"  q01={q01.tolist()}")
        logging.info(f"  q99={q99.tolist()}")
        return {"q01": q01, "q99": q99}

    def __getitem__(self, idx: int) -> dict:
        """Get a sample with value target included.

        Args:
            idx: Frame index.

        Returns:
            Sample dict from underlying dataset with 'value_target' added.
        """
        sample = super().__getitem__(idx)
        sample["value_target"] = self.value_targets[idx]
        return sample

    def create_dataloader(self, batch_size: int | None = None) -> torch.utils.data.DataLoader:
        """Create a DataLoader with episode-aware sampling.

        Args:
            batch_size: Override batch size (defaults to cfg.batch_size).

        Returns:
            DataLoader configured with EpisodeAwareSampler.
        """
        sampler = EpisodeAwareSampler(
            self.meta.episodes["dataset_from_index"],
            self.meta.episodes["dataset_to_index"],
            episode_indices_to_use=self.episodes,
            drop_n_last_frames=self.cfg.drop_n_last_frames,
            shuffle=self.cfg.shuffle,
        )

        dataloader = torch.utils.data.DataLoader(
            self,
            batch_size=batch_size or self.cfg.batch_size,
            sampler=sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=2 if self.cfg.num_workers > 0 else None,
        )

        return dataloader
