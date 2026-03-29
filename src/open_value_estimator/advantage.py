"""Offline advantage dataset generation and threshold helpers."""

import argparse
import copy
import logging
import math
from typing import Sequence

import numpy as np
import torch
from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import write_stats
from tqdm import tqdm

from open_value_estimator.config import AdvantageMode, Config, load_config, parse_cli_overrides
from open_value_estimator.dataset import (
    ValueDataset,
    get_scalar_feature_tensor,
    get_episode_boundaries,
)
from open_value_estimator.utils import preprocess_batch
from open_value_estimator.value_estimator import OpenValueEstimator

ADVANTAGE_FEATURE = "advantage"
ADVANTAGE_POSITIVE = "Advantage: Positive"
ADVANTAGE_NEGATIVE = "Advantage: Negative"


def build_advantage_feature_info() -> dict[str, object]:
    """Return LeRobot feature metadata for the scalar advantage column."""

    return {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    }


def format_quantile_key(quantile: float) -> str:
    """Format a quantile in [0, 1] as a LeRobot-style stat key."""

    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"Quantile must be in [0, 1], got {quantile}")

    percentile = quantile * 100.0
    rounded = int(round(percentile))
    if not math.isclose(percentile, rounded, abs_tol=1e-6):
        raise ValueError(
            f"Quantiles must map to whole-number percent keys, got {quantile}"
        )
    return f"q{rounded:02d}"


def compute_advantage_stats(
    advantages: torch.Tensor,
    quantiles: Sequence[float],
) -> dict[str, np.ndarray]:
    """Compute global summary stats for the scalar advantage feature."""

    flat = advantages.detach().cpu().float().view(-1)
    stats: dict[str, np.ndarray] = {
        "min": np.array([flat.min().item()], dtype=np.float32),
        "max": np.array([flat.max().item()], dtype=np.float32),
        "mean": np.array([flat.mean().item()], dtype=np.float32),
        "std": np.array([flat.std(unbiased=False).item()], dtype=np.float32),
        "count": np.array([flat.numel()], dtype=np.int64),
    }

    for quantile in quantiles:
        key = format_quantile_key(float(quantile))
        stats[key] = np.array(
            [torch.quantile(flat, float(quantile)).item()],
            dtype=np.float32,
        )

    return stats


@torch.no_grad()
def infer_dataset_values(
    model: OpenValueEstimator,
    dataset: ValueDataset,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Run value inference over every frame and scatter outputs by global index."""

    dataloader = dataset.create_dataloader(batch_size=batch_size)
    values = torch.full((dataset.meta.total_frames,), float("nan"), dtype=torch.float32)

    model.eval()
    for batch in tqdm(dataloader, desc="Estimating values"):
        batch = preprocess_batch(
            batch,
            device=device,
            state_stats=dataset.state_stats,
            augment=False,
        )
        indices = batch["index"].view(-1).long().detach().cpu()
        logits = model(batch)
        batch_values = model.get_expected_value(logits).float().detach().cpu()
        values[indices] = batch_values

    if torch.isnan(values).any():
        raise RuntimeError("Value inference did not produce predictions for every frame")

    return values


def compute_full_episode_advantages(
    predicted_values: torch.Tensor,
    value_targets: torch.Tensor,
) -> torch.Tensor:
    """Compute full-episode advantages using empirical returns to the episode end."""

    return value_targets.float().detach().cpu() - predicted_values.float().detach().cpu()


def compute_n_step_advantages(
    predicted_values: torch.Tensor,
    step_rewards: torch.Tensor,
    meta,
    n_step: int,
) -> torch.Tensor:
    """Compute n-step lookahead advantages for a full dataset."""

    if n_step < 1:
        raise ValueError(f"n_step must be >= 1, got {n_step}")

    values = predicted_values.float().detach().cpu()
    rewards = step_rewards.float().detach().cpu()
    advantages = torch.empty_like(values)
    ep_starts, ep_ends, _ = get_episode_boundaries(meta)

    for start_idx, end_idx in zip(ep_starts.tolist(), ep_ends.tolist(), strict=False):
        ep_rewards = rewards[start_idx:end_idx]
        ep_values = values[start_idx:end_idx]
        ep_len = ep_rewards.shape[0]
        local_indices = torch.arange(ep_len, dtype=torch.long)
        window_end = torch.clamp(local_indices + n_step, max=ep_len)

        reward_prefix = torch.cat([torch.zeros(1, dtype=ep_rewards.dtype), ep_rewards.cumsum(dim=0)])
        reward_sums = reward_prefix[window_end] - reward_prefix[local_indices]

        bootstrap = torch.zeros(ep_len, dtype=ep_values.dtype)
        valid_bootstrap = (local_indices + n_step) < ep_len
        bootstrap[valid_bootstrap] = ep_values[local_indices[valid_bootstrap] + n_step]

        advantages[start_idx:end_idx] = reward_sums + bootstrap - ep_values

    return advantages


def compute_task_advantage_thresholds_from_arrays(
    advantages: Sequence[float] | torch.Tensor,
    tasks: Sequence[str],
    percentile: float,
) -> dict[str, float]:
    """Compute per-task percentile thresholds from precomputed advantages."""

    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"Percentile must be in [0, 100], got {percentile}")

    flat_advantages = torch.as_tensor(advantages, dtype=torch.float32).view(-1)
    if len(tasks) != flat_advantages.numel():
        raise ValueError(
            f"Expected one task per advantage value, got {len(tasks)} tasks and "
            f"{flat_advantages.numel()} values"
        )

    grouped: dict[str, list[float]] = {}
    for value, task in zip(flat_advantages.tolist(), tasks, strict=False):
        grouped.setdefault(task, []).append(value)

    thresholds: dict[str, float] = {}
    quantile = percentile / 100.0
    for task, values in grouped.items():
        thresholds[task] = torch.quantile(
            torch.tensor(values, dtype=torch.float32),
            quantile,
        ).item()

    return thresholds


def binarize_advantages(
    advantages: Sequence[float] | torch.Tensor,
    task_thresholds: dict[str, float],
    tasks: Sequence[str],
) -> list[str]:
    """Convert scalar advantage values into per-task binary labels."""

    flat_advantages = torch.as_tensor(advantages, dtype=torch.float32).view(-1)
    if len(tasks) != flat_advantages.numel():
        raise ValueError(
            f"Expected one task per advantage value, got {len(tasks)} tasks and "
            f"{flat_advantages.numel()} values"
        )

    missing_tasks = sorted(set(tasks) - set(task_thresholds))
    if missing_tasks:
        raise ValueError(
            "Missing task thresholds for: " + ", ".join(missing_tasks)
        )

    labels = []
    for advantage, task in zip(flat_advantages.tolist(), tasks, strict=False):
        if advantage >= task_thresholds[task]:
            labels.append(ADVANTAGE_POSITIVE)
        else:
            labels.append(ADVANTAGE_NEGATIVE)
    return labels


def compute_task_advantage_thresholds(
    dataset: LeRobotDataset,
    percentile: float,
) -> dict[str, float]:
    """Compute per-task advantage thresholds on demand from a saved dataset."""

    if ADVANTAGE_FEATURE not in dataset.hf_dataset.column_names:
        raise ValueError(
            f"Dataset does not contain '{ADVANTAGE_FEATURE}'. "
            "Run compute_dataset_advantages first."
        )

    advantages = get_scalar_feature_tensor(dataset, ADVANTAGE_FEATURE)
    task_indices = torch.as_tensor(dataset.hf_dataset["task_index"])
    if task_indices.ndim > 1:
        task_indices = task_indices.reshape(task_indices.shape[0], -1)[:, 0]
    task_indices = task_indices.view(-1).long()

    tasks = [dataset.meta.tasks.iloc[int(task_idx)].name for task_idx in task_indices.tolist()]
    return compute_task_advantage_thresholds_from_arrays(advantages, tasks, percentile)


def write_advantage_stats(
    dataset: LeRobotDataset,
    advantages: torch.Tensor,
    quantiles: Sequence[float],
) -> None:
    """Persist global advantage stats into the derived dataset metadata."""

    stats = dict(dataset.meta.stats or {})
    stats[ADVANTAGE_FEATURE] = compute_advantage_stats(advantages, quantiles)
    write_stats(stats, dataset.meta.root)
    dataset.meta.stats = stats


def compute_dataset_advantages(cfg: Config) -> LeRobotDataset:
    """Compute a derived LeRobot dataset with a per-frame advantage column."""

    advantage_cfg = cfg.advantage
    if not advantage_cfg.checkpoint:
        raise ValueError("advantage.checkpoint is required")

    if advantage_cfg.mode == AdvantageMode.N_STEP and advantage_cfg.n_step < 1:
        raise ValueError("advantage.n_step must be >= 1")

    data_cfg = copy.deepcopy(cfg.data)
    data_cfg.shuffle = False
    data_cfg.drop_n_last_frames = 0
    data_cfg.augment_images = False

    dataset = ValueDataset(data_cfg)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Loading model from {advantage_cfg.checkpoint}")
    model = OpenValueEstimator.from_pretrained(
        path=advantage_cfg.checkpoint,
        device=device,
        use_ema=advantage_cfg.use_ema,
    )

    predicted_values = infer_dataset_values(
        model=model,
        dataset=dataset,
        batch_size=advantage_cfg.batch_size,
        device=device,
    )

    if advantage_cfg.mode == AdvantageMode.FULL_EPISODE:
        advantages = compute_full_episode_advantages(
            predicted_values=predicted_values,
            value_targets=dataset.value_targets,
        )
    else:
        advantages = compute_n_step_advantages(
            predicted_values=predicted_values,
            step_rewards=dataset.normalized_step_rewards,
            meta=dataset.meta,
            n_step=advantage_cfg.n_step,
        )

    output_repo_id = advantage_cfg.output_repo_id or f"{dataset.repo_id}_advantages"
    logging.info(
        f"Saving derived dataset '{output_repo_id}' "
        f"(mode={advantage_cfg.mode.value}, n_step={advantage_cfg.n_step})"
    )
    new_dataset = add_features(
        dataset=dataset,
        features={
            ADVANTAGE_FEATURE: (
                advantages.view(-1, 1).numpy().astype(np.float32),
                build_advantage_feature_info(),
            )
        },
        output_dir=advantage_cfg.output_root,
        repo_id=output_repo_id,
    )
    write_advantage_stats(
        dataset=new_dataset,
        advantages=advantages,
        quantiles=advantage_cfg.stats_quantiles,
    )
    logging.info(f"Saved derived dataset to {new_dataset.root}")

    return new_dataset


def main() -> None:
    """CLI entry point for offline advantage dataset generation."""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Generate a derived LeRobot dataset with a per-frame `advantage` feature "
            "from a pretrained open value estimator checkpoint."
        ),
        epilog=(
            "Example:\n"
            "  python -m open_value_estimator.advantage "
            "config=configs/advantage.yaml "
            "advantage.checkpoint=outputs/open_value_estimator/my_run/checkpoint_4000.safetensors"
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf-style key=value arguments.",
    )
    parser.add_argument(
        "--config",
        dest="legacy_config",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    override_dict = parse_cli_overrides(args.overrides)
    config_path = override_dict.pop("config", None) or args.legacy_config
    if not config_path or not isinstance(config_path, str):
        parser.error("config=... is required")

    config_dict = load_config(
        config_path=config_path,
        overrides=override_dict,
        require_run_name=False,
        append_run_name_to_output_dir=False,
    )
    logging.info(f"Configuration:\n{config_dict}")

    config = Config.from_dict(config_dict)
    compute_dataset_advantages(config)


if __name__ == "__main__":
    main()
