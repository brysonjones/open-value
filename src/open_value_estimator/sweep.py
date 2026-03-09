"""W&B sweep integration for the open value estimator.

This module provides a clean wrapper around train() for W&B hyperparameter sweeps.
The sweep logic is entirely contained here; train() remains sweep-unaware.

Usage:
    # Create a new sweep and run N trials locally:
    python -m open_value_estimator.sweep \
        config=configs/config.yaml \
        sweep_config=configs/sweeps/lr_batch_sweep.yaml \
        count=20

    # Join an existing sweep:
    python -m open_value_estimator.sweep \
        config=configs/config.yaml \
        sweep_id=<sweep_id> \
        count=5

    # Create sweep without running (prints the sweep ID):
    python -m open_value_estimator.sweep \
        config=configs/config.yaml \
        sweep_config=configs/sweeps/lr_batch_sweep.yaml \
        create_only=true
"""

import argparse
import copy
import logging
from pathlib import Path

import wandb
import yaml

from open_value_estimator.config import Config, load_config, parse_cli_overrides

logger = logging.getLogger(__name__)


def apply_sweep_params(base_dict: dict, sweep_params: dict) -> dict:
    """Merge W&B sweep parameters into a base config dict.

    Sweep parameters use dot notation (e.g., "training.learning_rate")
    that maps to the nested Config dataclass structure.

    Args:
        base_dict: Base configuration as a plain dict.
        sweep_params: Flat dict from wandb.config with dot-notation keys.

    Returns:
        New config dict with sweep parameters applied.
    """
    merged = copy.deepcopy(base_dict)
    for key, value in sweep_params.items():
        parts = key.split(".")
        target = merged
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
    return merged


def load_sweep_config(sweep_config_path: str | Path) -> dict:
    """Load a W&B sweep configuration from a YAML file.

    Args:
        sweep_config_path: Path to the sweep YAML file.

    Returns:
        Sweep config dict suitable for wandb.sweep().
    """
    with open(sweep_config_path) as f:
        sweep_config = yaml.safe_load(f)

    # Remove program key — we use function-based agents, not subprocess-based
    sweep_config.pop("program", None)

    return sweep_config


def make_sweep_train_fn(base_config_dict: dict):
    """Create the callback function that wandb.agent() calls for each trial.

    Each invocation:
    1. Calls wandb.init() (required by wandb.agent protocol)
    2. Reads sweep-selected params from wandb.config
    3. Merges them into the base config
    4. Generates a unique output_dir from the wandb run name
    5. Calls train()
    6. Calls wandb.finish()

    Args:
        base_config_dict: Base configuration as a plain dict.

    Returns:
        A zero-argument callable for wandb.agent(function=...).
    """
    def sweep_train_fn():
        from open_value_estimator.training import train

        run = wandb.init()

        sweep_params = dict(wandb.config)
        config_dict = apply_sweep_params(base_config_dict, sweep_params)

        # Unique output_dir and run_name per sweep trial
        sweep_run_name = run.name or run.id
        base_output = base_config_dict.get("output_dir", "outputs")
        config_dict["output_dir"] = f"{base_output}/sweep_{run.sweep_id}/{sweep_run_name}"
        config_dict["run_name"] = f"sweep_{sweep_run_name}"

        cfg = Config.from_dict(config_dict)

        logger.info(f"Sweep run {sweep_run_name}: {sweep_params}")
        try:
            train(cfg)
        except Exception as e:
            logger.error(f"Sweep run failed: {e}")
            raise
        finally:
            wandb.finish()

    return sweep_train_fn


def _load_base_config_for_sweep(
    base_config_path: str | Path,
    project_override: str | None = None,
    overrides: dict | None = None,
) -> dict:
    """Load the base config dict for sweep usage.

    Sweeps generate their own run_name and output_dir per trial, so the
    shared loader is used without enforcing run_name or mutating output_dir.

    Args:
        base_config_path: Path to the base training YAML config.
        project_override: Optional W&B project name override.
        overrides: Optional base-config overrides from the CLI.

    Returns:
        Base config as a plain dict.
    """
    config_dict = load_config(
        config_path=base_config_path,
        overrides=overrides,
        require_run_name=False,
        append_run_name_to_output_dir=False,
    )

    # Sweeps generate their own run_name per trial
    if not config_dict.get("run_name"):
        config_dict["run_name"] = "sweep_placeholder"

    if project_override:
        if "wandb" not in config_dict:
            config_dict["wandb"] = {}
        config_dict["wandb"]["project"] = project_override

    return config_dict


def main():
    """CLI entry point for local W&B sweeps."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Run W&B hyperparameter sweeps for the open value estimator.",
        epilog=(
            "Examples:\n"
            "  python -m open_value_estimator.sweep "
            "config=configs/config.yaml "
            "sweep_config=configs/sweeps/lr_batch_sweep.yaml "
            "count=20\n"
            "  python -m open_value_estimator.sweep "
            "config=configs/config.yaml "
            "sweep_id=<SWEEP_ID> count=10"
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
        type=str,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--sweep-config",
        dest="legacy_sweep_config",
        type=str,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--sweep-id",
        dest="legacy_sweep_id",
        type=str,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--count",
        dest="legacy_count",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--project",
        dest="legacy_project",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--create-only",
        dest="legacy_create_only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    override_dict = parse_cli_overrides(args.overrides)
    base_config_path = override_dict.pop("config", None) or args.legacy_config
    if not base_config_path or not isinstance(base_config_path, str):
        parser.error("config=... is required")

    sweep_config_path = override_dict.pop("sweep_config", None) or args.legacy_sweep_config
    sweep_id = override_dict.pop("sweep_id", None) or args.legacy_sweep_id

    count = override_dict.pop("count", None)
    if count is None:
        count = args.legacy_count if args.legacy_count is not None else 10
    if not isinstance(count, int):
        parser.error("count must be an integer")

    project = override_dict.pop("project", None) or args.legacy_project
    create_only = override_dict.pop("create_only", None)
    if create_only is None:
        create_only = args.legacy_create_only
    if not isinstance(create_only, bool):
        parser.error("create_only must be true or false")

    base_config_dict = _load_base_config_for_sweep(
        base_config_path=base_config_path,
        project_override=project,
        overrides=override_dict,
    )

    project = project or base_config_dict.get("wandb", {}).get(
        "project", "open-value-estimator"
    )

    if sweep_id:
        logger.info(f"Joining existing sweep: {sweep_id}")
    elif sweep_config_path:
        sweep_config = load_sweep_config(sweep_config_path)
        sweep_id = wandb.sweep(sweep_config, project=project)
        logger.info(f"Created sweep: {sweep_id}")
    else:
        parser.error("Either sweep_config=... (to create) or sweep_id=... (to join) is required")

    if create_only:
        print(f"Sweep ID: {sweep_id}")
        print(
            "Join with: "
            "python -m open_value_estimator.sweep "
            f"config={base_config_path} sweep_id={sweep_id} count={count}"
        )
        return

    train_fn = make_sweep_train_fn(base_config_dict)
    wandb.agent(sweep_id, function=train_fn, count=count, project=project)


if __name__ == "__main__":
    main()
