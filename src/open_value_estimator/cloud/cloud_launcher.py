"""Cloud training launcher for open value estimator.

This module provides cloud job launch integration using coalesce.

Usage:
    # Launch training in the cloud
    python -m open_value_estimator.cloud.cloud_launcher \
        config=configs/config.yaml

    # With custom config overrides
    python -m open_value_estimator.cloud.cloud_launcher \
        config=configs/config.yaml \
        run_name=my_experiment

    # With a different dataset
    python -m open_value_estimator.cloud.cloud_launcher \
        config=configs/config.yaml \
        dataset=my_dataset
"""

import argparse
import os
from pathlib import Path

from omegaconf import OmegaConf

from open_value_estimator.cloud import REMOTE_PACKAGES
from open_value_estimator.cloud.cloud_training import run_training_gcp
from open_value_estimator.config import load_config, load_gcp_settings, parse_cli_overrides


def _merge_override_dicts(base: dict, extra: dict) -> dict:
    """Merge two nested override dicts."""
    if not extra:
        return base
    if not base:
        return extra

    merged = OmegaConf.merge(OmegaConf.create(base), OmegaConf.create(extra))
    return OmegaConf.to_container(merged, resolve=True)


def _normalize_list_arg(value: object, name: str) -> list | None:
    """Normalize a scalar-or-list CLI/config value to a list."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, (str, int)):
        return [value]
    raise ValueError(f"{name} must be a list or scalar")


def _resolve_gcp_mount_path(path: str | None, gcs_mount_prefix: str) -> str | None:
    """Resolve a config path against the GCS FUSE mount when relative."""
    if path is None:
        return None

    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return str(expanded)

    return str(Path(gcs_mount_prefix) / expanded)


def build_eval_config(
    config_path: str,
    gcp_settings: dict,
    overrides: dict | None = None,
) -> tuple[dict, str]:
    """Build the eval job config dict and selected GPU from config overrides."""
    overrides = overrides or {}
    valid_gpus = {"t4", "t4x2", "t4x4", "a100", "a100x2", "a100x4", "a100x8", "h100", "h100x2", "h100x4", "h100x8"}

    checkpoint = overrides.pop("checkpoint", None)
    dataset = overrides.pop("dataset", None)
    output_dir = overrides.pop("output_dir", None)
    gpu = overrides.pop("gpu", None)
    batch_size = overrides.pop("batch_size", None)
    use_ema_override = overrides.pop("use_ema", None)
    show_ground_truth_reward_override = overrides.pop("show_ground_truth_reward", None)
    no_ema = overrides.pop("no_ema", None)

    episodes = _normalize_list_arg(overrides.pop("episodes", None), "episodes")
    camera_views = _normalize_list_arg(overrides.pop("camera_views", None), "camera_views")

    eval_config_defaults = load_config(
        config_path=config_path,
        overrides=overrides,
        require_run_name=False,
        append_run_name_to_output_dir=False,
    )
    eval_defaults = eval_config_defaults.get("eval", {})
    if eval_defaults and not isinstance(eval_defaults, dict):
        raise ValueError("eval section must be a mapping")

    checkpoint = checkpoint or eval_defaults.get("checkpoint")
    episodes = episodes or _normalize_list_arg(eval_defaults.get("episodes"), "eval.episodes")
    dataset = dataset or eval_config_defaults.get("data", {}).get("repo_id")
    output_dir = output_dir or eval_defaults.get("output_dir")
    camera_views = camera_views or _normalize_list_arg(eval_defaults.get("camera_views"), "eval.camera_views")
    gpu = gpu or eval_defaults.get("gpu") or "t4"
    batch_size = batch_size or eval_defaults.get("batch_size", 16)

    use_ema = eval_defaults.get("use_ema", True)
    if use_ema_override is not None:
        if not isinstance(use_ema_override, bool):
            raise ValueError("use_ema must be true or false")
        use_ema = use_ema_override

    show_ground_truth_reward = eval_defaults.get("show_ground_truth_reward", True)
    if show_ground_truth_reward_override is not None:
        if not isinstance(show_ground_truth_reward_override, bool):
            raise ValueError("show_ground_truth_reward must be true or false")
        show_ground_truth_reward = show_ground_truth_reward_override

    if no_ema is not None:
        if not isinstance(no_ema, bool):
            raise ValueError("no_ema must be true or false")
        if no_ema:
            use_ema = False

    if gpu not in valid_gpus:
        raise ValueError(f"gpu must be one of: {sorted(valid_gpus)}")
    if batch_size is not None and not isinstance(batch_size, int):
        raise ValueError("batch_size must be an integer")
    if not checkpoint:
        raise ValueError("checkpoint=... is required for eval mode")
    if not episodes:
        raise ValueError("episodes=[...] is required for eval mode")
    if not dataset:
        raise ValueError("data.repo_id or dataset=... is required for eval mode")

    gcs_mount_prefix = gcp_settings.get("mount_prefix", "/gcs/YOUR_GCS_BUCKET")
    checkpoint = _resolve_gcp_mount_path(checkpoint, gcs_mount_prefix)
    resolved_output_dir = _resolve_gcp_mount_path(output_dir, gcs_mount_prefix)
    eval_config = {
        "checkpoint": checkpoint,
        "dataset": dataset,
        "data_root": f"{gcs_mount_prefix}/datasets/{dataset}",
        "episodes": episodes,
        "output_dir": resolved_output_dir or f"{gcs_mount_prefix}/outputs/open_value_estimator/eval",
        "batch_size": batch_size,
        "use_ema": use_ema,
        "show_ground_truth_reward": show_ground_truth_reward,
        "camera_views": camera_views,
    }
    return eval_config, gpu


def load_config_gcp(
    config_path: str,
    gcp_settings: dict,
    overrides: dict | None = None,
) -> dict:
    """Load config with GCP-specific path overrides.

    Calls the shared load_config() then overrides output_dir and data.root
    to use GCS FUSE mount paths on Vertex AI.

    Args:
        config_path: Path to the training YAML config file.
        gcp_settings: GCP deployment settings loaded from the cloud config.
        overrides: Optional dict of config overrides.

    Returns:
        Config as a plain dict with GCS paths.
    """
    config_dict = load_config(config_path=config_path, overrides=overrides)
    gcs_mount_prefix = gcp_settings.get("mount_prefix", "/gcs/YOUR_GCS_BUCKET")

    run_name = config_dict.get("run_name", "training")

    # Override paths for GCS-mounted filesystem on Vertex AI
    config_dict["output_dir"] = f"{gcs_mount_prefix}/outputs/open_value_estimator/{run_name}"

    if "data" in config_dict:
        repo_id = config_dict["data"].get("repo_id", "")
        config_dict["data"]["root"] = f"{gcs_mount_prefix}/datasets/{repo_id}"

    print(f"GCP output dir: {config_dict['output_dir']}")
    print(f"GCP data root: {config_dict['data']['root']}")

    return config_dict


def launch_training(
    config_dict: dict,
    gcp_settings: dict,
    gpu: str = "a100",
    sync: bool = True,
):
    """Launch training job on GCP Vertex AI.

    Args:
        config_dict: Training configuration dictionary.
        gpu: GPU type to use ("t4", "a100", "h100").
        sync: If True, wait for job completion.
    """
    from coalesce import launch_job

    gpu_configs = _get_gpu_configs()
    if gpu not in gpu_configs:
        raise ValueError(f"Unknown GPU type: {gpu}. Choose from: {list(gpu_configs.keys())}")

    gpu_config = gpu_configs[gpu]
    run_name = config_dict.get("run_name", "training")
    num_gpus = gpu_config["accelerator_count"]

    # Auto-sync accelerate num_processes with GPU count
    if "accelerate" not in config_dict:
        config_dict["accelerate"] = {}
    if num_gpus > 1:
        config_dict["accelerate"]["enabled"] = True
        config_dict["accelerate"]["num_processes"] = num_gpus
        print(f"  Multi-GPU detected: enabling accelerate with {num_gpus} processes")
    elif config_dict["accelerate"].get("enabled") and not config_dict["accelerate"].get("num_processes"):
        config_dict["accelerate"]["num_processes"] = 1

    print("\nLaunching GCP training job:")
    print(f"  Run name: {run_name}")
    print(f"  GPU: {gpu} ({num_gpus} GPU{'s' if num_gpus > 1 else ''})")
    print(f"  Dataset: {config_dict['data']['repo_id']}")

    # Set environment variables for the remote job
    env_vars = {}
    if "HF_TOKEN" in os.environ:
        env_vars["HF_TOKEN"] = os.environ["HF_TOKEN"]
    if "WANDB_API_KEY" in os.environ:
        env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    # A3 machine types (H100) require FLEX_START scheduling
    scheduling = "FLEX_START" if gpu_config["machine_type"].startswith("a3-") else "STANDARD"

    job = launch_job(
        func=run_training_gcp,
        project_id=gcp_settings.get("project_id", "YOUR_GCP_PROJECT_ID"),
        bucket=gcp_settings.get("bucket", "gs://YOUR_GCS_BUCKET"),
        region=gcp_settings.get("region", "us-central1"),
        container_uri=gcp_settings.get("container", "YOUR_CONTAINER_URI"),
        machine_type=gpu_config["machine_type"],
        accelerator_type=gpu_config["accelerator_type"],
        accelerator_count=gpu_config["accelerator_count"],
        sync_packages=["open_value_estimator"],
        config=config_dict,
        extra_packages=REMOTE_PACKAGES,
        env=env_vars,
        job_name=f"ove_{run_name}",
        sync=sync,
        scheduling_strategy=scheduling,
    )

    return job


def launch_sweep(
    config_dict: dict,
    gcp_settings: dict,
    sweep_id: str,
    sweep_count: int,
    agents: int,
    gpu: str = "a100",
    sync: bool = True,
):
    """Launch W&B sweep agents on GCP Vertex AI.

    Each agent is a separate GCP job that runs sweep_count trials sequentially.
    Multiple agents run in parallel, coordinated by the W&B sweep controller.

    Args:
        config_dict: Base training configuration dictionary.
        sweep_id: W&B sweep ID to join.
        sweep_count: Number of trials per agent.
        agents: Number of parallel GCP machines to spin up.
        gpu: GPU type to use.
        sync: If True, wait for all jobs to complete.
    """
    from coalesce import launch_job

    from open_value_estimator.cloud.cloud_training import run_sweep_gcp

    # GPU configurations (reuse from launch_training)
    gpu_configs = _get_gpu_configs()
    if gpu not in gpu_configs:
        raise ValueError(f"Unknown GPU type: {gpu}. Choose from: {list(gpu_configs.keys())}")

    gpu_config = gpu_configs[gpu]
    num_gpus = gpu_config["accelerator_count"]
    run_name = config_dict.get("run_name", "sweep")

    # Auto-sync accelerate with GPU count
    if "accelerate" not in config_dict:
        config_dict["accelerate"] = {}
    if num_gpus > 1:
        config_dict["accelerate"]["enabled"] = True
        config_dict["accelerate"]["num_processes"] = num_gpus

    # Environment variables for remote jobs
    env_vars = {}
    if "HF_TOKEN" in os.environ:
        env_vars["HF_TOKEN"] = os.environ["HF_TOKEN"]
    if "WANDB_API_KEY" in os.environ:
        env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    scheduling = "FLEX_START" if gpu_config["machine_type"].startswith("a3-") else "STANDARD"

    print(f"\nLaunching {agents} sweep agent(s) on GCP:")
    print(f"  Sweep ID: {sweep_id}")
    print(f"  Trials per agent: {sweep_count}")
    print(f"  Total trials: {agents * sweep_count}")
    print(f"  GPU: {gpu} ({num_gpus} GPU{'s' if num_gpus > 1 else ''} per agent)")

    jobs = []
    for i in range(agents):
        payload = {
            "base_config": config_dict,
            "sweep_id": sweep_id,
            "sweep_count": sweep_count,
        }
        job = launch_job(
            func=run_sweep_gcp,
            project_id=gcp_settings.get("project_id", "YOUR_GCP_PROJECT_ID"),
            bucket=gcp_settings.get("bucket", "gs://YOUR_GCS_BUCKET"),
            region=gcp_settings.get("region", "us-central1"),
            container_uri=gcp_settings.get("container", "YOUR_CONTAINER_URI"),
            machine_type=gpu_config["machine_type"],
            accelerator_type=gpu_config["accelerator_type"],
            accelerator_count=gpu_config["accelerator_count"],
            sync_packages=["open_value_estimator"],
            config=payload,
            extra_packages=REMOTE_PACKAGES,
            env=env_vars,
            job_name=f"ove_sweep_{run_name}_agent{i}",
            sync=False,  # Always launch async, wait below if needed
            scheduling_strategy=scheduling,
        )
        jobs.append(job)
        print(f"  Launched agent {i}")

    if sync:
        print(f"\nWaiting for {len(jobs)} agent(s) to complete...")
        for job in jobs:
            job.wait()
        print("All sweep agents completed.")

    return jobs


def launch_eval(
    config_dict: dict,
    gcp_settings: dict,
    gpu: str = "t4",
    sync: bool = True,
):
    """Launch evaluation job on GCP Vertex AI.

    Args:
        config_dict: Eval configuration dictionary with checkpoint, dataset,
            episodes, output_dir, etc.
        gpu: GPU type to use (eval is single-GPU).
        sync: If True, wait for job completion.
    """
    from coalesce import launch_job

    from open_value_estimator.cloud.cloud_training import run_eval_gcp

    gpu_configs = _get_gpu_configs()
    if gpu not in gpu_configs:
        raise ValueError(f"Unknown GPU type: {gpu}. Choose from: {list(gpu_configs.keys())}")

    gpu_config = gpu_configs[gpu]

    print("\nLaunching GCP eval job:")
    print(f"  Checkpoint: {config_dict['checkpoint']}")
    print(f"  Dataset: {config_dict['dataset']}")
    print(f"  Episodes: {config_dict['episodes']}")
    if config_dict.get("camera_views"):
        print(f"  Camera views (recording): {config_dict['camera_views']}")
    print(f"  GPU: {gpu}")

    env_vars = {}
    if "HF_TOKEN" in os.environ:
        env_vars["HF_TOKEN"] = os.environ["HF_TOKEN"]

    scheduling = "FLEX_START" if gpu_config["machine_type"].startswith("a3-") else "STANDARD"

    job = launch_job(
        func=run_eval_gcp,
        project_id=gcp_settings.get("project_id", "YOUR_GCP_PROJECT_ID"),
        bucket=gcp_settings.get("bucket", "gs://YOUR_GCS_BUCKET"),
        region=gcp_settings.get("region", "us-central1"),
        container_uri=gcp_settings.get("container", "YOUR_CONTAINER_URI"),
        machine_type=gpu_config["machine_type"],
        accelerator_type=gpu_config["accelerator_type"],
        accelerator_count=1,  # Eval is single-GPU
        sync_packages=["open_value_estimator"],
        config=config_dict,
        extra_packages=REMOTE_PACKAGES,
        env=env_vars,
        job_name="ove_eval",
        sync=sync,
        scheduling_strategy=scheduling,
    )

    return job


def _get_gpu_configs() -> dict:
    """Return the GPU configuration mapping."""
    return {
        "t4": {"machine_type": "n1-standard-4", "accelerator_type": "NVIDIA_TESLA_T4", "accelerator_count": 1},
        "t4x2": {"machine_type": "n1-standard-8", "accelerator_type": "NVIDIA_TESLA_T4", "accelerator_count": 2},
        "t4x4": {"machine_type": "n1-standard-16", "accelerator_type": "NVIDIA_TESLA_T4", "accelerator_count": 4},
        "a100": {"machine_type": "a2-highgpu-1g", "accelerator_type": "NVIDIA_TESLA_A100", "accelerator_count": 1},
        "a100x2": {"machine_type": "a2-highgpu-2g", "accelerator_type": "NVIDIA_TESLA_A100", "accelerator_count": 2},
        "a100x4": {"machine_type": "a2-highgpu-4g", "accelerator_type": "NVIDIA_TESLA_A100", "accelerator_count": 4},
        "a100x8": {"machine_type": "a2-highgpu-8g", "accelerator_type": "NVIDIA_TESLA_A100", "accelerator_count": 8},
        "h100": {"machine_type": "a3-highgpu-1g", "accelerator_type": "NVIDIA_H100_80GB", "accelerator_count": 1},
        "h100x2": {"machine_type": "a3-highgpu-2g", "accelerator_type": "NVIDIA_H100_80GB", "accelerator_count": 2},
        "h100x4": {"machine_type": "a3-highgpu-4g", "accelerator_type": "NVIDIA_H100_80GB", "accelerator_count": 4},
        "h100x8": {"machine_type": "a3-highgpu-8g", "accelerator_type": "NVIDIA_H100_80GB", "accelerator_count": 8},
    }


def main():
    """Main entry point for GCP training and sweeps."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Run open value estimator training on GCP Vertex AI with key=value arguments.",
        epilog=(
            "Examples:\n"
            "  python -m open_value_estimator.cloud.cloud_launcher "
            "config=configs/config.yaml run_name=my_run gpu=a100x4\n"
            "  python -m open_value_estimator.cloud.cloud_launcher "
            "config=configs/config.yaml eval=true checkpoint=... episodes=[0,5,10]\n"
            "  python -m open_value_estimator.cloud.cloud_launcher "
            "config=configs/config.yaml sweep_config=configs/sweeps/lr_batch_sweep.yaml "
            "agents=4 sweep_count=10 gpu=a100"
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
    parser.add_argument(
        "--gpu",
        dest="legacy_gpu",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--detached",
        dest="legacy_detached",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dataset",
        dest="legacy_dataset",
        type=str,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--run-name",
        dest="legacy_run_name",
        type=str,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--checkpoint",
        dest="legacy_checkpoint",
        type=str,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--eval",
        dest="legacy_eval",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--episodes",
        dest="legacy_episodes",
        type=int,
        nargs="+",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output-dir",
        dest="legacy_output_dir",
        type=str,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-ema",
        dest="legacy_no_ema",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--camera-views",
        dest="legacy_camera_views",
        type=str,
        nargs="+",
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
        "--sweep-count",
        dest="legacy_sweep_count",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--agents",
        dest="legacy_agents",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    override_dict = parse_cli_overrides(args.overrides)

    config_path = override_dict.pop("config", None) or args.legacy_config
    if not config_path or not isinstance(config_path, str):
        parser.error("config=... is required")

    gpu = override_dict.pop("gpu", None) or args.legacy_gpu
    valid_gpus = {"t4", "t4x2", "t4x4", "a100", "a100x2", "a100x4", "a100x8", "h100", "h100x2", "h100x4", "h100x8"}
    if gpu is not None and gpu not in valid_gpus:
        parser.error(f"gpu must be one of: {sorted(valid_gpus)}")

    detached = override_dict.pop("detached", None)
    if detached is None:
        detached = args.legacy_detached
    if not isinstance(detached, bool):
        parser.error("detached must be true or false")

    dataset = override_dict.pop("dataset", None) or args.legacy_dataset
    run_name = override_dict.pop("run_name", None) or args.legacy_run_name
    checkpoint = override_dict.pop("checkpoint", None) or args.legacy_checkpoint

    eval_mode = override_dict.pop("eval", None)
    if eval_mode is None:
        eval_mode = args.legacy_eval
    if not isinstance(eval_mode, bool):
        parser.error("eval must be true or false")

    episodes = override_dict.pop("episodes", None)
    if episodes is None:
        episodes = args.legacy_episodes
    try:
        episodes = _normalize_list_arg(episodes, "episodes")
    except ValueError as exc:
        parser.error(str(exc))

    output_dir = override_dict.pop("output_dir", None) or args.legacy_output_dir

    no_ema = override_dict.pop("no_ema", None)
    if no_ema is None:
        no_ema = args.legacy_no_ema
    if not isinstance(no_ema, bool):
        parser.error("no_ema must be true or false")

    camera_views = override_dict.pop("camera_views", None)
    if camera_views is None:
        camera_views = args.legacy_camera_views
    try:
        camera_views = _normalize_list_arg(camera_views, "camera_views")
    except ValueError as exc:
        parser.error(str(exc))

    sweep_config_path = override_dict.pop("sweep_config", None) or args.legacy_sweep_config
    sweep_id = override_dict.pop("sweep_id", None) or args.legacy_sweep_id

    sweep_count = override_dict.pop("sweep_count", None)
    if sweep_count is None:
        sweep_count = args.legacy_sweep_count if args.legacy_sweep_count is not None else 10
    if not isinstance(sweep_count, int):
        parser.error("sweep_count must be an integer")

    agents = override_dict.pop("agents", None)
    if agents is None:
        agents = args.legacy_agents if args.legacy_agents is not None else 1
    if not isinstance(agents, int):
        parser.error("agents must be an integer")

    try:
        gcp_settings = load_gcp_settings(config_path, overrides=override_dict)
    except ValueError as exc:
        parser.error(str(exc))

    # Check for required environment variables
    if "HF_TOKEN" not in os.environ:
        raise ValueError(
            "HF_TOKEN is not set. This is needed to use LeRobotDatasets. "
            "Please set it in your environment variables and try again."
        )

    # Build config overrides
    config_overrides = override_dict
    shorthand_overrides = {}
    if dataset:
        shorthand_overrides["data"] = {"repo_id": dataset}
    if run_name:
        shorthand_overrides["run_name"] = run_name
    if checkpoint:
        shorthand_overrides["training"] = {"resume_from": checkpoint}
    config_overrides = _merge_override_dicts(config_overrides, shorthand_overrides)

    # Eval mode
    if eval_mode:
        eval_overrides = config_overrides.copy()
        if gpu is not None:
            eval_overrides["gpu"] = gpu
        if checkpoint is not None:
            eval_overrides["checkpoint"] = checkpoint
        if episodes is not None:
            eval_overrides["episodes"] = episodes
        if output_dir is not None:
            eval_overrides["output_dir"] = output_dir
        if camera_views is not None:
            eval_overrides["camera_views"] = camera_views
        if no_ema:
            eval_overrides["no_ema"] = True

        try:
            eval_config, gpu = build_eval_config(
                config_path=config_path,
                gcp_settings=gcp_settings,
                overrides=eval_overrides,
            )
        except ValueError as exc:
            parser.error(str(exc))

        launch_eval(
            config_dict=eval_config,
            gcp_settings=gcp_settings,
            gpu=gpu,
            sync=not detached,
        )
        return

    # Sweeps generate their own run_name per trial; provide placeholder to pass validation
    is_sweep = sweep_config_path or sweep_id
    if is_sweep and not config_overrides.get("run_name"):
        config_overrides["run_name"] = "sweep"

    if gpu is None:
        gpu = "t4"

    # Load config with GCP path overrides
    config_dict = load_config_gcp(
        config_path=config_path,
        gcp_settings=gcp_settings,
        overrides=config_overrides,
    )

    # Sweep mode
    if sweep_config_path or sweep_id:
        import wandb

        from open_value_estimator.sweep import load_sweep_config

        if "WANDB_API_KEY" not in os.environ:
            raise ValueError("WANDB_API_KEY is required for sweeps")

        project = config_dict.get("wandb", {}).get("project", "open-value-estimator")

        if sweep_id:
            print(f"Joining existing sweep: {sweep_id}")
        else:
            sweep_config = load_sweep_config(sweep_config_path)
            sweep_id = wandb.sweep(sweep_config, project=project)
            print(f"Created sweep: {sweep_id}")

        launch_sweep(
            config_dict=config_dict,
            gcp_settings=gcp_settings,
            sweep_id=sweep_id,
            sweep_count=sweep_count,
            agents=agents,
            gpu=gpu,
            sync=not detached and agents == 1,
        )
        return

    # Normal training mode
    launch_training(
        config_dict=config_dict,
        gcp_settings=gcp_settings,
        gpu=gpu,
        sync=not detached,
    )


if __name__ == "__main__":
    main()
