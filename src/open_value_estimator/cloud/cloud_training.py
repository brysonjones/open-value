"""Cloud training function for open value estimator.

This module contains the training function that runs on cloud workers.
It is kept separate from cloud_launcher.py so that the function has a proper
__module__ attribute that can be imported remotely.

When accelerate is enabled, this module implements a self-launch pattern:
1. If not yet in distributed environment, re-launch via 'accelerate launch'
2. If already in distributed environment, create Accelerator and run training
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from open_value_estimator.config import Config
from open_value_estimator.training import train

logger = logging.getLogger(__name__)


def run_training_gcp(config: dict):
    """Training function that runs on GCP Vertex AI.

    This function is called remotely on the Vertex AI instance.
    It receives the config dict and runs the training loop.

    If accelerate is enabled, it handles the self-launch pattern:
    - Coordinator (first call): re-launches via 'accelerate launch' subprocess
    - Worker (after accelerate launch): creates Accelerator and runs training

    Args:
        config: Training configuration dictionary.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    # Extract accelerate config (don't modify original dict yet)
    accelerate_config = config.get("accelerate", {})
    use_accelerate = accelerate_config.get("enabled", False)

    # Check if already running inside accelerate-launched environment
    # Note: with num_processes=1, accelerate may not set LOCAL_RANK/RANK,
    # so we also check our own sentinel env var to prevent infinite re-launch.
    is_accelerate_child = os.environ.get("OVE_ACCELERATE_CHILD") == "1"
    is_distributed = os.environ.get("LOCAL_RANK") is not None or os.environ.get("RANK") is not None

    # === CASE 1: Coordinator - needs to launch accelerate ===
    if use_accelerate and not is_distributed and not is_accelerate_child:
        num_processes = str(accelerate_config.get("num_processes") or 1)
        mixed_precision = accelerate_config.get("mixed_precision", "no")

        logger.info("Launching training with accelerate (self-launch):")
        logger.info(f"  num_processes: {num_processes}")
        logger.info(f"  mixed_precision: {mixed_precision}")

        # Save config to temp file so subprocess can read it
        config_file = Path(tempfile.gettempdir()) / "ove_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f)
        logger.info(f"  config_file: {config_file}")

        # Set env vars so subprocess knows where to find config and doesn't re-launch
        os.environ["OVE_CONFIG_PATH"] = str(config_file)
        os.environ["OVE_ACCELERATE_CHILD"] = "1"

        # Construct accelerate launch command
        # We re-launch this module as a script
        cmd = [
            "accelerate", "launch",
            "--num_processes", num_processes,
            "--mixed_precision", mixed_precision,
            "--main_process_port", "29500",
            "-m", "open_value_estimator.cloud.cloud_training",
        ]

        logger.info(f"  Command: {' '.join(cmd)}")

        # Run accelerate launch as subprocess
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed with exit code {e.returncode}")
            sys.exit(e.returncode)

        return

    # === CASE 2: Worker process or single-process training ===
    accelerator = None

    if is_distributed:
        # We're inside accelerate-launched environment
        rank = os.environ.get("RANK", "0")
        local_rank = os.environ.get("LOCAL_RANK", "0")
        logger.info(f"Worker process started (Rank {rank}, Local Rank {local_rank})")

        # Create Accelerator that reads settings from environment
        from accelerate import Accelerator
        from accelerate.utils import DistributedDataParallelKwargs, GradientAccumulationPlugin

        grad_accum_steps = config.get("accelerate", {}).get("gradient_accumulation_steps", 1)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        gradient_accumulation_plugin = GradientAccumulationPlugin(
            num_steps=grad_accum_steps,
            adjust_scheduler=False,
            sync_with_dataloader=False,
        )
        accelerator = Accelerator(
            gradient_accumulation_plugin=gradient_accumulation_plugin,
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
        )
        logger.info(
            f"Created Accelerator: mixed_precision={accelerator.mixed_precision}, "
            f"num_processes={accelerator.num_processes}, device={accelerator.device}"
        )
    else:
        logger.info("Running single-process training (accelerate disabled)")

    # Convert dict to Config object
    cfg = Config.from_dict(config)

    logger.info(f"Starting cloud training run: {cfg.data.repo_id}")
    logger.info(f"Output directory: {cfg.output_dir}")
    logger.info(f"Run name: {cfg.run_name}")

    # Run training
    train(cfg, accelerator=accelerator)

    logger.info("Training completed!")


def run_sweep_gcp(payload: dict):
    """Run a W&B sweep agent on a GCP machine (supports multi-GPU via accelerate).

    This function is called remotely on a Vertex AI instance. It runs
    wandb.agent() which calls a sweep function for each trial. Each trial
    merges sweep params into the base config, then delegates to
    run_training_gcp() which handles the accelerate self-launch pattern.

    Args:
        payload: Dict with keys:
            - base_config: Base training configuration dict.
            - sweep_id: W&B sweep ID to join.
            - sweep_count: Number of trials to run.
    """
    import wandb

    from open_value_estimator.sweep import apply_sweep_params

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    base_config = payload["base_config"]
    sweep_id = payload["sweep_id"]
    sweep_count = payload["sweep_count"]
    project = base_config.get("wandb", {}).get("project", "open-value-estimator")

    logger.info(f"Starting sweep agent: sweep_id={sweep_id}, count={sweep_count}")

    def sweep_fn():
        run = wandb.init()
        sweep_params = dict(wandb.config)
        merged = apply_sweep_params(base_config, sweep_params)

        # Unique output_dir per sweep trial
        sweep_run_name = run.name or run.id
        base_output = base_config.get("output_dir", "/tmp/outputs")
        merged["output_dir"] = f"{base_output}/sweep_{run.sweep_id}/{sweep_run_name}"
        merged["run_name"] = f"sweep_{sweep_run_name}"

        logger.info(f"Sweep trial {sweep_run_name}: {sweep_params}")

        # Hand off wandb to the accelerate subprocess via env vars.
        # The subprocess's train() will resume this same wandb run.
        os.environ["WANDB_RUN_ID"] = run.id
        os.environ["WANDB_RESUME"] = "must"
        wandb.finish(quiet=True)

        # run_training_gcp handles accelerate self-launch for multi-GPU
        run_training_gcp(merged)

        # Clean up env vars for the next trial
        os.environ.pop("WANDB_RUN_ID", None)
        os.environ.pop("WANDB_RESUME", None)

    wandb.agent(sweep_id, function=sweep_fn, count=sweep_count, project=project)
    logger.info("Sweep agent completed all trials")


def run_eval_gcp(config: dict):
    """Evaluation function that runs on GCP Vertex AI.

    This function is called remotely on the Vertex AI instance.
    It loads a checkpoint, runs evaluation on specified episodes,
    and saves videos to the output directory.

    Args:
        config: Eval configuration dictionary with keys:
            - checkpoint: Path to safetensors checkpoint file.
            - dataset: Dataset repo ID.
            - data_root: Local root path for the dataset.
            - episodes: List of episode indices to evaluate.
            - output_dir: Directory to save output videos.
            - batch_size: Batch size for inference.
            - use_ema: Whether to use EMA weights.
            - camera_views: Optional list of camera views for video rendering.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    import torch

    from open_value_estimator.config import DataConfig
    from open_value_estimator.dataset import ValueDataset
    from open_value_estimator.eval import (
        create_evaluation_video,
        get_available_camera_view_keys,
        resolve_camera_view_keys,
    )
    from open_value_estimator.value_estimator import OpenValueEstimator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running eval on device: {device}")

    # Load model
    checkpoint = config["checkpoint"]
    use_ema = config.get("use_ema", True)
    logger.info(f"Loading model from {checkpoint}")
    model = OpenValueEstimator.from_pretrained(path=checkpoint, device=device, use_ema=use_ema)

    # Load dataset
    dataset_id = config["dataset"]
    data_root = config.get("data_root", dataset_id)
    data_cfg = DataConfig(repo_id=dataset_id, root=data_root)
    logger.info(f"Loading dataset: {dataset_id}")
    dataset = ValueDataset(data_cfg)
    available_camera_views = get_available_camera_view_keys(dataset)
    if not available_camera_views:
        raise ValueError(
            "No camera view columns found in dataset "
            "(expected keys like 'observation.images.<view>')."
        )

    requested_camera_views = config.get("camera_views")
    selected_camera_views = resolve_camera_view_keys(
        requested_camera_views,
        available_camera_views,
    )
    if selected_camera_views:
        logger.info(f"Selected camera views for recording: {selected_camera_views}")
    else:
        logger.info(f"Using all camera views for recording: {available_camera_views}")

    # Run eval for each episode
    episodes = config["episodes"]
    output_dir = Path(config.get("output_dir", "./eval_outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = config.get("batch_size", 16)
    show_ground_truth_reward = config.get("show_ground_truth_reward", True)

    logger.info(f"Evaluating {len(episodes)} episodes: {episodes}")
    for ep_idx in episodes:
        output_path = output_dir / f"eval_ep{ep_idx}.mp4"
        logger.info(f"Evaluating episode {ep_idx} -> {output_path}")
        create_evaluation_video(
            model=model,
            dataset=dataset,
            output_path=output_path,
            episode_idx=ep_idx,
            device=device,
            batch_size=batch_size,
            camera_views=selected_camera_views,
            show_ground_truth_reward=show_ground_truth_reward,
        )

    logger.info(f"Eval completed. Saved {len(episodes)} videos to {output_dir}")


def main():
    """Entry point when running as a module (for accelerate launch)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    # Load config from temp file (set by coordinator process)
    config_path = os.environ.get("OVE_CONFIG_PATH")
    if not config_path:
        raise ValueError("OVE_CONFIG_PATH environment variable not set")

    logger.info(f"Loading config from: {config_path}")
    with open(config_path) as f:
        config = json.load(f)

    # Run training (we're already in distributed environment)
    run_training_gcp(config)


if __name__ == "__main__":
    main()
