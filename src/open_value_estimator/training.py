"""Training utilities for the value estimator."""

import argparse
import copy
import json
import logging
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import wandb
from omegaconf import OmegaConf
from safetensors.torch import save_file as save_safetensors

from open_value_estimator.config import Config, load_config, parse_cli_overrides
from open_value_estimator.dataset import ValueDataset
from open_value_estimator.eval import evaluate
from open_value_estimator.utils import (
    compute_explained_variance,
    compute_td_error_magnitude,
    load_checkpoint,
    preprocess_batch,
)
from open_value_estimator.value_estimator import OpenValueEstimator

if TYPE_CHECKING:
    from accelerate import Accelerator


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    """Update EMA model weights: ema = decay * ema + (1 - decay) * model."""
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)


def flatten_optimizer_state(optimizer: torch.optim.Optimizer) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Flatten optimizer state into tensors and metadata for safetensors."""
    state_dict = optimizer.state_dict()
    tensors = {}
    metadata = {}

    # Store param_groups as JSON metadata
    metadata["param_groups"] = json.dumps(state_dict["param_groups"])

    # Flatten state tensors
    for param_id, param_state in state_dict["state"].items():
        for key, value in param_state.items():
            if isinstance(value, torch.Tensor):
                tensors[f"optimizer.{param_id}.{key}"] = value
            else:
                # Store scalars (like step) in metadata
                metadata[f"optimizer.{param_id}.{key}"] = json.dumps(value)

    return tensors, metadata


def lr_schedule_multiplier(current_step: int, warmup_steps: int, total_steps: int) -> float:
    """Compute the LR multiplier for linear warmup followed by cosine decay."""
    if current_step < warmup_steps:
        return current_step / max(1, warmup_steps)

    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train(cfg: Config, accelerator: "Accelerator | None" = None):
    """Main training loop.

    Args:
        cfg: Configuration object.
        accelerator: Optional HuggingFace Accelerator for distributed/mixed precision training.
    """
    # Set up device
    if accelerator:
        device = accelerator.device
    else:
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Validate run_name is provided
    if not cfg.run_name:
        raise ValueError("run_name is required for training")

    is_main_process = accelerator is None or accelerator.is_main_process
    output_dir = Path(cfg.output_dir)

    # Output dir creation and existence check — main process only
    # (non-main processes never write to output_dir)
    if is_main_process:
        if output_dir.exists() and not cfg.training.resume_from:
            raise ValueError(
                f"Output directory already exists: {output_dir}\n"
                "Use a different run_name or set training.resume_from to continue."
            )
        output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb only on main process
    use_wandb = "WANDB_API_KEY" in os.environ and is_main_process
    if use_wandb:
        run_name = cfg.wandb.name or cfg.run_name
        if wandb.run is None:
            wandb.init(
                project=cfg.wandb.project,
                name=run_name,
                config=asdict(cfg),
                dir=str(output_dir),
            )
        else:
            # Sweep mode: run already initialized by sweep agent
            wandb.config.update(asdict(cfg), allow_val_change=True)
        logging.info(f"WandB logging enabled: {cfg.wandb.project}/{run_name}")
    elif is_main_process:
        logging.info("WandB logging disabled (WANDB_API_KEY not set)")

    # Load dataset with precomputed value targets
    value_dataset = ValueDataset(cfg.data)
    logging.info(f"Features: {list(value_dataset.meta.features.keys())}")

    # Create dataloader (split batch across GPUs when using accelerator)
    logging.info("Creating dataloader with EpisodeAwareSampler")
    if accelerator and accelerator.num_processes > 1:
        per_gpu_batch_size = cfg.data.batch_size // accelerator.num_processes
        if per_gpu_batch_size < 1:
            per_gpu_batch_size = 1
        logging.info(
            f"Batch size: {cfg.data.batch_size} total -> {per_gpu_batch_size} per GPU "
            f"({accelerator.num_processes} GPUs)"
        )
        dataloader = value_dataset.create_dataloader(batch_size=per_gpu_batch_size)
    else:
        dataloader = value_dataset.create_dataloader()

    # Create model (loads directly to device)
    logging.info("Creating OpenValueEstimator model")
    model = OpenValueEstimator.from_config(cfg.model, device=device)

    # Create optimizer (Pi0.5 uses beta2=0.95)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=(cfg.training.adam_beta1, cfg.training.adam_beta2),
    )

    # Create learning rate scheduler: linear warmup then cosine decay to 0
    warmup_steps = cfg.training.warmup_steps
    total_steps = cfg.training.num_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda current_step: lr_schedule_multiplier(current_step, warmup_steps, total_steps),
    )

    # Prepare with accelerator if provided
    if accelerator:
        model, optimizer, dataloader, scheduler = accelerator.prepare(
            model, optimizer, dataloader, scheduler
        )
        logging.info("Prepared model, optimizer, dataloader, scheduler with Accelerator")

    # Load checkpoint if specified
    start_step = 0
    if cfg.training.resume_from:
        # Resume training: load model, optimizer, and scheduler
        checkpoint_info = load_checkpoint(
            cfg.training.resume_from,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        start_step = checkpoint_info["step"]
        logging.info(f"Resuming training from step {start_step}")
    elif cfg.training.pretrained:
        # Load pretrained weights only (fresh optimizer/scheduler)
        load_checkpoint(cfg.training.pretrained, model, device=device)
        logging.info("Loaded pretrained weights (starting fresh training)")

    # Initialize EMA model (shadow copy of weights for stable evaluation)
    use_ema = cfg.training.ema_decay > 0
    ema_model = None
    if use_ema:
        unwrapped = accelerator.unwrap_model(model) if accelerator else model
        ema_model = copy.deepcopy(unwrapped)
        for param in ema_model.parameters():
            param.requires_grad = False
        ema_model.eval()
        logging.info(f"EMA enabled with decay={cfg.training.ema_decay}")

    # Training loop
    # Each "step" is one optimizer update. With gradient accumulation,
    # each step runs `accum_steps` micro-steps (forward/backward passes).
    accum_steps = cfg.accelerate.gradient_accumulation_steps
    total_micro_steps = cfg.training.num_steps * accum_steps
    logging.info(
        f"Starting training for {cfg.training.num_steps} optimizer steps "
        f"({total_micro_steps} micro-steps, "
        f"warmup: {cfg.training.warmup_steps}, grad_accum: {accum_steps})"
    )
    model.train()
    data_iter = iter(dataloader)

    # Timing accumulators (reset at each log interval)
    timing = {
        "data_load": 0.0,
        "preprocess": 0.0,
        "forward": 0.0,
        "loss": 0.0,
        "backward": 0.0,
        "optimizer": 0.0,
    }
    timing_count = 0

    for step in range(start_step, cfg.training.num_steps):
        # --- Accumulation micro-steps: forward/backward passes ---
        for micro_step in range(accum_steps):
            # Get batch (cycle if exhausted)
            t0 = time.perf_counter()
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            timing["data_load"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            batch = preprocess_batch(
                batch,
                device=device,
                state_stats=value_dataset.state_stats,
                augment=cfg.data.augment_images,
            )
            targets = batch["value_target"]  # [B] scalar values
            timing["preprocess"] += time.perf_counter() - t0

            if accelerator:
                # accumulate() gates gradient sync and optimizer stepping
                with accelerator.accumulate(model):
                    t0 = time.perf_counter()
                    logits = model(batch)  # [B, num_bins]
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    timing["forward"] += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    unwrapped_model = accelerator.unwrap_model(model)
                    target_indices = unwrapped_model.get_target_bin_indices(targets)  # [B]
                    loss = torch.nn.functional.cross_entropy(logits, target_indices)
                    timing["loss"] += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    accelerator.backward(loss)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    timing["backward"] += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    if cfg.training.grad_clip_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    timing["optimizer"] += time.perf_counter() - t0
            else:
                t0 = time.perf_counter()
                logits = model(batch)  # [B, num_bins]
                if device.type == "cuda":
                    torch.cuda.synchronize()
                timing["forward"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                target_indices = model.get_target_bin_indices(targets)  # [B]
                loss = torch.nn.functional.cross_entropy(logits, target_indices)
                scaled_loss = loss / accum_steps
                timing["loss"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                scaled_loss.backward()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                timing["backward"] += time.perf_counter() - t0

        # --- Optimizer step (once per outer step) ---
        if accelerator:
            if accelerator.sync_gradients:
                scheduler.step()
                if use_ema:
                    update_ema(ema_model, accelerator.unwrap_model(model), cfg.training.ema_decay)
        else:
            t0 = time.perf_counter()
            if cfg.training.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if use_ema:
                update_ema(ema_model, model, cfg.training.ema_decay)
            timing["optimizer"] += time.perf_counter() - t0

        timing_count += 1

        # Logging (only on main process when using accelerator)
        if (step + 1) % cfg.training.log_freq == 0 and is_main_process:
            unwrapped_model = accelerator.unwrap_model(model) if accelerator else model
            expected_values = unwrapped_model.get_expected_value(logits)
            mae = (expected_values - targets).abs().mean().item()
            td_error_magnitude = compute_td_error_magnitude(expected_values, targets)
            explained_variance = compute_explained_variance(expected_values, targets)
            current_lr = scheduler.get_last_lr()[0]
            logging.info(
                f"Step {step + 1}/{cfg.training.num_steps}, Loss: {loss.item():.4f}, MAE: {mae:.4f}, "
                f"EV: {explained_variance:.3f}, LR: {current_lr:.2e}"
            )
            if use_wandb:
                avg_timing = {f"timing/{k}_ms": (v / timing_count) * 1000 for k, v in timing.items()}
                avg_timing["timing/total_ms"] = sum(avg_timing.values())

                wandb.log(
                    {
                        "loss": loss.item(),
                        "mae": mae,
                        "td_error_magnitude": td_error_magnitude,
                        "explained_variance": explained_variance,
                        "learning_rate": current_lr,
                        **avg_timing,
                    },
                    step=step + 1,
                )

            # Reset timing accumulators
            timing = {k: 0.0 for k in timing}
            timing_count = 0

        # Save checkpoint (only on main process when using accelerator)
        if (step + 1) % cfg.training.save_freq == 0 and is_main_process:
            checkpoint_path = output_dir / f"checkpoint_{step + 1}.safetensors"

            unwrapped_model = accelerator.unwrap_model(model) if accelerator else model

            # Clone tensors to handle weight-tied models (e.g., Gemma's embed_tokens/lm_head)
            tensors = {f"model.{k}": v.clone() for k, v in unwrapped_model.state_dict().items()}
            optimizer_tensors, optimizer_metadata = flatten_optimizer_state(optimizer)
            tensors.update({k: v.clone() for k, v in optimizer_tensors.items()})

            metadata = {
                "step": str(step + 1),
                "scheduler_state_dict": json.dumps(scheduler.state_dict()),
            }
            metadata.update(optimizer_metadata)

            save_safetensors(tensors, checkpoint_path, metadata=metadata)

            # Save EMA weights separately (these are the inference weights)
            if use_ema:
                ema_path = output_dir / f"checkpoint_{step + 1}_ema.safetensors"
                ema_tensors = {f"model.{k}": v.clone() for k, v in ema_model.state_dict().items()}
                save_safetensors(ema_tensors, ema_path, metadata={"step": str(step + 1)})
                logging.info(f"Saved EMA checkpoint to {ema_path}")

            config_path = checkpoint_path.with_suffix(".json")
            with open(config_path, "w") as f:
                json.dump(asdict(cfg.model), f, indent=2)

            logging.info(f"Saved checkpoint to {checkpoint_path}")

        # Run evaluation (only on main process when using accelerator)
        if cfg.training.eval_freq > 0 and (step + 1) % cfg.training.eval_freq == 0 and is_main_process:
            logging.info(f"Running evaluation at step {step + 1}")
            eval_model = ema_model if use_ema else (accelerator.unwrap_model(model) if accelerator else model)
            video_path = evaluate(eval_model, cfg, output_dir, step + 1, device, dataset=value_dataset)
            if use_wandb:
                wandb.log({"eval_video": wandb.Video(str(video_path), fps=10, format="mp4")}, step=step + 1)
            model.train()  # Switch back to train mode

    # Final logging (only on main process)
    if is_main_process:
        logging.info("Training complete!")
        if use_wandb:
            wandb.finish()


def main() -> None:
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Train the open value estimator with OmegaConf-style key=value arguments."
        ),
        epilog=(
            "Example:\n"
            "  python -m open_value_estimator.training "
            "config=configs/config.yaml "
            "run_name=my_run training.num_steps=10000"
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

    config_dict = load_config(config_path=config_path, overrides=override_dict)
    logging.info(f"Configuration:\n{OmegaConf.to_yaml(OmegaConf.create(config_dict))}")

    config = Config.from_dict(config_dict)
    train(config)


if __name__ == "__main__":
    main()
