"""Configuration dataclasses and loading utilities for the value estimator."""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from omegaconf import OmegaConf


def validate_model_config_keys(d: dict) -> None:
    """Raise ValueError if dict contains keys not in ModelConfig."""
    from dataclasses import fields as dataclass_fields

    valid_keys = {f.name for f in dataclass_fields(ModelConfig)}
    unknown = set(d) - valid_keys
    if unknown:
        raise ValueError(
            f"Unknown keys in model config: {unknown}. "
            f"Valid keys are: {sorted(valid_keys)}"
        )


def _resolve_config_path(config_path: str | Path, label: str) -> Path:
    """Resolve and validate a user-provided config path."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def parse_cli_overrides(overrides: list[str]) -> dict:
    """Parse OmegaConf dotlist CLI overrides into a plain dict."""
    if not overrides:
        return {}

    override_cfg = OmegaConf.from_dotlist(overrides)
    return OmegaConf.to_container(override_cfg, resolve=True)


@dataclass
class ModelConfig:
    """Model configuration."""

    model_name: str = "google/siglip-base-patch16-224"
    num_cameras: int = 4
    hidden_dim: int = 768
    num_bins: int = 201  # Distributional value head bins
    v_min: float = -1.0  # Min value for distribution
    v_max: float = 0.0   # Max value for distribution
    vlm_model_name: str = "google/gemma-3-270m"  # Gemma model for VLM backbone
    freeze_vision_encoder: bool = True  # Freeze SigLIP vision encoder (Pi0.5-style)
    value_head_depth: int = 1  # Number of hidden layers in the value head MLP
    threshold_percentile: float = 30.0  # Percentile threshold for binarized advantage


class AdvantageMode(str, Enum):
    """Supported advantage computation modes."""

    N_STEP = "n_step"
    FULL_EPISODE = "full_episode"


@dataclass
class DataConfig:
    """Data configuration."""

    repo_id: str = "datasets/fine_tuning_12_11_25_with_rewards"
    root: str = "./data/fine_tuning_12_11_25_with_rewards"
    batch_size: int = 8
    num_workers: int = 4
    drop_n_last_frames: int = 0
    shuffle: bool = True
    fail_penalty: float = 100.0  # Large negative constant for failure at episode end (terminal reward mode only)
    precomputed_rewards: bool = False  # If True, use per-step rewards from dataset directly (skip reward shaping)
    augment_images: bool = True  # Apply random image augmentations during training


@dataclass
class AdvantageConfig:
    """Configuration for offline advantage dataset generation."""

    checkpoint: str | None = None  # Path to pretrained value estimator checkpoint
    use_ema: bool = True  # Prefer EMA weights when available
    mode: AdvantageMode = AdvantageMode.N_STEP  # Advantage computation mode
    n_step: int = 50  # Lookahead horizon for n-step mode
    batch_size: int = 64  # Inference batch size for value prediction
    output_repo_id: str | None = None  # Name for the derived dataset
    output_root: str | None = None  # Output root for the derived dataset
    stats_quantiles: list[float] = field(
        default_factory=lambda: [0.01, 0.10, 0.50, 0.90, 0.99]
    )  # Global quantiles to store for the advantage feature


@dataclass
class TrainingConfig:
    """Training configuration."""

    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95  # Pi0.5-style (default PyTorch is 0.999)
    num_steps: int = 10000
    warmup_steps: int = 1000  # Linear warmup from 0 to learning_rate
    log_freq: int = 100
    save_freq: int = 1000
    eval_freq: int = 2000  # Run evaluation every N steps (0 to disable)
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.999  # EMA decay rate (0 to disable)
    resume_from: str | None = None  # Path to checkpoint for resuming training
    pretrained: str | None = None  # Path to checkpoint for loading weights only (no optimizer/scheduler)


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""

    project: str = "open-value-estimator"
    name: str | None = None  # Run name, uses output_dir name if None


@dataclass
class EvalConfig:
    """Evaluation visualization configuration."""

    show_ground_truth_reward: bool = True


@dataclass
class AccelerateConfig:
    """Accelerate configuration for distributed/mixed precision training."""

    enabled: bool = False  # Set to true to use accelerate
    num_processes: int = 1  # Number of GPUs/processes
    mixed_precision: str = "no"  # "no", "fp16", or "bf16"
    gradient_accumulation_steps: int = 1  # Gradient accumulation steps (handled by Accelerator)


@dataclass
class Config:
    """Main configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    advantage: AdvantageConfig = field(default_factory=AdvantageConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    accelerate: AccelerateConfig = field(default_factory=AccelerateConfig)
    output_dir: str = "outputs"
    run_name: str | None = None  # Required: name for this training run
    seed: int = 42
    device: str = "cuda"

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Create Config from a plain dict (handles nested configs)."""
        model_dict = d.get("model", {}).copy()
        validate_model_config_keys(model_dict)
        advantage_dict = d.get("advantage", {}).copy()
        if "mode" in advantage_dict and advantage_dict["mode"] is not None:
            advantage_dict["mode"] = AdvantageMode(advantage_dict["mode"])
        return cls(
            model=ModelConfig(**model_dict),
            data=DataConfig(**d.get("data", {})),
            advantage=AdvantageConfig(**advantage_dict),
            training=TrainingConfig(**d.get("training", {})),
            wandb=WandbConfig(**d.get("wandb", {})),
            eval=EvalConfig(**d.get("eval", {})),
            accelerate=AccelerateConfig(**d.get("accelerate", {})),
            output_dir=d.get("output_dir", "outputs"),
            run_name=d.get("run_name"),
            seed=d.get("seed", 42),
            device=d.get("device", "cuda"),
        )


def load_config(
    config_path: str | Path,
    overrides: dict | None = None,
    *,
    require_run_name: bool = True,
    append_run_name_to_output_dir: bool = True,
) -> dict:
    """Load config from YAML, apply overrides, and return as a plain dict.

    Cloud launchers should call this and then override paths
    (output_dir, data.root) as needed for their environment.

    Args:
        config_path: Path to YAML config file.
        overrides: Optional dict of config overrides to merge.
        require_run_name: Whether to require run_name to be set after overrides.
        append_run_name_to_output_dir: Whether to append run_name to output_dir.

    Returns:
        Config as a plain dict (serializable for remote execution).
    """
    resolved_config_path = _resolve_config_path(config_path, "Training config")

    cfg = OmegaConf.load(resolved_config_path)
    logging.info(f"Loaded config from {resolved_config_path}")

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))

    config_dict = OmegaConf.to_container(cfg, resolve=True)

    run_name = config_dict.get("run_name")
    if require_run_name and not run_name:
        raise ValueError("run_name is required. Set it in the config file or pass it as an override.")

    if append_run_name_to_output_dir and run_name:
        config_dict["output_dir"] = f"{config_dict['output_dir']}/{run_name}"

    if "data" in config_dict and "repo_id" in config_dict["data"]:
        logging.info(f"Using dataset: {config_dict['data']['repo_id']}")
    if run_name:
        logging.info(f"Run name: {run_name}")
    if "output_dir" in config_dict:
        logging.info(f"Output dir: {config_dict['output_dir']}")

    return config_dict


def load_gcp_settings(
    config_path: str | Path,
    overrides: dict | None = None,
) -> dict:
    """Load GCP deployment settings from the main config YAML.

    Args:
        config_path: Path to the main YAML config file.
        overrides: Optional overrides to merge before reading cloud settings.

    Returns:
        GCP config as a plain dict.
    """
    resolved_config_path = _resolve_config_path(config_path, "Config")
    cfg = OmegaConf.load(resolved_config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    cloud_cfg = config_dict.get("cloud", {})
    if not isinstance(cloud_cfg, dict):
        raise ValueError("cloud section must be a mapping")

    gcp_cfg = cloud_cfg.get("gcp")
    if not isinstance(gcp_cfg, dict):
        raise ValueError("cloud.gcp section is required for GCP launchers")

    required_fields = ["project_id", "bucket", "mount_prefix", "region", "container"]
    missing = [field for field in required_fields if not gcp_cfg.get(field)]
    if missing:
        raise ValueError(f"cloud.gcp is missing required fields: {missing}")

    logging.info(f"Loaded GCP settings from {resolved_config_path}")
    return gcp_cfg


def load_model_config_from_checkpoint(cfg: "Config") -> "Config":
    """Update Config with model architecture from a checkpoint JSON file.

    Loads the JSON sidecar file next to the checkpoint and reconstructs
    the ModelConfig from it. The caller is responsible for ensuring
    cfg.training.resume_from points to a valid path.

    Args:
        cfg: Config object with training.resume_from set.

    Returns:
        Updated Config with model settings from checkpoint.
    """
    if not cfg.training.resume_from:
        return cfg

    checkpoint_json = Path(cfg.training.resume_from).with_suffix(".json")
    if not checkpoint_json.exists():
        raise FileNotFoundError(
            f"Model config not found at {checkpoint_json}. "
            "A .json file must exist alongside the .safetensors checkpoint."
        )

    logging.info(f"Loading model architecture from {checkpoint_json}")
    with open(checkpoint_json) as f:
        model_cfg_dict = json.load(f)

    validate_model_config_keys(model_cfg_dict)
    cfg.model = ModelConfig(**model_cfg_dict)

    return cfg
