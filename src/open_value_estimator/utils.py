"""Shared utilities for open value estimator."""

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.functional as TF
from safetensors.torch import load_file as load_safetensors
from torchvision.transforms.functional import InterpolationMode


def load_checkpoint_metadata(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load metadata from a safetensors checkpoint.

    Args:
        checkpoint_path: Path to .safetensors checkpoint file.

    Returns:
        Dict with checkpoint metadata (step, model_config, etc.).
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with open(checkpoint_path, "rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        header_bytes = f.read(header_size)
        header = json.loads(header_bytes)

    return _extract_safetensors_metadata(header)


def load_model_weights(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    device: torch.device | str | None = None,
    strict: bool = False,
) -> None:
    """Load model weights from a safetensors checkpoint.

    Handles checkpoints saved with or without "model." prefix on keys.

    Args:
        checkpoint_path: Path to .safetensors checkpoint file.
        model: The model to load weights into.
        device: Device to load tensors to (defaults to "cpu").
        strict: Whether to require exact key matching.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device_str = str(device) if device else "cpu"
    tensors = load_safetensors(checkpoint_path, device=device_str)

    # Check if weights have "model." prefix (training checkpoint format)
    has_prefix = any(k.startswith("model.") for k in tensors.keys())

    if has_prefix:
        model_state = {k[6:]: v for k, v in tensors.items() if k.startswith("model.")}
    else:
        # Direct model weights (no prefix)
        model_state = {k: v for k, v in tensors.items() if not k.startswith("optimizer.")}

    missing, unexpected = model.load_state_dict(model_state, strict=strict)
    if missing:
        logging.warning(f"Missing keys in checkpoint: {missing}")
    if unexpected:
        logging.warning(f"Unexpected keys in checkpoint: {unexpected}")

    logging.info(f"Loaded {len(model_state)} model parameters from {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Load model weights and optionally optimizer/scheduler state from checkpoint.
    
    Args:
        checkpoint_path: Path to .safetensors checkpoint file.
        model: The model to load weights into.
        optimizer: Optional optimizer to restore state.
        scheduler: Optional scheduler to restore state.
        device: Device to load tensors to (uses model's device if None).
        
    Returns:
        Dict with checkpoint metadata including:
            - step: Training step when checkpoint was saved
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Load model weights
    load_model_weights(checkpoint_path, model, device=device)
    
    # Load tensors for optimizer/scheduler state
    tensors = load_safetensors(checkpoint_path, device=str(device) if device else "cpu")
    
    # Extract metadata from safetensors file
    # safetensors stores metadata as strings in the header
    with open(checkpoint_path, "rb") as f:
        # Read header size (first 8 bytes, little-endian)
        header_size = int.from_bytes(f.read(8), "little")
        header_bytes = f.read(header_size)
        header = json.loads(header_bytes)

    metadata = _extract_safetensors_metadata(header)
    
    result = {
        "step": int(metadata.get("step", 0)),
    }
    
    # Optionally restore optimizer state
    if optimizer is not None:
        optimizer_state = _reconstruct_optimizer_state(tensors, metadata)
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
            logging.info("Restored optimizer state")
    
    # Optionally restore scheduler state
    if scheduler is not None and "scheduler_state_dict" in metadata:
        scheduler_state = json.loads(metadata["scheduler_state_dict"])
        scheduler.load_state_dict(scheduler_state)
        logging.info("Restored scheduler state")
    
    return result


def _extract_safetensors_metadata(header: dict[str, Any]) -> dict[str, str]:
    """Extract user metadata from a safetensors header."""
    metadata = header.get("__metadata__", {})
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise ValueError("Invalid safetensors metadata: expected a dict in __metadata__")
    return metadata


def _reconstruct_optimizer_state(
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str],
) -> dict[str, Any] | None:
    """Reconstruct optimizer state dict from flattened checkpoint.
    
    Args:
        tensors: All tensors from checkpoint.
        metadata: Checkpoint metadata.
        
    Returns:
        Optimizer state dict or None if not present.
    """
    if "param_groups" not in metadata:
        return None
    
    state_dict = {
        "param_groups": json.loads(metadata["param_groups"]),
        "state": {},
    }
    
    # Reconstruct state from flattened tensors
    for key, value in tensors.items():
        if key.startswith("optimizer."):
            parts = key.split(".")
            if len(parts) >= 3:
                param_id = int(parts[1])
                state_key = ".".join(parts[2:])
                
                if param_id not in state_dict["state"]:
                    state_dict["state"][param_id] = {}
                state_dict["state"][param_id][state_key] = value
    
    # Restore scalar values from metadata
    for key, value in metadata.items():
        if key.startswith("optimizer."):
            parts = key.split(".")
            if len(parts) >= 3:
                param_id = int(parts[1])
                state_key = ".".join(parts[2:])
                
                if param_id not in state_dict["state"]:
                    state_dict["state"][param_id] = {}
                state_dict["state"][param_id][state_key] = json.loads(value)
    
    return state_dict


def augment_images(images: torch.Tensor) -> torch.Tensor:
    """Apply random augmentations to a batch of images (Pi0.5-style).

    Applies random crop (resized back), small rotation, and color jitter.
    All cameras in a batch sample get the same spatial augmentation
    but independent color jitter.

    Args:
        images: Tensor of shape [B, N, C, H, W] (uint8 or float).

    Returns:
        Augmented images with same shape and dtype.
    """
    B, N, C, H, W = images.shape
    dtype = images.dtype

    # Work in float for augmentation
    imgs = images.float()
    if imgs.max() > 1.0:
        imgs = imgs / 255.0
        was_uint8 = True
    else:
        was_uint8 = False

    # Flatten B*N for per-image operations
    imgs = imgs.view(B * N, C, H, W)

    # Random resized crop (scale 0.9-1.0 to avoid too aggressive cropping)
    crop_scale = torch.empty(B).uniform_(0.9, 1.0)
    crop_ratio = torch.empty(B).uniform_(0.95, 1.05)
    for b in range(B):
        s = crop_scale[b].item()
        r = crop_ratio[b].item()
        crop_h = int(H * s)
        crop_w = int(crop_h * r)
        crop_w = min(crop_w, W)
        top = torch.randint(0, max(1, H - crop_h + 1), (1,)).item()
        left = torch.randint(0, max(1, W - crop_w + 1), (1,)).item()
        # Apply same crop to all cameras for this sample
        for n in range(N):
            idx = b * N + n
            cropped = imgs[idx:idx+1, :, top:top+crop_h, left:left+crop_w]
            imgs[idx] = TF.resize(cropped, [H, W], antialias=True).squeeze(0)

    # Small random rotation (±5 degrees)
    angles = torch.empty(B).uniform_(-5.0, 5.0)
    for b in range(B):
        angle = angles[b].item()
        for n in range(N):
            idx = b * N + n
            imgs[idx] = TF.rotate(imgs[idx], angle)

    # Color jitter (independent per image): brightness, contrast, saturation
    for i in range(B * N):
        brightness = torch.empty(1).uniform_(0.9, 1.1).item()
        contrast = torch.empty(1).uniform_(0.9, 1.1).item()
        saturation = torch.empty(1).uniform_(0.9, 1.1).item()
        imgs[i] = TF.adjust_brightness(imgs[i], brightness)
        imgs[i] = TF.adjust_contrast(imgs[i], contrast)
        imgs[i] = TF.adjust_saturation(imgs[i], saturation)

    imgs = imgs.clamp(0.0, 1.0)
    if was_uint8:
        imgs = (imgs * 255.0).to(dtype)
    else:
        imgs = imgs.to(dtype)

    return imgs.view(B, N, C, H, W)


def preprocess_batch(
    batch: dict[str, Any],
    device: torch.device,
    state_stats: dict[str, torch.Tensor] | None = None,
    augment: bool = False,
    image_keys: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Preprocess a batch of data.

    Combines multiple camera images into a single tensor, normalizes observations,
    and moves all tensors to device.

    Args:
        batch: Raw batch dict from dataloader.
        device: Target device.
        state_stats: Optional dict with 'q01' and 'q99' tensors for normalizing
            observation.state to [-1, 1] range.
        augment: Whether to apply image augmentation.
        image_keys: Optional ordered list of camera tensor keys to stack into
            observation.images. If None, all camera keys are used.

    Returns:
        Preprocessed batch with observation.images stacked as [B, N, C, H, W]
        and observation.state normalized if stats provided.
    """
    # Combine images into a single tensor along an extra dimension (N) for number of cameras
    # Only include keys that are actual image tensors (4D or 5D tensors)
    all_image_keys = [k for k in batch if "observation.images" in k and k != "observation.images"]
    if image_keys is None:
        selected_image_keys = all_image_keys
    else:
        available = set(all_image_keys)
        missing = [k for k in image_keys if k not in available]
        if missing:
            raise ValueError(
                f"Requested camera view keys not found in batch: {missing}. "
                f"Available: {sorted(all_image_keys)}"
            )
        selected_image_keys = image_keys

    images = []
    for k in selected_image_keys:
        tensor = batch[k]
        # Only include tensors that are actually images (4D: [B, C, H, W] or 5D: [B, T, C, H, W])
        if isinstance(tensor, torch.Tensor) and tensor.dim() >= 4:
            # If 5D tensor [B, T, C, H, W], take only the last timestep to get [B, C, H, W]
            # The value estimator expects [B, N, C, H, W] where N is number of cameras
            if tensor.dim() == 5:
                tensor = tensor[:, -1, ...]  # Take last timestep: [B, T, C, H, W] -> [B, C, H, W]
            images.append(tensor)
    
    # Remove all individual camera keys from batch (they are represented in observation.images)
    for k in all_image_keys:
        if k in batch:
            del batch[k]

    if images:
        # Stack along dim=1 to create [B, N, C, H, W] where N is number of cameras
        batch["observation.images"] = torch.stack(images, dim=1)

    # Apply image augmentation (crop, rotation, color jitter) during training
    if augment and "observation.images" in batch:
        batch["observation.images"] = augment_images(batch["observation.images"])

    # Normalize observation.state using quantile normalization to [-1, 1] (Pi0.5-style)
    if state_stats is not None and "observation.state" in batch:
        state = batch["observation.state"]
        q01 = state_stats["q01"].to(state.device)
        q99 = state_stats["q99"].to(state.device)
        # Avoid division by zero
        range_vals = q99 - q01
        range_vals = torch.where(range_vals < 1e-6, torch.ones_like(range_vals), range_vals)
        # Quantile normalization to [-1, 1]; values outside q01/q99 get clamped
        batch["observation.state"] = ((state - q01) / range_vals * 2.0 - 1.0).clamp(-1.0, 1.0)

    # Move all tensors to device
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)

    return batch


def compute_td_error_magnitude(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute TD error magnitude (RMSE) between predictions and targets.
    
    Args:
        predictions: Predicted values tensor.
        targets: Target values tensor.
        
    Returns:
        Root mean squared error as a float.
    """
    td_errors = predictions - targets
    return td_errors.pow(2).mean().sqrt().item()


def compute_explained_variance(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute explained variance between predictions and targets.
    
    Explained variance measures how much of the target variance is explained
    by the predictions: EV = 1 - Var(residuals) / Var(targets)
    
    Args:
        predictions: Predicted values tensor.
        targets: Target values tensor.
        
    Returns:
        Explained variance as a float:
            - 1.0 = perfect predictions
            - 0.0 = predictions no better than mean
            - <0  = predictions worse than mean
    """
    residuals = predictions - targets
    target_var = targets.var().item()
    residual_var = residuals.var().item()
    return 1.0 - (residual_var / target_var) if target_var > 1e-8 else 0.0


def _normalize_to_chw(img: torch.Tensor) -> torch.Tensor:
    """Ensure a 3D image tensor is in CHW format with 3 channels.

    Accepts CHW or HWC layout. Expands single-channel grayscale to 3 channels.
    """
    if img.ndim != 3:
        raise ValueError(f"Expected 3D tensor (CHW or HWC). Got shape={tuple(img.shape)}")

    # HWC -> CHW
    if img.shape[-1] == 3 and img.shape[0] != 3:
        img = img.permute(2, 0, 1)

    # Grayscale 1-channel -> 3 channels
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)

    if img.shape[0] != 3:
        raise ValueError(f"Expected 3 channels after conversion. Got shape={tuple(img.shape)}")

    return img


def _rescale_and_normalize(t: torch.Tensor) -> torch.Tensor:
    """Rescale to [0, 1] if needed, then normalize to [-1, 1] for SigLIP.

    uint8 tensors are divided by 255. Float tensors are assumed to already
    be in [0, 1] (standard for torchvision/dataloader pipelines).
    """
    if t.dtype == torch.uint8:
        t = t.to(torch.float32) / 255.0
    else:
        t = t.to(torch.float32)
    # Normalize: (x - 0.5) / 0.5  -> [-1, 1]
    return (t - 0.5) / 0.5


def siglip_preprocess(
    images,
    size: int | tuple[int, int] = 224,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float16,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    antialias: bool = True,
) -> torch.Tensor:
    """Implements HF SigLIP preprocessing in torch/torchvision.

    Converts to RGB, resizes, rescales by 1/255 if inputs look like uint8,
    and normalizes with mean=std=[0.5,0.5,0.5] to produce pixel_values in [-1, 1].

    Returns: float tensor (B, 3, H, W) on `device` and `dtype`.
    """
    if isinstance(size, int):
        out_h, out_w = size, size
    else:
        out_h, out_w = size

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Handle batched tensor input (B, C, H, W) efficiently
    if torch.is_tensor(images) and images.ndim == 4:
        t = images.to(device)

        # Ensure CHW format
        if t.shape[1] != 3 and t.shape[-1] == 3:
            t = t.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)

        # Handle grayscale -> RGB
        if t.shape[1] == 1:
            t = t.repeat(1, 3, 1, 1)

        t = TF.resize(t, size=[out_h, out_w], interpolation=interpolation, antialias=antialias)
        t = _rescale_and_normalize(t)
        return t.to(dtype)

    # Handle list/tuple of images (fallback for compatibility)
    if not isinstance(images, (list, tuple)):
        images = [images]

    batch = []
    for img in images:
        if not torch.is_tensor(img):
            raise TypeError("images must be a torch.Tensor, or a list/tuple of those.")
        t = _normalize_to_chw(img).to(device)
        t = TF.resize(t, size=[out_h, out_w], interpolation=interpolation, antialias=antialias)
        t = _rescale_and_normalize(t)
        batch.append(t.to(dtype))

    return torch.stack(batch, dim=0)  # (B, 3, H, W)
