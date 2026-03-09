"""Value estimator using SigLIP vision and text encoders."""

import logging
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import save_file
from transformers import (
    AutoTokenizer,
    Gemma3ForCausalLM,
    SiglipVisionModel,
)

from open_value_estimator.utils import siglip_preprocess

OBS_IMAGES = "observation.images"

# Binarized advantage labels
ADVANTAGE_POSITIVE = "Advantage: Positive"
ADVANTAGE_NEGATIVE = "Advantage: Negative"


class MLPHead(nn.Module):
    """MLP projection head with GELU activations.

    With depth=1: Linear(in_dim, hidden_dim) -> GELU -> Linear(hidden_dim, out_dim)
    With depth>1: repeated Linear -> GELU blocks, final Linear to out_dim.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 1):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        layers = []
        for i in range(depth):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OpenValueEstimator(nn.Module):
    """Value estimator using SigLIP vision encoder with Gemma VLM backbone.

    Architecture:
        1. SigLIP vision encoder produces patch embeddings for each camera
        2. Linear projection converts SigLIP features to Gemma token dimension
        3. Proprioception is discretized into 256 bins and encoded as text (Pi0.5-style)
        4. Task + discretized state prompt is tokenized and embedded via Gemma
        5. Visual tokens and text tokens are concatenated
        6. Gemma processes the combined sequence with bidirectional attention
        7. Final hidden states are pooled and projected to value distribution
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        vlm_model_name: str = "google/gemma-3-270m",
        num_cameras: int = 4,
        hidden_dim: int = 768,
        num_bins: int = 201,
        v_min: float = -1.0,
        v_max: float = 0.0,
        freeze_vision_encoder: bool = True,
        value_head_depth: int = 1,
        device: torch.device | str = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.num_bins = num_bins
        self.v_min = v_min
        self.v_max = v_max
        self.num_cameras = num_cameras

        # Precompute bin centers for expected value calculation
        self.register_buffer(
            "bin_centers",
            torch.linspace(v_min, v_max, num_bins, device=self.device)
        )

        # Load SigLIP vision encoder
        self.vision_encoder = SiglipVisionModel.from_pretrained(model_name).to(self.device)
        vision_hidden_dim = self.vision_encoder.config.hidden_size

        # Load Gemma model and tokenizer (use HF token for gated models)
        hf_token = os.environ.get("HF_TOKEN")
        self.vlm = Gemma3ForCausalLM.from_pretrained(
            vlm_model_name,
            token=hf_token,
            dtype=torch.bfloat16,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(vlm_model_name, token=hf_token)
        
        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get Gemma embedding dimension
        gemma_embed_dim = self.vlm.config.hidden_size

        # All tokens attend to all other tokens, not just preceding ones.
        # Padding is still masked correctly by the 2D attention_mask we pass.
        self.vlm.model.config.use_bidirectional_attention = True

        # Optionally freeze vision encoder (Pi0.5 always freezes SigLIP)
        if freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            self.vision_encoder.eval()

        # Projection layer: SigLIP features -> Gemma token dimension
        # Projects each patch embedding to match Gemma's hidden size,
        # then scales by sqrt(embed_dim) to match Gemma's embedding scale (Pi0.5-style)
        self.vision_projection = nn.Linear(vision_hidden_dim, gemma_embed_dim).to(self.device)
        self._vision_scale = gemma_embed_dim ** 0.5

        # Discretization bins for proprioceptive state (Pi0.5-style)
        self.register_buffer(
            "_state_bins",
            torch.linspace(-1, 1, 257, device=self.device)[:-1],  # 256 uniform left edges over [-1, 1]
        )

        # Value head: pool Gemma outputs and project to value distribution
        self.value_head = MLPHead(gemma_embed_dim, hidden_dim, num_bins, depth=value_head_depth).to(self.device)

    @classmethod
    def from_config(cls, model_cfg, device: torch.device | str = "cuda") -> "OpenValueEstimator":
        """Create a OpenValueEstimator from a ModelConfig dataclass.

        Args:
            model_cfg: ModelConfig instance with model hyperparameters.
            device: Device to load model on.

        Returns:
            Initialized OpenValueEstimator.
        """
        return cls(
            model_name=model_cfg.model_name,
            vlm_model_name=model_cfg.vlm_model_name,
            num_cameras=model_cfg.num_cameras,
            hidden_dim=model_cfg.hidden_dim,
            num_bins=model_cfg.num_bins,
            v_min=model_cfg.v_min,
            v_max=model_cfg.v_max,
            freeze_vision_encoder=model_cfg.freeze_vision_encoder,
            value_head_depth=model_cfg.value_head_depth,
            device=device,
        )

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        """Forward pass returning logits over value bins.

        Args:
            batch: Dict with:
                - 'observation.images': [B, N, C, H, W] camera images
                - 'task': list[str] task descriptions
                - 'observation.state': [B, state_dim] joint angles / proprioception

        Returns:
            Logits of shape [B, num_bins] over discretized value distribution.
        """
        B, N, C, H, W = batch["observation.images"].shape
        images = rearrange(batch["observation.images"], "b n c h w -> (b n) c h w")
        device = next(self.parameters()).device

        # 1. ENCODE: SigLIP vision encoder -> patch embeddings [B*N, num_patches, vision_dim]
        input_size = self.vision_encoder.config.image_size
        pixel_values = siglip_preprocess(
            images, size=input_size, device=device, dtype=torch.float32,
        )
        vision_output = self.vision_encoder(pixel_values=pixel_values)
        patch_embeddings = vision_output.last_hidden_state

        # 2. PROJECT: SigLIP features -> Gemma token dimension, scaled by sqrt(embed_dim)
        visual_tokens = self.vision_projection(patch_embeddings) * self._vision_scale
        num_patches = visual_tokens.shape[1]
        visual_tokens = rearrange(visual_tokens, "(b n) p d -> b (n p) d", b=B, n=N)

        # 3. DISCRETIZE STATE: proprioception -> Pi0.5-style text (256 uniform bins)
        state = batch["observation.state"].to(device)
        if state.dim() == 3:
            state = state[:, -1, ...]
        state_strings = self.discretize_state(state)

        # 4. TOKENIZE: format prompt with task + discretized state, then embed
        prompts = [
            f"Task: {task}, State: {state_str};"
            for task, state_str in zip(batch['task'], state_strings)
        ]
        text_inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=128,
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_embeddings = self.vlm.model.embed_tokens(text_inputs['input_ids'])

        # 5. CONCATENATE: visual tokens then text tokens
        visual_tokens_bf16 = visual_tokens.to(text_embeddings.dtype)
        combined_embeddings = torch.cat([visual_tokens_bf16, text_embeddings], dim=1)
        visual_attention = torch.ones(B, N * num_patches, device=device, dtype=text_inputs['attention_mask'].dtype)
        combined_attention_mask = torch.cat([visual_attention, text_inputs['attention_mask']], dim=1)

        # 6. PROCESS: Gemma with bidirectional attention
        vlm_output = self.vlm.model(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            output_hidden_states=True,
        )
        hidden_states = vlm_output.last_hidden_state  # [B, total_seq_len, gemma_dim]

        # 7. POOL: last valid token's hidden state per sequence
        seq_lengths = combined_attention_mask.sum(dim=1) - 1  # [B]
        pooled_output = hidden_states[torch.arange(B, device=device), seq_lengths.long()]

        # 8. VALUE HEAD: project to value distribution
        logits = self.value_head(pooled_output.float())  # [B, num_bins]

        return logits

    def discretize_state(self, state: torch.Tensor) -> list[str]:
        """Discretize normalized [-1,1] state into Pi0.5-style text strings.

        Uses 256 uniform bins over [-1, 1]. Each state dimension is mapped to
        an integer index 0-255, then formatted as a space-separated string.

        Args:
            state: Normalized state values in [-1, 1], shape [B, state_dim].

        Returns:
            List of B strings, each containing space-separated bin indices.
        """
        indices = torch.bucketize(state, self._state_bins) - 1
        indices = indices.clamp(0, 255)
        return [" ".join(str(int(v)) for v in row) for row in indices]

    def get_expected_value(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute expected value from logits over bins.

        Args:
            logits: Logits of shape [B, num_bins].

        Returns:
            Expected values of shape [B].
        """
        probs = F.softmax(logits, dim=-1)  # [B, num_bins]
        expected_value = (probs * self.bin_centers).sum(dim=-1)  # [B]
        return expected_value

    def get_target_bin_indices(self, values: torch.Tensor) -> torch.Tensor:
        """Convert scalar values to target bin indices.

        Args:
            values: Target values of shape [B], in range [v_min, v_max].

        Returns:
            Bin indices of shape [B] (long tensor).
        """
        values = values.clamp(self.v_min, self.v_max)
        bin_indices = ((values - self.v_min) / (self.v_max - self.v_min) * (self.num_bins - 1)).round().long()
        bin_indices = bin_indices.clamp(0, self.num_bins - 1)
        return bin_indices

    def save_pretrained(self, path: str) -> None:
        """Save model weights to a safetensors file.

        Args:
            path: Path to save the safetensors file.
        """
        save_file(self.state_dict(), path)

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        device: torch.device | str = "cpu",
        freeze: bool = True,
        use_ema: bool = True,
    ) -> "OpenValueEstimator":
        """Load a pretrained OpenValueEstimator from a safetensors file.

        Model configuration is loaded from a JSON file alongside the checkpoint.
        E.g., for checkpoint_1000.safetensors, config is in checkpoint_1000.json.

        Args:
            path: Path to the safetensors checkpoint file.
            device: Device to load model on.
            freeze: Whether to freeze all parameters and set to eval mode.
            use_ema: If True, load EMA weights when available (recommended for
                inference). Falls back to regular weights if no EMA file exists.

        Returns:
            OpenValueEstimator with loaded weights.
        """
        import json
        from pathlib import Path

        from open_value_estimator.utils import load_model_weights

        checkpoint_path = Path(path)

        # Resolve which weights file to load
        if use_ema:
            ema_path = checkpoint_path.with_name(
                checkpoint_path.stem.replace("_ema", "") + "_ema" + checkpoint_path.suffix
            )
            if ema_path.exists() and ema_path != checkpoint_path:
                logging.info(f"Loading EMA weights from {ema_path}")
                weights_path = ema_path
            elif checkpoint_path.stem.endswith("_ema"):
                # Already pointing at an EMA file
                weights_path = checkpoint_path
            else:
                logging.info("No EMA checkpoint found, loading regular weights")
                weights_path = checkpoint_path
        else:
            weights_path = checkpoint_path

        # Load config from JSON file alongside checkpoint (strip _ema suffix for config lookup)
        config_stem = checkpoint_path.stem.replace("_ema", "")
        config_path = checkpoint_path.with_name(config_stem + ".json")
        if not config_path.exists():
            raise ValueError(
                f"Config file not found: {config_path}. "
                "Please use a checkpoint saved with the updated training code."
            )
        with open(config_path) as f:
            model_config = json.load(f)

        # Validate config keys match current ModelConfig fields
        from open_value_estimator.config import validate_model_config_keys
        validate_model_config_keys(model_config)

        # Create model from loaded config
        model = cls(
            model_name=model_config["model_name"],
            vlm_model_name=model_config["vlm_model_name"],
            num_cameras=model_config["num_cameras"],
            hidden_dim=model_config["hidden_dim"],
            num_bins=model_config["num_bins"],
            v_min=model_config["v_min"],
            v_max=model_config["v_max"],
            freeze_vision_encoder=model_config.get("freeze_vision_encoder", True),
            value_head_depth=model_config.get("value_head_depth", 1),
            device=device,
        )

        # Load pretrained weights
        load_model_weights(weights_path, model, device=device)

        # Optionally freeze all parameters and set to eval mode
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
            model.eval()

        return model
