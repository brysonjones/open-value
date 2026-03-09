"""Cloud deployment modules for open_value_estimator."""

# Shared pip packages for remote cloud environments (GCP Vertex AI).
# Keep in sync with pyproject.toml [project.dependencies].
REMOTE_PACKAGES = [
    "torch>=2.7.0",
    "lerobot>=0.4.2",
    "huggingface_hub",
    "transformers>=4.57.0",
    "sentencepiece>=0.2.0",
    "einops",
    "hydra-core>=1.3.0",
    "omegaconf>=2.3.0",
    "wandb",
    "safetensors",
    "opencv-python>=4.8.0",
    "accelerate>=0.25.0",
    "matplotlib>=3.8.0",
]
