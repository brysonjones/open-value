"""Shared pytest fixtures for open_value_estimator tests."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from omegaconf import OmegaConf


@pytest.fixture
def model_config_dict() -> dict:
    return {
        "model_name": "google/siglip-base-patch16-224",
        "num_cameras": 3,
        "hidden_dim": 128,
        "num_bins": 51,
        "v_min": -1.0,
        "v_max": 0.0,
        "vlm_model_name": "google/gemma-3-270m",
        "freeze_vision_encoder": True,
        "value_head_depth": 2,
        "threshold_percentile": 25.0,
    }


@pytest.fixture
def full_config_dict(model_config_dict: dict) -> dict:
    return {
        "model": copy.deepcopy(model_config_dict),
        "data": {
            "repo_id": "dummy_dataset",
            "root": "./data/dummy_dataset",
            "batch_size": 8,
            "num_workers": 0,
            "drop_n_last_frames": 0,
            "shuffle": True,
            "fail_penalty": 100.0,
            "precomputed_rewards": False,
            "augment_images": True,
        },
        "advantage": {
            "checkpoint": "outputs/model/checkpoint_100.safetensors",
            "use_ema": True,
            "mode": "n_step",
            "n_step": 25,
            "batch_size": 16,
            "output_repo_id": "dummy_advantages",
            "output_root": "./data/dummy_advantages",
            "stats_quantiles": [0.01, 0.5, 0.99],
        },
        "training": {
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "num_steps": 100,
            "warmup_steps": 10,
            "log_freq": 5,
            "save_freq": 20,
            "eval_freq": 20,
            "grad_clip_norm": 1.0,
            "ema_decay": 0.999,
            "resume_from": None,
            "pretrained": None,
        },
        "wandb": {
            "project": "open-value-estimator",
            "name": "test_run_name",
        },
        "eval": {
            "show_ground_truth_reward": False,
            "video_fps": 24.0,
        },
        "accelerate": {
            "enabled": False,
            "num_processes": 1,
            "mixed_precision": "no",
            "gradient_accumulation_steps": 2,
        },
        "cloud": {
            "gcp": {
                "project_id": "test-project",
                "bucket": "gs://test-bucket",
                "mount_prefix": "/gcs/test-bucket",
                "region": "us-central1",
                "container": "us-docker.pkg.dev/test/container:latest",
            }
        },
        "output_dir": "./outputs/open_value_estimator",
        "run_name": "test_run",
        "seed": 123,
        "device": "cpu",
    }


@pytest.fixture
def gcp_settings(full_config_dict: dict) -> dict:
    return copy.deepcopy(full_config_dict["cloud"]["gcp"])


@pytest.fixture
def make_config_yaml(tmp_path: Path, full_config_dict: dict):
    def _make_config_yaml(config_dict: dict | None = None, name: str = "config.yaml") -> Path:
        path = tmp_path / name
        OmegaConf.save(config=OmegaConf.create(config_dict or full_config_dict), f=path)
        return path

    return _make_config_yaml
