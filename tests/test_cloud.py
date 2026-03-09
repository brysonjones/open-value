"""Tests for cloud launcher helpers."""

from __future__ import annotations

import copy

import pytest

from open_value_estimator.cloud import REMOTE_PACKAGES
from open_value_estimator.cloud.cloud_launcher import (
    _get_gpu_configs,
    _merge_override_dicts,
    _normalize_list_arg,
    _resolve_gcp_mount_path,
    build_eval_config,
    load_config_gcp,
)


def test_remote_packages_include_core_dependencies() -> None:
    expected = {"torch>=2.7.0", "lerobot>=0.4.2", "transformers>=4.57.0"}
    assert expected.issubset(set(REMOTE_PACKAGES))


def test_merge_override_dicts_merges_nested_mappings() -> None:
    merged = _merge_override_dicts(
        {"training": {"num_steps": 100}, "data": {"repo_id": "a"}},
        {"training": {"warmup_steps": 10}, "data": {"repo_id": "b"}},
    )

    assert merged["training"] == {"num_steps": 100, "warmup_steps": 10}
    assert merged["data"]["repo_id"] == "b"


def test_normalize_list_arg_accepts_scalar_list_and_none() -> None:
    assert _normalize_list_arg(None, "episodes") is None
    assert _normalize_list_arg("top", "camera_views") == ["top"]
    assert _normalize_list_arg(3, "episodes") == [3]
    assert _normalize_list_arg([1, 2], "episodes") == [1, 2]


def test_normalize_list_arg_rejects_invalid_type() -> None:
    with pytest.raises(ValueError, match="episodes must be a list or scalar"):
        _normalize_list_arg({"bad": True}, "episodes")


def test_resolve_gcp_mount_path_handles_relative_absolute_and_none() -> None:
    assert _resolve_gcp_mount_path(None, "/gcs/test") is None
    assert _resolve_gcp_mount_path("outputs/run", "/gcs/test") == "/gcs/test/outputs/run"
    assert _resolve_gcp_mount_path("/tmp/run", "/gcs/test") == "/tmp/run"


def test_build_eval_config_uses_defaults_from_config(make_config_yaml, full_config_dict: dict, gcp_settings: dict) -> None:
    config_dict = copy.deepcopy(full_config_dict)
    config_dict["eval"].update(
        {
            "checkpoint": "outputs/model/checkpoint_100.safetensors",
            "episodes": [1, 3],
            "gpu": "t4",
            "camera_views": ["top", "wrist"],
            "batch_size": 32,
            "use_ema": True,
            "output_dir": "outputs/eval",
        }
    )
    config_path = make_config_yaml(config_dict, name="eval_defaults.yaml")

    eval_config, gpu = build_eval_config(str(config_path), gcp_settings)

    assert gpu == "t4"
    assert eval_config["checkpoint"] == "/gcs/test-bucket/outputs/model/checkpoint_100.safetensors"
    assert eval_config["dataset"] == "dummy_dataset"
    assert eval_config["data_root"] == "/gcs/test-bucket/datasets/dummy_dataset"
    assert eval_config["episodes"] == [1, 3]
    assert eval_config["output_dir"] == "/gcs/test-bucket/outputs/eval"
    assert eval_config["camera_views"] == ["top", "wrist"]
    assert eval_config["use_ema"] is True
    assert eval_config["show_ground_truth_reward"] is False


def test_build_eval_config_applies_overrides_and_no_ema(make_config_yaml, full_config_dict: dict, gcp_settings: dict) -> None:
    config_dict = copy.deepcopy(full_config_dict)
    config_dict["eval"].update(
        {
            "checkpoint": "outputs/model/checkpoint_100.safetensors",
            "episodes": [1, 3],
            "gpu": "t4",
            "camera_views": ["top", "wrist"],
            "batch_size": 32,
            "use_ema": True,
            "output_dir": "outputs/eval",
        }
    )
    config_path = make_config_yaml(config_dict, name="eval_overrides.yaml")

    eval_config, gpu = build_eval_config(
        str(config_path),
        gcp_settings,
        overrides={
            "checkpoint": "checkpoints/model.safetensors",
            "episodes": 5,
            "camera_views": "front",
            "output_dir": "custom_eval",
            "gpu": "h100",
            "batch_size": 64,
            "no_ema": True,
        },
    )

    assert gpu == "h100"
    assert eval_config["checkpoint"] == "/gcs/test-bucket/checkpoints/model.safetensors"
    assert eval_config["episodes"] == [5]
    assert eval_config["camera_views"] == ["front"]
    assert eval_config["output_dir"] == "/gcs/test-bucket/custom_eval"
    assert eval_config["batch_size"] == 64
    assert eval_config["use_ema"] is False


def test_build_eval_config_requires_checkpoint(make_config_yaml, full_config_dict: dict, gcp_settings: dict) -> None:
    config_dict = copy.deepcopy(full_config_dict)
    config_dict["eval"].update({"checkpoint": None, "episodes": [1]})
    config_path = make_config_yaml(config_dict, name="missing_checkpoint.yaml")

    with pytest.raises(ValueError, match="checkpoint=... is required for eval mode"):
        build_eval_config(str(config_path), gcp_settings, overrides={"episodes": [1]})


def test_load_config_gcp_rewrites_output_dir_and_data_root(make_config_yaml, full_config_dict: dict, gcp_settings: dict) -> None:
    config_path = make_config_yaml(copy.deepcopy(full_config_dict), name="train.yaml")

    config_dict = load_config_gcp(str(config_path), gcp_settings)

    assert config_dict["output_dir"] == "/gcs/test-bucket/outputs/open_value_estimator/test_run"
    assert config_dict["data"]["root"] == "/gcs/test-bucket/datasets/dummy_dataset"


def test_get_gpu_configs_contains_expected_gpu_families() -> None:
    gpu_configs = _get_gpu_configs()

    for gpu in ["t4", "a100", "h100"]:
        assert gpu in gpu_configs
        assert set(gpu_configs[gpu]) == {"machine_type", "accelerator_type", "accelerator_count"}
