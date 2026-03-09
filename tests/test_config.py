"""Tests for configuration loading and validation."""

from __future__ import annotations

import copy
import json

import pytest

from open_value_estimator.config import (
    AdvantageConfig,
    AdvantageMode,
    Config,
    DataConfig,
    EvalConfig,
    ModelConfig,
    TrainingConfig,
    _resolve_config_path,
    load_config,
    load_gcp_settings,
    load_model_config_from_checkpoint,
    parse_cli_overrides,
    validate_model_config_keys,
)


def test_validate_model_config_keys_accepts_all_fields(model_config_dict: dict) -> None:
    validate_model_config_keys(model_config_dict)


def test_validate_model_config_keys_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match="bad_key"):
        validate_model_config_keys({"bad_key": 1})


def test_validate_model_config_keys_accepts_partial_fields() -> None:
    validate_model_config_keys({"num_bins": 101, "value_head_depth": 3})


def test_resolve_config_path_returns_existing_path(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("run_name: test\n", encoding="utf-8")

    resolved = _resolve_config_path(path, "Training config")

    assert resolved == path.resolve()


def test_resolve_config_path_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Training config not found"):
        _resolve_config_path(tmp_path / "missing.yaml", "Training config")


def test_parse_cli_overrides_parses_nested_values() -> None:
    overrides = parse_cli_overrides(
        [
            "run_name=test_run",
            "training.num_steps=200",
            "data.shuffle=false",
            "eval.episodes=[1,2,3]",
        ]
    )

    assert overrides["run_name"] == "test_run"
    assert overrides["training"]["num_steps"] == 200
    assert overrides["data"]["shuffle"] is False
    assert overrides["eval"]["episodes"] == [1, 2, 3]


def test_config_from_dict_creates_nested_subconfigs(full_config_dict: dict) -> None:
    cfg = Config.from_dict(full_config_dict)

    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.data, DataConfig)
    assert isinstance(cfg.advantage, AdvantageConfig)
    assert isinstance(cfg.training, TrainingConfig)
    assert isinstance(cfg.eval, EvalConfig)
    assert cfg.model.hidden_dim == full_config_dict["model"]["hidden_dim"]
    assert cfg.advantage.mode == AdvantageMode.N_STEP
    assert cfg.eval.show_ground_truth_reward is False


def test_config_from_dict_uses_defaults_for_omitted_sections() -> None:
    cfg = Config.from_dict({"run_name": "minimal"})

    assert cfg.model == ModelConfig()
    assert cfg.data == DataConfig()
    assert cfg.training == TrainingConfig()
    assert cfg.eval == EvalConfig()


def test_config_from_dict_rejects_unknown_model_key(full_config_dict: dict) -> None:
    bad = copy.deepcopy(full_config_dict)
    bad["model"]["typo_field"] = True

    with pytest.raises(ValueError, match="typo_field"):
        Config.from_dict(bad)


def test_load_config_appends_run_name_to_output_dir(make_config_yaml) -> None:
    config_path = make_config_yaml()

    config_dict = load_config(config_path=config_path)

    assert config_dict["output_dir"].endswith("/test_run")


def test_load_config_requires_run_name_by_default(make_config_yaml, full_config_dict: dict) -> None:
    config_dict = copy.deepcopy(full_config_dict)
    config_dict["run_name"] = None
    config_path = make_config_yaml(config_dict, name="missing_run_name.yaml")

    with pytest.raises(ValueError, match="run_name is required"):
        load_config(config_path=config_path)


def test_load_config_can_skip_run_name_requirement(make_config_yaml, full_config_dict: dict) -> None:
    config_dict = copy.deepcopy(full_config_dict)
    config_dict["run_name"] = None
    config_path = make_config_yaml(config_dict, name="eval_config.yaml")

    loaded = load_config(config_path=config_path, require_run_name=False)

    assert loaded["run_name"] is None
    assert loaded["output_dir"] == full_config_dict["output_dir"]


def test_load_config_can_skip_output_dir_append(make_config_yaml) -> None:
    config_path = make_config_yaml()

    config_dict = load_config(config_path=config_path, append_run_name_to_output_dir=False)

    assert config_dict["output_dir"] == "./outputs/open_value_estimator"


def test_load_gcp_settings_reads_cloud_section(make_config_yaml, gcp_settings: dict) -> None:
    config_path = make_config_yaml()

    loaded = load_gcp_settings(config_path=config_path)

    assert loaded == gcp_settings


def test_load_gcp_settings_missing_required_field_raises(make_config_yaml, full_config_dict: dict) -> None:
    config_dict = copy.deepcopy(full_config_dict)
    config_dict["cloud"]["gcp"]["container"] = ""
    config_path = make_config_yaml(config_dict, name="bad_cloud.yaml")

    with pytest.raises(ValueError, match="missing required fields"):
        load_gcp_settings(config_path=config_path)


def test_load_model_config_from_checkpoint_updates_model(tmp_path, full_config_dict: dict, model_config_dict: dict) -> None:
    checkpoint_path = tmp_path / "checkpoint_100.safetensors"
    checkpoint_path.write_bytes(b"placeholder")
    json_path = checkpoint_path.with_suffix(".json")
    updated_model = copy.deepcopy(model_config_dict)
    updated_model["num_bins"] = 73
    updated_model["value_head_depth"] = 4
    json_path.write_text(json.dumps(updated_model), encoding="utf-8")

    cfg = Config.from_dict(full_config_dict)
    cfg.training.resume_from = str(checkpoint_path)

    updated = load_model_config_from_checkpoint(cfg)

    assert updated.model.num_bins == 73
    assert updated.model.value_head_depth == 4


def test_load_model_config_from_checkpoint_missing_sidecar_raises(tmp_path, full_config_dict: dict) -> None:
    checkpoint_path = tmp_path / "checkpoint_100.safetensors"
    checkpoint_path.write_bytes(b"placeholder")

    cfg = Config.from_dict(full_config_dict)
    cfg.training.resume_from = str(checkpoint_path)

    with pytest.raises(FileNotFoundError, match="Model config not found"):
        load_model_config_from_checkpoint(cfg)

