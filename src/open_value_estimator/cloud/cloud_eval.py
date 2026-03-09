"""Dedicated cloud launcher for value estimator evaluation."""

import argparse
import os

from open_value_estimator.cloud.cloud_launcher import (
    build_eval_config,
    launch_eval,
)
from open_value_estimator.config import load_gcp_settings, parse_cli_overrides


def main() -> None:
    """CLI entry point for cloud eval jobs."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Run open value estimator eval in the cloud with key=value arguments.",
        epilog=(
            "Examples:\n"
            "  python -m open_value_estimator.cloud.cloud_eval "
            "config=configs/eval.yaml\n"
            "  python -m open_value_estimator.cloud.cloud_eval "
            "config=configs/eval.yaml "
            "episodes=[0,5,10] gpu=h100 detached=true"
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf-style key=value arguments.",
    )
    args = parser.parse_args()

    override_dict = parse_cli_overrides(args.overrides)

    config_path = override_dict.pop("config", None)
    if not config_path or not isinstance(config_path, str):
        parser.error("config=... is required")

    detached = override_dict.pop("detached", False)
    if not isinstance(detached, bool):
        parser.error("detached must be true or false")

    try:
        gcp_settings = load_gcp_settings(config_path, overrides=override_dict)
    except ValueError as exc:
        parser.error(str(exc))

    if "HF_TOKEN" not in os.environ:
        raise ValueError(
            "HF_TOKEN is not set. This is needed to use LeRobotDatasets. "
            "Please set it in your environment variables and try again."
        )

    try:
        eval_config, gpu = build_eval_config(
            config_path=config_path,
            gcp_settings=gcp_settings,
            overrides=override_dict,
        )
    except ValueError as exc:
        parser.error(str(exc))

    launch_eval(
        config_dict=eval_config,
        gcp_settings=gcp_settings,
        gpu=gpu,
        sync=not detached,
    )


if __name__ == "__main__":
    main()
