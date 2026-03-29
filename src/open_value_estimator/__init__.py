"""Open Value Estimator package."""

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

from open_value_estimator.config import (
    AdvantageConfig,
    AdvantageMode,
    Config,
    DataConfig,
    ModelConfig,
    TrainingConfig,
)
from open_value_estimator.dataset import ValueDataset
from open_value_estimator.value_estimator import OpenValueEstimator

if TYPE_CHECKING:
    from open_value_estimator.advantage import compute_dataset_advantages

__all__ = [
    "AdvantageConfig",
    "AdvantageMode",
    "Config",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "ValueDataset",
    "OpenValueEstimator",
    "compute_dataset_advantages",
]


def __getattr__(name: str) -> Any:
    if name == "compute_dataset_advantages":
        from open_value_estimator.advantage import compute_dataset_advantages

        return compute_dataset_advantages
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
