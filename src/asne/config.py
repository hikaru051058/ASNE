from __future__ import annotations

from pathlib import Path

from .steering import REQUIRED_FIELDS, SteeringCondition, load_steering_conditions


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STEERING_CONFIG = PROJECT_ROOT / "configs" / "steering_conditions.example.yaml"


def load_default_steering_conditions() -> list[SteeringCondition]:
    return load_steering_conditions(DEFAULT_STEERING_CONFIG)


def validate_steering_config(path: str | Path = DEFAULT_STEERING_CONFIG) -> list[SteeringCondition]:
    return load_steering_conditions(path)


__all__ = [
    "DEFAULT_STEERING_CONFIG",
    "PROJECT_ROOT",
    "REQUIRED_FIELDS",
    "SteeringCondition",
    "load_default_steering_conditions",
    "validate_steering_config",
]
