from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REQUIRED_FIELDS = (
    "id",
    "name",
    "description",
    "prompt_prefix",
    "intended_analysis",
    "safety_note",
)


@dataclass(frozen=True)
class SteeringCondition:
    id: str
    name: str
    description: str
    prompt_prefix: str
    intended_analysis: str
    safety_note: str


def condition_from_mapping(data: dict[str, Any]) -> SteeringCondition:
    missing = [field for field in REQUIRED_FIELDS if not data.get(field)]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Steering condition is missing required fields: {joined}")

    return SteeringCondition(
        id=str(data["id"]),
        name=str(data["name"]),
        description=str(data["description"]),
        prompt_prefix=str(data["prompt_prefix"]),
        intended_analysis=str(data["intended_analysis"]),
        safety_note=str(data["safety_note"]),
    )


def load_steering_conditions(path: str | Path) -> list[SteeringCondition]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    entries = raw.get("steering_conditions")
    if not isinstance(entries, list):
        raise ValueError("Config must contain a 'steering_conditions' list")

    conditions = [condition_from_mapping(entry) for entry in entries]
    ids = [condition.id for condition in conditions]
    duplicates = sorted({condition_id for condition_id in ids if ids.count(condition_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate steering condition ids: {', '.join(duplicates)}")

    return conditions


def apply_prompt_prefix(condition: SteeringCondition, stimulus_text: str) -> str:
    return f"{condition.prompt_prefix}\n\nStimulus: {stimulus_text}"
