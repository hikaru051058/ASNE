from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__


MANIFEST_SAFETY_NOTE = (
    "This manifest records a mock ASNE run for reproducibility and auditability. "
    "Mock predicted cortical responses and placeholder ROI summaries are not "
    "scientifically valid neural measurements."
)


def create_experiment_manifest(
    experiment_id: str,
    stimulus: str,
    adapter_type: str,
    steering_config_path: str | Path,
    roi_map_path: str | Path | None,
    steering_condition_ids: list[str],
    output_paths: dict[str, Any],
    timestamp_utc: str | None = None,
    package_version: str | None = None,
    safety_note: str = MANIFEST_SAFETY_NOTE,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "timestamp_utc": timestamp_utc or datetime.now(timezone.utc).isoformat(),
        "stimulus": stimulus,
        "adapter_type": adapter_type,
        "steering_config_path": str(steering_config_path),
        "roi_map_path": str(roi_map_path) if roi_map_path else None,
        "steering_condition_ids": steering_condition_ids,
        "output_paths": output_paths,
        "package_version": package_version or __version__,
        "safety_note": safety_note,
    }


def save_experiment_manifest(
    manifest: dict[str, Any],
    output_dir: str | Path = "outputs/manifests",
) -> Path:
    manifest_dir = Path(output_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{manifest['experiment_id']}.json"
    import json

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
