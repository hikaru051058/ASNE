from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_roi_map(path: str | Path) -> dict[str, Any]:
    roi_path = Path(path)
    raw = yaml.safe_load(roi_path.read_text(encoding="utf-8")) or {}
    config = raw.get("mock_roi_map")
    if not isinstance(config, dict):
        raise ValueError("ROI config must contain a 'mock_roi_map' mapping")

    rois = config.get("rois")
    if not isinstance(rois, list) or not rois:
        raise ValueError("ROI config must contain a non-empty 'rois' list")

    by_index: dict[int, str] = {}
    by_roi: dict[str, list[int]] = {}
    labels: dict[str, str] = {}

    for roi in rois:
        roi_id = str(roi.get("id", ""))
        if not roi_id:
            raise ValueError("Each ROI entry must include an id")
        start = int(roi["index_start"])
        end = int(roi["index_end"])
        if end < start:
            raise ValueError(f"ROI {roi_id} has index_end before index_start")

        indices = list(range(start, end + 1))
        for index in indices:
            if index in by_index:
                raise ValueError(f"Index {index} appears in more than one ROI")
            by_index[index] = roi_id

        by_roi[roi_id] = indices
        labels[roi_id] = str(roi.get("name", roi_id))

    response_size = int(config.get("response_size", max(by_index) + 1))
    missing = sorted(set(range(response_size)) - set(by_index))
    if missing:
        raise ValueError(f"ROI map does not cover response indices: {missing}")

    return {
        "response_size": response_size,
        "description": str(config.get("description", "")),
        "by_index": by_index,
        "by_roi": by_roi,
        "labels": labels,
    }


def index_to_roi(index: int, roi_map: dict[str, Any]) -> str:
    try:
        return roi_map["by_index"][int(index)]
    except KeyError as exc:
        raise KeyError(f"Index {index} is not present in ROI map") from exc


def summarize_delta_by_roi(delta: np.ndarray, roi_map: dict[str, Any]) -> dict[str, dict[str, Any]]:
    delta_array = np.asarray(delta, dtype=float).reshape(-1)
    if delta_array.size != roi_map["response_size"]:
        raise ValueError(
            f"Delta size {delta_array.size} does not match ROI map size {roi_map['response_size']}"
        )

    summaries: dict[str, dict[str, Any]] = {}
    for roi_id, indices in roi_map["by_roi"].items():
        values = delta_array[indices]
        abs_values = np.abs(values)
        summaries[roi_id] = {
            "mean_abs_delta": float(np.mean(abs_values)),
            "max_abs_delta": float(np.max(abs_values)),
            "n_indices": len(indices),
        }
    return summaries
