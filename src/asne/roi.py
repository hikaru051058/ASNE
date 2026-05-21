from __future__ import annotations

from pathlib import Path
import csv
import json
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


def load_parcellation(path: str | Path) -> dict[str, Any]:
    parcellation_path = Path(path)
    suffix = parcellation_path.suffix.lower()
    if suffix == ".csv":
        vertices = _load_parcellation_csv(parcellation_path)
    elif suffix == ".json":
        vertices = _load_parcellation_json(parcellation_path)
    else:
        raise ValueError(f"Unsupported parcellation format: {parcellation_path.suffix}")
    return _build_parcellation(vertices, source_path=str(parcellation_path))


def validate_parcellation(parcellation: dict[str, Any], expected_vertices: int = 20484) -> None:
    by_index = parcellation.get("by_index")
    if not isinstance(by_index, dict):
        raise ValueError("Parcellation must include a by_index mapping.")
    indices = sorted(int(index) for index in by_index)
    expected = list(range(expected_vertices))
    if indices != expected:
        missing = sorted(set(expected) - set(indices))
        extra = sorted(set(indices) - set(expected))
        details = []
        if missing:
            details.append(f"missing vertex indices: {missing[:10]}")
        if extra:
            details.append(f"unexpected vertex indices: {extra[:10]}")
        raise ValueError("; ".join(details) or "Parcellation vertex indices are not contiguous.")
    for index, item in by_index.items():
        if not str(item.get("parcel_id", "")).strip():
            raise ValueError(f"Vertex {index} is missing parcel_id.")
        if not str(item.get("parcel_name", "")).strip():
            raise ValueError(f"Vertex {index} is missing parcel_name.")


def aggregate_vertices_to_parcels(
    response_vector: Any,
    parcel_labels: dict[str, Any],
    method: str = "mean",
) -> dict[str, float]:
    vector = np.asarray(response_vector, dtype=float).reshape(-1)
    if vector.size != int(parcel_labels["vertex_count"]):
        raise ValueError(f"Response vector has {vector.size} vertices; parcellation expects {parcel_labels['vertex_count']}.")
    aggregated = {}
    for parcel_id, indices in parcel_labels["by_parcel"].items():
        values = vector[np.asarray(indices, dtype=int)]
        aggregated[parcel_id] = _aggregate(values, method=method)
    return aggregated


def aggregate_segments_to_parcels(
    raw_segments: Any,
    parcel_labels: dict[str, Any],
    method: str = "mean",
) -> dict[str, Any]:
    segments = np.asarray(raw_segments, dtype=float)
    if segments.ndim == 1:
        segments = segments.reshape(1, -1)
    if segments.ndim > 2:
        segments = segments.reshape(segments.shape[0], -1)
    if segments.shape[1] != int(parcel_labels["vertex_count"]):
        raise ValueError(f"Raw segments have {segments.shape[1]} vertices; parcellation expects {parcel_labels['vertex_count']}.")
    parcel_ids = list(parcel_labels["by_parcel"])
    values = []
    for segment in segments:
        aggregated = aggregate_vertices_to_parcels(segment, parcel_labels, method=method)
        values.append([aggregated[parcel_id] for parcel_id in parcel_ids])
    return {
        "parcel_ids": parcel_ids,
        "parcel_names": [parcel_labels["parcel_names"][parcel_id] for parcel_id in parcel_ids],
        "values": np.asarray(values, dtype=float),
    }


def compute_parcel_delta(category_a_parcels: dict[str, float], category_b_parcels: dict[str, float]) -> dict[str, float]:
    if set(category_a_parcels) != set(category_b_parcels):
        missing_a = sorted(set(category_b_parcels) - set(category_a_parcels))
        missing_b = sorted(set(category_a_parcels) - set(category_b_parcels))
        raise ValueError(f"Parcel sets differ; missing from A={missing_a}, missing from B={missing_b}")
    return {
        parcel_id: float(category_b_parcels[parcel_id] - category_a_parcels[parcel_id])
        for parcel_id in category_a_parcels
    }


def rank_parcels_by_abs_delta(parcel_delta: dict[str, float], top_k: int = 20) -> list[dict[str, Any]]:
    ranked = sorted(parcel_delta.items(), key=lambda item: abs(float(item[1])), reverse=True)
    return [
        {"parcel_id": parcel_id, "delta": float(delta), "abs_delta": abs(float(delta))}
        for parcel_id, delta in ranked[:top_k]
    ]


def _aggregate(values: np.ndarray, method: str = "mean") -> float:
    if method == "mean":
        return float(np.mean(values))
    if method == "median":
        return float(np.median(values))
    if method == "max_abs":
        return float(values[np.argmax(np.abs(values))])
    raise ValueError(f"Unsupported parcel aggregation method: {method}")


def _load_parcellation_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"vertex_index", "parcel_id", "parcel_name"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError("CSV parcellation must include columns: vertex_index, parcel_id, parcel_name")
        return [
            {
                "vertex_index": int(row["vertex_index"]),
                "parcel_id": str(row["parcel_id"]),
                "parcel_name": str(row["parcel_name"]),
            }
            for row in reader
        ]


def _load_parcellation_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if isinstance(payload.get("vertices"), list):
            payload = payload["vertices"]
        else:
            payload = [
                {"vertex_index": int(index), **value}
                for index, value in payload.items()
            ]
    if not isinstance(payload, list):
        raise ValueError("JSON parcellation must be a list or mapping of vertex entries.")
    vertices = []
    for item in payload:
        vertices.append(
            {
                "vertex_index": int(item["vertex_index"]),
                "parcel_id": str(item["parcel_id"]),
                "parcel_name": str(item.get("parcel_name", item["parcel_id"])),
            }
        )
    return vertices


def _build_parcellation(vertices: list[dict[str, Any]], source_path: str) -> dict[str, Any]:
    by_index: dict[int, dict[str, str]] = {}
    by_parcel: dict[str, list[int]] = {}
    parcel_names: dict[str, str] = {}
    for vertex in vertices:
        index = int(vertex["vertex_index"])
        if index in by_index:
            raise ValueError(f"Duplicate vertex index in parcellation: {index}")
        parcel_id = str(vertex["parcel_id"])
        parcel_name = str(vertex.get("parcel_name", parcel_id))
        by_index[index] = {"parcel_id": parcel_id, "parcel_name": parcel_name}
        by_parcel.setdefault(parcel_id, []).append(index)
        parcel_names.setdefault(parcel_id, parcel_name)
    for indices in by_parcel.values():
        indices.sort()
    return {
        "source_path": source_path,
        "vertex_count": len(by_index),
        "parcel_count": len(by_parcel),
        "by_index": by_index,
        "by_parcel": by_parcel,
        "parcel_names": parcel_names,
    }
