from __future__ import annotations

from typing import Any

import numpy as np


def _response_array(prediction: dict[str, Any]) -> np.ndarray:
    if "response" not in prediction:
        raise KeyError("Prediction dictionary must contain a 'response' field")
    return np.asarray(prediction["response"], dtype=float)


def compute_delta(baseline: dict[str, Any], steered: dict[str, Any]) -> np.ndarray:
    baseline_response = _response_array(baseline)
    steered_response = _response_array(steered)
    if baseline_response.shape != steered_response.shape:
        raise ValueError(
            f"Response shapes must match, got {baseline_response.shape} and {steered_response.shape}"
        )
    return steered_response - baseline_response


def summarize_delta(delta: np.ndarray, top_k: int = 10) -> dict[str, Any]:
    delta_array = np.asarray(delta, dtype=float)
    if delta_array.ndim != 1:
        delta_array = delta_array.reshape(-1)

    abs_delta = np.abs(delta_array)
    count = min(top_k, abs_delta.size)
    top_indices = np.argsort(abs_delta)[-count:][::-1]

    return {
        "mean_abs_delta": float(np.mean(abs_delta)),
        "max_abs_delta": float(np.max(abs_delta)),
        "top_indices": top_indices.astype(int).tolist(),
        "top_values": delta_array[top_indices].astype(float).tolist(),
    }
