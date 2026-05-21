from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from asne.dictionary import (
    build_signature_records,
    compute_category_centroids,
    compute_temporal_movement_signatures,
    load_dictionary_records,
    top_abs_indices,
)


def discover_contrast_summaries(root: str | Path = "outputs/asne_evals/contrasts") -> list[Path]:
    """Find ASNE contrast evaluation summary JSON files."""

    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(root_path.glob("**/*_summary.json"))


def discover_temporal_summaries(root: str | Path = "outputs/asne_temporal_evals") -> list[Path]:
    """Find ASNE temporal contrast evaluation summary JSON files."""

    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(root_path.glob("**/*_temporal_summary.json"))


def load_summary(path: str | Path) -> dict[str, Any]:
    summary_path = Path(path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["_summary_path"] = str(summary_path)
    return summary


def infer_failure_report_path(summary_path: str | Path) -> str | None:
    path = Path(summary_path)
    failure_path = path.with_name(path.name.replace("_summary.json", "_failure_report.md"))
    return str(failure_path) if failure_path.exists() else None


def summarize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    summary_path = summary.get("_summary_path") or summary.get("summary_path") or ""
    timestamp = Path(summary_path).name.replace("_summary.json", "") if summary_path else summary.get("created_at")
    contrast_name = (
        summary.get("contrast_name")
        or summary.get("eval_name")
        or Path(summary_path).parent.name
        or "unknown"
    )
    total = int(summary.get("total_examples") or 0)
    correct = int(summary.get("correct_top1") or 0)
    return {
        "contrast_name": contrast_name,
        "timestamp": timestamp,
        "scoring_mode": summary.get("scoring_mode"),
        "signature_mode": summary.get("signature_mode"),
        "aggregation_mode": summary.get("aggregation_mode"),
        "top1_correct": correct,
        "total": total,
        "top1_accuracy": float(summary.get("top1_accuracy") or 0.0),
        "top2_accuracy": float(summary.get("top2_accuracy") or 0.0),
        "mean_rank_expected": summary.get("mean_rank_expected"),
        "per_category_accuracy": summary.get("per_category_accuracy", {}),
        "summary_path": summary_path,
        "failure_report_path": infer_failure_report_path(summary_path) if summary_path else None,
    }


def load_summary_rows(root: str | Path = "outputs/asne_evals/contrasts") -> list[dict[str, Any]]:
    rows = []
    for path in discover_contrast_summaries(root):
        rows.append(summarize_summary(load_summary(path)))
    return rows


def summarize_temporal_summary(summary: dict[str, Any]) -> dict[str, Any]:
    summary_path = summary.get("_summary_path") or summary.get("summary_path") or ""
    timestamp = Path(summary_path).name.replace("_temporal_summary.json", "") if summary_path else summary.get("created_at")
    return {
        "contrast_name": summary.get("contrast_name") or Path(summary_path).parent.name or "unknown",
        "timestamp": timestamp,
        "total_examples": int(summary.get("total_examples") or 0),
        "mean_response_accuracy": summary.get("mean_response_accuracy"),
        "early_accuracy": float(summary.get("early_accuracy") or 0.0),
        "late_accuracy": float(summary.get("late_accuracy") or 0.0),
        "final_segment_accuracy": float(summary.get("final_segment_accuracy") or 0.0),
        "majority_segment_accuracy": float(summary.get("majority_segment_accuracy") or 0.0),
        "average_switch_count": float(summary.get("average_switch_count") or 0.0),
        "examples_with_category_shift": summary.get("examples_with_category_shift", []),
        "examples_where_mean_failed_but_late_succeeded": summary.get("examples_where_mean_failed_but_late_succeeded", []),
        "examples_where_mean_failed_but_temporal_succeeded": summary.get("examples_where_mean_failed_but_temporal_succeeded", []),
        "summary_path": summary_path,
    }


def load_temporal_summary_rows(root: str | Path = "outputs/asne_temporal_evals") -> list[dict[str, Any]]:
    rows = []
    for path in discover_temporal_summaries(root):
        payload = load_summary(path)
        rows.append(summarize_temporal_summary(payload))
    return rows


def summaries_to_dataframe(rows: list[dict[str, Any]]):
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(
        ["top1_accuracy", "top2_accuracy", "mean_rank_expected"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def best_methods_by_contrast(rows: list[dict[str, Any]]):
    df = summaries_to_dataframe(rows)
    if df.empty:
        return df
    return (
        df.sort_values(
            ["contrast_name", "top1_accuracy", "top2_accuracy", "mean_rank_expected"],
            ascending=[True, False, False, True],
        )
        .groupby("contrast_name", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def compact_contrast_table(rows: list[dict[str, Any]]):
    df = best_methods_by_contrast(rows)
    if df.empty:
        return df
    compact = df.copy()
    compact["score"] = compact.apply(
        lambda row: f"{int(row['top1_correct'])}/{int(row['total'])} ({row['top1_accuracy']:.2f})",
        axis=1,
    )
    return compact[
        [
            "contrast_name",
            "scoring_mode",
            "score",
            "top2_accuracy",
            "mean_rank_expected",
            "timestamp",
        ]
    ]


def executive_summary_lines(rows: list[dict[str, Any]]) -> list[str]:
    best = best_methods_by_contrast(rows)
    if best.empty:
        return ["No contrast evaluation summaries were found."]

    lines = [
        "ASNE Contrast Executive Summary",
        "",
        "Best available result per contrast:",
    ]
    for row in best.itertuples(index=False):
        lines.append(
            f"- {row.contrast_name}: {int(row.top1_correct)}/{int(row.total)} "
            f"top-1 = {row.top1_accuracy:.2f} using {row.scoring_mode}"
        )

    centroid = best[best["scoring_mode"] == "centroid_raw"]
    if not centroid.empty:
        lines.extend(
            [
                "",
                "Current practical setting: signature=mean_response, aggregation=centroid, scoring=centroid_raw.",
            ]
        )

    weak = best[best["top1_accuracy"] <= 0.5]
    if not weak.empty:
        lines.append(
            "Weak or unresolved axes: "
            + ", ".join(str(value) for value in weak["contrast_name"].tolist())
            + "."
        )

    strong = best[best["top1_accuracy"] >= 0.8]
    if not strong.empty:
        lines.append(
            "Promising axes: "
            + ", ".join(str(value) for value in strong["contrast_name"].tolist())
            + "."
        )

    lines.extend(
        [
            "",
            "Interpretation: these are predicted response-similarity results, not emotion detection, diagnosis, or measurement of a person's mental state.",
        ]
    )
    return lines


def score_gap_from_ranking(ranking: list[dict[str, Any]], expected_category: str | None) -> float | None:
    if not ranking or not expected_category:
        return None
    top_score = _ranking_score(ranking[0])
    expected_score = None
    for item in ranking:
        if item.get("category") == expected_category:
            expected_score = _ranking_score(item)
            break
    if top_score is None or expected_score is None:
        return None
    return float(top_score - expected_score)


def extract_failures(summary: dict[str, Any]) -> list[dict[str, Any]]:
    failures = []
    for item in summary.get("failed_examples", []):
        ranking = item.get("per_category_ranking") or []
        failures.append(
            {
                "id": item.get("id"),
                "input_text": item.get("input_text"),
                "expected_category": item.get("expected_category"),
                "predicted_category": item.get("top_category"),
                "rank_of_expected": item.get("rank_of_expected"),
                "score_gap": score_gap_from_ranking(ranking, item.get("expected_category")),
                "ranking": ranking,
                "comparison_output_path": item.get("comparison_output_path"),
            }
        )
    return failures


def _ranking_score(item: dict[str, Any]) -> float | None:
    for key in (
        "score",
        "centroid_cosine_similarity",
        "best_cosine_similarity",
        "mean_cosine_similarity",
        "vote_count",
    ):
        if item.get(key) is not None:
            return float(item[key])
    return None


def plot_contrast_scores(df, metric: str = "top1_accuracy", title: str | None = None):
    import matplotlib.pyplot as plt

    if df.empty:
        print("No summary rows available to plot.")
        return None
    if metric not in df.columns:
        raise ValueError(f"Unknown metric column: {metric}")
    grouped = (
        df.sort_values(["contrast_name", metric], ascending=[True, False])
        .groupby("contrast_name", as_index=False)
        .head(1)
        .sort_values(metric, ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(grouped["contrast_name"], grouped[metric])
    ax.set_ylabel(metric)
    ax.set_xlabel("contrast_name")
    ax.set_title(title or f"{metric} by contrast")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    return fig


def load_raw_segments(stimulus_json_or_path: str | Path | dict[str, Any]) -> np.ndarray:
    """Load raw segment predictions from a stimulus JSON, record dict, or .npy path."""

    if isinstance(stimulus_json_or_path, dict):
        payload = stimulus_json_or_path
        raw_path = _raw_segment_path_from_payload(payload)
        if not raw_path:
            raise ValueError("Stimulus record does not include a raw segment prediction path.")
        return _normalize_raw_segments(np.load(raw_path))

    path = Path(stimulus_json_or_path)
    if path.suffix == ".npy":
        return _normalize_raw_segments(np.load(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_path = _raw_segment_path_from_payload(payload)
    if not raw_path:
        raise ValueError(f"Stimulus JSON does not include a raw segment prediction path: {path}")
    return _normalize_raw_segments(np.load(raw_path))


def _raw_segment_path_from_payload(payload: dict[str, Any]) -> str | None:
    return (
        payload.get("raw_segment_prediction_path")
        or payload.get("raw_segments_path")
        or payload.get("query_metadata", {}).get("raw_prediction_path")
    )


def compute_response_norm_timeline(raw_segments: Any) -> np.ndarray:
    raw = _normalize_raw_segments(raw_segments)
    return np.linalg.norm(raw, axis=1)


def compute_transition_norm_timeline(raw_segments: Any) -> np.ndarray:
    raw = _normalize_raw_segments(raw_segments)
    if raw.shape[0] <= 1:
        return np.asarray([], dtype=float)
    return np.linalg.norm(np.diff(raw, axis=0), axis=1)


def compute_segment_similarity_timeline(
    raw_segments: Any,
    category_centroids: dict[str, Any],
    metric: str = "cosine",
) -> dict[str, Any]:
    raw = _normalize_raw_segments(raw_segments)
    categories = list(category_centroids)
    similarities = []
    for segment in raw:
        row = []
        for category in categories:
            centroid = category_centroids[category]
            if isinstance(centroid, dict):
                centroid = centroid.get("vector")
            row.append(_similarity(segment, np.asarray(centroid, dtype=float), metric=metric))
        similarities.append(row)
    return {
        "metric": metric,
        "segments": list(range(raw.shape[0])),
        "categories": categories,
        "similarities": similarities,
    }


def get_top_moving_dimensions(raw_segments: Any, k: int = 20) -> dict[str, list[dict[str, float | int]]]:
    movement = compute_temporal_movement_signatures(raw_segments)
    return {
        "net_movement": top_abs_indices(movement["net_movement"], k=k),
        "abs_total_movement": top_abs_indices(movement["abs_total_movement"], k=k),
    }


def load_mean_response_category_centroids(dictionary_index: str | Path) -> dict[str, dict[str, Any]]:
    records = load_dictionary_records(dictionary_index)
    return compute_category_centroids(build_signature_records(records, signature="mean_response"))


def temporal_report_payload(
    stimulus_output: str | Path,
    dictionary_index: str | Path,
    metric: str = "cosine",
    top_k: int = 20,
) -> dict[str, Any]:
    raw = load_raw_segments(stimulus_output)
    centroids = load_mean_response_category_centroids(dictionary_index)
    response_norm = compute_response_norm_timeline(raw)
    transition_norm = compute_transition_norm_timeline(raw)
    similarity = compute_segment_similarity_timeline(raw, centroids, metric=metric)
    top_dims = get_top_moving_dimensions(raw, k=top_k)
    top_categories = _top_categories_by_segment(similarity)
    late_shift = bool(top_categories and top_categories[0] != top_categories[-1])
    return {
        "stimulus_output": str(stimulus_output),
        "dictionary_index": str(dictionary_index),
        "raw_shape": list(raw.shape),
        "segment_count": int(raw.shape[0]),
        "response_norm_timeline": response_norm.tolist(),
        "transition_norm_timeline": transition_norm.tolist(),
        "segment_similarity_timeline": similarity,
        "top_categories_by_segment": top_categories,
        "late_shift_detected": late_shift,
        "top_moving_dimensions": top_dims,
    }


def _top_categories_by_segment(similarity: dict[str, Any]) -> list[str | None]:
    categories = similarity.get("categories", [])
    rows = similarity.get("similarities", [])
    top = []
    for row in rows:
        if not row or not categories:
            top.append(None)
            continue
        top.append(categories[int(np.argmax(row))])
    return top


def _normalize_raw_segments(raw_segments: Any) -> np.ndarray:
    raw = np.asarray(raw_segments, dtype=float)
    if raw.ndim == 0:
        return raw.reshape(1, 1)
    if raw.ndim == 1:
        return raw.reshape(1, -1)
    if raw.ndim > 2:
        return raw.reshape(raw.shape[0], -1)
    return raw


def _similarity(a: np.ndarray, b: np.ndarray, metric: str = "cosine") -> float:
    if a.shape != b.shape:
        raise ValueError(f"Cannot compare vectors with shapes {a.shape} and {b.shape}.")
    if metric == "cosine":
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return 0.0 if denom == 0.0 else float(np.dot(a, b) / denom)
    if metric == "pearson":
        a_centered = a - np.mean(a)
        b_centered = b - np.mean(b)
        denom = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
        return 0.0 if denom == 0.0 else float(np.dot(a_centered, b_centered) / denom)
    raise ValueError(f"Unsupported similarity metric: {metric}")
