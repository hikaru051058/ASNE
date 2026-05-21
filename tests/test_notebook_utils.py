from __future__ import annotations

import json
from pathlib import Path

import pytest
import numpy as np

from asne.notebook_utils import (
    best_methods_by_contrast,
    compact_contrast_table,
    compute_response_norm_timeline,
    compute_segment_similarity_timeline,
    compute_transition_norm_timeline,
    discover_temporal_summaries,
    discover_contrast_summaries,
    executive_summary_lines,
    extract_failures,
    get_top_moving_dimensions,
    load_summary,
    load_temporal_summary_rows,
    summarize_summary,
    summaries_to_dataframe,
)


def _write_summary(path: Path, *, contrast: str, scoring: str, top1: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "contrast_name": contrast,
                "scoring_mode": scoring,
                "signature_mode": "mean_response",
                "aggregation_mode": "centroid",
                "total_examples": 2,
                "correct_top1": int(top1 * 2),
                "top1_accuracy": top1,
                "top2_accuracy": 1.0,
                "mean_rank_expected": 1.0 if top1 == 1.0 else 1.5,
                "per_category_accuracy": {"a": {"correct": 1, "total": 1}},
                "failed_examples": [],
            }
        ),
        encoding="utf-8",
    )


def test_discover_contrast_summaries(tmp_path: Path) -> None:
    _write_summary(tmp_path / "contrast_a" / "20260101T000000Z_summary.json", contrast="a", scoring="full", top1=1.0)
    (tmp_path / "contrast_a" / "notes.txt").write_text("ignore", encoding="utf-8")

    paths = discover_contrast_summaries(tmp_path)

    assert len(paths) == 1
    assert paths[0].name.endswith("_summary.json")


def test_discover_temporal_summaries(tmp_path: Path) -> None:
    path = tmp_path / "contrast_a" / "20260101T000000Z_temporal_summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"contrast_name": "contrast_a"}), encoding="utf-8")

    paths = discover_temporal_summaries(tmp_path)

    assert paths == [path]


def test_load_temporal_summary_rows(tmp_path: Path) -> None:
    path = tmp_path / "contrast_a" / "20260101T000000Z_temporal_summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "contrast_name": "contrast_a",
                "total_examples": 2,
                "mean_response_accuracy": 0.5,
                "early_accuracy": 0.5,
                "late_accuracy": 1.0,
                "final_segment_accuracy": 1.0,
                "majority_segment_accuracy": 0.5,
                "average_switch_count": 1.5,
                "examples_with_category_shift": ["a"],
                "examples_where_mean_failed_but_temporal_succeeded": ["b"],
            }
        ),
        encoding="utf-8",
    )

    rows = load_temporal_summary_rows(tmp_path)

    assert rows[0]["contrast_name"] == "contrast_a"
    assert rows[0]["late_accuracy"] == 1.0
    assert rows[0]["examples_with_category_shift"] == ["a"]
    assert rows[0]["examples_where_mean_failed_but_temporal_succeeded"] == ["b"]


def test_summary_row_extraction_from_fake_summary(tmp_path: Path) -> None:
    path = tmp_path / "contrast_a" / "20260101T000000Z_summary.json"
    _write_summary(path, contrast="contrast_a", scoring="centroid_raw", top1=1.0)
    (path.parent / "20260101T000000Z_failure_report.md").write_text("# Failures\n", encoding="utf-8")

    row = summarize_summary(load_summary(path))

    assert row["contrast_name"] == "contrast_a"
    assert row["timestamp"] == "20260101T000000Z"
    assert row["scoring_mode"] == "centroid_raw"
    assert row["top1_accuracy"] == 1.0
    assert row["failure_report_path"].endswith("_failure_report.md")


def test_best_method_grouping() -> None:
    rows = [
        {"contrast_name": "a", "scoring_mode": "full", "top1_accuracy": 0.5, "top2_accuracy": 1.0, "mean_rank_expected": 1.5},
        {"contrast_name": "a", "scoring_mode": "centroid_raw", "top1_accuracy": 1.0, "top2_accuracy": 1.0, "mean_rank_expected": 1.0},
        {"contrast_name": "b", "scoring_mode": "paired_vote", "top1_accuracy": 0.5, "top2_accuracy": 1.0, "mean_rank_expected": 1.5},
    ]

    best = best_methods_by_contrast(rows)

    assert list(best["contrast_name"]) == ["a", "b"]
    assert best.loc[best["contrast_name"] == "a", "scoring_mode"].iloc[0] == "centroid_raw"


def test_compact_contrast_table() -> None:
    rows = [
        {"contrast_name": "semantic", "scoring_mode": "centroid_raw", "top1_correct": 5, "total": 6, "top1_accuracy": 5 / 6, "top2_accuracy": 1.0, "mean_rank_expected": 1.17, "timestamp": "t1"},
    ]

    compact = compact_contrast_table(rows)

    assert list(compact.columns) == [
        "contrast_name",
        "scoring_mode",
        "score",
        "top2_accuracy",
        "mean_rank_expected",
        "timestamp",
    ]
    assert compact.iloc[0]["score"] == "5/6 (0.83)"


def test_executive_summary_lines() -> None:
    rows = [
        {"contrast_name": "semantic", "scoring_mode": "centroid_raw", "top1_correct": 5, "total": 6, "top1_accuracy": 0.83, "top2_accuracy": 1.0, "mean_rank_expected": 1.17},
        {"contrast_name": "motion", "scoring_mode": "centroid_raw", "top1_correct": 3, "total": 6, "top1_accuracy": 0.5, "top2_accuracy": 1.0, "mean_rank_expected": 1.5},
    ]

    lines = executive_summary_lines(rows)

    assert any("semantic: 5/6" in line for line in lines)
    assert any("Weak or unresolved axes: motion" in line for line in lines)
    assert any("predicted response-similarity" in line for line in lines)


def test_failure_extraction_with_score_gap() -> None:
    summary = {
        "failed_examples": [
            {
                "id": "x1",
                "input_text": "sample",
                "expected_category": "a",
                "top_category": "b",
                "rank_of_expected": 2,
                "per_category_ranking": [
                    {"category": "b", "score": 0.7},
                    {"category": "a", "score": 0.4},
                ],
            }
        ]
    }

    failures = extract_failures(summary)

    assert failures[0]["predicted_category"] == "b"
    assert failures[0]["score_gap"] == pytest.approx(0.3)


def test_summaries_to_dataframe_empty() -> None:
    df = summaries_to_dataframe([])

    assert df.empty


def test_response_norm_timeline_shape() -> None:
    raw = np.asarray([[3.0, 4.0], [0.0, 2.0], [1.0, 2.0]])

    timeline = compute_response_norm_timeline(raw)

    assert timeline.shape == (3,)
    assert timeline[0] == pytest.approx(5.0)


def test_transition_norm_timeline_shape() -> None:
    raw = np.asarray([[0.0, 0.0], [3.0, 4.0], [3.0, 8.0]])

    timeline = compute_transition_norm_timeline(raw)

    assert timeline.shape == (2,)
    assert timeline.tolist() == pytest.approx([5.0, 4.0])


def test_transition_norm_timeline_handles_one_segment() -> None:
    timeline = compute_transition_norm_timeline(np.asarray([[1.0, 2.0]]))

    assert timeline.shape == (0,)


def test_top_moving_dimensions_ranking() -> None:
    raw = np.asarray(
        [
            [0.0, 0.0, 5.0],
            [1.0, 4.0, 0.0],
            [2.0, 8.0, 5.0],
        ]
    )

    top = get_top_moving_dimensions(raw, k=2)

    assert top["net_movement"][0]["index"] == 1
    assert top["net_movement"][0]["value"] == pytest.approx(8.0)
    assert top["abs_total_movement"][0]["index"] == 2
    assert top["abs_total_movement"][0]["value"] == pytest.approx(10.0)


def test_segment_similarity_timeline_shape() -> None:
    raw = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    centroids = {
        "a": {"vector": np.asarray([1.0, 0.0])},
        "b": {"vector": np.asarray([0.0, 1.0])},
    }

    timeline = compute_segment_similarity_timeline(raw, centroids)

    assert timeline["segments"] == [0, 1]
    assert timeline["categories"] == ["a", "b"]
    assert len(timeline["similarities"]) == 2
    assert len(timeline["similarities"][0]) == 2
    assert timeline["similarities"][0][0] == pytest.approx(1.0)
