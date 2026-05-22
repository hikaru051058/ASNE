from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_asne_pair_margins import (
    average_margin_by_expected,
    enrich_margin_rows,
    load_eval_pair_map,
    margin_report,
    ranking_score,
    score_gap_top1_to_expected,
)


def test_ranking_score_uses_centroid_cosine() -> None:
    assert ranking_score({"centroid_cosine_similarity": 0.91}) == 0.91


def test_score_gap_top1_to_expected_from_ranking() -> None:
    result = {
        "expected_category": "b",
        "per_category_ranking": [
            {"category": "a", "centroid_cosine_similarity": 0.90},
            {"category": "b", "centroid_cosine_similarity": 0.85},
        ],
    }

    assert score_gap_top1_to_expected(result) == pytest.approx(0.05)


def test_load_eval_pair_map(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": "eval_a_01",
                        "pair_id": "pair_01",
                        "expected_category": "a",
                        "text": "A person reads a note. The label matches.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    pair_map = load_eval_pair_map(eval_path)

    assert pair_map["eval_a_01"]["pair_id"] == "pair_01"


def test_enrich_margin_rows_adds_pair_id_and_rank2_flag() -> None:
    summary = {
        "results": [
            {
                "id": "eval_b_01",
                "expected_category": "b",
                "top_category": "a",
                "rank_of_expected": 2,
                "per_category_ranking": [
                    {"category": "a", "centroid_cosine_similarity": 0.90},
                    {"category": "b", "centroid_cosine_similarity": 0.89},
                ],
            }
        ]
    }

    rows = enrich_margin_rows(summary, {"eval_b_01": {"pair_id": "pair_01", "text": "Text."}})

    assert rows[0]["pair_id"] == "pair_01"
    assert rows[0]["expected_rank2"] is True
    assert rows[0]["score_gap"] == pytest.approx(0.01)


def test_average_margin_by_expected_counts_failures() -> None:
    rows = [
        {"expected_category": "a", "predicted_category": "a", "score_gap": 0.0},
        {"expected_category": "a", "predicted_category": "b", "score_gap": 0.2},
    ]

    stats = average_margin_by_expected(rows)

    assert stats["a"]["failures"] == 1
    assert stats["a"]["average_gap"] == 0.1


def test_margin_report_contains_disclaimer_and_rows() -> None:
    summary = {
        "eval_name": "fake_eval",
        "correct_top1": 1,
        "total_examples": 2,
        "top1_accuracy": 0.5,
        "correct_top2": 2,
        "top2_accuracy": 1.0,
        "scoring_mode": "centroid_raw",
    }
    rows = [
        {
            "id": "eval_a_01",
            "pair_id": "pair_01",
            "expected_category": "a",
            "predicted_category": "b",
            "rank_of_expected": 2,
            "expected_rank2": True,
            "score_gap": 0.03,
            "input_text": "A person reads a note.",
        }
    ]

    report = margin_report(summary, rows)

    assert "ASNE Pair Margin Analysis" in report
    assert "not emotion detection" in report
    assert "eval_a_01" in report
