from __future__ import annotations

from scripts.analyze_asne_eval_failures import (
    dominant_attractor_categories,
    generate_diagnostic_report,
    score_gap_top1_to_expected,
    top2_high_top1_low,
)
from scripts.remap_asne_eval_categories import compute_grouped_accuracy


def _ranking(*categories: tuple[str, float]) -> list[dict[str, object]]:
    return [
        {"category": category, "score": score, "best_stimulus_id": f"{category}_1"}
        for category, score in categories
    ]


def test_score_gap_computation() -> None:
    result = {
        "expected_category": "threat",
        "per_category_ranking": _ranking(("relief", 0.42), ("threat", 0.30)),
    }

    assert score_gap_top1_to_expected(result) == 0.12


def test_top2_high_top1_low_flag() -> None:
    assert top2_high_top1_low({"top1_accuracy": 0.33, "top2_accuracy": 0.87})
    assert not top2_high_top1_low({"top1_accuracy": 0.70, "top2_accuracy": 0.80})


def test_attractor_category_detection() -> None:
    results = [
        {"is_top1_correct": False, "top_category": "confusion"},
        {"is_top1_correct": False, "top_category": "confusion"},
        {"is_top1_correct": False, "top_category": "relief"},
        {"is_top1_correct": True, "top_category": "neutral"},
    ]

    attractors = dominant_attractor_categories(results)

    assert attractors[0]["category"] == "confusion"
    assert attractors[0]["count"] == 2


def test_diagnostic_report_generation() -> None:
    summary = {
        "eval_name": "fake_eval",
        "total_examples": 3,
        "correct_top1": 1,
        "correct_top2": 3,
        "top1_accuracy": 1 / 3,
        "top2_accuracy": 1.0,
        "mean_rank_expected": 1.67,
        "signature_mode": "delta_from_neutral",
        "aggregation_mode": "centroid",
        "scoring_mode": "full",
        "confusion_matrix": {
            "threat": {"threat": 1, "sadness": 0},
            "sadness": {"threat": 1, "sadness": 1},
        },
        "results": [
            {
                "id": "s1",
                "input_text": "A person folds a letter and leaves the room.",
                "expected_category": "sadness",
                "top_category": "threat",
                "is_top1_correct": False,
                "rank_of_expected": 2,
                "per_category_ranking": _ranking(("threat", 0.4), ("sadness", 0.3)),
            },
            {
                "id": "s2",
                "expected_category": "threat",
                "top_category": "threat",
                "is_top1_correct": True,
                "rank_of_expected": 1,
                "per_category_ranking": _ranking(("threat", 0.5), ("sadness", 0.1)),
            },
        ],
    }

    report = generate_diagnostic_report(summary)

    assert "ASNE Evaluation Diagnostics" in report
    assert "Score gap top1-minus-expected: 0.1000" in report
    assert "Top-2 accuracy is high while top-1 accuracy is low" in report
    assert "diagnosis" in report.lower()
    assert "clinical" not in report.lower()


def test_grouped_category_accuracy() -> None:
    summary = {
        "eval_name": "fake_eval",
        "results": [
            {
                "id": "t1",
                "expected_category": "threat",
                "top_category": "sadness",
                "per_category_ranking": _ranking(("sadness", 0.4), ("threat", 0.3)),
            },
            {
                "id": "n1",
                "expected_category": "neutral",
                "top_category": "confusion",
                "per_category_ranking": _ranking(("confusion", 0.4), ("neutral", 0.3)),
            },
        ],
    }

    grouped = compute_grouped_accuracy(summary)

    assert grouped["correct_grouped_top1"] == 1
    assert grouped["grouped_top1_accuracy"] == 0.5
    assert grouped["per_group_accuracy"]["negative_context"]["accuracy"] == 1.0
    assert grouped["per_group_accuracy"]["ordinary"]["accuracy"] == 0.0
    assert "diagnosis" in grouped["disclaimer"].lower()
