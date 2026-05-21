from __future__ import annotations

from asne.dictionary import COMPARISON_DISCLAIMER
from asne.evaluation import (
    build_confusion_matrix,
    compute_eval_summary,
    generate_failure_report,
    load_eval_set,
    top_category_from_comparison,
)


def test_eval_schema_loading() -> None:
    payload = load_eval_set("data/stimuli/evals/emotion_context_eval_v0.json")

    assert payload["name"] == "emotion_context_eval_v0"
    assert len(payload["items"]) == 10
    categories = {item["expected_category"] for item in payload["items"]}
    assert categories == {"neutral", "threat", "sadness", "confusion", "relief"}


def test_top1_and_per_category_accuracy() -> None:
    results = [
        {"id": "n1", "expected_category": "neutral", "top_category": "neutral", "is_top1_correct": True, "rank_of_expected": 1},
        {"id": "n2", "expected_category": "neutral", "top_category": "threat", "is_top1_correct": False, "rank_of_expected": 2},
        {"id": "t1", "expected_category": "threat", "top_category": "threat", "is_top1_correct": True, "rank_of_expected": 1},
    ]

    summary = compute_eval_summary(
        eval_name="fake_eval",
        results=results,
        categories=["neutral", "threat"],
        signature_mode="delta_from_neutral",
        metric_mode="both",
        dictionary_index_path="dictionary_index.json",
    )

    assert summary["total_examples"] == 3
    assert summary["correct_top1"] == 2
    assert summary["top1_accuracy"] == 2 / 3
    assert summary["correct_top2"] == 3
    assert summary["top2_accuracy"] == 1.0
    assert summary["mean_rank_expected"] == 4 / 3
    assert summary["scoring_mode"] == "full"
    assert summary["top_k"] == 1000
    assert summary["per_category_accuracy"]["neutral"]["correct"] == 1
    assert summary["per_category_accuracy"]["neutral"]["total"] == 2
    assert summary["per_category_accuracy"]["threat"]["accuracy"] == 1.0


def test_confusion_matrix() -> None:
    results = [
        {"expected_category": "neutral", "top_category": "neutral"},
        {"expected_category": "neutral", "top_category": "threat"},
        {"expected_category": "threat", "top_category": "threat"},
    ]

    matrix = build_confusion_matrix(["neutral", "threat"], results)

    assert matrix["neutral"]["neutral"] == 1
    assert matrix["neutral"]["threat"] == 1
    assert matrix["threat"]["threat"] == 1


def test_summary_json_contains_disclaimer_without_clinical_wording() -> None:
    summary = compute_eval_summary(
        eval_name="fake_eval",
        results=[
            {
                "id": "n1",
                "expected_category": "neutral",
                "top_category": "neutral",
                "is_top1_correct": True,
                "rank_of_expected": 1,
            }
        ],
        categories=["neutral"],
        signature_mode="delta_from_neutral",
        metric_mode="both",
        dictionary_index_path="dictionary_index.json",
    )

    assert summary["disclaimer"] == COMPARISON_DISCLAIMER
    lowered = summary["disclaimer"].lower()
    assert "diagnosis" in lowered
    assert "emotion detection" in lowered
    assert "clinical" not in lowered
    assert "mental-health" not in lowered


def test_failure_report_generation_without_clinical_wording() -> None:
    summary = compute_eval_summary(
        eval_name="fake_eval",
        results=[
            {
                "id": "x1",
                "input_text": "A controlled stimulus.",
                "expected_category": "confusion",
                "top_category": "sadness",
                "is_top1_correct": False,
                "rank_of_expected": 2,
                "comparison_output_path": "comparison.json",
                "query_delta_norm": 1.25,
                "per_category_centroid_norm": {"confusion": 2.0, "sadness": 3.0},
                "per_category_ranking": [
                    {
                        "category": "sadness",
                        "centroid_cosine_similarity": 0.3,
                        "centroid_pearson_correlation": 0.2,
                        "best_stimulus_id": "s1",
                    },
                    {
                        "category": "confusion",
                        "centroid_cosine_similarity": 0.2,
                        "centroid_pearson_correlation": 0.1,
                        "best_stimulus_id": "c1",
                    },
                ],
            },
            {
                "id": "x2",
                "expected_category": "relief",
                "top_category": "sadness",
                "is_top1_correct": False,
                "rank_of_expected": 3,
                "per_category_ranking": [],
            },
        ],
        categories=["confusion", "relief", "sadness"],
        signature_mode="delta_from_neutral",
        metric_mode="both",
        aggregation_mode="centroid",
        dictionary_index_path="dictionary_index.json",
    )

    report = generate_failure_report(summary)

    assert "ASNE Evaluation Failure Report" in report
    assert "One category accounts for more than half" in report
    assert "Query delta norm" in report
    lowered = report.lower()
    assert "diagnosis" in lowered
    assert "clinical" not in lowered


def test_top_category_from_comparison() -> None:
    comparison = {
        "per_category_ranking": [
            {"category": "threat", "mean_cosine_similarity": 0.2},
            {"category": "neutral", "mean_cosine_similarity": 0.0},
        ]
    }

    assert top_category_from_comparison(comparison) == "threat"
