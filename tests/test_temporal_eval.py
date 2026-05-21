from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_temporal_eval_module():
    path = Path("scripts/evaluate_asne_temporal_contrast.py")
    spec = importlib.util.spec_from_file_location("evaluate_asne_temporal_contrast", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_segment_winner_computation() -> None:
    module = _load_temporal_eval_module()

    ranked = module.ranked_categories_for_segment([0.2, 0.9], ["a", "b"])

    assert ranked[0]["category"] == "b"
    assert ranked[0]["similarity"] == pytest.approx(0.9)


def test_early_late_split_even_and_odd_counts() -> None:
    module = _load_temporal_eval_module()

    even_early, even_late = module.split_early_late(np.zeros((4, 2)))
    odd_early, odd_late = module.split_early_late(np.zeros((5, 2)))

    assert even_early.shape == (2, 2)
    assert even_late.shape == (2, 2)
    assert odd_early.shape == (2, 2)
    assert odd_late.shape == (3, 2)


def test_majority_winner_tie_uses_category_order() -> None:
    module = _load_temporal_eval_module()

    winner = module.majority_winner(["b", "a"], ["a", "b"])

    assert winner == "a"


def test_switch_count() -> None:
    module = _load_temporal_eval_module()

    assert module.count_switches(["a", "a", "b", "a"]) == 2


def test_largest_transition_index() -> None:
    module = _load_temporal_eval_module()

    index, value = module.largest_transition(np.asarray([0.1, 2.5, 1.0]))

    assert index == 1
    assert value == pytest.approx(2.5)


def test_summary_metrics() -> None:
    module = _load_temporal_eval_module()
    examples = [
        {
            "id": "x1",
            "early_correct": False,
            "late_correct": True,
            "final_segment_correct": True,
            "majority_segment_correct": True,
            "number_of_switches": 1,
            "category_shift": True,
            "late_matches_expected_after_mean_failed": True,
            "temporal_matches_expected_after_mean_failed": True,
        },
        {
            "id": "x2",
            "early_correct": True,
            "late_correct": False,
            "final_segment_correct": False,
            "majority_segment_correct": True,
            "number_of_switches": 0,
            "category_shift": False,
            "late_matches_expected_after_mean_failed": False,
            "temporal_matches_expected_after_mean_failed": False,
        },
    ]

    summary = module.compute_temporal_summary_metrics(examples, mean_response_accuracy=0.5)

    assert summary["early_accuracy"] == pytest.approx(0.5)
    assert summary["late_accuracy"] == pytest.approx(0.5)
    assert summary["majority_segment_accuracy"] == pytest.approx(1.0)
    assert summary["average_switch_count"] == pytest.approx(0.5)
    assert summary["examples_with_category_shift"] == ["x1"]
    assert summary["examples_where_mean_failed_but_late_succeeded"] == ["x1"]
    assert summary["examples_where_mean_failed_but_temporal_succeeded"] == ["x1"]


def test_analyze_temporal_example_with_late_shift() -> None:
    module = _load_temporal_eval_module()
    raw = np.asarray(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    centroids = {
        "expected": {"vector": np.asarray([1.0, 0.0])},
        "other": {"vector": np.asarray([0.0, 1.0])},
    }

    example = module.analyze_temporal_example(
        item_id="e1",
        input_text="sample",
        expected_category="expected",
        raw_segments=raw,
        category_centroids=centroids,
        mean_response_top_category="other",
    )

    assert example["segment_winners"] == ["other", "other", "expected", "expected", "expected"]
    assert example["early_winner"] == "other"
    assert example["late_winner"] == "expected"
    assert example["final_segment_winner"] == "expected"
    assert example["majority_segment_winner"] == "expected"
    assert example["late_matches_expected_after_mean_failed"] is True
    assert example["temporal_matches_expected_after_mean_failed"] is True


def test_markdown_report_generation() -> None:
    module = _load_temporal_eval_module()
    summary = {
        "contrast_name": "fake_contrast",
        "dictionary_index_path": "dictionary_index.json",
        "eval_path": "eval.json",
        "eval_summary_path": "summary.json",
        "total_examples": 1,
        "mean_response_accuracy": 0.0,
        "early_accuracy": 0.0,
        "late_accuracy": 1.0,
        "final_segment_accuracy": 1.0,
        "majority_segment_accuracy": 1.0,
        "average_switch_count": 1.0,
        "examples": [
            {
                "id": "e1",
                "input_text": "sample",
                "expected_category": "expected",
                "mean_response_top_category": "other",
                "segment_winners": ["other", "expected"],
                "late_winner": "expected",
                "final_segment_winner": "expected",
                "majority_segment_winner": "expected",
                "number_of_switches": 1,
                "largest_transition_norm": 2.0,
            }
        ],
    }

    report = module.generate_temporal_report(summary)

    assert "ASNE Temporal Contrast Evaluation" in report
    assert "predicted stimulus-response similarity" in report
    assert "fake_contrast" in report
    assert "Segment-Level Differences" in report
