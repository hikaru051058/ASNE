from __future__ import annotations

import argparse
import json
from pathlib import Path

from asne.dictionary import (
    compute_neutral_baseline,
    validate_dictionary_expansion,
)
from scripts.run_asne_dictionary import _infer_neutral_category
from scripts.summarize_asne_contrasts import format_summary_table, summarize_contrasts


def test_arbitrary_baseline_category_from_metadata() -> None:
    payload = {
        "baseline_category": "consistent_information",
        "categories": {
            "consistent_information": [{"id": "c1", "text": "A person reads one matching sign."}],
            "contradictory_information": [{"id": "m1", "text": "A person reads two different signs."}],
        },
    }
    records = [
        {"category": "consistent_information", "mean_response": [1.0, 3.0]},
        {"category": "consistent_information", "mean_response": [3.0, 5.0]},
        {"category": "contradictory_information", "mean_response": [9.0, 9.0]},
    ]

    baseline_category = _infer_neutral_category(payload, requested=None)
    baseline = compute_neutral_baseline(records, neutral_category=baseline_category)

    assert baseline_category == "consistent_information"
    assert baseline.tolist() == [2.0, 4.0]


def test_no_label_leakage_validator() -> None:
    payload = {
        "categories": {
            "static_scene": [
                {"id": "s1", "text": "A person watches rain cross the window."},
                {"id": "s2", "text": "Someone reads a note beside the counter."},
            ]
        }
    }

    warnings = validate_dictionary_expansion(payload, expected_examples_per_category=2)

    assert warnings == []


def test_label_leakage_validator_flags_contrast_terms() -> None:
    payload = {
        "categories": {
            "object_missing": [{"id": "m1", "text": "The missing item is described directly."}]
        }
    }

    warnings = validate_dictionary_expansion(payload, expected_examples_per_category=1)

    assert any("category leakage term" in warning for warning in warnings)


def test_binary_eval_accuracy_summary() -> None:
    rows = summarize_contrasts(
        [
            {
                "_summary_path": "summary.json",
                "contrast_name": "fake_contrast",
                "top1_accuracy": 5 / 6,
                "top2_accuracy": 1.0,
                "correct_top1": 5,
                "total_examples": 6,
                "mean_rank_expected": 1.16,
                "confusion_matrix": {"a": {"a": 3}, "b": {"b": 2, "a": 1}},
            }
        ],
        threshold=0.75,
    )

    assert rows[0]["separable"] is True
    assert rows[0]["correct_top1"] == 5


def test_contrast_summary_generation() -> None:
    rows = [
        {
            "contrast_name": "fake_contrast",
            "top1_accuracy": 0.5,
            "top2_accuracy": 1.0,
            "correct_top1": 3,
            "total_examples": 6,
            "mean_rank_expected": 1.5,
            "separable": False,
        }
    ]

    table = format_summary_table(rows)

    assert "fake_contrast" in table
    assert "3/6 (0.50)" in table
