from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from asne.dictionary import load_stimulus_dictionary, validate_paired_contrast_benchmark
from asne.evaluation import load_eval_set


def test_v03_benchmark_files_validate() -> None:
    module = _load_script()

    warnings = module.validate_v03_inputs()

    assert warnings == []
    assert len(module.V03_CONTRASTS) == 4


def test_v03_dictionary_and_eval_counts() -> None:
    for dictionary_path in Path("data/stimuli/dictionaries/contrasts_v03").glob("*_paired.json"):
        payload = load_stimulus_dictionary(dictionary_path)
        assert payload["benchmark_name"] == "semantic_contrast_v0_3_lite"
        assert len(payload["categories"]) == 2
        assert all(len(items) == 10 for items in payload["categories"].values())

    for eval_path in Path("data/stimuli/evals/contrasts_v03").glob("*_paired_eval.json"):
        payload = load_eval_set(eval_path)
        assert payload["benchmark_name"] == "semantic_contrast_v0_3_lite"
        assert len(payload["items"]) == 10


def test_paired_validator_flags_missing_pair_id() -> None:
    dictionary = {
        "categories": {
            "a": [{"id": "a1", "text": "A person reads a card. The card says five."}],
            "b": [{"id": "b1", "pair_id": "p1", "text": "A person reads a card. The card says six."}],
        }
    }

    warnings = validate_paired_contrast_benchmark(dictionary, expected_dictionary_pairs=1)

    assert any("missing pair_id" in warning for warning in warnings)


def test_v03_report_format_includes_confidence_warning() -> None:
    module = _load_script()

    report = module.format_report(
        [
            {
                "contrast": "contrast_a",
                "classification": "stable",
                "v02_top1": 0.83,
                "vertex_top1": 0.8,
                "parcel_top1": 0.8,
                "vertex_top2": 1.0,
                "parcel_top2": 1.0,
                "vertex_mean_rank": 1.2,
                "parcel_mean_rank": 1.2,
                "mean_score_margin": 0.05,
                "temporal_late": 0.7,
                "temporal_final": 0.8,
                "temporal_majority": 0.8,
                "average_switch_count": 0.4,
                "vertex_summary_path": "vertex.json",
                "parcel_summary_path": "parcel.json",
                "temporal_summary_path": "temporal.json",
            }
        ],
        warnings=[],
    )

    assert "ASNE Semantic Contrast Benchmark v0.3-lite" in report
    assert "Confidence Warning" in report
    assert "classification" in report
    assert "contrast_a" in report


def test_v03_latest_summary_filters_feature_space(tmp_path: Path) -> None:
    module = _load_script()
    root = tmp_path / "summaries"
    vertex = root / "contrast_a" / "20260101T000000Z_summary.json"
    parcel = root / "contrast_a" / "20260102T000000Z_summary.json"
    vertex.parent.mkdir(parents=True)
    vertex.write_text(
        json.dumps({"signature_mode": "mean_response", "aggregation_mode": "centroid", "scoring_mode": "centroid_raw"}),
        encoding="utf-8",
    )
    parcel.write_text(
        json.dumps({"feature_space": "parcel", "signature_mode": "mean_response", "aggregation_mode": "centroid", "scoring_mode": "centroid_raw"}),
        encoding="utf-8",
    )

    assert module.latest_summary("contrast_a", feature_space="vertex", root=root)["_path"] == str(vertex)
    assert module.latest_summary("contrast_a", feature_space="parcel", root=root)["_path"] == str(parcel)


def test_v03_latest_summary_filters_eval_path(tmp_path: Path) -> None:
    module = _load_script()
    root = tmp_path / "summaries"
    expected = root / "contrast_a" / "20260101T000000Z_summary.json"
    other = root / "contrast_a" / "20260102T000000Z_summary.json"
    expected.parent.mkdir(parents=True)
    base = {"signature_mode": "mean_response", "aggregation_mode": "centroid", "scoring_mode": "centroid_raw"}
    expected.write_text(json.dumps({**base, "eval_path": "eval_a.json"}), encoding="utf-8")
    other.write_text(json.dumps({**base, "eval_path": "eval_b.json"}), encoding="utf-8")

    assert module.latest_summary("contrast_a", feature_space="vertex", root=root, eval_json="eval_a.json")["_path"] == str(expected)


def test_v03_select_contrasts_filters_by_name() -> None:
    module = _load_script()

    selected = module.select_contrasts(["expected_vs_unexpected_paired"])

    assert len(selected) == 1
    assert selected[0].name == "expected_vs_unexpected_paired"


def test_v03_classification_labels() -> None:
    module = _load_script()

    assert module.classify_contrast({"top1_accuracy": 0.8}, {"top1_accuracy": 0.7}, None) == "stable"
    assert module.classify_contrast({"top1_accuracy": 0.6}, {"top1_accuracy": 0.5}, None) == "promising but unstable"
    assert module.classify_contrast({"top1_accuracy": 0.4}, {"top1_accuracy": 0.5}, None) == "weak/deprioritized"


def _load_script():
    path = Path("scripts/run_asne_v03_benchmark.py")
    spec = importlib.util.spec_from_file_location("run_asne_v03_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_asne_v03_benchmark"] = module
    spec.loader.exec_module(module)
    return module
