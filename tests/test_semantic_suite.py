from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path


def _load_script(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_suite_config_contains_all_four_contrasts() -> None:
    module = _load_script("scripts/run_asne_semantic_contrast_suite.py", "semantic_suite")

    names = [spec.name for spec in module.CONTRAST_SUITE]

    assert names == [
        "contradiction_vs_consistency_paired",
        "expected_vs_unexpected_paired",
        "approach_vs_static_paired",
        "cause_effect_valid_vs_invalid_paired",
    ]


def test_latest_summary_discovery(tmp_path: Path) -> None:
    module = _load_script("scripts/run_asne_semantic_contrast_suite.py", "semantic_suite_latest")
    older = tmp_path / "contrast_a" / "20260101T000000Z_summary.json"
    newer = tmp_path / "contrast_a" / "20260102T000000Z_summary.json"
    older.parent.mkdir(parents=True)
    older.write_text("{}", encoding="utf-8")
    newer.write_text("{}", encoding="utf-8")

    assert module.latest_static_summary("contrast_a", root=tmp_path) == newer


def test_report_generation_with_fake_summaries(tmp_path: Path) -> None:
    module = _load_script("scripts/generate_asne_contrast_report.py", "contrast_report")
    static = tmp_path / "static" / "fake_contrast" / "20260101T000000Z_summary.json"
    temporal = tmp_path / "temporal" / "fake_contrast" / "20260101T000000Z_temporal_summary.json"
    static.parent.mkdir(parents=True)
    temporal.parent.mkdir(parents=True)
    static.write_text(
        json.dumps(
            {
                "signature_mode": "mean_response",
                "aggregation_mode": "centroid",
                "scoring_mode": "centroid_raw",
                "top1_accuracy": 1.0,
                "top2_accuracy": 1.0,
                "per_category_accuracy": {"a": {"correct": 1, "total": 1}},
            }
        ),
        encoding="utf-8",
    )
    temporal.write_text(
        json.dumps(
            {
                "mean_response_accuracy": 1.0,
                "early_accuracy": 0.5,
                "late_accuracy": 1.0,
                "final_segment_accuracy": 1.0,
                "majority_segment_accuracy": 1.0,
                "average_switch_count": 1.0,
                "examples_where_mean_failed_but_temporal_succeeded": ["x1"],
            }
        ),
        encoding="utf-8",
    )

    report = module.build_report(
        contrasts=["fake_contrast"],
        static_root=tmp_path / "static",
        temporal_root=tmp_path / "temporal",
    )

    assert "ASNE Semantic Contrast Report v0" in report
    assert "fake_contrast" in report
    assert "predicted stimulus-response similarity" in report


def test_suite_dry_run_without_tribe(monkeypatch, capsys) -> None:
    module = _load_script("scripts/run_asne_semantic_contrast_suite.py", "semantic_suite_dry")
    spec = module.ContrastSpec(
        name="fake_contrast",
        dictionary_json="fake_dictionary.json",
        eval_json="fake_eval.json",
        dictionary_output="fake_output",
    )
    monkeypatch.setattr(module, "CONTRAST_SUITE", [spec])
    monkeypatch.setattr(module, "latest_static_summary", lambda contrast_name: Path("fake_summary.json"))
    monkeypatch.setattr(module, "latest_temporal_summary", lambda contrast_name: Path("fake_temporal_summary.json"))

    module.run_suite(
        Namespace(
            skip_build=True,
            run_build=False,
            force_eval=False,
            force_temporal=False,
            dry_run=True,
            tts_backend="macos_say",
            tribev2_package_path="./tribev2",
            cache_folder="./cache",
            feature_device="cpu",
        )
    )
    output = capsys.readouterr().out

    assert "fake_contrast" in output
    assert "WOULD RUN" in output
    assert "fake_summary.json" in output


def test_report_html_export_contains_safety_text() -> None:
    module = _load_script("scripts/generate_asne_contrast_report.py", "contrast_report_html")

    html = module.markdown_to_simple_html("# Title\n\nThis is predicted stimulus-response similarity.")

    assert "<h1>Title</h1>" in html
    assert "predicted stimulus-response similarity" in html
