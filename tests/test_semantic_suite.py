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
    roi = tmp_path / "roi" / "fake_contrast" / "roi_report.md"
    vertex_vs_parcel = Path("outputs/asne_reports/vertex_vs_parcel_scoring_v0.md")
    static.parent.mkdir(parents=True)
    temporal.parent.mkdir(parents=True)
    roi.parent.mkdir(parents=True)
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
    roi.write_text(
        "\n".join(
            [
                "# ASNE ROI / Parcel Contrast Report",
                "",
                "## Top Parcels By Absolute Contrast Delta",
                "",
                "| rank | parcel_id | parcel_name | delta | abs_delta |",
                "|---:|---|---|---:|---:|",
                "| 1 | `p1` | Parcel 1 | 0.300000 | 0.300000 |",
                "| 2 | `p2` | Parcel 2 | -0.200000 | 0.200000 |",
                "",
                "## Top Parcels By Late Minus Early",
                "",
                "| rank | parcel_id | parcel_name | delta | abs_delta |",
                "|---:|---|---|---:|---:|",
                "| 1 | `p3` | Parcel 3 | 0.400000 | 0.400000 |",
            ]
        ),
        encoding="utf-8",
    )

    report = module.build_report(
        contrasts=["fake_contrast"],
        static_root=tmp_path / "static",
        temporal_root=tmp_path / "temporal",
        roi_root=tmp_path / "roi",
    )

    assert "ASNE Semantic Contrast Report v0" in report
    assert "fake_contrast" in report
    assert "ROI / Parcel-Level Predicted Response Summary" in report
    assert "p1" in report
    assert "p3" in report
    assert "Add ROI or parcel-level aggregation for interpretability" not in report
    assert "predicted stimulus-response similarity" in report
    if vertex_vs_parcel.exists():
        assert "Vertex vs Parcel Scoring" in report


def test_roi_report_discovery_finds_canonical_and_timestamped_names(tmp_path: Path) -> None:
    module = _load_script("scripts/generate_asne_contrast_report.py", "contrast_report_roi_discovery")
    canonical = tmp_path / "roi" / "fake_contrast" / "roi_report.md"
    timestamped = tmp_path / "roi" / "fake_contrast" / "20260101T000000Z_roi_report.md"
    canonical.parent.mkdir(parents=True)
    canonical.write_text("# canonical", encoding="utf-8")
    timestamped.write_text("# timestamped", encoding="utf-8")

    paths = module.find_roi_reports("fake_contrast", tmp_path / "roi")

    assert str(canonical) in paths
    assert str(timestamped) in paths


def test_extract_top_roi_parcels_from_named_section(tmp_path: Path) -> None:
    module = _load_script("scripts/generate_asne_contrast_report.py", "contrast_report_roi_parse")
    report = tmp_path / "roi_report.md"
    report.write_text(
        "\n".join(
            [
                "## Top Parcels By Absolute Contrast Delta",
                "",
                "| rank | parcel_id | parcel_name | delta | abs_delta |",
                "|---:|---|---|---:|---:|",
                "| 1 | `p1` | Parcel 1 | -0.500000 | 0.500000 |",
                "| 2 | `p2` | Parcel 2 | 0.300000 | 0.300000 |",
                "",
                "## Top Parcels By Late Minus Early",
                "",
                "| rank | parcel_id | parcel_name | delta | abs_delta |",
                "|---:|---|---|---:|---:|",
                "| 1 | `p3` | Parcel 3 | 0.700000 | 0.700000 |",
            ]
        ),
        encoding="utf-8",
    )

    contrast_rows = module.extract_top_roi_parcels(report, top_k=1)
    late_rows = module.extract_top_roi_parcels(report, top_k=1, section="Top Parcels By Late Minus Early")

    assert contrast_rows == [
        {"rank": "1", "parcel_id": "p1", "parcel_name": "Parcel 1", "delta": "-0.500000", "abs_delta": "0.500000"}
    ]
    assert late_rows[0]["parcel_id"] == "p3"


def test_extract_vertex_vs_parcel_rows(tmp_path: Path) -> None:
    module = _load_script("scripts/generate_asne_contrast_report.py", "contrast_report_vertex_parcel")
    report = tmp_path / "vertex_vs_parcel.md"
    report.write_text(
        "\n".join(
            [
                "| contrast | vertex_top1 | parcel_top1 | vertex_top2 | parcel_top2 | vertex_mean_rank | parcel_mean_rank | difference | recommendation |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|",
                "| `contrast_a` | 0.83 | 0.83 | 1.00 | 1.00 | 1.17 | 1.17 | 0.00 | parcel viable |",
            ]
        ),
        encoding="utf-8",
    )

    rows = module.extract_vertex_vs_parcel_rows(report)

    assert rows == [
        {
            "contrast": "contrast_a",
            "vertex_top1": "0.83",
            "parcel_top1": "0.83",
            "vertex_top2": "1.00",
            "parcel_top2": "1.00",
            "vertex_mean_rank": "1.17",
            "parcel_mean_rank": "1.17",
            "difference": "0.00",
            "recommendation": "parcel viable",
        }
    ]


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
