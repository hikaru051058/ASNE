from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def test_latest_summary_filters_by_feature_space(tmp_path: Path) -> None:
    module = _load_script()
    vertex = tmp_path / "contrast_a" / "20260101T000000Z_summary.json"
    parcel = tmp_path / "contrast_a" / "20260102T000000Z_summary.json"
    vertex.parent.mkdir(parents=True)
    vertex.write_text(
        json.dumps(
            {
                "signature_mode": "mean_response",
                "aggregation_mode": "centroid",
                "scoring_mode": "centroid_raw",
                "top1_accuracy": 0.5,
            }
        ),
        encoding="utf-8",
    )
    parcel.write_text(
        json.dumps(
            {
                "feature_space": "parcel",
                "signature_mode": "mean_response",
                "aggregation_mode": "centroid",
                "scoring_mode": "centroid_raw",
                "top1_accuracy": 1.0,
            }
        ),
        encoding="utf-8",
    )

    assert module.latest_summary("contrast_a", "vertex", tmp_path)["_path"] == str(vertex)
    assert module.latest_summary("contrast_a", "parcel", tmp_path)["_path"] == str(parcel)


def test_vertex_vs_parcel_report_generation_with_fake_summaries() -> None:
    module = _load_script()
    rows = module.build_rows(
        {
            "contrast_a": {
                "_path": "vertex.json",
                "top1_accuracy": 0.5,
                "top2_accuracy": 1.0,
                "mean_rank_expected": 1.5,
            }
        },
        {
            "contrast_a": {
                "_path": "parcel.json",
                "top1_accuracy": 1.0,
                "top2_accuracy": 1.0,
                "mean_rank_expected": 1.0,
            }
        },
    )
    report = module.format_report(rows, "parcellation.csv")

    assert rows[0]["difference"] == 0.5
    assert "parcel viable" in rows[0]["recommendation"]
    assert "ASNE Vertex vs Parcel Scoring Report v0" in report
    assert "contrast_a" in report
    assert "parcellation.csv" in report


def _load_script():
    path = Path("scripts/compare_vertex_vs_parcel_scoring.py")
    spec = importlib.util.spec_from_file_location("compare_vertex_vs_parcel_scoring", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["compare_vertex_vs_parcel_scoring"] = module
    spec.loader.exec_module(module)
    return module
