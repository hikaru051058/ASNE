import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from asne.roi import (
    aggregate_segments_to_parcels,
    aggregate_vertices_to_parcels,
    compute_parcel_delta,
    index_to_roi,
    load_parcellation,
    load_roi_map,
    rank_parcels_by_abs_delta,
    summarize_delta_by_roi,
    validate_parcellation,
)


def test_load_roi_map_covers_128_mock_indices():
    roi_map = load_roi_map("configs/mock_rois.example.yaml")

    assert roi_map["response_size"] == 128
    assert len(roi_map["by_index"]) == 128
    assert index_to_roi(0, roi_map) == "visual_primary"
    assert index_to_roi(127, roi_map) == "uncertainty_monitoring_proxy"


def test_summarize_delta_by_roi_returns_roi_metrics():
    roi_map = load_roi_map("configs/mock_rois.example.yaml")
    delta = np.zeros(128)
    delta[0] = 2.0
    delta[1] = -1.0

    summaries = summarize_delta_by_roi(delta, roi_map)

    assert summaries["visual_primary"]["n_indices"] == 11
    assert summaries["visual_primary"]["max_abs_delta"] == 2.0
    assert summaries["visual_primary"]["mean_abs_delta"] == 3.0 / 11.0


def test_parcellation_validation_passes_with_mock_data():
    parcellation = load_parcellation("tests/fixtures/mock_parcellation_20_vertices.csv")

    validate_parcellation(parcellation, expected_vertices=20)

    assert parcellation["vertex_count"] == 20
    assert parcellation["parcel_count"] == 4


def test_parcellation_validation_fails_for_missing_vertex(tmp_path: Path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "vertex_index,parcel_id,parcel_name\n0,p0,P0\n2,p0,P0\n",
        encoding="utf-8",
    )
    parcellation = load_parcellation(path)

    with pytest.raises(ValueError, match="missing vertex"):
        validate_parcellation(parcellation, expected_vertices=3)


def test_aggregate_vertices_to_parcels_returns_expected_means():
    parcellation = load_parcellation("tests/fixtures/mock_parcellation_20_vertices.csv")
    vector = np.arange(20, dtype=float)

    parcels = aggregate_vertices_to_parcels(vector, parcellation)

    assert parcels["p0"] == pytest.approx(2.0)
    assert parcels["p1"] == pytest.approx(7.0)
    assert parcels["p2"] == pytest.approx(12.0)
    assert parcels["p3"] == pytest.approx(17.0)


def test_aggregate_segments_to_parcels_returns_expected_shape():
    parcellation = load_parcellation("tests/fixtures/mock_parcellation_20_vertices.csv")
    raw = np.vstack([np.arange(20, dtype=float), np.arange(20, dtype=float) + 10.0])

    parcels = aggregate_segments_to_parcels(raw, parcellation)

    assert parcels["values"].shape == (2, 4)
    assert parcels["parcel_ids"] == ["p0", "p1", "p2", "p3"]
    assert parcels["values"][1, 0] == pytest.approx(12.0)


def test_rank_parcels_by_abs_delta_returns_sorted_parcels():
    ranked = rank_parcels_by_abs_delta({"p0": 0.1, "p1": -3.0, "p2": 2.0}, top_k=2)

    assert [item["parcel_id"] for item in ranked] == ["p1", "p2"]
    assert ranked[0]["abs_delta"] == pytest.approx(3.0)


def test_compute_parcel_delta():
    delta = compute_parcel_delta({"p0": 1.0, "p1": 2.0}, {"p0": 4.0, "p1": -1.0})

    assert delta == {"p0": 3.0, "p1": -3.0}


def test_roi_report_generation_with_fake_dictionary_outputs(tmp_path: Path):
    module = _load_roi_script()
    parcellation = Path("tests/fixtures/mock_parcellation_20_vertices.csv")
    output_dir = tmp_path / "fake_dictionary"
    raw_dir = output_dir / "raw_segments"
    raw_dir.mkdir(parents=True)
    stimuli = []
    for category, offset in [("a", 0.0), ("b", 10.0)]:
        raw_path = raw_dir / f"{category}.npy"
        raw = np.vstack([np.arange(20, dtype=float) + offset, np.arange(20, dtype=float) + offset + 1.0])
        np.save(raw_path, raw)
        record_path = output_dir / f"{category}.json"
        record = {
            "dictionary_name": "fake_dictionary",
            "category": category,
            "stimulus_id": f"{category}_01",
            "mean_response": raw.mean(axis=0).tolist(),
            "raw_segment_prediction_path": str(raw_path),
        }
        record_path.write_text(json.dumps(record), encoding="utf-8")
        stimuli.append({"category": category, "stimulus_id": f"{category}_01", "output_path": str(record_path)})
    index_path = output_dir / "dictionary_index.json"
    index_path.write_text(json.dumps({"dictionary_name": "fake_dictionary", "stimuli": stimuli}), encoding="utf-8")
    report_path = tmp_path / "roi_report.md"

    result = module.analyze_roi_contrast(
        Namespace(
            dictionary=str(index_path),
            eval_summary=None,
            temporal_summary=None,
            parcellation=str(parcellation),
            output=str(report_path),
            top_k=2,
            expected_vertices=20,
        )
    )
    report = Path(result["output_path"]).read_text(encoding="utf-8")

    assert "ASNE ROI / Parcel Contrast Report" in report
    assert "predicted response similarity from TRIBE outputs" in report
    assert "Top Parcels By Absolute Contrast Delta" in report


def test_real_fsaverage5_parcellation_validates_if_present():
    paths = sorted(Path("data/parcellations").glob("fsaverage5_*.csv"))
    if not paths:
        pytest.skip("No real fsaverage5 parcellation CSV has been generated.")
    for path in paths:
        parcellation = load_parcellation(path)
        validate_parcellation(parcellation, expected_vertices=20484)


def _load_roi_script():
    path = Path("scripts/analyze_asne_roi_contrast.py")
    spec = importlib.util.spec_from_file_location("analyze_asne_roi_contrast", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
