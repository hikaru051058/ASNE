import numpy as np

from asne.roi import index_to_roi, load_roi_map, summarize_delta_by_roi


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
