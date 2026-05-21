import json
from pathlib import Path

from asne.visualize import generate_mock_output_plots


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_generate_mock_output_plots_from_sample_outputs(tmp_path):
    output_root = tmp_path / "outputs"
    _write_json(
        output_root / "baseline" / "baseline.json",
        {"condition_id": "neutral", "response": [0.0, 0.0, 0.0]},
    )
    _write_json(
        output_root / "steered" / "threat.json",
        {"condition_id": "threat", "response": [0.5, -0.2, 1.0]},
    )
    _write_json(
        output_root / "deltas" / "threat.json",
        {
            "condition_id": "threat",
            "baseline_condition_id": "neutral",
            "delta": [0.5, -0.2, 1.0],
            "summary": {
                "mean_abs_delta": 0.5666666667,
                "max_abs_delta": 1.0,
                "top_indices": [2, 0, 1],
                "top_values": [1.0, 0.5, -0.2],
                "top_rois": [
                    {
                        "roi": "visual_primary",
                        "mean_abs_delta": 0.5666666667,
                        "max_abs_delta": 1.0,
                        "n_indices": 3,
                    }
                ],
            },
            "roi_summary": {
                "visual_primary": {
                    "mean_abs_delta": 0.5666666667,
                    "max_abs_delta": 1.0,
                    "n_indices": 3,
                }
            },
        },
    )

    paths = generate_mock_output_plots(output_root)

    assert Path(paths["condition_comparison"]).exists()
    assert Path(paths["per_condition"]["threat"]).exists()
    assert Path(paths["roi_condition_summary"]).exists()
    assert Path(paths["roi_per_condition"]["threat"]).exists()
