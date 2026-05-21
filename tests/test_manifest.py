import json

from asne.manifest import create_experiment_manifest, save_experiment_manifest


def test_create_experiment_manifest_records_run_metadata():
    manifest = create_experiment_manifest(
        experiment_id="mock_20260101T000000Z",
        timestamp_utc="2026-01-01T00:00:00+00:00",
        stimulus="A test stimulus.",
        adapter_type="MockTribeAdapter",
        steering_config_path="configs/steering_conditions.example.yaml",
        roi_map_path="configs/mock_rois.example.yaml",
        steering_condition_ids=["neutral", "threat"],
        output_paths={
            "baseline": "outputs/baseline/mock_20260101T000000Z.json",
            "steered": {"threat": "outputs/steered/mock_20260101T000000Z_threat.json"},
            "deltas": {"threat": "outputs/deltas/mock_20260101T000000Z_threat.json"},
            "report": "outputs/reports/mock_20260101T000000Z.md",
            "assets": {},
        },
        package_version="0.1.0",
    )

    assert manifest["experiment_id"] == "mock_20260101T000000Z"
    assert manifest["adapter_type"] == "MockTribeAdapter"
    assert manifest["steering_condition_ids"] == ["neutral", "threat"]
    assert manifest["package_version"] == "0.1.0"
    assert "mock predicted cortical responses" in manifest["safety_note"].lower()


def test_save_experiment_manifest_writes_json(tmp_path):
    manifest = create_experiment_manifest(
        experiment_id="mock_20260101T000000Z",
        stimulus="A test stimulus.",
        adapter_type="MockTribeAdapter",
        steering_config_path="configs/steering_conditions.example.yaml",
        roi_map_path=None,
        steering_condition_ids=["neutral"],
        output_paths={"baseline": "baseline.json"},
    )

    path = save_experiment_manifest(manifest, tmp_path)
    saved = json.loads(path.read_text(encoding="utf-8"))

    assert path.name == "mock_20260101T000000Z.json"
    assert saved["experiment_id"] == "mock_20260101T000000Z"
