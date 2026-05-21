from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np


def _load_compare_module():
    path = Path("scripts/compare_asne_text.py")
    spec = importlib.util.spec_from_file_location("compare_asne_text", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compare_cli_accepts_signature_and_metric() -> None:
    module = _load_compare_module()
    parser = module.build_parser()

    args = parser.parse_args(
        [
            "--dictionary",
            "outputs/asne_dictionaries/emotion_context_v0/dictionary_index.json",
            "--text",
            "sample",
            "--signature",
            "mean_response",
            "--metric",
            "pearson",
            "--aggregation",
            "best",
        ]
    )

    assert args.signature == "mean_response"
    assert args.metric == "pearson"
    assert args.aggregation == "best"
    assert args.scoring == "full"
    assert args.top_k == 1000

    binary_args = parser.parse_args(
        [
            "--dictionary",
            "outputs/asne_dictionaries/contradiction_vs_consistency_paired/dictionary_index.json",
            "--text",
            "sample",
            "--scoring",
            "binary_axis",
        ]
    )
    assert binary_args.scoring == "binary_axis"

    vote_args = parser.parse_args(
        [
            "--dictionary",
            "outputs/asne_dictionaries/expected_vs_unexpected_paired/dictionary_index.json",
            "--text",
            "sample",
            "--scoring",
            "paired_vote",
        ]
    )
    assert vote_args.scoring == "paired_vote"

    parcel_args = parser.parse_args(
        [
            "--dictionary",
            "outputs/asne_dictionaries/expected_vs_unexpected_paired/dictionary_index.json",
            "--text",
            "sample",
            "--feature-space",
            "parcel",
            "--parcellation",
            "data/parcellations/fsaverage5_hcp_mmp.csv",
        ]
    )
    assert parcel_args.feature_space == "parcel"
    assert parcel_args.parcellation == "data/parcellations/fsaverage5_hcp_mmp.csv"


def test_compare_cli_accepts_higgs_tts_backend() -> None:
    module = _load_compare_module()
    parser = module.build_parser()

    args = parser.parse_args(
        [
            "--dictionary",
            "outputs/asne_dictionaries/emotion_context_v0/dictionary_index.json",
            "--text",
            "sample",
            "--tts-backend",
            "higgs_audio",
            "--higgs-device",
            "mps",
        ]
    )

    assert args.tts_backend == "higgs_audio"
    assert args.higgs_device == "mps"


def test_evaluate_cli_accepts_feature_space_and_parcellation() -> None:
    sys.path.insert(0, str(Path("scripts").resolve()))
    path = Path("scripts/evaluate_asne_dictionary.py")
    spec = importlib.util.spec_from_file_location("evaluate_asne_dictionary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = module.build_parser()

    args = parser.parse_args(
        [
            "--dictionary",
            "dictionary_index.json",
            "--eval",
            "eval.json",
            "--feature-space",
            "parcel",
            "--parcellation",
            "data/parcellations/fsaverage5_hcp_mmp.csv",
        ]
    )

    assert args.feature_space == "parcel"
    assert args.parcellation == "data/parcellations/fsaverage5_hcp_mmp.csv"


def test_compare_json_includes_disclaimer(tmp_path: Path, monkeypatch) -> None:
    module = _load_compare_module()

    records = [
        {
            "dictionary_name": "fake",
            "category": "neutral",
            "stimulus_id": "n1",
            "mean_response": [10.0, 0.0, 0.0],
        },
        {
            "dictionary_name": "fake",
            "category": "neutral",
            "stimulus_id": "n2",
            "mean_response": [10.0, 0.0, 0.0],
        },
        {
            "dictionary_name": "fake",
            "category": "threat",
            "stimulus_id": "t1",
            "mean_response": [11.0, 1.0, 0.0],
        },
    ]
    stimuli = []
    for record in records:
        path = tmp_path / f"{record['stimulus_id']}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        stimuli.append(
            {
                "category": record["category"],
                "stimulus_id": record["stimulus_id"],
                "output_path": str(path),
            }
        )
    index_path = tmp_path / "dictionary_index.json"
    index_path.write_text(json.dumps({"dictionary_name": "fake", "stimuli": stimuli}), encoding="utf-8")

    class FakeAdapter:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def check_model_load(self, load_model: bool = True) -> None:
            return None

    monkeypatch.setattr(module, "TribeV2Adapter", FakeAdapter)
    monkeypatch.setattr(
        module,
        "_predict_text",
        lambda adapter, text, raw_prediction_path=None: {"response": [11.0, 1.0, 0.0], "metadata": {"fake": True}},
    )

    args = Namespace(
        dictionary=str(index_path),
        text="query",
        model_name="facebook/tribev2",
        cache_folder="./cache",
        tribev2_package_path="./tribev2",
        feature_device="cpu",
        output_root=str(tmp_path),
        signature="delta_from_neutral",
        metric="both",
        aggregation="centroid",
        scoring="topk",
        top_k=2,
        expected_category="threat",
    )
    result = module.compare_text(args)
    payload = json.loads(Path(result["output_path"]).read_text(encoding="utf-8"))

    assert payload["signature_mode"] == "delta_from_neutral"
    assert payload["metric_mode"] == "both"
    assert payload["aggregation_mode"] == "centroid"
    assert payload["scoring_mode"] == "topk"
    assert payload["top_k"] == 2
    assert payload["expected_category"] == "threat"
    assert payload["rank_of_expected"] == 1
    assert "query_delta_norm" in payload
    assert "per_category_centroid_norm" in payload
    assert payload["selected_dim_count"]["threat"] == 2
    assert Path(payload["selected_dim_indices_path"]["threat"]).exists()
    assert "category_delta_norm" in payload
    assert payload["per_stimulus_results"][0]["delta_norm"] >= 0.0
    assert payload["neutral_baseline_source_count"] == 2
    assert payload["per_category_ranking"][0]["category"] == "threat"
    assert payload["disclaimer"] == (
        "This is predicted stimulus-response similarity, not emotion detection, diagnosis, "
        "or measurement of an individual person's mental state."
    )


def test_binary_axis_comparison_json_contains_margin_diagnostics(tmp_path: Path, monkeypatch) -> None:
    module = _load_compare_module()

    records = [
        {
            "dictionary_name": "fake_binary",
            "category": "baseline",
            "stimulus_id": "b1",
            "mean_response": [10.0, 0.0],
        },
        {
            "dictionary_name": "fake_binary",
            "category": "contrast",
            "stimulus_id": "c1",
            "mean_response": [12.0, 0.0],
        },
    ]
    stimuli = []
    for record in records:
        path = tmp_path / f"{record['stimulus_id']}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        stimuli.append(
            {
                "category": record["category"],
                "stimulus_id": record["stimulus_id"],
                "output_path": str(path),
            }
        )
    index_path = tmp_path / "dictionary_index.json"
    index_path.write_text(
        json.dumps(
            {
                "dictionary_name": "fake_binary",
                "neutral_baseline_category": "baseline",
                "stimuli": stimuli,
            }
        ),
        encoding="utf-8",
    )

    class FakeAdapter:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def check_model_load(self, load_model: bool = True) -> None:
            return None

    monkeypatch.setattr(module, "TribeV2Adapter", FakeAdapter)
    monkeypatch.setattr(
        module,
        "_predict_text",
        lambda adapter, text, raw_prediction_path=None: {"response": [11.8, 0.0], "metadata": {"fake": True}},
    )

    args = Namespace(
        dictionary=str(index_path),
        text="query",
        model_name="facebook/tribev2",
        cache_folder="./cache",
        tribev2_package_path="./tribev2",
        feature_device="cpu",
        output_root=str(tmp_path),
        signature="delta_from_neutral",
        metric="both",
        aggregation="centroid",
        scoring="binary_axis",
        top_k=2,
        expected_category="contrast",
        neutral_category="baseline",
    )
    result = module.compare_text(args)
    payload = json.loads(Path(result["output_path"]).read_text(encoding="utf-8"))

    assert payload["scoring_mode"] == "binary_axis"
    assert payload["binary_axis"]["signed_score"] > 0.0
    assert payload["binary_axis"]["predicted_side"] == "category_b"
    assert payload["per_category_ranking"][0]["category"] == "contrast"
    assert payload["rank_of_expected"] == 1
    assert payload["category_delta_norm"]["baseline"] > 0.0


def test_binary_axis_rejects_non_binary_dictionary(tmp_path: Path, monkeypatch) -> None:
    module = _load_compare_module()
    records = [
        {"dictionary_name": "fake", "category": "baseline", "stimulus_id": "b1", "mean_response": [0.0]},
        {"dictionary_name": "fake", "category": "contrast", "stimulus_id": "c1", "mean_response": [1.0]},
        {"dictionary_name": "fake", "category": "other", "stimulus_id": "o1", "mean_response": [2.0]},
    ]
    stimuli = []
    for record in records:
        path = tmp_path / f"{record['stimulus_id']}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        stimuli.append({"category": record["category"], "stimulus_id": record["stimulus_id"], "output_path": str(path)})
    index_path = tmp_path / "dictionary_index.json"
    index_path.write_text(
        json.dumps({"dictionary_name": "fake", "neutral_baseline_category": "baseline", "stimuli": stimuli}),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "_predict_text", lambda adapter, text, raw_prediction_path=None: {"response": [1.0], "metadata": {}})

    args = Namespace(
        dictionary=str(index_path),
        text="query",
        output_root=str(tmp_path),
        signature="delta_from_neutral",
        metric="both",
        aggregation="centroid",
        scoring="binary_axis",
        top_k=2,
        expected_category=None,
        neutral_category="baseline",
    )

    try:
        module.compare_text_with_adapter(args, object())
    except ValueError as exc:
        assert "exactly 2 categories" in str(exc)
    else:
        raise AssertionError("binary_axis should reject non-binary dictionaries")


def test_paired_vote_comparison_json_contains_vote_diagnostics(tmp_path: Path, monkeypatch) -> None:
    module = _load_compare_module()
    records = [
        {"dictionary_name": "fake", "category": "a", "stimulus_id": "a_01", "pair_id": "p1", "mean_response": [1.0, 0.0]},
        {"dictionary_name": "fake", "category": "b", "stimulus_id": "b_01", "pair_id": "p1", "mean_response": [0.0, 1.0]},
        {"dictionary_name": "fake", "category": "a", "stimulus_id": "a_02", "pair_id": "p2", "mean_response": [1.0, 0.1]},
        {"dictionary_name": "fake", "category": "b", "stimulus_id": "b_02", "pair_id": "p2", "mean_response": [0.0, 1.0]},
    ]
    stimuli = []
    for record in records:
        path = tmp_path / f"{record['stimulus_id']}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        stimuli.append({"category": record["category"], "stimulus_id": record["stimulus_id"], "output_path": str(path)})
    index_path = tmp_path / "dictionary_index.json"
    index_path.write_text(
        json.dumps({"dictionary_name": "fake", "neutral_baseline_category": "a", "stimuli": stimuli}),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "_predict_text", lambda adapter, text, raw_prediction_path=None: {"response": [0.0, 1.0], "metadata": {}})

    args = Namespace(
        dictionary=str(index_path),
        text="query",
        output_root=str(tmp_path),
        signature="mean_response",
        metric="both",
        aggregation="centroid",
        scoring="paired_vote",
        top_k=2,
        expected_category="b",
        neutral_category="a",
    )
    result = module.compare_text_with_adapter(args, object())
    payload = json.loads(Path(result["output_path"]).read_text(encoding="utf-8"))

    assert payload["scoring_mode"] == "paired_vote"
    assert payload["paired_vote"]["vote_counts"]["b"] == 2
    assert len(payload["paired_vote"]["votes"]) == 2
    assert payload["per_category_ranking"][0]["category"] == "b"
    assert payload["rank_of_expected"] == 1


def test_parcel_feature_space_requires_parcellation(tmp_path: Path, monkeypatch) -> None:
    module = _load_compare_module()
    index_path = tmp_path / "dictionary_index.json"
    index_path.write_text(json.dumps({"dictionary_name": "fake", "stimuli": []}), encoding="utf-8")
    monkeypatch.setattr(module, "_predict_text", lambda adapter, text, raw_prediction_path=None: {"response": [1.0], "metadata": {}})

    args = Namespace(
        dictionary=str(index_path),
        text="query",
        output_root=str(tmp_path),
        signature="mean_response",
        metric="both",
        aggregation="centroid",
        scoring="centroid_raw",
        top_k=2,
        feature_space="parcel",
        parcellation=None,
        expected_category=None,
        neutral_category="a",
    )

    try:
        module.compare_text_with_adapter(args, object(), records=[])
    except ValueError as exc:
        assert "--parcellation is required" in str(exc)
    else:
        raise AssertionError("parcel feature space should require --parcellation")


def test_parcel_feature_space_aggregates_vectors_for_scoring(tmp_path: Path, monkeypatch) -> None:
    module = _load_compare_module()
    parcellation = tmp_path / "parcellation.csv"
    parcellation.write_text(
        "\n".join(
            [
                "vertex_index,parcel_id,parcel_name",
                "0,p0,Parcel 0",
                "1,p0,Parcel 0",
                "2,p1,Parcel 1",
                "3,p1,Parcel 1",
            ]
        ),
        encoding="utf-8",
    )
    records = [
        {"dictionary_name": "fake", "category": "a", "stimulus_id": "a1", "mean_response": [1.0, 1.0, 0.0, 0.0]},
        {"dictionary_name": "fake", "category": "b", "stimulus_id": "b1", "mean_response": [0.0, 0.0, 2.0, 2.0]},
    ]
    stimuli = []
    for record in records:
        path = tmp_path / f"{record['stimulus_id']}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        stimuli.append({"category": record["category"], "stimulus_id": record["stimulus_id"], "output_path": str(path)})
    index_path = tmp_path / "dictionary_index.json"
    index_path.write_text(
        json.dumps({"dictionary_name": "fake", "neutral_baseline_category": "a", "stimuli": stimuli}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "_predict_text",
        lambda adapter, text, raw_prediction_path=None: {"response": [0.0, 0.0, 3.0, 3.0], "metadata": {}},
    )

    args = Namespace(
        dictionary=str(index_path),
        text="query",
        output_root=str(tmp_path),
        signature="mean_response",
        metric="both",
        aggregation="centroid",
        scoring="centroid_raw",
        top_k=2,
        feature_space="parcel",
        parcellation=str(parcellation),
        expected_vertices=4,
        expected_category="b",
        neutral_category="a",
    )
    result = module.compare_text_with_adapter(args, object())
    payload = json.loads(Path(result["output_path"]).read_text(encoding="utf-8"))

    assert payload["feature_space"] == "parcel"
    assert payload["parcel_count"] == 2
    assert payload["parcellation_path"] == str(parcellation)
    assert payload["selected_dim_count"]["b"] == 2
    assert payload["per_category_ranking"][0]["category"] == "b"
    assert payload["rank_of_expected"] == 1


def test_temporal_signature_comparison_json_contains_movement_diagnostics(tmp_path: Path, monkeypatch) -> None:
    module = _load_compare_module()
    records = [
        {
            "dictionary_name": "fake_temporal",
            "category": "steady",
            "stimulus_id": "steady_01",
            "mean_response": [0.0, 0.0],
        },
        {
            "dictionary_name": "fake_temporal",
            "category": "moving",
            "stimulus_id": "moving_01",
            "mean_response": [0.0, 1.0],
        },
    ]
    raw_arrays = {
        "steady_01": np.asarray([[0.0, 0.0], [0.0, 0.0]]),
        "moving_01": np.asarray([[0.0, 0.0], [0.0, 2.0]]),
    }
    stimuli = []
    for record in records:
        raw_path = tmp_path / f"{record['stimulus_id']}_raw.npy"
        np.save(raw_path, raw_arrays[record["stimulus_id"]])
        record["raw_segment_prediction_path"] = str(raw_path)
        path = tmp_path / f"{record['stimulus_id']}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        stimuli.append({"category": record["category"], "stimulus_id": record["stimulus_id"], "output_path": str(path)})
    index_path = tmp_path / "dictionary_index.json"
    index_path.write_text(
        json.dumps({"dictionary_name": "fake_temporal", "neutral_baseline_category": "steady", "stimuli": stimuli}),
        encoding="utf-8",
    )

    def fake_predict(adapter, text, raw_prediction_path=None):
        assert raw_prediction_path is not None
        Path(raw_prediction_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(raw_prediction_path, np.asarray([[0.0, 0.0], [0.0, 3.0]]))
        return {"response": [0.0, 1.5], "metadata": {"fake": True}}

    monkeypatch.setattr(module, "_predict_text", fake_predict)

    args = Namespace(
        dictionary=str(index_path),
        text="query",
        output_root=str(tmp_path),
        signature="net_movement",
        metric="both",
        aggregation="centroid",
        scoring="centroid_raw",
        top_k=2,
        expected_category="moving",
        neutral_category="steady",
    )
    result = module.compare_text_with_adapter(args, object())
    payload = json.loads(Path(result["output_path"]).read_text(encoding="utf-8"))

    assert payload["signature_mode"] == "net_movement"
    assert payload["scoring_mode"] == "centroid_raw"
    assert payload["per_category_ranking"][0]["category"] == "moving"
    assert payload["query_movement_diagnostics"]["segment_count"] == 2
    assert payload["query_movement_diagnostics"]["transition_count"] == 1
    assert payload["query_movement_diagnostics"]["movement_norm"] == 3.0
    assert payload["query_movement_diagnostics"]["top_moving_dimensions"][0]["index"] == 1
