from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from asne.dictionary import (
    aggregate_categories,
    aggregate_category_scores,
    aggregate_response_segments,
    build_signature_records,
    classify_binary_contrast_axis,
    compute_binary_contrast_axis,
    compute_neutral_baseline,
    compute_temporal_movement_signatures,
    compute_late_minus_early,
    score_binary_contrast_axis,
    score_paired_vote,
    score_category_deltas,
    select_top_k_dims,
    load_dictionary_records,
    load_stimulus_dictionary,
    rank_dictionary_signatures,
    rank_signature_vectors,
    signature_vector,
    validate_stimulus_dictionary,
    validate_dictionary_expansion,
    validate_eval_expansion,
    weighted_cosine,
)


def test_sample_dictionary_schema() -> None:
    payload = load_stimulus_dictionary("data/stimuli/dictionaries/emotion_context_v0.json")
    validate_stimulus_dictionary(payload)

    assert set(payload["categories"]) == {
        "neutral",
        "threat",
        "sadness",
        "confusion",
        "relief",
    }
    assert all(len(stimuli) == 2 for stimuli in payload["categories"].values())


def test_aggregate_response_segments_with_fake_large_vectors() -> None:
    segments = np.vstack(
        [
            np.ones(20484),
            np.full(20484, 3.0),
            np.full(20484, -1.0),
        ]
    )

    result = aggregate_response_segments(segments, percentile=95)

    assert result["raw_shape"] == [3, 20484]
    assert result["response_shape"] == [20484]
    assert result["segments"] == 3
    assert result["mean_response"].shape == (20484,)
    assert np.allclose(result["mean_response"][:3], [1.0, 1.0, 1.0])
    assert np.allclose(result["peak_abs_response"][:3], [3.0, 3.0, 3.0])


def test_temporal_movement_signatures_for_increasing_sequence() -> None:
    raw = np.asarray([0.0, 2.0, 3.0, 5.0, 8.0])[:, None]

    movement = compute_temporal_movement_signatures(raw)

    assert movement["segment_count"] == 5
    assert movement["transition_count"] == 4
    assert movement["start_response"][0] == 0.0
    assert movement["end_response"][0] == 8.0
    assert movement["net_movement"][0] == 8.0
    assert movement["avg_slope_by_count"][0] == 8.0 / 5.0
    assert movement["avg_slope_per_transition"][0] == 8.0 / 4.0
    assert np.allclose(movement["step_deltas"][:, 0], [2.0, 1.0, 2.0, 3.0])
    assert movement["abs_total_movement"][0] == 8.0
    assert movement["avg_abs_step"][0] == 2.0


def test_temporal_movement_signatures_for_decreasing_sequence() -> None:
    raw = np.asarray([8.0, 5.0, 3.0])[:, None]

    movement = compute_temporal_movement_signatures(raw)

    assert movement["net_movement"][0] == -5.0
    assert movement["avg_slope_per_transition"][0] == -2.5
    assert movement["abs_total_movement"][0] == 5.0


def test_temporal_movement_signatures_for_oscillating_sequence() -> None:
    raw = np.asarray([0.0, 2.0, 0.0, 2.0, 0.0])[:, None]

    movement = compute_temporal_movement_signatures(raw)

    assert movement["net_movement"][0] == 0.0
    assert movement["abs_total_movement"][0] == 8.0
    assert movement["avg_abs_step"][0] == 2.0


def test_temporal_movement_signatures_for_one_segment_are_safe() -> None:
    raw = np.asarray([[4.0, -2.0]])

    movement = compute_temporal_movement_signatures(raw)

    assert movement["segment_count"] == 1
    assert movement["transition_count"] == 0
    assert np.allclose(movement["net_movement"], [0.0, 0.0])
    assert np.allclose(movement["avg_slope_per_transition"], [0.0, 0.0])
    assert np.allclose(movement["abs_total_movement"], [0.0, 0.0])
    assert movement["step_deltas"].shape == (0, 2)


def test_late_minus_early_signature_splits_segments() -> None:
    raw = np.asarray([[0.0], [2.0], [4.0], [8.0]])

    assert np.allclose(compute_late_minus_early(raw), [5.0])


def test_similarity_ranking_with_fake_vectors() -> None:
    records = [
        {"category": "neutral", "stimulus_id": "n1", "mean_response": [1.0, 0.0, 0.0]},
        {"category": "threat", "stimulus_id": "t1", "mean_response": [0.0, 1.0, 0.0]},
        {"category": "relief", "stimulus_id": "r1", "mean_response": [0.8, 0.1, 0.0]},
    ]

    ranked = rank_dictionary_signatures([1.0, 0.0, 0.0], records)
    categories = aggregate_category_scores(ranked)

    assert ranked[0]["stimulus_id"] == "n1"
    assert categories[0]["category"] == "neutral"
    assert categories[0]["mean_cosine_similarity"] > categories[-1]["mean_cosine_similarity"]


def test_delta_from_neutral_changes_ranking_compared_with_raw_mean_response() -> None:
    records = [
        {"category": "neutral", "stimulus_id": "n1", "mean_response": [100.0, 0.0, 0.0]},
        {"category": "neutral", "stimulus_id": "n2", "mean_response": [100.0, 0.0, 0.0]},
        {"category": "threat", "stimulus_id": "t1", "mean_response": [100.0, 10.0, 0.0]},
        {"category": "confusion", "stimulus_id": "c1", "mean_response": [100.0, 0.0, 10.0]},
    ]
    neutral_baseline = compute_neutral_baseline(records)
    query_raw = [100.0, 1.0, 0.0]
    query_delta = np.asarray(query_raw) - neutral_baseline

    raw_ranked = rank_dictionary_signatures(query_raw, records)
    delta_ranked = rank_dictionary_signatures(
        query_delta,
        records,
        signature="delta_from_neutral",
        neutral_baseline=neutral_baseline,
    )

    assert raw_ranked[0]["category"] == "neutral"
    assert delta_ranked[0]["category"] == "threat"


def test_neutral_baseline_averages_neutral_examples() -> None:
    records = [
        {"category": "neutral", "stimulus_id": "n1", "mean_response": [1.0, 3.0]},
        {"category": "neutral", "stimulus_id": "n2", "mean_response": [3.0, 5.0]},
        {"category": "threat", "stimulus_id": "t1", "mean_response": [100.0, 100.0]},
    ]

    baseline = compute_neutral_baseline(records)

    assert np.allclose(baseline, [2.0, 4.0])
    assert np.allclose(
        signature_vector(records[2], "delta_from_neutral", neutral_baseline=baseline),
        [98.0, 96.0],
    )


def test_category_aggregation_returns_best_stimulus_for_selected_metric() -> None:
    ranked = [
        {
            "category": "threat",
            "stimulus_id": "t1",
            "cosine_similarity": 0.2,
            "pearson_correlation": 0.9,
        },
        {
            "category": "threat",
            "stimulus_id": "t2",
            "cosine_similarity": 0.8,
            "pearson_correlation": 0.1,
        },
    ]

    cosine_scores = aggregate_category_scores(ranked, sort_metric="cosine")
    pearson_scores = aggregate_category_scores(ranked, sort_metric="pearson")

    assert cosine_scores[0]["best_stimulus_id"] == "t2"
    assert pearson_scores[0]["best_stimulus_id"] == "t1"


def test_centroid_aggregation_ranks_expected_category_correctly() -> None:
    records = [
        {"category": "neutral", "stimulus_id": "n1", "mean_response": [0.0, 0.0]},
        {"category": "neutral", "stimulus_id": "n2", "mean_response": [0.0, 0.0]},
        {"category": "threat", "stimulus_id": "t1", "mean_response": [1.0, 0.8]},
        {"category": "threat", "stimulus_id": "t2", "mean_response": [1.0, 1.2]},
        {"category": "sadness", "stimulus_id": "s1", "mean_response": [0.0, 1.0]},
        {"category": "sadness", "stimulus_id": "s2", "mean_response": [0.0, 1.2]},
    ]
    baseline = compute_neutral_baseline(records)
    signatures = build_signature_records(records, "delta_from_neutral", baseline)
    query = np.asarray([1.0, 1.0])
    ranked = rank_signature_vectors(query, signatures)

    categories = aggregate_categories(
        query,
        signatures,
        ranked,
        aggregation="centroid",
        sort_metric="cosine",
    )

    assert categories[0]["category"] == "threat"
    assert categories[0]["centroid_norm"] > 0.0


def test_top_k_selects_expected_dimensions() -> None:
    dims = select_top_k_dims([0.1, -5.0, 2.0, 0.0, 4.0], 2)

    assert dims.tolist() == [1, 4]


def test_top_k_scoring_ignores_noisy_irrelevant_dimensions() -> None:
    query = np.asarray([1.0, 0.0, 100.0])
    category_deltas = {
        "expected": {
            "category": "expected",
            "delta": np.asarray([1.0, 0.0, 0.0]),
            "category_delta_norm": 1.0,
            "n_signatures": 1,
        },
        "other": {
            "category": "other",
            "delta": np.asarray([0.0, 1.0, 0.0]),
            "category_delta_norm": 1.0,
            "n_signatures": 1,
        },
    }

    full_scores, _ = score_category_deltas(query, category_deltas, scoring="full")
    topk_scores, diagnostics = score_category_deltas(query, category_deltas, scoring="topk", top_k=1)

    assert topk_scores[0]["category"] == "expected"
    assert topk_scores[0]["score"] > full_scores[0]["score"]
    assert diagnostics["selected_dim_count"]["expected"] == 1


def test_weighted_cosine_favors_high_weight_dimensions() -> None:
    query = [1.0, 1.0]
    expected = [1.0, -1.0]
    unweighted = weighted_cosine(query, expected, [1.0, 1.0])
    weighted = weighted_cosine(query, expected, [10.0, 0.1])

    assert weighted > unweighted


def test_binary_contrast_axis_predicts_category_a_for_negative_side() -> None:
    category_a = {"category": "baseline", "vector": np.asarray([0.0, 0.0]), "n_signatures": 2}
    category_b = {"category": "contrast", "vector": np.asarray([2.0, 0.0]), "n_signatures": 2}

    result = classify_binary_contrast_axis([0.25, 0.0], category_a, category_b)

    assert result["signed_score"] < 0.0
    assert result["ranking"][0]["category"] == "baseline"
    assert result["ranking"][0]["category_side"] == "category_a"
    assert result["axis_norm"] > 0.0


def test_binary_contrast_axis_predicts_category_b_for_positive_side() -> None:
    score = score_binary_contrast_axis([1.75, 0.0], [0.0, 0.0], [2.0, 0.0])

    assert score["signed_score"] > 0.0
    assert score["predicted_side"] == "category_b"


def test_binary_contrast_axis_baseline_centroid_is_not_zero_delta_vector() -> None:
    axis = compute_binary_contrast_axis([10.0, 1.0], [12.0, 1.0])
    result = classify_binary_contrast_axis(
        [10.2, 1.0],
        {"category": "consistent_information", "vector": [10.0, 1.0]},
        {"category": "contradictory_information", "vector": [12.0, 1.0]},
    )

    assert axis["midpoint_norm"] > 0.0
    assert result["ranking"][0]["category"] == "consistent_information"
    assert result["ranking"][0]["centroid_norm"] > 0.0


def test_paired_vote_majority_predicts_expected_category() -> None:
    records = [
        {"category": "a", "stimulus_id": "a_01", "pair_id": "p1", "mean_response": [1.0, 0.0]},
        {"category": "b", "stimulus_id": "b_01", "pair_id": "p1", "mean_response": [0.0, 1.0]},
        {"category": "a", "stimulus_id": "a_02", "pair_id": "p2", "mean_response": [1.0, 0.1]},
        {"category": "b", "stimulus_id": "b_02", "pair_id": "p2", "mean_response": [0.0, 1.0]},
        {"category": "a", "stimulus_id": "a_03", "pair_id": "p3", "mean_response": [1.0, 0.0]},
        {"category": "b", "stimulus_id": "b_03", "pair_id": "p3", "mean_response": [0.2, 1.0]},
    ]

    ranking, diagnostics = score_paired_vote([0.0, 1.0], records)

    assert ranking[0]["category"] == "b"
    assert ranking[0]["vote_count"] == 3
    assert diagnostics["vote_counts"]["b"] == 3
    assert len(diagnostics["votes"]) == 3


def test_paired_vote_tie_is_deterministic_to_category_a() -> None:
    records = [
        {"category": "a", "stimulus_id": "a_01", "mean_response": [1.0, 0.0]},
        {"category": "b", "stimulus_id": "b_01", "mean_response": [0.0, 1.0]},
        {"category": "a", "stimulus_id": "a_02", "mean_response": [1.0, 0.0]},
        {"category": "b", "stimulus_id": "b_02", "mean_response": [0.0, 1.0]},
    ]

    ranking, diagnostics = score_paired_vote([1.0, 1.0], records)

    assert ranking[0]["category"] == "a"
    assert diagnostics["predicted_category"] == "a"
    assert "choose category_a" in diagnostics["tie_handling_rule"]


def test_paired_vote_rejects_non_binary_records() -> None:
    records = [
        {"category": "a", "stimulus_id": "a_01", "mean_response": [1.0]},
        {"category": "b", "stimulus_id": "b_01", "mean_response": [2.0]},
        {"category": "c", "stimulus_id": "c_01", "mean_response": [3.0]},
    ]

    try:
        score_paired_vote([1.0], records)
    except ValueError as exc:
        assert "exactly 2 categories" in str(exc)
    else:
        raise AssertionError("paired_vote should reject non-binary records")


def test_paired_vote_requires_matched_pairs() -> None:
    records = [
        {"category": "a", "stimulus_id": "a_alpha", "mean_response": [1.0]},
        {"category": "b", "stimulus_id": "b_01", "mean_response": [2.0]},
    ]

    try:
        score_paired_vote([1.0], records)
    except ValueError as exc:
        assert "pair_id" in str(exc) or "inferable numeric" in str(exc)
    else:
        raise AssertionError("paired_vote should reject unsafe pair inference")


def test_index_generation_format_loads_records(tmp_path: Path) -> None:
    record_path = tmp_path / "neutral_n1.json"
    record_path.write_text(
        json.dumps(
            {
                "dictionary_name": "fake_dictionary",
                "category": "neutral",
                "stimulus_id": "n1",
                "mean_response": [1.0, 2.0, 3.0],
            }
        ),
        encoding="utf-8",
    )
    index_path = tmp_path / "dictionary_index.json"
    index_path.write_text(
        json.dumps(
            {
                "dictionary_name": "fake_dictionary",
                "stimuli": [
                    {
                        "category": "neutral",
                        "stimulus_id": "n1",
                        "output_path": str(record_path),
                        "response_shape": [3],
                        "segments": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    records = load_dictionary_records(index_path)

    assert len(records) == 1
    assert records[0]["dictionary_name"] == "fake_dictionary"
    assert records[0]["mean_response"] == [1.0, 2.0, 3.0]


def test_dictionary_expansion_validator_flags_direct_category_and_counts() -> None:
    payload = {
        "categories": {
            "threat": [
                {"id": "t1", "text": "The word threat appears here."},
                {"id": "t2", "text": "A clinical phrase appears here."},
            ]
        }
    }

    warnings = validate_dictionary_expansion(payload, expected_examples_per_category=10)

    assert any("expected 10" in warning for warning in warnings)
    assert any("category leakage term" in warning for warning in warnings)
    assert any("prohibited wording" in warning for warning in warnings)


def test_eval_expansion_validator_flags_dictionary_duplicates() -> None:
    dictionary_payload = {
        "categories": {
            "neutral": [{"id": "n1", "text": "A person checks the time."}],
        }
    }
    eval_payload = {
        "items": [
            {
                "id": "e1",
                "expected_category": "neutral",
                "text": "A person checks the time.",
            }
        ]
    }

    warnings = validate_eval_expansion(
        eval_payload,
        dictionary_payload=dictionary_payload,
        expected_examples_per_category=5,
    )

    assert any("duplicates a dictionary stimulus" in warning for warning in warnings)
    assert any("expected 5" in warning for warning in warnings)
