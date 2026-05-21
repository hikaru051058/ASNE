#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from asne.dictionary import (
    COMPARISON_DISCLAIMER,
    aggregate_categories,
    build_signature_records,
    classify_binary_contrast_axis,
    compute_neutral_baseline,
    compute_category_centroids,
    compute_category_deltas,
    compute_late_minus_early,
    compute_temporal_movement_signatures,
    count_neutral_records,
    load_dictionary_records,
    rank_signature_vectors,
    score_paired_vote,
    score_category_deltas,
    signature_vector,
    slugify,
    top_abs_indices,
    vector_norm,
)
from asne.tribe_adapter import TribeV2Adapter


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def neutral_category_from_dictionary_index(path: str | Path, requested: str | None = None) -> str:
    if requested:
        return requested
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return str(payload.get("neutral_baseline_category") or "neutral")


TEMPORAL_SIGNATURE_CHOICES = [
    "mean_response",
    "delta_from_neutral",
    "net_movement",
    "avg_slope_per_transition",
    "abs_total_movement",
    "late_minus_early",
]


def _predict_text(
    adapter: TribeV2Adapter,
    text: str,
    raw_prediction_path: str | Path | None = None,
) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    try:
        return adapter.predict(str(temp_path), condition_id=None, raw_prediction_path=raw_prediction_path)
    finally:
        temp_path.unlink(missing_ok=True)


def compare_text_with_adapter(
    args: argparse.Namespace,
    adapter: TribeV2Adapter,
    records: list[dict[str, Any]] | None = None,
    neutral_baseline: Any | None = None,
    neutral_baseline_source_count: int | None = None,
) -> dict[str, Any]:
    records = records if records is not None else load_dictionary_records(args.dictionary)
    neutral_category = neutral_category_from_dictionary_index(
        args.dictionary,
        getattr(args, "neutral_category", None),
    )
    neutral_baseline = (
        neutral_baseline
        if neutral_baseline is not None
        else compute_neutral_baseline(records, neutral_category=neutral_category)
    )
    if neutral_baseline_source_count is None:
        neutral_baseline_source_count = count_neutral_records(records, neutral_category=neutral_category)

    timestamp = datetime.now(timezone.utc)
    output_dir = Path(args.output_root) / "asne_comparisons"
    output_name = f"comparison_{timestamp.strftime('%Y%m%dT%H%M%SZ')}_{slugify(args.text[:32])}.json"
    output_path = output_dir / output_name
    query_raw_path = output_dir / "query_raw_segments" / f"{output_path.stem}.npy"

    prediction = _predict_text(adapter, args.text, raw_prediction_path=query_raw_path)
    query_record = {
        "mean_response": prediction["response"],
        "raw_segment_prediction_path": str(query_raw_path) if query_raw_path.exists() else None,
    }
    query_response = np.asarray(prediction["response"], dtype=float)
    query_vector = signature_vector(
        query_record,
        signature=args.signature,
        neutral_baseline=neutral_baseline,
    )
    sort_metric = "cosine" if args.metric in {"cosine", "both"} else "pearson"
    signature_records = build_signature_records(
        records,
        signature=args.signature,
        neutral_baseline=neutral_baseline,
    )
    ranked_signatures = rank_signature_vectors(
        query_vector,
        signature_records,
        sort_metric=sort_metric,
    )
    scoring = getattr(args, "scoring", "topk")
    top_k = int(getattr(args, "top_k", 1000))
    category_scoring_diagnostics: dict[str, Any] = {}
    scoring_warning = None
    if scoring == "binary_axis":
        raw_signature_records = build_signature_records(records, signature="mean_response")
        raw_centroids = compute_category_centroids(raw_signature_records)
        if len(raw_centroids) != 2:
            raise ValueError("binary_axis scoring requires a dictionary with exactly 2 categories.")
        raw_ranked_signatures = rank_signature_vectors(
            query_response,
            raw_signature_records,
            sort_metric=sort_metric,
        )
        best_by_category = {}
        for item in raw_ranked_signatures:
            best_by_category.setdefault(item["category"], item["stimulus_id"])
        category_a_name, category_b_name = list(raw_centroids)
        classification = classify_binary_contrast_axis(
            query_response,
            raw_centroids[category_a_name],
            raw_centroids[category_b_name],
        )
        category_scores = classification["ranking"]
        for item in category_scores:
            item["best_stimulus_id"] = best_by_category.get(item["category"])
        category_scoring_diagnostics = {
            "binary_axis": {
                "category_a": classification["category_a"],
                "category_b": classification["category_b"],
                "signed_score": classification["signed_score"],
                "axis_norm": classification["axis_norm"],
                "midpoint_norm": classification["midpoint_norm"],
                "predicted_side": classification["predicted_side"],
            },
            "category_delta_norm": {
                category: values["centroid_norm"]
                for category, values in raw_centroids.items()
            },
            "selected_dim_count": {
                category: int(np.asarray(values["vector"]).size)
                for category, values in raw_centroids.items()
            },
        }
        if args.signature != "mean_response":
            scoring_warning = (
                "binary_axis scoring uses raw mean_response centroids internally; "
                f"requested signature {args.signature!r} is retained only for non-axis diagnostics."
            )
    elif scoring == "paired_vote":
        category_scores, paired_vote_diagnostics = score_paired_vote(query_response, records)
        raw_signature_records = build_signature_records(records, signature="mean_response")
        raw_ranked_signatures = rank_signature_vectors(
            query_response,
            raw_signature_records,
            sort_metric=sort_metric,
        )
        best_by_category = {}
        for item in raw_ranked_signatures:
            best_by_category.setdefault(item["category"], item["stimulus_id"])
        for item in category_scores:
            item["best_stimulus_id"] = best_by_category.get(item["category"])
        category_scoring_diagnostics = {
            "paired_vote": paired_vote_diagnostics,
            "selected_dim_count": {
                item["category"]: int(query_response.size)
                for item in category_scores
            },
        }
        if args.signature != "mean_response":
            scoring_warning = (
                "paired_vote scoring uses raw mean_response stimulus vectors internally; "
                f"requested signature {args.signature!r} is retained only for non-vote diagnostics."
            )
    elif scoring == "centroid_raw":
        raw_signature_records = build_signature_records(
            records,
            signature=args.signature,
            neutral_baseline=neutral_baseline,
        )
        raw_ranked_signatures = rank_signature_vectors(
            query_vector,
            raw_signature_records,
            sort_metric=sort_metric,
        )
        category_scores = aggregate_categories(
            query_vector,
            raw_signature_records,
            raw_ranked_signatures,
            aggregation="centroid",
            sort_metric=sort_metric,
        )
        category_scoring_diagnostics = {
            "category_delta_norm": {
                item["category"]: item.get("centroid_norm", 0.0)
                for item in category_scores
            },
            "selected_dim_count": {
                item["category"]: int(query_response.size)
                for item in category_scores
            },
        }
    elif scoring == "full" and args.aggregation != "centroid":
        category_scores = aggregate_categories(
            query_vector,
            signature_records,
            ranked_signatures,
            aggregation=args.aggregation,
            sort_metric=sort_metric,
        )
    else:
        category_deltas = compute_category_deltas(records, neutral_baseline)
        category_scores, category_scoring_diagnostics = score_category_deltas(
            query_vector,
            category_deltas,
            scoring=scoring,
            top_k=top_k,
        )
        best_by_category = {}
        for item in ranked_signatures:
            best_by_category.setdefault(item["category"], item["stimulus_id"])
        for item in category_scores:
            item["best_stimulus_id"] = best_by_category.get(item["category"])
    category_centroids = compute_category_centroids(signature_records)
    centroid_norms = {
        category: values["centroid_norm"]
        for category, values in category_centroids.items()
    }
    expected_category = getattr(args, "expected_category", None)
    rank_of_expected = None
    if expected_category:
        for index, item in enumerate(category_scores, start=1):
            if item["category"] == expected_category:
                rank_of_expected = index
                break

    selected_dim_indices_path = {}
    selected_dim_indices = category_scoring_diagnostics.get("selected_dim_indices", {})
    if selected_dim_indices:
        dims_dir = output_dir / "selected_dims" / output_path.stem
        for category, indices in selected_dim_indices.items():
            dim_path = dims_dir / f"{category}_selected_dims.npy"
            dim_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(dim_path, np.asarray(indices, dtype=int))
            selected_dim_indices_path[category] = str(dim_path)
    movement_diagnostics = {}
    if query_raw_path.exists():
        raw_segments = np.load(query_raw_path)
        movement = compute_temporal_movement_signatures(raw_segments)
        late_minus_early = compute_late_minus_early(raw_segments)
        movement_diagnostics = {
            "segment_count": movement["segment_count"],
            "transition_count": movement["transition_count"],
            "movement_norm": vector_norm(movement["net_movement"]),
            "avg_slope_norm": vector_norm(movement["avg_slope_per_transition"]),
            "abs_total_movement_norm": vector_norm(movement["abs_total_movement"]),
            "late_minus_early_norm": vector_norm(late_minus_early),
            "top_moving_dimensions": top_abs_indices(movement["net_movement"]),
            "top_abs_movement_dimensions": top_abs_indices(movement["abs_total_movement"]),
        }

    payload = {
        "created_at": timestamp.isoformat(),
        "dictionary_index_path": str(args.dictionary),
        "input_text": args.text,
        "tts_backend": getattr(args, "tts_backend", "tribev2_gtts"),
        "signature_mode": args.signature,
        "metric_mode": args.metric,
        "aggregation_mode": args.aggregation,
        "scoring_mode": scoring,
        "top_k": top_k,
        "scoring_warning": scoring_warning,
        "expected_category": expected_category,
        "rank_of_expected": rank_of_expected,
        "neutral_baseline_source_count": neutral_baseline_source_count,
        "neutral_baseline_category": neutral_category,
        "query_delta_norm": vector_norm(query_vector),
        "query_response_norm": vector_norm(query_response),
        "query_movement_diagnostics": movement_diagnostics,
        "per_category_centroid_norm": centroid_norms,
        "binary_axis": category_scoring_diagnostics.get("binary_axis"),
        "paired_vote": category_scoring_diagnostics.get("paired_vote"),
        "selected_dim_count": category_scoring_diagnostics.get("selected_dim_count", {}),
        "selected_dim_indices_path": selected_dim_indices_path,
        "category_delta_norm": category_scoring_diagnostics.get("category_delta_norm", {}),
        "query_metadata": prediction.get("metadata", {}),
        "per_category_ranking": category_scores,
        "per_stimulus_results": ranked_signatures,
        "disclaimer": COMPARISON_DISCLAIMER,
    }
    _write_json(output_path, payload)
    return {"output_path": str(output_path), "category_scores": category_scores}


def compare_text(args: argparse.Namespace) -> dict[str, Any]:
    records = load_dictionary_records(args.dictionary)
    neutral_category = neutral_category_from_dictionary_index(
        args.dictionary,
        getattr(args, "neutral_category", None),
    )
    neutral_baseline = compute_neutral_baseline(records, neutral_category=neutral_category)
    neutral_baseline_source_count = count_neutral_records(records, neutral_category=neutral_category)
    adapter = TribeV2Adapter(
        model_name=args.model_name,
        cache_folder=args.cache_folder,
        package_path=args.tribev2_package_path,
        feature_device=args.feature_device,
        tts_backend=getattr(args, "tts_backend", "tribev2_gtts"),
        tts_cache_dir=getattr(args, "tts_cache_dir", "./cache/asne_tts"),
        tts_voice=getattr(args, "tts_voice", "Samantha"),
        tts_rate=getattr(args, "tts_rate", 180),
        openai_tts_model=getattr(args, "openai_tts_model", "gpt-4o-mini-tts"),
        openai_tts_voice=getattr(args, "openai_tts_voice", "alloy"),
        openai_tts_format=getattr(args, "openai_tts_format", "mp3"),
        higgs_model=getattr(args, "higgs_model", "bosonai/higgs-audio-v2-generation-3B-base"),
        higgs_tokenizer=getattr(args, "higgs_tokenizer", "bosonai/higgs-audio-v2-tokenizer"),
        higgs_device=getattr(args, "higgs_device", "auto"),
        higgs_max_new_tokens=getattr(args, "higgs_max_new_tokens", 1024),
        higgs_temperature=getattr(args, "higgs_temperature", 0.3),
        higgs_top_p=getattr(args, "higgs_top_p", 0.95),
        higgs_top_k=getattr(args, "higgs_top_k", 50),
        higgs_scene_description=getattr(args, "higgs_scene_description", "Audio is recorded from a quiet room."),
    )
    adapter.check_model_load(load_model=True)
    return compare_text_with_adapter(
        args,
        adapter,
        records,
        neutral_baseline,
        neutral_baseline_source_count,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare raw text against an ASNE stimulus-response dictionary.")
    parser.add_argument("--dictionary", required=True, help="Path to dictionary_index.json.")
    parser.add_argument("--text", required=True, help="Raw text to compare.")
    parser.add_argument("--tribev2-package-path", default="./tribev2")
    parser.add_argument("--cache-folder", default="./cache")
    parser.add_argument("--model-name", default="facebook/tribev2")
    parser.add_argument("--feature-device", default="cpu")
    parser.add_argument(
        "--tts-backend",
        choices=["tribev2_gtts", "macos_say", "openai", "higgs_audio"],
        default="tribev2_gtts",
        help=(
            "Text stimulus audio backend. 'tribev2_gtts' preserves upstream TRIBE behavior; "
            "'macos_say', 'higgs_audio', and 'openai' generate cached audio before calling TRIBE audio mode."
        ),
    )
    parser.add_argument("--tts-cache-dir", default="./cache/asne_tts")
    parser.add_argument("--tts-voice", default="Samantha", help="Voice name for --tts-backend macos_say.")
    parser.add_argument("--tts-rate", type=int, default=180, help="Speech rate for --tts-backend macos_say.")
    parser.add_argument("--openai-tts-model", default="gpt-4o-mini-tts")
    parser.add_argument("--openai-tts-voice", default="alloy")
    parser.add_argument("--openai-tts-format", default="mp3", choices=["mp3", "wav", "opus", "aac", "flac"])
    parser.add_argument("--higgs-model", default="bosonai/higgs-audio-v2-generation-3B-base")
    parser.add_argument("--higgs-tokenizer", default="bosonai/higgs-audio-v2-tokenizer")
    parser.add_argument("--higgs-device", default="auto", help="Device for Higgs Audio, e.g. auto, mps, cpu, cuda.")
    parser.add_argument("--higgs-max-new-tokens", type=int, default=1024)
    parser.add_argument("--higgs-temperature", type=float, default=0.3)
    parser.add_argument("--higgs-top-p", type=float, default=0.95)
    parser.add_argument("--higgs-top-k", type=int, default=50)
    parser.add_argument("--higgs-scene-description", default="Audio is recorded from a quiet room.")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument(
        "--neutral-category",
        default=None,
        help="Override the baseline category stored in dictionary_index.json.",
    )
    parser.add_argument(
        "--signature",
        choices=TEMPORAL_SIGNATURE_CHOICES,
        default="delta_from_neutral",
        help="Response signature to compare. Defaults to neutral-baseline deltas.",
    )
    parser.add_argument(
        "--metric",
        choices=["cosine", "pearson", "both"],
        default="both",
        help="Metric to prioritize and print. 'both' sorts by cosine and prints both metrics.",
    )
    parser.add_argument(
        "--aggregation",
        choices=["best", "centroid", "mean"],
        default="centroid",
        help="How to aggregate stimulus signatures into category scores.",
    )
    parser.add_argument(
        "--scoring",
        choices=["full", "topk", "weighted", "binary_axis", "centroid_raw", "paired_vote"],
        default="full",
        help=(
            "Category scoring mode. 'full' is the current default. "
            "'topk' and 'weighted' are experimental category-sensitive dimension modes. "
            "'binary_axis' is experimental signed projection along the two-category raw centroid axis. "
            "'centroid_raw' compares raw response centroids directly and is recommended for binary contrast experiments. "
            "'paired_vote' compares against matched stimulus pairs and votes per pair."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1000,
        help="Number of category-sensitive dimensions to use when --scoring topk. Experimental.",
    )
    parser.add_argument(
        "--expected-category",
        default=None,
        help="Optional expected category for diagnostics only.",
    )
    parser.add_argument("--debug-traceback", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = compare_text(args)
    except Exception as exc:
        print(f"ASNE text comparison failed: {type(exc).__name__}: {exc}")
        if args.debug_traceback:
            print("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip())
        return 1

    print("ASNE predicted response similarity ranking:")
    print(f"Signature: {args.signature}")
    print(f"Metric: {args.metric}")
    print(f"Aggregation: {args.aggregation}")
    print(f"Scoring: {args.scoring}")
    for index, item in enumerate(result["category_scores"], start=1):
        if args.metric == "cosine":
            score = item.get("score", item.get("centroid_cosine_similarity", item.get("best_cosine_similarity", item.get("mean_cosine_similarity", 0.0))))
            metric_text = f"cosine={score:.4f}"
        elif args.metric == "pearson":
            score = item.get("centroid_pearson_correlation", item.get("best_pearson_correlation", item.get("mean_pearson_correlation", 0.0)))
            metric_text = f"pearson={score:.4f}"
        else:
            cosine = item.get("score", item.get("centroid_cosine_similarity", item.get("best_cosine_similarity", item.get("mean_cosine_similarity", 0.0))))
            pearson = item.get("centroid_pearson_correlation", item.get("best_pearson_correlation", item.get("mean_pearson_correlation", 0.0)))
            metric_text = (
                f"cosine={cosine:.4f} "
                f"pearson={pearson:.4f}"
            )
        print(f"{index}. {item['category']} {metric_text} best={item['best_stimulus_id']}")
    print(f"Output: {result['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
