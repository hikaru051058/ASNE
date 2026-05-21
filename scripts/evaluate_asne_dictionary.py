#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from asne.dictionary import slugify
from asne.dictionary import compute_neutral_baseline, count_neutral_records, load_dictionary_records
from asne.evaluation import (
    compute_eval_summary,
    generate_failure_report,
    load_eval_set,
    rank_of_expected_category,
    top_category_from_comparison,
)
from asne.tribe_adapter import TribeV2Adapter

from compare_asne_text import compare_text_with_adapter
from compare_asne_text import neutral_category_from_dictionary_index


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evaluate_dictionary(args: argparse.Namespace) -> dict[str, Any]:
    eval_payload = load_eval_set(args.eval)
    eval_name = slugify(str(eval_payload.get("name") or Path(args.eval).stem))
    output_group = eval_payload.get("output_group")
    contrast_name = eval_payload.get("contrast_name")
    if output_group is None and "contrasts" in Path(args.eval).parts:
        output_group = "contrasts"
    eval_output_name = slugify(str(contrast_name or eval_name))
    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ")
    if output_group:
        eval_output_dir = Path(args.output_root) / "asne_evals" / slugify(str(output_group)) / eval_output_name
    else:
        eval_output_dir = Path(args.output_root) / "asne_evals" / eval_output_name
    comparison_output_root = eval_output_dir / run_id / "comparisons"
    dictionary_records = load_dictionary_records(args.dictionary)
    neutral_category = neutral_category_from_dictionary_index(args.dictionary, args.neutral_category)
    neutral_baseline = compute_neutral_baseline(dictionary_records, neutral_category=neutral_category)
    neutral_baseline_source_count = count_neutral_records(dictionary_records, neutral_category=neutral_category)
    adapter = TribeV2Adapter(
        model_name=args.model_name,
        cache_folder=args.cache_folder,
        package_path=args.tribev2_package_path,
        feature_device=args.feature_device,
        tts_backend=args.tts_backend,
        tts_cache_dir=args.tts_cache_dir,
        tts_voice=args.tts_voice,
        tts_rate=args.tts_rate,
        openai_tts_model=args.openai_tts_model,
        openai_tts_voice=args.openai_tts_voice,
        openai_tts_format=args.openai_tts_format,
        higgs_model=args.higgs_model,
        higgs_tokenizer=args.higgs_tokenizer,
        higgs_device=args.higgs_device,
        higgs_max_new_tokens=args.higgs_max_new_tokens,
        higgs_temperature=args.higgs_temperature,
        higgs_top_p=args.higgs_top_p,
        higgs_top_k=args.higgs_top_k,
        higgs_scene_description=args.higgs_scene_description,
    )
    adapter.check_model_load(load_model=True)

    categories = []
    results = []
    for item in eval_payload["items"]:
        expected = str(item["expected_category"])
        if expected not in categories:
            categories.append(expected)

        compare_args = argparse.Namespace(
            dictionary=args.dictionary,
            text=item["text"],
            tribev2_package_path=args.tribev2_package_path,
            cache_folder=args.cache_folder,
            model_name=args.model_name,
            feature_device=args.feature_device,
            output_root=str(comparison_output_root),
            tts_backend=args.tts_backend,
            tts_cache_dir=args.tts_cache_dir,
            tts_voice=args.tts_voice,
            tts_rate=args.tts_rate,
            openai_tts_model=args.openai_tts_model,
            openai_tts_voice=args.openai_tts_voice,
            openai_tts_format=args.openai_tts_format,
            higgs_model=args.higgs_model,
            higgs_tokenizer=args.higgs_tokenizer,
            higgs_device=args.higgs_device,
            higgs_max_new_tokens=args.higgs_max_new_tokens,
            higgs_temperature=args.higgs_temperature,
            higgs_top_p=args.higgs_top_p,
            higgs_top_k=args.higgs_top_k,
            higgs_scene_description=args.higgs_scene_description,
            signature=args.signature,
            metric=args.metric,
            aggregation=args.aggregation,
            scoring=args.scoring,
            top_k=args.top_k,
            expected_category=expected,
            neutral_category=neutral_category,
        )
        print(f"Evaluating {item['id']} expected={expected}")
        comparison_result = compare_text_with_adapter(
            compare_args,
            adapter,
            records=dictionary_records,
            neutral_baseline=neutral_baseline,
            neutral_baseline_source_count=neutral_baseline_source_count,
        )
        comparison_path = Path(comparison_result["output_path"])
        comparison_payload = json.loads(comparison_path.read_text(encoding="utf-8"))
        ranking = comparison_payload.get("per_category_ranking", [])
        top_category = top_category_from_comparison(comparison_payload)
        rank_of_expected = rank_of_expected_category(ranking, expected)
        results.append(
            {
                "id": item["id"],
                "input_text": item["text"],
                "expected_category": expected,
                "top_category": top_category,
                "is_top1_correct": expected == top_category,
                "is_top2_correct": rank_of_expected is not None and rank_of_expected <= 2,
                "rank_of_expected": rank_of_expected,
                "comparison_output_path": str(comparison_path),
                "per_category_ranking": ranking,
                "query_delta_norm": comparison_payload.get("query_delta_norm"),
                "query_response_norm": comparison_payload.get("query_response_norm"),
                "query_movement_diagnostics": comparison_payload.get("query_movement_diagnostics", {}),
                "binary_axis": comparison_payload.get("binary_axis"),
                "paired_vote": comparison_payload.get("paired_vote"),
                "per_category_centroid_norm": comparison_payload.get("per_category_centroid_norm", {}),
                "category_delta_norm": comparison_payload.get("category_delta_norm", {}),
                "selected_dim_count": comparison_payload.get("selected_dim_count", {}),
                "selected_dim_indices_path": comparison_payload.get("selected_dim_indices_path", {}),
            }
        )

    summary = compute_eval_summary(
        eval_name=eval_name,
        results=results,
        categories=categories,
        signature_mode=args.signature,
        metric_mode=args.metric,
        dictionary_index_path=args.dictionary,
        aggregation_mode=args.aggregation,
        scoring_mode=args.scoring,
        top_k=args.top_k,
    )
    summary["created_at"] = timestamp.isoformat()
    summary["eval_path"] = args.eval
    summary["tts_backend"] = args.tts_backend
    summary["tts_config"] = {
        "tts_cache_dir": args.tts_cache_dir,
        "tts_voice": args.tts_voice,
        "tts_rate": args.tts_rate,
        "openai_tts_model": args.openai_tts_model,
        "openai_tts_voice": args.openai_tts_voice,
        "openai_tts_format": args.openai_tts_format,
        "higgs_model": args.higgs_model,
        "higgs_tokenizer": args.higgs_tokenizer,
        "higgs_device": args.higgs_device,
        "higgs_max_new_tokens": args.higgs_max_new_tokens,
        "higgs_temperature": args.higgs_temperature,
        "higgs_top_p": args.higgs_top_p,
        "higgs_top_k": args.higgs_top_k,
        "higgs_scene_description": args.higgs_scene_description,
    }
    if contrast_name:
        summary["contrast_name"] = contrast_name
    if output_group:
        summary["output_group"] = output_group
    summary["neutral_baseline_category"] = neutral_category
    summary["results"] = results

    summary_path = eval_output_dir / f"{run_id}_summary.json"
    _write_json(summary_path, summary)
    failure_report_path = eval_output_dir / f"{run_id}_failure_report.md"
    failure_report_path.write_text(generate_failure_report(summary), encoding="utf-8")
    return {
        "summary_path": str(summary_path),
        "failure_report_path": str(failure_report_path),
        "summary": summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an ASNE stimulus-response dictionary.")
    parser.add_argument("--dictionary", required=True, help="Path to dictionary_index.json.")
    parser.add_argument("--eval", required=True, help="Path to eval JSON.")
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
        choices=[
            "mean_response",
            "delta_from_neutral",
            "net_movement",
            "avg_slope_per_transition",
            "abs_total_movement",
            "late_minus_early",
        ],
        default="delta_from_neutral",
    )
    parser.add_argument(
        "--metric",
        choices=["cosine", "pearson", "both"],
        default="both",
    )
    parser.add_argument(
        "--aggregation",
        choices=["best", "centroid", "mean"],
        default="centroid",
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
        default=500,
        help="Number of category-sensitive dimensions to use when --scoring topk. Experimental.",
    )
    parser.add_argument("--debug-traceback", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = evaluate_dictionary(args)
    except Exception as exc:
        print(f"ASNE eval failed: {type(exc).__name__}: {exc}")
        if args.debug_traceback:
            print("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip())
        return 1

    summary = result["summary"]
    print("ASNE eval complete.")
    print(
        f"total={summary['total_examples']} "
        f"top1_correct={summary['correct_top1']} "
        f"top1_accuracy={summary['top1_accuracy']:.2f} "
        f"top2_accuracy={summary['top2_accuracy']:.2f} "
        f"mean_rank_expected={summary['mean_rank_expected']:.2f} "
        f"scoring={summary['scoring_mode']} "
        f"top_k={summary['top_k']}"
    )
    for category, stats in summary["per_category_accuracy"].items():
        print(f"{category}: {stats['correct']}/{stats['total']}")
    print(f"Summary: {result['summary_path']}")
    print(f"Failure report: {result['failure_report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
