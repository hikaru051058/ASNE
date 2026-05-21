#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from asne.dictionary import COMPARISON_DISCLAIMER, slugify
from asne.evaluation import load_eval_set, rank_of_expected_category
from asne.notebook_utils import (
    compute_response_norm_timeline,
    compute_segment_similarity_timeline,
    compute_transition_norm_timeline,
    load_mean_response_category_centroids,
    load_raw_segments,
)


def ranked_categories_for_segment(
    similarities: list[float],
    categories: list[str],
) -> list[dict[str, Any]]:
    ranked = sorted(
        (
            {"category": category, "similarity": float(score)}
            for category, score in zip(categories, similarities, strict=False)
        ),
        key=lambda item: item["similarity"],
        reverse=True,
    )
    return ranked


def winner_for_vector(vector: np.ndarray, category_centroids: dict[str, Any], metric: str = "cosine") -> tuple[str | None, list[dict[str, Any]]]:
    timeline = compute_segment_similarity_timeline(np.asarray(vector, dtype=float).reshape(1, -1), category_centroids, metric=metric)
    categories = timeline["categories"]
    ranked = ranked_categories_for_segment(timeline["similarities"][0], categories)
    return (ranked[0]["category"] if ranked else None), ranked


def split_early_late(raw_segments: Any) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw_segments, dtype=float)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.ndim > 2:
        raw = raw.reshape(raw.shape[0], -1)
    if raw.shape[0] <= 1:
        return raw, raw
    split = raw.shape[0] // 2
    return raw[:split], raw[split:]


def count_switches(winners: list[str | None]) -> int:
    return sum(1 for previous, current in zip(winners, winners[1:], strict=False) if previous != current)


def majority_winner(winners: list[str | None], categories: list[str]) -> str | None:
    counts = Counter(winner for winner in winners if winner is not None)
    if not counts:
        return None
    max_count = max(counts.values())
    tied = {category for category, count in counts.items() if count == max_count}
    for category in categories:
        if category in tied:
            return category
    return sorted(tied)[0]


def largest_transition(transition_norm: np.ndarray) -> tuple[int | None, float | None]:
    if transition_norm.size == 0:
        return None, None
    index = int(np.argmax(transition_norm))
    return index, float(transition_norm[index])


def analyze_temporal_example(
    *,
    item_id: str,
    input_text: str,
    expected_category: str,
    raw_segments: Any,
    category_centroids: dict[str, Any],
    mean_response_top_category: str | None = None,
    mean_response_rank_of_expected: int | None = None,
    comparison_output_path: str | None = None,
    metric: str = "cosine",
) -> dict[str, Any]:
    raw = np.asarray(raw_segments, dtype=float)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.ndim > 2:
        raw = raw.reshape(raw.shape[0], -1)
    similarity = compute_segment_similarity_timeline(raw, category_centroids, metric=metric)
    categories = similarity["categories"]
    segment_rankings = [
        ranked_categories_for_segment(row, categories)
        for row in similarity["similarities"]
    ]
    segment_winners = [ranking[0]["category"] if ranking else None for ranking in segment_rankings]
    expected_rank_by_segment = [
        rank_of_expected_category(ranking, expected_category)
        for ranking in segment_rankings
    ]

    early, late = split_early_late(raw)
    early_winner, early_ranking = winner_for_vector(early.mean(axis=0), category_centroids, metric=metric)
    late_winner, late_ranking = winner_for_vector(late.mean(axis=0), category_centroids, metric=metric)
    final_segment_winner = segment_winners[-1] if segment_winners else None
    majority = majority_winner(segment_winners, categories)
    response_norm = compute_response_norm_timeline(raw)
    transition_norm = compute_transition_norm_timeline(raw)
    transition_index, transition_value = largest_transition(transition_norm)
    return {
        "id": item_id,
        "input_text": input_text,
        "expected_category": expected_category,
        "comparison_output_path": comparison_output_path,
        "segment_count": int(raw.shape[0]),
        "segment_winners": segment_winners,
        "segment_rankings": segment_rankings,
        "expected_rank_by_segment": expected_rank_by_segment,
        "early_winner": early_winner,
        "early_ranking": early_ranking,
        "late_winner": late_winner,
        "late_ranking": late_ranking,
        "final_segment_winner": final_segment_winner,
        "majority_segment_winner": majority,
        "number_of_switches": count_switches(segment_winners),
        "largest_transition_index": transition_index,
        "largest_transition_norm": transition_value,
        "response_norm_timeline": response_norm.tolist(),
        "transition_norm_timeline": transition_norm.tolist(),
        "mean_response_top_category": mean_response_top_category,
        "mean_response_rank_of_expected": mean_response_rank_of_expected,
        "mean_response_correct": mean_response_top_category == expected_category if mean_response_top_category else None,
        "early_correct": early_winner == expected_category,
        "late_correct": late_winner == expected_category,
        "final_segment_correct": final_segment_winner == expected_category,
        "majority_segment_correct": majority == expected_category,
        "category_shift": count_switches(segment_winners) > 0,
        "late_matches_expected_after_mean_failed": (
            mean_response_top_category is not None
            and mean_response_top_category != expected_category
            and late_winner == expected_category
        ),
        "temporal_matches_expected_after_mean_failed": (
            mean_response_top_category is not None
            and mean_response_top_category != expected_category
            and (
                late_winner == expected_category
                or final_segment_winner == expected_category
                or majority == expected_category
            )
        ),
    }


def compute_temporal_summary_metrics(examples: list[dict[str, Any]], mean_response_accuracy: float | None = None) -> dict[str, Any]:
    total = len(examples)

    def accuracy(field: str) -> float:
        return sum(1 for example in examples if example.get(field)) / total if total else 0.0

    return {
        "total_examples": total,
        "early_accuracy": accuracy("early_correct"),
        "late_accuracy": accuracy("late_correct"),
        "final_segment_accuracy": accuracy("final_segment_correct"),
        "majority_segment_accuracy": accuracy("majority_segment_correct"),
        "mean_response_accuracy": mean_response_accuracy,
        "average_switch_count": (
            sum(int(example.get("number_of_switches") or 0) for example in examples) / total if total else 0.0
        ),
        "examples_with_category_shift": [
            example["id"] for example in examples if example.get("category_shift")
        ],
        "examples_where_late_matches_expected": [
            example["id"] for example in examples if example.get("late_correct")
        ],
        "examples_where_mean_failed_but_late_succeeded": [
            example["id"] for example in examples if example.get("late_matches_expected_after_mean_failed")
        ],
        "examples_where_mean_failed_but_temporal_succeeded": [
            example["id"] for example in examples if example.get("temporal_matches_expected_after_mean_failed")
        ],
    }


def evaluate_temporal_contrast(args: argparse.Namespace) -> dict[str, Any]:
    eval_payload = load_eval_set(args.eval)
    eval_name = slugify(str(eval_payload.get("name") or Path(args.eval).stem))
    contrast_name = slugify(str(eval_payload.get("contrast_name") or eval_name))
    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%SZ")
    eval_summary = load_eval_summary(args.eval_summary, args.eval, args.dictionary)
    result_by_id = {result["id"]: result for result in eval_summary.get("results", [])} if eval_summary else {}
    centroids = load_mean_response_category_centroids(args.dictionary)
    examples = []
    skipped = []
    for item in eval_payload["items"]:
        result = result_by_id.get(item["id"], {})
        comparison_path = result.get("comparison_output_path")
        if not comparison_path:
            skipped.append({"id": item["id"], "reason": "missing comparison_output_path"})
            continue
        comparison_payload = json.loads(Path(comparison_path).read_text(encoding="utf-8"))
        try:
            raw_segments = load_raw_segments(comparison_payload)
        except (FileNotFoundError, ValueError) as exc:
            skipped.append({"id": item["id"], "reason": str(exc), "comparison_output_path": comparison_path})
            continue
        examples.append(
            analyze_temporal_example(
                item_id=item["id"],
                input_text=item["text"],
                expected_category=str(item["expected_category"]),
                raw_segments=raw_segments,
                category_centroids=centroids,
                mean_response_top_category=result.get("top_category"),
                mean_response_rank_of_expected=result.get("rank_of_expected"),
                comparison_output_path=comparison_path,
                metric=args.metric,
            )
        )
    metrics = compute_temporal_summary_metrics(
        examples,
        mean_response_accuracy=eval_summary.get("top1_accuracy") if eval_summary else None,
    )
    payload = {
        "created_at": timestamp.isoformat(),
        "contrast_name": contrast_name,
        "dictionary_index_path": args.dictionary,
        "eval_path": args.eval,
        "eval_summary_path": args.eval_summary,
        "metric": args.metric,
        **metrics,
        "skipped_examples": skipped,
        "examples": examples,
        "disclaimer": COMPARISON_DISCLAIMER,
    }
    output_path = Path(args.output) if args.output else Path("outputs/asne_temporal_evals") / contrast_name / f"{run_id}_temporal_summary.json"
    report_path = output_path.with_name(output_path.name.replace("_temporal_summary.json", "_temporal_report.md"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path.write_text(generate_temporal_report(payload), encoding="utf-8")
    return {"summary_path": str(output_path), "report_path": str(report_path), "summary": payload}


def load_eval_summary(eval_summary_path: str | None, eval_path: str, dictionary_path: str) -> dict[str, Any]:
    if eval_summary_path:
        return json.loads(Path(eval_summary_path).read_text(encoding="utf-8"))
    candidates = sorted(Path("outputs/asne_evals").glob("**/*_summary.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("eval_path") == eval_path and payload.get("dictionary_index_path") == dictionary_path:
            payload["_summary_path"] = str(path)
            return payload
    raise ValueError("No eval summary was provided and no matching existing summary was found. Run evaluate_asne_dictionary.py first or pass --eval-summary.")


def generate_temporal_report(summary: dict[str, Any]) -> str:
    lines = [
        "# ASNE Temporal Contrast Evaluation",
        "",
        COMPARISON_DISCLAIMER,
        "",
        f"- Contrast: `{summary.get('contrast_name')}`",
        f"- Dictionary: `{summary.get('dictionary_index_path')}`",
        f"- Eval: `{summary.get('eval_path')}`",
        f"- Eval summary: `{summary.get('eval_summary_path')}`",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| total examples | {summary.get('total_examples')} |",
        f"| mean response accuracy | {_format_metric(summary.get('mean_response_accuracy'))} |",
        f"| early accuracy | {_format_metric(summary.get('early_accuracy'))} |",
        f"| late accuracy | {_format_metric(summary.get('late_accuracy'))} |",
        f"| final segment accuracy | {_format_metric(summary.get('final_segment_accuracy'))} |",
        f"| majority segment accuracy | {_format_metric(summary.get('majority_segment_accuracy'))} |",
        f"| average switch count | {_format_metric(summary.get('average_switch_count'))} |",
        "",
        "## Per-Example Temporal Winners",
        "",
        "| id | expected | mean top | early | late | final | majority | switches | largest transition |",
        "|---|---|---|---|---|---|---|---:|---:|",
    ]
    for example in summary.get("examples", []):
        lines.append(
            f"| {example.get('id')} | {example.get('expected_category')} | "
            f"{example.get('mean_response_top_category')} | {example.get('early_winner')} | "
            f"{example.get('late_winner')} | {example.get('final_segment_winner')} | "
            f"{example.get('majority_segment_winner')} | {example.get('number_of_switches')} | "
            f"{_format_metric(example.get('largest_transition_norm'))} |"
        )
    lines.extend(["", "## Segment-Level Differences From Mean Response", ""])
    differing = [
        example
        for example in summary.get("examples", [])
        if example.get("mean_response_top_category") not in {
            example.get("late_winner"),
            example.get("final_segment_winner"),
            example.get("majority_segment_winner"),
        }
    ]
    if not differing:
        lines.append("No examples differed from the mean-response top category under late/final/majority segment views.")
    for example in differing:
        lines.extend(
            [
                f"### {example.get('id')}",
                "",
                f"- Input: {example.get('input_text')}",
                f"- Expected: {example.get('expected_category')}",
                f"- Mean-response top category: {example.get('mean_response_top_category')}",
                f"- Segment winners: {example.get('segment_winners')}",
                f"- Late winner: {example.get('late_winner')}",
                f"- Final segment winner: {example.get('final_segment_winner')}",
                f"- Majority segment winner: {example.get('majority_segment_winner')}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate segment-level temporal ASNE behavior over a contrast eval set.")
    parser.add_argument("--dictionary", required=True, help="Path to dictionary_index.json.")
    parser.add_argument("--eval", required=True, help="Path to eval JSON.")
    parser.add_argument("--eval-summary", default=None, help="Optional existing evaluate_asne_dictionary.py summary JSON.")
    parser.add_argument("--output", default=None, help="Temporal summary JSON output path.")
    parser.add_argument("--metric", choices=["cosine", "pearson"], default="cosine")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate_temporal_contrast(args)
    summary = result["summary"]
    print("ASNE temporal contrast eval complete.")
    print(f"Summary: {result['summary_path']}")
    print(f"Report: {result['report_path']}")
    print(
        "mean={mean} early={early:.2f} late={late:.2f} final={final:.2f} majority={majority:.2f}".format(
            mean=_format_metric(summary.get("mean_response_accuracy")),
            early=summary.get("early_accuracy", 0.0),
            late=summary.get("late_accuracy", 0.0),
            final=summary.get("final_segment_accuracy", 0.0),
            majority=summary.get("majority_segment_accuracy", 0.0),
        )
    )


if __name__ == "__main__":
    main()
