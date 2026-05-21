from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .dictionary import COMPARISON_DISCLAIMER


REQUIRED_EVAL_FIELDS = {"id", "expected_category", "text"}


def load_eval_set(path: str | Path) -> dict[str, Any]:
    eval_path = Path(path)
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    validate_eval_set(payload)
    if not payload.get("name"):
        payload["name"] = eval_path.stem
    return payload


def validate_eval_set(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Evaluation payload must be a JSON object.")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Evaluation payload must include a non-empty 'items' list.")
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each evaluation item must be an object.")
        missing = REQUIRED_EVAL_FIELDS - set(item)
        if missing:
            raise ValueError(f"Evaluation item is missing required fields: {sorted(missing)}")
        for field in REQUIRED_EVAL_FIELDS:
            if not str(item[field]).strip():
                raise ValueError(f"Evaluation item field {field!r} cannot be empty.")


def top_category_from_comparison(comparison: dict[str, Any]) -> str | None:
    ranking = comparison.get("per_category_ranking") or comparison.get("category_scores") or []
    if not ranking:
        return None
    return ranking[0].get("category")


def rank_of_expected_category(ranking: list[dict[str, Any]], expected_category: str) -> int | None:
    for index, item in enumerate(ranking, start=1):
        if item.get("category") == expected_category:
            return index
    return None


def build_confusion_matrix(categories: list[str], results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    observed = set(categories)
    for result in results:
        observed.add(result["expected_category"])
        if result.get("top_category"):
            observed.add(result["top_category"])
    ordered = [category for category in categories if category in observed]
    ordered.extend(sorted(observed - set(ordered)))

    matrix = {expected: {predicted: 0 for predicted in ordered} for expected in ordered}
    for result in results:
        expected = result["expected_category"]
        predicted = result.get("top_category") or "<none>"
        if expected not in matrix:
            matrix[expected] = {category: 0 for category in ordered}
        if predicted not in matrix[expected]:
            for row in matrix.values():
                row[predicted] = 0
        matrix[expected][predicted] += 1
    return matrix


def compute_eval_summary(
    eval_name: str,
    results: list[dict[str, Any]],
    categories: list[str],
    signature_mode: str,
    metric_mode: str,
    dictionary_index_path: str,
    aggregation_mode: str = "centroid",
    scoring_mode: str = "full",
    top_k: int = 1000,
) -> dict[str, Any]:
    total = len(results)
    correct_top1 = sum(1 for result in results if result.get("is_top1_correct"))
    correct_top2 = sum(
        1
        for result in results
        if result.get("rank_of_expected") is not None and result["rank_of_expected"] <= 2
    )
    ranks = [
        result["rank_of_expected"]
        for result in results
        if result.get("rank_of_expected") is not None
    ]
    per_category_accuracy = {}
    for category in categories:
        category_results = [result for result in results if result["expected_category"] == category]
        correct = sum(1 for result in category_results if result.get("is_top1_correct"))
        total_category = len(category_results)
        per_category_accuracy[category] = {
            "correct": correct,
            "total": total_category,
            "accuracy": (correct / total_category) if total_category else 0.0,
        }

    failed_examples = [
        {
            "id": result["id"],
            "expected_category": result["expected_category"],
            "top_category": result.get("top_category"),
            "comparison_output_path": result.get("comparison_output_path"),
            "rank_of_expected": result.get("rank_of_expected"),
            "input_text": result.get("input_text"),
            "query_delta_norm": result.get("query_delta_norm"),
            "query_response_norm": result.get("query_response_norm"),
            "binary_axis": result.get("binary_axis"),
            "paired_vote": result.get("paired_vote"),
            "per_category_centroid_norm": result.get("per_category_centroid_norm", {}),
            "per_category_ranking": result.get("per_category_ranking", []),
        }
        for result in results
        if not result.get("is_top1_correct")
    ]

    return {
        "eval_name": eval_name,
        "dictionary_index_path": dictionary_index_path,
        "total_examples": total,
        "correct_top1": correct_top1,
        "top1_accuracy": (correct_top1 / total) if total else 0.0,
        "correct_top2": correct_top2,
        "top2_accuracy": (correct_top2 / total) if total else 0.0,
        "mean_rank_expected": (sum(ranks) / len(ranks)) if ranks else None,
        "per_category_accuracy": per_category_accuracy,
        "confusion_matrix": build_confusion_matrix(categories, results),
        "failed_examples": failed_examples,
        "signature_mode": signature_mode,
        "metric_mode": metric_mode,
        "aggregation_mode": aggregation_mode,
        "scoring_mode": scoring_mode,
        "top_k": top_k,
        "disclaimer": COMPARISON_DISCLAIMER,
    }


def generate_failure_report(summary: dict[str, Any]) -> str:
    lines = [
        "# ASNE Evaluation Failure Report",
        "",
        COMPARISON_DISCLAIMER,
        "",
        f"Evaluation: {summary.get('eval_name')}",
        f"Signature mode: {summary.get('signature_mode')}",
        f"Metric mode: {summary.get('metric_mode')}",
        f"Aggregation mode: {summary.get('aggregation_mode')}",
        f"Scoring mode: {summary.get('scoring_mode')}",
        f"Top-k: {summary.get('top_k')}",
        "",
        "## Failed Examples",
        "",
    ]
    failures = summary.get("failed_examples", [])
    if not failures:
        lines.append("No top-1 failures.")
    for failure in failures:
        lines.extend(
            [
                f"### {failure.get('id')}",
                "",
                f"- Expected: {failure.get('expected_category')}",
                f"- Predicted top category: {failure.get('top_category')}",
                f"- Rank of expected category: {failure.get('rank_of_expected')}",
                f"- Comparison output: {failure.get('comparison_output_path')}",
            ]
        )
        input_text = failure.get("input_text")
        if input_text:
            lines.append(f"- Input: {input_text}")
        query_delta_norm = failure.get("query_delta_norm")
        if query_delta_norm is not None:
            lines.append(f"- Query delta norm: {query_delta_norm:.6f}")
        query_response_norm = failure.get("query_response_norm")
        if query_response_norm is not None:
            lines.append(f"- Query response norm: {query_response_norm:.6f}")
        binary_axis = failure.get("binary_axis") or {}
        if binary_axis:
            lines.extend(
                [
                    f"- Binary axis signed score: {binary_axis.get('signed_score'):.6f}",
                    f"- Binary axis norm: {binary_axis.get('axis_norm'):.6f}",
                    f"- Binary axis midpoint norm: {binary_axis.get('midpoint_norm'):.6f}",
                    f"- Binary axis predicted side: {binary_axis.get('predicted_side')}",
                ]
            )
        paired_vote = failure.get("paired_vote") or {}
        if paired_vote:
            lines.append(f"- Paired vote counts: {paired_vote.get('vote_counts')}")
            lines.append(f"- Paired vote tie handling: {paired_vote.get('tie_handling_rule')}")
        centroid_norms = failure.get("per_category_centroid_norm") or {}
        if centroid_norms:
            lines.append("- Category centroid norms:")
            for category, value in sorted(centroid_norms.items()):
                lines.append(f"  - {category}: {value:.6f}")
        lines.append("- Full ranking:")
        for index, item in enumerate(failure.get("per_category_ranking", []), start=1):
            cosine = item.get("centroid_cosine_similarity", item.get("best_cosine_similarity", item.get("mean_cosine_similarity")))
            pearson = item.get("centroid_pearson_correlation", item.get("best_pearson_correlation", item.get("mean_pearson_correlation")))
            best = item.get("best_stimulus_id")
            lines.append(
                f"  - {index}. {item.get('category')}: cosine={_format_score(cosine)} "
                f"pearson={_format_score(pearson)} best={best}"
            )
        lines.append("")

    predicted_failures: dict[str, int] = {}
    for failure in failures:
        category = failure.get("top_category")
        if category:
            predicted_failures[category] = predicted_failures.get(category, 0) + 1
    dominant = [
        category
        for category, count in predicted_failures.items()
        if failures and count > len(failures) / 2
    ]
    if dominant:
        lines.extend(
            [
                "## Dominance Warning",
                "",
                (
                    "One category accounts for more than half of top-1 failures: "
                    + ", ".join(sorted(dominant))
                    + ". This may indicate an attractor category in the current stimulus-response dictionary."
                ),
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _format_score(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"
