from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SIMILARITY_DISCLAIMER = (
    "This report analyzes predicted stimulus-response similarity rankings. "
    "It is not emotion detection, diagnosis, or measurement of an individual person's mental state."
)


def score_gap_top1_to_expected(result: dict[str, Any]) -> float | None:
    ranking = result.get("per_category_ranking") or []
    expected = result.get("expected_category")
    if not ranking or not expected:
        return None
    top_score = ranking[0].get("score")
    expected_score = None
    for item in ranking:
        if item.get("category") == expected:
            expected_score = item.get("score")
            break
    if top_score is None or expected_score is None:
        return None
    return float(top_score) - float(expected_score)


def top2_category(result: dict[str, Any]) -> str | None:
    ranking = result.get("per_category_ranking") or []
    if len(ranking) < 2:
        return None
    return ranking[1].get("category")


def expected_often_rank2(results: list[dict[str, Any]]) -> bool:
    failures = [result for result in results if not result.get("is_top1_correct")]
    if not failures:
        return False
    rank2_count = sum(1 for result in failures if result.get("rank_of_expected") == 2)
    return rank2_count >= max(2, len(failures) / 2)


def top2_high_top1_low(summary: dict[str, Any]) -> bool:
    top1 = float(summary.get("top1_accuracy") or 0.0)
    top2 = float(summary.get("top2_accuracy") or 0.0)
    return top2 >= 0.75 and (top2 - top1) >= 0.30


def dominant_attractor_categories(results: list[dict[str, Any]], min_share: float = 0.30) -> list[dict[str, Any]]:
    failures = [result for result in results if not result.get("is_top1_correct")]
    counts = Counter(result.get("top_category") for result in failures if result.get("top_category"))
    if not failures:
        return []
    threshold = max(2, int(len(failures) * min_share + 0.999))
    return [
        {"category": category, "count": count, "share": count / len(failures)}
        for category, count in counts.most_common()
        if count >= threshold
    ]


def grouped_failures(
    failures: list[dict[str, Any]],
    group_key: str,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for failure in failures:
        grouped[str(failure.get(group_key) or "<none>")].append(failure)
    return dict(sorted(grouped.items()))


def confusion_matrix_markdown(matrix: dict[str, dict[str, int]]) -> list[str]:
    categories = list(matrix)
    for row in matrix.values():
        for category in row:
            if category not in categories:
                categories.append(category)
    lines = ["| expected \\ predicted | " + " | ".join(categories) + " |"]
    lines.append("| --- | " + " | ".join("---" for _ in categories) + " |")
    for expected in categories:
        row = matrix.get(expected, {})
        lines.append("| " + expected + " | " + " | ".join(str(row.get(category, 0)) for category in categories) + " |")
    return lines


def ranking_lines(ranking: list[dict[str, Any]]) -> list[str]:
    lines = []
    for index, item in enumerate(ranking, start=1):
        score = item.get("score")
        score_text = "n/a" if score is None else f"{float(score):.4f}"
        best = item.get("best_stimulus_id")
        lines.append(f"  - {index}. {item.get('category')}: score={score_text} best={best}")
    return lines


def generate_diagnostic_report(summary: dict[str, Any]) -> str:
    results = summary.get("results") or []
    failures = [result for result in results if not result.get("is_top1_correct")]
    rank2 = expected_often_rank2(results)
    attractors = dominant_attractor_categories(results)

    lines = [
        "# ASNE Evaluation Diagnostics",
        "",
        SIMILARITY_DISCLAIMER,
        "",
        "## Overall",
        "",
        f"- Evaluation: {summary.get('eval_name')}",
        f"- Signature mode: {summary.get('signature_mode')}",
        f"- Aggregation mode: {summary.get('aggregation_mode')}",
        f"- Scoring mode: {summary.get('scoring_mode')}",
        f"- Top-1 accuracy: {summary.get('correct_top1')}/{summary.get('total_examples')} ({float(summary.get('top1_accuracy') or 0.0):.2f})",
        f"- Top-2 accuracy: {summary.get('correct_top2')}/{summary.get('total_examples')} ({float(summary.get('top2_accuracy') or 0.0):.2f})",
        f"- Mean rank of expected category: {float(summary.get('mean_rank_expected') or 0.0):.2f}",
        "",
        "## Confusion Matrix",
        "",
    ]
    lines.extend(confusion_matrix_markdown(summary.get("confusion_matrix") or {}))
    lines.extend(["", "## Label-Structure Notes", ""])
    if top2_high_top1_low(summary):
        lines.append("- Top-2 accuracy is high while top-1 accuracy is low; the signal appears better suited to ranking analysis than strict 5-way top-1 classification.")
    if rank2:
        lines.append("- The expected category is often ranked second among top-1 failures.")
    if attractors:
        attractor_text = ", ".join(f"{item['category']} ({item['count']})" for item in attractors)
        lines.append(f"- Dominant attractor categories among top-1 failures: {attractor_text}.")
    else:
        lines.append("- No single predicted category dominates failures; labels appear to swap across several categories.")
    if top2_high_top1_low(summary) and not attractors:
        lines.append("- Categories may be too abstract or semantically adjacent for the current dictionary size.")

    lines.extend(["", "## Failed Examples by Expected Category", ""])
    for category, items in grouped_failures(failures, "expected_category").items():
        lines.append(f"### {category}")
        lines.append("")
        for item in items:
            append_failure_detail(lines, item)

    lines.extend(["", "## Failed Examples by Predicted Top Category", ""])
    for category, items in grouped_failures(failures, "top_category").items():
        lines.append(f"### {category}")
        lines.append("")
        for item in items:
            lines.append(f"- {item.get('id')} expected={item.get('expected_category')} rank_expected={item.get('rank_of_expected')}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def append_failure_detail(lines: list[str], item: dict[str, Any]) -> None:
    ranking = item.get("per_category_ranking") or []
    gap = score_gap_top1_to_expected(item)
    gap_text = "n/a" if gap is None else f"{gap:.4f}"
    lines.extend(
        [
            f"#### {item.get('id')}",
            "",
            f"- Input: {item.get('input_text')}",
            f"- Expected category: {item.get('expected_category')}",
            f"- Top-1 category: {item.get('top_category')}",
            f"- Top-2 category: {top2_category(item)}",
            f"- Rank of expected: {item.get('rank_of_expected')}",
            f"- Score gap top1-minus-expected: {gap_text}",
            "- Full ranking:",
        ]
    )
    lines.extend(ranking_lines(ranking))
    lines.append("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate diagnostics from an ASNE evaluation summary JSON.")
    parser.add_argument("--summary", required=True, help="Path to an ASNE evaluation summary JSON.")
    parser.add_argument(
        "--output",
        default=None,
        help="Path for the markdown diagnostics report. Defaults to '<summary_stem>_diagnostics.md'.",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    output_path = Path(args.output) if args.output else summary_path.with_name(
        summary_path.stem.replace("_summary", "") + "_diagnostics.md"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report = generate_diagnostic_report(summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print("ASNE diagnostics complete.")
    print(f"Summary: {summary_path}")
    print(f"Output: {output_path}")
    print(f"top1={summary.get('correct_top1')}/{summary.get('total_examples')} top2={summary.get('correct_top2')}/{summary.get('total_examples')} mean_rank={summary.get('mean_rank_expected')}")
    attractors = dominant_attractor_categories(summary.get("results") or [])
    if attractors:
        print("Attractors: " + ", ".join(f"{item['category']}={item['count']}" for item in attractors))
    else:
        print("Attractors: none dominant")


if __name__ == "__main__":
    main()
