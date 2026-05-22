from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DISCLAIMER = (
    "This report analyzes predicted stimulus-response similarity margins. "
    "It is not emotion detection, diagnosis, or measurement of an individual person's mental state."
)


COSINE_KEYS = ("score", "centroid_cosine_similarity", "best_cosine_similarity", "mean_cosine_similarity")
PEARSON_KEYS = ("centroid_pearson_correlation", "best_pearson_correlation", "mean_pearson_correlation")


def load_eval_pair_map(eval_path: str | Path | None) -> dict[str, dict[str, str]]:
    if not eval_path:
        return {}
    payload = json.loads(Path(eval_path).read_text(encoding="utf-8"))
    pair_map = {}
    for item in payload.get("items", []):
        item_id = str(item.get("id") or "")
        if not item_id:
            continue
        pair_map[item_id] = {
            "pair_id": str(item.get("pair_id") or ""),
            "text": str(item.get("text") or ""),
        }
    return pair_map


def ranking_score(item: dict[str, Any], metric: str = "cosine") -> float | None:
    keys = PEARSON_KEYS if metric == "pearson" else COSINE_KEYS
    for key in keys:
        value = item.get(key)
        if value is not None:
            return float(value)
    return None


def score_gap_top1_to_expected(result: dict[str, Any], metric: str = "cosine") -> float | None:
    ranking = result.get("per_category_ranking") or []
    expected = result.get("expected_category")
    if not ranking or not expected:
        return None
    top_score = ranking_score(ranking[0], metric=metric)
    expected_score = None
    for item in ranking:
        if item.get("category") == expected:
            expected_score = ranking_score(item, metric=metric)
            break
    if top_score is None or expected_score is None:
        return None
    return top_score - expected_score


def enrich_margin_rows(
    summary: dict[str, Any],
    pair_map: dict[str, dict[str, str]] | None = None,
    metric: str = "cosine",
) -> list[dict[str, Any]]:
    pair_map = pair_map or {}
    rows = []
    for result in summary.get("results", []):
        result_id = str(result.get("id") or "")
        eval_meta = pair_map.get(result_id, {})
        ranking = result.get("per_category_ranking") or []
        expected = result.get("expected_category")
        expected_score = None
        for item in ranking:
            if item.get("category") == expected:
                expected_score = ranking_score(item, metric=metric)
                break
        top_score = ranking_score(ranking[0], metric=metric) if ranking else None
        rows.append(
            {
                "id": result_id,
                "pair_id": eval_meta.get("pair_id") or result.get("pair_id") or infer_pair_id(result_id),
                "expected_category": expected,
                "predicted_category": result.get("top_category"),
                "rank_of_expected": result.get("rank_of_expected"),
                "expected_rank2": result.get("rank_of_expected") == 2,
                "score_gap": score_gap_top1_to_expected(result, metric=metric),
                "top_score": top_score,
                "expected_score": expected_score,
                "input_text": result.get("input_text") or eval_meta.get("text") or "",
            }
        )
    return rows


def infer_pair_id(result_id: str) -> str:
    parts = result_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return f"pair_{parts[1]}"
    return ""


def average_margin_by_expected(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    failures: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    for row in rows:
        category = str(row.get("expected_category") or "")
        if not category:
            continue
        totals[category] += 1
        if row.get("predicted_category") != row.get("expected_category"):
            failures[category] += 1
        gap = row.get("score_gap")
        if gap is not None:
            grouped[category].append(float(gap))
    output = {}
    for category in sorted(totals):
        values = grouped.get(category, [])
        output[category] = {
            "total": totals[category],
            "failures": failures.get(category, 0),
            "average_gap": sum(values) / len(values) if values else 0.0,
            "max_gap": max(values) if values else 0.0,
        }
    return output


def margin_report(summary: dict[str, Any], rows: list[dict[str, Any]], metric: str = "cosine") -> str:
    aggregate = average_margin_by_expected(rows)
    lines = [
        "# ASNE Pair Margin Analysis",
        "",
        DISCLAIMER,
        "",
        f"Evaluation: {summary.get('eval_name')}",
        f"Feature space: {summary.get('feature_space', 'vertex')}",
        f"Scoring mode: {summary.get('scoring_mode')}",
        f"Metric used for margin: {metric}",
        f"Top-1 accuracy: {summary.get('correct_top1')}/{summary.get('total_examples')} ({float(summary.get('top1_accuracy') or 0.0):.2f})",
        f"Top-2 accuracy: {summary.get('correct_top2')}/{summary.get('total_examples')} ({float(summary.get('top2_accuracy') or 0.0):.2f})",
        "",
        "## Average Margin By Expected Category",
        "",
        "| expected_category | total | failures | average_top1_minus_expected | max_top1_minus_expected |",
        "|---|---:|---:|---:|---:|",
    ]
    for category, stats in aggregate.items():
        lines.append(
            f"| {category} | {stats['total']} | {stats['failures']} | "
            f"{float(stats['average_gap']):.6f} | {float(stats['max_gap']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Per-Example Margins",
            "",
            "| id | pair_id | expected | predicted | rank_expected | expected_rank2 | score_gap | input |",
            "|---|---|---|---|---:|---|---:|---|",
        ]
    )
    for row in rows:
        gap = row.get("score_gap")
        gap_text = "n/a" if gap is None else f"{float(gap):.6f}"
        text = str(row.get("input_text") or "").replace("|", "\\|")
        lines.append(
            f"| {row.get('id')} | {row.get('pair_id')} | {row.get('expected_category')} | "
            f"{row.get('predicted_category')} | {row.get('rank_of_expected')} | "
            f"{row.get('expected_rank2')} | {gap_text} | {text} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "pair_id",
        "expected_category",
        "predicted_category",
        "rank_of_expected",
        "expected_rank2",
        "score_gap",
        "top_score",
        "expected_score",
        "input_text",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paired ASNE eval margins from a summary JSON.")
    parser.add_argument("--summary", required=True, help="Path to an ASNE eval summary JSON.")
    parser.add_argument("--eval", default=None, help="Optional eval JSON for pair_id lookup.")
    parser.add_argument("--metric", choices=["cosine", "pearson"], default="cosine")
    parser.add_argument("--output", default=None, help="Markdown output path.")
    parser.add_argument("--csv-output", default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = enrich_margin_rows(summary, load_eval_pair_map(args.eval), metric=args.metric)
    output_path = Path(args.output) if args.output else summary_path.with_name(
        summary_path.stem.replace("_summary", "") + "_pair_margins.md"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(margin_report(summary, rows, metric=args.metric), encoding="utf-8")
    csv_output = Path(args.csv_output) if args.csv_output else output_path.with_suffix(".csv")
    write_csv(rows, csv_output)

    aggregate = average_margin_by_expected(rows)
    print("ASNE pair margin analysis complete.")
    print(f"Summary: {summary_path}")
    print(f"Report: {output_path}")
    print(f"CSV: {csv_output}")
    for category, stats in aggregate.items():
        print(
            f"{category}: failures={stats['failures']}/{stats['total']} "
            f"avg_gap={float(stats['average_gap']):.6f}"
        )


if __name__ == "__main__":
    main()
