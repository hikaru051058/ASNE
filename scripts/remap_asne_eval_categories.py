from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


BROAD_AXIS_MAP = {
    "neutral": "ordinary",
    "threat": "negative_context",
    "sadness": "negative_context",
    "confusion": "uncertainty",
    "relief": "resolution",
}

REMAP_DISCLAIMER = (
    "This remapping analyzes predicted stimulus-response similarity under broader labels. "
    "It is not emotion detection, diagnosis, or measurement of an individual person's mental state."
)


def map_category(category: str | None, mapping: dict[str, str] | None = None) -> str | None:
    if category is None:
        return None
    category_map = mapping or BROAD_AXIS_MAP
    return category_map.get(category, category)


def collapse_ranking_to_groups(
    ranking: list[dict[str, Any]],
    mapping: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    seen = set()
    collapsed = []
    for item in ranking:
        group = map_category(item.get("category"), mapping)
        if group is None or group in seen:
            continue
        seen.add(group)
        collapsed.append(
            {
                "group": group,
                "source_category": item.get("category"),
                "score": item.get("score"),
                "best_stimulus_id": item.get("best_stimulus_id"),
            }
        )
    return collapsed


def compute_grouped_accuracy(
    summary: dict[str, Any],
    mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    results = summary.get("results") or []
    grouped_results = []
    correct_top1 = 0
    correct_top2 = 0
    per_group: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for result in results:
        expected_group = map_category(result.get("expected_category"), mapping)
        grouped_ranking = collapse_ranking_to_groups(result.get("per_category_ranking") or [], mapping)
        top_group = grouped_ranking[0]["group"] if grouped_ranking else None
        top2_groups = [item["group"] for item in grouped_ranking[:2]]
        is_top1 = expected_group == top_group
        is_top2 = expected_group in top2_groups
        correct_top1 += int(is_top1)
        correct_top2 += int(is_top2)
        if expected_group is not None:
            per_group[expected_group]["total"] += 1
            per_group[expected_group]["correct"] += int(is_top1)
            confusion[expected_group][top_group or "<none>"] += 1
        grouped_results.append(
            {
                "id": result.get("id"),
                "input_text": result.get("input_text"),
                "expected_category": result.get("expected_category"),
                "expected_group": expected_group,
                "top_category": result.get("top_category"),
                "top_group": top_group,
                "is_group_top1_correct": is_top1,
                "is_group_top2_correct": is_top2,
                "grouped_ranking": grouped_ranking,
            }
        )

    total = len(results)
    per_group_accuracy = {
        group: {
            "correct": counts["correct"],
            "total": counts["total"],
            "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0.0,
        }
        for group, counts in sorted(per_group.items())
    }
    normalized_confusion = {
        expected: dict(sorted(predicted.items()))
        for expected, predicted in sorted(confusion.items())
    }
    return {
        "eval_name": summary.get("eval_name"),
        "source_summary": summary.get("summary_path"),
        "grouping": "broad_axes_v0",
        "category_map": mapping or BROAD_AXIS_MAP,
        "total_examples": total,
        "correct_grouped_top1": correct_top1,
        "grouped_top1_accuracy": correct_top1 / total if total else 0.0,
        "correct_grouped_top2": correct_top2,
        "grouped_top2_accuracy": correct_top2 / total if total else 0.0,
        "per_group_accuracy": per_group_accuracy,
        "grouped_confusion_matrix": normalized_confusion,
        "results": grouped_results,
        "disclaimer": REMAP_DISCLAIMER,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute broad-axis grouped ASNE evaluation accuracy from saved rankings.")
    parser.add_argument("--summary", required=True, help="Path to an ASNE evaluation summary JSON.")
    parser.add_argument("--output", required=True, help="Path for grouped remap JSON output.")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    output_path = Path(args.output)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["summary_path"] = str(summary_path)
    result = compute_grouped_accuracy(summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("ASNE category remap complete.")
    print(f"Summary: {summary_path}")
    print(f"Output: {output_path}")
    print(
        "grouped_top1="
        f"{result['correct_grouped_top1']}/{result['total_examples']} "
        f"({result['grouped_top1_accuracy']:.2f})"
    )
    print(
        "grouped_top2="
        f"{result['correct_grouped_top2']}/{result['total_examples']} "
        f"({result['grouped_top2_accuracy']:.2f})"
    )


if __name__ == "__main__":
    main()
