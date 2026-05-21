from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_contrast_summaries(paths: list[str]) -> list[dict[str, Any]]:
    summaries = []
    for raw_path in paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_summary_path"] = str(path)
        summaries.append(payload)
    return summaries


def is_separable(summary: dict[str, Any], threshold: float = 0.75) -> bool:
    return float(summary.get("top1_accuracy") or 0.0) >= threshold


def summarize_contrasts(summaries: list[dict[str, Any]], threshold: float = 0.75) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        contrast_name = summary.get("contrast_name") or summary.get("eval_name") or Path(summary["_summary_path"]).parent.name
        rows.append(
            {
                "contrast_name": contrast_name,
                "summary_path": summary["_summary_path"],
                "top1_accuracy": float(summary.get("top1_accuracy") or 0.0),
                "top2_accuracy": float(summary.get("top2_accuracy") or 0.0),
                "correct_top1": int(summary.get("correct_top1") or 0),
                "total_examples": int(summary.get("total_examples") or 0),
                "mean_rank_expected": summary.get("mean_rank_expected"),
                "confusion_matrix": summary.get("confusion_matrix") or {},
                "separable": is_separable(summary, threshold=threshold),
            }
        )
    return sorted(rows, key=lambda row: row["contrast_name"])


def format_summary_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| contrast | top1 | top2 | mean_rank | separable |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        total = row["total_examples"]
        top1 = f"{row['correct_top1']}/{total} ({row['top1_accuracy']:.2f})"
        top2 = f"{row['top2_accuracy']:.2f}"
        mean_rank = "n/a" if row["mean_rank_expected"] is None else f"{float(row['mean_rank_expected']):.2f}"
        lines.append(
            f"| {row['contrast_name']} | {top1} | {top2} | {mean_rank} | {row['separable']} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize ASNE binary contrast evaluation summaries.")
    parser.add_argument("--summaries", nargs="+", required=True, help="One or more ASNE eval summary JSON paths.")
    parser.add_argument("--output", default=None, help="Optional markdown output path.")
    parser.add_argument("--threshold", type=float, default=0.75, help="Top-1 accuracy threshold for separability.")
    args = parser.parse_args()

    rows = summarize_contrasts(load_contrast_summaries(args.summaries), threshold=args.threshold)
    table = format_summary_table(rows)
    lines = [
        "# ASNE Contrast Summary",
        "",
        "This report summarizes predicted stimulus-response similarity for controlled binary contrasts. It is not emotion detection, diagnosis, or measurement of an individual person's mental state.",
        "",
        table,
        "",
        "## Confusion Matrices",
        "",
    ]
    for row in rows:
        lines.append(f"### {row['contrast_name']}")
        lines.append("")
        lines.append(json.dumps(row["confusion_matrix"], indent=2))
        lines.append("")
    report = "\n".join(lines).rstrip() + "\n"
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
    print(table)
    if args.output:
        print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
