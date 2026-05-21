#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from asne.notebook_utils import temporal_report_payload


def _format_float_list(values: list[float], precision: int = 4) -> str:
    return ", ".join(f"{value:.{precision}f}" for value in values)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    similarity = payload["segment_similarity_timeline"]
    categories = similarity["categories"]
    lines = [
        "# ASNE Temporal Response Diagnostic",
        "",
        "This report describes predicted response movement across TRIBE v2 output segments. "
        "It is a diagnostic view of predicted response similarity, not emotion detection, "
        "diagnosis, or measurement of an individual person's mental state.",
        "",
        f"- Stimulus output: `{payload['stimulus_output']}`",
        f"- Dictionary index: `{payload['dictionary_index']}`",
        f"- Raw shape: `{payload['raw_shape']}`",
        f"- Segment count: `{payload['segment_count']}`",
        f"- Late category shift detected: `{payload['late_shift_detected']}`",
        "",
        "## Response Norm Per Segment",
        "",
        _format_float_list(payload["response_norm_timeline"]),
        "",
        "## Transition Movement Norm Per Segment",
        "",
        _format_float_list(payload["transition_norm_timeline"]) or "No transitions; only one segment.",
        "",
        "## Segment-Level Category Similarity",
        "",
        "| segment | top_category | " + " | ".join(categories) + " |",
        "|---:|---|" + "---:|" * len(categories),
    ]
    top_categories = payload["top_categories_by_segment"]
    for segment, row in zip(similarity["segments"], similarity["similarities"], strict=False):
        top_category = top_categories[segment] if segment < len(top_categories) else ""
        lines.append(
            f"| {segment} | {top_category} | "
            + " | ".join(f"{value:.4f}" for value in row)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Top Moving Dimensions By Net Movement",
            "",
        ]
    )
    for item in payload["top_moving_dimensions"]["net_movement"]:
        lines.append(f"- index {item['index']}: {item['value']:.6f}")

    lines.extend(
        [
            "",
            "## Top Moving Dimensions By Absolute Total Movement",
            "",
        ]
    )
    for item in payload["top_moving_dimensions"]["abs_total_movement"]:
        lines.append(f"- index {item['index']}: {item['value']:.6f}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_temporal_response(args: argparse.Namespace) -> dict[str, Any]:
    payload = temporal_report_payload(
        args.stimulus_output,
        args.dictionary,
        metric=args.metric,
        top_k=args.top_k,
    )
    output = Path(args.output) if args.output else _default_output_path(args.stimulus_output)
    _write_report(output, payload)
    return {"output_path": str(output), "payload": payload}


def _default_output_path(stimulus_output: str | Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = Path(stimulus_output).stem
    return Path("outputs/asne_temporal_reports") / f"{timestamp}_{stem}.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze ASNE segment-level temporal response diagnostics.")
    parser.add_argument("--stimulus-output", required=True, help="Path to a dictionary/eval stimulus output JSON.")
    parser.add_argument("--dictionary", required=True, help="Path to the dictionary_index.json used for category centroids.")
    parser.add_argument("--output", default=None, help="Markdown output path. Defaults under outputs/asne_temporal_reports/.")
    parser.add_argument("--metric", choices=["cosine", "pearson"], default="cosine")
    parser.add_argument("--top-k", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = analyze_temporal_response(args)
    payload = result["payload"]
    print("ASNE temporal response diagnostic complete.")
    print(f"Output: {result['output_path']}")
    print(f"Raw shape: {payload['raw_shape']}")
    print(f"Segment count: {payload['segment_count']}")
    print(f"Late category shift detected: {payload['late_shift_detected']}")


if __name__ == "__main__":
    main()
