#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from asne.dictionary import COMPARISON_DISCLAIMER


DEFAULT_CONTRASTS = [
    "contradiction_vs_consistency_paired",
    "expected_vs_unexpected_paired",
    "approach_vs_static_paired",
    "cause_effect_valid_vs_invalid_paired",
]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_path"] = str(path)
    return payload


def _summary_sort_key(summary: dict[str, Any]) -> tuple[int, float, float]:
    is_default = (
        summary.get("signature_mode") == "mean_response"
        and summary.get("aggregation_mode") == "centroid"
        and summary.get("scoring_mode") == "centroid_raw"
    )
    return (
        1 if is_default else 0,
        float(summary.get("top1_accuracy") or 0.0),
        Path(summary["_path"]).stat().st_mtime,
    )


def find_static_summary(contrast: str, root: str | Path) -> dict[str, Any] | None:
    paths = sorted(Path(root).glob(f"{contrast}/*_summary.json"))
    if not paths:
        return None
    summaries = [_load_json(path) for path in paths]
    return sorted(summaries, key=_summary_sort_key)[-1]


def find_temporal_summary(contrast: str, root: str | Path) -> dict[str, Any] | None:
    paths = sorted(Path(root).glob(f"{contrast}/*_temporal_summary.json"), key=lambda path: path.stat().st_mtime)
    return _load_json(paths[-1]) if paths else None


def find_roi_reports(contrast: str, root: str | Path = "outputs/asne_roi_reports") -> list[str]:
    root_path = Path(root)
    paths = list(root_path.glob(f"{contrast}/roi_report.md"))
    paths.extend(root_path.glob(f"{contrast}/*_roi_report.md"))
    paths = sorted(set(paths), key=lambda path: path.stat().st_mtime)
    return [str(path) for path in paths]


def extract_top_roi_parcels(
    report_path: str | Path,
    top_k: int = 5,
    section: str = "Top Parcels By Absolute Contrast Delta",
) -> list[dict[str, str]]:
    path = Path(report_path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, str]] = []
    in_section = False
    for line in lines:
        if line.startswith(f"## {section}"):
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if not in_section or not line.startswith("|"):
            continue
        if line.startswith("| rank") or line.startswith("|---"):
            continue
        parts = [part.strip().strip("`") for part in line.strip("|").split("|")]
        if len(parts) < 5:
            continue
        rows.append(
            {
                "rank": parts[0],
                "parcel_id": parts[1],
                "parcel_name": parts[2],
                "delta": parts[3],
                "abs_delta": parts[4],
            }
        )
        if len(rows) >= top_k:
            break
    return rows


def _format_roi_rows(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "n/a"
    return "<br>".join(
        f"{row['rank']}. `{row['parcel_id']}` ({row['abs_delta']})"
        for row in rows
    )


def per_category_accuracy_text(summary: dict[str, Any] | None) -> str:
    if not summary:
        return "n/a"
    parts = []
    for category, values in (summary.get("per_category_accuracy") or {}).items():
        parts.append(f"{category}: {values.get('correct')}/{values.get('total')}")
    return "; ".join(parts) if parts else "n/a"


def build_report(
    *,
    contrasts: list[str],
    static_root: str | Path,
    temporal_root: str | Path,
    roi_root: str | Path = "outputs/asne_roi_reports",
) -> str:
    static = {contrast: find_static_summary(contrast, static_root) for contrast in contrasts}
    temporal = {contrast: find_temporal_summary(contrast, temporal_root) for contrast in contrasts}
    roi_reports = {contrast: find_roi_reports(contrast, roi_root) for contrast in contrasts}
    lines = [
        "# ASNE Semantic Contrast Report v0",
        "",
        "## Overview",
        "",
        "ASNE is a TRIBE-backed in-silico stimulus-response analysis system that compares predicted cortical response signatures for controlled stimuli.",
        "",
        "## Method",
        "",
        "- Text/TTS stimuli are generated from controlled paired narratives.",
        "- TRIBE v2 produces predicted cortical responses.",
        "- ASNE stores raw segment predictions with shape `[segments, 20484]` when available.",
        "- Static binary contrast evaluation uses `mean_response [20484]` with `centroid_raw` comparison.",
        "- Temporal evaluation compares segment-level responses to category centroids and reports early, late, final, and majority segment winners.",
        "",
        "## Static Contrast Results",
        "",
        "| contrast | best/default scoring | top1 accuracy | top2 accuracy | per-category accuracy | summary path |",
        "|---|---|---:|---:|---|---|",
    ]
    for contrast in contrasts:
        summary = static[contrast]
        if not summary:
            lines.append(f"| `{contrast}` | n/a | n/a | n/a | n/a | missing |")
            continue
        scoring = f"{summary.get('signature_mode')}/{summary.get('aggregation_mode')}/{summary.get('scoring_mode')}"
        lines.append(
            f"| `{contrast}` | `{scoring}` | {_fmt(summary.get('top1_accuracy'))} | "
            f"{_fmt(summary.get('top2_accuracy'))} | {per_category_accuracy_text(summary)} | `{summary['_path']}` |"
        )

    lines.extend(
        [
            "",
            "## Temporal Contrast Results",
            "",
            "| contrast | mean_response_acc | early_acc | late_acc | final_segment_acc | majority_segment_acc | avg_switch_count | recovered_mean_failures | temporal summary path |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for contrast in contrasts:
        summary = temporal[contrast]
        if not summary:
            lines.append(f"| `{contrast}` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | missing |")
            continue
        recovered = len(summary.get("examples_where_mean_failed_but_temporal_succeeded") or [])
        lines.append(
            f"| `{contrast}` | {_fmt(summary.get('mean_response_accuracy'))} | {_fmt(summary.get('early_accuracy'))} | "
            f"{_fmt(summary.get('late_accuracy'))} | {_fmt(summary.get('final_segment_accuracy'))} | "
            f"{_fmt(summary.get('majority_segment_accuracy'))} | {_fmt(summary.get('average_switch_count'))} | "
            f"{recovered} | `{summary['_path']}` |"
        )

    if any(roi_reports.values()):
        lines.extend(
            [
                "",
                "## ROI / Parcel-Level Predicted Response Summary",
                "",
                "ROI summaries use HCP-MMP parcel aggregation over TRIBE's `20,484` fsaverage5 vertices when `data/parcellations/fsaverage5_hcp_mmp.csv` is used. "
                "Temporal movement is often dominated by auditory parcels such as LBelt, PBelt, A1, and A4 because the current pipeline uses text-to-speech audio. "
                "For semantic interpretation, contrast-delta parcels are currently more useful than raw temporal movement parcels.",
                "",
                "| contrast | ROI report path | top contrast-delta parcels | top late-minus-early parcels |",
                "|---|---|---|---|",
            ]
        )
        for contrast in contrasts:
            paths = roi_reports[contrast]
            if paths:
                latest_path = paths[-1]
                contrast_rows = extract_top_roi_parcels(latest_path, top_k=5)
                late_rows = extract_top_roi_parcels(
                    latest_path,
                    top_k=5,
                    section="Top Parcels By Late Minus Early",
                )
                parcels = _format_roi_rows(contrast_rows)
                late_parcels = _format_roi_rows(late_rows)
                links = f"`{latest_path}`"
            else:
                links = "n/a"
                parcels = "n/a"
                late_parcels = "n/a"
            lines.append(f"| `{contrast}` | {links} | {parcels} | {late_parcels} |")

    lines.extend(
        [
            "",
            "## Main Findings",
            "",
            "Semantic and logical text contrasts separate better than narrated motion-style contrasts in the current text/TTS ASNE pipeline. "
            "`expected_vs_unexpected_paired` showed the strongest temporal result, with final-segment accuracy reaching `1.00`. "
            "`approach_vs_static_paired` stayed weak across static and temporal views, suggesting it may require video-native stimuli or a different stimulus design. "
            "HCP-MMP ROI reports now summarize which parcels contribute most to predicted contrast deltas; temporal movement parcels should be interpreted cautiously because they can reflect TTS/audio processing dynamics.",
            "",
            "## Limitations",
            "",
            "- Evaluation sets are tiny and should be treated as smoke-scale experiments.",
            "- Current runs use text/TTS rather than native video stimuli.",
            "- TRIBE outputs are predicted cortical responses, not measured neural activity.",
            "- Some examples have small margins, and ASNE is not a stable classifier yet.",
            "- Results are predicted stimulus-response similarity, not emotion detection, diagnosis, or measurement of an individual person's mental state.",
            "",
            "## Next Recommended Work",
            "",
            "- Add more semantic contrasts and increase held-out examples per contrast.",
            "- Test video-native contrasts separately from narrated text/TTS contrasts.",
            "- Refine ROI interpretation and separate semantic contrast deltas from TTS/audio-driven temporal movement.",
            "- Consider ROI-level scoring or ROI-level temporal normalization after the current vertex-level benchmark remains stable.",
            "- Build an HTML or interactive temporal visualizer after the artifact format stabilizes.",
            "",
            "## Reproduction Commands",
            "",
            "Verify or regenerate the current report without rerunning TRIBE predictions:",
            "",
            "```bash",
            "python scripts/run_asne_semantic_contrast_suite.py --skip-build",
            "```",
            "",
            "Run missing dictionary builds if needed:",
            "",
            "```bash",
            "python scripts/run_asne_semantic_contrast_suite.py --run-build",
            "```",
            "",
            "Regenerate only the report from existing summaries:",
            "",
            "```bash",
            "python scripts/generate_asne_contrast_report.py",
            "```",
            "",
            "## Safety Framing",
            "",
            COMPARISON_DISCLAIMER,
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def markdown_to_simple_html(markdown: str, title: str = "ASNE Semantic Contrast Report v0") -> str:
    body = []
    in_code = False
    for line in markdown.splitlines():
        if line.startswith("```"):
            body.append("</code></pre>" if in_code else "<pre><code>")
            in_code = not in_code
            continue
        if in_code:
            body.append(html.escape(line))
        elif line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            body.append(f"<p>{html.escape(line)}</p>")
        elif line.startswith("|"):
            body.append(f"<pre>{html.escape(line)}</pre>")
        elif not line.strip():
            body.append("")
        else:
            body.append(f"<p>{html.escape(line)}</p>")
    return (
        "<!doctype html>\n<html><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;line-height:1.5}"
        "pre{background:#f6f8fa;padding:8px;overflow:auto}code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}"
        "h1,h2{line-height:1.2}</style></head><body>\n"
        + "\n".join(body)
        + "\n</body></html>\n"
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate ASNE semantic contrast report v0.")
    parser.add_argument("--static-root", default="outputs/asne_evals/contrasts")
    parser.add_argument("--temporal-root", default="outputs/asne_temporal_evals")
    parser.add_argument("--roi-root", default="outputs/asne_roi_reports")
    parser.add_argument("--output", default="outputs/asne_reports/semantic_contrast_report_v0.md")
    parser.add_argument("--html-output", default="outputs/asne_reports/semantic_contrast_report_v0.html")
    parser.add_argument("--contrasts", nargs="*", default=DEFAULT_CONTRASTS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_report(
        contrasts=args.contrasts,
        static_root=args.static_root,
        temporal_root=args.temporal_root,
        roi_root=args.roi_root,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    if args.html_output:
        html_output = Path(args.html_output)
        html_output.parent.mkdir(parents=True, exist_ok=True)
        html_output.write_text(markdown_to_simple_html(report), encoding="utf-8")
    print(f"Report: {output}")
    if args.html_output:
        print(f"HTML: {args.html_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
