#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from asne.dictionary import (
    COMPARISON_DISCLAIMER,
    load_stimulus_dictionary,
    validate_paired_contrast_benchmark,
)
from asne.evaluation import load_eval_set
from asne.dictionary import slugify


BENCHMARK_NAME = "semantic_contrast_v0_3_lite"
STATIC_ROOT = Path("outputs/asne_evals/contrasts_v03")
TEMPORAL_ROOT = Path("outputs/asne_temporal_evals/contrasts_v03")


@dataclass(frozen=True)
class V03ContrastSpec:
    name: str
    dictionary_json: str
    eval_json: str
    dictionary_output: str

    @property
    def dictionary_index(self) -> str:
        return f"outputs/asne_dictionaries/{self.dictionary_output}/dictionary_index.json"


V03_CONTRASTS = [
    V03ContrastSpec(
        name="contradiction_vs_consistency_paired",
        dictionary_json="data/stimuli/dictionaries/contrasts_v03/contradiction_vs_consistency_paired.json",
        eval_json="data/stimuli/evals/contrasts_v03/contradiction_vs_consistency_paired_eval.json",
        dictionary_output="semantic_contrast_v0_3_lite_contradiction_vs_consistency_paired_tts_macos_say_samantha_180",
    ),
    V03ContrastSpec(
        name="expected_vs_unexpected_paired",
        dictionary_json="data/stimuli/dictionaries/contrasts_v03/expected_vs_unexpected_paired.json",
        eval_json="data/stimuli/evals/contrasts_v03/expected_vs_unexpected_paired_eval.json",
        dictionary_output="semantic_contrast_v0_3_lite_expected_vs_unexpected_paired_tts_macos_say_samantha_180",
    ),
    V03ContrastSpec(
        name="cause_effect_valid_vs_invalid_paired",
        dictionary_json="data/stimuli/dictionaries/contrasts_v03/cause_effect_valid_vs_invalid_paired.json",
        eval_json="data/stimuli/evals/contrasts_v03/cause_effect_valid_vs_invalid_paired_eval.json",
        dictionary_output="semantic_contrast_v0_3_lite_cause_effect_valid_vs_invalid_paired_tts_macos_say_samantha_180",
    ),
    V03ContrastSpec(
        name="approach_vs_static_paired",
        dictionary_json="data/stimuli/dictionaries/contrasts_v03/approach_vs_static_paired.json",
        eval_json="data/stimuli/evals/contrasts_v03/approach_vs_static_paired_eval.json",
        dictionary_output="semantic_contrast_v0_3_lite_approach_vs_static_paired_tts_macos_say_samantha_180",
    ),
]

V02_BASELINE = {
    "contradiction_vs_consistency_paired": 0.83,
    "expected_vs_unexpected_paired": 0.83,
    "cause_effect_valid_vs_invalid_paired": 0.83,
    "approach_vs_static_paired": 0.50,
}


def select_contrasts(names: list[str] | None, specs: list[V03ContrastSpec] = V03_CONTRASTS) -> list[V03ContrastSpec]:
    if not names:
        return list(specs)
    by_name = {spec.name: spec for spec in specs}
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise ValueError(f"Unknown v0.3 contrast(s): {', '.join(unknown)}")
    return [by_name[name] for name in names]


def validate_v03_inputs(specs: list[V03ContrastSpec] = V03_CONTRASTS) -> list[str]:
    warnings: list[str] = []
    for spec in specs:
        dictionary = load_stimulus_dictionary(spec.dictionary_json)
        eval_set = load_eval_set(spec.eval_json)
        warnings.extend(
            validate_paired_contrast_benchmark(
                dictionary,
                eval_set,
                expected_dictionary_pairs=10,
                expected_eval_pairs=5,
                banned_terms=_contrast_banned_terms(spec.name),
            )
        )
    return warnings


def latest_summary(
    contrast_name: str,
    *,
    feature_space: str,
    root: str | Path = STATIC_ROOT,
    eval_json: str | None = None,
    dictionary_index: str | None = None,
) -> dict[str, Any] | None:
    summaries = []
    for path in sorted(Path(root).glob(f"{contrast_name}/*_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload_feature_space = payload.get("feature_space") or "vertex"
        if (
            payload_feature_space == feature_space
            and payload.get("signature_mode") == "mean_response"
            and payload.get("aggregation_mode") == "centroid"
            and payload.get("scoring_mode") == "centroid_raw"
        ):
            if eval_json and payload.get("eval_path") != eval_json:
                continue
            if dictionary_index and payload.get("dictionary_index_path") != dictionary_index:
                continue
            payload["_path"] = str(path)
            summaries.append(payload)
    return summaries[-1] if summaries else None


def latest_temporal_summary(
    contrast_name: str,
    root: str | Path = TEMPORAL_ROOT,
    *,
    eval_json: str | None = None,
    dictionary_index: str | None = None,
) -> dict[str, Any] | None:
    paths = sorted(Path(root).glob(f"{contrast_name}/*_temporal_summary.json"), key=lambda path: path.stat().st_mtime)
    legacy_root = Path("outputs/asne_temporal_evals")
    paths.extend(
        sorted(legacy_root.glob(f"{contrast_name}/*_temporal_summary.json"), key=lambda path: path.stat().st_mtime)
    )
    paths = sorted(set(paths), key=lambda path: path.stat().st_mtime)
    if not paths:
        return None
    candidates = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if eval_json and payload.get("eval_path") != eval_json:
            continue
        if dictionary_index and payload.get("dictionary_index_path") != dictionary_index:
            continue
        payload["_path"] = str(path)
        candidates.append(payload)
    return candidates[-1] if candidates else None


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    specs = select_contrasts(args.contrasts)
    warnings = validate_v03_inputs(specs)
    if warnings:
        print("Validation warnings:")
        for warning in warnings:
            print(f"- {warning}")
        if not args.allow_validation_warnings:
            raise ValueError("v0.3 benchmark validation produced warnings; pass --allow-validation-warnings to continue.")
    else:
        print("Validation OK: v0.3 paired benchmark files passed structural checks.")

    rows = []
    for spec in specs:
        print(f"\n== {spec.name} ==")
        dictionary_index = Path(spec.dictionary_index)
        if dictionary_index.exists():
            print(f"dictionary index exists: {dictionary_index}")
        elif args.skip_build:
            print(f"missing dictionary index, skip-build set: {dictionary_index}")
        else:
            _run_or_print(_build_dictionary_command(spec, args), dry_run=args.dry_run)

        vertex_summary = None if args.force_eval else latest_summary(
            spec.name,
            feature_space="vertex",
            eval_json=spec.eval_json,
            dictionary_index=spec.dictionary_index,
        )
        if vertex_summary:
            print(f"vertex summary exists: {vertex_summary['_path']}")
        elif dictionary_index.exists() or args.dry_run:
            _run_or_print(_eval_command(spec, args, feature_space="vertex"), dry_run=args.dry_run)
            vertex_summary = latest_summary(
                spec.name,
                feature_space="vertex",
                eval_json=spec.eval_json,
                dictionary_index=spec.dictionary_index,
            )

        parcel_summary = None if args.force_eval else latest_summary(
            spec.name,
            feature_space="parcel",
            eval_json=spec.eval_json,
            dictionary_index=spec.dictionary_index,
        )
        if parcel_summary:
            print(f"parcel summary exists: {parcel_summary['_path']}")
        elif dictionary_index.exists() or args.dry_run:
            _run_or_print(_eval_command(spec, args, feature_space="parcel"), dry_run=args.dry_run)
            parcel_summary = latest_summary(
                spec.name,
                feature_space="parcel",
                eval_json=spec.eval_json,
                dictionary_index=spec.dictionary_index,
            )

        temporal_summary = None if args.force_temporal else latest_temporal_summary(
            spec.name,
            eval_json=spec.eval_json,
            dictionary_index=spec.dictionary_index,
        )
        if temporal_summary:
            print(f"temporal summary exists: {temporal_summary['_path']}")
        elif vertex_summary:
            temporal_output = TEMPORAL_ROOT / spec.name / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_temporal_summary.json"
            _run_or_print(_temporal_command(spec, vertex_summary["_path"], temporal_output), dry_run=args.dry_run)
            temporal_summary = latest_temporal_summary(
                spec.name,
                eval_json=spec.eval_json,
                dictionary_index=spec.dictionary_index,
            )

        rows.append(_row_from_summaries(spec, vertex_summary, parcel_summary, temporal_summary))

    report = format_report(rows, warnings)
    output = Path(args.output)
    html_output = Path(args.html_output)
    if not args.dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        html_output.parent.mkdir(parents=True, exist_ok=True)
        html_output.write_text(markdown_to_simple_html(report), encoding="utf-8")
        print(f"\nReport: {output}")
        print(f"HTML: {html_output}")
    return {"rows": rows, "warnings": warnings, "report": report}


def format_report(rows: list[dict[str, Any]], warnings: list[str] | None = None) -> str:
    warnings = warnings or []
    lines = [
        "# ASNE Semantic Contrast Benchmark v0.3-lite",
        "",
        COMPARISON_DISCLAIMER,
        "",
        "This benchmark expands the frozen semantic contrast suite from tiny six-example evals to ten held-out examples per contrast. "
        "It is still small and should be treated as a stability check, not a definitive accuracy claim.",
        "",
        "## Static Vertex vs Parcel Results",
        "",
        "| contrast | classification | v0.2 top1 | vertex top1 | parcel top1 | vertex top2 | parcel top2 | vertex mean rank | parcel mean rank | mean score margin |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['contrast']}` | {row['classification']} | {_fmt(row['v02_top1'])} | {_fmt(row['vertex_top1'])} | {_fmt(row['parcel_top1'])} | "
            f"{_fmt(row['vertex_top2'])} | {_fmt(row['parcel_top2'])} | {_fmt(row['vertex_mean_rank'])} | "
            f"{_fmt(row['parcel_mean_rank'])} | {_fmt(row['mean_score_margin'])} |"
        )
    lines.extend(
        [
            "",
            "## Temporal Results",
            "",
            "| contrast | late accuracy | final accuracy | majority accuracy | average switches |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['contrast']}` | {_fmt(row['temporal_late'])} | {_fmt(row['temporal_final'])} | "
            f"{_fmt(row['temporal_majority'])} | {_fmt(row['average_switch_count'])} |"
        )
    lines.extend(
        [
            "",
            "## Summary Paths",
            "",
            "| contrast | vertex summary | parcel summary | temporal summary |",
            "|---|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['contrast']}` | `{row['vertex_summary_path'] or 'missing'}` | "
            f"`{row['parcel_summary_path'] or 'missing'}` | `{row['temporal_summary_path'] or 'missing'}` |"
        )
    lines.extend(
        [
            "",
            "## Confidence Warning",
            "",
            "Each v0.3-lite contrast has ten held-out examples. This is larger than v0.2 but still too small for stable accuracy claims. "
            "Use the results to decide whether the signal is worth expanding, not as a production classifier benchmark.",
            "",
            "Classification labels are coarse triage labels: `stable` means both vertex and parcel top-1 are at least 0.70; "
            "`promising but unstable` means at least one static feature space reaches 0.60 or temporal late/final reaches 0.70; "
            "`weak/deprioritized` means no current v0.3 view clears those thresholds.",
        ]
    )
    if warnings:
        lines.extend(["", "## Validation Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines).rstrip() + "\n"


def markdown_to_simple_html(markdown: str) -> str:
    body = []
    for line in markdown.splitlines():
        if line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("|"):
            body.append(f"<pre>{html.escape(line)}</pre>")
        elif line.startswith("- "):
            body.append(f"<p>{html.escape(line)}</p>")
        elif not line.strip():
            body.append("")
        else:
            body.append(f"<p>{html.escape(line)}</p>")
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>ASNE v0.3 Benchmark</title>"
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;line-height:1.5}"
        "pre{background:#f6f8fa;padding:8px;overflow:auto}</style></head><body>"
        + "\n".join(body)
        + "</body></html>\n"
    )


def _row_from_summaries(
    spec: V03ContrastSpec,
    vertex: dict[str, Any] | None,
    parcel: dict[str, Any] | None,
    temporal: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "contrast": spec.name,
        "v02_top1": V02_BASELINE.get(spec.name),
        "vertex_top1": _value(vertex, "top1_accuracy"),
        "parcel_top1": _value(parcel, "top1_accuracy"),
        "vertex_top2": _value(vertex, "top2_accuracy"),
        "parcel_top2": _value(parcel, "top2_accuracy"),
        "vertex_mean_rank": _value(vertex, "mean_rank_expected"),
        "parcel_mean_rank": _value(parcel, "mean_rank_expected"),
        "mean_score_margin": _mean_score_margin(vertex),
        "temporal_late": _value(temporal, "late_accuracy"),
        "temporal_final": _value(temporal, "final_segment_accuracy"),
        "temporal_majority": _value(temporal, "majority_segment_accuracy"),
        "average_switch_count": _value(temporal, "average_switch_count"),
        "vertex_summary_path": vertex.get("_path") if vertex else None,
        "parcel_summary_path": parcel.get("_path") if parcel else None,
        "temporal_summary_path": temporal.get("_path") if temporal else None,
        "classification": classify_contrast(vertex, parcel, temporal),
    }


def classify_contrast(
    vertex: dict[str, Any] | None,
    parcel: dict[str, Any] | None,
    temporal: dict[str, Any] | None,
) -> str:
    vertex_top1 = _value(vertex, "top1_accuracy")
    parcel_top1 = _value(parcel, "top1_accuracy")
    temporal_late = _value(temporal, "late_accuracy")
    temporal_final = _value(temporal, "final_segment_accuracy")
    if vertex_top1 is None and parcel_top1 is None and temporal_late is None and temporal_final is None:
        return "pending"
    if (
        vertex_top1 is not None
        and parcel_top1 is not None
        and vertex_top1 >= 0.70
        and parcel_top1 >= 0.70
    ):
        return "stable"
    if max(value for value in [vertex_top1, parcel_top1, temporal_late, temporal_final] if value is not None) >= 0.60:
        return "promising but unstable"
    return "weak/deprioritized"


def _mean_score_margin(summary: dict[str, Any] | None) -> float | None:
    if not summary:
        return None
    margins = []
    for result in summary.get("results", []):
        ranking = result.get("per_category_ranking") or []
        if len(ranking) < 2:
            continue
        top = _score_value(ranking[0])
        second = _score_value(ranking[1])
        if top is not None and second is not None:
            margins.append(top - second)
    return sum(margins) / len(margins) if margins else None


def _score_value(item: dict[str, Any]) -> float | None:
    for key in ("score", "centroid_cosine_similarity", "mean_cosine_similarity", "best_cosine_similarity"):
        if item.get(key) is not None:
            return float(item[key])
    return None


def _value(summary: dict[str, Any] | None, key: str) -> float | None:
    if not summary or summary.get(key) is None:
        return None
    return float(summary[key])


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2f}"


def _run_or_print(command: list[str], *, dry_run: bool) -> None:
    if dry_run:
        print("WOULD RUN:", " ".join(command))
        return
    subprocess.run(command, check=True)


def _build_dictionary_command(spec: V03ContrastSpec, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "scripts/run_asne_dictionary.py",
        "--dictionary",
        spec.dictionary_json,
        "--tribev2-package-path",
        args.tribev2_package_path,
        "--cache-folder",
        args.cache_folder,
        "--feature-device",
        args.feature_device,
        "--tts-backend",
        args.tts_backend,
        "--debug-traceback",
    ]


def _eval_command(spec: V03ContrastSpec, args: argparse.Namespace, *, feature_space: str) -> list[str]:
    command = [
        sys.executable,
        "scripts/evaluate_asne_dictionary.py",
        "--dictionary",
        spec.dictionary_index,
        "--eval",
        spec.eval_json,
        "--tribev2-package-path",
        args.tribev2_package_path,
        "--cache-folder",
        args.cache_folder,
        "--feature-device",
        args.feature_device,
        "--tts-backend",
        args.tts_backend,
        "--signature",
        "mean_response",
        "--aggregation",
        "centroid",
        "--scoring",
        "centroid_raw",
        "--feature-space",
        feature_space,
        "--debug-traceback",
    ]
    if feature_space == "parcel":
        command.extend(["--parcellation", args.parcellation])
    return command


def _temporal_command(spec: V03ContrastSpec, static_summary: str, output: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/evaluate_asne_temporal_contrast.py",
        "--dictionary",
        spec.dictionary_index,
        "--eval",
        spec.eval_json,
        "--eval-summary",
        static_summary,
        "--output",
        str(output),
    ]


def _contrast_banned_terms(contrast_name: str) -> set[str]:
    common = {"diagnosis", "clinical", "surveillance", "emotion", "mental state"}
    by_contrast = {
        "contradiction_vs_consistency_paired": {"consistent", "contradictory", "contradiction", "conflict", "mismatch"},
        "expected_vs_unexpected_paired": {"expected", "unexpected", "surprising", "strange", "impossible", "mismatch", "contradict"},
        "cause_effect_valid_vs_invalid_paired": {"valid", "invalid", "cause", "effect", "impossible", "strange", "surprising", "unexpected", "contradiction", "mismatch"},
        "approach_vs_static_paired": {"static", "approach", "approaching", "closer", "threat", "fear"},
    }
    return common | by_contrast.get(contrast_name, set())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ASNE semantic_contrast_v0_3_lite benchmark.")
    parser.add_argument("--skip-build", action="store_true", help="Do not build missing dictionaries.")
    parser.add_argument("--force-eval", action="store_true", help="Rerun static vertex and parcel evals.")
    parser.add_argument("--force-temporal", action="store_true", help="Rerun temporal evals.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--contrasts",
        nargs="+",
        default=None,
        help="Optional subset of v0.3 contrast names to run. Use this for focused local runs.",
    )
    parser.add_argument("--allow-validation-warnings", action="store_true")
    parser.add_argument("--tts-backend", default="macos_say", choices=["tribev2_gtts", "macos_say", "openai", "higgs_audio"])
    parser.add_argument("--tribev2-package-path", default="./tribev2")
    parser.add_argument("--cache-folder", default="./cache")
    parser.add_argument("--feature-device", default="cpu")
    parser.add_argument("--parcellation", default="data/parcellations/fsaverage5_hcp_mmp.csv")
    parser.add_argument("--output", default="outputs/asne_reports/semantic_contrast_benchmark_v03.md")
    parser.add_argument("--html-output", default="outputs/asne_reports/semantic_contrast_benchmark_v03.html")
    return parser


def main() -> int:
    run_benchmark(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
