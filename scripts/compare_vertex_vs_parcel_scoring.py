#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONTRAST_OUTPUT_ROOT = Path("outputs/asne_evals/contrasts")


@dataclass(frozen=True)
class ContrastSpec:
    name: str
    dictionary: str
    eval_path: str


CONTRASTS = [
    ContrastSpec(
        name="contradiction_vs_consistency_paired",
        dictionary="outputs/asne_dictionaries/contradiction_vs_consistency_paired_tts_macos_say_samantha_180/dictionary_index.json",
        eval_path="data/stimuli/evals/contrasts/contradiction_vs_consistency_paired_eval.json",
    ),
    ContrastSpec(
        name="expected_vs_unexpected_paired",
        dictionary="outputs/asne_dictionaries/expected_vs_unexpected_paired_tts_macos_say_samantha_180/dictionary_index.json",
        eval_path="data/stimuli/evals/contrasts/expected_vs_unexpected_paired_eval.json",
    ),
    ContrastSpec(
        name="approach_vs_static_paired",
        dictionary="outputs/asne_dictionaries/approach_vs_static_paired_tts_macos_say_samantha_180/dictionary_index.json",
        eval_path="data/stimuli/evals/contrasts/approach_vs_static_paired_eval.json",
    ),
    ContrastSpec(
        name="cause_effect_valid_vs_invalid_paired",
        dictionary="outputs/asne_dictionaries/cause_effect_valid_vs_invalid_paired_tts_macos_say_samantha_180/dictionary_index.json",
        eval_path="data/stimuli/evals/contrasts/cause_effect_valid_vs_invalid_paired_eval.json",
    ),
]


def load_summary(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload["_path"] = str(path)
    return payload


def latest_summary(
    contrast: str,
    feature_space: str,
    root: str | Path = CONTRAST_OUTPUT_ROOT,
) -> dict[str, Any] | None:
    candidates = []
    for path in sorted(Path(root).glob(f"{contrast}/*_summary.json")):
        payload = load_summary(path)
        payload_feature_space = payload.get("feature_space") or "vertex"
        if (
            payload_feature_space == feature_space
            and payload.get("signature_mode") == "mean_response"
            and payload.get("aggregation_mode") == "centroid"
            and payload.get("scoring_mode") == "centroid_raw"
        ):
            candidates.append(payload)
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: Path(item["_path"]).stat().st_mtime)[-1]


def evaluate_if_needed(args: argparse.Namespace, spec: ContrastSpec, feature_space: str) -> dict[str, Any]:
    existing = None if args.force else latest_summary(spec.name, feature_space, args.static_root)
    if existing is not None:
        return existing
    command = [
        sys.executable,
        "scripts/evaluate_asne_dictionary.py",
        "--dictionary",
        spec.dictionary,
        "--eval",
        spec.eval_path,
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
    ]
    if feature_space == "parcel":
        command.extend(["--parcellation", args.parcellation])
    if args.debug_traceback:
        command.append("--debug-traceback")
    if args.dry_run:
        print("WOULD RUN:", " ".join(command))
        return {
            "_path": "<dry-run>",
            "contrast_name": spec.name,
            "feature_space": feature_space,
            "top1_accuracy": None,
            "top2_accuracy": None,
            "mean_rank_expected": None,
        }
    subprocess.run(command, check=True)
    summary = latest_summary(spec.name, feature_space, args.static_root)
    if summary is None:
        raise RuntimeError(f"No {feature_space} summary was produced for {spec.name}.")
    return summary


def build_rows(vertex_summaries: dict[str, dict[str, Any]], parcel_summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for contrast in vertex_summaries:
        vertex = vertex_summaries[contrast]
        parcel = parcel_summaries[contrast]
        vertex_top1 = _as_float(vertex.get("top1_accuracy"))
        parcel_top1 = _as_float(parcel.get("top1_accuracy"))
        difference = None if vertex_top1 is None or parcel_top1 is None else parcel_top1 - vertex_top1
        if difference is None:
            recommendation = "pending"
        elif difference >= 0.0:
            recommendation = "parcel viable for scoring experiment"
        else:
            recommendation = "keep vertex benchmark as primary"
        rows.append(
            {
                "contrast": contrast,
                "vertex_top1": vertex_top1,
                "parcel_top1": parcel_top1,
                "vertex_top2": _as_float(vertex.get("top2_accuracy")),
                "parcel_top2": _as_float(parcel.get("top2_accuracy")),
                "vertex_mean_rank": _as_float(vertex.get("mean_rank_expected")),
                "parcel_mean_rank": _as_float(parcel.get("mean_rank_expected")),
                "difference": difference,
                "recommendation": recommendation,
                "vertex_summary_path": vertex.get("_path"),
                "parcel_summary_path": parcel.get("_path"),
            }
        )
    return rows


def format_report(rows: list[dict[str, Any]], parcellation: str) -> str:
    lines = [
        "# ASNE Vertex vs Parcel Scoring Report v0",
        "",
        "This compares the frozen vertex-level benchmark against HCP-MMP parcel-level centroid scoring.",
        "",
        f"- Parcel parcellation: `{parcellation}`",
        "- Vertex setting: `feature_space=vertex`, `signature=mean_response`, `aggregation=centroid`, `scoring=centroid_raw`",
        "- Parcel setting: `feature_space=parcel`, `signature=mean_response`, `aggregation=centroid`, `scoring=centroid_raw`",
        "",
        "| contrast | vertex_top1 | parcel_top1 | vertex_top2 | parcel_top2 | vertex_mean_rank | parcel_mean_rank | difference | recommendation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['contrast']}` | {_fmt(row['vertex_top1'])} | {_fmt(row['parcel_top1'])} | "
            f"{_fmt(row['vertex_top2'])} | {_fmt(row['parcel_top2'])} | "
            f"{_fmt(row['vertex_mean_rank'])} | {_fmt(row['parcel_mean_rank'])} | "
            f"{_fmt(row['difference'])} | {row['recommendation']} |"
        )
    lines.extend(
        [
            "",
            "## Summary Paths",
            "",
            "| contrast | vertex summary | parcel summary |",
            "|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['contrast']}` | `{row['vertex_summary_path']}` | `{row['parcel_summary_path']}` |"
        )
    lines.extend(
        [
            "",
            "## Safety Framing",
            "",
            "This is predicted stimulus-response similarity from TRIBE outputs, not measured brain activity, diagnosis, or measurement of a person's mental state.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    vertex_summaries = {}
    parcel_summaries = {}
    for spec in CONTRASTS:
        vertex_summaries[spec.name] = evaluate_if_needed(args, spec, "vertex")
        parcel_summaries[spec.name] = evaluate_if_needed(args, spec, "parcel")
    rows = build_rows(vertex_summaries, parcel_summaries)
    report = format_report(rows, args.parcellation)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    return {"output_path": str(output), "rows": rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare ASNE vertex-level and parcel-level scoring.")
    parser.add_argument("--parcellation", default="data/parcellations/fsaverage5_hcp_mmp.csv")
    parser.add_argument("--output", default="outputs/asne_reports/vertex_vs_parcel_scoring_v0.md")
    parser.add_argument("--static-root", default=str(CONTRAST_OUTPUT_ROOT))
    parser.add_argument("--tribev2-package-path", default="./tribev2")
    parser.add_argument("--cache-folder", default="./cache")
    parser.add_argument("--feature-device", default="cpu")
    parser.add_argument("--tts-backend", default="macos_say")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug-traceback", action="store_true")
    return parser


def _as_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2f}"


def main() -> int:
    result = run(build_parser().parse_args())
    print(f"Report: {result['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
