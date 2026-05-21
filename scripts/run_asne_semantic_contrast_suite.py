#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_asne_contrast_report import DEFAULT_CONTRASTS


@dataclass(frozen=True)
class ContrastSpec:
    name: str
    dictionary_json: str
    eval_json: str
    dictionary_output: str

    @property
    def dictionary_index(self) -> str:
        return f"outputs/asne_dictionaries/{self.dictionary_output}/dictionary_index.json"


CONTRAST_SUITE: list[ContrastSpec] = [
    ContrastSpec(
        name="contradiction_vs_consistency_paired",
        dictionary_json="data/stimuli/dictionaries/contrasts/contradiction_vs_consistency_paired.json",
        eval_json="data/stimuli/evals/contrasts/contradiction_vs_consistency_paired_eval.json",
        dictionary_output="contradiction_vs_consistency_paired_tts_macos_say_samantha_180",
    ),
    ContrastSpec(
        name="expected_vs_unexpected_paired",
        dictionary_json="data/stimuli/dictionaries/contrasts/expected_vs_unexpected_paired.json",
        eval_json="data/stimuli/evals/contrasts/expected_vs_unexpected_paired_eval.json",
        dictionary_output="expected_vs_unexpected_paired_tts_macos_say_samantha_180",
    ),
    ContrastSpec(
        name="approach_vs_static_paired",
        dictionary_json="data/stimuli/dictionaries/contrasts/approach_vs_static_paired.json",
        eval_json="data/stimuli/evals/contrasts/approach_vs_static_paired_eval.json",
        dictionary_output="approach_vs_static_paired_tts_macos_say_samantha_180",
    ),
    ContrastSpec(
        name="cause_effect_valid_vs_invalid_paired",
        dictionary_json="data/stimuli/dictionaries/contrasts/cause_effect_valid_vs_invalid_paired.json",
        eval_json="data/stimuli/evals/contrasts/cause_effect_valid_vs_invalid_paired_eval.json",
        dictionary_output="cause_effect_valid_vs_invalid_paired_tts_macos_say_samantha_180",
    ),
]


def latest_static_summary(contrast_name: str, root: str | Path = "outputs/asne_evals/contrasts") -> Path | None:
    paths = sorted(Path(root).glob(f"{contrast_name}/*_summary.json"), key=lambda path: path.stat().st_mtime)
    if not paths:
        return None
    default_paths = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("signature_mode") == "mean_response"
            and payload.get("aggregation_mode") == "centroid"
            and payload.get("scoring_mode") == "centroid_raw"
        ):
            default_paths.append(path)
    return default_paths[-1] if default_paths else paths[-1]


def latest_temporal_summary(contrast_name: str, root: str | Path = "outputs/asne_temporal_evals") -> Path | None:
    paths = sorted(Path(root).glob(f"{contrast_name}/*_temporal_summary.json"), key=lambda path: path.stat().st_mtime)
    return paths[-1] if paths else None


def build_dictionary_command(spec: ContrastSpec, args: argparse.Namespace) -> list[str]:
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


def evaluate_command(spec: ContrastSpec, args: argparse.Namespace) -> list[str]:
    return [
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
        "--debug-traceback",
    ]


def temporal_command(spec: ContrastSpec, static_summary: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/evaluate_asne_temporal_contrast.py",
        "--dictionary",
        spec.dictionary_index,
        "--eval",
        spec.eval_json,
        "--eval-summary",
        str(static_summary),
    ]


def report_command() -> list[str]:
    return [sys.executable, "scripts/generate_asne_contrast_report.py"]


def run_or_print(command: list[str], *, dry_run: bool) -> None:
    if dry_run:
        print("WOULD RUN:", " ".join(command))
        return
    subprocess.run(command, check=True)


def run_suite(args: argparse.Namespace) -> dict[str, list[str]]:
    static_paths: list[str] = []
    temporal_paths: list[str] = []
    for spec in CONTRAST_SUITE:
        print(f"\n== {spec.name} ==")
        dictionary_index = Path(spec.dictionary_index)
        if dictionary_index.exists():
            print(f"dictionary index exists: {dictionary_index}")
        elif args.run_build and not args.skip_build:
            run_or_print(build_dictionary_command(spec, args), dry_run=args.dry_run)
        else:
            print(f"missing dictionary index: {dictionary_index}")
            if args.skip_build:
                print("skip-build is set; dictionary build will not run.")
            else:
                print("default behavior will not build missing dictionaries; pass --run-build to build.")

        static_summary = latest_static_summary(spec.name)
        if static_summary and not args.force_eval:
            print(f"static summary exists: {static_summary}")
        else:
            if not dictionary_index.exists() and not args.dry_run:
                print("cannot run static eval without dictionary index.")
            else:
                run_or_print(evaluate_command(spec, args), dry_run=args.dry_run)
                static_summary = latest_static_summary(spec.name)
        if static_summary:
            static_paths.append(str(static_summary))

        temporal_summary = latest_temporal_summary(spec.name)
        if temporal_summary and not args.force_temporal:
            print(f"temporal summary exists: {temporal_summary}")
        elif static_summary:
            run_or_print(temporal_command(spec, static_summary), dry_run=args.dry_run)
            temporal_summary = latest_temporal_summary(spec.name)
        else:
            print("cannot run temporal eval without a static summary.")
        if temporal_summary:
            temporal_paths.append(str(temporal_summary))

    run_or_print(report_command(), dry_run=args.dry_run)
    print("\nCollected static summaries:")
    for path in static_paths:
        print(f"- {path}")
    print("Collected temporal summaries:")
    for path in temporal_paths:
        print(f"- {path}")
    if not args.dry_run:
        print("Report: outputs/asne_reports/semantic_contrast_report_v0.md")
    return {"static": static_paths, "temporal": temporal_paths}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or verify the frozen ASNE semantic contrast suite.")
    parser.add_argument("--skip-build", action="store_true", help="Do not build missing dictionary outputs.")
    parser.add_argument("--run-build", action="store_true", help="Build missing dictionary outputs.")
    parser.add_argument("--force-eval", action="store_true", help="Rerun static contrast evaluations.")
    parser.add_argument("--force-temporal", action="store_true", help="Rerun temporal contrast evaluations.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without running commands.")
    parser.add_argument("--tts-backend", default="macos_say", choices=["tribev2_gtts", "macos_say", "openai", "higgs_audio"])
    parser.add_argument("--tribev2-package-path", default="./tribev2")
    parser.add_argument("--cache-folder", default="./cache")
    parser.add_argument("--feature-device", default="cpu")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if [spec.name for spec in CONTRAST_SUITE] != DEFAULT_CONTRASTS:
        print("Warning: suite contrast order differs from report generator defaults.")
    run_suite(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
