from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import DEFAULT_STEERING_CONFIG
from .delta import compute_delta, summarize_delta
from .manifest import create_experiment_manifest, save_experiment_manifest
from .report import generate_markdown_report
from .roi import load_roi_map, summarize_delta_by_roi
from .steering import load_steering_conditions
from .tribe_adapter import AdapterUnavailableError, MockTribeAdapter, TribeV2Adapter
from .visualize import generate_mock_output_plots


def _json_ready_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    ready = dict(prediction)
    response = ready.get("response")
    if isinstance(response, np.ndarray):
        ready["response"] = response.astype(float).tolist()
    return ready


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_mock_experiment(
    stimulus: str,
    config_path: str | Path = DEFAULT_STEERING_CONFIG,
    output_root: str | Path = "outputs",
    with_plots: bool = False,
    roi_map_path: str | Path | None = "configs/mock_rois.example.yaml",
    adapter_name: str = "mock",
    tribev2_package_path: str | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root)
    timestamp = datetime.now(timezone.utc)
    timestamp_compact = timestamp.strftime("%Y%m%dT%H%M%SZ")
    experiment_id = f"{adapter_name}_{timestamp_compact}"
    conditions = load_steering_conditions(config_path)
    roi_map = None
    if roi_map_path and Path(roi_map_path).exists():
        roi_map = load_roi_map(roi_map_path)
    neutral = next((condition for condition in conditions if condition.id == "neutral"), None)
    baseline_condition_id = neutral.id if neutral else "baseline"

    if adapter_name == "mock":
        adapter = MockTribeAdapter()
    elif adapter_name == "tribev2":
        adapter = TribeV2Adapter(package_path=tribev2_package_path)
    else:
        raise ValueError(f"Unknown adapter: {adapter_name}")

    baseline = adapter.predict(stimulus, condition_id=baseline_condition_id)
    baseline_path = output_root / "baseline" / f"{experiment_id}.json"
    _write_json(baseline_path, _json_ready_prediction(baseline))

    steered_paths: dict[str, str] = {}
    delta_paths: dict[str, str] = {}
    delta_summaries: dict[str, dict[str, Any]] = {}

    for condition in conditions:
        if condition.id == baseline_condition_id:
            continue

        steered = adapter.predict(stimulus, condition_id=condition.id)
        steered_path = output_root / "steered" / f"{experiment_id}_{condition.id}.json"
        _write_json(steered_path, _json_ready_prediction(steered))
        steered_paths[condition.id] = str(steered_path)

        delta = compute_delta(baseline, steered)
        summary = summarize_delta(delta)
        roi_summary = None
        if roi_map:
            roi_summary = summarize_delta_by_roi(delta, roi_map)
            summary["top_rois"] = [
                {
                    "roi": roi_id,
                    "mean_abs_delta": values["mean_abs_delta"],
                    "max_abs_delta": values["max_abs_delta"],
                    "n_indices": values["n_indices"],
                }
                for roi_id, values in sorted(
                    roi_summary.items(),
                    key=lambda item: item[1]["mean_abs_delta"],
                    reverse=True,
                )[:5]
            ]
        delta_payload = {
            "condition_id": condition.id,
            "baseline_condition_id": baseline_condition_id,
            "delta": delta.astype(float).tolist(),
            "summary": summary,
        }
        if roi_summary:
            delta_payload["roi_summary"] = roi_summary
        delta_path = output_root / "deltas" / f"{experiment_id}_{condition.id}.json"
        _write_json(delta_path, delta_payload)
        delta_paths[condition.id] = str(delta_path)
        delta_summaries[condition.id] = summary

    chart_paths = (
        generate_mock_output_plots(
            output_root,
            delta_paths=delta_paths,
            experiment_id=experiment_id,
        )
        if with_plots
        else None
    )
    report_path = generate_markdown_report(
        stimulus=stimulus,
        baseline_condition=baseline_condition_id,
        delta_summaries=delta_summaries,
        output_dir=output_root / "reports",
        chart_paths=chart_paths,
        roi_map_description=roi_map["description"].strip() if roi_map else None,
        experiment_id=experiment_id,
        manifest_path=output_root / "manifests" / f"{experiment_id}.json",
        adapter_type=adapter.__class__.__name__,
        steering_config_path=config_path,
        roi_map_path=roi_map_path if roi_map else None,
    )
    output_paths = {
        "baseline": str(baseline_path),
        "steered": steered_paths,
        "deltas": delta_paths,
        "report": str(report_path),
        "assets": chart_paths or {},
    }
    manifest = create_experiment_manifest(
        experiment_id=experiment_id,
        timestamp_utc=timestamp.isoformat(),
        stimulus=stimulus,
        adapter_type=adapter.__class__.__name__,
        steering_config_path=config_path,
        roi_map_path=roi_map_path if roi_map else None,
        steering_condition_ids=[condition.id for condition in conditions],
        output_paths=output_paths,
    )
    manifest_path = save_experiment_manifest(manifest, output_root / "manifests")

    return {
        "experiment_id": experiment_id,
        "baseline_path": str(baseline_path),
        "steered_paths": steered_paths,
        "delta_paths": delta_paths,
        "report_path": str(report_path),
        "asset_paths": chart_paths or {},
        "manifest_path": str(manifest_path),
    }


def check_adapter(
    adapter_name: str,
    tribev2_package_path: str | None = None,
    load_model: bool = False,
    model_name: str = "facebook/tribev2",
    cache_folder: str = "./cache",
) -> str:
    if adapter_name == "mock":
        adapter = MockTribeAdapter()
        prediction = adapter.predict("adapter check", condition_id="neutral")
        response = prediction["response"]
        return f"Adapter available: mock ({adapter.__class__.__name__}, response_size={len(response)})"
    if adapter_name == "tribev2":
        adapter = TribeV2Adapter(
            model_name=model_name,
            cache_folder=cache_folder,
            package_path=tribev2_package_path,
        )
        result = adapter.check_model_load(load_model=load_model)
        if result["loaded"]:
            return (
                "Adapter model load available: tribev2 "
                f"(model_name={result['model_name']}, cache_folder={result['cache_folder']}, "
                f"model_type={result['model_type']})"
            )
        return (
            "Adapter import available: tribev2 "
            f"(TribeModel={result['model_class']}, model_name={result['model_name']}, "
            f"cache_folder={result['cache_folder']})"
        )
    raise ValueError(f"Unknown adapter: {adapter_name}")


def run_tribev2_smoke(
    stimulus_path: str,
    tribev2_package_path: str | None = None,
    model_name: str = "facebook/tribev2",
    cache_folder: str = "./cache",
    output_root: str | Path = "outputs",
    feature_device: str | None = None,
    tts_backend: str = "tribev2_gtts",
    tts_cache_dir: str = "./cache/asne_tts",
    tts_voice: str = "Samantha",
    tts_rate: int = 180,
    openai_tts_model: str = "gpt-4o-mini-tts",
    openai_tts_voice: str = "alloy",
    openai_tts_format: str = "mp3",
    higgs_model: str = "bosonai/higgs-audio-v2-generation-3B-base",
    higgs_tokenizer: str = "bosonai/higgs-audio-v2-tokenizer",
    higgs_device: str = "auto",
    higgs_max_new_tokens: int = 1024,
    higgs_temperature: float = 0.3,
    higgs_top_p: float = 0.95,
    higgs_top_k: int = 50,
    higgs_scene_description: str = "Audio is recorded from a quiet room.",
) -> dict[str, Any]:
    stimulus = Path(stimulus_path).expanduser()
    if not stimulus.is_file():
        raise FileNotFoundError(f"Stimulus file does not exist: {stimulus}")

    timestamp = datetime.now(timezone.utc)
    experiment_id = f"tribev2_smoke_{timestamp.strftime('%Y%m%dT%H%M%SZ')}"
    adapter = TribeV2Adapter(
        model_name=model_name,
        cache_folder=cache_folder,
        package_path=tribev2_package_path,
        feature_device=feature_device,
        tts_backend=tts_backend,
        tts_cache_dir=tts_cache_dir,
        tts_voice=tts_voice,
        tts_rate=tts_rate,
        openai_tts_model=openai_tts_model,
        openai_tts_voice=openai_tts_voice,
        openai_tts_format=openai_tts_format,
        higgs_model=higgs_model,
        higgs_tokenizer=higgs_tokenizer,
        higgs_device=higgs_device,
        higgs_max_new_tokens=higgs_max_new_tokens,
        higgs_temperature=higgs_temperature,
        higgs_top_p=higgs_top_p,
        higgs_top_k=higgs_top_k,
        higgs_scene_description=higgs_scene_description,
    )
    adapter.check_model_load(load_model=True)
    prediction = adapter.predict(str(stimulus), condition_id=None)

    output_path = Path(output_root) / "tribev2_smoke" / f"{experiment_id}.json"
    payload = {
        "experiment_id": experiment_id,
        "timestamp_utc": timestamp.isoformat(),
        "prediction": prediction,
    }
    _write_json(output_path, payload)
    return {
        "experiment_id": experiment_id,
        "output_path": str(output_path),
        "response_shape": prediction["metadata"]["response_shape"],
        "raw_shape": prediction["metadata"]["raw_shape"],
        "segments_count": prediction["metadata"]["segments_count"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="asne")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_mock = subparsers.add_parser("run-mock", help="Run a deterministic mock ASNE experiment")
    run_mock.add_argument("--stimulus", required=True, help="Stimulus text to evaluate")
    run_mock.add_argument(
        "--adapter",
        choices=["mock", "tribev2"],
        default="mock",
        help="Neural encoding adapter to use. The TRIBE v2 adapter is a placeholder.",
    )
    run_mock.add_argument(
        "--tribev2-package-path",
        default=None,
        help="Local TRIBE v2 checkout/package root to prepend during lazy import, e.g. ./tribev2.",
    )
    run_mock.add_argument(
        "--config",
        default=str(DEFAULT_STEERING_CONFIG),
        help="Path to steering condition YAML config",
    )
    run_mock.add_argument(
        "--output-root",
        default="outputs",
        help="Directory where baseline, steered, delta, and report outputs are written",
    )
    run_mock.add_argument(
        "--with-plots",
        action="store_true",
        help="Generate matplotlib charts and include them in the markdown report",
    )
    run_mock.add_argument(
        "--roi-map",
        default="configs/mock_rois.example.yaml",
        help="Path to a mock ROI map YAML file. If absent, ROI summaries are skipped.",
    )

    check = subparsers.add_parser("check-adapter", help="Check adapter importability without inference")
    check.add_argument(
        "--adapter",
        choices=["mock", "tribev2"],
        default="mock",
        help="Adapter to check.",
    )
    check.add_argument(
        "--tribev2-package-path",
        default=None,
        help="Local TRIBE v2 checkout/package root to prepend during lazy import, e.g. ./tribev2.",
    )
    check.add_argument(
        "--load-model",
        action="store_true",
        help="Also call TribeModel.from_pretrained. May download model artifacts or initialize heavy dependencies.",
    )
    check.add_argument(
        "--model-name",
        default="facebook/tribev2",
        help="TRIBE v2 model name or checkpoint path for load-model diagnostics.",
    )
    check.add_argument(
        "--cache-folder",
        default="./cache",
        help="Cache folder passed to TribeModel.from_pretrained during load-model diagnostics.",
    )

    smoke = subparsers.add_parser(
        "run-tribev2-smoke",
        help="Run one TRIBE v2 video/text prediction smoke test without steering conditions",
    )
    smoke.add_argument(
        "--stimulus-path",
        required=True,
        help="Path to a tiny local video stimulus or .txt text stimulus",
    )
    smoke.add_argument(
        "--tribev2-package-path",
        default=None,
        help="Local TRIBE v2 checkout/package root to prepend during lazy import, e.g. ./tribev2.",
    )
    smoke.add_argument(
        "--model-name",
        default="facebook/tribev2",
        help="TRIBE v2 model name or checkpoint path.",
    )
    smoke.add_argument(
        "--cache-folder",
        default="./cache",
        help="Cache folder passed to TribeModel.from_pretrained.",
    )
    smoke.add_argument(
        "--output-root",
        default="outputs",
        help="Directory where smoke outputs are written.",
    )
    smoke.add_argument(
        "--feature-device",
        default=None,
        help="Best-effort device override for TRIBE feature extractors, e.g. cpu on Apple Silicon.",
    )
    smoke.add_argument(
        "--tts-backend",
        choices=["tribev2_gtts", "macos_say", "openai", "higgs_audio"],
        default="tribev2_gtts",
        help=(
            "Text stimulus audio backend. 'tribev2_gtts' preserves upstream TRIBE behavior; "
            "'macos_say', 'higgs_audio', and 'openai' generate cached audio before calling TRIBE audio mode."
        ),
    )
    smoke.add_argument("--tts-cache-dir", default="./cache/asne_tts")
    smoke.add_argument("--tts-voice", default="Samantha", help="Voice name for --tts-backend macos_say.")
    smoke.add_argument("--tts-rate", type=int, default=180, help="Speech rate for --tts-backend macos_say.")
    smoke.add_argument("--openai-tts-model", default="gpt-4o-mini-tts")
    smoke.add_argument("--openai-tts-voice", default="alloy")
    smoke.add_argument("--openai-tts-format", default="mp3", choices=["mp3", "wav", "opus", "aac", "flac"])
    smoke.add_argument("--higgs-model", default="bosonai/higgs-audio-v2-generation-3B-base")
    smoke.add_argument("--higgs-tokenizer", default="bosonai/higgs-audio-v2-tokenizer")
    smoke.add_argument("--higgs-device", default="auto", help="Device for Higgs Audio, e.g. auto, mps, cpu, cuda.")
    smoke.add_argument("--higgs-max-new-tokens", type=int, default=1024)
    smoke.add_argument("--higgs-temperature", type=float, default=0.3)
    smoke.add_argument("--higgs-top-p", type=float, default=0.95)
    smoke.add_argument("--higgs-top-k", type=int, default=50)
    smoke.add_argument("--higgs-scene-description", default="Audio is recorded from a quiet room.")
    smoke.add_argument(
        "--debug-traceback",
        action="store_true",
        help="Print the full chained traceback if TRIBE smoke prediction fails.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run-mock":
        try:
            paths = run_mock_experiment(
                stimulus=args.stimulus,
                config_path=args.config,
                output_root=args.output_root,
                with_plots=args.with_plots,
                roi_map_path=args.roi_map,
                adapter_name=args.adapter,
                tribev2_package_path=args.tribev2_package_path,
            )
        except AdapterUnavailableError as exc:
            print(f"Adapter unavailable: {exc}")
            return 1
        except NotImplementedError as exc:
            print(f"Adapter not implemented: {exc}")
            return 1
        print("ASNE mock experiment complete.")
        print(f"Experiment ID: {paths['experiment_id']}")
        print(f"Baseline: {paths['baseline_path']}")
        print("Steered:")
        for condition_id, path in paths["steered_paths"].items():
            print(f"  {condition_id}: {path}")
        print("Deltas:")
        for condition_id, path in paths["delta_paths"].items():
            print(f"  {condition_id}: {path}")
        if paths["asset_paths"]:
            print("Assets:")
            print(f"  condition_comparison: {paths['asset_paths']['condition_comparison']}")
            for condition_id, path in paths["asset_paths"]["per_condition"].items():
                print(f"  {condition_id}: {path}")
            if paths["asset_paths"].get("roi_condition_summary"):
                print(f"  roi_condition_summary: {paths['asset_paths']['roi_condition_summary']}")
            for condition_id, path in paths["asset_paths"].get("roi_per_condition", {}).items():
                print(f"  {condition_id}_roi: {path}")
        print(f"Report: {paths['report_path']}")
        print(f"Manifest: {paths['manifest_path']}")
        return 0

    if args.command == "check-adapter":
        try:
            if args.adapter == "tribev2" and args.load_model:
                print(
                    "Warning: --load-model calls TribeModel.from_pretrained and may download "
                    "model artifacts or initialize heavy dependencies. No stimulus prediction "
                    "will be run."
                )
            print(
                check_adapter(
                    args.adapter,
                    tribev2_package_path=args.tribev2_package_path,
                    load_model=args.load_model,
                    model_name=args.model_name,
                    cache_folder=args.cache_folder,
                )
            )
        except AdapterUnavailableError as exc:
            print(f"Adapter unavailable: {exc}")
            return 1
        return 0

    if args.command == "run-tribev2-smoke":
        try:
            print(
                "Warning: run-tribev2-smoke loads TRIBE v2 and runs one video/text prediction. "
                "It may initialize heavy dependencies and use cached model artifacts. "
                "No ASNE steering conditions are applied. Text input may trigger upstream "
                "text-to-speech/audio preprocessing."
            )
            result = run_tribev2_smoke(
                stimulus_path=args.stimulus_path,
                tribev2_package_path=args.tribev2_package_path,
                model_name=args.model_name,
                cache_folder=args.cache_folder,
                output_root=args.output_root,
                feature_device=args.feature_device,
                tts_backend=args.tts_backend,
                tts_cache_dir=args.tts_cache_dir,
                tts_voice=args.tts_voice,
                tts_rate=args.tts_rate,
                openai_tts_model=args.openai_tts_model,
                openai_tts_voice=args.openai_tts_voice,
                openai_tts_format=args.openai_tts_format,
                higgs_model=args.higgs_model,
                higgs_tokenizer=args.higgs_tokenizer,
                higgs_device=args.higgs_device,
                higgs_max_new_tokens=args.higgs_max_new_tokens,
                higgs_temperature=args.higgs_temperature,
                higgs_top_p=args.higgs_top_p,
                higgs_top_k=args.higgs_top_k,
                higgs_scene_description=args.higgs_scene_description,
            )
        except (AdapterUnavailableError, FileNotFoundError) as exc:
            print(f"TRIBE v2 smoke failed: {exc}")
            return 1
        except Exception as exc:
            summary = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            print(f"TRIBE v2 smoke failed during prediction: {summary}")
            if args.debug_traceback:
                print("Full traceback:")
                print("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip())
            print(
                "Action: confirm the stimulus is a short supported video or .txt file, TRIBE v2 dependencies "
                "are installed, and required model artifacts are available in cache or allowed "
                "to download by your environment."
            )
            return 1
        print("TRIBE v2 smoke prediction complete.")
        print(f"Output: {result['output_path']}")
        print(f"Raw shape: {result['raw_shape']}")
        print(f"Response shape: {result['response_shape']}")
        print(f"Segments: {result['segments_count']}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
