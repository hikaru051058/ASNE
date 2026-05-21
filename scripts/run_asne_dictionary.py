#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from asne.dictionary import (
    add_delta_from_neutral,
    aggregate_response_segments,
    compute_late_minus_early,
    compute_neutral_baseline,
    compute_temporal_movement_signatures,
    iter_dictionary_stimuli,
    load_stimulus_dictionary,
    slugify,
    top_abs_indices,
    vector_norm,
)
from asne.tribe_adapter import TribeV2Adapter


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _predict_text(
    adapter: TribeV2Adapter,
    text: str,
    raw_prediction_path: Path,
) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    try:
        return adapter.predict(
            str(temp_path),
            condition_id=None,
            raw_prediction_path=raw_prediction_path,
        )
    finally:
        temp_path.unlink(missing_ok=True)


def _infer_neutral_category(payload: dict[str, Any], requested: str | None) -> str:
    if requested:
        return requested
    if payload.get("baseline_category"):
        return str(payload["baseline_category"])
    categories = list(payload["categories"])
    if "neutral" in categories:
        return "neutral"
    if "ordinary_observation" in categories:
        return "ordinary_observation"
    return categories[0]


def _dictionary_run_name(dictionary_name: str, args: argparse.Namespace) -> str:
    if args.tts_backend == "tribev2_gtts":
        return dictionary_name
    if args.tts_backend == "macos_say":
        suffix = f"tts_macos_say_{args.tts_voice}_{args.tts_rate}"
    elif args.tts_backend == "higgs_audio":
        suffix = f"tts_higgs_{args.higgs_device}_{args.higgs_temperature}_{args.higgs_top_p}_{args.higgs_top_k}"
    else:
        suffix = f"tts_openai_{args.openai_tts_model}_{args.openai_tts_voice}_{args.openai_tts_format}"
    return f"{dictionary_name}_{slugify(suffix)}"


def _record_from_raw_segments(
    *,
    dictionary_name: str,
    category: str,
    stimulus_id: str,
    text: str,
    pair_id: str | None,
    raw_path: Path,
    percentile: float,
    created_at: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    raw_array = np.load(raw_path)
    aggregates = aggregate_response_segments(raw_array, percentile=percentile)
    return {
        "dictionary_name": dictionary_name,
        "category": category,
        "stimulus_id": stimulus_id,
        "pair_id": pair_id,
        "input_text": text,
        "segments": aggregates["segments"],
        "raw_shape": aggregates["raw_shape"],
        "response_shape": aggregates["response_shape"],
        "mean_response": aggregates["mean_response"].astype(float).tolist(),
        "raw_segment_prediction_path": str(raw_path),
        "created_at": created_at,
        "model_metadata": {
            "adapter": "tribev2",
            "model": "tribev2",
            "input_mode": "text",
            "model_name": args.model_name,
            "cache_folder": args.cache_folder,
            "feature_device": args.feature_device,
            "tts_backend": args.tts_backend,
            "aggregation": "mean_over_segments",
            "warning": (
                "ASNE dictionary signatures are predicted response similarities for "
                "controlled stimuli. They are not diagnoses and are not measurements "
                "of an individual person's mental state."
            ),
        },
        "peak_abs_response_summary": {
            "max": float(np.max(aggregates["peak_abs_response"])),
            "mean": float(np.mean(aggregates["peak_abs_response"])),
        },
        "percentile_abs_response_summary": {
            "percentile": aggregates["percentile"],
            "max": float(np.max(aggregates["percentile_abs_response"])),
            "mean": float(np.mean(aggregates["percentile_abs_response"])),
        },
    }


def _attach_temporal_movement(record: dict[str, Any], raw_path: Path) -> dict[str, Any]:
    if not raw_path.exists():
        return record
    raw_array = np.load(raw_path)
    movement = compute_temporal_movement_signatures(raw_array)
    late_minus_early = compute_late_minus_early(raw_array)
    vector_dir = raw_path.parent.parent / "movement_vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{record['category']}_{record['stimulus_id']}"
    vectors = {
        "net_movement": movement["net_movement"],
        "avg_slope_per_transition": movement["avg_slope_per_transition"],
        "abs_total_movement": movement["abs_total_movement"],
        "late_minus_early": late_minus_early,
    }
    for name, vector in vectors.items():
        path = vector_dir / f"{prefix}_{name}.npy"
        np.save(path, np.asarray(vector, dtype=float))
        record[f"{name}_path"] = str(path)
    record["movement_summary"] = {
        "segment_count": movement["segment_count"],
        "transition_count": movement["transition_count"],
        "movement_norm": vector_norm(movement["net_movement"]),
        "avg_slope_norm": vector_norm(movement["avg_slope_per_transition"]),
        "abs_total_movement_norm": vector_norm(movement["abs_total_movement"]),
        "late_minus_early_norm": vector_norm(late_minus_early),
        "top_moving_dimensions": top_abs_indices(movement["net_movement"]),
        "top_abs_movement_dimensions": top_abs_indices(movement["abs_total_movement"]),
    }
    return record


def build_dictionary(args: argparse.Namespace) -> dict[str, Any]:
    payload = load_stimulus_dictionary(args.dictionary)
    source_dictionary_name = slugify(str(payload.get("name") or Path(args.dictionary).stem))
    dictionary_name = _dictionary_run_name(source_dictionary_name, args)
    output_dir = Path(args.output_root) / "asne_dictionaries" / dictionary_name
    raw_dir = output_dir / "raw_segments"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    neutral_category = _infer_neutral_category(payload, args.neutral_category)
    adapter: TribeV2Adapter | None = None

    created_at = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []
    index_items: list[dict[str, Any]] = []

    stimuli = iter_dictionary_stimuli(payload)
    if args.limit is not None:
        stimuli = stimuli[: args.limit]

    for stimulus in stimuli:
        print(f"Running {stimulus.category}/{stimulus.stimulus_id}")
        raw_path = raw_dir / f"{stimulus.category}_{stimulus.stimulus_id}.npy"
        if raw_path.exists() and not args.force_recompute:
            record = _record_from_raw_segments(
                dictionary_name=dictionary_name,
                category=stimulus.category,
                stimulus_id=stimulus.stimulus_id,
                text=stimulus.text,
                pair_id=stimulus.pair_id,
                raw_path=raw_path,
                percentile=args.percentile,
                created_at=created_at,
                args=args,
            )
        else:
            if adapter is None:
                adapter = TribeV2Adapter(
                    model_name=args.model_name,
                    cache_folder=args.cache_folder,
                    package_path=args.tribev2_package_path,
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
                adapter.check_model_load(load_model=True)
            prediction = _predict_text(adapter, stimulus.text, raw_path)
            response = np.asarray(prediction["response"], dtype=float)
            metadata = prediction.get("metadata", {})

            record = {
                "dictionary_name": dictionary_name,
                "category": stimulus.category,
                "stimulus_id": stimulus.stimulus_id,
                "pair_id": stimulus.pair_id,
                "input_text": stimulus.text,
                "segments": metadata.get("segments_count"),
                "raw_shape": metadata.get("raw_shape"),
                "response_shape": metadata.get("response_shape"),
                "mean_response": response.astype(float).tolist(),
                "raw_segment_prediction_path": metadata.get("raw_prediction_path"),
                "created_at": created_at,
                "model_metadata": {
                    "adapter": prediction.get("adapter"),
                    "model": prediction.get("model"),
                    "input_mode": prediction.get("input_mode"),
                    "model_name": metadata.get("model_name"),
                    "cache_folder": metadata.get("cache_folder"),
                    "feature_device": metadata.get("feature_device"),
                    "tts_backend": metadata.get("tts_backend"),
                    "tts_metadata": metadata.get("tts_metadata"),
                    "aggregation": metadata.get("aggregation"),
                    "warning": (
                        "ASNE dictionary signatures are predicted response similarities for "
                        "controlled stimuli. They are not diagnoses and are not measurements "
                        "of an individual person's mental state."
                    ),
                },
            }
            if raw_path.exists():
                raw_array = np.load(raw_path)
                aggregates = aggregate_response_segments(raw_array, percentile=args.percentile)
                record["peak_abs_response_summary"] = {
                    "max": float(np.max(aggregates["peak_abs_response"])),
                    "mean": float(np.mean(aggregates["peak_abs_response"])),
                }
                record["percentile_abs_response_summary"] = {
                    "percentile": aggregates["percentile"],
                    "max": float(np.max(aggregates["percentile_abs_response"])),
                    "mean": float(np.mean(aggregates["percentile_abs_response"])),
                }

        record = _attach_temporal_movement(record, raw_path)
        records.append(record)

    neutral_baseline = compute_neutral_baseline(records, neutral_category=neutral_category)
    baseline_path = output_dir / "neutral_baseline.npy"
    np.save(baseline_path, neutral_baseline)

    for record in records:
        record_with_delta = add_delta_from_neutral(
            record,
            neutral_baseline,
            neutral_category=neutral_category,
        )
        output_path = output_dir / f"{record['category']}_{record['stimulus_id']}.json"
        _write_json(output_path, record_with_delta)
        index_items.append(
            {
                "category": record["category"],
                "stimulus_id": record["stimulus_id"],
                "pair_id": record.get("pair_id"),
                "output_path": str(output_path),
                "response_shape": record["response_shape"],
                "segments": record["segments"],
            }
        )

    index = {
        "dictionary_name": dictionary_name,
        "source_dictionary_name": source_dictionary_name,
        "source_dictionary_path": str(args.dictionary),
        "created_at": created_at,
        "tts_backend": args.tts_backend,
        "tts_config": {
            "tts_cache_dir": args.tts_cache_dir,
            "tts_voice": args.tts_voice,
            "tts_rate": args.tts_rate,
            "openai_tts_model": args.openai_tts_model,
            "openai_tts_voice": args.openai_tts_voice,
            "openai_tts_format": args.openai_tts_format,
            "higgs_model": args.higgs_model,
            "higgs_tokenizer": args.higgs_tokenizer,
            "higgs_device": args.higgs_device,
            "higgs_max_new_tokens": args.higgs_max_new_tokens,
            "higgs_temperature": args.higgs_temperature,
            "higgs_top_p": args.higgs_top_p,
            "higgs_top_k": args.higgs_top_k,
            "higgs_scene_description": args.higgs_scene_description,
        },
        "neutral_baseline_path": str(baseline_path),
        "neutral_baseline_category": neutral_category,
        "stimuli": index_items,
        "safety_note": (
            "This is a stimulus-response dictionary for predicted response similarity. "
            "It is not a diagnosis and is not a measurement of an individual person's mental state."
        ),
    }
    index_path = output_dir / "dictionary_index.json"
    _write_json(index_path, index)
    return {
        "dictionary_name": dictionary_name,
        "output_dir": str(output_dir),
        "index_path": str(index_path),
        "n_stimuli": len(index_items),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an ASNE stimulus-response dictionary.")
    parser.add_argument(
        "--dictionary",
        default="data/stimuli/dictionaries/emotion_context_v0.json",
        help="Path to dictionary JSON.",
    )
    parser.add_argument("--tribev2-package-path", default="./tribev2")
    parser.add_argument("--cache-folder", default="./cache")
    parser.add_argument("--model-name", default="facebook/tribev2")
    parser.add_argument("--feature-device", default="cpu")
    parser.add_argument(
        "--tts-backend",
        choices=["tribev2_gtts", "macos_say", "openai", "higgs_audio"],
        default="tribev2_gtts",
        help=(
            "Text stimulus audio backend. 'tribev2_gtts' preserves upstream TRIBE behavior; "
            "'macos_say', 'higgs_audio', and 'openai' generate cached audio before calling TRIBE audio mode."
        ),
    )
    parser.add_argument("--tts-cache-dir", default="./cache/asne_tts")
    parser.add_argument("--tts-voice", default="Samantha", help="Voice name for --tts-backend macos_say.")
    parser.add_argument("--tts-rate", type=int, default=180, help="Speech rate for --tts-backend macos_say.")
    parser.add_argument("--openai-tts-model", default="gpt-4o-mini-tts")
    parser.add_argument("--openai-tts-voice", default="alloy")
    parser.add_argument("--openai-tts-format", default="mp3", choices=["mp3", "wav", "opus", "aac", "flac"])
    parser.add_argument("--higgs-model", default="bosonai/higgs-audio-v2-generation-3B-base")
    parser.add_argument("--higgs-tokenizer", default="bosonai/higgs-audio-v2-tokenizer")
    parser.add_argument("--higgs-device", default="auto", help="Device for Higgs Audio, e.g. auto, mps, cpu, cuda.")
    parser.add_argument("--higgs-max-new-tokens", type=int, default=1024)
    parser.add_argument("--higgs-temperature", type=float, default=0.3)
    parser.add_argument("--higgs-top-p", type=float, default=0.95)
    parser.add_argument("--higgs-top-k", type=int, default=50)
    parser.add_argument("--higgs-scene-description", default="Audio is recorded from a quiet room.")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument(
        "--neutral-category",
        default=None,
        help=(
            "Category used as the baseline for delta_from_neutral. "
            "Defaults to 'neutral' when present, otherwise 'ordinary_observation' when present."
        ),
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore existing raw segment .npy files and run TRIBE predictions again.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional development limit. Omit to run every dictionary stimulus.",
    )
    parser.add_argument("--debug-traceback", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = build_dictionary(args)
    except Exception as exc:
        print(f"ASNE dictionary build failed: {type(exc).__name__}: {exc}")
        if args.debug_traceback:
            print("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip())
        return 1

    print("ASNE dictionary build complete.")
    print(f"Dictionary: {result['dictionary_name']}")
    print(f"Stimuli: {result['n_stimuli']}")
    print(f"Output directory: {result['output_dir']}")
    print(f"Index: {result['index_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
