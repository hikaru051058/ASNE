#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from asne.dictionary import COMPARISON_DISCLAIMER, load_dictionary_records
from asne.roi import (
    aggregate_segments_to_parcels,
    aggregate_vertices_to_parcels,
    compute_parcel_delta,
    load_parcellation,
    rank_parcels_by_abs_delta,
    validate_parcellation,
)


ROI_DISCLAIMER = (
    "This is predicted response similarity from TRIBE outputs, not measured brain activity, "
    "diagnosis, or measurement of a person's mental state."
)


def _mean_parcels(records: list[dict[str, Any]], category: str, parcellation: dict[str, Any]) -> dict[str, float]:
    vectors = [
        aggregate_vertices_to_parcels(record["mean_response"], parcellation)
        for record in records
        if record["category"] == category
    ]
    if not vectors:
        raise ValueError(f"No dictionary records found for category: {category}")
    parcel_ids = vectors[0].keys()
    return {
        parcel_id: float(np.mean([vector[parcel_id] for vector in vectors]))
        for parcel_id in parcel_ids
    }


def _categories_from_records(records: list[dict[str, Any]]) -> list[str]:
    categories = []
    for record in records:
        if record["category"] not in categories:
            categories.append(record["category"])
    return categories


def _temporal_rankings(records: list[dict[str, Any]], parcellation: dict[str, Any], top_k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    late_minus_early_by_parcel: dict[str, list[float]] = {}
    movement_by_parcel: dict[str, list[float]] = {}
    for record in records:
        raw_path = record.get("raw_segment_prediction_path") or record.get("raw_segments_path")
        if not raw_path or not Path(raw_path).exists():
            continue
        raw = np.load(raw_path)
        parcel_segments = aggregate_segments_to_parcels(raw, parcellation)["values"]
        if parcel_segments.shape[0] <= 1:
            continue
        split = parcel_segments.shape[0] // 2
        early = parcel_segments[:split].mean(axis=0)
        late = parcel_segments[split:].mean(axis=0)
        late_minus_early = late - early
        movement = np.sum(np.abs(np.diff(parcel_segments, axis=0)), axis=0)
        parcel_ids = list(parcellation["by_parcel"])
        for index, parcel_id in enumerate(parcel_ids):
            late_minus_early_by_parcel.setdefault(parcel_id, []).append(float(late_minus_early[index]))
            movement_by_parcel.setdefault(parcel_id, []).append(float(movement[index]))

    late_delta = {
        parcel_id: float(np.mean(values))
        for parcel_id, values in late_minus_early_by_parcel.items()
    }
    movement_delta = {
        parcel_id: float(np.mean(values))
        for parcel_id, values in movement_by_parcel.items()
    }
    return (
        _attach_names(rank_parcels_by_abs_delta(late_delta, top_k=top_k), parcellation),
        _attach_names(rank_parcels_by_abs_delta(movement_delta, top_k=top_k), parcellation),
    )


def _attach_names(rows: list[dict[str, Any]], parcellation: dict[str, Any]) -> list[dict[str, Any]]:
    for row in rows:
        row["parcel_name"] = parcellation["parcel_names"].get(row["parcel_id"], row["parcel_id"])
    return rows


def analyze_roi_contrast(args: argparse.Namespace) -> dict[str, Any]:
    records = load_dictionary_records(args.dictionary)
    categories = _categories_from_records(records)
    if len(categories) != 2:
        raise ValueError("ROI contrast analysis currently expects a binary dictionary.")
    parcellation = load_parcellation(args.parcellation)
    validate_parcellation(parcellation, expected_vertices=args.expected_vertices)
    category_a, category_b = categories
    a_centroid = _mean_parcels(records, category_a, parcellation)
    b_centroid = _mean_parcels(records, category_b, parcellation)
    parcel_delta = compute_parcel_delta(a_centroid, b_centroid)
    top_contrast = _attach_names(rank_parcels_by_abs_delta(parcel_delta, top_k=args.top_k), parcellation)
    top_late, top_movement = _temporal_rankings(records, parcellation, top_k=args.top_k)
    contrast_name = Path(args.dictionary).parent.name
    timestamp = datetime.now(timezone.utc)
    output = Path(args.output) if args.output else Path("outputs/asne_roi_reports") / contrast_name / f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}_roi_report.md"
    payload = {
        "contrast_name": contrast_name,
        "dictionary": args.dictionary,
        "eval_summary": args.eval_summary,
        "temporal_summary": args.temporal_summary,
        "parcellation": args.parcellation,
        "categories": [category_a, category_b],
        "parcel_count": parcellation["parcel_count"],
        "top_contrast_delta": top_contrast,
        "top_late_minus_early": top_late,
        "top_temporal_movement": top_movement,
        "disclaimer": ROI_DISCLAIMER,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(format_roi_report(payload), encoding="utf-8")
    return {"output_path": str(output), "payload": payload}


def format_roi_report(payload: dict[str, Any]) -> str:
    lines = [
        "# ASNE ROI / Parcel Contrast Report",
        "",
        ROI_DISCLAIMER,
        "",
        f"- Contrast: `{payload['contrast_name']}`",
        f"- Categories: `{payload['categories'][0]}` vs `{payload['categories'][1]}`",
        f"- Dictionary: `{payload['dictionary']}`",
        f"- Parcellation: `{payload['parcellation']}`",
        f"- Number of parcels: `{payload['parcel_count']}`",
        "",
        "## Top Parcels By Absolute Contrast Delta",
        "",
        _table(payload["top_contrast_delta"]),
        "",
        "## Top Parcels By Late Minus Early",
        "",
        _table(payload["top_late_minus_early"]),
        "",
        "## Top Parcels By Temporal Movement",
        "",
        _table(payload["top_temporal_movement"]),
    ]
    return "\n".join(lines).rstrip() + "\n"


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No temporal parcel data available."
    lines = ["| rank | parcel_id | parcel_name | delta | abs_delta |", "|---:|---|---|---:|---:|"]
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"| {index} | `{row['parcel_id']}` | {row.get('parcel_name', row['parcel_id'])} | "
            f"{float(row['delta']):.6f} | {float(row['abs_delta']):.6f} |"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate ROI/parcel-level ASNE contrast diagnostics.")
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--eval-summary", default=None)
    parser.add_argument("--temporal-summary", default=None)
    parser.add_argument("--parcellation", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--expected-vertices", type=int, default=20484)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = analyze_roi_contrast(args)
    print(f"ROI report: {result['output_path']}")


if __name__ == "__main__":
    main()
