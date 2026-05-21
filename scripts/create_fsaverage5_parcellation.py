#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


EXPECTED_FSAVERAGE5_VERTICES = 20484
HEMISPHERE_VERTICES = 10242
SUPPORTED_ATLASES = ("hcp_mmp", "schaefer100", "schaefer200", "schaefer400")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create an fsaverage5 parcellation CSV for ASNE ROI/parcel reports."
    )
    parser.add_argument("--atlas", choices=SUPPORTED_ATLASES, default="hcp_mmp")
    parser.add_argument("--tribev2-package-path", default="./tribev2")
    parser.add_argument("--output", default=None)
    parser.add_argument("--allow-fetch", action="store_true", help="Allow MNE/TRIBE to fetch external atlas data.")
    parser.add_argument("--dry-run", action="store_true", help="Print dependency/source status without writing a CSV.")
    parser.add_argument(
        "--bilateral",
        action="store_true",
        help="Use bilateral parcel IDs when the source exposes matching left/right labels.",
    )
    return parser


def default_output_path(atlas: str) -> Path:
    return Path("data/parcellations") / f"fsaverage5_{atlas}.csv"


def describe_source(args: argparse.Namespace) -> dict[str, Any]:
    if args.atlas.startswith("schaefer"):
        return {
            "atlas": args.atlas,
            "available": False,
            "reason": (
                "No verified direct Schaefer fsaverage5 mapping is available in the current local "
                "dependencies. Install or provide a licensed fsaverage5 Schaefer source before generating "
                "scientific parcel labels."
            ),
        }
    hcp_available, hcp_reason = _check_hcp_dependencies(args.tribev2_package_path)
    return {
        "atlas": "hcp_mmp",
        "available": hcp_available,
        "reason": hcp_reason,
        "requires_allow_fetch": True,
        "source": "MNE HCP-MMP parcellation fetched through TRIBE's fsaverage5 helper.",
    }


def create_hcp_mmp_rows(
    *,
    tribev2_package_path: str | Path,
    allow_fetch: bool,
    bilateral: bool,
) -> list[dict[str, Any]]:
    if not allow_fetch:
        raise RuntimeError(
            "HCP-MMP generation may fetch external atlas data through MNE. Re-run with --allow-fetch "
            "after confirming the atlas source and license are acceptable for your use."
        )
    _add_tribev2_to_path(tribev2_package_path)
    from tribev2 import utils as tribe_utils  # type: ignore

    rows: list[dict[str, Any]] = []
    if bilateral:
        labels = tribe_utils.get_hcp_labels(mesh="fsaverage5", combine=False, hemi="both")
        vertex_to_label = [""] * EXPECTED_FSAVERAGE5_VERTICES
        for label, vertices in labels.items():
            for vertex in vertices:
                vertex_to_label[int(vertex)] = str(label)
        for vertex_index, label in enumerate(vertex_to_label):
            if not label:
                raise RuntimeError(f"HCP-MMP labels did not cover fsaverage5 vertex {vertex_index}.")
            rows.append(
                {
                    "vertex_index": vertex_index,
                    "parcel_id": _clean_id(label),
                    "parcel_name": label,
                }
            )
        return rows

    left = tribe_utils.get_hcp_labels(mesh="fsaverage5", combine=False, hemi="left")
    right = tribe_utils.get_hcp_labels(mesh="fsaverage5", combine=False, hemi="right")
    vertex_to_label = [""] * EXPECTED_FSAVERAGE5_VERTICES
    for label, vertices in left.items():
        for vertex in vertices:
            vertex_to_label[int(vertex)] = f"lh_{label}"
    for label, vertices in right.items():
        for vertex in vertices:
            vertex_to_label[int(vertex)] = f"rh_{label}"
    for vertex_index, label in enumerate(vertex_to_label):
        if not label:
            raise RuntimeError(f"HCP-MMP labels did not cover fsaverage5 vertex {vertex_index}.")
        rows.append(
            {
                "vertex_index": vertex_index,
                "parcel_id": _clean_id(label),
                "parcel_name": label,
            }
        )
    return rows


def write_rows_csv(rows: list[dict[str, Any]], output: str | Path) -> None:
    if len(rows) != EXPECTED_FSAVERAGE5_VERTICES:
        raise ValueError(f"Expected {EXPECTED_FSAVERAGE5_VERTICES} rows, got {len(rows)}.")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["vertex_index", "parcel_id", "parcel_name"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(args: argparse.Namespace) -> int:
    source = describe_source(args)
    output = Path(args.output) if args.output else default_output_path(args.atlas)
    print(f"Atlas: {source['atlas']}")
    print(f"Output: {output}")
    print(f"Available: {source['available']}")
    print(f"Source: {source.get('source', 'not available')}")
    print(f"Status: {source['reason']}")
    if args.dry_run:
        if source.get("requires_allow_fetch") and not args.allow_fetch:
            print("Dry run only: generation would require --allow-fetch.")
        return 0 if source["available"] else 2
    if not source["available"]:
        raise RuntimeError(source["reason"])
    if args.atlas != "hcp_mmp":
        raise RuntimeError("Only HCP-MMP generation is implemented without a separate verified atlas source.")
    rows = create_hcp_mmp_rows(
        tribev2_package_path=args.tribev2_package_path,
        allow_fetch=args.allow_fetch,
        bilateral=args.bilateral,
    )
    write_rows_csv(rows, output)
    print(f"Wrote {len(rows)} rows to {output}")
    return 0


def _check_hcp_dependencies(tribev2_package_path: str | Path) -> tuple[bool, str]:
    try:
        import mne  # noqa: F401
    except Exception as exc:
        return False, f"MNE is required for HCP-MMP fetching and is not importable: {exc}"
    try:
        _add_tribev2_to_path(tribev2_package_path)
        from tribev2 import utils as tribe_utils  # type: ignore
    except Exception as exc:
        return False, f"TRIBE v2 utilities are required and are not importable: {exc}"
    if not hasattr(tribe_utils, "get_hcp_labels"):
        return False, "TRIBE v2 utilities do not expose get_hcp_labels."
    return True, (
        "HCP-MMP helper is importable. Generation may download atlas files with "
        "mne.datasets.fetch_hcp_mmp_parcellation and therefore requires --allow-fetch."
    )


def _add_tribev2_to_path(path: str | Path) -> None:
    resolved = str(Path(path).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _clean_id(label: str) -> str:
    return (
        label.strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
        .lower()
    )


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
