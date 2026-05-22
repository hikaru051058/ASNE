#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


REPORT_FILES = [
    "index.html",
    "semantic_contrast_benchmark_v03.html",
    "semantic_contrast_report_v0.html",
    "semantic_contrast_benchmark_v03.md",
    "semantic_contrast_report_v0.md",
    "vertex_vs_parcel_scoring_v0.md",
]

GENERATED_TOP_LEVEL = set(REPORT_FILES) | {"README.md"}
GENERATED_DIRS = ["roi", "temporal"]

DISCLAIMER = (
    "ASNE compares predicted TRIBE cortical response signatures. It is not measured brain activity, "
    "emotion detection, diagnosis, or measurement of a person's mental state."
)


def export_docs_site(
    *,
    docs_root: str | Path = "docs",
    reports_root: str | Path = "outputs/asne_reports",
    roi_root: str | Path = "outputs/asne_roi_reports",
    temporal_root: str | Path = "outputs/asne_temporal_evals",
) -> dict[str, list[Path]]:
    docs_root = Path(docs_root)
    reports_root = Path(reports_root)
    roi_root = Path(roi_root)
    temporal_root = Path(temporal_root)

    docs_root.mkdir(parents=True, exist_ok=True)
    clean_generated_docs(docs_root)

    copied: dict[str, list[Path]] = {"reports": [], "roi": [], "temporal": [], "generated": []}
    for filename in REPORT_FILES:
        source = reports_root / filename
        destination = docs_root / filename
        if source.exists():
            copy_file(source, destination)
            copied["reports"].append(destination)

    index_path = docs_root / "index.html"
    if index_path.exists():
        rewrite_index_links(index_path)

    if roi_root.exists():
        for report in sorted(roi_root.glob("*/roi_report.md")):
            contrast = report.parent.name
            destination = docs_root / "roi" / contrast / "roi_report.md"
            copy_file(report, destination)
            copied["roi"].append(destination)

    if temporal_root.exists():
        for report in sorted(temporal_root.glob("**/*_temporal_report.md")):
            relative_parts = report.relative_to(temporal_root).parts
            if len(relative_parts) < 2:
                continue
            contrast = relative_parts[-2]
            destination = docs_root / "temporal" / contrast / report.name
            copy_file(report, destination)
            copied["temporal"].append(destination)

    readme_path = docs_root / "README.md"
    readme_path.write_text(render_generated_readme(), encoding="utf-8")
    copied["generated"].append(readme_path)
    return copied


def clean_generated_docs(docs_root: Path) -> None:
    for name in GENERATED_TOP_LEVEL:
        path = docs_root / name
        if path.exists() and path.is_file():
            path.unlink()
    for name in GENERATED_DIRS:
        path = docs_root / name
        if path.exists():
            shutil.rmtree(path)


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def rewrite_index_links(index_path: Path) -> None:
    content = index_path.read_text(encoding="utf-8")
    replacements = {
        "../asne_roi_reports/": "roi/",
        "../asne_temporal_evals/": "temporal/",
        "../asne_reports/": "",
        "outputs/asne_reports/": "",
        "outputs/asne_roi_reports/": "roi/",
        "outputs/asne_temporal_evals/": "temporal/",
    }
    for old, new in replacements.items():
        content = content.replace(old, new)
    content = re.sub(r'href="temporal/contrasts_v03/([^/]+)/([^"]+)"', r'href="temporal/\1/\2"', content)
    index_path.write_text(content, encoding="utf-8")


def render_generated_readme() -> str:
    return (
        "# ASNE GitHub Pages Export\n\n"
        "This folder contains generated static report copies exported from ASNE outputs.\n\n"
        "Do not manually edit generated report copies in this folder. Regenerate them with:\n\n"
        "```bash\n"
        "python scripts/export_asne_docs_site.py\n"
        "```\n\n"
        f"{DISCLAIMER}\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export ASNE static report artifacts into docs/ for GitHub Pages.")
    parser.add_argument("--docs-root", default="docs")
    parser.add_argument("--reports-root", default="outputs/asne_reports")
    parser.add_argument("--roi-root", default="outputs/asne_roi_reports")
    parser.add_argument("--temporal-root", default="outputs/asne_temporal_evals")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    copied = export_docs_site(
        docs_root=args.docs_root,
        reports_root=args.reports_root,
        roi_root=args.roi_root,
        temporal_root=args.temporal_root,
    )
    print("ASNE docs site export complete.")
    for key, paths in copied.items():
        print(f"{key}: {len(paths)}")
        for path in paths[:8]:
            print(f"  {path}")
        if len(paths) > 8:
            print(f"  ... {len(paths) - 8} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
