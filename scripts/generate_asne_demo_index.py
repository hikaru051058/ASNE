#!/usr/bin/env python
from __future__ import annotations

import argparse
import html
import os
from pathlib import Path
from typing import Any


DISCLAIMER = (
    "ASNE compares predicted TRIBE cortical response signatures. It is not measured brain activity, "
    "emotion detection, diagnosis, or measurement of a person's mental state."
)

V03_ROWS = [
    {
        "contrast": "contradiction_vs_consistency_paired",
        "status": "weak/deprioritized",
        "vertex_top1": "0.40",
        "parcel_top1": "0.50",
        "temporal_late": "0.50",
        "temporal_final": "0.40",
        "read": "Did not survive v0.3 expansion; retained as an unstable contrast case.",
    },
    {
        "contrast": "expected_vs_unexpected_paired",
        "status": "stable",
        "vertex_top1": "0.80",
        "parcel_top1": "0.90",
        "temporal_late": "0.90",
        "temporal_final": "0.80",
        "read": "Strongest current contrast; parcel and late temporal views improve the result.",
    },
    {
        "contrast": "cause_effect_valid_vs_invalid_paired",
        "status": "stable/viable",
        "vertex_top1": "0.70",
        "parcel_top1": "0.80",
        "temporal_late": "0.50",
        "temporal_final": "0.40",
        "majority": "0.70",
        "read": "Viable semantic contrast; parcel scoring is the strongest static view.",
    },
    {
        "contrast": "approach_vs_static_paired",
        "status": "pending / low priority",
        "vertex_top1": "n/a",
        "parcel_top1": "n/a",
        "temporal_late": "n/a",
        "temporal_final": "n/a",
        "read": "Pending in v0.3; prior smaller runs suggest text/TTS is a weak fit.",
    },
]

V02_ROWS = [
    {
        "contrast": "contradiction_vs_consistency_paired",
        "vertex_top1": "0.83",
        "parcel_top1": "0.83",
        "parcel_status": "preserved",
    },
    {
        "contrast": "expected_vs_unexpected_paired",
        "vertex_top1": "0.83",
        "parcel_top1": "0.83",
        "parcel_status": "preserved",
    },
    {
        "contrast": "approach_vs_static_paired",
        "vertex_top1": "0.50",
        "parcel_top1": "0.50",
        "parcel_status": "preserved weak result",
    },
    {
        "contrast": "cause_effect_valid_vs_invalid_paired",
        "vertex_top1": "0.83",
        "parcel_top1": "0.83",
        "parcel_status": "preserved",
    },
]


def discover_artifacts(
    *,
    reports_root: str | Path = "outputs/asne_reports",
    roi_root: str | Path = "outputs/asne_roi_reports",
    temporal_root: str | Path = "outputs/asne_temporal_evals",
    notebook_path: str | Path = "notebooks/asne_contrast_explorer.ipynb",
) -> dict[str, Any]:
    reports_root = Path(reports_root)
    roi_root = Path(roi_root)
    temporal_root = Path(temporal_root)
    artifacts = {
        "main_reports": [
            reports_root / "semantic_contrast_benchmark_v03.html",
            reports_root / "semantic_contrast_benchmark_v03.md",
            reports_root / "semantic_contrast_report_v0.html",
            reports_root / "semantic_contrast_report_v0.md",
            reports_root / "vertex_vs_parcel_scoring_v0.md",
        ],
        "roi_reports": [],
        "temporal_reports": [],
        "notebook": Path(notebook_path),
    }
    if roi_root.exists():
        artifacts["roi_reports"] = sorted(roi_root.glob("*/roi_report.md"))
    if temporal_root.exists():
        artifacts["temporal_reports"] = latest_temporal_reports(temporal_root)
    return artifacts


def latest_temporal_reports(temporal_root: str | Path) -> list[Path]:
    root = Path(temporal_root)
    latest: list[Path] = []
    for contrast_dir in sorted(path for path in root.glob("*") if path.is_dir()):
        reports = sorted(contrast_dir.glob("*_temporal_report.md"), key=lambda path: path.stat().st_mtime)
        if reports:
            latest.append(reports[-1])
    for group_dir in sorted(path for path in root.glob("*") if path.is_dir()):
        for contrast_dir in sorted(path for path in group_dir.glob("*") if path.is_dir()):
            reports = sorted(contrast_dir.glob("*_temporal_report.md"), key=lambda path: path.stat().st_mtime)
            if reports:
                latest.append(reports[-1])
    return sorted(set(latest))


def generate_index(
    *,
    output: str | Path = "outputs/asne_reports/index.html",
    reports_root: str | Path = "outputs/asne_reports",
    roi_root: str | Path = "outputs/asne_roi_reports",
    temporal_root: str | Path = "outputs/asne_temporal_evals",
    notebook_path: str | Path = "notebooks/asne_contrast_explorer.ipynb",
) -> Path:
    output_path = Path(output)
    artifacts = discover_artifacts(
        reports_root=reports_root,
        roi_root=roi_root,
        temporal_root=temporal_root,
        notebook_path=notebook_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(artifacts, output_path), encoding="utf-8")
    return output_path


def render_html(artifacts: dict[str, Any], output_path: Path) -> str:
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            "<title>ASNE v0.3 Demo Report</title>",
            f"<style>{_css()}</style>",
            "</head>",
            "<body>",
            '<main class="page">',
            "<header>",
            "<p class=\"eyebrow\">Static demo report</p>",
            "<h1>ASNE v0.3 Demo Report</h1>",
            "<p class=\"lede\">ASNE is a TRIBE-backed in-silico semantic contrast analysis system for comparing predicted cortical response signatures across controlled stimuli.</p>",
            f'<p class="disclaimer">{html.escape(DISCLAIMER)}</p>',
            '<p class="warning">v0.3-lite uses ten held-out examples per contrast. Treat this as a stability check, not a production classifier benchmark.</p>',
            "</header>",
            '<section class="card result-card">',
            "<h2>Main result</h2>",
            "<p>Expected/unexpected remained stable under v0.3 expansion; cause/effect remains viable, especially with parcel scoring; contradiction/consistency became unstable and is now weak/deprioritized.</p>",
            "</section>",
            '<section class="card">',
            "<h2>v0.3 semantic contrast benchmark</h2>",
            _v03_table(),
            "</section>",
            '<section class="card">',
            "<h2>Previous milestone: v0.2 parcel scoring</h2>",
            "<p class=\"muted\">The earlier v0.2 milestone showed that HCP-MMP parcel scoring preserved the small-suite vertex benchmark. v0.3 is now the headline stability benchmark.</p>",
            _v02_vertex_parcel_table(),
            "</section>",
            '<section class="card">',
            "<h2>Artifacts</h2>",
            _artifact_links(artifacts, output_path),
            "</section>",
            '<section class="card">',
            "<h2>How to reproduce</h2>",
            '<pre><code>python scripts/run_asne_v03_benchmark.py --skip-build\npython scripts/generate_asne_demo_index.py</code></pre>',
            "</section>",
            "</main>",
            "</body>",
            "</html>",
        ]
    ) + "\n"


def _v03_table() -> str:
    rows = [
        "<table>",
        "<thead><tr><th>contrast</th><th>status</th><th>vertex top1</th><th>parcel top1</th><th>late accuracy</th><th>final accuracy</th><th>read</th></tr></thead>",
        "<tbody>",
    ]
    for row in V03_ROWS:
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(row['contrast'])}</code></td>"
            f"<td>{html.escape(row['status'])}</td>"
            f"<td>{row['vertex_top1']}</td>"
            f"<td>{row['parcel_top1']}</td>"
            f"<td>{row['temporal_late']}</td>"
            f"<td>{row['temporal_final']}</td>"
            f"<td>{html.escape(row['read'])}</td>"
            "</tr>"
        )
    rows.extend(["</tbody>", "</table>"])
    return "\n".join(rows)


def _v02_vertex_parcel_table() -> str:
    rows = [
        "<table>",
        "<thead><tr><th>contrast</th><th>vertex top1</th><th>parcel top1</th><th>status</th></tr></thead>",
        "<tbody>",
    ]
    for row in V02_ROWS:
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(row['contrast'])}</code></td>"
            f"<td>{row['vertex_top1']}</td>"
            f"<td>{row['parcel_top1']}</td>"
            f"<td>{html.escape(row['parcel_status'])}</td>"
            "</tr>"
        )
    rows.extend(["</tbody>", "</table>"])
    return "\n".join(rows)


def _artifact_links(artifacts: dict[str, Any], output_path: Path) -> str:
    sections = [
        _link_group("Core reports", artifacts["main_reports"], output_path),
        _link_group("ROI reports", artifacts["roi_reports"], output_path),
        _link_group("Temporal reports", artifacts["temporal_reports"], output_path),
        _link_group("Notebook", [artifacts["notebook"]], output_path),
    ]
    return "\n".join(sections)


def _link_group(title: str, paths: list[Path], output_path: Path) -> str:
    existing = [Path(path) for path in paths if Path(path).exists()]
    lines = [f"<h3>{html.escape(title)}</h3>"]
    if not existing:
        lines.append('<p class="muted">No matching artifacts found.</p>')
        return "\n".join(lines)
    lines.append("<ul>")
    for path in existing:
        label = str(path)
        href = _relative_href(path, output_path)
        lines.append(f'<li><a href="{html.escape(href)}">{html.escape(label)}</a></li>')
    lines.append("</ul>")
    return "\n".join(lines)


def _relative_href(path: str | Path, output_path: Path) -> str:
    return os.path.relpath(Path(path).resolve(), output_path.parent.resolve())


def _css() -> str:
    return """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
  margin: 0;
  background: #f7f7f5;
  color: #191919;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.5;
}
.page { max-width: 1120px; margin: 0 auto; padding: 48px 24px; }
header { margin-bottom: 24px; }
.eyebrow { color: #666; font-size: 13px; text-transform: uppercase; letter-spacing: .08em; margin: 0 0 8px; }
h1 { font-size: 42px; line-height: 1.1; margin: 0 0 14px; letter-spacing: 0; }
h2 { font-size: 22px; margin: 0 0 14px; letter-spacing: 0; }
h3 { font-size: 15px; margin: 20px 0 8px; letter-spacing: 0; }
.lede { max-width: 780px; font-size: 18px; margin: 0 0 16px; color: #333; }
.disclaimer { max-width: 880px; border-left: 3px solid #191919; padding-left: 14px; color: #333; }
.card {
  background: #fff;
  border: 1px solid #deded9;
  border-radius: 8px;
  padding: 22px;
  margin: 18px 0;
  box-shadow: 0 1px 2px rgba(0,0,0,.04);
}
.result-card { border-color: #191919; }
.result-card p { font-size: 19px; margin: 0; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { border-bottom: 1px solid #e6e6e1; padding: 10px 8px; text-align: left; vertical-align: top; }
th { color: #555; font-weight: 600; background: #fafafa; }
code {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: .92em;
  background: #f1f1ee;
  padding: 1px 4px;
  border-radius: 4px;
}
pre {
  margin: 0;
  overflow: auto;
  background: #f1f1ee;
  border-radius: 6px;
  padding: 14px;
}
pre code { background: transparent; padding: 0; }
a { color: #111; text-decoration: underline; text-underline-offset: 2px; }
ul { margin: 8px 0 0; padding-left: 20px; }
li { margin: 4px 0; }
.muted { color: #777; margin: 0; }
@media (max-width: 760px) {
  .page { padding: 28px 14px; }
  h1 { font-size: 32px; }
  table { display: block; overflow-x: auto; white-space: nowrap; }
}
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the static ASNE demo report index.")
    parser.add_argument("--output", default="outputs/asne_reports/index.html")
    parser.add_argument("--reports-root", default="outputs/asne_reports")
    parser.add_argument("--roi-root", default="outputs/asne_roi_reports")
    parser.add_argument("--temporal-root", default="outputs/asne_temporal_evals")
    parser.add_argument("--notebook-path", default="notebooks/asne_contrast_explorer.ipynb")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = generate_index(
        output=args.output,
        reports_root=args.reports_root,
        roi_root=args.roi_root,
        temporal_root=args.temporal_root,
        notebook_path=args.notebook_path,
    )
    print(f"Demo index: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
