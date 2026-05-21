from __future__ import annotations

from pathlib import Path
from typing import Any


SAFETY_DISCLAIMER = (
    "ASNE reports predicted cortical responses from computational models under defined "
    "steering conditions. Response deltas are not detected emotions, diagnoses, thoughts, "
    "or direct human neural measurements. Mock ROI labels are placeholder groupings for "
    "prototype workflow testing until a real cortical parcellation is integrated."
)


def _markdown_path(path: str | Path, report_dir: Path) -> str:
    image_path = Path(path)
    try:
        return image_path.relative_to(report_dir).as_posix()
    except ValueError:
        return image_path.as_posix()


def generate_markdown_report(
    stimulus: str,
    baseline_condition: str,
    delta_summaries: dict[str, dict[str, Any]],
    output_dir: str | Path = "outputs/reports",
    chart_paths: dict[str, Any] | None = None,
    roi_map_description: str | None = None,
    experiment_id: str | None = None,
    manifest_path: str | Path | None = None,
    adapter_type: str | None = None,
    steering_config_path: str | Path | None = None,
    roi_map_path: str | Path | None = None,
) -> Path:
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_stem = experiment_id or "mock_experiment"
    report_path = report_dir / f"{report_stem}.md"

    tested_conditions = sorted(delta_summaries)
    lines = [
        "# ASNE Mock Experiment Report",
        "",
    ]
    if experiment_id or manifest_path or adapter_type or steering_config_path or roi_map_path:
        lines.extend(
            [
                "## Run Metadata",
                "",
                f"- experiment_id: {experiment_id or 'not recorded'}",
                f"- adapter_type: {adapter_type or 'not recorded'}",
                f"- steering_config_path: {steering_config_path or 'not recorded'}",
                f"- roi_map_path: {roi_map_path or 'not recorded'}",
                f"- manifest_path: {manifest_path or 'not recorded'}",
                "",
            ]
        )
    lines.extend(
        [
        "## Stimulus",
        "",
        stimulus,
        "",
        "## Baseline Condition",
        "",
        baseline_condition,
        "",
        "## Steering Conditions Tested",
        "",
        ]
    )
    lines.extend(f"- {condition_id}" for condition_id in tested_conditions)
    if chart_paths and chart_paths.get("condition_comparison"):
        lines.extend(
            [
                "",
                "## Charts",
                "",
                f"![Condition comparison]({_markdown_path(chart_paths['condition_comparison'], report_dir)})",
                "",
            ]
        )
    if chart_paths and chart_paths.get("roi_condition_summary"):
        lines.extend(
            [
                f"![Top mock ROI by condition]({_markdown_path(chart_paths['roi_condition_summary'], report_dir)})",
                "",
            ]
        )

    if roi_map_description:
        lines.extend(
            [
                "## Mock ROI Note",
                "",
                roi_map_description,
                "",
            ]
        )

    lines.extend(
        [
            "",
            "## Condition Comparison",
            "",
            "| Steering condition | mean_abs_delta | max_abs_delta |",
            "| --- | ---: | ---: |",
        ]
    )
    for condition_id in tested_conditions:
        summary = delta_summaries[condition_id]
        lines.append(
            f"| {condition_id} | {summary['mean_abs_delta']:.6f} | {summary['max_abs_delta']:.6f} |"
        )

    conditions_with_rois = [
        condition_id for condition_id in tested_conditions if delta_summaries[condition_id].get("top_rois")
    ]
    if conditions_with_rois:
        lines.extend(
            [
                "",
                "## ROI-Level Delta Summary",
                "",
                "| Steering condition | Top mock ROI | mean_abs_delta | max_abs_delta | n_indices |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for condition_id in conditions_with_rois:
            top_roi = delta_summaries[condition_id]["top_rois"][0]
            lines.append(
                "| "
                f"{condition_id} | {top_roi['roi']} | {top_roi['mean_abs_delta']:.6f} | "
                f"{top_roi['max_abs_delta']:.6f} | {top_roi['n_indices']} |"
            )

    lines.extend(["", "## Delta Summaries", ""])

    for condition_id in tested_conditions:
        summary = delta_summaries[condition_id]
        per_condition_path = None
        roi_condition_path = None
        if chart_paths:
            per_condition_path = chart_paths.get("per_condition", {}).get(condition_id)
            roi_condition_path = chart_paths.get("roi_per_condition", {}).get(condition_id)
        lines.extend(
            [
                f"### {condition_id}",
                "",
                f"- mean_abs_delta: {summary['mean_abs_delta']:.6f}",
                f"- max_abs_delta: {summary['max_abs_delta']:.6f}",
                "",
            ]
        )
        if per_condition_path:
            lines.extend(
                [
                    f"![{condition_id} top response deltas]({_markdown_path(per_condition_path, report_dir)})",
                    "",
                ]
            )
        if roi_condition_path:
            lines.extend(
                [
                    f"![{condition_id} mock ROI response deltas]({_markdown_path(roi_condition_path, report_dir)})",
                    "",
                ]
            )

        if summary.get("top_rois"):
            lines.extend(
                [
                    "#### Top Mock ROIs",
                    "",
                    "| Mock ROI | mean_abs_delta | max_abs_delta | n_indices |",
                    "| --- | ---: | ---: | ---: |",
                ]
            )
            for roi in summary["top_rois"]:
                lines.append(
                    f"| {roi['roi']} | {roi['mean_abs_delta']:.6f} | "
                    f"{roi['max_abs_delta']:.6f} | {roi['n_indices']} |"
                )
            lines.append("")

        lines.extend(
            [
                "#### Top Delta Indices",
                "",
                "| Index | Response delta |",
                "| ---: | ---: |",
            ]
        )
        lines.extend(
            f"| {index} | {value:.6f} |"
            for index, value in zip(summary["top_indices"], summary["top_values"], strict=True)
        )
        lines.append("")

    lines.extend(["## Safety Disclaimer", "", SAFETY_DISCLAIMER, ""])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
