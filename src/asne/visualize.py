from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_mock_outputs(output_root: str | Path = "outputs") -> dict[str, Any]:
    root = Path(output_root)
    baseline_path = root / "baseline" / "baseline.json"
    steered_dir = root / "steered"
    deltas_dir = root / "deltas"

    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline output: {baseline_path}")
    if not deltas_dir.exists():
        raise FileNotFoundError(f"Missing deltas directory: {deltas_dir}")

    baseline = _read_json(baseline_path)
    steered = {
        path.stem: _read_json(path)
        for path in sorted(steered_dir.glob("*.json"))
        if path.is_file()
    }
    deltas = {
        path.stem: _read_json(path)
        for path in sorted(deltas_dir.glob("*.json"))
        if path.is_file()
    }
    return {
        "baseline": baseline,
        "steered": steered,
        "deltas": deltas,
    }


def generate_delta_bar_chart(
    condition_id: str,
    summary: dict[str, Any],
    assets_dir: str | Path,
    filename_prefix: str | None = None,
) -> Path:
    output_dir = Path(assets_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{filename_prefix}_" if filename_prefix else ""
    chart_path = output_dir / f"{prefix}{condition_id}_top_deltas.png"

    top_indices = [str(index) for index in summary["top_indices"]]
    top_values = summary["top_values"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(top_indices, top_values)
    ax.set_title(f"{condition_id}: Top Response Deltas")
    ax.set_xlabel("Mock cortical response index")
    ax.set_ylabel("Response delta")
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return chart_path


def generate_condition_comparison_chart(
    delta_summaries: dict[str, dict[str, Any]],
    assets_dir: str | Path,
    filename_prefix: str | None = None,
) -> Path:
    output_dir = Path(assets_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{filename_prefix}_" if filename_prefix else ""
    chart_path = output_dir / f"{prefix}condition_comparison.png"

    condition_ids = sorted(delta_summaries)
    mean_values = [delta_summaries[condition_id]["mean_abs_delta"] for condition_id in condition_ids]
    max_values = [delta_summaries[condition_id]["max_abs_delta"] for condition_id in condition_ids]
    positions = range(len(condition_ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([position - width / 2 for position in positions], mean_values, width, label="mean_abs_delta")
    ax.bar([position + width / 2 for position in positions], max_values, width, label="max_abs_delta")
    ax.set_title("Condition Comparison")
    ax.set_xlabel("Steering condition")
    ax.set_ylabel("Response delta summary")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(condition_ids, rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return chart_path


def generate_roi_comparison_chart(
    condition_id: str,
    roi_summary: dict[str, dict[str, Any]],
    assets_dir: str | Path,
    filename_prefix: str | None = None,
) -> Path:
    output_dir = Path(assets_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{filename_prefix}_" if filename_prefix else ""
    chart_path = output_dir / f"{prefix}{condition_id}_roi_deltas.png"

    roi_ids = sorted(roi_summary)
    mean_values = [roi_summary[roi_id]["mean_abs_delta"] for roi_id in roi_ids]
    positions = range(len(roi_ids))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(positions, mean_values)
    ax.set_title(f"{condition_id}: Mock ROI Response Deltas")
    ax.set_xlabel("Placeholder ROI")
    ax.set_ylabel("mean_abs_delta")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(roi_ids, rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return chart_path


def generate_roi_condition_summary_chart(
    delta_summaries: dict[str, dict[str, Any]],
    assets_dir: str | Path,
    filename_prefix: str | None = None,
) -> Path:
    output_dir = Path(assets_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{filename_prefix}_" if filename_prefix else ""
    chart_path = output_dir / f"{prefix}roi_condition_summary.png"

    rows = [
        (condition_id, summary["top_rois"][0])
        for condition_id, summary in sorted(delta_summaries.items())
        if summary.get("top_rois")
    ]
    if not rows:
        raise ValueError("No top ROI summaries found for plotting")

    labels = [f"{condition_id}\n{top_roi['roi']}" for condition_id, top_roi in rows]
    values = [top_roi["mean_abs_delta"] for _, top_roi in rows]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(labels, values)
    ax.set_title("Top Mock ROI by Steering Condition")
    ax.set_xlabel("Condition and top placeholder ROI")
    ax.set_ylabel("mean_abs_delta")
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return chart_path


def generate_mock_output_plots(
    output_root: str | Path = "outputs",
    delta_paths: dict[str, str] | None = None,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    outputs = load_mock_outputs(root)
    assets_dir = root / "reports" / "assets"
    deltas = (
        {condition_id: _read_json(Path(path)) for condition_id, path in delta_paths.items()}
        if delta_paths
        else outputs["deltas"]
    )

    delta_summaries = {
        condition_id: payload["summary"]
        for condition_id, payload in deltas.items()
    }
    if not delta_summaries:
        raise ValueError("No delta summaries found for plotting")

    per_condition = {
        condition_id: str(
            generate_delta_bar_chart(
                condition_id,
                summary,
                assets_dir,
                filename_prefix=experiment_id,
            )
        )
        for condition_id, summary in sorted(delta_summaries.items())
    }
    comparison = str(
        generate_condition_comparison_chart(
            delta_summaries,
            assets_dir,
            filename_prefix=experiment_id,
        )
    )
    roi_per_condition = {
        condition_id: str(
            generate_roi_comparison_chart(
                condition_id,
                payload["roi_summary"],
                assets_dir,
                filename_prefix=experiment_id,
            )
        )
        for condition_id, payload in sorted(deltas.items())
        if payload.get("roi_summary")
    }
    roi_condition_summary = None
    if any(summary.get("top_rois") for summary in delta_summaries.values()):
        roi_condition_summary = str(
            generate_roi_condition_summary_chart(
                delta_summaries,
                assets_dir,
                filename_prefix=experiment_id,
            )
        )

    paths: dict[str, Any] = {
        "condition_comparison": comparison,
        "per_condition": per_condition,
    }
    if roi_per_condition:
        paths["roi_per_condition"] = roi_per_condition
    if roi_condition_summary:
        paths["roi_condition_summary"] = roi_condition_summary
    return paths
