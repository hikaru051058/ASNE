from __future__ import annotations

import importlib.util
from pathlib import Path


def test_demo_index_generation_creates_file(tmp_path: Path) -> None:
    module = _load_script()
    reports = tmp_path / "reports"
    roi = tmp_path / "roi" / "contrast_a"
    temporal = tmp_path / "temporal" / "contrast_a"
    notebook = tmp_path / "notebook.ipynb"
    reports.mkdir()
    roi.mkdir(parents=True)
    temporal.mkdir(parents=True)
    (reports / "semantic_contrast_report_v0.html").write_text("<html></html>", encoding="utf-8")
    (reports / "semantic_contrast_benchmark_v03.html").write_text("<html></html>", encoding="utf-8")
    (reports / "semantic_contrast_benchmark_v03.md").write_text("# report", encoding="utf-8")
    (reports / "vertex_vs_parcel_scoring_v0.md").write_text("# report", encoding="utf-8")
    (roi / "roi_report.md").write_text("# roi", encoding="utf-8")
    (temporal / "20260101T000000Z_temporal_report.md").write_text("# temporal", encoding="utf-8")
    notebook.write_text("{}", encoding="utf-8")
    output = tmp_path / "reports" / "index.html"

    path = module.generate_index(
        output=output,
        reports_root=reports,
        roi_root=tmp_path / "roi",
        temporal_root=tmp_path / "temporal",
        notebook_path=notebook,
    )
    html = path.read_text(encoding="utf-8")

    assert path.exists()
    assert "ASNE | Artificial Semantic Neural Evaluation" in html
    assert "<h1>ASNE</h1>" in html
    assert "A benchmark and visualization framework" in html
    assert "ASNE compares predicted TRIBE cortical response signatures" in html
    assert "https://github.com/hikaru051058/ASNE" in html
    assert "Expected/unexpected remained stable under v0.3 expansion" in html
    assert "expected_vs_unexpected_paired" in html
    assert "stable" in html
    assert "cause_effect_valid_vs_invalid_paired" in html
    assert "contradiction_vs_consistency_paired" in html
    assert "weak/deprioritized" in html
    assert "Demo reports and artifacts" in html
    assert "View v0.3 Report" in html
    assert "Previous milestone: v0.2 parcel scoring" in html
    assert "semantic_contrast_benchmark_v03.html" in html
    assert "semantic_contrast_report_v0.html" in html
    assert "roi_report.md" in html
    assert "temporal_report.md" in html
    assert "notebook.ipynb" in html


def test_demo_index_missing_artifacts_do_not_crash(tmp_path: Path) -> None:
    module = _load_script()
    output = tmp_path / "index.html"

    path = module.generate_index(
        output=output,
        reports_root=tmp_path / "missing_reports",
        roi_root=tmp_path / "missing_roi",
        temporal_root=tmp_path / "missing_temporal",
        notebook_path=tmp_path / "missing.ipynb",
    )
    html = path.read_text(encoding="utf-8")

    assert path.exists()
    assert "No matching artifacts found." in html
    assert "<h1>ASNE</h1>" in html


def _load_script():
    path = Path("scripts/generate_asne_demo_index.py")
    spec = importlib.util.spec_from_file_location("generate_asne_demo_index", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
