from __future__ import annotations

from pathlib import Path

from scripts.export_asne_docs_site import export_docs_site


def test_export_creates_docs_index_and_reports(tmp_path: Path) -> None:
    reports = tmp_path / "outputs" / "asne_reports"
    docs = tmp_path / "docs"
    reports.mkdir(parents=True)
    (reports / "index.html").write_text(
        '<html><body>ASNE compares predicted TRIBE cortical response signatures. '
        '<a href="../asne_roi_reports/contrast_a/roi_report.md">roi</a></body></html>',
        encoding="utf-8",
    )
    (reports / "semantic_contrast_benchmark_v03.html").write_text("<html>v03</html>", encoding="utf-8")
    (reports / "semantic_contrast_benchmark_v03.md").write_text("# v03", encoding="utf-8")

    copied = export_docs_site(docs_root=docs, reports_root=reports, roi_root=tmp_path / "missing_roi", temporal_root=tmp_path / "missing_temporal")

    assert (docs / "index.html").exists()
    assert (docs / "semantic_contrast_benchmark_v03.html").exists()
    assert (docs / "semantic_contrast_benchmark_v03.md").exists()
    assert "roi/contrast_a/roi_report.md" in (docs / "index.html").read_text(encoding="utf-8")
    assert copied["reports"]


def test_export_copies_roi_reports(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    roi = tmp_path / "roi" / "contrast_a"
    docs = tmp_path / "docs"
    reports.mkdir()
    roi.mkdir(parents=True)
    (reports / "index.html").write_text("ASNE compares predicted TRIBE cortical response signatures.", encoding="utf-8")
    (roi / "roi_report.md").write_text("# ROI", encoding="utf-8")

    export_docs_site(docs_root=docs, reports_root=reports, roi_root=tmp_path / "roi", temporal_root=tmp_path / "missing_temporal")

    assert (docs / "roi" / "contrast_a" / "roi_report.md").read_text(encoding="utf-8") == "# ROI"


def test_export_copies_temporal_reports_and_flattens_group(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    temporal = tmp_path / "temporal" / "contrasts_v03" / "contrast_a"
    docs = tmp_path / "docs"
    reports.mkdir()
    temporal.mkdir(parents=True)
    (reports / "index.html").write_text("ASNE compares predicted TRIBE cortical response signatures.", encoding="utf-8")
    (temporal / "20260101T000000Z_temporal_report.md").write_text("# temporal", encoding="utf-8")

    export_docs_site(docs_root=docs, reports_root=reports, roi_root=tmp_path / "missing_roi", temporal_root=tmp_path / "temporal")

    assert (docs / "temporal" / "contrast_a" / "20260101T000000Z_temporal_report.md").exists()


def test_export_does_not_crash_when_optional_reports_missing(tmp_path: Path) -> None:
    docs = tmp_path / "docs"

    export_docs_site(docs_root=docs, reports_root=tmp_path / "missing_reports", roi_root=tmp_path / "missing_roi", temporal_root=tmp_path / "missing_temporal")

    assert (docs / "README.md").exists()
    assert not (docs / "index.html").exists()


def test_export_docs_readme_contains_safety_disclaimer(tmp_path: Path) -> None:
    docs = tmp_path / "docs"

    export_docs_site(docs_root=docs, reports_root=tmp_path / "missing_reports", roi_root=tmp_path / "missing_roi", temporal_root=tmp_path / "missing_temporal")

    text = (docs / "README.md").read_text(encoding="utf-8")
    assert "not measured brain activity" in text
    assert "python scripts/export_asne_docs_site.py" in text
