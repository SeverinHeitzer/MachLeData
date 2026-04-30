"""Tests for local data helpers and future BigQuery access seams."""

from pathlib import Path

from machledata.data import describe_bigquery_source, load_sample_paths


def test_describe_bigquery_source() -> None:
    """Builds a stable source identifier for logs and docs."""
    assert describe_bigquery_source("project", "dataset") == "project.dataset"


def test_load_sample_paths_filters_non_images(tmp_path: Path) -> None:
    """Only image fixtures should enter the detection dataset seam."""
    (tmp_path / "image.jpg").write_text("sample", encoding="utf-8")
    (tmp_path / "README.md").write_text("docs", encoding="utf-8")

    assert load_sample_paths(tmp_path) == [tmp_path / "image.jpg"]
