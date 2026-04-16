"""Tests for local data helpers and future BigQuery access seams."""

from machledata.data import describe_bigquery_source


def test_describe_bigquery_source() -> None:
    """Builds a stable source identifier for logs and docs."""
    assert describe_bigquery_source("project", "dataset") == "project.dataset"

