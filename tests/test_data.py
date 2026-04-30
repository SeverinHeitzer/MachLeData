"""Tests for local data helpers and BigQuery access seams."""

from pathlib import Path

from machledata.data import (
    BigQueryDatasetConfig,
    build_bigquery_annotations_query,
    describe_bigquery_source,
    load_bigquery_object_detection_rows,
    load_sample_paths,
)


def test_describe_bigquery_source() -> None:
    """Builds a stable source identifier for logs and docs."""
    assert describe_bigquery_source("project", "dataset") == "project.dataset"


def test_load_sample_paths_filters_non_images(tmp_path: Path) -> None:
    """Only image fixtures should enter the detection dataset seam."""
    (tmp_path / "image.jpg").write_text("sample", encoding="utf-8")
    (tmp_path / "README.md").write_text("docs", encoding="utf-8")

    assert load_sample_paths(tmp_path) == [tmp_path / "image.jpg"]


def test_bigquery_images_labels_query_uses_normalized_schema() -> None:
    """The BigQuery contract joins image metadata with per-box labels."""
    query = build_bigquery_annotations_query(
        BigQueryDatasetConfig(
            project_id="project-id",
            dataset="dataset",
            images_table="images",
            labels_table="labels",
            split="train",
            limit=10,
        )
    )

    assert "`project-id.dataset.images`" in query
    assert "`project-id.dataset.labels`" in query
    assert "LEFT JOIN" in query
    assert "AS image_uri" in query
    assert "AS class_name" in query
    assert "AS x_min" in query
    assert "LIMIT @limit" in query


def test_load_bigquery_rows_maps_pixel_xyxy_records() -> None:
    """BigQuery rows are normalized into the dataset descriptor row contract."""

    class FakeQueryResult:
        def result(self):
            return [
                {
                    "image_id": "img-1",
                    "image_uri": "gs://bucket/image.jpg",
                    "split": "train",
                    "width": 640,
                    "height": 480,
                    "class_name": "car",
                    "x_min": 1.0,
                    "y_min": 2.0,
                    "x_max": 3.0,
                    "y_max": 4.0,
                }
            ]

    class FakeClient:
        def __init__(self):
            self.query_text = ""
            self.job_config = None

        def query(self, query_text, job_config=None):
            self.query_text = query_text
            self.job_config = job_config
            return FakeQueryResult()

    client = FakeClient()
    rows = load_bigquery_object_detection_rows(
        BigQueryDatasetConfig(
            project_id="project-id",
            dataset="dataset",
            split="train",
        ),
        client=client,
    )

    assert rows == [
        {
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "class_name": "car",
            "height": 480,
            "image_id": "img-1",
            "image_uri": "gs://bucket/image.jpg",
            "split": "train",
            "width": 640,
        }
    ]
    assert "@split" in client.query_text
    assert client.job_config.query_parameters[0].name == "split"
