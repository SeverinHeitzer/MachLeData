"""Data access helpers for BigQuery-backed object detection datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.cloud import bigquery


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class BigQueryDatasetConfig:
    """Configuration for the normalized images plus labels BigQuery schema."""

    project_id: str
    dataset: str
    images_table: str = "images"
    labels_table: str = "labels"
    image_id_column: str = "image_id"
    image_uri_column: str = "image_uri"
    split_column: str = "split"
    width_column: str = "width"
    height_column: str = "height"
    label_image_id_column: str = "image_id"
    class_column: str = "class_name"
    x_min_column: str = "x_min"
    y_min_column: str = "y_min"
    x_max_column: str = "x_max"
    y_max_column: str = "y_max"
    split: str | None = None
    limit: int | None = None


def load_sample_paths(samples_dir: str | Path = "data/samples") -> list[Path]:
    """Return local sample files used for tests and offline demos.

    Args:
        samples_dir: Directory containing tiny versioned sample assets.

    Returns:
        Sorted paths for files directly inside the sample directory.
    """
    path = Path(samples_dir)
    if not path.exists():
        return []
    return sorted(
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def describe_bigquery_source(project_id: str, dataset: str) -> str:
    """Build a human-readable BigQuery source identifier.

    Args:
        project_id: Google Cloud project that owns the dataset.
        dataset: BigQuery dataset name.

    Returns:
        A `project.dataset` string for logs and documentation.
    """
    return f"{project_id}.{dataset}"


def build_bigquery_annotations_query(config: BigQueryDatasetConfig) -> str:
    """Build the images plus labels query for the object detection dataset."""
    images_ref = _table_ref(config.project_id, config.dataset, config.images_table)
    labels_ref = _table_ref(config.project_id, config.dataset, config.labels_table)
    query = f"""
SELECT
  i.{_identifier(config.image_id_column)} AS image_id,
  i.{_identifier(config.image_uri_column)} AS image_uri,
  i.{_identifier(config.split_column)} AS split,
  i.{_identifier(config.width_column)} AS width,
  i.{_identifier(config.height_column)} AS height,
  l.{_identifier(config.class_column)} AS class_name,
  l.{_identifier(config.x_min_column)} AS x_min,
  l.{_identifier(config.y_min_column)} AS y_min,
  l.{_identifier(config.x_max_column)} AS x_max,
  l.{_identifier(config.y_max_column)} AS y_max
FROM `{images_ref}` AS i
LEFT JOIN `{labels_ref}` AS l
  ON i.{_identifier(config.image_id_column)}
    = l.{_identifier(config.label_image_id_column)}
WHERE (@split IS NULL OR i.{_identifier(config.split_column)} = @split)
ORDER BY image_id, class_name
""".strip()
    if config.limit is not None:
        query += "\nLIMIT @limit"
    return query


def load_bigquery_object_detection_rows(
    config: BigQueryDatasetConfig,
    *,
    client: Any | None = None,
) -> list[dict[str, Any]]:
    """Load normalized image and bounding-box rows from BigQuery."""
    active_client = client or bigquery.Client(project=config.project_id)
    query_parameters = [
        bigquery.ScalarQueryParameter("split", "STRING", config.split),
    ]
    if config.limit is not None:
        query_parameters.append(
            bigquery.ScalarQueryParameter("limit", "INT64", config.limit)
        )
    job_config = bigquery.QueryJobConfig(query_parameters=query_parameters)
    rows = active_client.query(
        build_bigquery_annotations_query(config),
        job_config=job_config,
    ).result()
    return [_row_to_detection_record(row) for row in rows]


def _row_to_detection_record(row: Any) -> dict[str, Any]:
    payload = dict(row.items()) if hasattr(row, "items") else dict(row)
    bbox = [
        payload.get("x_min"),
        payload.get("y_min"),
        payload.get("x_max"),
        payload.get("y_max"),
    ]
    return {
        "image_id": payload.get("image_id"),
        "image_uri": payload.get("image_uri"),
        "split": payload.get("split"),
        "width": payload.get("width"),
        "height": payload.get("height"),
        "class_name": payload.get("class_name"),
        "bbox": bbox,
    }


def _table_ref(project_id: str, dataset: str, table: str) -> str:
    return ".".join(_table_part(part) for part in (project_id, dataset, table))


def _identifier(value: str) -> str:
    if not value or not value.replace("_", "").isalnum():
        raise ValueError(f"Unsafe BigQuery identifier: {value!r}")
    return value


def _table_part(value: str) -> str:
    if not value or not value.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Unsafe BigQuery table reference part: {value!r}")
    return value
