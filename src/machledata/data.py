"""Data access helpers for BigQuery-backed object detection datasets.

Future implementations should load metadata from BigQuery, retrieve image
references, and prepare small YOLO-compatible training or test samples.
"""

from pathlib import Path


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
    return sorted(item for item in path.iterdir() if item.is_file())


def describe_bigquery_source(project_id: str, dataset: str) -> str:
    """Build a human-readable BigQuery source identifier.

    Args:
        project_id: Google Cloud project that owns the dataset.
        dataset: BigQuery dataset name.

    Returns:
        A `project.dataset` string for logs and documentation.
    """
    return f"{project_id}.{dataset}"

