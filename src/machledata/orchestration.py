"""Serializable orchestration helpers shared by Kubeflow and local CLIs."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from machledata.data import (
    BigQueryDatasetConfig,
    describe_bigquery_source,
    load_bigquery_object_detection_rows,
    load_sample_paths,
)
from machledata.metrics import summarize_detections
from machledata.train import create_training_run


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = os.getenv("MACHLEDATA_ARTIFACT_ROOT", "artifacts")
DEFAULT_DATA_CONFIG = "configs/data.yaml"
DEFAULT_MODEL_CONFIG = "configs/model.yaml"
DEFAULT_APP_CONFIG = "configs/app.yaml"


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a plain dictionary."""
    content = yaml.safe_load(_project_path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(content, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return _expand_env_placeholders(content)


def config_value(config: dict[str, Any], key: str, default: Any = None) -> Any:
    """Read a config value while ignoring unresolved ${ENV_VAR} placeholders."""
    value = config.get(key, default)
    if isinstance(value, str) and re.fullmatch(r"\$\{[^}]+\}", value.strip()):
        return default
    return value


def prepare_dataset(
    dataset_id: str,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    *,
    samples_dir: str | Path = "data/samples",
    project_id: str | None = None,
    bigquery_dataset: str | None = None,
    images_table: str | None = None,
    labels_table: str | None = None,
    split: str | None = None,
    max_rows: int | None = None,
    run_label: str | None = None,
    data_config_path: str | Path = DEFAULT_DATA_CONFIG,
    descriptor_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create a serializable dataset descriptor for downstream tasks."""
    label = _normalize_label(run_label or dataset_id)
    prepared_dir = Path(artifact_root) / "prepared" / label
    prepared_dir.mkdir(parents=True, exist_ok=True)

    sample_paths = load_sample_paths(_project_path(samples_dir))
    data_config = load_yaml_config(data_config_path)
    annotation_rows = _load_bigquery_rows(
        data_config=data_config,
        project_id=project_id,
        bigquery_dataset=bigquery_dataset,
        images_table=images_table,
        labels_table=labels_table,
        split=split,
        max_rows=max_rows,
    )
    annotation_path = None
    if annotation_rows:
        annotation_path = _artifact_sibling_path(
            descriptor_output_path,
            prepared_dir / "annotations.jsonl",
            ".annotations.jsonl",
        )
        _write_jsonl(annotation_path, annotation_rows)
    image_count = (
        len({row["image_id"] for row in annotation_rows})
        if annotation_rows
        else len(sample_paths)
    )
    source_uri = (
        describe_bigquery_source(project_id, bigquery_dataset)
        if project_id and bigquery_dataset
        else dataset_id
    )
    descriptor = {
        "dataset_id": dataset_id,
        "run_label": run_label or label,
        "prepared_dir": str(prepared_dir.resolve()),
        "samples_dir": str(_project_path(samples_dir)),
        "sample_count": image_count,
        "sample_files": [path.name for path in sample_paths],
        "annotation_count": len(annotation_rows),
        "annotations_path": str(annotation_path.resolve()) if annotation_path else None,
        "split": split or config_value(data_config, "split"),
        "source_type": "bigquery" if project_id and bigquery_dataset else "local",
        "source_uri": source_uri,
        "prepared_at": _utc_timestamp(),
        "config_snapshot": data_config,
    }
    descriptor_path = (
        Path(descriptor_output_path)
        if descriptor_output_path
        else prepared_dir / "dataset_descriptor.json"
    )
    _write_json(descriptor_path, descriptor)
    descriptor["descriptor_path"] = str(descriptor_path.resolve())
    return descriptor


def train_model(
    prepared_dataset: dict[str, Any],
    model_name: str,
    epochs: int,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    *,
    run_label: str | None = None,
    model_config_path: str | Path = DEFAULT_MODEL_CONFIG,
    model_artifact_path: str | Path | None = None,
    training_metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create placeholder training outputs and run metadata."""
    run_id = _build_run_id(model_name, run_label)
    training_dir = Path(artifact_root) / "training" / run_id
    training_dir.mkdir(parents=True, exist_ok=True)

    training_run = create_training_run(
        model_name=model_name,
        epochs=epochs,
        artifact_dir=str(training_dir.resolve()),
    )
    output_model_path = (
        Path(model_artifact_path)
        if model_artifact_path
        else training_dir / "model-placeholder.bin"
    )
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    output_model_path.write_text(
        f"Placeholder model artifact for {model_name} ({epochs} epochs)\n",
        encoding="utf-8",
    )

    run_metadata = {
        "run_id": run_id,
        "run_label": run_label or run_id,
        "model_name": training_run.model_name,
        "epochs": training_run.epochs,
        "artifact_dir": training_run.artifact_dir,
        "model_artifact_path": str(output_model_path.resolve()),
        "prepared_dataset_id": prepared_dataset["dataset_id"],
        "prepared_descriptor_path": prepared_dataset["descriptor_path"],
        "status": "completed",
        "trained_at": _utc_timestamp(),
        "config_snapshot": load_yaml_config(model_config_path),
    }
    metadata_path = (
        Path(training_metadata_path)
        if training_metadata_path
        else training_dir / "training_run.json"
    )
    _write_json(metadata_path, run_metadata)
    run_metadata["training_metadata_path"] = str(metadata_path.resolve())
    return run_metadata


def evaluate_model(
    prepared_dataset: dict[str, Any],
    training_run: dict[str, Any],
    *,
    min_detections_per_image: float = 0.1,
    model_artifact_path: str | Path | None = None,
    evaluation_output_path: str | Path | None = None,
    metrics_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate the placeholder training output and persist a summary."""
    total_images = int(prepared_dataset.get("sample_count", 0))
    total_detections = total_images * 2
    metrics = summarize_detections(
        total_images=total_images,
        total_detections=total_detections,
    )
    checked_artifact = Path(model_artifact_path or training_run["model_artifact_path"])
    model_artifact_exists = checked_artifact.exists()
    passed = model_artifact_exists and (
        metrics["detections_per_image"] >= min_detections_per_image
        or total_images == 0
    )

    evaluation_summary = {
        "run_id": training_run["run_id"],
        "dataset_id": prepared_dataset["dataset_id"],
        "artifact_checked": str(checked_artifact.resolve()),
        "artifact_exists": model_artifact_exists,
        "metrics": metrics,
        "thresholds": {
            "min_detections_per_image": min_detections_per_image,
        },
        "passed": passed,
        "evaluated_at": _utc_timestamp(),
    }
    evaluation_path = (
        Path(evaluation_output_path)
        if evaluation_output_path
        else Path(training_run["artifact_dir"]) / "evaluation_summary.json"
    )
    _write_json(evaluation_path, evaluation_summary)
    if metrics_output_path:
        _write_json(Path(metrics_output_path), metrics)
        evaluation_summary["metrics_path"] = str(Path(metrics_output_path).resolve())
    evaluation_summary["evaluation_path"] = str(evaluation_path.resolve())
    return evaluation_summary


def publish_artifact_manifest(
    prepared_dataset: dict[str, Any],
    training_run: dict[str, Any],
    evaluation_summary: dict[str, Any],
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    *,
    app_config_path: str | Path = DEFAULT_APP_CONFIG,
    manifest_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write deployment-facing metadata once evaluation has passed."""
    if not evaluation_summary.get("passed"):
        raise ValueError("Evaluation must pass before publishing artifact metadata")

    publish_dir = (
        Path(manifest_output_path).parent
        if manifest_output_path
        else Path(artifact_root) / "published" / training_run["run_id"]
    )
    publish_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": training_run["run_id"],
        "published_at": _utc_timestamp(),
        "dataset": {
            "dataset_id": prepared_dataset["dataset_id"],
            "source_uri": prepared_dataset["source_uri"],
            "descriptor_path": prepared_dataset["descriptor_path"],
        },
        "model": {
            "model_name": training_run["model_name"],
            "epochs": training_run["epochs"],
            "artifact_path": training_run["model_artifact_path"],
            "training_metadata_path": training_run["training_metadata_path"],
        },
        "evaluation": {
            "passed": evaluation_summary["passed"],
            "summary_path": evaluation_summary["evaluation_path"],
            "metrics": evaluation_summary["metrics"],
        },
        "serving": {
            "model_artifact_path": training_run["model_artifact_path"],
            "config_snapshot": load_yaml_config(app_config_path),
            "deployment_target": "gcp-serving-handoff",
        },
    }
    manifest_path = (
        Path(manifest_output_path)
        if manifest_output_path
        else publish_dir / "artifact_manifest.json"
    )
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path.resolve())
    return manifest


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _utc_timestamp() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


def _normalize_label(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or "run"


def _build_run_id(model_name: str, run_label: str | None) -> str:
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    parts = [_normalize_label(model_name)]
    if run_label:
        parts.append(_normalize_label(run_label))
    parts.append(timestamp)
    return "-".join(parts)


def _expand_env_placeholders(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env_placeholders(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_placeholders(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def _project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _artifact_sibling_path(
    output_path: str | Path | None,
    default_path: Path,
    suffix: str,
) -> Path:
    if not output_path:
        return default_path
    output = Path(output_path)
    return output.with_name(output.name + suffix)


def _load_bigquery_rows(
    *,
    data_config: dict[str, Any],
    project_id: str | None,
    bigquery_dataset: str | None,
    images_table: str | None,
    labels_table: str | None,
    split: str | None,
    max_rows: int | None,
) -> list[dict[str, Any]]:
    if (
        not project_id
        or not bigquery_dataset
        or not (images_table or data_config.get("images_table"))
    ):
        return []
    bbox_columns = data_config.get("bbox_columns", {})
    config = BigQueryDatasetConfig(
        project_id=project_id,
        dataset=bigquery_dataset,
        images_table=images_table
        or config_value(data_config, "images_table", "images"),
        labels_table=labels_table
        or config_value(data_config, "labels_table", "labels"),
        image_id_column=config_value(data_config, "image_id_column", "image_id"),
        image_uri_column=config_value(data_config, "image_uri_column", "image_uri"),
        split_column=config_value(data_config, "split_column", "split"),
        width_column=config_value(data_config, "width_column", "width"),
        height_column=config_value(data_config, "height_column", "height"),
        label_image_id_column=config_value(
            data_config,
            "label_image_id_column",
            "image_id",
        ),
        class_column=config_value(data_config, "class_column", "class_name"),
        x_min_column=config_value(bbox_columns, "x_min", "x_min"),
        y_min_column=config_value(bbox_columns, "y_min", "y_min"),
        x_max_column=config_value(bbox_columns, "x_max", "x_max"),
        y_max_column=config_value(bbox_columns, "y_max", "y_max"),
        split=split or config_value(data_config, "split"),
        limit=max_rows,
    )
    return load_bigquery_object_detection_rows(config)
