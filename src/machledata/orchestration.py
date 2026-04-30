"""Serializable orchestration helpers shared by Kubeflow and local CLIs."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from machledata.data import describe_bigquery_source, load_sample_paths
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
    run_label: str | None = None,
    data_config_path: str | Path = DEFAULT_DATA_CONFIG,
) -> dict[str, Any]:
    """Create a serializable dataset descriptor for downstream tasks."""
    label = _normalize_label(run_label or dataset_id)
    prepared_dir = Path(artifact_root) / "prepared" / label
    prepared_dir.mkdir(parents=True, exist_ok=True)

    sample_paths = load_sample_paths(_project_path(samples_dir))
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
        "sample_count": len(sample_paths),
        "sample_files": [path.name for path in sample_paths],
        "source_type": "bigquery" if project_id and bigquery_dataset else "local",
        "source_uri": source_uri,
        "prepared_at": _utc_timestamp(),
        "config_snapshot": load_yaml_config(data_config_path),
    }
    descriptor_path = prepared_dir / "dataset_descriptor.json"
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
    model_artifact_path = training_dir / "model-placeholder.bin"
    model_artifact_path.write_text(
        f"Placeholder model artifact for {model_name} ({epochs} epochs)\n",
        encoding="utf-8",
    )

    run_metadata = {
        "run_id": run_id,
        "run_label": run_label or run_id,
        "model_name": training_run.model_name,
        "epochs": training_run.epochs,
        "artifact_dir": training_run.artifact_dir,
        "model_artifact_path": str(model_artifact_path.resolve()),
        "prepared_dataset_id": prepared_dataset["dataset_id"],
        "prepared_descriptor_path": prepared_dataset["descriptor_path"],
        "status": "completed",
        "trained_at": _utc_timestamp(),
        "config_snapshot": load_yaml_config(model_config_path),
    }
    metadata_path = training_dir / "training_run.json"
    _write_json(metadata_path, run_metadata)
    run_metadata["training_metadata_path"] = str(metadata_path.resolve())
    return run_metadata


def evaluate_model(
    prepared_dataset: dict[str, Any],
    training_run: dict[str, Any],
    *,
    min_detections_per_image: float = 0.1,
) -> dict[str, Any]:
    """Evaluate the placeholder training output and persist a summary."""
    total_images = int(prepared_dataset.get("sample_count", 0))
    total_detections = total_images * 2
    metrics = summarize_detections(
        total_images=total_images,
        total_detections=total_detections,
    )
    model_artifact_exists = Path(training_run["model_artifact_path"]).exists()
    passed = model_artifact_exists and (
        metrics["detections_per_image"] >= min_detections_per_image
        or total_images == 0
    )

    evaluation_summary = {
        "run_id": training_run["run_id"],
        "dataset_id": prepared_dataset["dataset_id"],
        "artifact_checked": training_run["model_artifact_path"],
        "artifact_exists": model_artifact_exists,
        "metrics": metrics,
        "thresholds": {
            "min_detections_per_image": min_detections_per_image,
        },
        "passed": passed,
        "evaluated_at": _utc_timestamp(),
    }
    evaluation_path = Path(training_run["artifact_dir"]) / "evaluation_summary.json"
    _write_json(evaluation_path, evaluation_summary)
    evaluation_summary["evaluation_path"] = str(evaluation_path.resolve())
    return evaluation_summary


def publish_artifact_manifest(
    prepared_dataset: dict[str, Any],
    training_run: dict[str, Any],
    evaluation_summary: dict[str, Any],
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    *,
    app_config_path: str | Path = DEFAULT_APP_CONFIG,
) -> dict[str, Any]:
    """Write deployment-facing metadata once evaluation has passed."""
    if not evaluation_summary.get("passed"):
        raise ValueError("Evaluation must pass before publishing artifact metadata")

    publish_dir = Path(artifact_root) / "published" / training_run["run_id"]
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
    manifest_path = publish_dir / "artifact_manifest.json"
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path.resolve())
    return manifest


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
