"""Training orchestration helpers for local and Kubeflow-triggered runs.

This module handles fine-tuning of YOLO models, writing metrics, and
persisting model artifacts outside Git.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from ultralytics import YOLO

from machledata.model import ModelConfig, build_model_config, save_model


@dataclass(frozen=True)
class TrainingRun:
    """Summary metadata for a model training run."""

    model_name: str
    epochs: int
    artifact_dir: str


def create_training_run(
    model_name: str,
    epochs: int,
    artifact_dir: str,
) -> TrainingRun:
    """Create training metadata before invoking the real trainer."""
    return TrainingRun(model_name=model_name, epochs=epochs, artifact_dir=artifact_dir)


def train_yolo_model(
    config: ModelConfig,
    dataset_path: str | Path,
    epochs: int = 10,
    batch_size: int = 8,
    artifact_dir: str | Path | None = None,
    device: str | None = None,
) -> tuple[YOLO, dict]:
    """Train a YOLO model on the provided dataset.
    ...
    """
    # Use a writable cache/config dir for Ultralytics.
    import os
    os.environ["YOLO_CONFIG_DIR"] = "/tmp/yolo_config"
    os.makedirs("/tmp/yolo_config", exist_ok=True)
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load base model
    model = YOLO(config.model_name)

    # Prepare artifact directory
    if artifact_dir:
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

    # Run training
    results = model.train(
        data=str(dataset_path),
        epochs=epochs,
        imgsz=config.image_size,
        batch=batch_size,
        device=device,
        patience=10,  # Early stopping patience
        save=True,
        project=str(artifact_dir) if artifact_dir else None,
        name="yolo_train",
        verbose=True,
    )

    # Extract metrics
    metrics = {
        "model_name": config.model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": device,
        "training_status": "completed",
    }

    # Add training results if available
    if hasattr(results, 'results_dict'):
        metrics.update(results.results_dict)

    # Save model to artifact directory if specified
    if artifact_dir:
        model_path = artifact_dir / "best_model.pt"
        save_model(model, model_path)
        metrics["saved_model_path"] = str(model_path.resolve())

    return model, metrics


def validate_model(
    model: YOLO,
    dataset_path: str | Path,
    device: str | None = None,
) -> dict:
    """Validate a trained YOLO model.

    Args:
        model: Trained YOLO model.
        dataset_path: Path to validation dataset.
        device: Device to validate on.

    Returns:
        Validation metrics dictionary.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = model.val(
        data=str(dataset_path),
        device=device,
        verbose=False,
    )

    validation_metrics = {
        "validation_status": "completed",
    }

    if hasattr(results, 'results_dict'):
        validation_metrics.update(results.results_dict)

    return validation_metrics

