"""Training orchestration helpers for local and Kubeflow-triggered runs.

The concrete trainer should fine-tune the selected YOLO model, write metrics,
and persist model artifacts outside Git.
"""

from dataclasses import dataclass


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
