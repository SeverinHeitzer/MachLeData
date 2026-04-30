"""Kubeflow Pipelines definition for the MachLeData ML workflow."""

import os

from kfp import dsl


PIPELINE_IMAGE = os.getenv("MACHLEDATA_PIPELINE_IMAGE", "machledata:local")
PIPELINE_NAME = "machledata-ml-pipeline"
TASK_SEQUENCE = (
    "prepare-data",
    "train-model",
    "evaluate-model",
    "publish-artifact-metadata",
)


@dsl.container_component
def prepare_data_component(
    dataset_id: str,
    samples_dir: str,
    project_id: str,
    bigquery_dataset: str,
    artifact_root: str,
    run_label: str,
    prepared_dataset: dsl.Output[dsl.Artifact],
) -> dsl.ContainerSpec:
    return dsl.ContainerSpec(
        image=PIPELINE_IMAGE,
        command=["python", "-m", "machledata.pipeline_steps", "prepare-data"],
        args=[
            "--dataset-id",
            dataset_id,
            "--samples-dir",
            samples_dir,
            "--project-id",
            project_id,
            "--bigquery-dataset",
            bigquery_dataset,
            "--artifact-root",
            artifact_root,
            "--run-label",
            run_label,
            "--output-path",
            prepared_dataset.path,
        ],
    )


@dsl.container_component
def train_model_component(
    prepared_dataset: dsl.Input[dsl.Artifact],
    model_name: str,
    epochs: int,
    artifact_root: str,
    run_label: str,
    training_run: dsl.Output[dsl.Artifact],
) -> dsl.ContainerSpec:
    return dsl.ContainerSpec(
        image=PIPELINE_IMAGE,
        command=["python", "-m", "machledata.pipeline_steps", "train-model"],
        args=[
            "--prepared-dataset-path",
            prepared_dataset.path,
            "--model-name",
            model_name,
            "--epochs",
            epochs,
            "--artifact-root",
            artifact_root,
            "--run-label",
            run_label,
            "--output-path",
            training_run.path,
        ],
    )


@dsl.container_component
def evaluate_model_component(
    prepared_dataset: dsl.Input[dsl.Artifact],
    training_run: dsl.Input[dsl.Artifact],
    min_detections_per_image: float,
    evaluation_summary: dsl.Output[dsl.Artifact],
) -> dsl.ContainerSpec:
    return dsl.ContainerSpec(
        image=PIPELINE_IMAGE,
        command=["python", "-m", "machledata.pipeline_steps", "evaluate-model"],
        args=[
            "--prepared-dataset-path",
            prepared_dataset.path,
            "--training-run-path",
            training_run.path,
            "--min-detections-per-image",
            min_detections_per_image,
            "--output-path",
            evaluation_summary.path,
        ],
    )


@dsl.container_component
def publish_artifact_metadata_component(
    prepared_dataset: dsl.Input[dsl.Artifact],
    training_run: dsl.Input[dsl.Artifact],
    evaluation_summary: dsl.Input[dsl.Artifact],
    artifact_root: str,
    artifact_manifest: dsl.Output[dsl.Artifact],
) -> dsl.ContainerSpec:
    return dsl.ContainerSpec(
        image=PIPELINE_IMAGE,
        command=[
            "python",
            "-m",
            "machledata.pipeline_steps",
            "publish-artifact-metadata",
        ],
        args=[
            "--prepared-dataset-path",
            prepared_dataset.path,
            "--training-run-path",
            training_run.path,
            "--evaluation-summary-path",
            evaluation_summary.path,
            "--artifact-root",
            artifact_root,
            "--output-path",
            artifact_manifest.path,
        ],
    )


@dsl.pipeline(name=PIPELINE_NAME)
def machledata_pipeline(
    dataset_id: str = "local-demo",
    samples_dir: str = "data/samples",
    project_id: str = "",
    bigquery_dataset: str = "",
    model_name: str = "yolov8n",
    epochs: int = 10,
    artifact_root: str = "/tmp/machledata-artifacts",
    run_label: str = "manual",
    min_detections_per_image: float = 0.1,
) -> None:
    """Prepare data, create model artifacts, evaluate, and publish metadata."""
    prepared = prepare_data_component(
        dataset_id=dataset_id,
        samples_dir=samples_dir,
        project_id=project_id,
        bigquery_dataset=bigquery_dataset,
        artifact_root=artifact_root,
        run_label=run_label,
    )
    trained = train_model_component(
        prepared_dataset=prepared.outputs["prepared_dataset"],
        model_name=model_name,
        epochs=epochs,
        artifact_root=artifact_root,
        run_label=run_label,
    )
    evaluated = evaluate_model_component(
        prepared_dataset=prepared.outputs["prepared_dataset"],
        training_run=trained.outputs["training_run"],
        min_detections_per_image=min_detections_per_image,
    )
    publish_artifact_metadata_component(
        prepared_dataset=prepared.outputs["prepared_dataset"],
        training_run=trained.outputs["training_run"],
        evaluation_summary=evaluated.outputs["evaluation_summary"],
        artifact_root=artifact_root,
    )
