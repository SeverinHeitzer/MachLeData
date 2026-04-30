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
    images_table: str,
    labels_table: str,
    split: str,
    max_rows: int,
    artifact_root: str,
    run_label: str,
    prepared_dataset: dsl.Output[dsl.Dataset],
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
            "--images-table",
            images_table,
            "--labels-table",
            labels_table,
            "--split",
            split,
            "--max-rows",
            max_rows,
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
    prepared_dataset: dsl.Input[dsl.Dataset],
    model_name: str,
    epochs: int,
    artifact_root: str,
    run_label: str,
    model_artifact: dsl.Output[dsl.Model],
    training_metadata: dsl.Output[dsl.Artifact],
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
            "--model-output-path",
            model_artifact.path,
            "--metadata-output-path",
            training_metadata.path,
        ],
    )


@dsl.container_component
def evaluate_model_component(
    prepared_dataset: dsl.Input[dsl.Dataset],
    model_artifact: dsl.Input[dsl.Model],
    training_run: dsl.Input[dsl.Artifact],
    min_detections_per_image: float,
    evaluation_summary: dsl.Output[dsl.Artifact],
    evaluation_metrics: dsl.Output[dsl.Metrics],
) -> dsl.ContainerSpec:
    return dsl.ContainerSpec(
        image=PIPELINE_IMAGE,
        command=["python", "-m", "machledata.pipeline_steps", "evaluate-model"],
        args=[
            "--prepared-dataset-path",
            prepared_dataset.path,
            "--training-run-path",
            training_run.path,
            "--model-artifact-path",
            model_artifact.path,
            "--min-detections-per-image",
            min_detections_per_image,
            "--evaluation-output-path",
            evaluation_summary.path,
            "--metrics-output-path",
            evaluation_metrics.path,
        ],
    )


@dsl.container_component
def publish_artifact_metadata_component(
    prepared_dataset: dsl.Input[dsl.Dataset],
    model_artifact: dsl.Input[dsl.Model],
    training_run: dsl.Input[dsl.Artifact],
    evaluation_summary: dsl.Input[dsl.Artifact],
    evaluation_metrics: dsl.Input[dsl.Metrics],
    artifact_root: str,
    artifact_manifest: dsl.Output[dsl.Artifact],
) -> dsl.ContainerSpec:
    _ = model_artifact, evaluation_metrics
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
            "--manifest-output-path",
            artifact_manifest.path,
        ],
    )


@dsl.pipeline(name=PIPELINE_NAME)
def machledata_pipeline(
    dataset_id: str = "local-demo",
    samples_dir: str = "data/samples",
    project_id: str = "",
    bigquery_dataset: str = "",
    images_table: str = "images",
    labels_table: str = "labels",
    split: str = "train",
    max_rows: int = 0,
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
        images_table=images_table,
        labels_table=labels_table,
        split=split,
        max_rows=max_rows,
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
        model_artifact=trained.outputs["model_artifact"],
        training_run=trained.outputs["training_metadata"],
        min_detections_per_image=min_detections_per_image,
    )
    publish_artifact_metadata_component(
        prepared_dataset=prepared.outputs["prepared_dataset"],
        model_artifact=trained.outputs["model_artifact"],
        training_run=trained.outputs["training_metadata"],
        evaluation_summary=evaluated.outputs["evaluation_summary"],
        evaluation_metrics=evaluated.outputs["evaluation_metrics"],
        artifact_root=artifact_root,
    )
