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
    """Initializes and prepares the raw dataset for training."""
    return dsl.ContainerSpec(
        image=PIPELINE_IMAGE,
        command=["python", "-m", "machledata.pipeline_steps", "prepare-data"],
        args=[
            "--dataset-id", dataset_id,
            "--samples-dir", samples_dir,
            "--project-id", project_id,
            "--bigquery-dataset", bigquery_dataset,
            "--images-table", images_table,
            "--labels-table", labels_table,
            "--split", split,
            "--max-rows", str(max_rows), 
            "--artifact-root", artifact_root,
            "--run-label", run_label,
            "--output-path", prepared_dataset.path,
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
    """Trains the ML model using the prepared dataset."""
    return dsl.ContainerSpec(
        image=PIPELINE_IMAGE,
        command=["python", "-m", "machledata.pipeline_steps", "train-model"],
        args=[
            "--prepared-dataset-path", prepared_dataset.path,
            "--model-name", model_name,
            "--epochs", str(epochs), 
            "--artifact-root", artifact_root,
            "--run-label", run_label,
            "--model-output-path", model_artifact.path,
            "--metadata-output-path", training_metadata.path,
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
    """Evaluates model performance and generates metrics."""
    return dsl.ContainerSpec(
        image=PIPELINE_IMAGE,
        command=["python", "-m", "machledata.pipeline_steps", "evaluate-model"],
        args=[
            "--prepared-dataset-path", prepared_dataset.path,
            "--training-run-path", training_run.path,
            "--model-artifact-path", model_artifact.path,
            "--min-detections-per-image", str(min_detections_per_image), 
            "--evaluation-output-path", evaluation_summary.path,
            "--metrics-output-path", evaluation_metrics.path,
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
    """Compiles and publishes the final manifest for the run."""
    return dsl.ContainerSpec(
        image=PIPELINE_IMAGE,
        command=["python", "-m", "machledata.pipeline_steps", "publish-artifact-metadata"],
        args=[
            "--prepared-dataset-path", prepared_dataset.path,
            "--training-run-path", training_run.path,
            "--evaluation-summary-path", evaluation_summary.path,
            "--artifact-root", artifact_root,
            "--manifest-output-path", artifact_manifest.path,
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
    use_gpu: bool = False,
) -> None:
    """Orchestrates the full ML workflow DAG."""
    prep_task = prepare_data_component(
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
    )  # type: ignore
    prep_task.set_caching_options(False)

    train_task = train_model_component(
        prepared_dataset=prep_task.outputs["prepared_dataset"],
        model_name=model_name,
        epochs=epochs,
        artifact_root=artifact_root,
        run_label=run_label,
    )  # type: ignore
    train_task.set_caching_options(False)
    #if use_gpu:
        #train_task = train_task.set_accelerator_type("NVIDIA_TESLA_T4").set_accelerator_limit("1")

    eval_task = evaluate_model_component(
        prepared_dataset=prep_task.outputs["prepared_dataset"],
        model_artifact=train_task.outputs["model_artifact"],
        training_run=train_task.outputs["training_metadata"],
        min_detections_per_image=min_detections_per_image,
    )  # type: ignore
    eval_task.set_caching_options(False)

    publish_task = publish_artifact_metadata_component(
        prepared_dataset=prep_task.outputs["prepared_dataset"],
        model_artifact=train_task.outputs["model_artifact"],
        training_run=train_task.outputs["training_metadata"],
        evaluation_summary=eval_task.outputs["evaluation_summary"],
        evaluation_metrics=eval_task.outputs["evaluation_metrics"],
        artifact_root=artifact_root,
    )  # type: ignore
    publish_task.set_caching_options(False)