"""Airflow DAG for the MachLeData object detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from machledata.orchestration import (
    DEFAULT_APP_CONFIG,
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_DATA_CONFIG,
    DEFAULT_MODEL_CONFIG,
    config_value,
    evaluate_model,
    load_yaml_config,
    prepare_dataset,
    publish_artifact_manifest,
    train_model,
)

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:  # pragma: no cover - keeps local imports working without Airflow.
    DAG = None
    PythonOperator = None


DAG_ID = "machledata_ml_pipeline"
TASK_SEQUENCE = (
    "prepare_data",
    "train_model",
    "evaluate_model",
    "publish_artifact_metadata",
)


@dataclass
class FallbackTask:
    """Minimal task object used to inspect the graph without Airflow installed."""

    task_id: str
    upstream_task_ids: set[str] = field(default_factory=set)
    downstream_task_ids: set[str] = field(default_factory=set)

    def set_downstream(self, other: "FallbackTask") -> "FallbackTask":
        self.downstream_task_ids.add(other.task_id)
        other.upstream_task_ids.add(self.task_id)
        return other

    def __rshift__(self, other: "FallbackTask") -> "FallbackTask":
        return self.set_downstream(other)


@dataclass
class FallbackDag:
    """Minimal DAG object used for tests and local imports without Airflow."""

    dag_id: str
    description: str
    schedule: Any
    catchup: bool
    params: dict[str, Any]
    task_dict: dict[str, FallbackTask] = field(default_factory=dict)

    @property
    def tasks(self) -> list[FallbackTask]:
        return list(self.task_dict.values())


def _default_params() -> dict[str, Any]:
    data_config = load_yaml_config(DEFAULT_DATA_CONFIG)
    model_config = load_yaml_config(DEFAULT_MODEL_CONFIG)
    return {
        "dataset_id": config_value(data_config, "dataset", "local-sample-dataset"),
        "samples_dir": config_value(data_config, "samples_dir", "data/samples"),
        "project_id": config_value(data_config, "project_id"),
        "bigquery_dataset": config_value(data_config, "dataset"),
        "model_name": config_value(model_config, "model_name", "yolov8n"),
        "epochs": int(config_value(model_config, "epochs", 10)),
        "artifact_root": DEFAULT_ARTIFACT_ROOT,
        "run_label": "manual",
        "min_detections_per_image": 0.1,
    }


def _resolve_runtime_config(context: dict[str, Any]) -> dict[str, Any]:
    params = dict(context.get("params", {}))
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None) or {}
    params.update(conf)
    return params


def _prepare_task(**context: Any) -> dict[str, Any]:
    cfg = _resolve_runtime_config(context)
    return prepare_dataset(
        dataset_id=cfg["dataset_id"],
        artifact_root=cfg["artifact_root"],
        samples_dir=cfg["samples_dir"],
        project_id=cfg.get("project_id"),
        bigquery_dataset=cfg.get("bigquery_dataset"),
        run_label=cfg.get("run_label"),
    )


def _train_task(**context: Any) -> dict[str, Any]:
    cfg = _resolve_runtime_config(context)
    prepared_dataset = context["ti"].xcom_pull(task_ids="prepare_data")
    return train_model(
        prepared_dataset=prepared_dataset,
        model_name=cfg["model_name"],
        epochs=int(cfg["epochs"]),
        artifact_root=cfg["artifact_root"],
        run_label=cfg.get("run_label"),
    )


def _evaluate_task(**context: Any) -> dict[str, Any]:
    cfg = _resolve_runtime_config(context)
    prepared_dataset = context["ti"].xcom_pull(task_ids="prepare_data")
    training_run = context["ti"].xcom_pull(task_ids="train_model")
    return evaluate_model(
        prepared_dataset=prepared_dataset,
        training_run=training_run,
        min_detections_per_image=float(cfg["min_detections_per_image"]),
    )


def _publish_task(**context: Any) -> dict[str, Any]:
    cfg = _resolve_runtime_config(context)
    prepared_dataset = context["ti"].xcom_pull(task_ids="prepare_data")
    training_run = context["ti"].xcom_pull(task_ids="train_model")
    evaluation_summary = context["ti"].xcom_pull(task_ids="evaluate_model")
    return publish_artifact_manifest(
        prepared_dataset=prepared_dataset,
        training_run=training_run,
        evaluation_summary=evaluation_summary,
        artifact_root=cfg["artifact_root"],
        app_config_path=DEFAULT_APP_CONFIG,
    )


def build_dag() -> Any:
    """Create the Airflow DAG or an inspectable fallback when Airflow is absent."""
    params = _default_params()
    description = (
        "Manual, backfill-ready MachLeData pipeline that prepares data, trains a "
        "model, evaluates outputs, and writes a deployment-facing artifact manifest."
    )
    if DAG is None or PythonOperator is None:
        dag = FallbackDag(
            dag_id=DAG_ID,
            description=description,
            schedule=None,
            catchup=True,
            params=params,
        )
        prepare = FallbackTask("prepare_data")
        train = FallbackTask("train_model")
        evaluate = FallbackTask("evaluate_model")
        publish = FallbackTask("publish_artifact_metadata")
        prepare >> train >> evaluate >> publish
        dag.task_dict = {
            task.task_id: task
            for task in (prepare, train, evaluate, publish)
        }
        return dag

    with DAG(
        dag_id=DAG_ID,
        description=description,
        schedule=None,
        start_date=datetime(2024, 1, 1),
        catchup=True,
        render_template_as_native_obj=True,
        params=params,
        tags=["machledata", "mlops", "manual"],
    ) as dag:
        prepare = PythonOperator(task_id="prepare_data", python_callable=_prepare_task)
        train = PythonOperator(task_id="train_model", python_callable=_train_task)
        evaluate = PythonOperator(
            task_id="evaluate_model",
            python_callable=_evaluate_task,
        )
        publish = PythonOperator(
            task_id="publish_artifact_metadata",
            python_callable=_publish_task,
        )

        prepare >> train >> evaluate >> publish

    return dag


dag = build_dag()
