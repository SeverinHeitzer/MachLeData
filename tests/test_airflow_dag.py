"""Tests for the MachLeData Airflow DAG definition."""

from workflows.airflow_dag import DAG_ID, TASK_SEQUENCE, dag


def test_airflow_dag_loads_without_airflow_installed() -> None:
    """Importing the DAG module should stay safe in non-Airflow environments."""
    assert dag is not None
    assert dag.dag_id == DAG_ID


def test_airflow_dag_exposes_expected_task_graph() -> None:
    """The pipeline should keep the intended linear stage ordering."""
    task_ids = [task.task_id for task in dag.tasks]

    assert task_ids == list(TASK_SEQUENCE)
    assert dag.task_dict["prepare_data"].downstream_task_ids == {"train_model"}
    assert dag.task_dict["train_model"].downstream_task_ids == {"evaluate_model"}
    assert dag.task_dict["evaluate_model"].downstream_task_ids == {
        "publish_artifact_metadata"
    }
