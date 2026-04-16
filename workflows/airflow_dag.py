"""Airflow DAG skeleton for the MachLeData object detection workflow.

The final DAG should coordinate data preparation, training, evaluation, and
optional deployment steps while keeping task logic inside `src/machledata`.
"""

try:
    from airflow import DAG
except ImportError:  # pragma: no cover - keeps local imports working without Airflow.
    DAG = None


def build_dag():
    """Create the Airflow DAG once Airflow is installed in the runtime."""
    if DAG is None:
        return None
    return DAG(dag_id="machledata_object_detection")


dag = build_dag()

