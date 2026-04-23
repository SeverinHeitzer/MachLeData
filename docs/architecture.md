# Architecture Notes

## End-to-End Shape

MachLeData is structured around a thin-interface pattern:

- `src/machledata/` owns reusable data, training, evaluation, and orchestration code.
- `scripts/` provide local smoke-testable entry points over the same package seams.
- `workflows/airflow_dag.py` orchestrates the ML pipeline without embedding business logic.
- `apps/api.py` and `apps/dashboard.py` are the downstream consumers of publish-ready artifacts and metadata.

## Airflow Pipeline

The Docker-based Airflow stack is the local orchestration runtime for the course deliverable. The canonical DAG is `machledata_ml_pipeline`, and it is manual-trigger-first with `catchup=True` so the structure is compatible with later backfills.

Pipeline stages:

1. `prepare_data`
2. `train_model`
3. `evaluate_model`
4. `publish_artifact_metadata`

Stage contract:

- `prepare_data` writes a dataset descriptor with source information, sample inventory, and a config snapshot.
- `train_model` writes training metadata plus a model artifact placeholder path.
- `evaluate_model` writes an evaluation summary with metrics and a pass/fail gate.
- `publish_artifact_metadata` writes a serving-facing manifest only when evaluation passes.

Simple DAG diagram:

```text
prepare_data -> train_model -> evaluate_model -> publish_artifact_metadata
```

## Deployment Handoff

The first Airflow version intentionally stops at validated, publish-ready outputs. That handoff boundary is the `artifact_manifest.json` produced under `artifacts/published/<run_id>/`.

That manifest is designed to feed a later GCP-oriented serving path, for example:

- model artifacts stored in GCS or another object store
- metadata read by a FastAPI inference service
- reports surfaced through the Streamlit dashboard
- future deployment orchestration via Cloud Run, Composer, or CI/CD pipelines

Website hosting and rollout automation are intentionally outside the initial DAG scope.

## Local Runtime

`docker/docker-compose.yml` runs:

- Postgres for Airflow metadata
- `airflow-init` for database migration and user bootstrapping
- Airflow webserver, scheduler, and triggerer
- the existing API container for local app work

The Airflow containers mount the repo `workflows/`, `src/`, `configs/`, and `scripts/` directories plus writable local directories for logs and artifacts.
