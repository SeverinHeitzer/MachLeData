# MachLeData

MachLeData is a course project skeleton for a YOLO-style object detection pipeline with a small MLOps-oriented layout. The repository now includes reusable package modules, CLI entry points, a FastAPI app, a Streamlit dashboard, config files, tests, and a Docker-based Airflow pipeline that prepares data, trains, evaluates, and publishes deployment-facing metadata.

## Current Status

The project is intentionally lightweight today. The scaffolding that already works includes:

- Python package entry point in `src/machledata/`
- FastAPI health and placeholder prediction endpoints in `apps/api.py`
- Streamlit dashboard starter in `apps/dashboard.py`
- Local CLI scripts for training, prediction, and evaluation in `scripts/`
- Config files in `configs/`
- Smoke tests for data helpers, inference, and API behavior in `tests/`
- A manual, backfill-ready Airflow DAG in `workflows/airflow_dag.py`

The model training, dataset integration, and real inference logic are still stubs, which makes this a good foundation for iterative implementation.

## Repository Layout

- `src/machledata/`: package code for data access, model config, training, metrics, and inference
- `apps/`: FastAPI and Streamlit entry points
- `scripts/`: local command-line helpers for train, predict, and evaluate flows
- `configs/`: committed YAML configuration files
- `tests/`: pytest coverage for package and app smoke tests
- `workflows/`: orchestration entry points such as the Airflow DAG
- `docs/`: architecture and data notes
- `docker/`: Dockerfile and Compose scaffolding
- `data/samples/`: tiny tracked fixtures only

## Quick Start

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the project with development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run the package smoke test:

```bash
python -m machledata
```

## Common Commands

Run the test suite:

```bash
python -m pytest
```

Run the FastAPI app locally:

```bash
python -m uvicorn apps.api:app --reload
```

Run the Streamlit dashboard:

```bash
streamlit run apps/dashboard.py
```

Smoke-test the CLI scripts:

```bash
python scripts/train.py
python scripts/predict.py
python scripts/evaluate.py
```

Run the local Airflow stack:

```bash
docker compose -f docker/docker-compose.yml up airflow-init
docker compose -f docker/docker-compose.yml up airflow-webserver airflow-scheduler airflow-triggerer
```

## Package Overview

The core package is designed to keep application and orchestration code thin:

- `machledata.data`: sample-file discovery and BigQuery source description helpers
- `machledata.model`: typed model configuration for future YOLO integration
- `machledata.train`: training run metadata creation
- `machledata.infer`: shared detection schema and prediction interface
- `machledata.metrics`: simple summary metrics for demos and smoke tests
- `machledata.orchestration`: serializable prepare, train, evaluate, and publish contracts for Airflow and CLIs

## Airflow Pipeline

The orchestration entry point is `workflows/airflow_dag.py`. It defines one manual DAG with four linear stages:

1. `prepare_data`
2. `train_model`
3. `evaluate_model`
4. `publish_artifact_metadata`

The publish step stops at writing validated, deployment-facing metadata such as model artifact location, config snapshots, and evaluation outputs. It does not deploy the FastAPI app or dashboard.

### Local Airflow Run

Initialize the Airflow metadata database and admin user:

```bash
docker compose -f docker/docker-compose.yml up airflow-init
```

Start the Airflow services:

```bash
docker compose -f docker/docker-compose.yml up airflow-webserver airflow-scheduler airflow-triggerer
```

Open the UI at `http://localhost:8080` and sign in with `admin` / `admin`.

Trigger `machledata_ml_pipeline` manually and optionally pass DAG run config such as:

```json
{
  "dataset_id": "machledata-demo",
  "model_name": "yolov8n",
  "epochs": 5,
  "artifact_root": "/opt/airflow/artifacts",
  "run_label": "course-demo"
}
```

Published outputs are written under `artifacts/` locally and are mounted into the Airflow containers at `/opt/airflow/artifacts`.

## API and Dashboard

The FastAPI app currently exposes:

- `GET /health`: returns `{"status": "ok"}`
- `POST /predict`: returns an empty detection list until inference is implemented

The Streamlit dashboard is a starter page for demos, prediction inspection, and model metrics.

The Airflow pipeline is the orchestration path for preparing data, training a model, evaluating outputs, and publishing deployment-facing metadata. It stops at artifact and manifest creation so the API or dashboard can consume validated outputs later.

## Configuration

Committed YAML files in `configs/` define the initial project shape:

- `configs/model.yaml`: model name, image size, batch size, epochs, artifact directory
- `configs/data.yaml`: BigQuery-oriented dataset settings and sample directory
- `configs/app.yaml`: app title, model artifact path, and confidence threshold

Keep secrets and environment-specific values out of Git. Use `.env` or deployment configuration for credentials and sensitive settings.

For the Airflow pipeline, the most relevant variables are:

- `GOOGLE_CLOUD_PROJECT`
- `BIGQUERY_DATASET`
- `MODEL_ARTIFACT_PATH`
- `AIRFLOW_UID` for local Docker file ownership if needed

## Development Notes

- Target Python 3.10 or newer
- Keep package logic in `src/machledata/` and leave apps, scripts, and workflows thin
- Avoid committing generated artifacts, large datasets, caches, virtual environments, or trained model weights
- Prefer adding tests alongside new behavior, especially for inference, metrics, API routes, and data helpers

## Related Docs

- `docs/architecture.md`
- `docs/data.md`
- `data/README.md`
- `data/samples/README.md`
