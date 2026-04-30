# MachLeData

MachLeData is a course project for a YOLO-style object detection pipeline with an MLOps-oriented layout. The repository includes reusable package modules, CLI entry points, a FastAPI app, a Streamlit dashboard, configuration files, tests, a Docker-based Airflow pipeline, and a GitHub Actions workflow for test, image build, and deployment automation.

## Current Status

The project is intentionally lightweight, but the main MLOps seams are in place. The pieces that already work include:

- Python package entry point in `src/machledata/`
- FastAPI health and placeholder prediction endpoints in `apps/api.py`
- Streamlit dashboard for image upload, API-backed prediction display, detection metrics, and optional annotated images in `apps/dashboard.py`
- Local CLI scripts for training, prediction, and evaluation in `scripts/`
- Config files in `configs/`
- Pytest coverage for data helpers, inference, orchestration, Airflow DAG structure, and API behavior in `tests/`
- A manual, backfill-ready Airflow DAG in `workflows/airflow_dag.py`
- Docker Compose services for Airflow, Postgres, and the API in `docker/docker-compose.yml`
- GitHub Actions CI/CD in `.github/workflows/deploy.yaml`

The real YOLO training, dataset integration, and production inference logic are still placeholders. The current implementation is best understood as a working MLOps scaffold that can be extended with a concrete model and dataset.

## Repository Layout

- `src/machledata/`: package code for data access, model config, training, metrics, and inference
- `apps/`: FastAPI and Streamlit entry points
- `scripts/`: local command-line helpers for train, predict, and evaluate flows
- `configs/`: committed YAML configuration files
- `tests/`: pytest coverage for package and app smoke tests
- `workflows/`: orchestration entry points such as the Airflow DAG
- `docs/`: architecture and data notes
- `docker/`: Dockerfiles and Compose configuration for local services
- `.github/workflows/`: CI/CD workflow definitions
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

Run the same style of coverage command used in CI:

```bash
python -m pytest -v --cov=src --cov-report=term
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

Run the API service through Docker Compose:

```bash
docker compose -f docker/docker-compose.yml up api
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

The Streamlit dashboard provides a local monitoring and demo surface. It accepts image uploads, sends them to the API, displays detection counts, renders a confidence chart and detection table when detections are present, and can show an annotated image when the API returns one.

Because the API prediction endpoint is still a placeholder, the dashboard currently represents the intended end-to-end user flow rather than real model inference. Before using it as a live demo, align the API response contract with the dashboard fields: `detections`, `class_name`, `confidence`, `bbox`, and optional `annotated_image_base64`.

The Airflow pipeline is the orchestration path for preparing data, training a model, evaluating outputs, and publishing deployment-facing metadata. It stops at artifact and manifest creation so the API or dashboard can consume validated outputs later.

## CI/CD and Deployment

The GitHub Actions workflow in `.github/workflows/deploy.yaml` runs on pushes and pull requests targeting `main` or `develop`, and on version tags matching `v*`.

The workflow has three stages:

1. `test`: installs the package with development dependencies, runs pytest with coverage, and uploads coverage output to Codecov.
2. `build`: builds `docker/Dockerfile` with Docker Buildx and pushes the image to Docker Hub on push events.
3. `deploy`: on pushes to `main`, connects to a remote VPS over SSH, pulls the latest image, and restarts the Docker Compose stack.

The deployment workflow expects these repository secrets:

- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`
- `REMOTE_HOST`
- `REMOTE_USER`
- `SSH_PRIVATE_KEY`
- `REMOTE_APP_DIR`

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

For CI/CD, configure Docker Hub and remote server credentials as GitHub Actions secrets rather than local environment variables.

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
