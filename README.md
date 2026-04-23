# MachLeData

MachLeData is a course project skeleton for a YOLO-style object detection pipeline with a small MLOps-oriented layout. The repository already includes reusable package modules, CLI entry points, a FastAPI app, a Streamlit dashboard, config files, tests, and an Airflow DAG stub so the project can grow without reorganizing the codebase later.

## Current Status

The project is intentionally lightweight today. The scaffolding that already works includes:

- Python package entry point in `src/machledata/`
- FastAPI health and placeholder prediction endpoints in `apps/api.py`
- Streamlit dashboard starter in `apps/dashboard.py`
- Local CLI scripts for training, prediction, and evaluation in `scripts/`
- Config files in `configs/`
- Smoke tests for data helpers, inference, and API behavior in `tests/`
- An Airflow DAG placeholder in `workflows/airflow_dag.py`

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

## Package Overview

The core package is designed to keep application and orchestration code thin:

- `machledata.data`: sample-file discovery and BigQuery source description helpers
- `machledata.model`: typed model configuration for future YOLO integration
- `machledata.train`: training run metadata creation
- `machledata.infer`: shared detection schema and prediction interface
- `machledata.metrics`: simple summary metrics for demos and smoke tests

## API and Dashboard

The FastAPI app currently exposes:

- `GET /health`: returns `{"status": "ok"}`
- `POST /predict`: returns an empty detection list until inference is implemented

The Streamlit dashboard is a starter page for demos, prediction inspection, and model metrics.

## Configuration

Committed YAML files in `configs/` define the initial project shape:

- `configs/model.yaml`: model name, image size, batch size, epochs, artifact directory
- `configs/data.yaml`: BigQuery-oriented dataset settings and sample directory
- `configs/app.yaml`: app title, model artifact path, and confidence threshold

Keep secrets and environment-specific values out of Git. Use `.env` or deployment configuration for credentials and sensitive settings.

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
