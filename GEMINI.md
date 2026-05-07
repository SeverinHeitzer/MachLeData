# MachLeData: MLOps YOLOv8 Pipeline

MachLeData is a production-oriented object detection pipeline using YOLOv8, orchestrated via Kubeflow Pipelines v2 and Vertex AI. It provides a complete lifecycle from data preparation to inference via a FastAPI service and a Streamlit dashboard.

## Project Overview

- **Core Package**: Located in `src/machledata/`, containing reusable logic for data, training, inference, and metrics.
- **Orchestration**: Kubeflow Pipelines v2 (KFP v2) definitions in `workflows/`, with Vertex AI support.
- **Serving**: FastAPI for real-time inference and Streamlit for interactive data visualization and live detection.
- **Infrastructure**: Containerized via Docker, with CI/CD pipelines for automated testing and deployment.

## Key Technologies

- **Language**: Python 3.11+
- **Model**: Ultralytics YOLOv8
- **Orchestration**: Kubeflow Pipelines v2, Google Vertex AI Pipelines
- **Data**: Google BigQuery, Google Cloud Storage (GCS)
- **API/UI**: FastAPI, Streamlit, Uvicorn
- **Testing/Linting**: Pytest

## Building and Running

### Local Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,kubeflow,vertex]"
```

### Common Commands
- **Run Tests**: `python -m pytest`
- **FastAPI (Inference)**: `uvicorn apps.api:app --reload`
- **Streamlit (Dashboard)**: `streamlit run apps/dashboard.py`
- **Compile Pipeline**: `python scripts/compile_pipeline.py --image-uri machledata:local`

### CLI Scripts
- `python scripts/train.py`: Local YOLOv8 training.
- `python scripts/predict.py --image path/to/image.jpg`: Local inference check.
- `python scripts/evaluate.py`: Performance metrics evaluation.

## Development Conventions

### Architecture
- **Isolation**: Keep core business logic inside `src/machledata/`. Keep `apps/`, `scripts/`, and `workflows/` as thin entry points.
- **Pipeline Steps**: `machledata.pipeline_steps` acts as the container adapter for Kubeflow. It handles the translation between CLI arguments and KFP v2 typed artifacts (JSON payloads).
- **Type Safety**: Use Pydantic models for API contracts and shared types (e.g., `machledata.infer.Detection`).

### Testing
- Always add tests for new features in the `tests/` directory.
- Verify pipeline compilation before committing changes to `workflows/`.

### Configuration
- Use `configs/*.yaml` for static, version-controlled defaults.
- Use `.env` or system environment variables for secrets (GCP project IDs, credentials, etc.).
- Refer to `docs/gcp_setup.md` for one-time environment preparation.
