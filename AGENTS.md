# Repository Guidelines

## Project Structure & Module Organization

This repository is the MachLeData course project: a YOLO-oriented object detection pipeline with an MLOps-friendly layout. Keep the root focused and use the existing top-level directories consistently:

- `src/machledata/` for reusable Python package code
- `apps/` for thin FastAPI and Streamlit entry points
- `workflows/` for orchestration definitions such as Kubeflow pipelines
- `scripts/` for local command-line helpers
- `configs/` for YAML settings that are safe to commit
- `tests/` for pytest coverage of package and app behavior
- `docs/` for architecture, data-source, and process documentation
- `docker/` for containerization assets
- `data/samples/` for tiny fixtures only; keep larger datasets and artifacts out of Git

Avoid committing generated outputs, caches, virtual environments, large datasets, or trained model artifacts.

## Build, Test, and Development Commands

Use the Python project metadata in `pyproject.toml`:

- `python -m venv .venv` creates a local virtual environment
- `source .venv/bin/activate` activates it on Linux or macOS shells
- `python -m pip install -e ".[dev,kubeflow,vertex]"` installs the package, tests, Kubeflow compiler, and Vertex submit dependencies
- `python -m machledata` smoke-tests package importability
- `python -m pytest` runs the test suite
- `python scripts/train.py` smoke-tests the training entry point
- `python scripts/predict.py` smoke-tests the prediction CLI
- `python scripts/evaluate.py` smoke-tests the evaluation CLI
- `python scripts/compile_pipeline.py --image-uri machledata:local` compiles the Kubeflow pipeline package
- `python scripts/submit_vertex_pipeline.py --project-id "$GOOGLE_CLOUD_PROJECT" --region "$VERTEX_REGION" --pipeline-root "$VERTEX_PIPELINE_ROOT" --template-path artifacts/pipelines/machledata_pipeline.yaml` submits the compiled package to Vertex AI
- `python -m uvicorn apps.api:app --reload` runs the API locally
- `streamlit run apps/dashboard.py` runs the dashboard locally

Update this section if a `Makefile` or another task runner becomes the canonical entry point.

## Coding Style & Naming Conventions

Write Python using PEP 8 with 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and lowercase module names. Keep modules focused and name scripts descriptively, for example `train.py` or `predict.py`.

Prefer type hints and concise docstrings. Skeleton files should include short module docstrings explaining their role in the pipeline, especially when the concrete implementation is still a stub.

For Kubeflow work, keep pipeline code thin and push logic into `src/machledata/` or `scripts/` so local CLI runs and orchestrated runs stay aligned. Preserve typed KFP v2 artifact interfaces for Vertex-facing pipeline components.

## Testing Guidelines

Use `pytest` for unit and integration tests. Name test files `test_*.py` and test functions `test_*`. Mirror behavior where practical, for example `tests/test_infer.py` for `machledata.infer`.

Prioritize tests for data access helpers, inference behavior, metrics, CLI seams, API routes, Kubeflow pipeline compilation, and Vertex submit parameter wiring. Avoid tests that require GPU access, cloud credentials, Vertex AI services, or large local datasets.

## Commit & Pull Request Guidelines

Use short, imperative commit messages such as `Add preprocessing pipeline` or `Document experiment setup`.

Pull requests should include a brief summary, test results, and assumptions about data sources, model settings, or environment variables. Link related issues when available. Include screenshots or plots only when they clarify dashboards, reports, or visual outputs.

## Security & Configuration Tips

Do not commit secrets, API keys, credentials, or private datasets. Store local configuration in ignored files such as `.env`, and document required variables in example configs or project docs.
