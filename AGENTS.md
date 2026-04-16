# Repository Guidelines

## Project Structure & Module Organization

This repository is for the MachLeData course project: a YOLO-oriented object detection pipeline with MLOps practices. Keep the root small and use the existing top-level directories consistently:

- `src/machledata/` for reusable Python package code.
- `apps/` for thin FastAPI and Streamlit entry points.
- `workflows/` for Airflow DAGs and future orchestration files.
- `scripts/` for local command-line helpers.
- `configs/` for YAML settings that are safe to commit.
- `tests/` for pytest tests that exercise package and app behavior.
- `data/samples/` for tiny fixtures only; document BigQuery sources in `docs/`.

Avoid committing generated outputs, caches, virtual environments, large datasets, or trained model artifacts.

## Build, Test, and Development Commands

Use the Python project metadata in `pyproject.toml`:

- `python -m venv .venv` creates a local virtual environment.
- `source .venv/bin/activate` activates it on Linux/macOS shells.
- `python -m pip install -e ".[dev]"` installs the package and test dependencies.
- `python -m pytest` runs the test suite.
- `python scripts/train.py` smoke-tests the training entry point.
- `python -m uvicorn apps.api:app --reload` runs the API locally.

Update this section if a `Makefile` or task runner becomes canonical.

## Coding Style & Naming Conventions

Write Python using PEP 8: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and lowercase module names. Keep modules focused and name scripts descriptively, for example `train.py` or `predict.py`.

Prefer type hints and concise docstrings. Skeleton files should include clear module docstrings explaining their role in the MLOps pipeline.

## Testing Guidelines

Use `pytest` for unit and integration tests. Name test files `test_*.py` and test functions `test_*`. Mirror behavior where practical, for example `tests/test_infer.py` for `machledata.infer`.

Prioritize tests for data access helpers, inference behavior, metrics, and API routes. Avoid tests that require GPU, cloud credentials, or large local datasets.

## Commit & Pull Request Guidelines

The current history contains only `Initial commit`, so no detailed convention has emerged yet. Use short, imperative commit messages such as `Add preprocessing pipeline` or `Document experiment setup`.

Pull requests should include a brief summary, test results, and assumptions about data, model settings, or environment variables. Link related issues when available. Include screenshots or plots only when they clarify notebooks, reports, or visual outputs.

## Security & Configuration Tips

Do not commit secrets, API keys, credentials, or private datasets. Store local configuration in ignored files such as `.env`, and document required variables in an example file.
