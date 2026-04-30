"""Tests for the Kubeflow pipeline definition."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def test_kubeflow_pipeline_compiles(tmp_path: Path, monkeypatch) -> None:
    """The Kubeflow v2 pipeline compiles without GCP credentials."""
    compiler = pytest.importorskip("kfp.compiler")

    monkeypatch.setenv("MACHLEDATA_PIPELINE_IMAGE", "example/machledata:test")
    import workflows.kubeflow_pipeline as pipeline_module

    pipeline_module = importlib.reload(pipeline_module)
    package_path = tmp_path / "machledata_pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=pipeline_module.machledata_pipeline,
        package_path=str(package_path),
    )

    compiled = package_path.read_text(encoding="utf-8")
    assert "machledata-ml-pipeline" in compiled
    assert "example/machledata:test" in compiled
    assert "system.Dataset" in compiled
    assert "system.Model" in compiled
    assert "system.Metrics" in compiled
    for task_id in pipeline_module.TASK_SEQUENCE:
        assert task_id in compiled
