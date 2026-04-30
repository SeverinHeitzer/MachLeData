"""Tests for Vertex AI submit wiring without contacting GCP."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

from scripts import submit_vertex_pipeline


def test_submit_vertex_pipeline_uses_compiled_template(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Submit script builds a PipelineJob with expected parameters."""
    template = tmp_path / "pipeline.yaml"
    template.write_text("pipeline", encoding="utf-8")
    created: dict[str, object] = {}

    class FakePipelineJob:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.display_name = kwargs["display_name"]

        def submit(self, service_account=None):
            created["service_account"] = service_account

    fake_aiplatform = ModuleType("google.cloud.aiplatform")
    fake_aiplatform.PipelineJob = FakePipelineJob
    fake_aiplatform.init = lambda **kwargs: created.update({"init": kwargs})
    fake_cloud = ModuleType("google.cloud")
    fake_cloud.aiplatform = fake_aiplatform
    fake_google = ModuleType("google")
    fake_google.cloud = fake_cloud
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.cloud", fake_cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.aiplatform", fake_aiplatform)
    monkeypatch.setattr(
        "sys.argv",
        [
            "submit_vertex_pipeline",
            "--project-id",
            "project",
            "--region",
            "europe-west4",
            "--pipeline-root",
            "gs://bucket/root",
            "--template-path",
            str(template),
            "--service-account",
            "runner@example.iam.gserviceaccount.com",
        ],
    )

    submit_vertex_pipeline.main()

    assert created["init"] == {"project": "project", "location": "europe-west4"}
    assert created["template_path"] == str(template)
    assert created["pipeline_root"] == "gs://bucket/root"
    assert created["parameter_values"]["project_id"] == "project"
    assert created["service_account"] == "runner@example.iam.gserviceaccount.com"
