"""Tests for Kubeflow container step adapters."""

from __future__ import annotations

import json
from pathlib import Path

from machledata.pipeline_steps import main


def test_prepare_train_evaluate_publish_step_cli_round_trip(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The step CLI exchanges JSON files between pipeline stages."""
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "one.jpg").write_text("sample", encoding="utf-8")
    artifact_root = tmp_path / "artifacts"

    prepared_path = tmp_path / "prepared_dataset.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "pipeline_steps",
            "prepare-data",
            "--dataset-id",
            "demo",
            "--samples-dir",
            str(samples),
            "--artifact-root",
            str(artifact_root),
            "--run-label",
            "unit",
            "--output-path",
            str(prepared_path),
        ],
    )
    main()

    training_path = tmp_path / "training_run.json"
    model_path = tmp_path / "model" / "model-placeholder.bin"
    monkeypatch.setattr(
        "sys.argv",
        [
            "pipeline_steps",
            "train-model",
            "--prepared-dataset-path",
            str(prepared_path),
            "--model-name",
            "yolov8n",
            "--epochs",
            "2",
            "--artifact-root",
            str(artifact_root),
            "--run-label",
            "unit",
            "--model-output-path",
            str(model_path),
            "--metadata-output-path",
            str(training_path),
        ],
    )
    main()

    evaluation_path = tmp_path / "evaluation_summary.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "pipeline_steps",
            "evaluate-model",
            "--prepared-dataset-path",
            str(prepared_path),
            "--training-run-path",
            str(training_path),
            "--model-artifact-path",
            str(model_path),
            "--evaluation-output-path",
            str(evaluation_path),
            "--metrics-output-path",
            str(tmp_path / "metrics.json"),
        ],
    )
    main()

    manifest_path = tmp_path / "artifact_manifest.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "pipeline_steps",
            "publish-artifact-metadata",
            "--prepared-dataset-path",
            str(prepared_path),
            "--training-run-path",
            str(training_path),
            "--evaluation-summary-path",
            str(evaluation_path),
            "--artifact-root",
            str(artifact_root),
            "--manifest-output-path",
            str(manifest_path),
        ],
    )
    main()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset"]["dataset_id"] == "demo"
    assert manifest["model"]["model_name"] == "yolov8n"
    assert manifest["model"]["artifact_path"] == str(model_path.resolve())
    assert manifest["evaluation"]["passed"] is True
