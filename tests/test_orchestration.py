"""Tests for orchestration-friendly pipeline helpers."""

from __future__ import annotations

from pathlib import Path

from machledata.orchestration import (
    evaluate_model,
    prepare_dataset,
    publish_artifact_manifest,
    train_model,
)


def test_prepare_dataset_returns_serializable_descriptor(tmp_path: Path) -> None:
    """Dataset preparation returns a descriptor and writes it to disk."""
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    (samples_dir / "one.jpg").write_text("sample", encoding="utf-8")

    descriptor = prepare_dataset(
        dataset_id="demo-dataset",
        artifact_root=tmp_path / "artifacts",
        samples_dir=samples_dir,
        run_label="unit-test",
    )

    assert descriptor["dataset_id"] == "demo-dataset"
    assert descriptor["sample_count"] == 1
    assert Path(descriptor["descriptor_path"]).exists()


def test_train_evaluate_publish_contracts_round_trip(tmp_path: Path) -> None:
    """The placeholder pipeline produces training, evaluation, and publish outputs."""
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    (samples_dir / "one.jpg").write_text("sample", encoding="utf-8")

    prepared = prepare_dataset(
        dataset_id="demo-dataset",
        artifact_root=tmp_path / "artifacts",
        samples_dir=samples_dir,
        run_label="unit-test",
    )
    training_run = train_model(
        prepared_dataset=prepared,
        model_name="yolov8n",
        epochs=5,
        artifact_root=tmp_path / "artifacts",
        run_label="unit-test",
    )
    evaluation_summary = evaluate_model(
        prepared_dataset=prepared,
        training_run=training_run,
    )
    manifest = publish_artifact_manifest(
        prepared_dataset=prepared,
        training_run=training_run,
        evaluation_summary=evaluation_summary,
        artifact_root=tmp_path / "artifacts",
    )

    assert training_run["epochs"] == 5
    assert Path(training_run["model_artifact_path"]).exists()
    assert evaluation_summary["passed"] is True
    assert manifest["model"]["artifact_path"] == training_run["model_artifact_path"]
    assert Path(manifest["manifest_path"]).exists()
