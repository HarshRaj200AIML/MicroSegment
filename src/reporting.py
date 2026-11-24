"""Export segmentation outputs and reproducibility metadata."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import joblib
import pandas as pd

from .utils import PROJECT_ROOT, ExportValidationError, get_logger


LOGGER = get_logger("microsegment.reporting")


def create_run_id(now: datetime | None = None) -> tuple[str, datetime]:
    timestamp = now or datetime.now(timezone.utc)
    local_timestamp = timestamp.astimezone()
    run_id = f"run_{local_timestamp.strftime('%Y%m%d_%H%M%S')}"
    return run_id, timestamp


def _simple_report_html(profile: pd.DataFrame) -> str:
    table = profile.to_html(border=0)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>MicroSegment Cluster Profiles</title></head><body>"
        "<h1>MicroSegment Cluster Profiles</h1>"
        f"{table}</body></html>"
    )


def _unique_backup_path(path: Path, run_id: str) -> Path:
    counter = 1
    candidate = path.with_name(f"{path.stem}_{run_id}{path.suffix}")
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{run_id}_{counter:03d}{path.suffix}")
        counter += 1
    return candidate


def _archive_existing_legacy_file(path: Path, new_payload: bytes, run_id: str) -> None:
    if not path.exists():
        return
    existing_payload = path.read_bytes()
    if existing_payload == new_payload:
        return
    backup = _unique_backup_path(path, run_id)
    backup.write_bytes(existing_payload)


def build_assignment_csv_payload(frame: pd.DataFrame) -> bytes:
    """Serialize the existing assignment CSV schema without adding metadata columns."""
    return frame.to_csv(index=False).encode("utf-8")


def build_assignment_metadata_payload(metadata: dict[str, Any]) -> bytes:
    return json.dumps(metadata, indent=2, ensure_ascii=False).encode("utf-8")


def build_assignment_zip_payload(csv_payload: bytes, metadata_payload: bytes) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("microsegment_new_assignments.csv", csv_payload)
        archive.writestr("microsegment_assignment_metadata.json", metadata_payload)
    return buffer.getvalue()


def save_outputs(
    frame: pd.DataFrame,
    profile: pd.DataFrame,
    preprocessor: Any,
    model: Any,
    config: dict[str, Any],
    project_root: Path | None = None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Path]:
    root = project_root or PROJECT_ROOT
    if manifest is None:
        raise ExportValidationError("A model manifest is required before exporting outputs.")
    if len(frame) != int(manifest.get("dataset_row_count", -1)):
        raise ExportValidationError("The segmented row count does not match the model provenance.")
    if "cluster" not in frame.columns:
        raise ExportValidationError("The export does not contain a cluster label column.")

    run_id, timestamp = create_run_id()

    folders = {key: root / "outputs" / key for key in ("segmented_data", "cluster_profiles")}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    models = root / "models"
    models.mkdir(exist_ok=True)
    data_path = folders["segmented_data"] / "segmented_data.csv"
    profile_path = folders["cluster_profiles"] / "cluster_profiles.csv"
    model_path = models / "clustering_pipeline.joblib"
    config_path = models / "preprocessing_config.json"

    data_payload = frame.to_csv(index=False).encode("utf-8")
    profile_payload = profile.to_csv().encode("utf-8")
    config_payload = json.dumps(config, indent=2, default=str).encode("utf-8")

    _archive_existing_legacy_file(data_path, data_payload, run_id)
    _archive_existing_legacy_file(profile_path, profile_payload, run_id)
    _archive_existing_legacy_file(config_path, config_payload, run_id)

    data_path.write_bytes(data_payload)
    profile_path.write_bytes(profile_payload)
    config_path.write_bytes(config_payload)

    if model_path.exists():
        backup = _unique_backup_path(model_path, run_id)
        backup.write_bytes(model_path.read_bytes())
    joblib.dump({"preprocessor": preprocessor, "model": model, "manifest": manifest}, model_path)
    experiments_root = root / "outputs" / "experiments"
    experiment_dir = experiments_root / run_id
    counter = 1
    while experiment_dir.exists():
        experiment_dir = experiments_root / f"{run_id}_{counter:03d}"
        counter += 1
    experiment_dir.mkdir(parents=True)

    experiment_data = experiment_dir / "segmented_data.csv"
    experiment_profile = experiment_dir / "cluster_profiles.csv"
    experiment_model = experiment_dir / "clustering_pipeline.joblib"
    experiment_config = experiment_dir / "model_config.json"
    experiment_manifest = experiment_dir / "model_manifest.json"
    experiment_report = experiment_dir / "report.html"

    frame.to_csv(experiment_data, index=False)
    profile.to_csv(experiment_profile)
    joblib.dump({"preprocessor": preprocessor, "model": model, "manifest": manifest}, experiment_model)

    enriched_config = dict(config)
    enriched_config["run_id"] = run_id
    enriched_config["timestamp"] = timestamp.isoformat()
    enriched_config["dataset_fingerprint"] = manifest["dataset_fingerprint"]
    enriched_config["schema_fingerprint"] = manifest["schema_fingerprint"]
    enriched_config["selected_features"] = manifest["selected_features"]
    enriched_config["algorithm"] = manifest["algorithm"]
    enriched_config["algorithm_parameters"] = manifest["algorithm_parameters"]
    enriched_config["random_seed"] = manifest["random_seed"]
    enriched_config["preprocessing"] = manifest["preprocessing"]
    enriched_config["metrics"] = manifest["metrics"]

    export_manifest = dict(manifest)
    export_manifest["run_id"] = run_id
    export_manifest["timestamp"] = timestamp.isoformat()
    export_manifest["export_type"] = "validated_microsegment_run"

    experiment_config.write_text(json.dumps(enriched_config, indent=2, default=str), encoding="utf-8")
    experiment_manifest.write_text(json.dumps(export_manifest, indent=2, default=str), encoding="utf-8")
    experiment_report.write_text(_simple_report_html(profile), encoding="utf-8")

    LOGGER.info("Saved validated experiment %s", experiment_dir)
    return {
        "segmented_data": data_path,
        "cluster_profiles": profile_path,
        "model": model_path,
        "config": config_path,
        "experiment": experiment_dir,
        "experiment_segmented_data": experiment_data,
        "experiment_cluster_profiles": experiment_profile,
        "experiment_model": experiment_model,
        "experiment_config": experiment_config,
        "experiment_manifest": experiment_manifest,
        "experiment_report": experiment_report,
    }