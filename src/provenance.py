"""Deterministic dataset and model provenance helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def schema_fingerprint(frame: pd.DataFrame) -> str:
    schema = [
        {"name": str(column), "dtype": str(frame[column].dtype)}
        for column in frame.columns
    ]
    payload = json.dumps(schema, ensure_ascii=False, sort_keys=False).encode("utf-8")
    return _hash_bytes(payload)


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    """Hash actual row values and the source schema in a deterministic format."""
    row_hashes = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256()
    digest.update(json.dumps(
        [
            {"name": str(column), "dtype": str(frame[column].dtype)}
            for column in frame.columns
        ],
        ensure_ascii=False,
    ).encode("utf-8"))
    digest.update(json.dumps(row_hashes.tolist(), separators=(",", ":")).encode("ascii"))
    digest.update(str(len(frame)).encode("ascii"))
    return digest.hexdigest()


def dataset_column_info(frame: pd.DataFrame) -> dict[str, str]:
    return {str(column): str(frame[column].dtype) for column in frame.columns}


def create_model_run_id() -> str:
    """Create one genuinely unique ID for a successfully fitted model."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"model_{timestamp}_{uuid4().hex[:12]}"


def preprocessing_metadata(preprocessor: Any, selected_features: list[str]) -> dict[str, Any]:
    numeric: list[str] = []
    categorical: list[str] = []
    transformers = getattr(preprocessor, "transformers_", [])
    for name, pipeline, columns in transformers:
        if name == "numeric":
            numeric = [str(column) for column in columns]
        elif name == "categorical":
            categorical = [str(column) for column in columns]
    return {
        "selected_features": list(selected_features),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "numeric_steps": ["median_imputation", "standard_scaling"],
        "categorical_steps": ["most_frequent_imputation", "one_hot_encoding"],
    }


def build_model_manifest(
    *,
    frame: pd.DataFrame,
    source_name: str,
    selected_features: list[str],
    algorithm: str,
    algorithm_parameters: dict[str, Any],
    random_seed: int | None,
    preprocessor: Any,
    metrics: dict[str, Any],
    model_run_id: str | None = None,
    seed_applicable: bool | None = None,
) -> dict[str, Any]:
    return {
        "model_run_id": model_run_id,
        "dataset_fingerprint": dataset_fingerprint(frame),
        "schema_fingerprint": schema_fingerprint(frame),
        "dataset_row_count": int(len(frame)),
        "dataset_column_info": dataset_column_info(frame),
        "dataset_name": Path(source_name).name,
        "selected_features": list(selected_features),
        "algorithm": algorithm,
        "algorithm_parameters": algorithm_parameters,
        "random_seed": None if random_seed is None else int(random_seed),
        "seed_applicable": random_seed is not None if seed_applicable is None else bool(seed_applicable),
        "preprocessing": preprocessing_metadata(preprocessor, selected_features),
        "metrics": metrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def validate_model_manifest(
    manifest: dict[str, Any] | None,
    *,
    frame: pd.DataFrame,
    labels: Any,
    selected_features: list[str],
    preprocessor: Any,
    model: Any,
) -> tuple[bool, str]:
    if not manifest:
        return False, "No trained model provenance is available."
    actual_data = dataset_fingerprint(frame)
    actual_schema = schema_fingerprint(frame)
    if manifest.get("dataset_fingerprint") != actual_data:
        return False, "The active dataset differs from the dataset used to train this model."
    if manifest.get("schema_fingerprint") != actual_schema:
        return False, "The active dataset schema differs from the model training schema."
    if int(manifest.get("dataset_row_count", -1)) != len(frame):
        return False, "The active dataset row count differs from the model training data."
    if manifest.get("dataset_column_info") != dataset_column_info(frame):
        return False, "The active dataset columns or column types differ from the training data."
    if list(manifest.get("selected_features", [])) != list(selected_features):
        return False, "The selected clustering features differ from the model training features."
    if labels is None:
        return False, "The model does not contain cluster labels."
    if len(labels) != len(frame):
        return False, "The cluster label count does not match the active dataset row count."
    if preprocessor is None:
        return False, "The trained preprocessing pipeline is unavailable."
    if model is None:
        return False, "The trained clustering model is unavailable."
    return True, "The model matches the active dataset and selected features."


def is_valid_random_seed(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, Integral):
        return False
    return 0 <= int(value) <= 2**32 - 1


def validate_assignment_random_seed(manifest: dict[str, Any]) -> tuple[bool, str]:
    algorithm = manifest.get("algorithm")
    if algorithm == "DBSCAN":
        if manifest.get("random_seed") is not None:
            return False, "DBSCAN assignment provenance must use random_seed=null."
        if manifest.get("seed_applicable") is not False:
            return False, "DBSCAN assignment provenance must set seed_applicable=false."
        return True, "DBSCAN seed provenance is consistent."
    if algorithm not in {"K-Means", "K-Medoids"}:
        return False, "Assignment provenance has an unsupported algorithm."
    if "random_seed" not in manifest:
        return False, f"{algorithm} assignment provenance is missing random_seed. Retrain the model."
    seed = manifest.get("random_seed")
    if seed is None:
        return False, f"{algorithm} assignment provenance requires a non-null random_seed. Retrain the model."
    if not is_valid_random_seed(seed):
        return False, f"{algorithm} assignment provenance has an invalid random_seed. Retrain the model."
    if manifest.get("seed_applicable") is not True:
        return False, f"{algorithm} assignment provenance must set seed_applicable=true."
    return True, "Random seed provenance is verified."


def validate_cluster_count_parameter(manifest: dict[str, Any]) -> tuple[bool, str]:
    algorithm = manifest.get("algorithm")
    parameters = manifest.get("algorithm_parameters")
    if algorithm == "DBSCAN":
        return True, "DBSCAN does not require n_clusters."
    if algorithm not in {"K-Means", "K-Medoids"}:
        return False, "Assignment provenance has an unsupported algorithm."
    if not isinstance(parameters, dict):
        return False, f"{algorithm} assignment provenance is missing algorithm_parameters. Retrain the model."
    if "n_clusters" not in parameters:
        return False, f"{algorithm} assignment provenance is missing n_clusters. Retrain the model."
    n_clusters = parameters.get("n_clusters")
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, Integral):
        return False, f"{algorithm} assignment provenance has an invalid n_clusters value. Retrain the model."
    if int(n_clusters) <= 0:
        return False, f"{algorithm} assignment provenance requires n_clusters to be positive. Retrain the model."
    return True, "Cluster count parameter is verified."


def validate_assignment_provenance(manifest: dict[str, Any] | None) -> tuple[bool, str]:
    if not manifest:
        return False, "No trained model provenance is available."
    if not manifest.get("model_run_id"):
        return False, "This model is missing model_run_id provenance. Retrain the model to create a new traced model."
    if not manifest.get("dataset_fingerprint"):
        return False, "This model is missing its training dataset fingerprint. Retrain the model."
    if not manifest.get("selected_features"):
        return False, "This model is missing selected feature provenance. Retrain the model."
    algorithm = manifest.get("algorithm")
    seed_valid, seed_reason = validate_assignment_random_seed(manifest)
    if not seed_valid:
        return False, seed_reason
    cluster_count_valid, cluster_count_reason = validate_cluster_count_parameter(manifest)
    if not cluster_count_valid:
        return False, cluster_count_reason
    if algorithm not in {"K-Means", "K-Medoids"}:
        return False, f"Assignment export is only supported for K-Means or verified K-Medoids models; this model uses {algorithm or 'an unknown algorithm'}."
    if not isinstance(manifest.get("algorithm_parameters"), dict):
        return False, "This model is missing verified algorithm parameters. Retrain the model."
    if algorithm == "DBSCAN":
        return False, "DBSCAN does not support the current assignment prediction workflow."
    return True, "Assignment provenance is complete."


def build_assignment_manifest(
    *,
    manifest: dict[str, Any],
    assigned_frame: pd.DataFrame,
    original_input_filename: str,
    cluster_labels: Any,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    valid, reason = validate_assignment_provenance(manifest)
    if not valid:
        raise ValueError(reason)

    labels = np.asarray(cluster_labels)
    distinct_clusters = set(int(label) for label in np.unique(labels) if int(label) != -1)
    algorithm = str(manifest["algorithm"])
    parameters = dict(manifest["algorithm_parameters"])

    if algorithm == "K-Means":
        model_cluster_count = int(parameters["n_clusters"])
    elif algorithm == "K-Medoids":
        model_cluster_count = int(parameters["n_clusters"])
    else:
        raise ValueError("Assignment metadata is unavailable for this algorithm.")

    assignment_timestamp = timestamp or datetime.now(timezone.utc)
    if assignment_timestamp.tzinfo is None:
        assignment_timestamp = assignment_timestamp.replace(tzinfo=timezone.utc)

    return {
        "export_type": "microsegment_assignment",
        "assignment_timestamp_utc": assignment_timestamp.astimezone(timezone.utc).isoformat(),
        "original_input_filename": Path(str(original_input_filename)).name,
        "training_dataset_fingerprint": manifest["dataset_fingerprint"],
        "model_run_id": manifest["model_run_id"],
        "selected_features": list(manifest["selected_features"]),
        "algorithm": algorithm,
        "actual_random_seed": manifest.get("random_seed"),
        "seed_applicable": bool(manifest.get("seed_applicable", manifest.get("random_seed") is not None)),
        "algorithm_parameters": parameters,
        "model_cluster_count": model_cluster_count,
        "assigned_cluster_count": len(distinct_clusters),
        "assigned_record_count": int(len(assigned_frame)),
    }
