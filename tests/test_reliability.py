import os

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.clustering import (
    fit_dbscan,
    fit_kmeans,
    validate_matrix_memory,
)
from src.data_loader import (
    DatasetLimits,
    load_dataset,
)
from src.provenance import (
    build_assignment_manifest,
    build_model_manifest,
    create_model_run_id,
    dataset_fingerprint,
    schema_fingerprint,
    validate_assignment_provenance,
    validate_model_manifest,
    validate_assignment_random_seed,
    validate_cluster_count_parameter,
)
from src.reporting import (
    build_assignment_csv_payload,
    build_assignment_metadata_payload,
    build_assignment_zip_payload,
    save_outputs,
)
from src.utils import DatasetValidationError, MemorySafetyError


def test_same_row_count_different_records_produces_different_fingerprints():
    dataset_a = pd.DataFrame({"measure": [1, 2, 3], "category": ["A", "A", "B"]})
    dataset_b = pd.DataFrame({"measure": [10, 20, 30], "category": ["A", "A", "B"]})
    assert len(dataset_a) == len(dataset_b)
    assert dataset_fingerprint(dataset_a) != dataset_fingerprint(dataset_b)


def test_schema_fingerprint_is_independent_from_values():
    dataset_a = pd.DataFrame({"measure": [1, 2, 3]})
    dataset_b = pd.DataFrame({"measure": [10, 20, 30]})
    assert schema_fingerprint(dataset_a) == schema_fingerprint(dataset_b)
    assert dataset_fingerprint(dataset_a) != dataset_fingerprint(dataset_b)


def test_model_manifest_rejects_changed_dataset_and_features():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "category": ["x", "y", "x"]})
    matrix = frame[["a", "b"]].values
    result = fit_kmeans(matrix, n_clusters=2)
    manifest = build_model_manifest(
        frame=frame,
        source_name="synthetic.csv",
        selected_features=["a", "b"],
        algorithm="K-Means",
        algorithm_parameters={"n_clusters": 2, "n_init": 10},
        random_seed=42,
        preprocessor=object(),
        metrics=result.metrics,
    )
    valid, _ = validate_model_manifest(
        manifest,
        frame=frame,
        labels=result.labels,
        selected_features=["a", "b"],
        preprocessor=object(),
        model=result.model,
    )
    assert valid

    changed = frame.copy()
    changed["a"] = changed["a"] * 100
    valid, reason = validate_model_manifest(
        manifest,
        frame=changed,
        labels=result.labels,
        selected_features=["a", "b"],
        preprocessor=object(),
        model=result.model,
    )
    assert not valid
    assert "dataset differs" in reason.lower()

    valid, reason = validate_model_manifest(
        manifest,
        frame=frame,
        labels=result.labels,
        selected_features=["a", "category"],
        preprocessor=object(),
        model=result.model,
    )
    assert not valid
    assert "selected clustering features" in reason.lower()


def test_sparse_and_dense_kmeans_behavior_is_preserved():
    dense = np.array([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]])
    dense_result = fit_kmeans(dense, n_clusters=2)
    sparse_result = fit_kmeans(sp.csr_matrix(dense), n_clusters=2)
    assert len(dense_result.labels) == 4
    assert len(sparse_result.labels) == 4
    assert sparse_result.metrics["n_clusters"] == 2


def test_sparse_dbscan_behavior():
    matrix = sp.csr_matrix([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1]])
    result = fit_dbscan(matrix, eps=0.5, min_samples=2)
    assert len(result.labels) == 4


def test_configurable_memory_guard_rejects_large_matrix(monkeypatch):
    matrix = sp.csr_matrix(np.ones((100, 100)))
    monkeypatch.setenv("MICROSEGMENT_MAX_CLUSTERING_BYTES", "1")
    with pytest.raises(MemorySafetyError):
        validate_matrix_memory(matrix)


def test_duplicate_and_blank_columns_are_rejected(tmp_path):
    duplicate = tmp_path / "duplicate.csv"
    pd.DataFrame([[1, 2]], columns=["a", "a"]).to_csv(duplicate, index=False)
    with pytest.raises(ValueError, match="duplicate column"):
        load_dataset(duplicate)

    blank = tmp_path / "blank.csv"
    pd.DataFrame([[1]], columns=[" "]).to_csv(blank, index=False)
    with pytest.raises(ValueError, match="blank column"):
        load_dataset(blank)


def test_row_and_column_limits_are_configurable(tmp_path):
    source = tmp_path / "limited.csv"
    pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).to_csv(source, index=False)
    with pytest.raises(ValueError, match="row limit"):
        load_dataset(source, DatasetLimits(max_rows=2))
    with pytest.raises(ValueError, match="column limit"):
        load_dataset(source, DatasetLimits(max_columns=1))


def test_corrupted_stata_reports_loader_error(tmp_path):
    source = tmp_path / "corrupt.dta"
    source.write_bytes(b"not a valid stata file")
    with pytest.raises(ValueError, match="Could not load this Stata"):
        load_dataset(source)


def test_versioned_exports_never_overwrite_previous_run(tmp_path):
    frame = pd.DataFrame({"a": [1, 2], "cluster": [0, 1], "segment_name": ["A", "B"]})
    profile = pd.DataFrame({"size": [1, 1]}, index=[0, 1])
    matrix = np.array([[1.0], [2.0]])
    result = fit_kmeans(matrix, n_clusters=2)
    manifest = build_model_manifest(
        frame=frame.drop(columns=["cluster", "segment_name"]),
        source_name="synthetic.csv",
        selected_features=["a"],
        algorithm="K-Means",
        algorithm_parameters={"n_clusters": 2, "n_init": 10},
        random_seed=42,
        preprocessor=object(),
        metrics=result.metrics,
    )

    first = save_outputs(
        frame,
        profile,
        object(),
        result.model,
        {"source": "synthetic.csv", "features": ["a"]},
        project_root=tmp_path,
        manifest=manifest,
    )
    second = save_outputs(
        frame,
        profile,
        object(),
        result.model,
        {"source": "synthetic.csv", "features": ["a"]},
        project_root=tmp_path,
        manifest=manifest,
    )
    assert first["experiment"] != second["experiment"]
    assert first["experiment_segmented_data"].exists()
    assert second["experiment_segmented_data"].exists()


def test_optional_kmedoids_is_clear_when_unavailable():
    try:
        from src.clustering import fit_kmedoids

        fit_kmedoids(np.array([[0.0], [1.0], [5.0]]), n_clusters=2)
    except ImportError as exc:
        assert "scikit-learn-extra" in str(exc)


def test_pandas_string_dtype_gets_high_cardinality_warning(tmp_path):
    source = tmp_path / "high_cardinality.csv"
    pd.DataFrame({"category": [f"value-{index}" for index in range(120)]}).to_csv(
        source, index=False
    )
    info = load_dataset(source)
    assert any("High-cardinality categorical feature 'category'" in warning for warning in info.warnings)


def test_numeric_column_does_not_get_categorical_warning(tmp_path):
    source = tmp_path / "numeric.csv"
    pd.DataFrame({"measure": list(range(120))}).to_csv(source, index=False)
    info = load_dataset(source)
    assert not any("High-cardinality" in warning for warning in info.warnings)


def test_duplicate_and_blank_csv_errors_remain_dataset_validation_errors(tmp_path):
    duplicate = tmp_path / "duplicate.csv"
    pd.DataFrame([[1, 2]], columns=["duplicate", "duplicate"]).to_csv(duplicate, index=False)
    with pytest.raises(DatasetValidationError, match="duplicate column"):
        load_dataset(duplicate)

    blank = tmp_path / "blank.csv"
    blank.write_text(" ,value\n1,2\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="blank column"):
        load_dataset(blank)


def test_legacy_exports_are_backed_up_before_replacing(tmp_path):
    frame_a = pd.DataFrame({"a": [1, 2], "cluster": [0, 1], "segment_name": ["A", "B"]})
    profile_a = pd.DataFrame({"size": [1, 1]}, index=[0, 1])
    frame_b = frame_a.copy()
    frame_b["a"] = [99, 100]

    manifest = {
        "dataset_row_count": 2,
        "dataset_fingerprint": "dataset-a",
        "schema_fingerprint": "schema-a",
        "selected_features": ["a"],
        "algorithm": "K-Means",
        "algorithm_parameters": {"n_clusters": 2},
        "random_seed": 42,
        "preprocessing": {},
        "metrics": {},
    }

    first = save_outputs(frame_a, profile_a, object(), object(), {}, project_root=tmp_path, manifest=manifest)
    first_segmented = first["segmented_data"].read_bytes()

    manifest_b = dict(manifest)
    manifest_b["dataset_fingerprint"] = "dataset-b"
    second = save_outputs(frame_b, profile_a, object(), object(), {}, project_root=tmp_path, manifest=manifest_b)

    backups = list((tmp_path / "outputs" / "segmented_data").glob("segmented_data_*.csv"))
    assert first["segmented_data"].read_bytes() != first_segmented or len(backups) >= 1
    assert second["segmented_data"].read_bytes() == frame_b.to_csv(index=False).encode("utf-8")
    assert len(backups) >= 1
    assert backups[0].read_bytes() == first_segmented


def test_successful_model_fits_get_unique_model_run_ids():
    first = create_model_run_id()
    second = create_model_run_id()
    assert first.startswith("model_")
    assert second.startswith("model_")
    assert first != second


def test_assignment_manifest_contains_verified_provenance():
    timestamp = pd.Timestamp("2026-11-23T01:30:00Z").to_pydatetime()
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    manifest = build_model_manifest(
        frame=frame,
        source_name="districts.csv",
        selected_features=["a", "b"],
        algorithm="K-Means",
        algorithm_parameters={"n_clusters": 2, "n_init": 10},
        random_seed=42,
        preprocessor=object(),
        metrics={"n_clusters": 2},
        model_run_id="model_verified_123",
        seed_applicable=True,
    )
    assigned = pd.DataFrame({
        "a": [1.0, 9.0],
        "b": [4.0, 10.0],
        "cluster": [0, 1],
        "segment_name": ["A", "B"],
    })
    metadata = build_assignment_manifest(
        manifest=manifest,
        assigned_frame=assigned,
        original_input_filename=r"C:\uploads\districts.csv",
        cluster_labels=[0, 1],
        timestamp=timestamp,
    )
    assert metadata["export_type"] == "microsegment_assignment"
    assert metadata["assignment_timestamp_utc"] == "2026-11-23T01:30:00+00:00"
    assert metadata["original_input_filename"] == "districts.csv"
    assert metadata["training_dataset_fingerprint"] == manifest["dataset_fingerprint"]
    assert metadata["model_run_id"] == "model_verified_123"
    assert metadata["selected_features"] == ["a", "b"]
    assert metadata["algorithm"] == "K-Means"
    assert metadata["actual_random_seed"] == 42
    assert metadata["model_cluster_count"] == 2
    assert metadata["assigned_cluster_count"] == 2
    assert metadata["assigned_record_count"] == 2


def test_assignment_metadata_uses_null_seed_when_not_applicable():
    frame = pd.DataFrame({"a": [1.0, 2.0]})
    manifest = build_model_manifest(
        frame=frame,
        source_name="districts.csv",
        selected_features=["a"],
        algorithm="DBSCAN",
        algorithm_parameters={"eps": 0.5, "min_samples": 5},
        random_seed=None,
        preprocessor=object(),
        metrics={"n_clusters": 1},
        model_run_id="model_dbscan_verified",
        seed_applicable=False,
    )
    # DBSCAN assignment is intentionally blocked; no assignment provenance is fabricated.
    valid, reason = validate_assignment_provenance(manifest)
    assert not valid
    assert "only supported" in reason


def test_assignment_noise_labels_are_excluded_from_assigned_cluster_count():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    manifest = build_model_manifest(
        frame=frame,
        source_name="districts.csv",
        selected_features=["a"],
        algorithm="K-Means",
        algorithm_parameters={"n_clusters": 3, "n_init": 10},
        random_seed=42,
        preprocessor=object(),
        metrics={"n_clusters": 3},
        model_run_id="model_noise_test",
        seed_applicable=True,
    )
    assigned = pd.DataFrame({"a": [1.0, 2.0, 3.0], "cluster": [0, -1, 1], "segment_name": ["A", "Noise", "B"]})
    metadata = build_assignment_manifest(
        manifest=manifest,
        assigned_frame=assigned,
        original_input_filename="records.csv",
        cluster_labels=[0, -1, 1],
    )
    assert metadata["model_cluster_count"] == 3
    assert metadata["assigned_cluster_count"] == 2
    assert metadata["assigned_record_count"] == 3


def test_missing_model_run_id_blocks_assignment_without_fabrication():
    manifest = {
        "dataset_fingerprint": "verified",
        "selected_features": ["a"],
        "algorithm": "K-Means",
        "algorithm_parameters": {"n_clusters": 2, "n_init": 10},
        "random_seed": 42,
    }
    valid, reason = validate_assignment_provenance(manifest)
    assert not valid
    assert "model_run_id" in reason
    assert "Retrain" in reason


def test_assignment_csv_json_zip_preserve_schema_and_timestamp():
    import json
    from zipfile import ZipFile
    from io import BytesIO

    assigned = pd.DataFrame({
        "a": [1.0, 2.0],
        "cluster": [0, 1],
        "segment_name": ["A", "B"],
    })
    metadata = {
        "export_type": "microsegment_assignment",
        "assignment_timestamp_utc": "2026-11-23T01:30:00+00:00",
        "model_run_id": "model_verified_123",
    }
    csv_payload = build_assignment_csv_payload(assigned)
    metadata_payload = build_assignment_metadata_payload(metadata)
    zip_payload = build_assignment_zip_payload(csv_payload, metadata_payload)

    csv_text = csv_payload.decode("utf-8")
    assert csv_text.splitlines()[0] == "a,cluster,segment_name"
    assert json.loads(metadata_payload)["assignment_timestamp_utc"] == "2026-11-23T01:30:00+00:00"
    assert json.loads(metadata_payload)["model_run_id"] == "model_verified_123"

    with ZipFile(BytesIO(zip_payload), "r") as archive:
        assert archive.namelist() == [
            "microsegment_new_assignments.csv",
            "microsegment_assignment_metadata.json",
        ]
        assert archive.read("microsegment_new_assignments.csv") == csv_payload
        assert archive.read("microsegment_assignment_metadata.json") == metadata_payload


def test_repeated_assignments_reuse_model_run_id_and_timestamp():
    from datetime import datetime, timezone

    frame = pd.DataFrame({"a": [1.0, 2.0]})
    manifest = build_model_manifest(
        frame=frame,
        source_name="districts.csv",
        selected_features=["a"],
        algorithm="K-Means",
        algorithm_parameters={"n_clusters": 2, "n_init": 10},
        random_seed=42,
        preprocessor=object(),
        metrics={"n_clusters": 2},
        model_run_id="model_repeated_assignment",
        seed_applicable=True,
    )
    timestamp = datetime(2026, 11, 23, 1, 45, 0, tzinfo=timezone.utc)
    first = build_assignment_manifest(
        manifest=manifest,
        assigned_frame=pd.DataFrame({"a": [1.0], "cluster": [0], "segment_name": ["A"]}),
        original_input_filename="first.csv",
        cluster_labels=[0],
        timestamp=timestamp,
    )
    second = build_assignment_manifest(
        manifest=manifest,
        assigned_frame=pd.DataFrame({"a": [2.0], "cluster": [1], "segment_name": ["B"]}),
        original_input_filename="second.csv",
        cluster_labels=[1],
        timestamp=timestamp,
    )
    assert first["model_run_id"] == second["model_run_id"] == "model_repeated_assignment"
    assert first["assignment_timestamp_utc"] == second["assignment_timestamp_utc"]


@pytest.mark.parametrize("seed", [None, "42", -1, 2**32, True])
def test_kmeans_invalid_or_missing_random_seed_is_rejected(seed):
    manifest = {
        "model_run_id": "model_seed_validation",
        "dataset_fingerprint": "fingerprint",
        "selected_features": ["a"],
        "algorithm": "K-Means",
        "algorithm_parameters": {"n_clusters": 2, "n_init": 10},
        "random_seed": seed,
        "seed_applicable": seed is not None,
    }
    valid, reason = validate_assignment_provenance(manifest)
    assert not valid
    assert "random_seed" in reason


def test_kmeans_missing_random_seed_field_is_rejected():
    manifest = {
        "model_run_id": "model_seed_validation",
        "dataset_fingerprint": "fingerprint",
        "selected_features": ["a"],
        "algorithm": "K-Means",
        "algorithm_parameters": {"n_clusters": 2, "n_init": 10},
        "seed_applicable": True,
    }
    valid, reason = validate_assignment_provenance(manifest)
    assert not valid
    assert "missing random_seed" in reason


def test_kmeans_valid_random_seed_is_accepted():
    manifest = {
        "model_run_id": "model_seed_validation",
        "dataset_fingerprint": "fingerprint",
        "selected_features": ["a"],
        "algorithm": "K-Means",
        "algorithm_parameters": {"n_clusters": 2, "n_init": 10},
        "random_seed": 42,
        "seed_applicable": True,
    }
    valid, reason = validate_assignment_random_seed(manifest)
    assert valid
    assert reason == "Random seed provenance is verified."


def test_dbscan_null_seed_and_not_applicable_are_preserved():
    manifest = {
        "algorithm": "DBSCAN",
        "algorithm_parameters": {"eps": 0.5, "min_samples": 5},
        "random_seed": None,
        "seed_applicable": False,
    }
    valid, reason = validate_assignment_random_seed(manifest)
    assert valid
    assert reason == "DBSCAN seed provenance is consistent."


@pytest.mark.parametrize("n_clusters", [None, 0, -1, True, "2"])
def test_invalid_n_clusters_is_rejected(n_clusters):
    manifest = {
        "algorithm": "K-Means",
        "algorithm_parameters": {"n_clusters": n_clusters},
    }
    valid, reason = validate_cluster_count_parameter(manifest)
    assert not valid
    assert "n_clusters" in reason


def test_missing_n_clusters_is_rejected():
    manifest = {
        "algorithm": "K-Means",
        "algorithm_parameters": {"n_init": 10},
    }
    valid, reason = validate_cluster_count_parameter(manifest)
    assert not valid
    assert "missing n_clusters" in reason


@pytest.mark.parametrize("algorithm", ["K-Means", "K-Medoids"])
def test_valid_cluster_parameter_is_accepted(algorithm):
    manifest = {
        "algorithm": algorithm,
        "algorithm_parameters": {"n_clusters": 4},
    }
    valid, reason = validate_cluster_count_parameter(manifest)
    assert valid
    assert reason == "Cluster count parameter is verified."


def test_dbscan_does_not_require_n_clusters():
    manifest = {
        "algorithm": "DBSCAN",
        "algorithm_parameters": {"eps": 0.5, "min_samples": 5},
    }
    valid, reason = validate_cluster_count_parameter(manifest)
    assert valid
    assert "does not require n_clusters" in reason
