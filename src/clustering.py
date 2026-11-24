"""Clustering algorithms and reproducible model results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import os
import scipy.sparse as sp
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA

from .evaluation import evaluate_clustering
from .utils import MemorySafetyError, RANDOM_SEED, get_logger


LOGGER = get_logger("microsegment.clustering")


@dataclass
class ClusteringResult:
    algorithm: str
    labels: np.ndarray
    model: Any
    metrics: dict[str, float | int | None]
    inertia: float | None = None


def _dense(matrix: Any) -> np.ndarray:
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def estimated_matrix_bytes(matrix: Any) -> int:
    if sp.issparse(matrix):
        return int(matrix.nnz * (8 + 4) + matrix.shape[0] * 8 + matrix.shape[1] * 8)
    values = np.asarray(matrix)
    return int(values.nbytes)


def clustering_memory_limit_bytes() -> int:
    try:
        return int(os.getenv("MICROSEGMENT_MAX_CLUSTERING_BYTES", str(512 * 1024 * 1024)))
    except ValueError:
        LOGGER.warning("Ignoring invalid MICROSEGMENT_MAX_CLUSTERING_BYTES value")
        return 512 * 1024 * 1024


def validate_matrix_memory(matrix: Any) -> dict[str, int | bool]:
    estimated = estimated_matrix_bytes(matrix)
    limit = clustering_memory_limit_bytes()
    if estimated > limit:
        raise MemorySafetyError(
            f"Estimated clustering memory is {estimated:,} bytes, exceeding "
            f"the configured limit of {limit:,} bytes."
        )
    return {"estimated_bytes": estimated, "limit_bytes": limit, "is_sparse": bool(sp.issparse(matrix))}


def matrix_row_count(matrix: Any) -> int:
    return int(matrix.shape[0])


def fit_kmeans(matrix: Any, n_clusters: int = 4, random_state: int = RANDOM_SEED) -> ClusteringResult:
    validate_matrix_memory(matrix)
    values = matrix
    row_count = matrix_row_count(values)
    if row_count < 2 or not 2 <= n_clusters <= row_count:
        raise ValueError("K must be between 2 and the number of rows.")
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(values)
    return ClusteringResult("K-Means", labels, model, evaluate_clustering(values, labels), float(model.inertia_))


def fit_dbscan(matrix: Any, eps: float = 0.5, min_samples: int = 5) -> ClusteringResult:
    validate_matrix_memory(matrix)
    values = matrix
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(values)
    return ClusteringResult("DBSCAN", labels, model, evaluate_clustering(values, labels))


def fit_kmedoids(matrix: Any, n_clusters: int = 4, random_state: int = RANDOM_SEED) -> ClusteringResult:
    try:
        from sklearn_extra.cluster import KMedoids
    except ImportError as exc:
        raise ImportError("K-Medoids requires the optional scikit-learn-extra package.") from exc
    validate_matrix_memory(matrix)
    values = _dense(matrix)
    model = KMedoids(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(values)
    return ClusteringResult("K-Medoids", labels, model, evaluate_clustering(values, labels))


def elbow_scores(matrix: Any, maximum_k: int = 10, random_state: int = RANDOM_SEED) -> list[dict[str, float | int]]:
    validate_matrix_memory(matrix)
    values = matrix
    results = []
    row_count = matrix_row_count(values)
    for k in range(2, min(maximum_k, row_count - 1) + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(values)
        results.append({"k": k, "inertia": float(model.inertia_)})
    return results


def pca_projection(matrix: Any) -> tuple[np.ndarray, list[float]]:
    validate_matrix_memory(matrix)
    if sp.issparse(matrix):
        dense_estimate = int(matrix.shape[0] * matrix.shape[1] * 8)
        limit = clustering_memory_limit_bytes()
        if dense_estimate > limit:
            raise MemorySafetyError(
                f"PCA would require approximately {dense_estimate:,} bytes, exceeding "
                f"the configured limit of {limit:,} bytes."
            )
    values = _dense(matrix)
    if values.shape[1] == 0:
        raise ValueError("No transformed features are available for PCA.")
    components = min(2, values.shape[0], values.shape[1])
    reducer = PCA(n_components=components, random_state=RANDOM_SEED)
    projection = reducer.fit_transform(values)
    if components == 1:
        projection = np.column_stack((projection[:, 0], np.zeros(len(projection))))
    return projection, [float(value) for value in reducer.explained_variance_ratio_]