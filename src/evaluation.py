"""Internal clustering metrics with explicit handling for noise and edge cases."""

from __future__ import annotations

from typing import Any

import numpy as np
import os
import scipy.sparse as sp
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def evaluate_clustering(matrix: Any, labels: Any) -> dict[str, float | int | None]:
    labels_array = np.asarray(labels)
    values = matrix
    non_noise = labels_array != -1
    valid_values = values[non_noise]
    valid_labels = labels_array[non_noise]
    clusters = np.unique(valid_labels)
    scores: dict[str, float | int | None] = {
        "n_clusters": int(len(np.unique(labels_array[labels_array != -1]))),
        "noise_points": int(np.sum(labels_array == -1)),
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
    }
    if len(clusters) < 2 or len(valid_labels) <= len(clusters):
        return scores
    try:
        scores["silhouette"] = float(silhouette_score(valid_values, valid_labels, sample_size=min(len(valid_labels), 5000), random_state=42))
    except (ValueError, FloatingPointError):
        scores["silhouette"] = None
    dense_values = valid_values
    if sp.issparse(valid_values):
        try:
            dense_limit = int(os.getenv("MICROSEGMENT_MAX_CLUSTERING_BYTES", str(512 * 1024 * 1024)))
        except ValueError:
            dense_limit = 512 * 1024 * 1024
        dense_bytes = int(valid_values.shape[0] * valid_values.shape[1] * 8)
        dense_values = valid_values.toarray() if dense_bytes <= dense_limit else None
    if dense_values is not None:
        try:
            scores["davies_bouldin"] = float(davies_bouldin_score(dense_values, valid_labels))
        except (ValueError, FloatingPointError):
            scores["davies_bouldin"] = None
        try:
            scores["calinski_harabasz"] = float(calinski_harabasz_score(dense_values, valid_labels))
        except (ValueError, FloatingPointError):
            scores["calinski_harabasz"] = None
    return scores