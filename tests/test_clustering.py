import numpy as np
import pandas as pd

from src.clustering import fit_dbscan, fit_kmeans, pca_projection
from src.profiling import cluster_profile


def test_kmeans_metrics_labels_profiles_and_projection():
    values = np.array([[0.0, 0.0], [0.1, 0.2], [-0.1, 0.1], [5.0, 5.0], [5.1, 5.2], [4.9, 5.1]])
    result = fit_kmeans(values, n_clusters=2)
    assert len(result.labels) == len(values)
    assert result.metrics["n_clusters"] == 2
    assert result.metrics["silhouette"] is not None
    profile, names = cluster_profile(pd.DataFrame(values, columns=["measure_a", "measure_b"]), result.labels)
    assert profile["size"].sum() == len(values)
    assert len(names) == 2
    projection, variance = pca_projection(values)
    assert projection.shape == (6, 2)
    assert len(variance) == 2


def test_dbscan_single_cluster_metrics_are_undefined_not_crash():
    result = fit_dbscan(np.zeros((5, 2)), eps=1.0, min_samples=2)
    assert result.metrics["n_clusters"] == 1
    assert result.metrics["silhouette"] is None