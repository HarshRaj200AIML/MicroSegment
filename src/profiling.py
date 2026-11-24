"""Cluster summaries and cautious, data-derived segment labels."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from .data_loader import infer_column_roles


def kmeans_stability(matrix: object, n_clusters: int = 4, repeats: int = 5) -> float | None:
    values = matrix
    assignments = [KMeans(n_clusters=n_clusters, n_init=1, random_state=seed).fit_predict(values) for seed in range(repeats)]
    if len(assignments) < 2:
        return None
    scores = [adjusted_rand_score(assignments[0], current) for current in assignments[1:]]
    return float(sum(scores) / len(scores))


def cluster_profile(frame: pd.DataFrame, labels: object) -> tuple[pd.DataFrame, dict[int, str]]:
    data = frame.copy()
    data["cluster"] = labels
    roles = infer_column_roles(frame)
    numeric = [column for column in roles["numeric"] + roles["binary"] if column in data.columns]
    profile = data.groupby("cluster", dropna=False).agg(
        size=("cluster", "size"),
        **{f"{column}__mean": (column, "mean") for column in numeric},
        **{f"{column}__median": (column, "median") for column in numeric},
    )
    names: dict[int, str] = {}
    overall = frame[numeric].median(numeric_only=True) if len(numeric) else pd.Series(dtype=float)
    for cluster, row in profile.iterrows():
        differentiators = []
        for column in numeric:
            median = overall.get(column)
            value = row.get(f"{column}__median")
            if pd.notna(median) and pd.notna(value) and median != 0:
                ratio = value / median
                if ratio >= 1.25:
                    differentiators.append(f"higher {column}")
                elif ratio <= 0.75:
                    differentiators.append(f"lower {column}")
        suffix = ", ".join(differentiators[:2]) or "near overall numeric medians"
        names[int(cluster)] = f"Segment {cluster}: {suffix}"
    return profile, names


def categorical_modes(frame: pd.DataFrame, labels: object) -> pd.DataFrame:
    data = frame.copy()
    data["cluster"] = labels
    roles = infer_column_roles(frame)
    categorical = [column for column in roles["categorical"] + roles["binary"] if column in data.columns]
    if len(categorical) == 0:
        return pd.DataFrame(index=sorted(data["cluster"].dropna().unique()))
    return data.groupby("cluster")[categorical].agg(lambda values: values.mode().iloc[0] if not values.mode().empty else None)