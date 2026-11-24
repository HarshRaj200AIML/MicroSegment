"""Conservative, schema-independent feature selection helpers."""

from __future__ import annotations

import pandas as pd

from .data_loader import infer_column_roles


def default_clustering_features(frame: pd.DataFrame) -> list[str]:
    roles = infer_column_roles(frame)
    candidates = roles["numeric"] + roles["categorical"] + roles["binary"]
    return [column for column in candidates if frame[column].nunique(dropna=True) > 1]


def feature_audit(frame: pd.DataFrame) -> pd.DataFrame:
    roles = infer_column_roles(frame)
    role_lookup = {column: role for role, columns in roles.items() for column in columns}
    return pd.DataFrame({
        "column": frame.columns.astype(str),
        "dtype": [str(frame[column].dtype) for column in frame.columns],
        "role": [role_lookup.get(column, "other") for column in frame.columns],
        "missing": [int(frame[column].isna().sum()) for column in frame.columns],
        "unique_values": [int(frame[column].nunique(dropna=True)) for column in frame.columns],
    })