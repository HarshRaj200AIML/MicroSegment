"""Schema-aware preprocessing without assuming dataset-specific variables."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_loader import infer_column_roles


def build_preprocessor(frame: pd.DataFrame, features: Sequence[str] | None = None) -> tuple[ColumnTransformer, list[str]]:
    selected = list(features) if features is not None else [column for column in frame.columns if column not in infer_column_roles(frame)["identifier"]]
    selected = [column for column in selected if column in frame.columns]
    if not selected:
        raise ValueError("Select at least one non-identifier feature.")
    numeric = [column for column in selected if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in selected if column not in numeric]
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scaler", StandardScaler())])
        transformers.append(("numeric", numeric_pipeline, numeric))
    if categorical:
        categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))])
        transformers.append(("categorical", categorical_pipeline, categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True), selected


def preprocess_data(frame: pd.DataFrame, features: Sequence[str] | None = None):
    preprocessor, selected = build_preprocessor(frame, features)
    matrix = preprocessor.fit_transform(frame[selected])
    return matrix, preprocessor, selected


def quality_report(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "shape": frame.shape,
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing": frame.isna().sum().sort_values(ascending=False),
        "constant_columns": [column for column in frame if frame[column].nunique(dropna=False) <= 1],
    }