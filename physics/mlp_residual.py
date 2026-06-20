"""
E7 learned correction: OpenAP → MLP residual predictor.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from physics.eval_framework import CATEGORICAL, RANDOM_STATE


def make_mlp_pipeline(feature_cols: list[str]) -> Pipeline:
    numeric = [c for c in feature_cols if c not in CATEGORICAL]
    cat = [c for c in feature_cols if c in CATEGORICAL]
    prep = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ],
        remainder="drop",
    )
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("prep", prep), ("model", mlp)])


def train_mlp_residual(
    feature_cols: list[str],
    X_train,
    X_test,
    y_residual_train: np.ndarray,
    physics_test: np.ndarray,
) -> np.ndarray:
    pipe = make_mlp_pipeline(feature_cols)
    pipe.fit(X_train, y_residual_train)
    return physics_test + pipe.predict(X_test)