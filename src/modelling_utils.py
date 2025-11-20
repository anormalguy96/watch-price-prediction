from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def get_feature_lists(
    df: pd.DataFrame,
    target_col: str,
    extra_drop_cols: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    drop_cols = [target_col]
    if extra_drop_cols:
        drop_cols.extend(extra_drop_cols)

    drop_cols = [c for c in drop_cols if c in df.columns]

    feature_cols = [c for c in df.columns if c not in drop_cols]

    numeric_features: List[str] = []
    categorical_features: List[str] = []

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    return feature_cols, numeric_features, categorical_features


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def build_regression_pipeline(
    model,
    numeric_features: List[str],
    categorical_features: List[str],
) -> Pipeline:
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])

    return pipe


def evaluate_regression_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "Model",
    return_metrics: bool = False,
) -> Optional[Dict[str, float]]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)  # no 'squared' argument for older sklearn
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)

    print(f"{label} performance:")
    print(f"  MAE  : {mae:,.2f}")
    print(f"  RMSE : {rmse:,.2f}")
    print(f"  R²   : {r2:.3f}")
    print("-" * 40)

    if return_metrics:
        return {"mae": mae, "rmse": rmse, "r2": r2}
    return None


def fit_and_evaluate_regression(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    label: str = "Model",
    return_metrics: bool = False,
) -> Optional[Dict[str, float]]:
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    return evaluate_regression_model(
        y_true=y_test,
        y_pred=y_pred,
        label=label,
        return_metrics=return_metrics,
    )