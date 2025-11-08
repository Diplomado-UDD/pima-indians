"""Feature engineering and preprocessing for diabetes prediction."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ZeroImputer(BaseEstimator, TransformerMixin):
    """Replace zero values with median (for biological measurements that cannot be zero)."""

    def __init__(self, columns_to_impute=None):
        """Initialize imputer.

        Args:
            columns_to_impute: List of column names to impute zeros
        """
        self.columns_to_impute = columns_to_impute or []
        self.medians_ = {}

    def fit(self, X, y=None):
        """Fit imputer by calculating medians.

        Args:
            X: Input features
            y: Target (unused)

        Returns:
            self
        """
        X_df = pd.DataFrame(X, columns=self.columns_to_impute)

        for col in self.columns_to_impute:
            non_zero_values = X_df[col][X_df[col] > 0]
            if len(non_zero_values) > 0:
                self.medians_[col] = non_zero_values.median()
            else:
                self.medians_[col] = X_df[col].median()

        return self

    def transform(self, X):
        """Transform by replacing zeros with medians.

        Args:
            X: Input features

        Returns:
            Transformed array
        """
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X, columns=self.columns_to_impute)

        for col in self.columns_to_impute:
            if col in X_df.columns:
                X_df.loc[X_df[col] == 0, col] = self.medians_[col]

        return X_df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create derived features for diabetes prediction."""

    def __init__(self):
        """Initialize feature engineer."""
        self.feature_names_ = []

    def fit(self, X, y=None):
        """Fit feature engineer (stateless).

        Args:
            X: Input features
            y: Target (unused)

        Returns:
            self
        """
        return self

    def transform(self, X):
        """Transform by creating derived features.

        Args:
            X: Input features (DataFrame or array)

        Returns:
            Transformed array with original + derived features
        """
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(
                X,
                columns=[
                    "Pregnancies",
                    "Glucose",
                    "BloodPressure",
                    "SkinThickness",
                    "Insulin",
                    "BMI",
                    "DiabetesPedigreeFunction",
                    "Age",
                ],
            )

        X_df["BMI_Category"] = pd.cut(
            X_df["BMI"],
            bins=[0, 18.5, 25, 30, 100],
            labels=[0, 1, 2, 3],
        ).astype(int)

        X_df["Age_Group"] = pd.cut(
            X_df["Age"],
            bins=[0, 30, 45, 60, 120],
            labels=[0, 1, 2, 3],
        ).astype(int)

        X_df["Glucose_BMI_Interaction"] = X_df["Glucose"] * X_df["BMI"]

        self.feature_names_ = X_df.columns.tolist()

        return X_df.values


def create_preprocessing_pipeline(zero_impute_columns=None):
    """Create preprocessing pipeline.

    Args:
        zero_impute_columns: Columns where zeros should be imputed

    Returns:
        sklearn Pipeline
    """
    if zero_impute_columns is None:
        zero_impute_columns = ["Glucose", "BloodPressure", "SkinThickness", "BMI"]

    pipeline = Pipeline(
        [
            ("zero_imputer", ZeroImputer(columns_to_impute=zero_impute_columns)),
            ("feature_engineer", FeatureEngineer()),
            ("scaler", StandardScaler()),
        ]
    )

    return pipeline


def get_feature_names(preprocessor):
    """Extract feature names from fitted preprocessor.

    Args:
        preprocessor: Fitted preprocessing pipeline

    Returns:
        List of feature names
    """
    if hasattr(preprocessor.named_steps["feature_engineer"], "feature_names_"):
        return preprocessor.named_steps["feature_engineer"].feature_names_
    return [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]
