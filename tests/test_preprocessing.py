"""Tests for preprocessing pipeline."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.features.preprocess import ZeroImputer, FeatureEngineer, create_preprocessing_pipeline


class TestZeroImputer:
    """Test ZeroImputer transformer."""

    def test_imputer_replaces_zeros(self):
        """Test that zeros are replaced with median of non-zero values."""
        X = pd.DataFrame([[0, 100], [0, 120], [80, 110], [90, 0]], columns=["col1", "col2"])

        imputer = ZeroImputer(columns_to_impute=["col1", "col2"])
        imputer.fit(X)
        X_transformed = imputer.transform(X)

        assert X_transformed.iloc[0, 0] == 85.0
        assert X_transformed.iloc[1, 0] == 85.0
        assert X_transformed.iloc[0, 1] == 100.0

    def test_imputer_preserves_non_zeros(self):
        """Test that non-zero values are preserved."""
        X = pd.DataFrame([[50, 100], [60, 120]], columns=["col1", "col2"])

        imputer = ZeroImputer(columns_to_impute=["col1", "col2"])
        imputer.fit(X)
        X_transformed = imputer.transform(X)

        pd.testing.assert_frame_equal(X, X_transformed)


class TestFeatureEngineer:
    """Test FeatureEngineer transformer."""

    def test_creates_derived_features(self):
        """Test that derived features are created."""
        X = np.array([[1, 120, 70, 20, 80, 25.0, 0.5, 35]])
        engineer = FeatureEngineer()
        X_transformed = engineer.fit_transform(X)

        assert X_transformed.shape[1] > X.shape[1]
        assert engineer.feature_names_[-1] == "Glucose_BMI_Interaction"

    def test_bmi_categories(self):
        """Test BMI categorization."""
        X = np.array([
            [1, 120, 70, 20, 80, 18.0, 0.5, 35],
            [1, 120, 70, 20, 80, 24.0, 0.5, 35],
            [1, 120, 70, 20, 80, 29.0, 0.5, 35],
            [1, 120, 70, 20, 80, 35.0, 0.5, 35],
        ])

        engineer = FeatureEngineer()
        X_transformed = engineer.fit_transform(X)
        feature_idx = engineer.feature_names_.index("BMI_Category")

        assert X_transformed[0, feature_idx] == 0
        assert X_transformed[1, feature_idx] == 1
        assert X_transformed[2, feature_idx] == 2
        assert X_transformed[3, feature_idx] == 3


class TestPreprocessingPipeline:
    """Test full preprocessing pipeline."""

    def test_pipeline_end_to_end(self):
        """Test full preprocessing pipeline."""
        X = pd.DataFrame({
            "Pregnancies": [1, 2],
            "Glucose": [0, 120],
            "BloodPressure": [70, 0],
            "SkinThickness": [20, 30],
            "Insulin": [80, 0],
            "BMI": [25.0, 30.0],
            "DiabetesPedigreeFunction": [0.5, 0.6],
            "Age": [35, 45],
        })

        pipeline = create_preprocessing_pipeline()
        X_transformed = pipeline.fit_transform(X)

        assert X_transformed.shape[0] == 2
        assert not np.isnan(X_transformed).any()

    def test_pipeline_is_sklearn_compatible(self):
        """Test that pipeline is a valid sklearn Pipeline."""
        pipeline = create_preprocessing_pipeline()
        assert isinstance(pipeline, Pipeline)
        assert hasattr(pipeline, "fit")
        assert hasattr(pipeline, "transform")
