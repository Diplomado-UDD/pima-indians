"""Tests for data validation."""

import pandas as pd
import pytest

from src.data.validate_input import DiabetesDataValidator


class TestDiabetesDataValidator:
    """Test data validation."""

    def test_valid_data_passes(self):
        """Test that valid data passes validation."""
        df = pd.DataFrame({
            "Pregnancies": [1, 2],
            "Glucose": [100.0, 120.0],
            "BloodPressure": [70.0, 80.0],
            "SkinThickness": [20.0, 30.0],
            "Insulin": [80.0, 100.0],
            "BMI": [25.0, 30.0],
            "DiabetesPedigreeFunction": [0.5, 0.6],
            "Age": [35, 45],
        })

        validator = DiabetesDataValidator()
        is_valid, errors = validator.validate_schema(df)

        assert is_valid, f"Validation failed with errors: {errors}"
        assert len(errors) == 0

    def test_missing_columns_rejected(self):
        """Test that missing required columns are rejected."""
        df = pd.DataFrame({
            "Pregnancies": [1, 2],
            "Glucose": [100, 120],
        })

        validator = DiabetesDataValidator()
        is_valid, errors = validator.validate_schema(df)

        assert not is_valid
        assert len(errors) > 0
        assert "Missing required columns" in errors[0]

    def test_negative_values_rejected(self):
        """Test that negative values are rejected."""
        df = pd.DataFrame({
            "Pregnancies": [1, -1],
            "Glucose": [100, 120],
            "BloodPressure": [70, 80],
            "SkinThickness": [20, 30],
            "Insulin": [80, 100],
            "BMI": [25.0, 30.0],
            "DiabetesPedigreeFunction": [0.5, 0.6],
            "Age": [35, 45],
        })

        validator = DiabetesDataValidator()
        is_valid, errors = validator.validate_schema(df)

        assert not is_valid

    def test_age_validation(self):
        """Test age validation (must be <= 120)."""
        df = pd.DataFrame({
            "Pregnancies": [1],
            "Glucose": [100],
            "BloodPressure": [70],
            "SkinThickness": [20],
            "Insulin": [80],
            "BMI": [25.0],
            "DiabetesPedigreeFunction": [0.5],
            "Age": [150],
        })

        validator = DiabetesDataValidator()
        is_valid, errors = validator.validate_schema(df)

        assert not is_valid

    def test_outlier_detection(self):
        """Test outlier detection returns dictionary."""
        df = pd.DataFrame({
            "Pregnancies": [1, 2, 3, 2, 1],
            "Glucose": [100.0, 110.0, 120.0, 105.0, 115.0],
            "BloodPressure": [70.0, 75.0, 80.0, 72.0, 78.0],
            "SkinThickness": [20.0, 25.0, 30.0, 22.0, 28.0],
            "Insulin": [80.0, 90.0, 100.0, 85.0, 95.0],
            "BMI": [25.0, 27.0, 29.0, 26.0, 28.0],
            "DiabetesPedigreeFunction": [0.5, 0.6, 0.7, 0.55, 0.65],
            "Age": [35, 40, 45, 38, 42],
        })

        validator = DiabetesDataValidator()
        outliers = validator.detect_outliers(df)

        assert isinstance(outliers, dict)
