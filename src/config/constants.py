"""Shared constants for diabetes prediction pipeline."""

# Required feature columns for diabetes prediction
REQUIRED_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# Target column name
TARGET_COLUMN = "Outcome"

# Columns where zero values should be treated as missing (biological impossibility)
ZERO_IMPUTE_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "BMI",
]

# Default ID column for batch predictions
DEFAULT_ID_COLUMN = "patient_id"


