"""Data validation module for diabetes prediction pipeline."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema

from src.config.constants import REQUIRED_COLUMNS


class DiabetesDataValidator:
    """Validates input data for diabetes prediction."""

    REQUIRED_COLUMNS = REQUIRED_COLUMNS

    def __init__(self):
        """Initialize validator with schema."""
        self.schema = DataFrameSchema(
            {
                "Pregnancies": Column(int, checks=[pa.Check.ge(0)], nullable=False),
                "Glucose": Column(float, checks=[pa.Check.ge(0)], nullable=False),
                "BloodPressure": Column(float, checks=[pa.Check.ge(0)], nullable=False),
                "SkinThickness": Column(float, checks=[pa.Check.ge(0)], nullable=False),
                "Insulin": Column(float, checks=[pa.Check.ge(0)], nullable=False),
                "BMI": Column(float, checks=[pa.Check.ge(0)], nullable=False),
                "DiabetesPedigreeFunction": Column(float, checks=[pa.Check.ge(0)], nullable=False),
                "Age": Column(int, checks=[pa.Check.ge(0), pa.Check.le(120)], nullable=False),
            },
            strict=False,
        )

    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate dataframe schema against required columns and data types.

        Checks for:
        - Missing required columns
        - Data type mismatches
        - Value constraints (e.g., non-negative values, age <= 120)

        Args:
            df: Input dataframe

        Returns:
            Tuple of (is_valid, error_messages) where error_messages is a list
            of validation error descriptions if validation fails
        """
        errors = []

        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return False, errors

        try:
            self.schema.validate(df[self.REQUIRED_COLUMNS], lazy=True)
            return True, []
        except pa.errors.SchemaErrors as e:
            for _, row in e.failure_cases.iterrows():
                errors.append(
                    f"Column '{row['column']}' failed check '{row['check']}' "
                    f"at index {row['index']}"
                )
            return False, errors

    def detect_outliers(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """Detect outliers using z-score method.

        Args:
            df: Input dataframe

        Returns:
            Dictionary mapping column names to list of outlier indices
        """
        outliers = {}
        z_threshold = 3.0

        for col in self.REQUIRED_COLUMNS:
            if col in df.columns and len(df) > 3:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    z_scores = ((df[col] - mean) / std).abs()
                    outlier_indices = df[z_scores > z_threshold].index.tolist()
                    if outlier_indices:
                        outliers[col] = outlier_indices

        return outliers

    def validate_file(
        self, file_path: Path, rejected_dir: Path
    ) -> Tuple[pd.DataFrame, Dict]:
        """Validate input file and separate valid/invalid records.

        Args:
            file_path: Path to input CSV file
            rejected_dir: Directory to save rejected records

        Returns:
            Tuple of (valid_dataframe, validation_report)
        """
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            return None, {"status": "error", "message": f"Failed to read file: {str(e)}"}

        report = {
            "file": file_path.name,
            "total_records": len(df),
            "valid_records": 0,
            "rejected_records": 0,
            "errors": [],
            "warnings": [],
        }

        is_valid, errors = self.validate_schema(df)
        if not is_valid:
            report["status"] = "rejected"
            report["errors"] = errors
            rejected_path = rejected_dir / f"rejected_{file_path.name}"
            df.to_csv(rejected_path, index=False)

            error_report_path = rejected_dir / f"errors_{file_path.stem}.json"
            with open(error_report_path, "w") as f:
                json.dump(report, f, indent=2)

            return None, report

        outliers = self.detect_outliers(df)
        if outliers:
            report["warnings"].append(f"Outliers detected: {outliers}")

        report["status"] = "valid"
        report["valid_records"] = len(df)

        return df, report


def main():
    """CLI entry point for data validation."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate diabetes prediction input data")
    parser.add_argument("input_file", type=Path, help="Path to input CSV file")
    parser.add_argument("--output-dir", type=Path, default=Path("data/validated"))
    parser.add_argument("--rejected-dir", type=Path, default=Path("data/rejected"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.rejected_dir.mkdir(parents=True, exist_ok=True)

    validator = DiabetesDataValidator()
    valid_df, report = validator.validate_file(args.input_file, args.rejected_dir)

    print(f"Validation Report: {json.dumps(report, indent=2)}")

    if valid_df is not None:
        output_path = args.output_dir / f"validated_{args.input_file.name}"
        valid_df.to_csv(output_path, index=False)
        print(f"Valid data saved to: {output_path}")
    else:
        print("Validation failed. Check rejected directory for details.")


if __name__ == "__main__":
    main()
