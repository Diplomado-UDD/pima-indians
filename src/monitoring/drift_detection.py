"""Data drift detection for diabetes prediction model."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.config.constants import REQUIRED_COLUMNS


def detect_drift(reference_path: Path, current_path: Path, output_dir: Path, threshold: float = 0.05):
    """Detect data drift between reference and current datasets.

    Args:
        reference_path: Path to reference dataset (training data)
        current_path: Path to current dataset (production data)
        output_dir: Directory to save drift reports
        threshold: P-value threshold for drift detection

    Returns:
        Dictionary with drift results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)

    reference_features = reference_df[REQUIRED_COLUMNS]
    current_features = current_df[REQUIRED_COLUMNS]

    report = Report(metrics=[DataDriftPreset()])

    report.run(reference_data=reference_features, current_data=current_features)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"drift_report_{timestamp}.html"
    report.save_html(str(report_path))

    drift_results = report.as_dict()

    metrics = drift_results["metrics"][0]["result"]

    drift_summary = {
        "timestamp": datetime.now().isoformat(),
        "reference_file": str(reference_path),
        "current_file": str(current_path),
        "dataset_drift": metrics.get("dataset_drift", False),
        "number_of_drifted_columns": metrics.get("number_of_drifted_columns", 0),
        "share_of_drifted_columns": metrics.get("share_of_drifted_columns", 0.0),
        "drift_by_columns": {},
    }

    if "drift_by_columns" in metrics:
        for col, col_drift in metrics["drift_by_columns"].items():
            drift_summary["drift_by_columns"][col] = {
                "drift_detected": col_drift.get("drift_detected", False),
                "drift_score": col_drift.get("drift_score", 0.0),
            }

    summary_path = output_dir / f"drift_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(drift_summary, f, indent=2)

    print("Drift Detection Report")
    print("=" * 60)
    print(f"Dataset drift detected: {drift_summary['dataset_drift']}")
    print(f"Drifted columns: {drift_summary['number_of_drifted_columns']}/{len(REQUIRED_COLUMNS)}")
    print(f"Share of drifted columns: {drift_summary['share_of_drifted_columns']:.1%}")
    print()

    if drift_summary["drift_by_columns"]:
        print("Drift by column:")
        for col, info in drift_summary["drift_by_columns"].items():
            status = "DRIFT" if info["drift_detected"] else "OK"
            print(f"  {col:30s} {status:6s} (score: {info['drift_score']:.4f})")

    print()
    print(f"Full report: {report_path}")
    print(f"Summary JSON: {summary_path}")

    return drift_summary


def main():
    """CLI entry point for drift detection."""
    parser = argparse.ArgumentParser(description="Detect data drift in diabetes prediction data")
    parser.add_argument(
        "--reference", type=Path, required=True, help="Path to reference dataset (training data)"
    )
    parser.add_argument(
        "--current", type=Path, required=True, help="Path to current dataset (production data)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/drift"),
        help="Output directory for reports",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.05, help="P-value threshold for drift detection"
    )
    args = parser.parse_args()

    drift_summary = detect_drift(args.reference, args.current, args.output_dir, args.threshold)

    if drift_summary["dataset_drift"]:
        print("\n⚠️  WARNING: Data drift detected!")
        print("   Consider retraining the model or investigating data quality issues.")
        return 1
    else:
        print("\n✓ No significant drift detected")
        return 0


if __name__ == "__main__":
    exit(main())
