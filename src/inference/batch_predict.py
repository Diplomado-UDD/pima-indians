"""Batch prediction script for diabetes screening."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import yaml


def load_config(config_path: Path) -> dict:
    """Load inference configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model_artifacts(model_dir: Path):
    """Load production model artifacts.

    Args:
        model_dir: Directory containing model artifacts

    Returns:
        Dictionary with model, preprocessor, threshold, metadata
    """
    artifacts_path = model_dir / "model_artifacts.pkl"
    metadata_path = model_dir / "metadata.json"

    if not artifacts_path.exists():
        raise FileNotFoundError(f"Model artifacts not found: {artifacts_path}")

    artifacts = joblib.load(artifacts_path)

    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    return {**artifacts, "metadata": metadata}


def validate_input(df: pd.DataFrame, config: dict) -> tuple:
    """Validate input dataframe.

    Args:
        df: Input dataframe
        config: Inference configuration

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    required_cols = config["validation"]["required_columns"]

    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
        return False, errors

    if config["validation"]["id_column"] not in df.columns:
        if not config["validation"]["allow_missing_id"]:
            errors.append(f"Missing ID column: {config['validation']['id_column']}")
            return False, errors

    for col in required_cols:
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            errors.append(f"Column '{col}' has {null_count} null values")

    if errors:
        return False, errors

    return True, []


def categorize_risk(probabilities: pd.Series, thresholds: dict) -> pd.Series:
    """Categorize risk scores into low/medium/high.

    Args:
        probabilities: Predicted probabilities
        thresholds: Risk category thresholds

    Returns:
        Series with risk categories
    """
    categories = pd.cut(
        probabilities,
        bins=[thresholds["low"], thresholds["medium"], thresholds["high"], 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )
    return categories


def process_batch(
    input_path: Path, output_dir: Path, model_artifacts: dict, config: dict
) -> dict:
    """Process a single batch file.

    Args:
        input_path: Path to input CSV
        output_dir: Output directory
        model_artifacts: Loaded model artifacts
        config: Inference configuration

    Returns:
        Processing report dictionary
    """
    start_time = datetime.now()

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        return {
            "status": "error",
            "file": input_path.name,
            "error": f"Failed to read file: {str(e)}",
            "timestamp": start_time.isoformat(),
        }

    is_valid, errors = validate_input(df, config)
    if not is_valid:
        rejected_path = Path(config["batch"]["rejected_dir"]) / f"rejected_{input_path.name}"
        df.to_csv(rejected_path, index=False)

        error_report_path = (
            Path(config["batch"]["rejected_dir"]) / f"errors_{input_path.stem}.json"
        )
        with open(error_report_path, "w") as f:
            json.dump({"errors": errors, "file": input_path.name}, f, indent=2)

        return {
            "status": "rejected",
            "file": input_path.name,
            "errors": errors,
            "rejected_path": str(rejected_path),
            "timestamp": start_time.isoformat(),
        }

    model = model_artifacts["model"]
    preprocessor = model_artifacts["preprocessor"]
    threshold = model_artifacts["threshold"]
    metadata = model_artifacts.get("metadata", {})

    required_features = config["validation"]["required_columns"]
    X = df[required_features]

    try:
        X_processed = preprocessor.transform(X)
        probabilities = model.predict_proba(X_processed)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
    except Exception as e:
        return {
            "status": "error",
            "file": input_path.name,
            "error": f"Prediction failed: {str(e)}",
            "timestamp": start_time.isoformat(),
        }

    output_data = {}

    if config["validation"]["id_column"] in df.columns:
        output_data[config["validation"]["id_column"]] = df[config["validation"]["id_column"]]

    output_data["screening_recommendation"] = predictions

    if config["output"]["include_probability"]:
        output_data["diabetes_risk_score"] = probabilities

    if config["output"]["include_risk_category"]:
        output_data["risk_category"] = categorize_risk(
            pd.Series(probabilities), config["output"]["risk_thresholds"]
        )

    if config["output"]["include_model_version"]:
        output_data["model_version"] = metadata.get("version", "unknown")

    if config["output"]["include_timestamp"]:
        output_data["prediction_date"] = datetime.now().isoformat()

    output_df = pd.DataFrame(output_data)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"predictions_{input_path.stem}_{timestamp}.csv"
    output_path = output_dir / output_filename

    output_df.to_csv(output_path, index=False)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    positive_count = int(predictions.sum())
    positive_rate = float(predictions.mean())

    report = {
        "status": "success",
        "file": input_path.name,
        "output_file": output_filename,
        "records_processed": len(df),
        "positive_predictions": positive_count,
        "positive_rate": positive_rate,
        "model_version": metadata.get("version", "unknown"),
        "threshold": float(threshold),
        "duration_seconds": duration,
        "timestamp": start_time.isoformat(),
    }

    if config["logging"]["predictions_log_dir"]:
        log_dir = Path(config["logging"]["predictions_log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(report) + "\n")

    return report


def main():
    """CLI entry point for batch prediction."""
    parser = argparse.ArgumentParser(description="Batch diabetes prediction")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV file")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/inference_config.yaml"),
        help="Inference config",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/production"),
        help="Production model directory",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    output_dir = Path(config["batch"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {args.model_dir}")
    model_artifacts = load_model_artifacts(args.model_dir)

    print(f"Processing: {args.input}")
    report = process_batch(args.input, output_dir, model_artifacts, config)

    print(f"\n{'=' * 60}")
    print("BATCH PROCESSING REPORT")
    print(f"{'=' * 60}")
    print(json.dumps(report, indent=2))

    if report["status"] == "success":
        print(f"\nPredictions saved to: {output_dir / report['output_file']}")
        print(f"Records processed: {report['records_processed']}")
        print(f"Positive predictions: {report['positive_predictions']} ({report['positive_rate']:.1%})")
        print(f"Processing time: {report['duration_seconds']:.2f}s")
    else:
        print(f"\nProcessing failed: {report.get('error', 'Unknown error')}")
        if "errors" in report:
            print(f"Validation errors: {report['errors']}")

    if config["logging"]["batch_log_path"]:
        log_path = Path(config["logging"]["batch_log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
