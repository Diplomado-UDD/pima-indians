"""Batch prediction script for diabetes screening."""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import yaml

from src.config.constants import DEFAULT_ID_COLUMN, REQUIRED_COLUMNS
from src.data.validate_input import DiabetesDataValidator
from src.monitoring.drift_detection import detect_drift

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
    """Validate input dataframe using DiabetesDataValidator.

    Args:
        df: Input dataframe
        config: Inference configuration

    Returns:
        Tuple of (is_valid, error_messages)
    """
    validator = DiabetesDataValidator()

    # Validate schema using validator
    is_valid, errors = validator.validate_schema(df)
    if not is_valid:
        return False, errors

    # Check ID column if required
    id_column = config["validation"].get("id_column", DEFAULT_ID_COLUMN)
    if id_column not in df.columns:
        if not config["validation"].get("allow_missing_id", False):
            errors.append(f"Missing ID column: {id_column}")
            return False, errors

    # Check for null values (validator checks types but not nulls explicitly)
    required_cols = config["validation"]["required_columns"]
    for col in required_cols:
        if col in df.columns and df[col].isnull().any():
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
    input_path: Path,
    output_dir: Path,
    model_artifacts: dict,
    config: dict,
    max_retries: int = 1,
    backoff_seconds: int = 0,
) -> dict:
    """Process a single batch file with retry logic.

    Args:
        input_path: Path to input CSV
        output_dir: Output directory
        model_artifacts: Loaded model artifacts
        config: Inference configuration
        max_retries: Maximum number of retry attempts
        backoff_seconds: Seconds to wait between retries

    Returns:
        Processing report dictionary
    """
    start_time = datetime.now()

    for attempt in range(max_retries):
        try:
            df = pd.read_csv(input_path)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {backoff_seconds}s...")
                time.sleep(backoff_seconds)
                continue
            return {
                "status": "error",
                "file": input_path.name,
                "error": f"Failed to read file after {max_retries} attempts: {str(e)}",
                "timestamp": start_time.isoformat(),
            }

    is_valid, errors = validate_input(df, config)
    if not is_valid:
        rejected_dir = Path(config["batch"]["rejected_dir"])
        rejected_dir.mkdir(parents=True, exist_ok=True)
        rejected_path = rejected_dir / f"rejected_{input_path.name}"
        
        try:
            df.to_csv(rejected_path, index=False)
        except Exception as e:
            logger.error(f"Failed to save rejected file: {e}")

        error_report_path = rejected_dir / f"errors_{input_path.stem}.json"
        try:
            with open(error_report_path, "w") as f:
                json.dump({"errors": errors, "file": input_path.name}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save error report: {e}")

        logger.warning(f"Batch {input_path.name} rejected: {errors}")
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

    # Use required columns from config (should match REQUIRED_COLUMNS)
    required_features = config["validation"]["required_columns"]
    X = df[required_features]

    # Perform drift detection if enabled
    drift_summary = None
    if config.get("monitoring", {}).get("enable_drift_detection", False):
        try:
            reference_path = Path(config["monitoring"]["reference_data_path"])
            drift_threshold = config["monitoring"].get("drift_threshold", 0.05)
            drift_output_dir = Path(config["monitoring"]["drift_report_dir"])
            
            # Save current batch temporarily for drift detection
            temp_current_path = output_dir / f"temp_current_{input_path.stem}.csv"
            X.to_csv(temp_current_path, index=False)
            
            drift_summary = detect_drift(
                reference_path=reference_path,
                current_path=temp_current_path,
                output_dir=drift_output_dir,
                threshold=drift_threshold,
            )
            
            # Clean up temp file
            temp_current_path.unlink(missing_ok=True)
            
            if drift_summary.get("dataset_drift", False):
                logger.warning(
                    f"Data drift detected in {input_path.name}: "
                    f"{drift_summary['number_of_drifted_columns']} columns drifted"
                )
        except Exception as e:
            logger.warning(f"Drift detection failed: {e}. Continuing with prediction.")

    try:
        X_processed = preprocessor.transform(X)
        probabilities = model.predict_proba(X_processed)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
    except Exception as e:
        logger.error(f"Prediction failed for {input_path.name}: {e}")
        return {
            "status": "error",
            "file": input_path.name,
            "error": f"Prediction failed: {str(e)}",
            "timestamp": start_time.isoformat(),
        }

    output_data = {}

    id_column = config["validation"].get("id_column", DEFAULT_ID_COLUMN)
    if id_column in df.columns:
        output_data[id_column] = df[id_column]

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

    try:
        output_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save predictions: {e}")
        return {
            "status": "error",
            "file": input_path.name,
            "error": f"Failed to save predictions: {str(e)}",
            "timestamp": start_time.isoformat(),
        }

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

    if drift_summary:
        report["drift_detected"] = drift_summary.get("dataset_drift", False)
        report["drifted_columns"] = drift_summary.get("number_of_drifted_columns", 0)

    if config.get("logging", {}).get("predictions_log_dir"):
        log_dir = Path(config["logging"]["predictions_log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(report) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write prediction log: {e}")

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

    # Set log level from config if available
    config = load_config(args.config)
    log_level = config.get("logging", {}).get("log_level", "INFO")
    logging.getLogger().setLevel(getattr(logging, log_level.upper(), logging.INFO))

    output_dir = Path(config["batch"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model from: {args.model_dir}")
    try:
        model_artifacts = load_model_artifacts(args.model_dir)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    logger.info(f"Processing: {args.input}")
    
    # Get retry settings from config
    retry_config = config.get("retry", {})
    max_retries = retry_config.get("max_attempts", 1)
    backoff_seconds = retry_config.get("backoff_seconds", 0)
    
    report = process_batch(
        args.input, output_dir, model_artifacts, config, max_retries, backoff_seconds
    )

    logger.info(f"\n{'=' * 60}")
    logger.info("BATCH PROCESSING REPORT")
    logger.info(f"{'=' * 60}")
    logger.info(json.dumps(report, indent=2))

    if report["status"] == "success":
        logger.info(f"\nPredictions saved to: {output_dir / report['output_file']}")
        logger.info(f"Records processed: {report['records_processed']}")
        logger.info(
            f"Positive predictions: {report['positive_predictions']} "
            f"({report['positive_rate']:.1%})"
        )
        logger.info(f"Processing time: {report['duration_seconds']:.2f}s")
        if report.get("drift_detected"):
            logger.warning(f"Data drift detected: {report.get('drifted_columns', 0)} columns")
    else:
        logger.error(f"\nProcessing failed: {report.get('error', 'Unknown error')}")
        if "errors" in report:
            logger.error(f"Validation errors: {report['errors']}")
        return 1

    if config.get("logging", {}).get("batch_log_path"):
        log_path = Path(config["logging"]["batch_log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps(report) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write batch log: {e}")

    return 0


if __name__ == "__main__":
    exit(main())
