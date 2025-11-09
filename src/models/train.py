"""Training script for recall-optimized diabetes prediction model."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.features.preprocess import create_preprocessing_pipeline


def load_config(config_path: Path) -> dict:
    """Load training configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(data_path: Path, config: dict) -> tuple:
    """Load and split data.

    Args:
        data_path: Path to dataset
        config: Training configuration

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    df = pd.read_csv(data_path)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    random_seed = config["data"]["random_seed"]
    test_size = config["data"]["test_size"]
    val_size = config["data"]["val_size"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )

    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_seed, stratify=y_temp
    )

    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    print(f"Train positive rate: {y_train.mean():.3f}")
    print(f"Val positive rate: {y_val.mean():.3f}")
    print(f"Test positive rate: {y_test.mean():.3f}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def find_optimal_threshold(y_true, y_proba, min_recall=0.85):
    """Find threshold maximizing precision while maintaining minimum recall.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        min_recall: Minimum required recall

    Returns:
        Optimal threshold
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # precision_recall_curve returns n+1 precisions/recalls but n thresholds
    # We need to align them by excluding the last precision/recall
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    valid_idx = recalls >= min_recall
    if not any(valid_idx):
        print(f"Warning: Cannot achieve recall >= {min_recall}")
        return 0.5

    valid_precisions = precisions[valid_idx]
    valid_thresholds = thresholds[valid_idx]

    best_idx = np.argmax(valid_precisions)
    return valid_thresholds[best_idx]


class RecallOptimizer:
    """Optuna objective for recall-optimized hyperparameter tuning."""

    def __init__(self, X_train, y_train, X_val, y_val, config, preprocessor):
        """Initialize optimizer.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            config: Training configuration
            preprocessor: Fitted preprocessing pipeline
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.config = config
        self.preprocessor = preprocessor

    def __call__(self, trial):
        """Optuna objective function.

        Args:
            trial: Optuna trial

        Returns:
            F2-score (or -1 if constraints violated)
        """
        search_space = self.config["hyperparameter_tuning"]["search_space"]

        params = {
            "learning_rate": trial.suggest_float(
                "learning_rate",
                search_space["learning_rate"]["min"],
                search_space["learning_rate"]["max"],
                log=search_space["learning_rate"].get("log", False),
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                search_space["max_depth"]["min"],
                search_space["max_depth"]["max"],
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves",
                search_space["num_leaves"]["min"],
                search_space["num_leaves"]["max"],
            ),
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight",
                search_space["scale_pos_weight"]["min"],
                search_space["scale_pos_weight"]["max"],
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples",
                search_space["min_child_samples"]["min"],
                search_space["min_child_samples"]["max"],
            ),
            "subsample": trial.suggest_float(
                "subsample",
                search_space["subsample"]["min"],
                search_space["subsample"]["max"],
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                search_space["colsample_bytree"]["min"],
                search_space["colsample_bytree"]["max"],
            ),
            "random_state": self.config["model"]["random_state"],
            "verbose": -1,
        }

        model = LGBMClassifier(**params)
        model.fit(self.X_train, self.y_train)

        y_proba = model.predict_proba(self.X_val)[:, 1]

        threshold = find_optimal_threshold(
            self.y_val, y_proba, self.config["optimization"]["min_recall_constraint"]
        )

        y_pred = (y_proba >= threshold).astype(int)

        recall = recall_score(self.y_val, y_pred)
        precision = precision_score(self.y_val, y_pred, zero_division=0)

        if recall < self.config["optimization"]["min_recall_constraint"]:
            return -1.0
        if precision < self.config["optimization"]["min_precision_constraint"]:
            return -1.0

        f2 = fbeta_score(self.y_val, y_pred, beta=self.config["optimization"]["beta"])

        return f2


def train_model(config: dict, X_train, y_train, X_val, y_val, preprocessor):
    """Train recall-optimized model with hyperparameter tuning.

    Args:
        config: Training configuration
        X_train, y_train: Training data
        X_val, y_val: Validation data
        preprocessor: Fitted preprocessing pipeline

    Returns:
        Tuple of (best_model, best_params, optimal_threshold)
    """
    if config["hyperparameter_tuning"]["enabled"]:
        print("Running hyperparameter optimization...")

        optimizer = RecallOptimizer(X_train, y_train, X_val, y_val, config, preprocessor)

        study = optuna.create_study(direction="maximize")
        study.optimize(
            optimizer,
            n_trials=config["hyperparameter_tuning"]["n_trials"],
            timeout=config["hyperparameter_tuning"].get("timeout_seconds"),
            show_progress_bar=True,
        )

        print(f"Best F2-score: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")

        best_params = study.best_params
        best_params["random_state"] = config["model"]["random_state"]
        best_params["verbose"] = -1

    else:
        best_params = {
            "random_state": config["model"]["random_state"],
            "scale_pos_weight": config["model"]["class_weight_multiplier"],
        }

    best_model = LGBMClassifier(**best_params)
    best_model.fit(X_train, y_train)

    y_proba_val = best_model.predict_proba(X_val)[:, 1]
    optimal_threshold = find_optimal_threshold(
        y_val, y_proba_val, config["optimization"]["min_recall_constraint"]
    )

    print(f"Optimal threshold: {optimal_threshold:.4f}")

    return best_model, best_params, optimal_threshold


def evaluate_model(model, X, y, threshold, prefix=""):
    """Evaluate model with custom threshold.

    Args:
        model: Trained model
        X: Features
        y: True labels
        threshold: Decision threshold
        prefix: Metric prefix for logging

    Returns:
        Dictionary of metrics
    """
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        f"{prefix}recall": recall_score(y, y_pred),
        f"{prefix}precision": precision_score(y, y_pred, zero_division=0),
        f"{prefix}f2_score": fbeta_score(y, y_pred, beta=2.0),
        f"{prefix}auc_roc": roc_auc_score(y, y_proba),
        f"{prefix}accuracy": accuracy_score(y, y_pred),
        f"{prefix}false_negatives": ((y == 1) & (y_pred == 0)).sum(),
        f"{prefix}false_positives": ((y == 0) & (y_pred == 1)).sum(),
    }

    return metrics


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train diabetes prediction model")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/train_config.yaml"), help="Config file path"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/experiments"))
    args = parser.parse_args()

    config = load_config(args.config)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    data_path = Path(config["data"]["raw_path"])
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(data_path, config)

    preprocessor = create_preprocessing_pipeline(
        zero_impute_columns=config["preprocessing"]["handle_zeros_as_missing"]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    with mlflow.start_run():
        mlflow.log_params(config["optimization"])
        mlflow.log_params(config["data"])

        model, best_params, threshold = train_model(
            config, X_train_processed, y_train, X_val_processed, y_val, preprocessor
        )

        mlflow.log_params(best_params)
        mlflow.log_param("optimal_threshold", threshold)

        train_metrics = evaluate_model(model, X_train_processed, y_train, threshold, "train_")
        val_metrics = evaluate_model(model, X_val_processed, y_val, threshold, "val_")
        test_metrics = evaluate_model(model, X_test_processed, y_test, threshold, "test_")

        all_metrics = {**train_metrics, **val_metrics, **test_metrics}
        mlflow.log_metrics(all_metrics)

        print("\nTest Set Performance:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = args.output_dir / f"model_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)

        model_artifacts = {
            "model": model,
            "preprocessor": preprocessor,
            "threshold": threshold,
            "config": config,
            "metrics": test_metrics,
            "best_params": best_params,
        }

        joblib.dump(model_artifacts, model_dir / "model_artifacts.pkl")
        mlflow.log_artifact(str(model_dir / "model_artifacts.pkl"))

        metadata = {
            "version": timestamp,
            "algorithm": "LightGBM",
            "optimization_target": "recall",
            "optimal_threshold": float(threshold),
            "target_recall": config["optimization"]["min_recall_constraint"],
            "test_metrics": {k: float(v) if isinstance(v, (np.number, float)) else int(v) for k, v in test_metrics.items()},
            "training_date": datetime.now().isoformat(),
            "data_path": str(data_path),
            "best_params": best_params,
        }

        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        mlflow.log_artifact(str(model_dir / "metadata.json"))

        print(f"\nModel saved to: {model_dir}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
