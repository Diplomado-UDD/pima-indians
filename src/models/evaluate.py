"""Model evaluation script with recall-focused analysis."""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def load_model_artifacts(model_path: Path):
    """Load trained model artifacts.

    Args:
        model_path: Path to model artifacts file

    Returns:
        Dictionary containing model, preprocessor, threshold, etc.
    """
    return joblib.load(model_path)


def generate_confusion_matrix_plot(y_true, y_pred, output_path):
    """Generate confusion matrix visualization.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save plot
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)

    ax.set_title("Confusion Matrix (False Negatives Highlighted)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticklabels(["Negative", "Positive"])

    fn_count = cm[1, 0]
    ax.text(
        0.5,
        1.5,
        f"False Negatives: {fn_count}",
        ha="center",
        va="center",
        fontsize=12,
        color="red",
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Confusion matrix saved to: {output_path}")


def generate_precision_recall_curve(y_true, y_proba, threshold, output_path):
    """Generate precision-recall curve with operating point.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        threshold: Operating threshold
        output_path: Path to save plot
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    y_pred = (y_proba >= threshold).astype(int)
    from sklearn.metrics import precision_score, recall_score

    operating_precision = precision_score(y_true, y_pred)
    operating_recall = recall_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(recall, precision, linewidth=2, label=f"PR Curve (AUC = {pr_auc:.3f})")
    ax.scatter(
        [operating_recall],
        [operating_precision],
        color="red",
        s=200,
        marker="*",
        label=f"Operating Point (threshold={threshold:.3f})",
        zorder=5,
    )

    ax.axhline(y=0.85, color="orange", linestyle="--", alpha=0.7, label="Target Recall = 0.85")

    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Precision-recall curve saved to: {output_path}")


def generate_roc_curve(y_true, y_proba, output_path):
    """Generate ROC curve.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        output_path: Path to save plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"ROC curve saved to: {output_path}")


def generate_threshold_analysis(y_true, y_proba, output_path):
    """Generate threshold analysis table.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        output_path: Path to save CSV
    """
    from sklearn.metrics import fbeta_score, precision_score, recall_score

    thresholds = np.linspace(0.1, 0.9, 17)
    results = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        f2 = fbeta_score(y_true, y_pred, beta=2.0)
        fn_count = ((y_true == 1) & (y_pred == 0)).sum()

        results.append(
            {
                "threshold": threshold,
                "recall": recall,
                "precision": precision,
                "f2_score": f2,
                "fn_per_1000": int((fn_count / len(y_true)) * 1000),
            }
        )

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)

    print(f"Threshold analysis saved to: {output_path}")
    print("\nThreshold Analysis:")
    print(df_results.to_string(index=False))


def generate_evaluation_report(model_path: Path, test_data_path: Path, output_dir: Path):
    """Generate comprehensive evaluation report.

    Args:
        model_path: Path to model artifacts
        test_data_path: Path to test data
        output_dir: Directory to save evaluation outputs
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = load_model_artifacts(model_path)
    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    threshold = artifacts["threshold"]

    df_test = pd.read_csv(test_data_path)
    X_test = df_test.drop("Outcome", axis=1)
    y_test = df_test["Outcome"]

    X_test_processed = preprocessor.transform(X_test)
    y_proba = model.predict_proba(X_test_processed)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    generate_confusion_matrix_plot(y_test, y_pred, output_dir / "confusion_matrix.png")
    generate_precision_recall_curve(y_test, y_proba, threshold, output_dir / "pr_curve.png")
    generate_roc_curve(y_test, y_proba, output_dir / "roc_curve.png")
    generate_threshold_analysis(y_test, y_proba, output_dir / "threshold_analysis.csv")

    summary = {
        "model_path": str(model_path),
        "test_data_path": str(test_data_path),
        "threshold": float(threshold),
        "test_samples": len(y_test),
        "positive_rate": float(y_test.mean()),
        "evaluation_date": pd.Timestamp.now().isoformat(),
    }

    with open(output_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nEvaluation complete. Results saved to: {output_dir}")


def main():
    """CLI entry point for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate diabetes prediction model")
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to model artifacts file (model_artifacts.pkl)",
    )
    parser.add_argument(
        "--test-data", type=Path, default=Path("data/raw/diabetes.csv"), help="Path to test data"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/model_evaluation"), help="Output directory"
    )
    args = parser.parse_args()

    generate_evaluation_report(args.model_path, args.test_data, args.output_dir)


if __name__ == "__main__":
    main()
