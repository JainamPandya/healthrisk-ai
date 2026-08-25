import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(y_true, y_pred, y_probability):
    """
    Calculate classification metrics for a binary risk model.
    """

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_probability),
        "pr_auc": average_precision_score(y_true, y_probability),
    }

    return results


def print_evaluation(y_true, y_pred, y_probability):
    """
    Print detailed model evaluation.
    """

    results = evaluate_model(
        y_true,
        y_pred,
        y_probability,
    )

    print("\n===== MODEL EVALUATION =====")

    for metric, value in results.items():
        print(f"{metric.upper():10}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    return results