"""Unit tests for metric calculation, confusion matrix, and threshold evaluation."""
import numpy as np
import pytest

from src.evaluate import calculate_metrics


def test_calculate_metrics_perfect_classifier():
    """Test metrics for 100% correct classification."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = calculate_metrics(y_true, y_pred, y_proba)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["false_negatives"] == 0
    assert metrics["false_positives"] == 0


def test_calculate_metrics_with_errors():
    """Test metrics with specific known false positives and false negatives."""
    # 2 true negatives (y_true=0, y_pred=0)
    # 1 false positive (y_true=0, y_pred=1)
    # 1 false negative (y_true=1, y_pred=0)
    # 2 true positives (y_true=1, y_pred=1)
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.7, 0.4, 0.8, 0.9])

    metrics = calculate_metrics(y_true, y_pred, y_proba)
    assert metrics["confusion_matrix"]["true_negatives"] == 2
    assert metrics["confusion_matrix"]["false_positives"] == 1
    assert metrics["confusion_matrix"]["false_negatives"] == 1
    assert metrics["confusion_matrix"]["true_positives"] == 2

    # Recall = TP / (TP + FN) = 2 / 3 = 0.6667
    assert metrics["recall"] == round(2 / 3, 4)
    # Precision = TP / (TP + FP) = 2 / 3 = 0.6667
    assert metrics["precision"] == round(2 / 3, 4)
    # Specificity = TN / (TN + FP) = 2 / 3 = 0.6667
    assert metrics["specificity"] == round(2 / 3, 4)
