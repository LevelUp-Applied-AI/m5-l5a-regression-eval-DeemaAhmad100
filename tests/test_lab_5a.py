"""Autograder tests for Lab 5A — Regression & Evaluation."""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from starter.lab_regression import (load_data, split_data, build_logistic_pipeline,
                            build_ridge_pipeline, evaluate_classifier,
                            evaluate_regressor, run_cross_validation)


@pytest.fixture
def df():
    data = load_data(os.path.join(os.path.dirname(__file__), "..", "data", "telecom_churn.csv"))
    assert data is not None, "load_data returned None"
    return data


@pytest.fixture
def cls_split(df):
    numeric = ["tenure", "monthly_charges", "total_charges",
               "num_support_calls", "senior_citizen", "has_partner", "has_dependents"]
    df_cls = df[numeric + ["churned"]].dropna()
    result = split_data(df_cls, "churned")
    assert result is not None, "split_data returned None"
    return result


def test_data_loaded(df):
    assert df.shape[0] > 1000, f"Expected >1000 rows, got {df.shape[0]}"
    assert "churned" in df.columns, "Missing 'churned' column"


def test_train_test_split_sizes(cls_split):
    X_train, X_test, y_train, y_test = cls_split
    total = len(X_train) + len(X_test)
    test_ratio = len(X_test) / total
    assert 0.18 <= test_ratio <= 0.22, f"Test ratio {test_ratio:.2f} not ~0.20"


def test_stratification(cls_split):
    X_train, X_test, y_train, y_test = cls_split
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    assert abs(train_ratio - test_ratio) < 0.05, "Class ratios differ too much between train/test"


def test_logistic_pipeline_exists():
    pipe = build_logistic_pipeline()
    assert pipe is not None, "build_logistic_pipeline returned None"
    assert hasattr(pipe, "fit"), "Pipeline must have fit method"
    assert len(pipe.steps) >= 2, "Pipeline should have at least 2 steps"


def test_logistic_predictions(cls_split):
    X_train, X_test, y_train, y_test = cls_split
    pipe = build_logistic_pipeline()
    assert pipe is not None
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    assert len(preds) == len(X_test), "Prediction length mismatch"
    assert set(preds).issubset({0, 1}), "Predictions should be 0 or 1"


def test_classification_metrics_computed(cls_split):
    X_train, X_test, y_train, y_test = cls_split
    pipe = build_logistic_pipeline()
    assert pipe is not None
    metrics = evaluate_classifier(pipe, X_train, X_test, y_train, y_test)
    assert metrics is not None, "evaluate_classifier returned None"
    for key in ["accuracy", "precision", "recall", "f1"]:
        assert key in metrics, f"Missing key: {key}"
        assert metrics[key] > 0, f"{key} should be > 0"


def test_ridge_pipeline():
    pipe = build_ridge_pipeline()
    assert pipe is not None, "build_ridge_pipeline returned None"
    assert hasattr(pipe, "fit"), "Pipeline must have fit method"


def test_regression_metrics(df):
    cols = ["tenure", "total_charges", "num_support_calls",
            "senior_citizen", "has_partner", "has_dependents", "monthly_charges"]
    df_reg = df[cols].dropna()
    result = split_data(df_reg, "monthly_charges")
    assert result is not None
    X_tr, X_te, y_tr, y_te = result
    pipe = build_ridge_pipeline()
    assert pipe is not None
    metrics = evaluate_regressor(pipe, X_tr, X_te, y_tr, y_te)
    assert metrics is not None, "evaluate_regressor returned None"
    assert "mae" in metrics, "Missing 'mae'"
    assert "r2" in metrics, "Missing 'r2'"
    assert metrics["r2"] > 0, "R² should be > 0"


def test_cross_validation_runs(cls_split):
    X_train, X_test, y_train, y_test = cls_split
    pipe = build_logistic_pipeline()
    assert pipe is not None
    scores = run_cross_validation(pipe, X_train, y_train)
    assert scores is not None, "run_cross_validation returned None"
    assert len(scores) == 5, f"Expected 5 scores, got {len(scores)}"
    assert scores.mean() > 0.5, f"Mean CV score should be > 0.5"