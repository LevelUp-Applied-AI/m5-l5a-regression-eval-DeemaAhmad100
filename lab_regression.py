"""
Module 5 Week A — Lab: Regression & Evaluation
"""

import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             mean_absolute_error, r2_score, precision_recall_curve)


def load_data():
    """Load the telecom churn dataset and perform basic EDA."""
    df = pd.read_csv("data/telecom_churn.csv")
    
    print("=== Dataset Shape ===")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print(f"Shape: {df.shape}")
    
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    print(f"Total Missing Values: {df.isnull().sum().sum()}")
    
    print("\n=== Churned Distribution ===")
    print(df['churned'].value_counts())
    print(f"Churn Rate: {df['churned'].mean() * 100:.2f}%")
    
    print("\n=== First 5 rows ===")
    print(df.head())
    
    print("\n=== Data Info ===")
    df.info()
    
    return df


def split_data(df, target_col, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    stratify_param = y if y.nunique() == 2 else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    print(f"\n=== Split Summary for target: {target_col} ===")
    print(f"Train set: {X_train.shape[0]} rows")
    print(f"Test set : {X_test.shape[0]} rows")
    
    if y.nunique() == 2:
        print(f"Train {target_col} rate: {y_train.mean():.4f}")
        print(f"Test {target_col} rate : {y_test.mean():.4f}")
    
    return X_train, X_test, y_train, y_test


def build_logistic_pipeline():
    """Build a Pipeline with StandardScaler and LogisticRegression."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight="balanced"
        ))
    ])
    return pipeline


def build_ridge_pipeline():
    """Build a Pipeline with StandardScaler and Ridge regression."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=1.0, random_state=42))
    ])
    return pipeline


def evaluate_classifier(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return classification metrics."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred))
    }
    return metrics


def evaluate_regressor(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return regression metrics."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred))
    }
    return metrics


def threshold_tuning(pipeline, X_test, y_test):
    """Tune the decision threshold for churn prediction."""
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    results = []

    print("\n" + "="*60)
    print("CHALLENGE: Threshold Tuning for Churn Prediction")
    print("="*60)
    print(f"{'Threshold':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 55)

    for thresh in thresholds:
        y_pred_thresh = (y_proba >= thresh).astype(int)
        
        acc = accuracy_score(y_test, y_pred_thresh)
        prec = precision_score(y_test, y_pred_thresh, zero_division=0)
        rec = recall_score(y_test, y_pred_thresh, zero_division=0)
        f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
        
        results.append((thresh, acc, prec, rec, f1))
        
        print(f"{thresh:<10} {acc:.4f}     {prec:.4f}      {rec:.4f}      {f1:.4f}")

    # Plot
    precisions, recalls, threshs = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(10, 6))
    plt.plot(threshs, precisions[:-1], label='Precision', linewidth=2)
    plt.plot(threshs, recalls[:-1], label='Recall', linewidth=2)
    plt.axvline(x=0.5, color='gray', linestyle='--', label='Default (0.5)')
    plt.xlabel('Classification Threshold')
    plt.ylabel('Score')
    plt.title('Precision & Recall vs Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()

    best_idx = np.argmax([f1 for _, _, _, _, f1 in results])
    best_thresh = results[best_idx][0]
    best_f1 = results[best_idx][4]
    
    print(f"\nBest Threshold based on F1-score: {best_thresh} (F1 = {best_f1:.4f})")
    return results, best_thresh


def run_cv_on_pipeline(pipeline, X_train, y_train, n_folds=5, random_state=42):
    """Run 5-fold stratified cross-validation."""
    cv_splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv_splitter, scoring="accuracy")
    
    print("\n=== 5-Fold Cross Validation Results ===")
    for i, score in enumerate(scores, 1):
        print(f"Fold {i}: {score:.4f}")
    
    print(f"\nMean Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


# ====================== TIER 2: Config-Driven Sweep ======================
def run_config_sweep(config_path="models_config.yaml", X_train=None, y_train=None, n_folds=5):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("\n" + "="*70)
    print("TIER 2 CHALLENGE: Config-Driven Model Sweep")
    print("="*70)
    print(f"{'Model Name':<30} {'Mean Score':<12} {'Std':<10} {'Type'}")
    print("-" * 70)

    cv_splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for model_config in config['models']:
        name = model_config['name']
        model_type = model_config['type']
        params = model_config.get('params', {})

        if model_type == "LogisticRegression":
            model = LogisticRegression(**params)
            scoring = "accuracy"
        elif model_type == "Ridge":
            model = Ridge(**params)
            scoring = "r2"
        elif model_type == "Lasso":
            model = Lasso(**params)
            scoring = "r2"
        else:
            continue

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        scores = cross_val_score(pipeline, X_train, y_train, cv=cv_splitter, scoring=scoring)
        mean_score = scores.mean()
        std_score = scores.std()

        print(f"{name:<30} {mean_score:.4f}       {std_score:.4f}     {model_type}")


# ====================== TIER 3: From Scratch ======================
class LogisticRegressionFromScratch:
    def __init__(self, learning_rate=0.01, n_iterations=2000, regularization=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iterations):
            linear_pred = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_pred)

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            dw += (self.regularization / n_samples) * self.weights

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_pred)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ====================== MAIN ======================
if __name__ == "__main__":
    df = load_data()

    # Classification Task
    print("\n" + "="*60)
    print("CLASSIFICATION TASK - Predicting Churn")
    print("="*60)

    numeric_features = ["tenure", "monthly_charges", "total_charges",
                        "num_support_calls", "senior_citizen",
                        "has_partner", "has_dependents"]

    df_cls = df[numeric_features + ["churned"]].dropna()
    X_train, X_test, y_train, y_test = split_data(df_cls, target_col="churned")

    log_pipe = build_logistic_pipeline()
    metrics = evaluate_classifier(log_pipe, X_train, X_test, y_train, y_test)
    print(f"\nLogistic Regression Metrics:\n{metrics}")

    # Threshold Tuning Challenge
    results, best_threshold = threshold_tuning(log_pipe, X_test, y_test)

    # Cross-Validation
    run_cv_on_pipeline(log_pipe, X_train, y_train)

    # Regression Task
    print("\n" + "="*60)
    print("REGRESSION TASK - Predicting Monthly Charges")
    print("="*60)

    reg_features = ["tenure", "total_charges", "num_support_calls",
                    "senior_citizen", "has_partner", "has_dependents"]

    df_reg = df[reg_features + ["monthly_charges"]].dropna()
    X_tr, X_te, y_tr, y_te = split_data(df_reg, target_col="monthly_charges")

    ridge_pipe = build_ridge_pipeline()
    reg_metrics = evaluate_regressor(ridge_pipe, X_tr, X_te, y_tr, y_te)
    print(f"\nRidge Regression Metrics:\n{reg_metrics}")

    print("\nLab completed successfully!")