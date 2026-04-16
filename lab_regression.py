"""
Module 5 Week A — Lab: Regression & Evaluation
"""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             mean_absolute_error, r2_score)


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
    """Split data into train and test sets.
    Uses stratification only for classification targets (binary).
    """
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


def run_cv_on_pipeline(pipeline, X_train, y_train, n_folds=5, random_state=42):
    """Run 5-fold stratified cross-validation."""
    cv_splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv_splitter, scoring="accuracy")
    
    print("\n=== 5-Fold Cross Validation Results ===")
    for i, score in enumerate(scores, 1):
        print(f"Fold {i}: {score:.4f}")
    
    print(f"\nMean Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


# ====================== MAIN ======================
if __name__ == "__main__":
    df = load_data()

    # ====================== CLASSIFICATION ======================
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

    # Task 6: Cross-Validation
    run_cv_on_pipeline(log_pipe, X_train, y_train)

    # ====================== REGRESSION ======================
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