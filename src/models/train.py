import os
from typing import Tuple
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def train_model(X, y, preprocessor) -> Tuple:
    """
    Train a machine learning model, evaluate performance,
    detect overfitting, and save artifacts.

    Args:
        X: Feature matrix
        y: Target vector
        preprocessor: Fitted preprocessing pipeline

    Returns:
        model, test_accuracy
    """

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model Training
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluation

    # Train predictions
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    # Test predictions
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print("\n========== MODEL PERFORMANCE ==========")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy:  {test_accuracy:.4f}")

    print("\nClassification Report (Test):\n")
    print(classification_report(y_test, y_test_pred))

    # Overfitting Detection
    gap = train_accuracy - test_accuracy

    print("\n========== OVERFITTING CHECK ==========")
    print(f"Overfitting Gap: {gap:.4f}")

    if gap > 0.05:
        print("⚠️ Warning: Model may be overfitting.")
    else:
        print("✅ Model generalization looks good.")

    # Save Artifacts
    os.makedirs("artifacts", exist_ok=True)

    joblib.dump(model, "artifacts/model.joblib")
    joblib.dump(preprocessor, "artifacts/preprocessor.joblib")

    print("\nArtifacts saved successfully.")
    print(" - artifacts/model.joblib")
    print(" - artifacts/preprocessor.joblib")

    return model, test_accuracy