from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


def load_artifacts(model_path: str, preprocessor_path: str):
    """
    Load trained model and preprocessor.
    """
    model_file = Path(model_path)
    preprocessor_file = Path(preprocessor_path)

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not preprocessor_file.exists():
        raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")

    model = joblib.load(model_file)
    preprocessor = joblib.load(preprocessor_file)

    return model, preprocessor


def load_and_prepare_data(file_path: str) -> tuple:
    """
    Load external dataset and prepare features/target.
    """
    data_file = Path(file_path)

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(data_file)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Separate features and target
    X = df.drop(columns=["Exited"])
    y = df["Exited"]

    return X, y


def evaluate_on_external_data(
    data_path: str,
    model_path: str = "artifacts/model.joblib",
    preprocessor_path: str = "artifacts/preprocessor.joblib"
) -> None:
    """
    Evaluate trained model on external dataset.
    """

    print("\n========== EXTERNAL EVALUATION ==========")

    # Load artifacts
    model, preprocessor = load_artifacts(model_path, preprocessor_path)

    # Load data
    X, y = load_and_prepare_data(data_path)

    # Transform features
    X_processed = preprocessor.transform(X)

    # Predict
    y_pred = model.predict(X_processed)

    # Metrics
    accuracy = accuracy_score(y, y_pred)

    print(f"External Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y, y_pred))