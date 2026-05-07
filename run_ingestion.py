from src.data.ingestion import load_data
from src.data.validation import validate_data
from src.features.build_features import build_features
from src.models.train import train_model


def main():
    file_path = "data/raw/ChurnData.csv"

    # Load
    df = load_data(file_path)
    print("Data loaded successfully.")

    # Validate
    validate_data(df)
    print("Data validation passed.")

    # Feature Engineering
    X, y, preprocessor = build_features(df)
    print("Feature engineering completed.")

    # Train Model
    model, accuracy = train_model(X, y, preprocessor)
    print("Model training completed.")


if __name__ == "__main__":
    main()