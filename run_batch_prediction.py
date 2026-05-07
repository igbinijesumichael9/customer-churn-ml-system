from src.models.predict import ChurnPredictor
import pandas as pd


def main():
    file_path = "data/raw/test_data.csv"

    # Load data
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    # Initialize predictor
    predictor = ChurnPredictor(
        model_path="artifacts/model.joblib",
        preprocessor_path="artifacts/preprocessor.joblib"
    )

    # Make predictions
    predictions = []

    for _, row in df.iterrows():
        pred = predictor.predict(row.to_dict())
        predictions.append(pred)

    # Add predictions to dataframe
    df["Prediction"] = predictions

    # Save results
    output_path = "data/processed/predictions.csv"
    df.to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")
    print(df.head())


if __name__ == "__main__":
    main()