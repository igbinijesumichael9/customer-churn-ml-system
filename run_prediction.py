from src.models.predict import ChurnPredictor


def main():
    predictor = ChurnPredictor(
        model_path="artifacts/model.joblib",
        preprocessor_path="artifacts/preprocessor.joblib"
    )

    sample_input = {
        "CreditScore": 600,
        "Geography": "France",
        "Gender": "Male",
        "Age": 40,
        "Tenure": 5,
        "Balance": 60000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000
    }

    prediction = predictor.predict(sample_input)

    print(f"Prediction (1 = churn, 0 = no churn): {prediction}")


if __name__ == "__main__":
    main()