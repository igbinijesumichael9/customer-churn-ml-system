import joblib
import pandas as pd


class ChurnPredictor:
    """
    Inference pipeline for churn prediction.
    """

    def __init__(self, model_path: str, preprocessor_path: str):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def predict(self, input_data: dict):
        """
        Predict churn for a single input.

        Args:
            input_data (dict): Raw input features

        Returns:
            int: Predicted class (0 or 1)
        """

        # Convert input to DataFrame
        df = pd.DataFrame([input_data])

        # Apply preprocessing
        X_processed = self.preprocessor.transform(df)

        # Predict
        prediction = self.model.predict(X_processed)

        return int(prediction[0])