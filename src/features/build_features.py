from typing import Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def build_features(df: pd.DataFrame) -> Tuple:
    """
    Prepare dataset for model training.

    Returns:
        X (features), y (target), preprocessor (for reuse)
    """

    # Separate features and target
    X = df.drop(columns=["Exited"])
    y = df["Exited"]

    # Define feature types
    numerical_features = [
        "CreditScore", "Age", "Tenure", "Balance",
        "NumOfProducts", "EstimatedSalary"
    ]

    categorical_features = [
        "Geography", "Gender"
    ]

    # Create transformers
    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    # Combine transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Apply transformations
    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor