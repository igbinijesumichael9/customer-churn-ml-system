from typing import List
import pandas as pd


# Schema Definition (Bank Churn)
FEATURE_COLUMNS: List[str] = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
]

TARGET_COLUMN: str = "Exited"

REQUIRED_COLUMNS: List[str] = FEATURE_COLUMNS + [TARGET_COLUMN]


# Validation Functions
def validate_columns(df: pd.DataFrame) -> None:
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def validate_no_missing_values(df: pd.DataFrame) -> None:
    if df.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains missing values.")


def validate_data_types(df: pd.DataFrame) -> None:
    try:
        numeric_cols = [
            "CreditScore", "Age", "Tenure", "Balance",
            "NumOfProducts", "EstimatedSalary"
        ]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])

    except Exception:
        raise ValueError("Numeric columns contain invalid values.")


def validate_data(df: pd.DataFrame) -> None:
    validate_columns(df)
    validate_no_missing_values(df)
    validate_data_types(df)