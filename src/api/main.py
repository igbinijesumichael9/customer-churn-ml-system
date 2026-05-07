from fastapi import FastAPI
from pydantic import BaseModel

from src.models.predict import ChurnPredictor


# Initialize App
app = FastAPI(
    title="Churn Prediction API",
    version="1.0.0"
)


# Load Predictor
predictor = ChurnPredictor(
    model_path="artifacts/model.joblib",
    preprocessor_path="artifacts/preprocessor.joblib"
)


# Request Schema
class CustomerData(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float


# Health Check
@app.get("/")
def health_check():
    return {"status": "API is running"}


# Prediction Endpoint
@app.post("/predict")
def predict(customer: CustomerData):

    prediction = predictor.predict(customer.dict())

    return {
        "prediction": prediction
    }