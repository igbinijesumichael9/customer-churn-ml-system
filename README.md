# Customer Churn Prediction ML System

## Overview

This project is a production-style machine learning system for predicting customer churn using structured banking customer data.

The system was built using modular machine learning engineering principles and includes:

- Data ingestion
- Data validation
- Feature engineering
- Model training
- Overfitting detection
- Batch inference
- REST API deployment
- Docker containerization

The goal of the system is to predict whether a customer is likely to churn based on demographic and financial attributes.

---

# Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Docker
- Joblib
- Uvicorn

---

# Project Structure

```plaintext
churn_ml_system/
│
├── artifacts/
│   ├── model.joblib
│   ├── preprocessor.joblib
│
├── configs/
│
├── data/
│   ├── raw/
│   │   ├── ChurnData.csv
│   │   ├── test_data.csv
│   │
│   ├── processed/
│   │   ├── predictions.csv
│
├── src/
│   ├── api/
│   │   ├── main.py
│   │
│   ├── data/
│   │   ├── ingestion.py
│   │   ├── validation.py
│   │
│   ├── features/
│   │   ├── build_features.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── predict.py
│   │   ├── evaluate.py
│   │
│   ├── utils/
│
├── tests/
│
├── run_ingestion.py
├── run_prediction.py
├── run_batch_prediction.py
├── run_evaluation.py
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── README.md
```

---

# ML Pipeline Architecture

The system follows a modular ML engineering pipeline:

1. Data Ingestion
2. Data Validation
3. Feature Engineering
4. Model Training
5. Artifact Persistence
6. Batch Inference
7. REST API Deployment
8. Docker Containerization

---

# Model Training

The model was trained using:

- Logistic Regression
- Train/Test Split Validation
- Accuracy Evaluation
- Overfitting Detection

Artifacts generated during training:

- `model.joblib`
- `preprocessor.joblib`

---

# API Deployment

The project exposes a REST API using FastAPI.

## Health Endpoint

```http
GET /
```

Response:

```json
{
  "status": "API is running"
}
```

---

## Prediction Endpoint

```http
POST /predict
```

Request Body:

```json
{
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
```

Response:

```json
{
  "prediction": 0
}
```

---

# Running the Project Locally

## 1. Clone Repository

```bash
git clone <repository-url>
cd churn_ml_system
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Run API

```bash
python -m uvicorn src.api.main:app --reload
```

---

## 4. Open Swagger Documentation

```plaintext
http://127.0.0.1:8000/docs
```

---

# Docker Deployment

## Build Docker Image

```bash
docker build -t churn-api .
```

---

## Run Docker Container

```bash
docker run -p 8000:8000 churn-api
```

---

# Batch Prediction

Run predictions on unseen external datasets:

```bash
python run_batch_prediction.py
```

Predictions will be saved to:

```plaintext
data/processed/predictions.csv
```

---

# Future Improvements

Potential improvements for the system include:

- MLflow experiment tracking
- CI/CD pipelines
- Cloud deployment
- Model monitoring
- Logging and observability
- Automated retraining
- Kubernetes deployment
- Feature store integration

---

# Author

Ajali Augustine

Machine Learning Engineer | Data Scientist | Data Analyst
