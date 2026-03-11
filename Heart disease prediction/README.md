# Heart Disease Prediction

This project builds a machine learning model to predict heart disease risk and exposes it through a FastAPI REST API.

## Features
- Random Forest / XGBoost classifier
- ~88% cross-validation accuracy
- Preprocessing pipeline using ColumnTransformer
- FastAPI inference endpoint

## API

Run server:

python -m uvicorn app:app --reload

Open docs:

http://127.0.0.1:8000/docs

## Example Request

POST /predict

{
  "Age": 52,
  "Sex": 1,
  "Chest_pain_type": 4,
  ...
}