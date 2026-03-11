from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# load model and preprocessing
model = joblib.load("heart_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# define input schema
class Patient(BaseModel):
    Age: int
    Sex: int
    Chest_pain_type: int
    BP: int
    Cholesterol: int
    FBS_over_120: int
    EKG_results: int
    Max_HR: int
    Exercise_angina: int
    ST_depression: float
    Slope_of_ST: int
    Number_of_vessels_fluro: int
    Thallium: int
    
@app.get("/")
def home():
    return {"message": "Heart disease prediction API"}

@app.post("/predict")
def predict(data: Patient):
    df = pd.DataFrame([data.model_dump()])
    df.rename(columns={
        "Chest_pain_type": "Chest pain type",
        "FBS_over_120": "FBS over 120",
        "EKG_results": "EKG results",
        "Max_HR": "Max HR",
        "Exercise_angina": "Exercise angina",
        "ST_depression": "ST depression",
        "Slope_of_ST": "Slope of ST",
        "Number_of_vessels_fluro": "Number of vessels fluro"
    }, inplace=True)
    X = preprocessor.transform(df)
    prediction = model.predict(X)[0]
    return {"prediction": int(prediction)}