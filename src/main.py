from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Crop Yield Prediction API")
model = joblib.load('models/crop_yield_predictor.pkl')

class CropFeatures(BaseModel):
    area: float
    production: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Crop Yield Prediction API!"}

@app.post("/predict")
def predict_yield(features: CropFeatures):
    df = pd.DataFrame([[features.area, features.production]], columns=['Area_in_hectares', 'Production_in_tonnes'])
    prediction = model.predict(df)[0]
    return {"predicted_yield_tonnes_per_hectare": prediction}