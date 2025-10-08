from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Crop Yield Prediction API", version="2.0")

# This is NOT needed when deploying the UI with Azure Static Web Apps,
# but it's good practice to keep for local testing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = joblib.load('models/crop_yield_pipeline.pkl')

class CropFeatures(BaseModel):
    Crop: str
    Crop_Year: int
    Season: str
    State: str
    Area: float
    Annual_Rainfall: float
    Fertilizer: float
    Pesticide: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Crop Yield Prediction API! Visit /docs for more info."}

@app.post("/predict")
def predict_yield(features: CropFeatures):
    df = pd.DataFrame([features.model_dump()])
    prediction = pipeline.predict(df)[0]
    return {"predicted_yield": prediction}