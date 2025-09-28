from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize the app
app = FastAPI(title="Crop Yield Prediction API", version="2.0")

# Load the trained model pipeline
pipeline = joblib.load('models/crop_yield_pipeline.pkl')

# Define the input data model to match the training script
class CropFeatures(BaseModel):
    Crop: str
    Crop_Year: int
    Season: str
    State: str
    Area: float
    Annual_Rainfall: float
    Fertilizer: float
    Pesticide: float
    
    class Config:
        schema_extra = {
            "example": {
                "Crop": "Rice",
                "Crop_Year": 2024,
                "Season": "Kharif",
                "State": "Maharashtra",
                "Area": 100.0,
                "Annual_Rainfall": 1200.5,
                "Fertilizer": 5000.0,
                "Pesticide": 150.0
            }
        }

@app.get("/")
def read_root():
    """A simple endpoint to check if the API is live."""
    return {"message": "Welcome to the Crop Yield Prediction API! Visit /docs for more info."}

@app.post("/predict")
def predict_yield(features: CropFeatures):
    """Predicts crop yield based on agricultural features."""
    
    df = pd.DataFrame([features.dict()])

    prediction = pipeline.predict(df)[0]
    
    return {"predicted_yield": prediction}