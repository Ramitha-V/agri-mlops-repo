from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware # <--- Import this

app = FastAPI(title="Crop Yield Prediction API", version="2.0")

# --- Add this CORS middleware block ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# ------------------------------------

pipeline = joblib.load('models/crop_yield_pipeline.pkl')

# ... the rest of your main.py code remains the same ...
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
    return {"message": "Welcome to the Crop Yield Prediction API!"}

@app.post("/predict")
def predict_yield(features: CropFeatures):
    df = pd.DataFrame([features.dict()])
    prediction = pipeline.predict(df)[0]
    return {"predicted_yield": prediction}