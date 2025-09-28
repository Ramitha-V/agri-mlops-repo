from fastapi.testclient import TestClient
from src.main import app # Import your FastAPI app
import os
import pytest

# Create a TestClient instance
client = TestClient(app)

# A simple test for the root endpoint
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Crop Yield Prediction API! Visit /docs for more info."}

@pytest.mark.skipif(not os.path.exists('models/crop_yield_pipeline.pkl'), reason="Model file not found")
def test_predict_yield():
    # Define a valid sample payload
    sample_payload = {
      "Crop": "Rice",
      "Crop_Year": 2024,
      "Season": "Kharif",
      "State": "Punjab",
      "Area": 50000,
      "Annual_Rainfall": 850.5,
      "Fertilizer": 12000,
      "Pesticide": 850
    }

    response = client.post("/predict", json=sample_payload)

    # Check that the request was successful
    assert response.status_code == 200

    # Check that the response contains the prediction key
    response_json = response.json()
    assert "predicted_yield" in response_json

    # Check that the prediction is a number (float)
    assert isinstance(response_json["predicted_yield"], float)