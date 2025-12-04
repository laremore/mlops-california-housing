import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from src.inference.app import app

client = TestClient(app)

def test_health_endpoint():

    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

def test_predict_endpoint():

    test_data = {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252
    }
    
    response = client.post("/predict", json=test_data)
    
    assert response.status_code in [200, 503], f"Unexpected status code: {response.status_code}"
    
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "model_version" in data

def test_swagger_docs():
    response = client.get("/docs")
    assert response.status_code == 200
    
def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    
def test_redoc_endpoint():
    response = client.get("/redoc")
    assert response.status_code == 200