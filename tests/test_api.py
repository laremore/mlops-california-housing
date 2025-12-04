import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from src.inference.app import app

client = TestClient(app)

def test_health_endpoint():
    """Тест эндпоинта /health"""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

def test_predict_endpoint():
    """Тест эндпоинта /predict - не падаем если модель не загружена"""
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
    
    # Принимаем оба варианта: 200 (модель загружена) или 503 (модель не загружена)
    assert response.status_code in [200, 503], f"Unexpected status code: {response.status_code}"
    
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "model_version" in data

def test_swagger_docs():
    """Тест документации Swagger"""
    response = client.get("/docs")
    assert response.status_code == 200
    
def test_root_endpoint():
    """Тест корневого эндпоинта"""
    response = client.get("/")
    assert response.status_code == 200
    
def test_redoc_endpoint():
    """Тест ReDoc документации"""
    response = client.get("/redoc")
    assert response.status_code == 200