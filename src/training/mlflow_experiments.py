import mlflow
import mlflow.sklearn
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

PROJECT_ROOT = Path(__file__).parent.parent.parent

def setup_mlflow_windows():
    """Настройка MLflow для Windows"""
    # Вариант 1: Используем SQLite
    db_path = PROJECT_ROOT / "mlflow.db"
    tracking_uri = f"sqlite:///{db_path}"
    
    # Вариант 2: Или просто файловая система с правильным форматом
    # tracking_uri = str(PROJECT_ROOT / "mlruns")
    
    print(f"MLflow URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    
    return tracking_uri

def run_mlflow_experiment():
    """Запустить один эксперимент с MLflow"""
    
    # Настраиваем MLflow
    setup_mlflow_windows()
    
    # Устанавливаем эксперимент
    mlflow.set_experiment("california_housing_mlflow")
    
    # Загружаем данные
    df = pd.read_csv(PROJECT_ROOT / "data/processed/housing_processed_v2.csv")
    
    # Подготавливаем данные
    X = df.drop(columns=['median_house_value', 'median_house_value_log', 'ocean_proximity'], errors='ignore')
    y = df['median_house_value_log']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    with mlflow.start_run(run_name="xgboost_final"):
        # Параметры модели
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }
        
        # Логируем параметры
        mlflow.log_params(params)
        
        # Создаем и обучаем модель
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # Предсказания
        y_pred = model.predict(X_test)
        
        # Метрики
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Логируем метрики
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        
        # Логируем модель
        mlflow.sklearn.log_model(model, "model")
        
        print(f"MSE: {mse:.4f}")
        print(f"R²: {r2:.4f}")
        
        return model

if __name__ == "__main__":
    run_mlflow_experiment()