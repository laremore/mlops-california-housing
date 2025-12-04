import joblib
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any
import re

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.is_loaded = False
        self.model_version = "1.0"
        self.feature_names = None  
    def load(self):
        try:
            logger.info(f"Загрузка модели из: {self.model_path}")
            
            if not self.model_path.exists():
                logger.error(f"Файл модели не найден: {self.model_path}")
                return False
            
            self.model = joblib.load(str(self.model_path))
            
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = list(self.model.feature_names_in_)
                logger.info(f"Имена фич из модели: {self.feature_names}")
            elif hasattr(self.model, 'feature_importances_'):
                features_path = self.model_path.parent / "feature_names.joblib"
                if features_path.exists():
                    self.feature_names = joblib.load(features_path)
                    logger.info(f"Имена фич загружены из файла: {self.feature_names}")
            
            if not self._verify_model():
                logger.error("Модель не прошла проверку!")
                return False
            
            self.is_loaded = True
            logger.info(f"Модель загружена успешно: {type(self.model).__name__}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            return False
    
    def _normalize_feature_names(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Нормализация имен фичей для совместимости"""
        normalized = {}
        
        for key, value in features.items():
        
            normalized_key = str(key)
            
            if normalized_key.startswith('ocean_'):
                normalized_key = normalized_key.replace('ocean_', 'ocean__', 1)
            
            normalized_key = re.sub(r'[<>\[\]:=,()\s]', '_', normalized_key)
           
            while '__' in normalized_key:
                normalized_key = normalized_key.replace('__', '_')
            
            normalized[normalized_key] = value
        
        return normalized
    
    def _verify_model(self):
        try:
            test_features = {
                "longitude": -122.23,
                "latitude": 37.88,
                "housing_median_age": 41.0,
                "total_rooms": 880.0,
                "total_bedrooms": 129.0,
                "population": 322.0,
                "households": 126.0,
                "median_income": 8.3252,
                "bedrooms_per_room": 0.1466,
                "population_per_household": 2.5556,
                "households_per_population": 0.3913,
                "ocean__INLAND": 0,
                "ocean__NEAR_BAY": 1,
                "ocean__NEAR_OCEAN": 0,
                "ocean__ISLAND": 0,
                "ocean__1H_OCEAN": 0  
            }
            
            normalized_features = self._normalize_feature_names(test_features)
            
            df = self._prepare_features(normalized_features)
            
            if self.feature_names:
                missing = set(self.feature_names) - set(df.columns)
                if missing:
                    logger.warning(f"Отсутствующие фичи: {missing}")
                   
                    for feature in missing:
                        df[feature] = 0
            
            prediction = self.model.predict(df)[0]
            logger.info(f"Проверка модели: raw_prediction = {prediction}")
            
            if prediction < 20: 
                price = np.expm1(prediction)
                logger.info(f"Модель корректна: log({prediction:.4f}) → ${price:,.2f}")
                return True
            else:
                logger.error(f"Модель возвращает не логарифм: {prediction}")
                return False
                
        except Exception as e:
            logger.error(f"Ошибка проверки модели: {e}")
            return False
    
    def _prepare_features(self, features: Dict[str, Any]) -> pd.DataFrame:
        """Подготовка фичей в правильном порядке"""
        
        normalized_features = self._normalize_feature_names(features)
        
        default_features = [
            'longitude', 'latitude', 'housing_median_age',
            'total_rooms', 'total_bedrooms', 'population',
            'households', 'median_income', 'bedrooms_per_room',
            'population_per_household', 'households_per_population',
            'ocean__INLAND', 'ocean__NEAR_BAY', 
            'ocean__NEAR_OCEAN', 'ocean__ISLAND', 'ocean__1H_OCEAN'
        ]
        
        if self.feature_names:
            expected_features = self.feature_names
            logger.info(f"Используем имена фич из модели: {expected_features}")
        else:
            expected_features = default_features
        
        df = pd.DataFrame([normalized_features])
        
        defaults = {
            'longitude': -122.23,
            'latitude': 37.88,
            'housing_median_age': 41.0,
            'total_rooms': 880.0,
            'total_bedrooms': 129.0,
            'population': 322.0,
            'households': 126.0,
            'median_income': 8.3252,
            'bedrooms_per_room': 0.1466,
            'population_per_household': 2.5556,
            'households_per_population': 0.3913
        }
        
        for feature in expected_features:
            if feature not in df.columns:
                if feature.startswith('ocean__'):
                    df[feature] = 0  
                else:
                    df[feature] = defaults.get(feature, 0.0)
        
        df = df[expected_features]
        
        logger.debug(f"Подготовленные фичи: {df.columns.tolist()}")
        logger.debug(f"Форма данных: {df.shape}")
        
        return df
    
    def predict(self, features: Dict[str, Any]) -> float:
        if not self.is_loaded:
            raise ValueError("Модель не загружена")
        
        try:
            df = self._prepare_features(features)
            
            raw_prediction = self.model.predict(df)[0]
            
            price = np.expm1(raw_prediction)
            
            logger.debug(f"Предсказание: log={raw_prediction:.4f}, price=${price:,.2f}")
            
            price = max(50000, min(price, 1000000))
            
            logger.info(f"Предсказанная цена: ${price:,.2f}")
            return float(price)
            
        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            return 250000.0
    
    def predict_batch(self, features_list: list) -> list:
        if not self.is_loaded:
            raise ValueError("Модель не загружена")
        
        predictions = []
        for features in features_list:
            try:
                prediction = self.predict(features)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Ошибка предсказания для образца: {e}")
                predictions.append(250000.0)
        
        return predictions
    
    def get_info(self) -> Dict[str, Any]:
        info = {
            "is_loaded": self.is_loaded,
            "model_version": self.model_version,
            "model_type": type(self.model).__name__ if self.model else None,
            "model_path": str(self.model_path),
        }
        
        if self.feature_names:
            info["feature_count"] = len(self.feature_names)
            info["feature_names_sample"] = self.feature_names[:5]  # Первые 5 фич
        
        return info