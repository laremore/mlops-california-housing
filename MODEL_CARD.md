# Model Card: California Housing Price Predictor

## Model Details
- **Model Name**: California Housing Price Predictor
- **Version**: 1.0
- **Type**: Regression (XGBoost Regressor)
- **Author**: Цыпленков Константин, ШАД-311
- **Date**: Декабрь 2025

## Intended Use
- **Primary Use**: Прогнозирование медианной стоимости домов в Калифорнии
- **Target Users**: Студенты, исследователи, разработчики в области ML/MLOps
- **Limitations**: Модель обучена на данных переписи 1990 года, не подходит для прогнозирования текущих цен

## Training Data
- **Dataset**: California Housing Prices (1990 Census)
- **Size**: 20,640 samples (после предобработки)
- **Features**: 11 числовых признаков
- **Target**: `median_house_value` (медианная стоимость дома)
- **Data Split**: 80% train, 20% test
- **Data Version**: v2 (расширенная предобработка)

## Features
| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| longitude | Долгота | float | [-124.35, -114.31] |
| latitude | Широта | float | [32.54, 41.95] |
| housing_median_age | Медианный возраст дома | float | [1, 52] |
| total_rooms | Общее количество комнат | float | [2, 39320] |
| total_bedrooms | Общее количество спален | float | [1, 6445] |
| population | Население | float | [3, 35682] |
| households | Количество домохозяйств | float | [1, 6082] |
| median_income | Медианный доход (в десятках тыс. $) | float | [0.5, 15.0] |
| bedrooms_per_room | Отношение спален к комнатам | float | [0, 1] |
| population_per_household | Население на домохозяйство | float | [0, 55] |
| households_per_population | Домохозяйства на население | float | [0, 1] |

**One-hot encoded features:**
- `ocean__INLAND`, `ocean__NEAR_BAY`, `ocean__NEAR_OCEAN`, `ocean__ISLAND`

## Model Performance
### Метрики на тестовой выборке (4128 samples):
| Metric | Value | Description |
|--------|-------|-------------|
| R² Score | 0.8028 | Коэффициент детерминации |
| MSE | 2,584,714,064 | Среднеквадратичная ошибка |
| MAE | ~$31,741 | Средняя абсолютная ошибка |

### Сравнение моделей:
| Model | Data Version | R² Score | MSE |
|-------|--------------|----------|-----|
| Linear Regression | v1 | 0.6169 | 5,019,864,241 |
| Random Forest | v1 | 0.8003 (log) | 2,300,033,137,095 |
| **XGBoost** | **v2** | **0.8028** | **2,584,714,064** |

## Model Architecture
- **Algorithm**: XGBoost (Extreme Gradient Boosting)
- **Hyperparameters**:
  - `n_estimators`: 100
  - `max_depth`: 6
  - `learning_rate`: 0.1
  - `random_state`: 42
- **Training Time**: ~10 секунд
- **Inference Time**: < 10 мс на запрос

## Ethical Considerations
- **Bias**: Данные 1990 года могут не отражать текущую демографическую ситуацию
- **Fairness**: Модель может иметь региональные смещения
- **Transparency**: Исходный код и данные открыты для проверки

## Deployment
- **Framework**: FastAPI + Uvicorn
- **Container**: Docker (python:3.10-slim)
- **Endpoints**:
  - `GET /health` - проверка работоспособности
  - `POST /predict` - предсказание для одного образца
  - `POST /predict/batch` - батч-предсказания
- **API Documentation**: Swagger UI (`/docs`)

## Maintenance
- **Retraining**: При поступлении новых данных
- **Monitoring**: Логирование через MLflow
- **Versioning**: DVC для данных и моделей, Git для кода

## References
1. Pace, R. Kelley, and Ronald Barry. "Sparse spatial autoregressions." Statistics & Probability Letters 33.3 (1997)
2. Géron, A. "Hands-On Machine Learning with Scikit-Learn and TensorFlow" (2019)
3. Оригинальный датасет: https://www.kaggle.com/datasets/camnugent/california-housing-prices