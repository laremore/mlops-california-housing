# MLOps Pipeline for California Housing Prices Prediction

## О проекте
Курсовой проект по дисциплине "Операционализация моделей машинного обучения". 
MLOps-пайплайн для предсказания стоимости домов в Калифорнии на основе данных переписи 1990 года.

## Цели проекта
1. Реализация полного MLOps пайплайна
2. Версионирование данных и экспериментов
3. Контейнеризация ML-сервиса
4. Документирование всех этапов

## Данные
- Датасет: California Housing Prices (1990 Census)
- Целевая переменная: `median_house_value` (медианная стоимость дома)
- Признаки: 10 (география, демография, доходы)
- Размер: 20,640 
## Архитектура
mlops_california_housing/
├── data/ # Версионированные данные (DVC)
│ ├── raw/ # Исходные данные
│ └── processed/ # Обработанные данные (v1, v2)
├── src/ # Исходный код
│ ├── training/ # Обучение моделей
│ └── inference/ # FastAPI сервис
├── models/ # Сохраненные модели (DVC)
├── notebooks/ # EDA и эксперименты
├── configs/ # Конфигурационные файлы
├── scripts/ # Вспомогательные скрипты
├── tests/ # Тесты
└── mlruns/ # MLflow эксперименты
## Быстрый старт
 1. Клонирование и настройка
git clone <репозиторий>
cd mlops_california_housing
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
2. Запуск экспериментов
bash
cd src/training
python experiment_runner.py --experiment all
3. Запуск сервиса локально
bash
python run_mlflow_experiments.py
uvicorn src.inference.app:app --host 0.0.0.0 --port 8000 --reload
4. Запуск в Docker
docker build -t california-housing-ml .

# Запуск контейнера
docker run -d -p 8000:8000 --name housing-ml-service california-housing-ml

# Проверка
curl http://localhost:8000/health


Было протестировано 3 модели:
Linear Regression (базовая)
Random Forest (с GridSearch)
XGBoost (лучшая модель, R²=0.8028)

Технологический стек
Python 3.10 + основные ML библиотеки
DVC - версионирование данных
MLflow - трекинг экспериментов
FastAPI - инференс-сервис
Docker - контейнеризация
Git - контроль версий кода

#Pipeline:
1. Data Collection
   ↓
2. Data Versioning (DVC) → data/raw/housing.csv
   ↓
3. Preprocessing → data/processed/v1.csv, v2.csv
   ↓
4. Experiment Tracking (MLflow) → 3 эксперимента
   ↓
5. Model Selection → models/best_model.joblib
   ↓
6. Service Development (FastAPI)
   ↓
7. Containerization (Docker)
   ↓
8. Deployment & Monitoring