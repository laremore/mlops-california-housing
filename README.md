# MLOps Pipeline for California Housing Prices Prediction

## О проекте
Курсовой проект по дисциплине "Операционализация моделей машинного обучения". 
MLOps-пайплайн для предсказания стоимости домов в Калифорнии на основе данных переписи 1990 года.

## Цели проекта
1. Реализация полного MLOps пайплайна
2. Версионирование данных и экспериментов (MLflow)
3. Контейнеризация ML-сервиса с Docker Compose
4. Документирование всех этапов

## Данные
- **Датасет**: California Housing Prices (1990 Census)
- **Целевая переменная**: `median_house_value` (медианная стоимость дома)
- **Признаки**: географические, демографические, экономические
- **Размер**: 20,640 записей, 10 признаков
- **Версии данных**: v1 (базовая обработка), v2 (расширенные фичи)

## Архитектура проекта
```
mlops_california_housing/
├── data/                          
│   ├── raw/                        # Исходные данные
│   └── processed/                  # Обработанные данные (v1, v2)
├── src/                            
│   ├── training/                   
│   │   └── experiment_runner.py    # Запуск экспериментов
│   └── inference/                  # FastAPI сервис
│       ├── app.py                  
│       ├── model_loader.py        
│       ├── config.py              
│       └── schemas.py             
├── models/                         # Сохраненные модели после экспериментов
├── notebooks/                      # EDA и эксперименты Jupyter
├── configs/                        # Конфигурационные файлы
├── scripts/                        # Вспомогательные скрипты
├── tests/                          # Юнит-тесты
├── mlflow_data/                    # MLflow база данных (SQLite)
├── mlruns/                         # MLflow артефакты
├── docker-compose.yml              # Docker Compose конфигурация
├── Dockerfile                      # Основной Dockerfile
├── Dockerfile.mlflow               # MLflow UI Dockerfile
├── requirements.txt                # Зависимости
└── README.md                       # Документация
```

## Быстрый старт

### 1. Клонирование и настройка окружения
```bash
# Клонирование репозитория 
git clone <репозиторий>
cd mlops_california_housing

# Создание виртуального окружения (Windows)
python -m venv venv
venv\Scripts\activate

# Установка зависимостей
pip install -r requirements.txt
```

### 2. Подготовка данных
```bash
# Запуск предобработки данных
python scripts/preprocess_v1.py      # Базовая версия
python scripts/preprocess_v2.py      # Расширенная версия с дополнительными фичами
```

### 3. Обучение моделей и эксперименты
```bash
# Запуск всех моделей (сохранение в models/)
python src/training/experiment_runner.py --experiment all

# Запуск MLflow экспериментов 
python run_mlflow_experiments.py
```

### 4. Запуск сервисов

#### Вариант 1: Локальный запуск (без Docker)
```bash
# Запуск MLflow UI локально
mlflow server --backend-store-uri sqlite:///mlflow_data/mlflow.db --default-artifact-root file:///mlruns --host 0.0.0.0 --port 5000

# В другом терминале: запуск FastAPI сервиса
uvicorn src.inference.app:app --host 0.0.0.0 --port 8000 --reload
```

#### Вариант 2: Docker Compose 
```bash
# Сборка и запуск всех сервисов
docker-compose up --build

# Или с помощью скрипта
run_all.bat  # для Windows
```

#### Вариант 3: Один Docker контейнер
```bash
docker build -t california-housing-ml .
docker run -d -p 8000:8000 --name housing-ml-service california-housing-ml
```

### 5. Проверка работы
```bash
# Проверка здоровья сервиса
curl http://localhost:8000/health

# Пример предсказания
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41.0,
    "total_rooms": 880.0,
    "total_bedrooms": 129.0,
    "population": 322.0,
    "households": 126.0,
    "median_income": 8.3252
  }'
```

## Веб-интерфейсы

### FastAPI Swagger UI
- **URL**: http://localhost:8000/docs
- **Описание**: Интерактивная документация API
- **Эндпоинты**:
  - `GET /health` - проверка состояния сервиса
  - `POST /predict` - предсказание цены дома
  - `POST /predict/batch` - пакетное предсказание
  - `GET /debug-prediction` - отладка предсказания

### MLflow Tracking UI
- **URL**: http://localhost:5000
- **Описание**: Мониторинг экспериментов, метрик и моделей
- **Возможности**:
  - Просмотр всех экспериментов (3+ эксперимента)
  - Сравнение метрик моделей
  - Просмотр артефактов (модели, параметры)

## Результаты экспериментов

Было протестировано 3 модели с разными версиями данных:

### Эксперимент 1: Linear Regression
- **Данные**: v1 (базовая обработка)
- **Метрики**: 
  - R²: ~0.65
  - MSE: ~4.9e+09


### Эксперимент 2: Random Forest
- **Данные**: v1 (с логарифмической трансформацией целевой переменной)
- **Метрики**:
  - R² (log): ~0.80
  - R²: ~0.79


### Эксперимент 3: XGBoost 
- **Данные**: v2 (расширенные фичи + one-hot encoding)
- **Метрики**:
  - R² (log): ~0.82
  - R²: ~0.81
- **Особенности**: Лучшая модель, используется в продакшене
- **Сохраняется как**: `models/best_model.joblib`

## Технологический стек

| Компонент | Технология | Назначение |
|-----------|------------|------------|
| **Язык программирования** | Python 3.10 | Основной язык разработки |
| **ML библиотеки** | scikit-learn, XGBoost, pandas, numpy | Машинное обучение и обработка данных |
| **Эксперименты** | MLflow 3.7.0 | Трекинг экспериментов и метрик |
| **База данных** | SQLite | Хранение метаданных MLflow |
| **API фреймворк** | FastAPI 0.104.1 | REST API для инференса |
| **Контейнеризация** | Docker + Docker Compose | Упаковка и развертывание |
| **Конфигурация** | YAML, .env | Управление настройками |
| **Тестирование** | pytest | Юнит-тесты |
| **Документация** | Swagger UI | Автодокументирование API |

## Схема пайплайна

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

## Основные компоненты системы

### 1. Версионирование экспериментов (MLflow + SQLite)
- **Хранилище**: SQLite база данных (`mlflow_data/mlflow.db`)
- **Артефакты**: Локальная файловая система (`mlruns/`)
- **Эксперименты**: 3+ различных конфигураций
- **Модели**: Автоматическое логирование в MLflow

### 2. Инференс-сервис (FastAPI)
- **Эндпоинты**: `/health`, `/predict`, `/predict/batch`, `/debug-prediction`
- **Документация**: Автогенерируемая Swagger UI
- **Загрузка модели**: Автоматическая при старте сервиса
- **Обработка ошибок**: Полная валидация входных данных

### 3. Контейнеризация (Docker)
- **Основной сервис**: Python 3.10-slim + FastAPI
- **MLflow UI**: Отдельный контейнер для мониторинга
- **Вольюмы**: Примонтированные директории для данных и моделей
- **Сетевое взаимодействие**: Docker Compose для оркестрации

## Мониторинг и отладка

### Логирование
- **Уровни логов**: INFO, WARNING, ERROR
- **Файлы логов**: Контейнерные логи + MLflow журналы

### Отладка API
```bash
# Проверка состояния сервиса
curl http://localhost:8000/health

# Отладочный эндпоинт
curl http://localhost:8000/debug-prediction
```

### Просмотр экспериментов
1. Откройте http://localhost:5000
2. Выберите эксперимент "california_housing"
3. Сравните метки разных запусков
4. Просмотрите сохраненные модели и параметры

## Тестирование

```bash
# Запуск всех тестов
pytest tests/

# Тестирование API
python -m pytest tests/test_api.py -v

# Тестирование моделей
python -m pytest tests/test_model.py -v
```

## Структура конфигурации

```
configs/
├── data_config.yaml          # Настройки данных
├── experiment_config.yaml    # Параметры экспериментов
└── inference_config.yaml     # Настройки инференс-сервиса
```

## Развертывание


### Процесс деплоя
1. Клонировать репозиторий
2. Запустить предобработку данных
3. Обучить и сохранить модели
4. Собрать Docker образы
5. Запустить через Docker Compose

### Команды для развертывания
```bash
# Полный пайплайн одним скриптом (Windows)
run_all.bat

# Или пошагово
docker-compose build
docker-compose up -d

# Проверка
docker-compose ps
curl http://localhost:8000/health
```


## Контакты и ссылки

- **Датасет**: [California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
- **Документация FastAPI**: https://fastapi.tiangolo.com/
- **Документация MLflow**: https://mlflow.org/docs/
- **Курсовой проект**: Операционализация моделей машинного обучения

## Лицензия

Проект создан в образовательных целях для курсовой работы. Датасет California Housing Prices является общедоступным.

---
