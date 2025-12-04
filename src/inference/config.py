from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

MODEL_PATH = None

possible_names = [
    "best_model.joblib",     
    "fresh_model.joblib",     
    "model.joblib",          
]

models_dir = PROJECT_ROOT / "models"

for model_name in possible_names:
    candidate = models_dir / model_name
    if candidate.exists():
        MODEL_PATH = candidate
        print(f"Найдена модель: {model_name}")
        break

if MODEL_PATH is None:

    models_dir.mkdir(exist_ok=True)
    print(f"Модель не найдена в {models_dir}")
    MODEL_PATH = models_dir / "best_model.joblib"

print(f"Используем модель: {MODEL_PATH}")

class Settings:
    APP_NAME = "California Housing Price Predictor"
    VERSION = "1.0.0"
    MODEL_PATH = MODEL_PATH
    HOST = "0.0.0.0"
    PORT = 8000

settings = Settings()