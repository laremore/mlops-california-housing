# Dockerfile
# Используем официальный Python образ (соответствует требованиям курсовой)
FROM python:3.10-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Создаем необходимые папки
RUN mkdir -p models data/processed mlruns

# Загружаем данные через DVC (если нужно, но проще скопировать)
# RUN dvc pull || echo "DVC pull failed, using local data"

# Копируем модель и данные (проще чем через DVC в Docker)
COPY models/best_model.joblib models/
COPY data/processed/housing_processed_v2.csv data/processed/
COPY models/model_info.json models/
COPY models/example_features.json models/

# Экспонируем порт
EXPOSE 8000

# Команда запуска
CMD ["uvicorn", "src.inference.app:app", "--host", "0.0.0.0", "--port", "8000"]