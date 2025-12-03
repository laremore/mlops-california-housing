# scripts/preprocess_v1.py
import pandas as pd
import numpy as np
import os

print("=== Версия 1: Базовая предобработка ===")


df = pd.read_csv('data/raw/housing.csv')
print(f"Исходные данные: {len(df)} строк, {df.shape[1]} колонок")

median_bedrooms = df['total_bedrooms'].median()
df['total_bedrooms'] = df['total_bedrooms'].fillna(median_bedrooms)
print(f"Заполнено {df['total_bedrooms'].isnull().sum()} пропусков в total_bedrooms")

df['rooms_per_household'] = df['total_rooms'] / df['households']

threshold = df['median_house_value'].quantile(0.99)
df_clean = df[df['median_house_value'] <= threshold].copy()
print(f"Удалено {len(df) - len(df_clean)} строк-выбросов")

os.makedirs('data/processed', exist_ok=True)
output_path = 'data/processed/housing_processed_v1.csv'
df_clean.to_csv(output_path, index=False)

print(f"Сохранено: {output_path}")
print(f"Итоговый размер: {len(df_clean)} строк, {df_clean.shape[1]} колонок")