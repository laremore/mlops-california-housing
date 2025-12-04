
import pandas as pd
import numpy as np
import os


df = pd.read_csv('data/raw/housing.csv')
print(f"Исходные данные: {len(df)} строк, {df.shape[1]} колонок")


median_bedrooms = df['total_bedrooms'].median()
df['total_bedrooms'] = df['total_bedrooms'].fillna(median_bedrooms)
print(f"Заполнено {df['total_bedrooms'].isnull().sum()} пропусков в total_bedrooms")


df['households_safe'] = df['households'].replace(0, 1)
df['rooms_per_household'] = df['total_rooms'] / df['households_safe']

df = df.drop(columns=['households_safe'])


df.replace([np.inf, -np.inf], np.nan, inplace=True)


df = df.dropna()
print(f"После удаления NaN: {len(df)} строк")


threshold = df['median_house_value'].quantile(0.99)
df_clean = df[df['median_house_value'] <= threshold].copy()
print(f"Удалено {len(df) - len(df_clean)} строк-выбросов")


for col in df_clean.select_dtypes(include=[np.number]).columns:
    if np.isinf(df_clean[col]).any():
        print(f"ВНИМАНИЕ: В колонке {col} все еще есть бесконечные значения!")
        max_finite = df_clean[col][~np.isinf(df_clean[col])].max()
        df_clean[col] = df_clean[col].replace([np.inf, -np.inf], max_finite)


os.makedirs('data/processed', exist_ok=True)
output_path = 'data/processed/housing_processed_v1.csv'
df_clean.to_csv(output_path, index=False)

print(f"\nСохранено: {output_path}")
print(f"Итоговый размер: {len(df_clean)} строк, {df_clean.shape[1]} колонок")
