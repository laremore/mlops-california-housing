
import pandas as pd
import numpy as np
import os

print("=== Версия 2: Расширенная предобработка ===")

df = pd.read_csv('data/raw/housing.csv')
print(f"Исходные данные: {len(df)} строк, {df.shape[1]} колонок")


df['total_bedrooms'] = df['total_bedrooms'].fillna(0)
print("Заполнили пропуски в total_bedrooms нулями")


df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population'] / df['households']
df['households_per_population'] = df['households'] / df['population']


ocean_dummies = pd.get_dummies(df['ocean_proximity'], prefix='ocean')
df = pd.concat([df, ocean_dummies], axis=1)


df['median_house_value_log'] = np.log1p(df['median_house_value'])

numeric_cols = ['housing_median_age', 'total_rooms', 'total_bedrooms', 
                'population', 'households', 'median_income',
                'rooms_per_household', 'bedrooms_per_room', 
                'population_per_household', 'households_per_population']

for col in numeric_cols:
    if col in df.columns:
        df[col + '_norm'] = (df[col] - df[col].mean()) / df[col].std()


os.makedirs('data/processed', exist_ok=True)
output_path = 'data/processed/housing_processed_v2.csv'
df.to_csv(output_path, index=False)

print(f"Сохранено: {output_path}")
print(f"Итоговый размер: {len(df)} строк, {df.shape[1]} колонок")
print(f"Новые колонки: {list(df.columns[-15:])}") 