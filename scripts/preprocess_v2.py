
import pandas as pd
import numpy as np
import os



df = pd.read_csv('data/raw/housing.csv')
print(f"Исходные данные: {len(df)} строк, {df.shape[1]} колонок")


df['total_bedrooms'] = df['total_bedrooms'].fillna(0)
print("Заполнили пропуски в total_bedrooms нулями")


df['households_safe'] = df['households'].replace(0, 1)
df['total_rooms_safe'] = df['total_rooms'].replace(0, 1)
df['population_safe'] = df['population'].replace(0, 1)

df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms_safe']
df['population_per_household'] = df['population_safe'] / df['households_safe']
df['households_per_population'] = df['households'] / df['population_safe']


df = df.drop(columns=['households_safe', 'total_rooms_safe', 'population_safe'])


if 'ocean_proximity' in df.columns:
    ocean_dummies = pd.get_dummies(df['ocean_proximity'], prefix='ocean')
    df = pd.concat([df, ocean_dummies], axis=1)


df['median_house_value_log'] = np.log1p(df['median_house_value'])


df.replace([np.inf, -np.inf], np.nan, inplace=True)


for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].mean())


for col in df.select_dtypes(include=[np.number]).columns:
    if np.isinf(df[col]).any():
        print(f"Исправляем бесконечности в {col}")
        col_max = df[col][~np.isinf(df[col])].max()
        col_min = df[col][~np.isinf(df[col])].min()
        df[col] = df[col].replace(np.inf, col_max)
        df[col] = df[col].replace(-np.inf, col_min)


os.makedirs('data/processed', exist_ok=True)
output_path = 'data/processed/housing_processed_v2.csv'
df.to_csv(output_path, index=False)

print(f"\nСохранено: {output_path}")
print(f"Итоговый размер: {len(df)} строк, {df.shape[1]} колонок")
