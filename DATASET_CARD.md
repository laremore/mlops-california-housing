# Dataset Card: California Housing Prices

## Dataset Description
- **Name**: California Housing Prices
- **Version**: 1990 Census Data
- **Source**: U.S. Census Bureau 1990
- **License**: Public Domain (U.S. Government Works)
- **Original Source**: https://www.kaggle.com/datasets/camnugent/california-housing-prices

## Dataset Overview
Данные содержат информацию о домах в различных районах Калифорнии по данным переписи 1990 года.

### Статистика:
- **Всего записей**: 20,640
- **Признаков**: 10 (9 числовых + 1 категориальный)
- **Целевая переменная**: `median_house_value`
- **Период**: 1990 год

## Структура данных
### Исходные признаки:
| Column | Type | Description | Range/Values |
|--------|------|-------------|--------------|
| longitude | float | Долгота (западнее = больше) | [-124.35, -114.31] |
| latitude | float | Широта (севернее = больше) | [32.54, 41.95] |
| housing_median_age | float | Медианный возраст домов в блоке | [1, 52] лет |
| total_rooms | float | Общее количество комнат в блоке | [2, 39320] |
| total_bedrooms | float | Общее количество спален в блоке | [1, 6445] |
| population | float | Население блока | [3, 35682] |
| households | float | Количество домохозяйств | [1, 6082] |
| median_income | float | Медианный доход домохозяйств (в десятках тыс. $) | [0.5, 15.0] |
| median_house_value | float | Медианная стоимость дома (в $) | [14999, 500001] |
| ocean_proximity | categorical | Близость к океану | ['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'] |

### Версии датасета:
| Version | Описание | Размер | Особенности |
|---------|----------|--------|-------------|
| v0 (raw) | Исходные данные | 20,640 × 10 | Пропуски в total_bedrooms |
| v1 | Базовая предобработка | 20,397 × 10 | Заполнены пропуски, удалены выбросы |
| v2 | Расширенная обработка | 20,640 × 19 | Новые признаки, one-hot encoding, нормализация |

## Предобработка
### Версия 1 (basic):
1. Заполнение пропусков в `total_bedrooms` медианным значением
2. Создание признака `rooms_per_household`
3. Удаление выбросов (верхние 1% по `median_house_value`)

### Версия 2 (advanced):
1. Заполнение пропусков нулями
2. Создание новых признаков:
   - `bedrooms_per_room`
   - `population_per_household`
   - `households_per_population`
3. One-hot encoding для `ocean_proximity`
4. Логарифмирование целевой переменной
5. Нормализация числовых признаков

## Распределение данных
### Сплиты:
- **Train set**: 16,512 samples (80%)
- **Test set**: 4,128 samples (20%)
- **Validation**: Используется кросс-валидация в обучении

### Стратификация:
Разделение стратифицировано по квартилям целевой переменной для сохранения распределения.

## Проблемы данных
1. **Пропуски**: 207 пропущенных значений в `total_bedrooms`
2. **Выбросы**: Экстремальные значения в `median_house_value` (обрезаны при 99-м процентиле)
3. **Дисбаланс категорий**: `ocean_proximity` имеет неравномерное распределение
4. **Масштаб**: Признаки имеют разные масштабы (требуется нормализация)

## Этические соображения
- **Конфиденциальность**: Данные агрегированы на уровне блоков, нет персональной информации
- **Смещение**: Данные 1990 года могут не отражать текущие демографические тенденции
- **Применимость**: Модель для образовательных целей, не для реальных финансовых решений

## Использование
```python
import pandas as pd

# Загрузка данных
df = pd.read_csv('data/processed/housing_processed_v2.csv')

# Для обучения
X = df.drop(columns=['median_house_value', 'median_house_value_log'])
y = df['median_house_value_log']  # или df['median_house_value']