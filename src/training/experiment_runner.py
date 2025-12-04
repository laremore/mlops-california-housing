import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import yaml
import argparse
import os
import warnings
from pathlib import Path
import joblib

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent  

def get_absolute_path(relative_path):
    """Получить абсолютный путь"""
    path = PROJECT_ROOT / relative_path
    return str(path).replace('\\', '/')

def load_data(data_path):
    absolute_path = get_absolute_path(data_path)
    print(f"Загрузка данных из: {absolute_path}")
    if not os.path.exists(absolute_path):
        print(f"ОШИБКА: Файл не найден: {absolute_path}")
        raise FileNotFoundError(f"Файл не найден: {absolute_path}")
    return pd.read_csv(absolute_path)

def prepare_data(df, use_log_target=False, target_col="median_house_value"):
    df = df.copy() 
    if use_log_target and "median_house_value_log" in df.columns:
        y = df["median_house_value_log"]
        y_original = df[target_col]
        use_log = True
    else:
        y = df[target_col]
        y_original = y
        use_log = False
    
    features_to_drop = [target_col, "median_house_value_log", 
                       "median_house_value", "ocean_proximity"]
    features_to_drop = [col for col in features_to_drop if col in df.columns]
    
    X = df.drop(columns=features_to_drop)
    X = X.fillna(X.mean())
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    
    return X, y, y_original, use_log

def evaluate_model(y_true, y_pred, y_true_original=None, y_pred_original=None, use_log=False):
    metrics = {}
    y_pred_clean = np.nan_to_num(y_pred, nan=np.nanmean(y_pred), 
                                  posinf=np.nanmax(y_pred) if not np.all(np.isnan(y_pred)) else 1e6,
                                  neginf=np.nanmin(y_pred) if not np.all(np.isnan(y_pred)) else -1e6)
    
    if use_log and y_true_original is not None:
        metrics["mse_log"] = mean_squared_error(y_true, y_pred_clean)
        metrics["mae_log"] = mean_absolute_error(y_true, y_pred_clean)
        metrics["r2_log"] = r2_score(y_true, y_pred_clean)
        
        try:
            y_pred_exp = np.expm1(y_pred_clean)
            y_pred_exp = np.clip(y_pred_exp, 0, 5e6)
            
            metrics["mse"] = mean_squared_error(y_true_original, y_pred_exp)
            metrics["mae"] = mean_absolute_error(y_true_original, y_pred_exp)
            metrics["r2"] = r2_score(y_true_original, y_pred_exp)
        except:
            metrics["mse"] = np.nan
            metrics["mae"] = np.nan
            metrics["r2"] = np.nan
    else:
        metrics["mse"] = mean_squared_error(y_true, y_pred_clean)
        metrics["mae"] = mean_absolute_error(y_true, y_pred_clean)
        metrics["r2"] = r2_score(y_true, y_pred_clean)
    
    return metrics

def run_experiment_1_linear():
    print("\n" + "="*60)
    print("ЭКСПЕРИМЕНТ 1: Linear Regression (v1 data)")
    print("="*60)
    
    try:
        df = load_data("data/processed/housing_processed_v1.csv")
        X, y, y_orig, use_log = prepare_data(df, use_log_target=False)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Размер X_train: {X_train.shape}")
        print(f"Размер X_test: {X_test.shape}")
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = evaluate_model(y_test, y_pred, y_orig.iloc[X_test.index], y_pred, use_log=False)

        print("\nРезультаты Linear Regression:")
        print(f"R²: {metrics.get('r2', 0):.4f}")
        print(f"MSE: {metrics.get('mse', 0):.2f}")
        print(f"MAE: {metrics.get('mae', 0):.2f}")
        
        # Сохраняем модель
        models_dir = PROJECT_ROOT / "models"
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / "linear_model.joblib"
        joblib.dump(model, model_path)
        print(f"✓ Модель сохранена: {model_path}")
        
        return metrics
    except Exception as e:
        print(f"Ошибка в эксперименте 1: {e}")
        import traceback
        traceback.print_exc()
        return {}

def run_experiment_2_random_forest():
    print("\n" + "="*60)
    print("ЭКСПЕРИМЕНТ 2: Random Forest (v1 data)")
    print("="*60)
    
    try:
        df = load_data("data/processed/housing_processed_v1.csv")
        X, y, y_orig, use_log = prepare_data(df, use_log_target=True)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Размер X_train: {X_train.shape}")
        
        # Упрощаем для скорости
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = evaluate_model(y_test, y_pred, y_orig.iloc[X_test.index], y_pred, use_log=True)
        
        print("\nРезультаты Random Forest:")
        print(f"R² (log): {metrics.get('r2_log', 0):.4f}")
        print(f"R²: {metrics.get('r2', 0):.4f}")
        print(f"MSE: {metrics.get('mse', 0):.2f}")
        
        # Сохраняем модель
        models_dir = PROJECT_ROOT / "models"
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / "rf_model.joblib"
        joblib.dump(model, model_path)
        print(f"✓ Модель сохранена: {model_path}")
        
        return metrics
    except Exception as e:
        print(f"Ошибка в эксперименте 2: {e}")
        import traceback
        traceback.print_exc()
        return {}

def run_experiment_3_xgboost_v2():
    print("\n" + "="*60)
    print("ЭКСПЕРИМЕНТ 3: XGBoost (v2 data with more features)")
    print("="*60)
    
    try:
        df = load_data("data/processed/housing_processed_v2.csv")
        
        X, y, y_orig, use_log = prepare_data(df, use_log_target=True)
        
        print(f"Форма X: {X.shape}")
        print(f"Первые 5 колонок: {X.columns.tolist()[:5]}")
        print(f"Все колонки: {X.columns.tolist()}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Размер X_train: {X_train.shape}")
        print(f"Размер X_test: {X_test.shape}")
        
        model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = evaluate_model(y_test, y_pred, y_orig.iloc[X_test.index], y_pred, use_log=True)
        
        print("\nРезультаты XGBoost:")
        print(f"R² (log): {metrics.get('r2_log', 0):.4f}")
        print(f"R²: {metrics.get('r2', 0):.4f}")
        print(f"MSE: {metrics.get('mse', 0):.2f}")
        
        # СОХРАНЕНИЕ МОДЕЛИ (ГЛАВНОЕ!)
        models_dir = PROJECT_ROOT / "models"
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / "best_model.joblib"
        joblib.dump(model, model_path)
        print(f"✓ Модель сохранена: {model_path}")
        
        # Сохраняем также список фич, которые ожидает модель
        feature_names = X.columns.tolist()
        features_path = models_dir / "feature_names.joblib"
        joblib.dump(feature_names, features_path)
        print(f"✓ Имена фич сохранены: {features_path}")
        
        return metrics
    except Exception as e:
        print(f"Ошибка в эксперименте 3: {e}")
        import traceback
        traceback.print_exc()
        return {}

def main():
    parser = argparse.ArgumentParser(description='Run ML experiments')
    parser.add_argument('--experiment', type=str, default='all',
                      help='Which experiment to run: 1, 2, 3, or all')
    args = parser.parse_args()
    
    print(f"Запуск эксперимента: {args.experiment}")
    print(f"Рабочая директория: {os.getcwd()}")
    print(f"Корень проекта: {PROJECT_ROOT}")
    
    # Создаем необходимые директории
    (PROJECT_ROOT / "models").mkdir(exist_ok=True)
    (PROJECT_ROOT / "mlruns").mkdir(exist_ok=True)
    
    experiments = []
    
    if args.experiment in ['1', 'all']:
        experiments.append(run_experiment_1_linear())
    
    if args.experiment in ['2', 'all']:
        experiments.append(run_experiment_2_random_forest())
    
    if args.experiment in ['3', 'all']:
        experiments.append(run_experiment_3_xgboost_v2())
    
    if experiments:
        print("\n" + "="*60)
        print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ")
        print("="*60)
        
        best_r2 = -np.inf
        best_model_name = ""
        
        for i, metrics in enumerate(experiments, 1):
            if metrics:
                model_names = ["Linear Regression", "Random Forest", "XGBoost"]
                model_name = model_names[i-1] if i-1 < len(model_names) else f"Model {i}"
                
                print(f"\n{model_name}:")
                for k, v in metrics.items():
                    if not np.isnan(v):
                        print(f"  {k}: {v:.4f}")
                
                # Определяем лучшую модель по R²
                current_r2 = metrics.get('r2', -np.inf)
                if current_r2 > best_r2:
                    best_r2 = current_r2
                    best_model_name = model_name
        
        print(f"\n{'='*60}")
        print(f"ЛУЧШАЯ МОДЕЛЬ: {best_model_name} (R² = {best_r2:.4f})")
        print(f"{'='*60}")
        
        print("\n✓ Все модели сохранены в папке models/")
        print("✓ Основная модель: models/best_model.joblib")

if __name__ == "__main__":
    main()