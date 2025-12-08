
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib
from pathlib import Path
import warnings
import os

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
DATA_V1_PATH = PROJECT_ROOT / "data/processed/housing_processed_v1.csv"
DATA_V2_PATH = PROJECT_ROOT / "data/processed/housing_processed_v2.csv"
MODELS_DIR = PROJECT_ROOT / "models"

def setup_mlflow_with_sqlite():

    mlflow_data_dir = PROJECT_ROOT / "mlflow_data"
    mlruns_dir = PROJECT_ROOT / "mlruns"
    
    mlflow_data_dir.mkdir(exist_ok=True)
    mlruns_dir.mkdir(exist_ok=True)
    
    db_path = mlflow_data_dir / "mlflow.db"
    
    if not db_path.exists():
        db_path.touch()
        print(f"Создана новая база данных: {db_path}")
    
    tracking_uri = f"sqlite:///{db_path.absolute()}"
    artifact_location = f"file://{mlruns_dir.absolute()}"
    
    print(f"MLflow Tracking URI: {tracking_uri}")
    print(f"Artifact location: {artifact_location}")
    print(f"Database: {db_path}")
    
    mlflow.set_tracking_uri(tracking_uri)
    
    experiment_name = "california_housing"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=artifact_location
            )
            print(f"Создан новый эксперимент: {experiment_name} (ID: {experiment_id})")
        else:
            print(f"Используем существующий эксперимент: {experiment_name}")
            print(f"   Experiment ID: {experiment.experiment_id}")
            print(f"   Artifact location: {experiment.artifact_location}")
    except Exception as e:
        print(f"Ошибка при создании эксперимента: {e}")

        mlflow.set_experiment(experiment_name)
    
    return tracking_uri

def load_data(data_path):
    print(f"Загрузка данных из: {data_path}")
    if not data_path.exists():
        print(f"ОШИБКА: Файл не найден: {data_path}")
        return None
    return pd.read_csv(data_path)

def prepare_data(df, use_log_target=False):
    if df is None:
        return None, None, None, False
        
    df = df.copy()
    
    if use_log_target and "median_house_value_log" in df.columns:
        y = df["median_house_value_log"]
        y_original = df["median_house_value"]
        use_log = True
    else:
        y = df["median_house_value"]
        y_original = y
        use_log = False
    
    cols_to_drop = ["median_house_value", "median_house_value_log", "ocean_proximity"]
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    X = df.drop(columns=cols_to_drop)
    X = X.fillna(X.mean())
    
    return X, y, y_original, use_log

def evaluate_model(y_true, y_pred, y_true_original=None, use_log=False):
    metrics = {}
    
    if use_log and y_true_original is not None:
        metrics["mse_log"] = mean_squared_error(y_true, y_pred)
        metrics["mae_log"] = mean_absolute_error(y_true, y_pred)
        metrics["r2_log"] = r2_score(y_true, y_pred)
        
        try:
            y_pred_exp = np.expm1(y_pred)
            y_pred_exp = np.clip(y_pred_exp, 50000, 500000)
            
            metrics["mse"] = mean_squared_error(y_true_original, y_pred_exp)
            metrics["mae"] = mean_absolute_error(y_true_original, y_pred_exp)
            metrics["r2"] = r2_score(y_true_original, y_pred_exp)
        except Exception as e:
            print(f"Ошибка при преобразовании логарифма: {e}")
            metrics["mse"] = np.nan
            metrics["mae"] = np.nan
            metrics["r2"] = np.nan
    else:
        metrics["mse"] = mean_squared_error(y_true, y_pred)
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["r2"] = r2_score(y_true, y_pred)
    
    return metrics

def run_experiment_1():

    print("ЭКСПЕРИМЕНТ 1: Linear Regression (v1 data)")

    
    try:
        df = load_data(DATA_V1_PATH)
        if df is None:
            return None
            
        X, y, y_orig, use_log = prepare_data(df, use_log_target=False)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        with mlflow.start_run(run_name="linear_regression_v1"):
            mlflow.set_tag("model_type", "LinearRegression")
            mlflow.set_tag("data_version", "v1")
            mlflow.set_tag("mlflow.runName", "linear_regression_v1")
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            metrics = evaluate_model(y_test, y_pred, y_orig.iloc[X_test.index], use_log=False)
            
            mlflow.log_param("model", "LinearRegression")
            mlflow.log_param("data_version", "v1")
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)
            
            for k, v in metrics.items():
                if not np.isnan(v):
                    mlflow.log_metric(k, v)
            

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="linear_regression_model"
            )
            
            print(f"MSE: {metrics.get('mse', 0):.2f}")
            print(f"R2: {metrics.get('r2', 0):.4f}")
            
            models_dir = PROJECT_ROOT / "models"
            models_dir.mkdir(exist_ok=True)
            model_path = models_dir / "linear_regression_model.joblib"
            joblib.dump(model, model_path)
            print(f"Модель сохранена локально: {model_path}")
            
            return metrics
            
    except Exception as e:
        print(f"Ошибка в эксперименте 1: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_experiment_2():

    print("ЭКСПЕРИМЕНТ 2: Random Forest (v1 data with log transform)")

    
    try:
        df = load_data(DATA_V1_PATH)
        if df is None:
            return None
            
        X, y, y_orig, use_log = prepare_data(df, use_log_target=True)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        with mlflow.start_run(run_name="random_forest_v1_log"):
            mlflow.set_tag("model_type", "RandomForest")
            mlflow.set_tag("data_version", "v1")
            mlflow.set_tag("target_transform", "log")
            mlflow.set_tag("mlflow.runName", "random_forest_v1_log")
            
            model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=20, 
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            metrics = evaluate_model(y_test, y_pred, y_orig.iloc[X_test.index], use_log=True)
            
            mlflow.log_param("model", "RandomForestRegressor")
            mlflow.log_param("data_version", "v1")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 20)
            mlflow.log_param("target_transform", "log1p")
            
            for k, v in metrics.items():
                if not np.isnan(v):
                    mlflow.log_metric(k, v)
            
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="random_forest_model"
            )
            
            print(f"R2 (log): {metrics.get('r2_log', 0):.4f}")
            print(f"R2: {metrics.get('r2', 0):.4f}")
            
            models_dir = PROJECT_ROOT / "models"
            models_dir.mkdir(exist_ok=True)
            model_path = models_dir / "random_forest_model.joblib"
            joblib.dump(model, model_path)
            print(f"Модель сохранена локально: {model_path}")
            
            return metrics
            
    except Exception as e:
        print(f"Ошибка в эксперименте 2: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_experiment_3():

    print("ЭКСПЕРИМЕНТ 3: XGBoost (v2 data with engineered features)")

    
    try:
        df = load_data(DATA_V2_PATH)
        if df is None:
            return None
            
        X, y, y_orig, use_log = prepare_data(df, use_log_target=True)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        with mlflow.start_run(run_name="xgboost_v2_features"):
            mlflow.set_tag("model_type", "XGBoost")
            mlflow.set_tag("data_version", "v2")
            mlflow.set_tag("target_transform", "log")
            mlflow.set_tag("mlflow.runName", "xgboost_v2_features")
            
            model = XGBRegressor(
                n_estimators=100, 
                max_depth=6, 
                learning_rate=0.1, 
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            metrics = evaluate_model(y_test, y_pred, y_orig.iloc[X_test.index], use_log=True)
            
            mlflow.log_param("model", "XGBRegressor")
            mlflow.log_param("data_version", "v2")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 6)
            mlflow.log_param("learning_rate", 0.1)
            mlflow.log_param("feature_engineering", "extended")
            
            for k, v in metrics.items():
                if not np.isnan(v):
                    mlflow.log_metric(k, v)
            
            mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path="xgboost_model"
            )
            
            print(f"R2 (log): {metrics.get('r2_log', 0):.4f}")
            print(f"R2: {metrics.get('r2', 0):.4f}")
            
            models_dir = PROJECT_ROOT / "models"
            models_dir.mkdir(exist_ok=True)
            model_path = models_dir / "best_model.joblib"
            joblib.dump(model, model_path)
            print(f"Модель сохранена локально как основная: {model_path}")
            
            feature_names = X.columns.tolist()
            features_path = models_dir / "feature_names.joblib"
            joblib.dump(feature_names, features_path)
            print(f"Имена фич сохранены: {features_path}")
            
            return metrics
            
    except Exception as e:
        print(f"Ошибка в эксперименте 3: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():

    print("ЗАПУСК ЭКСПЕРИМЕНТОВ MLflow")

    
    tracking_uri = setup_mlflow_with_sqlite()

    
    results = []
    
    exp1_result = run_experiment_1()
    results.append(exp1_result)
    
    exp2_result = run_experiment_2()
    results.append(exp2_result)
    
    exp3_result = run_experiment_3()
    results.append(exp3_result)
    
    print("ИТОГИ ЭКСПЕРИМЕНТОВ")
 
    
    successful_experiments = 0
    for i, metrics in enumerate(results, 1):
        if metrics is not None:
            successful_experiments += 1
            print(f"\nЭксперимент {i} УСПЕШЕН:")
            for k, v in metrics.items():
                if not np.isnan(v):
                    print(f"  {k}: {v:.4f}")
        else:
            print(f"\nЭксперимент {i}: ПРОВАЛЕН")
    
    print(f"\nУспешных экспериментов: {successful_experiments} из 3")
    

    print("ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ")

    print(f"MLflow Tracking UI: http://localhost:5000")
    print(f"Database: {PROJECT_ROOT}/mlflow_data/mlflow.db")
    print(f"Artifacts: {PROJECT_ROOT}/mlruns")
    
    try:
        import sqlite3
        db_path = PROJECT_ROOT / "mlflow_data" / "mlflow.db"
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM experiments")
            exp_count = cursor.fetchone()[0]
            print(f"\nЭкспериментов в базе данных: {exp_count}")
            
            cursor.execute("SELECT COUNT(*) FROM runs")
            run_count = cursor.fetchone()[0]
            print(f"Запусков в базе данных: {run_count}")
            
            conn.close()
            
            if run_count > 0:
                print(f"Эксперименты успешно сохранены в базу данных!")
            else:
                print(f"В базе данных нет записей о запусках")
    except Exception as e:
        print(f"Не удалось проверить базу данных: {e}")
    

if __name__ == "__main__":
    main()