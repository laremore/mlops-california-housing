
import mlflow
import mlflow.sklearn
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

warnings.filterwarnings('ignore')


PROJECT_ROOT = Path(__file__).parent
DATA_V1_PATH = PROJECT_ROOT / "data/processed/housing_processed_v1.csv"
DATA_V2_PATH = PROJECT_ROOT / "data/processed/housing_processed_v2.csv"
MODELS_DIR = PROJECT_ROOT / "models"

def setup_mlflow():
    mlflow_data_dir = PROJECT_ROOT / "mlflow_data"
    mlflow_data_dir.mkdir(exist_ok=True)
    
    db_path = mlflow_data_dir / "mlflow.db"
    tracking_uri = f"sqlite:///{db_path}"
    
    print(f"MLflow Tracking URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    
    return tracking_uri

def load_data(data_path):
    print(f"Загрузка данных из: {data_path}")
    return pd.read_csv(data_path)

def prepare_data(df, use_log_target=False):
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

def safe_evaluate(y_true, y_pred, y_true_original=None, use_log=False, prefix=""):
   
    metrics = {}
    
    y_pred_clean = np.nan_to_num(y_pred, 
                                 nan=np.nanmean(y_pred) if not np.all(np.isnan(y_pred)) else 0,
                                 posinf=1e6,
                                 neginf=-1e6)
    
    if use_log and y_true_original is not None:

        metrics[f"{prefix}mse_log"] = mean_squared_error(y_true, y_pred_clean)
        metrics[f"{prefix}mae_log"] = mean_absolute_error(y_true, y_pred_clean)
        metrics[f"{prefix}r2_log"] = r2_score(y_true, y_pred_clean)
        
        try:
            y_pred_exp = np.expm1(y_pred_clean)

            y_pred_exp = np.clip(y_pred_exp, 50000, 500000)
            
            metrics[f"{prefix}mse"] = mean_squared_error(y_true_original, y_pred_exp)
            metrics[f"{prefix}mae"] = mean_absolute_error(y_true_original, y_pred_exp)
            metrics[f"{prefix}r2"] = r2_score(y_true_original, y_pred_exp)
        except:
            metrics[f"{prefix}mse"] = np.nan
            metrics[f"{prefix}mae"] = np.nan
            metrics[f"{prefix}r2"] = np.nan
    else:
        metrics[f"{prefix}mse"] = mean_squared_error(y_true, y_pred_clean)
        metrics[f"{prefix}mae"] = mean_absolute_error(y_true, y_pred_clean)
        metrics[f"{prefix}r2"] = r2_score(y_true, y_pred_clean)
    
    return metrics

def experiment_1_linear_regression():
  
    print("ЭКСПЕРИМЕНТ 1: Linear Regression (v1 data)")

    
    with mlflow.start_run(run_name="linear_regression_v1"):
        df = load_data(DATA_V1_PATH)
        X, y, y_orig, use_log = prepare_data(df, use_log_target=False)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("data_version", "v1")
        mlflow.log_param("features", X.shape[1])
        mlflow.log_param("use_log_target", False)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = safe_evaluate(y_test, y_pred, y_orig.iloc[X_test.index], use_log=False)
        
        for metric_name, value in metrics.items():
            if not np.isnan(value):
                mlflow.log_metric(metric_name, value)
        
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Результаты Linear Regression:")
        for k, v in metrics.items():
            if not np.isnan(v):
                print(f"  {k}: {v:.4f}")
        
        return metrics

def experiment_2_random_forest_simple():


    print("ЭКСПЕРИМЕНТ 2: Random Forest (v1 data - log scale only)")

    
    with mlflow.start_run(run_name="random_forest_v1"):
        df = load_data(DATA_V1_PATH)
        X, y, y_orig, use_log = prepare_data(df, use_log_target=True)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("data_version", "v1")
        mlflow.log_param("use_log_target", True)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 20)
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse_log = mean_squared_error(y_test, y_pred)
        mae_log = mean_absolute_error(y_test, y_pred)
        r2_log = r2_score(y_test, y_pred)
        
        mlflow.log_metric("mse_log", mse_log)
        mlflow.log_metric("mae_log", mae_log)
        mlflow.log_metric("r2_log", r2_log)
        
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Метрики в логарифмической шкале:")
        print(f"  MSE (log): {mse_log:.4f}")
        print(f"  MAE (log): {mae_log:.4f}")
        print(f"  R² (log): {r2_log:.4f}")
        
        return {"mse_log": mse_log, "mae_log": mae_log, "r2_log": r2_log}

def experiment_3_xgboost_v2_simple():
    
    print("ЭКСПЕРИМЕНТ 3: XGBoost (v2 data)")

    
    with mlflow.start_run(run_name="xgboost_v2"):
        df = load_data(DATA_V2_PATH)
        X, y, y_orig, use_log = prepare_data(df, use_log_target=True)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }
        
        mlflow.log_param("model_type", "XGBRegressor")
        mlflow.log_param("data_version", "v2")
        mlflow.log_param("use_log_target", True)
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse_log = mean_squared_error(y_test, y_pred)
        mae_log = mean_absolute_error(y_test, y_pred)
        r2_log = r2_score(y_test, y_pred)
        
        mlflow.log_metric("mse_log", mse_log)
        mlflow.log_metric("mae_log", mae_log)
        mlflow.log_metric("r2_log", r2_log)
        
        try:
            y_pred_exp = np.expm1(y_pred)
            y_pred_exp = np.clip(y_pred_exp, 50000, 500000)
            y_test_orig = y_orig.iloc[X_test.index].values
            
            mse = mean_squared_error(y_test_orig, y_pred_exp)
            mae = mean_absolute_error(y_test_orig, y_pred_exp)
            r2 = r2_score(y_test_orig, y_pred_exp)
            
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            
            print(f"Метрики в логарифмической шкале:")
            print(f"  MSE (log): {mse_log:.4f}")
            print(f"  MAE (log): {mae_log:.4f}")
            print(f"  R² (log): {r2_log:.4f}")
            print(f"\nМетрики в обычной шкале:")
            print(f"  MSE: {mse:.2f}")
            print(f"  MAE: {mae:.2f}")
            print(f"  R²: {r2:.4f}")
            
            metrics = {"mse_log": mse_log, "mae_log": mae_log, "r2_log": r2_log,
                      "mse": mse, "mae": mae, "r2": r2}
        except Exception as e:
            print(f"Ошибка при преобразовании: {e}")
            print(f"Метрики только в логарифмической шкале:")
            print(f"  MSE (log): {mse_log:.4f}")
            print(f"  MAE (log): {mae_log:.4f}")
            print(f"  R² (log): {r2_log:.4f}")
            metrics = {"mse_log": mse_log, "mae_log": mae_log, "r2_log": r2_log}
        
        mlflow.sklearn.log_model(model, "model")
        
        return metrics

def main():
    setup_mlflow()
    mlflow.set_experiment("california_housing_experiments")
    
    print("ЗАПУСК ЭКСПЕРИМЕНТОВ С MLFLOW")

    
    results = []
    
    results.append(experiment_1_linear_regression())
    results.append(experiment_2_random_forest_simple())
    results.append(experiment_3_xgboost_v2_simple())
    
    print("ИТОГИ ВСЕХ ЭКСПЕРИМЕНТОВ")

    
    experiment_names = ["Linear Regression", "Random Forest", "XGBoost"]
    
    for i, (name, metrics) in enumerate(zip(experiment_names, results)):
        print(f"\n{name}:")
        if 'r2' in metrics:
            print(f"  R²: {metrics.get('r2', 'N/A'):.4f}")
        print(f"  R² (log): {metrics.get('r2_log', metrics.get('r2', 'N/A')):.4f}")
        if 'mse' in metrics:
            print(f"  MSE: {metrics.get('mse', 0):.2f}")

if __name__ == "__main__":
    main()