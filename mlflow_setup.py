
import os
from pathlib import Path

def setup_mlflow_database():

    project_root = Path(__file__).parent
    mlflow_data_dir = project_root / "mlflow_data"

    mlflow_data_dir.mkdir(exist_ok=True)
    
    db_path = mlflow_data_dir / "mlflow.db"
    
    tracking_uri = f"sqlite:///{db_path.absolute()}"
    
    print(f"MLflow Tracking URI: {tracking_uri}")
    print(f"Database path: {db_path}")
    
    if not db_path.exists():
        db_path.touch()
        print("Создана новая база данных MLflow SQLite")
    else:
        print("База данных уже существует")
    
    return tracking_uri

if __name__ == "__main__":
    setup_mlflow_database()