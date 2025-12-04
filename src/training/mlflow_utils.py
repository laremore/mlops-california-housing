import os
from pathlib import Path
import mlflow

PROJECT_ROOT = Path(__file__).parent.parent.parent

def setup_mlflow():

    mlflow_dir = PROJECT_ROOT / "mlflow_data"
    mlflow_dir.mkdir(exist_ok=True)
    
    db_path = mlflow_dir / "mlflow.db"
    
    tracking_uri = f"sqlite:///{db_path.absolute()}"
    
    print(f"MLflow tracking URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    
    return tracking_uri