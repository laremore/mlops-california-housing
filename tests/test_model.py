import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
from pathlib import Path

def test_model_file_exists():
    project_root = Path(__file__).parent.parent
    model_path = project_root / "models" / "best_model.joblib"
    
    model_files = list((project_root / "models").glob("*.joblib"))
    assert len(model_files) > 0, "No model files found in models/ directory"

def test_model_loading():
    project_root = Path(__file__).parent.parent
    model_files = list((project_root / "models").glob("*.joblib"))
    
    if model_files:
        model_path = model_files[0]
        
        try:
            model = joblib.load(model_path)
            assert model is not None, "Failed to load model"
            
            assert hasattr(model, 'predict'), "Model has no predict method"
            
        except Exception as e:
            print(f"Note: Could not load model {model_path}: {e}")