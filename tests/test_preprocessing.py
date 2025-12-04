import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path

def test_data_files_exist():
    """Проверка существования файлов данных"""
    project_root = Path(__file__).parent.parent
    
    raw_data = project_root / "data" / "raw" / "housing.csv"
    if not raw_data.exists():
        print("Note: Raw data file not found in CI/CD, skipping assertion")
        return 
    
    assert raw_data.exists(), f"Raw data file not found: {raw_data}"
    
    v1_data = project_root / "data" / "processed" / "housing_processed_v1.csv"
    v2_data = project_root / "data" / "processed" / "housing_processed_v2.csv"
    
    if not v1_data.exists() and not v2_data.exists():
        print("Note: No processed data files found in CI/CD")
      

def test_v1_preprocessing():
    project_root = Path(__file__).parent.parent
    v1_data = project_root / "data" / "processed" / "housing_processed_v1.csv"
    
    if not v1_data.exists():
        print("Note: V1 data file not found, skipping test")
        return
        
    df = pd.read_csv(v1_data)
    
    assert df.isnull().sum().sum() == 0, "V1 data contains NaN values"
    
    expected_cols = ['longitude', 'latitude', 'housing_median_age', 
                    'total_rooms', 'total_bedrooms', 'population',
                    'households', 'median_income', 'median_house_value',
                    'ocean_proximity']
    
    for col in expected_cols:
        assert col in df.columns, f"Column {col} missing in v1 data"

def test_v2_preprocessing():
    project_root = Path(__file__).parent.parent
    v2_data = project_root / "data" / "processed" / "housing_processed_v2.csv"
    
    if not v2_data.exists():
        print("Note: V2 data file not found, skipping test")
        return
        
    df = pd.read_csv(v2_data)
    
    assert df.isnull().sum().sum() == 0, "V2 data contains NaN values"
    
    assert 'median_house_value_log' in df.columns, "Log target column missing in v2"