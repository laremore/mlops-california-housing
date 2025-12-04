import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path

def test_data_files_exist():
    project_root = Path(__file__).parent.parent
    
    raw_data = project_root / "data" / "raw" / "housing.csv"
    assert raw_data.exists(), f"Raw data file not found: {raw_data}"
    
    v1_data = project_root / "data" / "processed" / "housing_processed_v1.csv"
    v2_data = project_root / "data" / "processed" / "housing_processed_v2.csv"
    
    assert v1_data.exists() or v2_data.exists(), "Processed data files not found"

def test_v1_preprocessing():
    project_root = Path(__file__).parent.parent
    v1_data = project_root / "data" / "processed" / "housing_processed_v1.csv"
   
    if v1_data.exists():
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
    
    if v2_data.exists():
        df = pd.read_csv(v2_data)
        
        assert df.isnull().sum().sum() == 0, "V2 data contains NaN values"
        
        assert 'median_house_value_log' in df.columns, "Log target column missing in v2"
        
        new_features = ['bedrooms_per_room', 'population_per_household', 
                       'households_per_population']
        
        for feature in new_features:
            if feature in df.columns:
                assert df[feature].min() >= 0, f"Negative values in {feature}"