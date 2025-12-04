

experiment:
  name: "california_housing"
  tracking_uri: "./mlruns"  

datasets:
  v1_path: "data/processed/housing_processed_v1.csv"
  v2_path: "data/processed/housing_processed_v2.csv"

models:

  linear_regression:
    name: "LinearRegression"
    params:
      fit_intercept: true
      normalize: false
  
  random_forest:
    name: "RandomForestRegressor"
    params_grid:
      n_estimators: [50, 100, 200]
      max_depth: [5, 10, 20]
      min_samples_split: [2, 5, 10]
  
  xgboost:
    name: "XGBRegressor"
    params_grid:
      n_estimators: [50, 100]
      max_depth: [3, 6, 9]
      learning_rate: [0.01, 0.1, 0.3]

evaluation:
  metrics: ["mse", "mae", "r2"]
  target_column: "median_house_value"
  log_target_column: "median_house_value_log"