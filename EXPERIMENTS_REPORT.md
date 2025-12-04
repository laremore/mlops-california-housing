# ML Experiments Report

## Summary
- **Total Experiments**: 3
- **Best Model**: XGBoost (R² = 0.8028)
- **Generated**: 2025-12-03 22:07:59

## Experiment 1: Linear Regression
- **Date**: 2025-12-03
- **Data Version**: v1 (basic preprocessing)
- **Metrics**:
  - MSE: 5,019,864,241.06
  - MAE: 51,665.94
  - R²: 0.6169

## Experiment 2: Random Forest with GridSearch
- **Date**: 2025-12-03
- **Data Version**: v1 (basic preprocessing)
- **Best Parameters**: max_depth=None, min_samples_split=2, n_estimators=100
- **Metrics**:
  - MSE (log scale): 2,617,003,648.86
  - MAE (log scale): 33,526.89
  - R² (log scale): 0.8003

## Experiment 3: XGBoost
- **Date**: 2025-12-03
- **Data Version**: v2 (advanced preprocessing with feature engineering)
- **Parameters**: n_estimators=100, max_depth=6, learning_rate=0.1
- **Metrics**:
  - MSE (log): 0.0552
  - MAE (log): 0.1626
  - R² (log): 0.8298
  - R² (original scale): 0.8028

## Conclusion
The XGBoost model trained on v2 data performs best with R² = 0.8028.
