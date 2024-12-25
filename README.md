# Game-Search-Volume-Prediction-Machine-Learning-Models-and-Forecasting

![steam vr](https://github.com/user-attachments/assets/9d3113c8-ad73-473b-bf2d-ef7fe2beb02b)

This repository provides an analysis and implementation of machine learning models to predict search volumes for various games. The models evaluated include Random Forest, XGBoost, and LightGBM, with performance metrics including Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). Hyperparameter tuning is also performed for the LightGBM model using GridSearchCV to optimize its performance.

## Models and Performance

## 1. Random Forest Model
- MAE (Mean Absolute Error): 0.21
- RMSE (Root Mean Squared Error): 1.96
The Random Forest model delivers the most accurate predictions, with a low MAE and RMSE. This model is effective for this dataset, providing reliable and precise forecasts.

## 2. XGBoost Model
- MAE: 3.93
- RMSE: 24.79
The XGBoost model has higher error rates, with a significantly larger MAE and RMSE. It performs poorly with outliers and might require further data refinement or hyperparameter adjustments to improve its accuracy.

## 3. LightGBM Model
- MAE: 2.93
- RMSE: 16.34
The LightGBM model performs moderately well, offering a middle ground between Random Forest and XGBoost. While its performance is not as strong as Random Forest, it is more accurate than XGBoost and can be improved further.

## 4. Hyperparameter Tuning with GridSearchCV (LightGBM)
Best Parameters:
- learning_rate: 0.1
- n_estimators: 200
- num_leaves: 31
Best CV Score (MSE): 296.56 (Negative, indicating low error)
GridSearchCV was used to tune the hyperparameters of LightGBM, resulting in improved prediction accuracy. The optimized parameters helped reduce error rates and enhanced the model's performance.

## Conclusion

**Best Model:** Random Forest provides the best results with the lowest error rates (MAE: 0.21, RMSE: 1.96). It is the recommended model for predicting game search volumes.
**Improvement Areas:** The XGBoost model requires optimization, either through hyperparameter tuning or additional data, to reduce its error rates.
**LightGBM:** Though it performs better than XGBoost, it still lags behind Random Forest. Further fine-tuning of the model could lead to better results.

## Future Work
XGBoost: Further hyperparameter tuning and feature engineering can help improve its performance.
Data Expansion: Additional data, such as user ratings or reviews, could improve model predictions.
Additional Optimization: Experiment with different algorithms and more hyperparameter configurations to achieve even better results.


License
This project is licensed under the MIT License - see the LICENSE file for details.
