# Bike Sharing Demand Prediction

Linear Regression and Decision Tree 

## Dataset

Source: UCI Machine Learning Repository
File used: `day.csv`
Processed dataset: 728 daily observations

Target variable: `cnt`

Features used:

* season
* holiday
* workingday
* weathersit
* temp
* hum
* windspeed
* lag feature `cnt_2d_bfr`

---

## Preprocessing

* Selected relevant variables
* Created lag feature using `shift(2)`
* Removed missing rows
* Applied one hot encoding for linear regression
* Added intercept for OLS

The lag feature was introduced to capture temporal dependency.

---

## Model 1: Linear Regression

Method: Ordinary Least Squares
Library: `statsmodels`

Results:

* R² = 0.756
* Adjusted R² = 0.753

Key insight:

Lag demand and temperature are the strongest predictors. Weather and humidity significantly influence rentals.

---

## Model 2: Decision Tree Regressor

Library: `scikit-learn`
Parameters: `max_depth=5`, `min_samples_leaf=10`

Results:

* R² without lag ≈ 0.53
* R² with lag ≈ 0.67

Feature importance:

* `cnt_2d_bfr` dominant
* temperature second
* humidity and windspeed moderate

The tree captures nonlinear relationships but underperforms compared to linear regression.

---

## Model Comparison

| Model             | R²    |
| ----------------- | ----- |
| Linear Regression | 0.756 |
| Decision Tree     | 0.671 |

Linear regression performs better on this dataset. The decision tree provides rule based interpretability.

---

## Conclusion

Bike rental demand shows strong temporal persistence. Including lag features significantly improves performance.

Linear regression achieved higher predictive accuracy. Decision trees provided interpretable rule based structure.

---
