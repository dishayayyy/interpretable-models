# Bike Sharing Demand Prediction

This project compares Linear Regression, Decision Tree, and RuleFit models to predict daily bike rental demand using the UCI Bike Sharing dataset.

---

## Dataset

File: `day.csv`
Observations after preprocessing: 729

Target variable:
`cnt` → total daily rentals

Predictors:

* season
* yr
* mnth
* holiday
* weekday
* workingday
* weathersit
* temp
* atemp
* hum
* windspeed
* lag feature `cnt_2d_bfr`

The lag feature was created using `shift(2)` to model temporal persistence.

---

## Preprocessing

* Sorted dataset chronologically
* Created lag feature with shift(2)
* Removed missing rows
* Dropped leakage variables
* Applied one hot encoding for OLS
* Added intercept for linear regression

---

## Model 1: Linear Regression

Method: Ordinary Least Squares

R² = 0.756
Adjusted R² = 0.753

Observations:

* Lag demand is the strongest predictor
* Temperature has a positive and significant effect
* Weather condition and humidity significantly impact rentals

The model captures global linear trends but does not model nonlinear interactions.

---

## Model 2: Decision Tree Regressor

Parameters:
max_depth = 5
min_samples_leaf = 10

R² without lag ≈ 0.53
R² with lag ≈ 0.67

Observations:

* `cnt_2d_bfr` dominates importance
* Temperature is second most important
* Captures nonlinear relationships

Performance improves with lag but remains below linear regression.

---

## Model 3: RuleFit

Tree generator: Gradient Boosting
n_estimators = 300
max_depth = 3
L1 regularization enabled by default

1289 candidate rules generated
273 retained after L1 regularization

Performance:

R² = 0.98
RMSE ≈ 270

Top features by importance:

1. temp
2. hum
3. atemp
4. windspeed
5. yr

Interpretation:

* Temperature is the dominant driver of bike demand
* Humidity and perceived temperature strongly influence usage
* Wind speed negatively affects rentals
* Temporal persistence contributes but is secondary to weather

RuleFit achieves high predictive accuracy while maintaining interpretability through sparse rule selection.

---

## Model Comparison

| Model             | R²    |
| ----------------- | ----- |
| Linear Regression | 0.756 |
| Decision Tree     | 0.671 |
| RuleFit           | 0.980 |

RuleFit substantially outperforms the other models on training performance.

---

## Key Insights

* Weather conditions are the primary drivers of demand
* Temperature is consistently the strongest predictor
* Including lag features improves all models
* Hybrid rule based modeling captures nonlinear structure effectively
