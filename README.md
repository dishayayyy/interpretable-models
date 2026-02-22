# Bike Sharing Demand Prediction using Linear Regression

## Project Overview

You build a linear regression model to predict daily bike rentals using weather and calendar features.

You use the UCI Bike Sharing dataset and apply statistical modeling with `statsmodels`.

---

## Objective

Predict the daily count of rented bikes using:

* Season
* Holiday indicator
* Workday indicator
* Weather condition
* Temperature
* Humidity
* Wind speed
* Lag feature from two days before

---

## Dataset

Source: UCI Machine Learning Repository
Dataset: Bike Sharing Dataset

File used: `day.csv`

The dataset contains 728 processed daily observations after cleaning.

---

## Data Preprocessing

You performed the following steps:

* Selected relevant features
* Mapped season and weather codes to readable labels
* Renamed columns for clarity
* Created lag feature `cnt_2d_bfr` using `shift(2)`
* Removed rows with missing values
* Removed unrealistic humidity values
* Applied one hot encoding to categorical variables
* Added intercept term for regression

---

## Model

Model used: Ordinary Least Squares Linear Regression

Library: `statsmodels`

Regression equation:

cnt = β0 + β1X1 + β2X2 + ... + ε

---

## Results

* R² = 0.756
* Adjusted R² = 0.753
* Model statistically significant

Key findings:

* Temperature strongly increases bike rentals
* Wind speed strongly decreases rentals
* Humidity negatively affects rentals
* Workdays increase rentals
* Lag feature is the strongest predictor
* Weather conditions significantly impact demand

---

## Model Diagnostics

You evaluated:

* Actual vs Predicted plot
* Residual plot
* Residual distribution
* Feature weight visualization

Findings:

* Strong linear relationship
* Residuals centered around zero
* Slight heteroscedasticity at high rental values
* Assumptions reasonably satisfied

---

## How to Run

1. Install dependencies

```
pip install -r requirements.txt
```

2. Open the notebook

```
bike_sharing_linear_regression.ipynb
```

3. Run all cells

---

## Requirements

* pandas
* numpy
* matplotlib
* statsmodels

---

## Conclusion

The linear regression model explains a substantial portion of variability in daily bike rentals.

Weather, temperature, and temporal dependency play major roles in demand.

