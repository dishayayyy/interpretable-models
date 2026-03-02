# Machine Learning Models: Regression and Classification

This repository implements multiple interpretable machine learning models across regression and classification tasks.

* Regression task: Bike Sharing Demand Prediction
* Classification task: Breast Cancer Diagnosis

---

# Part 1: Bike Sharing Demand Prediction

## Dataset

UCI Bike Sharing Dataset
File: `day.csv`
Observations after preprocessing: 729

Target:
`cnt` → total daily rentals

Features include:

* season, yr, mnth
* holiday, weekday, workingday
* weathersit
* temp, atemp, hum, windspeed
* lag feature `cnt_2d_bfr` created using `shift(2)`

---

## Preprocessing

* Chronological sorting
* Lag feature creation
* Missing row removal
* Leakage variable removal
* One hot encoding for OLS
* Intercept added for linear regression

---

## Models

### 1. Linear Regression

Method: Ordinary Least Squares

R² = 0.756
Adjusted R² = 0.753

Key insights:

* Lag demand is dominant predictor
* Temperature positively impacts rentals
* Weather and humidity significantly affect demand

---

### 2. Decision Tree Regressor

Parameters:

* max_depth = 5
* min_samples_leaf = 10

R² without lag ≈ 0.53
R² with lag ≈ 0.67

Captures nonlinear structure but underperforms OLS.

---

### 3. RuleFit

Tree generator: Gradient Boosting
n_estimators = 300
max_depth = 3
L1 regularization applied

1289 candidate rules generated
273 retained

R² = 0.98
RMSE ≈ 270

Top drivers:

1. temp
2. hum
3. atemp
4. windspeed
5. yr

RuleFit achieves highest predictive accuracy with sparse rule based interpretability.

---

## Regression Model Comparison

| Model             | R²    |
| ----------------- | ----- |
| Linear Regression | 0.756 |
| Decision Tree     | 0.671 |
| RuleFit           | 0.980 |

---

# Part 2: Breast Cancer Classification

## Dataset

Breast Cancer Wisconsin Diagnostic Dataset
Observations: 569

Target:

* 1 → Malignant
* 0 → Benign

30 numeric tumor features used.

---

## Preprocessing

* Dropped `id` and empty column
* Encoded diagnosis using `.map()`
* Train test split with `stratify=y`
* Standardized features using `StandardScaler`

---

## Logistic Regression

Library: `sklearn.linear_model.LogisticRegression`
max_iter = 5000

Model estimates:

log odds = β0 + β1x1 + ... + βnxn

Probability:

P(Y=1) = 1 / (1 + e^(-z))

---

## Performance

Training accuracy = 98.68 percent
Testing accuracy = 96.49 percent

Confusion matrix:

* TN = 71
* FP = 1
* FN = 3
* TP = 39

Precision = 97.5 percent
Recall = 92.8 percent
ROC AUC = 0.996

---

## Key Takeaways

* Weather variables dominate bike demand prediction
* Hybrid rule based modeling captures nonlinear effects effectively
* Logistic regression provides strong probabilistic interpretation
* Odds ratios enable clear feature level explanations
* Breast cancer classes are highly separable under linear decision boundary

---

# Technical Stack

* Python 3.x
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn

Core modules used:

* `LinearRegression`
* `DecisionTreeRegressor`
* `LogisticRegression`
* `StandardScaler`
* `train_test_split`
* `confusion_matrix`
* `roc_auc_score`

---

# Project Structure

```
├── data/
│   ├── day.csv
│   ├── breast_cancer.csv
│
├── notebooks/
│   ├── linear_regression.ipynb
│   ├── decision_tree.ipynb
│   ├── rulefit.ipynb
│   ├── logistic_regression.ipynb
│
├── README.md
└── requirements.txt
```

---

# Installation

Clone the repository:

```
git clone <your_repo_link>
cd <repo_name>
```

Create environment:

```
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

