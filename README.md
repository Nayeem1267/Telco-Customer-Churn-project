# Telco Customer Churn Prediction Pipeline

An end-to-end Machine Learning project engineered to identify at-risk telecommunications subscribers and predict customer churn using demographics, signed services, and account analytics.

## 📌 Project Overview
Customer churn is one of the most critical metrics for subscription-based businesses. This project builds and optimizes a predictive pipeline using a **Random Forest Classifier** trained on customer account data. To combat the severe class imbalance typical of churn data, the final architecture leverages **SMOTE** (Synthetic Minority Over-sampling Technique) paired with tuned hyperparameter constraints to maximize recall for churned subscribers.

## 📊 Dataset Profile
* **Total Instances:** 7,043 customer accounts
* **Target Feature:** `Churn` (Binary classification: `1` for Churn, `0` for Retained)
* **Data Fields Included:** Demographics (Gender, SeniorCitizen, Partner, Dependents), Services (InternetService, OnlineSecurity, TechSupport, StreamingTV, etc.), and Financials (MonthlyCharges, TotalCharges, Tenure).

---

## 🛠️ Pipeline Architecture & Workflow

### 1. Data Cleaning & Feature Engineering
* **Dimensionality Reduction:** Dropped arbitrary identifiers (`customerID`) to prevent overfitting.
* **Target Encoding:** Mapped categorical text labels (`Yes`/`No`) into clear machine-readable binary values (`1`/`0`).
* **Categorical Encoding:** Applied automated One-Hot Encoding (`pd.get_dummies`) to translate diverse service matrices into sparse numeric representations while maintaining feature parity.

### 2. Class Imbalance Mitigation (SMOTE)
Standard classifiers often default to predicting the majority class ("Retained") due to skewness in data distribution. We introduced an **imblearn Pipeline** that isolates over-sampling to cross-validation training folds, generating synthetic minority rows without leaking data evaluation parameters into test sets.

### 3. Final Production Architecture
The optimal model parameters utilized for production stability include:
* **Estimators:** 100 decision trees
* **Max Depth:** 10 (Restricted to control model drift and over-adaptation)
* **Minimum Split Constraint:** 5 samples per internal node split
* **Class Balancing weight:** Uniform balance distribution array mapping

---

## 📈 Final Model Metrics Summary

| Evaluation Metric | Class 0 (Retained) | Class 1 (Churned) | Overall Model Performance |
| :--- | :--- | :--- | :--- |
| **Precision** | High | Balanced | *Reflects precise classification behaviors* |
| **Recall (Sensitivity)** | Optimized | Highly Sensitive | *Maximizes early detection flags* |
| **Accuracy** | — | — | **~78% - 82%** (Depending on validation fold splits) |

> **Business Impact:** The optimized model prioritizes **Recall for Churn (Class 1)**. In production, catching an at-risk customer early (preventing a False Negative) is substantially more valuable to retention marketing campaigns than avoiding a False Positive flag on a stable subscriber.

---

## 💾 Deployment & Usage

The entire operational asset group—including the SMOTE configuration transformers and the underlying tuned random forest state matrix—is encapsulated into a single operational object block file.

### Reloading the saved Model for Predictions:
```python
import joblib
import pandas as pd

# Load the deployment pipeline configuration
production_pipeline = joblib.load('optimized_telco_churn_pipeline.pkl')

# Feed in a preprocessed incoming DataFrame (must match column alignment arrays of X_test)
# new_predictions = production_pipeline.predict(new_customer_data)
# churn_probabilities = production_pipeline.predict_proba(new_customer_data)[:, 1]
```
