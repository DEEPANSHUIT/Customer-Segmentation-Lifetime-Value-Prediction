# 📊 Customer Lifetime Value (CLTV) Prediction Dashboard

This project presents an end-to-end pipeline to predict **Customer Lifetime Value (CLTV)** using regression models and visualize the insights using a **Streamlit dashboard**.

---

## 📌 Problem Statement

Businesses often struggle to determine how much future revenue they can expect from a customer. This project focuses on estimating the **6-month CLTV** for each customer based on historical transaction patterns and customer behavior using:

- **Linear Regression**
- **XGBoost Regressor**

---

## 📂 Dataset Description

The dataset contains the following features:

- **CustomerID**: Unique identifier for each customer.
- **Recency**: Days since the customer's last purchase.
- **Frequency**: Total number of repeat purchases.
- **Monetary**: Total amount spent by the customer.
- **Customer_Age**: Duration (in months) since the customer first appeared.

The target variable is **CLTV over the next 6 months**.

---

## 🔍 Feature Engineering

We engineered **RFM features** and calculated customer age from purchase dates. These features are critical in understanding customer behavior.

- **Recency**: How recently a customer purchased.
- **Frequency**: How often a customer purchases.
- **Monetary**: How much a customer spends.
- **Customer_Age**: How long they have been a customer.

We also performed log transformation on the CLTV values to handle skewness.

---

## ⚙️ Models Used

1. **Linear Regression**:
   - Baseline model using RFM and age features.
2. **XGBoost Regressor**:
   - Handles non-linear relationships, more robust to outliers.

---

## 📉 Results Summary

### Overall Model Performance:

| Model              | RMSE  | R² Score |
|--------------------|-------|----------|
| Linear Regression  | ~1.90 |  0.1222  |
| XGBoost Regressor  | ~2.00 |  0.0273  |

### Buyers Only Performance (CLTV > 0):

| Model              | RMSE  | R² Score |
|--------------------|-------|----------|
| Linear Regression  | ~1.04 | -0.1102  |
| XGBoost Regressor  | ~2.00 | -0.3619  |

---

## ⚠️ Why R² is Low in CLTV Prediction?

This is **common and expected** due to the nature of CLTV:

1. **Highly Skewed Data**: Most customers have low or zero CLTV, but a few have very high values.
2. **Zero Inflation**: Many customers don’t return, so the target variable has many zeros.
3. **Behavior Variability**: Customers behave unpredictably, especially over a long horizon.
4. **R² is not ideal for business value**: Even if R² is low, the model can still rank customers well, which is critical for prioritization and marketing.

---

## 📊 Streamlit Dashboard Features

- **Model selection** (Linear Regression or XGBoost)
- **Predicted vs Actual CLTV scatter plot**
- **Prediction Error Distribution**
- **RMSE and R² metrics**
- **Buyers-only performance metrics**
- **Export predictions as CSV**

---

## 💡 Future Improvements

- Use classification for purchase likelihood before CLTV regression.
- Add product category, channel, and seasonality features.
- Try models like Pareto/NBD or Gamma-Gamma (requires assumptions).

---

## 🛠 Tech Stack

- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Matplotlib, Seaborn
- Streamlit

---

## 🧾 File Structure

```
├── data/
│   └── transactions.csv
├── cltv_model.py
├── cltv_dashboard.py
├── requirements.txt
└── README.md
```

---

## 🙋‍♂️ Author

Deepanshu Khanna  
Project: CLTV Prediction and Dashboard Development