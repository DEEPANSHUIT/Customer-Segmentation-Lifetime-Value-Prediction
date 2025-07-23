import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title and Description
st.title("Customer Lifetime Value (CLTV) Prediction Dashboard")
st.markdown("""
This dashboard presents CLTV predictions using:
- **Linear Regression**
- **XGBoost Regressor**

We use RFM (Recency, Frequency, Monetary) and Customer Age features to predict customer value over the next 6 months.
""")

# Load the results (use your file path or session state)
@st.cache_data
def load_data():
    return pd.read_csv("cltv_predictions.csv")  # This should contain Actual_CLTV, Predicted_CLTV, etc.

data = load_data()

# Display raw data
if st.checkbox("Show raw data"):
    st.write(data.head())

# Select Model
model_choice = st.selectbox("Select Prediction Model", ["Linear Regression", "XGBoost"])
pred_column = "LR_Predicted_CLTV" if model_choice == "Linear Regression" else "XGB_Predicted_CLTV"

# Scatter Plot of Actual vs Predicted
st.subheader(f"{model_choice}: Actual vs Predicted CLTV")
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(x=data['Actual_CLTV'], y=data[pred_column], alpha=0.6)
ax.set_xlabel("Actual CLTV")
ax.set_ylabel(f"Predicted CLTV ({model_choice})")
ax.set_title("Actual vs Predicted CLTV")
st.pyplot(fig)

# Error Distribution
st.subheader("Prediction Error Distribution")
data['Error'] = data[pred_column] - data['Actual_CLTV']
fig2, ax2 = plt.subplots(figsize=(8, 4))
sns.histplot(data['Error'], bins=50, kde=True, color='orange')
ax2.set_title("Error Distribution")
ax2.set_xlabel("Prediction Error")
st.pyplot(fig2)

# Summary Stats
st.subheader("Model Summary")
rmse = ((data['Error'] ** 2).mean()) ** 0.5
r2 = 1 - sum((data[pred_column] - data['Actual_CLTV'])**2) / sum((data['Actual_CLTV'] - data['Actual_CLTV'].mean())**2)

st.write(f"**RMSE**: {rmse:.2f}")
st.write(f"**R² Score**: {r2:.4f}")

# Filter Buyers Only
if st.checkbox("Show Only Buyers (Actual CLTV > 0)"):
    buyers = data[data['Actual_CLTV'] > 0]
    st.write(buyers[["Recency", "Frequency", "Monetary", "Actual_CLTV", pred_column]].head())

    st.markdown("### Buyer Prediction Distribution")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.histplot(buyers[pred_column], bins=40, kde=True, color='green')
    ax3.set_title("Predicted CLTV Distribution for Buyers")
    ax3.set_xlabel("Predicted CLTV")
    st.pyplot(fig3)

    st.write("\n")
    buyer_rmse = ((buyers[pred_column] - buyers['Actual_CLTV'])**2).mean()**0.5
    buyer_r2 = 1 - sum((buyers[pred_column] - buyers['Actual_CLTV'])**2) / sum((buyers['Actual_CLTV'] - buyers['Actual_CLTV'].mean())**2)
    st.write(f"**Buyers Only RMSE**: {buyer_rmse:.2f}")
    st.write(f"**Buyers Only R² Score**: {buyer_r2:.4f}")

# Footer
st.markdown("---")
st.caption("Developed for Customer Segmentation & CLTV Prediction Analysis")
