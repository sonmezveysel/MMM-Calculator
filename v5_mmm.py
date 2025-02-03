import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained XGBoost model using Booster to avoid sklearn-related errors
MODEL_PATH = "optimized_mmm_xgboost_model.json"
best_xgb_model = xgb.Booster()
best_xgb_model.load_model(MODEL_PATH)

# Load dataset
weekly_mmm = pd.read_csv("weekly_data.csv")

# Apply Adstock transformation dynamically
def adstock_transform(series, carryover_factor=0.1):
    """
    Applies Adstock transformation dynamically based on user input.
    """
    adstocked_values = []
    prev_adstock = 0.1  # Initial carryover effect is 0
    for spend in series:
        current_adstock = spend + (prev_adstock * carryover_factor)
        adstocked_values.append(current_adstock)
        prev_adstock = current_adstock  # Update carryover effect
    return adstocked_values

# Identify spend columns
spend_columns = [col for col in weekly_mmm.columns if col.endswith("_Spend")]

# Apply Adstock transformation and store new columns
for col in spend_columns:
    weekly_mmm[col + "_Adstock"] = adstock_transform(weekly_mmm[col])

# Feature columns
spend_factors = [col + "_Adstock" for col in spend_columns]
macro_factors = ["Consumer_Index", "Inflation_Rate", "Gross_Rating_Point"]

# Ensure 'CTR' column exists before applying transformations
if "CTR" in weekly_mmm.columns:
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    weekly_mmm["CTR_Scaled"] = scaler.fit_transform(weekly_mmm[["CTR"]])
else:
    weekly_mmm["CTR_Scaled"] = 0  # Default value if 'CTR' is missing

# Ensure all expected feature columns exist
expected_features = spend_factors + macro_factors + ["CTR_Scaled"]
for feature in expected_features:
    if feature not in weekly_mmm.columns:
        weekly_mmm[feature] = 0  # Default value to avoid missing columns error

# Sidebar UI for What-If Scenario
st.sidebar.header("🔍 What-If Scenario Calculator")
st.sidebar.subheader("Adjust Media Spend Inputs")

# Create sliders for media spend and dynamically compute Adstock values
input_values = {}
adstock_values = {}
for col in spend_columns:
    min_val = float(weekly_mmm[col].min())
    max_val = float(weekly_mmm[col].max())
    input_values[col] = st.sidebar.slider(f"{col}", min_val, max_val, max_val * 0.8)
    
    # Compute Adstock dynamically based on user input
    adstock_values[col + "_Adstock"] = adstock_transform([input_values[col]])[0]

# Other macroeconomic factors (fixed for now)
input_values["Consumer_Index"] = weekly_mmm["Consumer_Index"].mean()
input_values["Inflation_Rate"] = weekly_mmm["Inflation_Rate"].mean()
input_values["Gross_Rating_Point"] = weekly_mmm["Gross_Rating_Point"].mean()
input_values["CTR_Scaled"] = weekly_mmm["CTR_Scaled"].mean()

# Create input DataFrame with dynamic Adstock values
input_data = {**adstock_values, **{factor: input_values[factor] for factor in macro_factors + ["CTR_Scaled"]}}
input_df = pd.DataFrame([input_data])

# Ensure column ordering matches the model's expected features
input_df = input_df[expected_features]

# Convert input data into DMatrix for XGBoost Booster model
input_dmatrix = xgb.DMatrix(input_df, feature_names=expected_features)

# Predict revenue dynamically
predicted_revenue = best_xgb_model.predict(input_dmatrix)[0]

st.sidebar.subheader("📊 Predicted Revenue")
st.sidebar.write(f"**${predicted_revenue:,.2f}**")

# Feature Importance Plot
st.subheader("📈 Feature Importance")
feature_importance_df = pd.DataFrame(best_xgb_model.get_score(importance_type='weight').items(), columns=["Feature", "Importance"])
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df, palette="viridis", ax=ax1)
ax1.set_title("Feature Importance in Revenue Prediction")
st.pyplot(fig1)

# Actual vs Predicted Revenue Trend
st.subheader("📊 Actual vs. Predicted Revenue Trend")
existing_columns = weekly_mmm.columns.intersection(expected_features)
weekly_dmatrix = xgb.DMatrix(weekly_mmm[existing_columns], feature_names=expected_features)
weekly_mmm["Predicted_Revenue"] = best_xgb_model.predict(weekly_dmatrix)
weekly_mmm["Error"] = np.abs(weekly_mmm["revenue"] - weekly_mmm["Predicted_Revenue"])

fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.lineplot(x=weekly_mmm.index, y=weekly_mmm["revenue"], label="Actual Revenue", color="blue", ax=ax2)
sns.lineplot(x=weekly_mmm.index, y=weekly_mmm["Predicted_Revenue"], label="Predicted Revenue", color="red", ax=ax2)
ax2.fill_between(weekly_mmm.index, weekly_mmm["Predicted_Revenue"] - weekly_mmm["Error"], 
                 weekly_mmm["Predicted_Revenue"] + weekly_mmm["Error"], color='red', alpha=0.2)
ax2.set_xlabel("Week")
ax2.set_ylabel("Revenue")
ax2.set_title("Predicted vs Actual Revenue Trend")
st.pyplot(fig2)

st.write("🚀 Adjust media spend and observe how the revenue changes dynamically!")