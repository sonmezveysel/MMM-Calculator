import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Set Streamlit page config
st.set_page_config(page_title="Media Mix Modeling: Impact Calculator", layout="wide")

# Load Your Pre-Trained MMM Model
@st.cache_resource
def load_trained_model():
    model = xgb.Booster()
    model.load_model("optimized_mmm_xgboost_model.json")
    return model

mmm_model = load_trained_model()

# ------------------------------------------
# 🎯 PAGE 1: Budget Input & Randomization
# ------------------------------------------
def page_budget_input():
    st.title("📊 Media Mix Modeling: Impact Calculator")

    # User Inputs for Monthly Budgets using Sliders
    offline_budget = st.slider("💰 Offline Budget (TV, Radio, OOH)", min_value=100000, max_value=10000000, value=5000000, step=100000)
    online_budget = st.slider("💻 Online Budget (Search, Display, Video, Digital Audio, Social)", min_value=100000, max_value=10000000, value=7000000, step=100000)

    # Generate 52-week synthetic data by randomizing budget distribution
    np.random.seed(42)  # Ensures reproducibility
    weeks = 52

    weekly_data = pd.DataFrame({
        "Week": range(1, weeks + 1),
        "TV_Spend_Adstock": np.random.dirichlet(np.ones(weeks), size=1)[0] * offline_budget,
        "Radio_Spend_Adstock": np.random.dirichlet(np.ones(weeks), size=1)[0] * offline_budget,
        "OOH_Spend_Adstock": np.random.dirichlet(np.ones(weeks), size=1)[0] * offline_budget,
        "Search_Spend_Adstock": np.random.dirichlet(np.ones(weeks), size=1)[0] * online_budget,
        "Display_Spend_Adstock": np.random.dirichlet(np.ones(weeks), size=1)[0] * online_budget,
        "Video_Spend_Adstock": np.random.dirichlet(np.ones(weeks), size=1)[0] * online_budget,
        "Social_Spend_Adstock": np.random.dirichlet(np.ones(weeks), size=1)[0] * online_budget,
        "Digital_Audio_Spend_Adstock": np.random.dirichlet(np.ones(weeks), size=1)[0] * online_budget
    })

    st.session_state.weekly_data = weekly_data
    st.success("✅ Budget Distributed! Move to Page 2 for Model Predictions.")

# ------------------------------------------
# 🎯 PAGE 2: Predict Revenue Using Pre-Trained MMM Model
# ------------------------------------------
def page_forecast():
    st.title("📈 MMM Model Predictions & Analysis")

    if "weekly_data" not in st.session_state:
        st.warning("⚠️ Please go back to Page 1 and enter your budget first!")
        return

    weekly_data = st.session_state.weekly_data
    weekly_data["Week"] = range(1, len(weekly_data) + 1)

    spend_factors = [
        "Search_Spend_Adstock", "Display_Spend_Adstock", "Video_Spend_Adstock",
        "Social_Spend_Adstock", "Digital_Audio_Spend_Adstock",
        "TV_Spend_Adstock", "Radio_Spend_Adstock", "OOH_Spend_Adstock"
    ]
    
    macro_factors = ["Promo_Exists", "Consumer_Index", "Inflation_Rate", "Gross_Rating_Point", "CTR_Scaled"]
    for col in macro_factors:
        weekly_data[col] = 0
    
    required_features = spend_factors + macro_factors
    model_input_data = weekly_data[required_features]
    
    X_test_dmatrix = xgb.DMatrix(model_input_data)
    weekly_data["Predicted_Revenue"] = mmm_model.predict(X_test_dmatrix)

    # Normalize values between 0 and 100
    scaler = MinMaxScaler(feature_range=(0, 100))
    weekly_data[["Predicted_Revenue"]] = scaler.fit_transform(weekly_data[["Predicted_Revenue"]])
    
    future_weeks = 12
    future_data = model_input_data.iloc[-1:].copy()
    future_predictions = []
    
    for _ in range(future_weeks):
        X_future_dmatrix = xgb.DMatrix(future_data)
        predicted_revenue = mmm_model.predict(X_future_dmatrix)[0]
        future_predictions.append(predicted_revenue)
    
    future_predictions = scaler.transform(np.array(future_predictions).reshape(-1, 1)).flatten()
    
    future_weeks_df = pd.DataFrame({
        "Week": range(53, 53 + future_weeks),
        "Predicted_Revenue": future_predictions
    })
    
    combined_predictions = pd.concat([
        weekly_data[["Week", "Predicted_Revenue"]].reset_index(drop=True),
        future_weeks_df
    ], ignore_index=True)
    
    feature_importance_dict = mmm_model.get_score(importance_type="weight")
    feature_importance_df = pd.DataFrame({
        "Feature": spend_factors,
        "Importance": [feature_importance_dict.get(feature, 0) for feature in spend_factors]
    })
    feature_importance_df["Importance"] = scaler.fit_transform(feature_importance_df[["Importance"]])
    
    roi_per_channel = {
        channel: (feature_importance_dict.get(channel, 0) * weekly_data["Predicted_Revenue"].sum()) / (weekly_data[channel].sum())
        for channel in spend_factors
    }
    roi_df = pd.DataFrame(list(roi_per_channel.items()), columns=["Media Channel", "ROI"])
    
    st.subheader("📌 Media Channel Impact")
    fig1, ax1 = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=feature_importance_df, ax=ax1)
    ax1.set_title("Feature Importance (0-100 Scale)")
    st.pyplot(fig1)
    
    st.subheader("📈 Predicted Revenue for Next 12 Weeks")
    fig2, ax2 = plt.subplots()
    ax2.bar(future_weeks_df["Week"], future_weeks_df["Predicted_Revenue"], yerr=5)
    ax2.set_title("Predicted Revenue (0-100 Scale)")
    st.pyplot(fig2)
    
    st.subheader("💰 ROI per Media Channel")
    fig3, ax3 = plt.subplots()
    sns.barplot(x="ROI", y="Media Channel", data=roi_df, ax=ax3)
    ax3.set_title("Revenue per Dollar Spent")
    st.pyplot(fig3)

st.sidebar.title("📌 Menu")
page = st.sidebar.radio("Go to:", ["Budget Input", "Forecast"])
if page == "Budget Input":
    page_budget_input()
else:
    page_forecast()
