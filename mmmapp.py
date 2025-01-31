import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Set Streamlit page config
st.set_page_config(page_title="MMM Budget Simulator", layout="wide")

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
    st.title("📊 Media Mix Modeling - Budget Simulator")

    # User Inputs for Monthly Budgets
    offline_budget = st.number_input("💰 Offline Budget (TV, Radio, OOH)", min_value=1000, value=50000, step=1000)
    online_budget = st.number_input("💻 Online Budget (Search, Display, Video, Digital Audio, Social)", min_value=1000, value=70000, step=1000)

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

    # Save the randomized dataset
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

    # Load distributed data
    weekly_data = st.session_state.weekly_data

    # Preserve Week column before modifying features
    weekly_data["Week"] = range(1, len(weekly_data) + 1)  # Recreate Week column if missing

    # Define Feature Columns
    spend_factors = [
        "Search_Spend_Adstock", "Display_Spend_Adstock", "Video_Spend_Adstock",
        "Social_Spend_Adstock", "Digital_Audio_Spend_Adstock",
        "TV_Spend_Adstock", "Radio_Spend_Adstock", "OOH_Spend_Adstock"
    ]

    # 🚨 Add missing macroeconomic features (since new budgets lack them)
    macro_factors = ["Promo_Exists", "Consumer_Index", "Inflation_Rate", "Gross_Rating_Point", "CTR_Scaled"]

    # Set default values for missing macro features
    for col in macro_factors:
        weekly_data[col] = 0  # Use default value (adjust if necessary)

    # Ensure feature order matches trained model
    required_features = spend_factors + macro_factors
    model_input_data = weekly_data[required_features]  # ✅ Keep only necessary features

    # ✅ Convert to XGBoost DMatrix format
    X_test_dmatrix = xgb.DMatrix(model_input_data)

    # ✅ Make predictions using pre-trained model
    weekly_data["Predicted_Revenue"] = mmm_model.predict(X_test_dmatrix)

    # 🚨 Remove `Predicted_Revenue` before future predictions
    future_weeks = 12
    future_data = model_input_data.iloc[-1:].copy()  # Use last row as future budget allocation

    future_predictions = []
    for _ in range(future_weeks):
        X_future_dmatrix = xgb.DMatrix(future_data)
        predicted_revenue = mmm_model.predict(X_future_dmatrix)[0]
        future_predictions.append(predicted_revenue)

    future_weeks_df = pd.DataFrame({
        "Week": range(53, 53 + future_weeks),
        "Predicted_Revenue": future_predictions
    })

    # ✅ Combine past and future predictions while keeping Week column
    combined_predictions = pd.concat([
        weekly_data[["Week", "Predicted_Revenue"]].reset_index(drop=True),
        future_weeks_df
    ], ignore_index=True)

    # 🚨 FIXED: Get feature importance from Booster model
    feature_importance_dict = mmm_model.get_score(importance_type="weight")

    # Ensure all spend factors exist in feature importance, default to 0 if missing
    feature_importance_df = pd.DataFrame({
        "Feature": spend_factors,
        "Importance": [feature_importance_dict.get(feature, 0) for feature in spend_factors]  # Default to 0 if missing
    }).sort_values(by="Importance", ascending=False)

    # ✅ ROI per Channel
    roi_per_channel = {
        channel: (feature_importance_dict.get(channel, 0) * weekly_data["Predicted_Revenue"].sum()) / (weekly_data[channel].sum())
        for channel in spend_factors
    }

    roi_df = pd.DataFrame(list(roi_per_channel.items()), columns=["Media Channel", "ROI"])

    # ----------------------------
    # 📊 PLOTTING RESULTS
    # ----------------------------

    # Feature Importance Plot
    st.subheader("📌 Media Channel Impact")
    fig1, ax1 = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=feature_importance_df, ax=ax1, palette="viridis")
    ax1.set_title("Feature Importance")
    st.pyplot(fig1)

    # Predicted Revenue Over Time
    st.subheader("📈 Predicted Revenue Trend (Past & Future)")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=combined_predictions["Week"], y=combined_predictions["Predicted_Revenue"], label="Predicted Revenue", ax=ax2, color="red")
    ax2.set_title("Predicted Revenue Trend")
    plt.xticks(rotation=60)
    st.pyplot(fig2)

    # ROI per Channel
    st.subheader("💰 ROI per Media Channel")
    fig3, ax3 = plt.subplots()
    sns.barplot(x="ROI", y="Media Channel", data=roi_df, ax=ax3, palette="coolwarm")
    ax3.set_title("Return on Investment (ROI)")
    st.pyplot(fig3)

# ------------------------------------------
# 🚀 STREAMLIT APP NAVIGATION
# ------------------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", ["Budget Input", "Forecast"])

if page == "Budget Input":
    page_budget_input()
else:
    page_forecast()