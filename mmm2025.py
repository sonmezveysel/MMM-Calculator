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
    try:
        model.load_model("optimized_mmm_xgboost_model.json")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    return model

mmm_model = load_trained_model()

# PAGE 1: Budget Input & Randomizationstrea
def page_budget_input():
    st.title("📊 Media Mix Modeling - Impact Calculator")
    offline_budget = st.slider("💰 Offline Budget (TV, Radio, OOH)", 100000, 10000000, 5000000, 50000)
    online_budget = st.slider("💻 Online Budget", 100000, 10000000, 5000000, 50000)
    search_budget_pct = st.slider("🔍 % of Online Budget to Search", 0, 100, 50)
    tv_budget_pct = st.slider("📺 % of Offline Budget to TV", 0, 100, 60)

    def allocate_remaining_budget(total_budget, fixed_allocation, num_channels):
        remaining_budget = total_budget - fixed_allocation
        return np.random.dirichlet(np.ones(num_channels)) * remaining_budget

    np.random.seed(42)  # Reproducibility
    weeks = 52
    tv_spend = (tv_budget_pct / 100) * offline_budget
    search_spend = (search_budget_pct / 100) * online_budget

    # Allocate Remaining Budgets
    remaining_offline_allocation = allocate_remaining_budget(offline_budget, tv_spend, 2)
    remaining_online_allocation = allocate_remaining_budget(online_budget, search_spend, 4)

    offline_allocation = np.array([tv_spend] + list(remaining_offline_allocation))
    online_allocation = np.array([search_spend] + list(remaining_online_allocation))

    weekly_data = pd.DataFrame({
        "Week": range(1, weeks + 1),
        "TV_Spend_Adstock": np.random.dirichlet(np.ones(weeks), size=1)[0] * offline_allocation[0],
        "Radio_Spend_Adstock": np.random.dirichlet(np.ones(weeks), size=1)[0] * offline_allocation[1],
        "OOH_Spend_Adstock": np.random.dirichlet(np.ones(weeks), size=1)[0] * offline_allocation[2],
        "Search_Spend_Adstock": np.random.dirichlet(np.ones(weeks), size=1)[0] * online_allocation[0],
        "Social_Spend_Adstock": np.random.dirichlet(np.ones(weeks), size=1)[0] * online_allocation[1],
        "Display_Spend_Adstock": np.random.dirichlet(np.ones(weeks), size=1)[0] * online_allocation[2],
        "Video_Spend_Adstock": np.random.dirichlet(np.ones(weeks), size=1)[0] * online_allocation[3],
        "Digital_Audio_Spend_Adstock": np.random.dirichlet(np.ones(weeks), size=1)[0] * online_allocation[4],
    })

    st.session_state.weekly_data = weekly_data
    st.success("✅ Budget Distributed! Move to 'What If' Tab for Predictions.")

# PAGE 2: Forecast Revenue Using MMM Model
def page_forecast():
    st.title("📈 MMM Model Predictions & Analysis")
    
    if "weekly_data" not in st.session_state:
        st.warning("⚠️ Please enter your budget first in the 'Input' Tab!")
        return

    weekly_data = st.session_state.weekly_data

    spend_factors = [
        "Search_Spend_Adstock", "Display_Spend_Adstock", "Video_Spend_Adstock",
        "Social_Spend_Adstock", "Digital_Audio_Spend_Adstock",
        "TV_Spend_Adstock", "Radio_Spend_Adstock", "OOH_Spend_Adstock"
    ]

    # Generate random macroeconomic factors
    weekly_data["Inflation_Rate"] = np.random.uniform(2, 8, len(weekly_data))
    weekly_data["Consumer_Index"] = np.random.uniform(100, 120, len(weekly_data))
    weekly_data["Gross_Rating_Point"] = np.random.uniform(40, 220, len(weekly_data))
    weekly_data["CTR_Scaled"] = np.random.uniform(1, 40, len(weekly_data))

    macro_factors = ["Consumer_Index", "Inflation_Rate", "Gross_Rating_Point", "CTR_Scaled"]

    model_input_data = weekly_data[spend_factors + macro_factors]
    X_test_dmatrix = xgb.DMatrix(model_input_data)
    
    weekly_data["Predicted_Revenue"] = mmm_model.predict(X_test_dmatrix)

    # Normalize Predicted Revenue (Index 0-100)
    scaler = MinMaxScaler(feature_range=(0, 100))
    weekly_data["Predicted_Revenue"] = scaler.fit_transform(weekly_data[["Predicted_Revenue"]])

    # 📈 Revenue Trend Graph (Last 12 Weeks)
    st.subheader("📈 12 Week Predicted Revenue Trend")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.set_theme(style="darkgrid")

    sns.lineplot(
        x=weekly_data["Week"].iloc[-12:], 
        y=weekly_data["Predicted_Revenue"].iloc[-12:], 
        marker="o", linewidth=2.5, color="royalblue", ax=ax
    )

    ax.set_ylim(0, 100)
    ax.set_xlabel("Week")
    ax.set_ylabel("Revenue Index (0-100)")
    ax.set_title("Predicted Revenue Over 12 Weeks")

    st.pyplot(fig)   

    # 📊 Bar Chart for Baseline vs. Predicted Revenue
    st.subheader("📊 Baseline vs. Predicted Revenue 12 Weeks")

    baseline_revenue = weekly_data["Predicted_Revenue"].iloc[-24:-12].sum()  # Baseline from prior 12 weeks
    predicted_revenue = weekly_data["Predicted_Revenue"].iloc[-12:].sum()  # Model Prediction for last 12 weeks

    fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
    ax_bar.bar(["Baseline Revenue", "Predicted Revenue"], [baseline_revenue, predicted_revenue], color=["gray", "blue"])
    
    ax_bar.set_ylabel("Total Revenue (Indexed)")
    ax_bar.set_title("Comparison of Baseline vs. Predicted Revenue")

    # Display the bar chart
    st.pyplot(fig_bar)

# 🚀 Streamlit App Navigation with Tabs
def budget_app():
    st.title("MMM: What If Scenarios")
    tab1, tab2 = st.tabs(["📊 Input", "📈 What If"])
    
    with tab1:
        page_budget_input()
        
    with tab2:
        page_forecast()

# Run the Streamlit App
if __name__ == "__main__":
    budget_app()