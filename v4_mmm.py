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
    st.title("📊 Media Mix Modeling - Impact Calculator")

    # User Inputs for Monthly Budgets
    offline_budget = st.slider("💰 Offline Budget (TV, Radio, OOH)", min_value=100000, max_value=10000000, value=5000000, step=50000)
    online_budget = st.slider("💻 Online Budget (Search, Display, Video, Digital Audio, Social)", min_value=100000, max_value=10000000, value=5000000, step=50000)

    # User Inputs for Search & TV Allocation
    search_budget_pct = st.slider("🔍 % of Online Budget Allocated to Search", min_value=0, max_value=100, value=50)
    tv_budget_pct = st.slider("📺 % of Offline Budget Allocated to TV", min_value=0, max_value=100, value=60)

    # Budget Allocation with Limited Randomization
    def allocate_remaining_budget(total_budget, fixed_allocation, num_channels):
        remaining_budget = total_budget - fixed_allocation
        random_distribution = np.random.dirichlet(np.ones(num_channels)) * remaining_budget
        return random_distribution

    weeks = 52
    np.random.seed(42)  # Ensures reproducibility

    # Allocate TV & Search Spend First
    tv_spend = (tv_budget_pct / 100) * offline_budget
    search_spend = (search_budget_pct / 100) * online_budget

    # Allocate the Remaining Budgets Randomly
    remaining_offline_allocation = allocate_remaining_budget(offline_budget, tv_spend, 2)  # Radio, OOH
    remaining_online_allocation = allocate_remaining_budget(online_budget, search_spend, 4)  # Social, Display, Video, Digital Audio

    # Assign Values to Offline Channels
    offline_allocation = np.array([tv_spend] + list(remaining_offline_allocation))

    # Assign Values to Online Channels
    online_allocation = np.array([search_spend] + list(remaining_online_allocation))

    # Create Weekly DataFrame
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
    st.success("✅ Budget Distributed! Move to Page 2 for Model Predictions.")

# ------------------------------------------
# 🎯 PAGE 2: Predict Revenue Using MMM Model
# ------------------------------------------
def page_forecast():
    st.title("📈 MMM Model Predictions & Analysis")
    
    if "weekly_data" not in st.session_state:
        st.warning("⚠️ Please go back to Page 1 and enter your budget first!")
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

    # Normalize Predicted Revenue (Index between 0 and 100)
    scaler = MinMaxScaler(feature_range=(0, 100))
    weekly_data["Predicted_Revenue"] = scaler.fit_transform(weekly_data[["Predicted_Revenue"]])

    # 📈 Predicted Revenue Trend Graph (Fully Indexed 0-100, Seaborn Styled)
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
    ax.set_title("Predicted Revenue Over the Last 12 Weeks")

    st.pyplot(fig)   
    

# 🚀 STREAMLIT APP NAVIGATION
st.sidebar.title("📌 Menu")
page = st.sidebar.radio("Go to:", ["Budget Input", "Forecast"])
if page == "Budget Input":
    page_budget_input()
else:
    page_forecast()

