import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Set Streamlit page config
st.set_page_config(page_title="Media Mix Modeling Dashboard", layout="wide")

# Load Pre-Trained MMM Model
@st.cache_resource
def load_trained_model():
    model = xgb.Booster()
    model.load_model("optimized_mmm_xgboost_model.json")
    return model

mmm_model = load_trained_model()

# Title
st.title("📊 Media Mix Modeling - Impact Calculator")

# Tabs for Budget Input & Forecast
tabs = st.tabs(["Budget Input & Allocation", "Forecast & Predictions"])

with tabs[0]:
    st.header("💰 Budget Input & Allocation")
    
    # Budget Inputs
    col1, col2 = st.columns(2)
    
    with col1:
        offline_budget = st.slider("Offline Budget (TV, Radio, OOH)", 100000, 10000000, 5000000, 50000)
        tv_budget_pct = st.slider("% of Offline Budget Allocated to TV", 0, 100, 60)
    
    with col2:
        online_budget = st.slider("Online Budget (Search, Display, Video, Digital Audio, Social)", 100000, 10000000, 5000000, 50000)
        search_budget_pct = st.slider("% of Online Budget Allocated to Search", 0, 100, 50)
    
    # Budget Allocation Logic
    def allocate_remaining_budget(total_budget, fixed_allocation, num_channels):
        remaining_budget = total_budget - fixed_allocation
        random_distribution = np.random.dirichlet(np.ones(num_channels)) * remaining_budget
        return random_distribution
    
    weeks = 52
    np.random.seed(42)
    
    # Allocate Fixed Spend
    tv_spend = (tv_budget_pct / 100) * offline_budget
    search_spend = (search_budget_pct / 100) * online_budget
    
    # Distribute Remaining Budget
    remaining_offline = allocate_remaining_budget(offline_budget, tv_spend, 2)
    remaining_online = allocate_remaining_budget(online_budget, search_spend, 4)
    
    # Assign Values
    offline_allocation = np.array([tv_spend] + list(remaining_offline))
    online_allocation = np.array([search_spend] + list(remaining_online))
    
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
    st.success("✅ Budget Allocated! Check the Forecast Tab.")

with tabs[1]:
    st.header("📈 Revenue Forecast & Predictions")
    
    if "weekly_data" not in st.session_state:
        st.warning("⚠️ Please allocate budgets first in the Budget Input tab!")
    else:
        weekly_data = st.session_state.weekly_data
        
        # Additional Factors
        weekly_data["Inflation_Rate"] = np.random.uniform(2, 8, len(weekly_data))
        weekly_data["Consumer_Index"] = np.random.uniform(100, 120, len(weekly_data))
        weekly_data["Gross_Rating_Point"] = np.random.uniform(40, 220, len(weekly_data))
        weekly_data["CTR_Scaled"] = np.random.uniform(1, 40, len(weekly_data))
        
        spend_factors = [
            "Search_Spend_Adstock", "Display_Spend_Adstock", "Video_Spend_Adstock",
            "Social_Spend_Adstock", "Digital_Audio_Spend_Adstock",
            "TV_Spend_Adstock", "Radio_Spend_Adstock", "OOH_Spend_Adstock"
        ]
        macro_factors = ["Consumer_Index", "Inflation_Rate", "Gross_Rating_Point", "CTR_Scaled"]
        
        model_input_data = weekly_data[spend_factors + macro_factors]
        X_test_dmatrix = xgb.DMatrix(model_input_data)
        weekly_data["Predicted_Revenue"] = mmm_model.predict(X_test_dmatrix)
        
        # Normalize Revenue
        scaler = MinMaxScaler(feature_range=(0, 100))
        weekly_data["Predicted_Revenue"] = scaler.fit_transform(weekly_data[["Predicted_Revenue"]])
        
        # 📊 Plot Revenue Trend
        st.subheader("📉 Predicted Revenue Trend")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(x=weekly_data["Week"].iloc[-12:], y=weekly_data["Predicted_Revenue"].iloc[-12:], marker="o", linewidth=2.5, ax=ax)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Week")
        ax.set_ylabel("Revenue Index (0-100)")
        ax.set_title("Predicted Revenue for 12 Weeks")
        st.pyplot(fig)
        