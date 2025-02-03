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
    offline_budget = st.slider("💰 Offline Budget (TV, Radio, OOH)", min_value=100000, max_value=10000000, value=500000, step=50000)
    online_budget = st.slider("💻 Online Budget (Search, Display, Video, Digital Audio, Social)", min_value=100000, max_value=10000000, value=700000, step=50000)
    
    # Budget Allocation with Limited Randomization
    def allocate_budget(total_budget, base_distribution, variation=0.2):
        base_allocation = np.array(base_distribution) * total_budget
        random_factors = 1 + np.random.uniform(-variation, variation, len(base_distribution))
        allocated_budget = base_allocation * random_factors
        return allocated_budget / allocated_budget.sum() * total_budget
    
    weeks = 52
    np.random.seed(42)  # Ensures reproducibility

    offline_allocation = allocate_budget(offline_budget, [0.6, 0.2, 0.2])
    online_allocation = allocate_budget(online_budget, [0.5, 0.2, 0.08, 0.15, 0.07])

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
    weekly_data["Week"] = range(1, len(weekly_data) + 1)

    spend_factors = [
        "Search_Spend_Adstock", "Display_Spend_Adstock", "Video_Spend_Adstock",
        "Social_Spend_Adstock", "Digital_Audio_Spend_Adstock",
        "TV_Spend_Adstock", "Radio_Spend_Adstock", "OOH_Spend_Adstock"
    ]

    macro_factors = ["Consumer_Index", "Inflation_Rate", "Gross_Rating_Point", "CTR_Scaled"]
    for col in macro_factors:
        weekly_data[col] = 0

    model_input_data = weekly_data[spend_factors + macro_factors]
    X_test_dmatrix = xgb.DMatrix(model_input_data)
    weekly_data["Predicted_Revenue"] = mmm_model.predict(X_test_dmatrix)

    scaler = MinMaxScaler(feature_range=(0, 100))
    weekly_data["Predicted_Revenue"] = scaler.fit_transform(weekly_data[["Predicted_Revenue"]])

    feature_importance_dict = mmm_model.get_score(importance_type="weight")
    feature_importance_df = pd.DataFrame({
        "Feature": [feature.replace("_Spend_Adstock", "") for feature in spend_factors],
        "Importance": [feature_importance_dict.get(feature, 0) for feature in spend_factors]
    })
    feature_importance_df["Importance"] = scaler.fit_transform(feature_importance_df[["Importance"]])
    
   #  st.subheader("📌 Media Channel Impact")
   #  fig1, ax1 = plt.subplots()
   # sns.barplot(x="Importance", y="Feature", data=feature_importance_df, ax=ax1)
   # ax1.set_xlim(0, 100)
   # ax1.set_title("Feature Importance")
   # st.pyplot(fig1)

    st.subheader("📈 Predicted Revenue Trend (Last 12 Weeks)")
    fig2, ax2 = plt.subplots()
    sns.barplot(x=[f"Week {i}" for i in range(1, 13)], y=weekly_data["Predicted_Revenue"].iloc[-12:], ax=ax2, palette="Blues")
    ax2.set_ylim(0, 100)
    ax2.set_title("Predicted Revenue Trend")
    st.pyplot(fig2)
    
    roi_df = pd.DataFrame({
        "Media Channel": [channel.replace("_Spend_Adstock", "") for channel in spend_factors],
        "ROI": [mmm_model.get_score(importance_type="weight").get(channel, 0) for channel in spend_factors]
    })
    
    st.subheader("💰 ROI per Media Channel")
    fig3, ax3 = plt.subplots()
    sns.barplot(x="ROI", y="Media Channel", data=roi_df, ax=ax3)
    ax3.set_xlabel("Revenue per Dollar Spent")
    st.pyplot(fig3)

# 🚀 STREAMLIT APP NAVIGATION
st.sidebar.title("📌 Menu")
page = st.sidebar.radio("Go to:", ["Budget Input", "Forecast"])
if page == "Budget Input":
    page_budget_input()
else:
    page_forecast()
