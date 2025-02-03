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

# Streamlit UI Setup
st.title("📊 Marketing Mix Modeling What-If Dashboard")
st.write("Simulate the impact of different marketing spends on sales using your MMM model.")

# User Inputs
TV_spend = st.slider("TV Spend ($)", 0, 500000, 100000, step=10000)
Social_spend = st.slider("Social Media Spend ($)", 0, 500000, 50000, step=10000)
Search_spend = st.slider("Search Ad Spend ($)", 0, 500000, 75000, step=10000)
Promo_discount = st.slider("Discount Rate (%)", 0, 50, 10, step=1)

# Feature Processing based on MMM Model
def create_feature_df(TV, Social, Search, Discount):
    return pd.DataFrame({
        "TV": [TV],
        "Social": [Social],
        "Search": [Search],
        "Promo": [Discount]
    })

# Generate Predictions using the MMM Model
def predict_sales(TV, Social, Search, Discount):
    features = create_feature_df(TV, Social, Search, Discount)
    sales_prediction = model.predict(features)[0]  # Use the trained model for predictions
    return sales_prediction

# Prediction
predicted_sales = predict_sales(TV_spend, Social_spend, Search_spend, Promo_discount)

# Visualization
fig, ax = plt.subplots()
scenarios = ["Base Case", "New Scenario"]
sales_values = [50000, predicted_sales]  # Base case is assumed to be 50,000 sales
ax.bar(scenarios, sales_values, color=['blue', 'red'])
ax.set_ylabel("Predicted Sales")
ax.set_title("Impact of Marketing Spend Changes")

st.pyplot(fig)

# Display Prediction
st.metric(label="Predicted Sales", value=f"{predicted_sales:,.0f} units")
import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

# Load Pre-Trained MMM Model (Dummy Model for Now)
def load_model():
    # Here, you would load the trained XGBoost model from the given MMM repository.
    # For demonstration, we assume a dummy model is loaded.
    model = xgb.XGBRegressor()
    return model

model = load_model()

# Streamlit UI Setup
st.title("📊 Marketing Mix Modeling What-If Dashboard")
st.write("Simulate the impact of different marketing spends on sales.")

# User Inputs
TV_spend = st.slider("TV Spend ($)", 0, 500000, 100000, step=10000)
Social_spend = st.slider("Social Media Spend ($)", 0, 500000, 50000, step=10000)
Search_spend = st.slider("Search Ad Spend ($)", 0, 500000, 75000, step=10000)
Promo_discount = st.slider("Discount Rate (%)", 0, 50, 10, step=1)

# Dummy Feature Processing (Replace with actual MMM model features)
def create_feature_df(TV, Social, Search, Discount):
    return pd.DataFrame({
        "TV": [TV],
        "Social": [Social],
        "Search": [Search],
        "Promo": [Discount]
    })

# Generate Predictions
def predict_sales(TV, Social, Search, Discount):
    features = create_feature_df(TV, Social, Search, Discount)
    # Prediction using the MMM model (currently mocked)
    sales_prediction = np.random.uniform(10000, 100000)  # Replace with model.predict(features)
    return sales_prediction

# Prediction
predicted_sales = predict_sales(TV_spend, Social_spend, Search_spend, Promo_discount)

# Visualization
fig, ax = plt.subplots()
scenarios = ["Base Case", "New Scenario"]
sales_values = [50000, predicted_sales]  # Base case is assumed to be 50,000 sales
ax.bar(scenarios, sales_values, color=['blue', 'red'])
ax.set_ylabel("Predicted Sales")
ax.set_title("Impact of Marketing Spend Changes")

st.pyplot(fig)

# Display Prediction
st.metric(label="Predicted Sales", value=f"{predicted_sales:,.0f} units")