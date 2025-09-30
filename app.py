import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import shap

# Page Configuration
st.set_page_config(
    page_title="Churn Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load Models & Assets
@st.cache_resource
def load_assets():
    """Loads all necessary machine learning assets."""
    imputer = joblib.load("imputer.joblib")
    scaler = joblib.load("scaler.joblib")
    model = joblib.load("xgb_churn_model.joblib")
    explainer = shap.TreeExplainer(model)
    return imputer, scaler, model, explainer

imputer, scaler, model, explainer = load_assets()

# Header
st.title("Customer Churn Predictor")
st.write("Predict churn with high accuracy by providing the complete customer profile.")

# Input Form
with st.form("churn_prediction_form"):
    st.header("Enter Full Customer Details")

    # Plans & Account
    st.subheader("Account and Plans")
    col1, col2 = st.columns(2)
    with col1:
        international_plan = st.radio("Has International Plan?", ("No", "Yes"), horizontal=True)
    with col2:
        voice_mail_plan = st.radio("Has Voice Mail Plan?", ("No", "Yes"), horizontal=True)
    
    account_length = st.slider("Account Length (days)", 1, 240, 100)
    customer_service_calls = st.slider("Customer Service Calls", 0, 10, 1)

    # Usage Details (Minutes)
    st.subheader("Usage Details (in Minutes)")
    col3, col4, col5 = st.columns(3)
    with col3:
        day_mins = st.number_input("Day Mins", min_value=0.0, value=180.0)
    with col4:
        evening_mins = st.number_input("Evening Mins", min_value=0.0, value=200.0)
    with col5:
        night_mins = st.number_input("Night Mins", min_value=0.0, value=200.0)
    
    international_mins = st.number_input("International Mins", min_value=0.0, value=10.0)
    
    # Call Details (Number of Calls)
    st.subheader("Number of Calls")
    col6, col7, col8 = st.columns(3)
    with col6:
        day_calls = st.number_input("Day Calls", min_value=0, value=100)
    with col7:
        evening_calls = st.number_input("Evening Calls", min_value=0, value=100)
    with col8:
        night_calls = st.number_input("Night Calls", min_value=0, value=100)

    international_calls = st.number_input("International Calls", min_value=0, value=4)

    # Submit Button
    st.markdown("---") # Visual separator
    predict_button = st.form_submit_button(label="Predict Churn", use_container_width=True)


# Prediction Logic
if predict_button:
    # Convert radio button inputs to numerical values
    int_plan_val = 1 if international_plan == "Yes" else 0
    vm_plan_val = 1 if voice_mail_plan == "Yes" else 0

    # Create the DataFrame with ALL user inputs 
    feature_names = [
        'account_length', 'voice_mail_plan', 'day_mins', 'evening_mins',
        'night_mins', 'international_mins', 'customer_service_calls',
        'international_plan', 'day_calls', 'evening_calls',
        'night_calls', 'international_calls'
    ]
    input_data = pd.DataFrame([[
        account_length, vm_plan_val, day_mins, evening_mins,
        night_mins, international_mins, customer_service_calls, int_plan_val,
        day_calls, evening_calls, night_calls, international_calls
    ]], columns=feature_names)

    # Preprocess and predict
    input_imputed = imputer.transform(input_data)
    input_scaled = scaler.transform(input_imputed)
    probability = model.predict_proba(input_scaled)[0][1]
    shap_values = explainer.shap_values(input_scaled)

    st.markdown("---")
    st.header("Prediction Result")
    
    # Prediction Result Visualization
    col1, col2 = st.columns(2)
    with col1:
        if probability > 0.5:
            st.error("Status: HIGH RISK")
        else:
            st.success("Status: LOW RISK")
    with col2:
        st.metric(label="Churn Probability", value=f"{probability:.1%}")

    # Explanation Visualization
    st.subheader("Top 5 Reasons for this Prediction")
    shap_df = pd.DataFrame({
        'feature': [f.replace('_', ' ').title() for f in input_data.columns],
        'shap_value': shap_values[0, :],
    })
    shap_df['color'] = np.where(shap_df['shap_value'] > 0, '#e53e3e', '#3182ce')
    shap_df['abs_shap'] = np.abs(shap_df['shap_value'])
    shap_df = shap_df.sort_values('abs_shap', ascending=False).head(5)

    fig = go.Figure(go.Bar(
        x=shap_df['shap_value'],
        y=shap_df['feature'],
        orientation='h',
        marker_color=shap_df['color']
    ))
    fig.update_layout(
        xaxis_title="Impact on Churn Risk (Red = Higher Risk)",
        yaxis=dict(autorange="reversed"),
        height=300,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    # This line makes the plot completely static
    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})


