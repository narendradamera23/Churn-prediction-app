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
st.write("An app to predict customer churn based on key metrics.")

# Input Form
with st.form("churn_prediction_form"):
    st.header("Enter Customer Details")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Plans")
        international_plan = st.radio("Has International Plan?", ("No", "Yes"), horizontal=True)
        voice_mail_plan = st.radio("Has Voice Mail Plan?", ("No", "Yes"), horizontal=True)
        
    with col2:
        st.subheader("Usage & Support")
        customer_service_calls = st.slider("Customer Service Calls", 0, 10, 1)
        day_mins = st.slider("Total Day Minutes", 0.0, 350.0, 180.0)
        international_mins = st.slider("Total International Minutes", 0.0, 20.0, 10.0)
        
    # Submit button
    predict_button = st.form_submit_button(label="Predict Churn", use_container_width=True)

# Prediction Logic
if predict_button:
    # Convert inputs
    int_plan_val = 1 if international_plan == "Yes" else 0
    vm_plan_val = 1 if voice_mail_plan == "Yes" else 0

    # Create the full feature DataFrame using defaults for non-user inputs
    feature_names = [
        'account_length', 'voice_mail_plan', 'day_mins', 'evening_mins',
        'night_mins', 'international_mins', 'customer_service_calls',
        'international_plan', 'day_calls', 'evening_calls',
        'night_calls', 'international_calls'
    ]
    input_data = pd.DataFrame([[
        101, vm_plan_val, day_mins, 200.0,
        200.0, international_mins, customer_service_calls, int_plan_val,
        100, 100, 100, 3
    ]], columns=feature_names)

    # Preprocess and predict
    input_imputed = imputer.transform(input_data)
    input_scaled = scaler.transform(input_imputed)
    probability = model.predict_proba(input_scaled)[0][1]
    shap_values = explainer.shap_values(input_scaled)

    st.markdown("---")
    st.header("Prediction Result")
    
    # Prediction Visualization
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
    # We use st.altair_chart for better theme integration
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
    st.plotly_chart(fig, use_container_width=True, config={'staticPlot': True})

