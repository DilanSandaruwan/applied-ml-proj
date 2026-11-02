import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Colombo PM2.5 Forecaster & XAI", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/gpr_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    features = joblib.load("models/features.pkl")
    explainer = joblib.load("models/shap_explainer.pkl")
    return model, scaler, features, explainer

try:
    gpr, scaler, features, explainer = load_artifacts()
except Exception as e:
    st.error(f"Error loading models. Run 02_model_training_xai.py first: {e}")
    st.stop()

st.title("Colombo Air Quality Predictor & Explainability Dashboard")
st.markdown(
    "Predicting **PM2.5 particulate pollution** in Colombo using **Gaussian Process Regression (GPR)** "
    "with Bayesian uncertainty bounds and SHAP local attribution."
)

# Input Sliders
st.sidebar.header("Meteorological & Temporal Inputs")
temp = st.sidebar.slider("Temperature (°C)", 22.0, 38.0, 29.5, 0.5)
humidity = st.sidebar.slider("Relative Humidity (%)", 40.0, 100.0, 78.0, 1.0)
pressure = st.sidebar.slider("Surface Pressure (hPa)", 1000.0, 1025.0, 1011.0, 0.5)
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.2, 12.0, 3.2, 0.1)
wind_dir = st.sidebar.slider("Wind Direction (°)", 0.0, 360.0, 240.0, 5.0)
precip = st.sidebar.slider("Precipitation (mm)", 0.0, 45.0, 0.0, 0.5)
hour = st.sidebar.slider("Hour of Day (0-23)", 0, 23, 8, 1)
day_of_week = st.sidebar.selectbox("Day of Week", options=list(range(7)), format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
is_weekend = 1 if day_of_week >= 5 else 0
pm25_lag = st.sidebar.slider("PM2.5 Lag 1-Hour (µg/m³)", 5.0, 120.0, 32.0, 1.0)

# Build Feature Vector
input_dict = {
    "temperature_2m": temp, "relative_humidity_2m": humidity,
    "surface_pressure": pressure, "wind_speed_10m": wind_speed,
    "wind_direction_10m": wind_dir, "precipitation": precip,
    "hour": hour, "day_of_week": day_of_week,
    "is_weekend": is_weekend, "pm25_lag_1h": pm25_lag
}
input_df = pd.DataFrame([input_dict])
input_scaled = scaler.transform(input_df)

col1, col2 = st.columns([1, 1.3])

with col1:
    st.subheader("Model Inference & Uncertainty")
    y_pred, y_std = gpr.predict(input_scaled, return_std=True)
    pred_val = float(y_pred[0])
    std_val = float(y_std[0])
    lower_ci = max(0.0, pred_val - 1.96 * std_val)
    upper_ci = pred_val + 1.96 * std_val
    
    st.metric(label="Predicted PM2.5 Concentration", value=f"{pred_val:.2f} µg/m³")
    st.info(f"**95% Predictive Credible Interval:** [{lower_ci:.2f} — {upper_ci:.2f}] µg/m³\n*(Predictive standard deviation $\\sigma$: {std_val:.2f})*")
    
    if pred_val <= 12.0:
        st.success("Air Quality Category: **Good** (0 - 12 µg/m³)")
    elif pred_val <= 35.4:
        st.warning("Air Quality Category: **Moderate** (12.1 - 35.4 µg/m³)")
    elif pred_val <= 55.4:
        st.warning("Air Quality Category: **Unhealthy for Sensitive Groups** (35.5 - 55.4 µg/m³)")
    else:
        st.error("Air Quality Category: **Unhealthy** (> 55.5 µg/m³)")

with col2:
    st.subheader("Local Explainability (SHAP Waterfall)")
    with st.spinner("Calculating SHAP attributions..."):
        shap_values_single = explainer.shap_values(input_scaled)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        explanation_obj = shap.Explanation(
            values=shap_values_single[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0].values,
            feature_names=features
        )
        shap.waterfall_plot(explanation_obj, max_display=7, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

st.divider()
st.subheader("Model Diagnostic Highlights")
c1, c2 = st.columns(2)
with c1:
    if os.path.exists("shap_summary.png"):
        st.image("shap_summary.png", caption="Global Feature Importance (SHAP Summary)")
with c2:
    if os.path.exists("pdp_plots.png"):
        st.image("pdp_plots.png", caption="Partial Dependence Plots (PDP)")