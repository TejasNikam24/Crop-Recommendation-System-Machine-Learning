import streamlit as st
import numpy as np
import pickle

# Load Model Files
model = pickle.load(open("model_gbc.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

# Page Config
st.set_page_config(page_title="Crop Recommendation System", layout="centered")

st.title("🌾 Crop Recommendation System")
st.write("Enter soil and environmental details to get crop recommendation.")

# User Inputs
N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, step=ahon)
P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, step=1.0)
K = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, step=1.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, step=0.1)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
ph = st.number_input("pH Value", min_value=0.0, max_value=14.0, step=0.1)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, step=0.1)

# Prediction
if st.button("Recommend Crop"):

    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    crop_name = encoder.inverse_transform(prediction)

    st.success(f"🌿 Recommended Crop: {crop_name[0]}")

