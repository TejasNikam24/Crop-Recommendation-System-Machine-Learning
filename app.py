import streamlit as st
import numpy as np
import pickle

# -------------------------------
# Load Model Files
# -------------------------------
model = pickle.load(open("model_gbc.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))  # Rename file properly
encoder = pickle.load(open("encoder.pkl", "rb"))

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="Crop Recommendation System", layout="wide")

# -------------------------------
# Custom CSS (Glassmorphism UI)
# -------------------------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), 
                    url("https://images.unsplash.com/photo-1500382017468-9049fed747ef?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
    }

    .input-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }

    h1, h3 {
        color: #ffffff !important;
    }

    .stButton>button {
        background-color: #2ecc71 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 24px !important;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #27ae60 !important;
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# App Header
# -------------------------------
st.title("🌾 Crop Recommendation System")
st.markdown("### Enter soil and environmental details to get the best crop recommendation")

# -------------------------------
# Input Section
# -------------------------------
with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        n = st.number_input("Nitrogen (N)", min_value=0.0, max_value=150.0, value=90.0)
        p = st.number_input("Phosphorus (P)", min_value=0.0, max_value=150.0, value=40.0)
        k = st.number_input("Potassium (K)", min_value=0.0, max_value=150.0, value=40.0)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
        hum = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        ph = st.number_input("pH Value", min_value=0.0, max_value=14.0, value=6.5)
        rain = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Prediction Logic
# -------------------------------
if st.button("Recommend Crop"):

    # Prepare input array
    input_data = np.array([[n, p, k, temp, hum, ph, rain]])

    # Scale input
    scaled_input = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_input)

    # Decode prediction
    crop_name = encoder.inverse_transform(prediction)

    st.success(f"### 🌱 Recommended Crop: {crop_name[0]}")
