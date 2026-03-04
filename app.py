import streamlit as st
import numpy as np
import pickle

# Load Model Files
model = pickle.load(open("model_gbc.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

st.set_page_config(page_title="Crop Recommendation System", layout="centered")

# Custom CSS to mimic your reference image
st.markdown("""
<style>

body {
    background-color: #f0f2f6;
}

.stApp {
    background: linear-gradient(to right, #ffffff, #e6faff);
}

.title {
    text-align: center;
    font-size: 2.5rem;
    font-weight: bold;
    color: #0a4a71;
}

.crop-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
    margin-top: 20px;
}

input {
    font-size: 1rem !important;
}

.stButton>button {
    background-color: #0a8fdb !important;
    color: white !important;
    font-size: 1rem;
    padding: 10px 24px;
    border-radius: 8px;
    transition: 0.3s;
}

.stButton>button:hover {
    background-color: #096fa8 !important;
    transform: scale(1.03);
}

</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="title">🌾 Crop Recommendation System</h1>', unsafe_allow_html=True)
st.write("---")

# Input Form in a card style
with st.container():
    st.markdown('<div class="crop-card">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        n = st.number_input("Nitrogen (N)", min_value=0.0, max_value=150.0, value=90.0)
        p = st.number_input("Phosphorus (P)", min_value=0.0, max_value=150.0, value=40.0)
        k = st.number_input("Potassium (K)", min_value=0.0, max_value=150.0, value=40.0)

    with col2:
        temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
        hum = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0)

    with col3:
        ph = st.number_input("pH Value", min_value=0.0, max_value=14.0, value=6.5)
        rain = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)

    st.markdown("</div>", unsafe_allow_html=True)

# Predict Button
if st.button("Recommend Crop"):
    input_data = np.array([[n, p, k, temp, hum, ph, rain]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    crop_name = encoder.inverse_transform(prediction)[0]

    st.success(f"### 🌱 Recommended Crop: **{crop_name}**", icon="✅")

    # Optional Crop Images (you can add links for images for each crop)
    crop_images = {
        "Rice": "https://example.com/rice.jpg",
        "Wheat": "https://example.com/wheat.jpg",
        "Jute": "https://example.com/jute.png"
    }

    if crop_name in crop_images:
        st.image(crop_images[crop_name], caption=f"{crop_name}", use_column_width=True)
