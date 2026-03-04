import streamlit as st
import numpy as np
import pickle

# ----------------------------
# Load Model
# ----------------------------
model = pickle.load(open("model_gbc.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

st.set_page_config(page_title="AgriAI Dashboard", layout="wide")

# ----------------------------
# Custom CSS – SaaS Dark Theme
# ----------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

section[data-testid="stSidebar"] {
    background: #0c1b1f;
    border-right: 1px solid rgba(0,255,150,0.2);
}

.navbar {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 12px;
    backdrop-filter: blur(10px);
    margin-bottom: 20px;
}

.card {
    background: rgba(255,255,255,0.07);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(0,255,150,0.2);
    box-shadow: 0 0 20px rgba(0,255,150,0.05);
}

.metric-title {
    font-size: 14px;
    color: #00ffa3;
}

.metric-value {
    font-size: 26px;
    font-weight: bold;
}

.stButton>button {
    background: linear-gradient(90deg, #00ffa3, #00c97b);
    border-radius: 10px;
    border: none;
    color: black;
    font-weight: bold;
    padding: 10px 20px;
}

.stButton>button:hover {
    transform: scale(1.05);
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("🌱 AgriAI")
page = st.sidebar.radio("Navigation", ["Dashboard", "Prediction", "Analytics", "About"])

# ----------------------------
# Top Navbar
# ----------------------------
st.markdown('<div class="navbar">🤖 AI Powered Smart Farming Dashboard | 👤 Admin</div>', unsafe_allow_html=True)

# =====================================================
# DASHBOARD PAGE
# =====================================================
if page == "Dashboard":

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card"><div class="metric-title">Soil Health</div><div class="metric-value">87%</div></div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><div class="metric-title">Weather Stability</div><div class="metric-value">72%</div></div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card"><div class="metric-title">Crop Suitability Score</div><div class="metric-value">AI Ready</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.image("https://images.unsplash.com/photo-1500382017468-9049fed747ef", use_column_width=True)

# =====================================================
# PREDICTION PAGE
# =====================================================
elif page == "Prediction":

    st.subheader("🔬 AI Crop Prediction Engine")

    col1, col2, col3 = st.columns(3)

    with col1:
        n = st.number_input("Nitrogen", 0.0, 150.0, 90.0)
        p = st.number_input("Phosphorus", 0.0, 150.0, 40.0)
        k = st.number_input("Potassium", 0.0, 150.0, 40.0)

    with col2:
        temp = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
        hum = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)

    with col3:
        ph = st.number_input("pH Value", 0.0, 14.0, 6.5)
        rain = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)

    if st.button("Run AI Prediction 🚀"):

        input_data = np.array([[n, p, k, temp, hum, ph, rain]])
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)
        probabilities = model.predict_proba(scaled_input)

        crop_name = encoder.inverse_transform(prediction)[0]
        confidence = round(np.max(probabilities) * 100, 2)

        st.markdown("### 🌾 Recommended Crop")
        st.markdown(f"<div class='card'><h2 style='color:#00ffa3'>{crop_name}</h2><p>Suitability Score: {confidence}%</p></div>", unsafe_allow_html=True)

        st.image("https://source.unsplash.com/600x300/?"+crop_name+",farm", use_column_width=True)

# =====================================================
# ANALYTICS PAGE
# =====================================================
elif page == "Analytics":

    st.subheader("📊 Agricultural Insights")

    st.markdown('<div class="card">AI analyzes soil composition, rainfall trends, and climate patterns to optimize crop yield and sustainability.</div>', unsafe_allow_html=True)

# =====================================================
# ABOUT PAGE
# =====================================================
elif page == "About":

    st.subheader("🌍 About AgriAI")
    st.markdown("""
    AgriAI is a next-generation AI-powered Crop Recommendation System designed 
    for precision agriculture and sustainable farming. 
    
    Built using Machine Learning and predictive analytics to help farmers 
    maximize productivity and reduce environmental impact.
    """)
