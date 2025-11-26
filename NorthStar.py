import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import os
import shap
import matplotlib.pyplot as plt

# ============================================================
# Page Config
# ============================================================
st.set_page_config(page_title="NorthStar — Process Advisor", layout="wide")


# ============================================================
# Load Video Base64
# ============================================================
def load_video_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

VIDEO_PATH = "static/Video.mp4"

if os.path.exists(VIDEO_PATH):
    video_base64 = load_video_base64(VIDEO_PATH)
else:
    st.error(f"❌ Video not found: {VIDEO_PATH}")
    st.stop()


# ============================================================
# DARK PREMIUM THEME
# ============================================================
st.markdown("""
<style>

body, .stApp {
    background-color: #0E0E0E;
    color: #E0E0E0;
}

/* Center containers */
.center-container {
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
}

/* Titles */
h1, h2, h3, h4 {
    color: #FFFFFF;
    text-align: center;
}

/* Input labels */
label {
    color: #CCCCCC !important;
}

/* Badges */
.success-badge {
    background-color: #0f5132;
    color: #d1e7dd;
    padding: 6px 12px;
    border-radius: 6px;
    font-weight: bold;
}

.fail-badge {
    background-color: #842029;
    color: #f8d7da;
    padding: 6px 12px;
    border-radius: 6px;
    font-weight: bold;
}

/* Video box */
.video-box {
    display: flex;
    justify-content: center;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# Title + Subtitle + Description
# ============================================================
st.markdown("""
<div class="center-container">

<h1>NorthStar — Process Advisor</h1>
<h3>Laser Powder Bed Fusion (L-PBF) Optimization</h3>

<p style="color:#BBBBBB; font-size:18px;">
AI-driven guidance for predicting and optimizing print success in L-PBF systems.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# Video (Centered)
# ============================================================
video_html = f"""
<div class="video-box">
    <video autoplay loop muted playsinline
           style="width:350px; border-radius: 12px;">
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
    </video>
</div>
"""

st.markdown(video_html, unsafe_allow_html=True)


# ============================================================
# Load Model
# ============================================================
MODEL_PATH = "best_model_smote.pkl"

pipeline = joblib.load(MODEL_PATH)
model = pipeline.named_steps["model"]
preprocess = pipeline.named_steps["prep"]


# ============================================================
# INPUT SECTION (2x2 GRID)
# ============================================================
st.write("---")
st.markdown("## 🔧 Process Parameters")

def explain(text):
    st.markdown(f"<p style='color:#888; font-size:13px;'>{text}</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    power = st.number_input("Power [W]", 50, 500, 180)
    explain("Laser energy delivered to the powder bed.")

with col2:
    velocity = st.number_input("Velocity [m/s]", 0.1, 3.0, 0.9)
    explain("Laser scan speed across the powder bed.")

with col1:
    hatch = st.number_input("Hatch Spacing [μm]", 10, 200, 90)
    explain("Distance between scan tracks.")

with col2:
    beam = st.number_input("Beam Diameter [μm]", 20, 200, 78)
    explain("Effective beam spot size.")

with col1:
    layer = st.number_input("Powder Layer Thickness [μm]", 10, 200, 50)
    explain("Thickness of each powder layer.")

with col2:
    d90 = st.number_input("d90 [μm]", 1, 60, 15)
    explain("Powder particle coarse fraction (90th percentile).")


# CATEGORICAL
st.markdown("## 🧩 Material & Process Settings")

col1, col2 = st.columns(2)

with col1:
    atomosphere = st.selectbox("Atmosphere of Build", ["Argon", "Nitrogen"])
    explain("Protective gas used during printing.")

with col2:
    nucleants = st.selectbox("Nucleants", ["N11", "N12", "N13"])
    explain("Powder additive to promote solidification.")

with col1:
    atom_atm = st.selectbox("Atomization Atmosphere", ["Gas", "Water"])
    explain("Powder production method.")

with col2:
    same_layer = st.selectbox("Same Layer Scanned?", ["0", "1"])
    explain("Laser rescans same layer.")


# ============================================================
# Build full input row
# ============================================================
X = pd.DataFrame([{
    "Power [W]": power,
    "Velocity [m/s]": velocity,
    "Hatch Spacing [μm]": hatch,
    "Beam Diameter [μm]": beam,
    "Powder Layer Thickness [μm]": layer,
    "d90 [um]": d90,
    "Atomosphere of build": atomosphere,
    "Nucleants": nucleants,
    "Atomization Atomosphere": atom_atm,
    "Same Layer Scanned?": same_layer,
    "Initial Powder Bed Temperature [K]": 300,
    "Substrate/Platform Temperature [K]": 100,
    "Surface Energy Density [J/mm2]": 2.5,
    "Linear Energy Density [J/mm]": 3.5,
    "Volume Energy Density [J/mm3]": 30,
    "Powder Recycled How Many Times?": 0,
    "Powder Shape [0-1]": 1.0,
    "Build Direction [degrees]": "90",
}])


# ============================================================
# PREDICTION BUTTON
# ============================================================
st.write("---")
if st.button("🔮 Predict"):
    pred = pipeline.predict(X)[0]
    prob = float(pipeline.predict_proba(X)[0][1])

    st.markdown("## 📊 Prediction Result")
    if pred == 1:
        st.markdown(f"<span class='success-badge'>SUCCESS</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span class='fail-badge'>FAIL</span>", unsafe_allow_html=True)

    st.markdown(f"**Probability of SUCCESS:** `{prob:.5f}`")


