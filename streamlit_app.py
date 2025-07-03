import streamlit as st
import numpy as np
import pickle
import easyocr
import re
from PIL import Image
import plotly.graph_objects as go

st.set_page_config(page_title="Parkinson's Predictor", layout="centered")
st.title("🧠 Parkinson's Detection from Voice Parameters")

st.markdown("Choose input mode and analyze Parkinson's disease risk using ML.")

# --- Model Mode Selector ---
mode = st.radio("Choose Prediction Mode", ["Quick (6 Features)", "Detailed (22 Features)"])

# --- Load Appropriate Model ---
if mode == "Quick (6 Features)":
    model = pickle.load(open("parkinson_model_6.pkl", "rb"))
    selected_mode = 6
else:
    model = pickle.load(open("parkinson_model_22.pkl", "rb"))
    selected_mode = 22

# --- OCR (Quick Mode Only) ---
values = {}
if selected_mode == 6:
    st.markdown("### 📷 Upload Report Image (Optional - Quick Mode Only)")
    uploaded = st.file_uploader("Upload Report Image (PNG/JPG)", type=['png', 'jpg', 'jpeg'])

    if uploaded:
        reader = easyocr.Reader(['en'], gpu=False)
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        results = reader.readtext(np.array(img), detail=0)
        text = ' '.join(results)

        def find(pattern):
            match = re.search(pattern, text)
            return float(match.group(1)) if match else None

        patterns = {
            'fo': r'Fo\(Hz\)[^\d]*([\d.]+)',
            'fhi': r'Fhi\(Hz\)[^\d]*([\d.]+)',
            'flo': r'Flo\(Hz\)[^\d]*([\d.]+)',
            'jitter_percent': r'Jitter\(.*?%\)[^\d]*([\d.]+)',
            'rap': r'RAP[^\d]*([\d.]+)',
            'ppe': r'PPE[^\d]*([\d.]+)'
        }

        values = {k: find(p) for k, p in patterns.items()}
        st.info("📄 OCR Results:")
        for k, v in values.items():
            st.write(f"{k.upper()}: {v}")

# --- Feature Input ---
st.markdown("### ✍️ Enter Feature Values Below:")

if selected_mode == 6:
    fo = st.number_input("Fo", value=values.get('fo', 0.0), step=0.1)
    fhi = st.number_input("Fhi", value=values.get('fhi', 0.0), step=0.1)
    flo = st.number_input("Flo", value=values.get('flo', 0.0), step=0.1)
    jitter = st.number_input("Jitter(%)", value=values.get('jitter_percent', 0.0), step=0.001)
    rap = st.number_input("RAP", value=values.get('rap', 0.0), step=0.001)
    ppe = st.number_input("PPE", value=values.get('ppe', 0.0), step=0.001)

    features = np.array([[fo, fhi, flo, jitter, rap, ppe]])
    labels = ['Fo', 'Fhi', 'Flo', 'Jitter(%)', 'RAP', 'PPE']
    vals = [fo, fhi, flo, jitter, rap, ppe]

else:
    feature_names = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
        'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
        'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3',
        'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR',
        'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
    ]

    vals = []
    for f in feature_names:
        val = st.number_input(f, min_value=0.0, step=0.001)
        vals.append(val)

    features = np.array([vals])
    labels = feature_names

# --- Prediction & Visuals ---
if st.button("🔍 Predict"):
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.warning("⚠️ Parkinson's likely!")
    else:
        st.success("✅ Parkinson's unlikely.")

    # --- Radar Chart ---
    max_vals = [300, 400, 300, 1.0, 0.2, 1.0] if selected_mode == 6 else [max(v, 0.01) for v in vals]  # Avoid zero division
    scaled = [v / m if m else 0 for v, m in zip(vals, max_vals)]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=scaled, theta=labels, fill='toself'))
    fig_radar.update_layout(title="Feature Radar Chart", polar=dict(radialaxis=dict(range=[0, 1])))
    st.plotly_chart(fig_radar)

    # --- Bar Chart ---
    fig_bar = go.Figure([go.Bar(x=labels, y=vals)])
    fig_bar.update_layout(title="Feature Values")
    st.plotly_chart(fig_bar)
