import streamlit as st
import numpy as np
import pickle
import easyocr
import re
from PIL import Image
import plotly.graph_objects as go

model = pickle.load(open('parkinson_model.pkl', 'rb'))
reader = easyocr.Reader(['en'], gpu=False)

st.set_page_config(page_title="Parkinson's Predictor", layout="centered")
st.title("🧠 Parkinson's Detection from Voice Parameters")

st.markdown("Use OCR or manual input to analyze Parkinson's risk.")

uploaded = st.file_uploader("📷 Upload Report Image (PNG/JPG)", type=['png', 'jpg', 'jpeg'])

values = {}

if uploaded:
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

# Input Form
fo = st.number_input("Fo", value=values.get('fo', 0.0), step=0.1)
fhi = st.number_input("Fhi", value=values.get('fhi', 0.0), step=0.1)
flo = st.number_input("Flo", value=values.get('flo', 0.0), step=0.1)
jitter = st.number_input("Jitter(%)", value=values.get('jitter_percent', 0.0), step=0.001)
rap = st.number_input("RAP", value=values.get('rap', 0.0), step=0.001)
ppe = st.number_input("PPE", value=values.get('ppe', 0.0), step=0.001)

features = np.array([[fo, fhi, flo, jitter, rap, ppe]])

if st.button("Predict"):
    res = model.predict(features)
    st.warning("⚠️ Parkinson's likely!" if res[0] else "✅ Parkinson's unlikely.")

    labels = ['Fo', 'Fhi', 'Flo', 'Jitter(%)', 'RAP', 'PPE']
    vals = [fo, fhi, flo, jitter, rap, ppe]
    scaled = [v / m for v, m in zip(vals, [300, 400, 300, 1.0, 0.2, 1.0])]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=scaled, theta=labels, fill='toself'))
    fig_radar.update_layout(title="Feature Radar Chart", polar=dict(radialaxis=dict(range=[0, 1])))
    st.plotly_chart(fig_radar)

    fig_bar = go.Figure([go.Bar(x=labels, y=vals)])
    fig_bar.update_layout(title="Feature Values")
    st.plotly_chart(fig_bar)
