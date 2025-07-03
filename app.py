import streamlit as st
import numpy as np
import pickle
import easyocr
import re
from PIL import Image
import plotly.graph_objects as go

# Load the trained model
model = pickle.load(open('parkinson_model.pkl', 'rb'))

st.set_page_config(page_title="Parkinson's Detector", layout="centered")

st.title("🧠 Parkinson's Disease Detection App")

st.markdown("Upload a voice report screenshot or manually enter features to check for Parkinson's disease.")

# OCR Upload Section
st.subheader("📸 Upload Screenshot for Auto-Fill")
uploaded_file = st.file_uploader("Upload Image (JPG/PNG)", type=["png", "jpg", "jpeg"])

ocr_data = {}

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Report", use_column_width=True)
    
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(np.array(image), detail=0)
    combined_text = ' '.join(result)
    
    def extract(pattern, text):
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

    ocr_data = {k: extract(p, combined_text) for k, p in patterns.items()}
    st.info("OCR Extracted Values:")
    for k, v in ocr_data.items():
        st.write(f"**{k.upper()}**: {v if v is not None else 'Not Detected'}")

# Manual Input Section
st.subheader("✍️ Manual Input (Auto-filled if OCR used)")

fo = st.number_input("MDVP:Fo(Hz)", value=ocr_data.get('fo', 0.0), step=0.1)
fhi = st.number_input("MDVP:Fhi(Hz)", value=ocr_data.get('fhi', 0.0), step=0.1)
flo = st.number_input("MDVP:Flo(Hz)", value=ocr_data.get('flo', 0.0), step=0.1)
jitter_percent = st.number_input("MDVP:Jitter(%)", value=ocr_data.get('jitter_percent', 0.0), step=0.001)
rap = st.number_input("MDVP:RAP", value=ocr_data.get('rap', 0.0), step=0.001)
ppe = st.number_input("PPE", value=ocr_data.get('ppe', 0.0), step=0.001)

input_data = np.array([[fo, fhi, flo, jitter_percent, rap, ppe]])

# Prediction + Visualization
if st.button("🔍 Detect Parkinson's"):
    result = model.predict(input_data)
    if result[0] == 1:
        st.error("⚠️ Parkinson’s disease detected.")
    else:
        st.success("✅ No signs of Parkinson’s disease.")

    # Radar Chart
    labels = ['Fo', 'Fhi', 'Flo', 'Jitter(%)', 'RAP', 'PPE']
    raw_vals = [fo, fhi, flo, jitter_percent, rap, ppe]
    max_vals = [300, 400, 300, 1.0, 0.2, 1.0]
    scaled_vals = [v / m if m != 0 else 0 for v, m in zip(raw_vals, max_vals)]

    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=scaled_vals,
        theta=labels,
        fill='toself',
        name='Patient Profile',
        line=dict(color='deepskyblue')
    ))
    radar_fig.update_layout(
        title="📈 Feature Radar Chart",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False
    )
    st.plotly_chart(radar_fig)

    # Bar Chart
    bar_fig = go.Figure(data=[
        go.Bar(x=labels, y=raw_vals, marker_color='lightskyblue')
    ])
    bar_fig.update_layout(title="📊 Feature Values", xaxis_title="Feature", yaxis_title="Value")
    st.plotly_chart(bar_fig)
