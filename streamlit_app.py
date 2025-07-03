import streamlit as st
import numpy as np
import pandas as pd
import pickle
import easyocr
import re
from PIL import Image
import plotly.graph_objects as go

# Load model
model = pickle.load(open('parkinson_model.pkl', 'rb'))
reader = easyocr.Reader(['en'], gpu=False)

# Page config
st.set_page_config(page_title="Parkinson's Predictor", layout="centered")
st.title("🧠 Parkinson's Detection from Voice Parameters")
st.markdown("Upload report via image, text, or CSV. Then analyze Parkinson's risk.")

values = {}

# ===== Upload Options =====
upload_mode = st.radio("Choose Input Type", ["Image Upload (OCR)", "Text Box Input", "CSV Upload"])

# ===== Image Upload & OCR =====
if upload_mode == "Image Upload (OCR)":
    uploaded = st.file_uploader("📷 Upload Report Image", type=['png', 'jpg', 'jpeg'])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        results = reader.readtext(np.array(img), detail=0)
        text = ' '.join(results)

# ===== Text Area Input =====
elif upload_mode == "Text Box Input":
    text = st.text_area("📄 Paste Report Text Here")
    
# ===== CSV Upload =====
elif upload_mode == "CSV Upload":
    csv_file = st.file_uploader("📄 Upload CSV File", type=['csv'])
    if csv_file:
        df = pd.read_csv(csv_file)
        st.dataframe(df.head())
        # Try to extract the same 6 features
        if set(['Fo', 'Fhi', 'Flo', 'Jitter(%)', 'RAP', 'PPE']).issubset(df.columns):
            features = df[['Fo', 'Fhi', 'Flo', 'Jitter(%)', 'RAP', 'PPE']].iloc[0].values.reshape(1, -1)
            csv_mode = True
        else:
            st.error("CSV must contain columns: Fo, Fhi, Flo, Jitter(%), RAP, PPE")
            csv_mode = False
    else:
        csv_mode = False

# ===== Feature Extraction from Text (OCR or TextBox) =====
if upload_mode in ["Image Upload (OCR)", "Text Box Input"] and text:
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
    st.info("📄 Extracted Features:")
    for k, v in values.items():
        st.write(f"{k.upper()}: {v}")

# ===== Manual Input for all modes (except valid CSV) =====
if not (upload_mode == "CSV Upload" and csv_mode):
    fo = st.number_input("Fo", value=values.get('fo', 0.0), step=0.1)
    fhi = st.number_input("Fhi", value=values.get('fhi', 0.0), step=0.1)
    flo = st.number_input("Flo", value=values.get('flo', 0.0), step=0.1)
    jitter = st.number_input("Jitter(%)", value=values.get('jitter_percent', 0.0), step=0.001)
    rap = st.number_input("RAP", value=values.get('rap', 0.0), step=0.001)
    ppe = st.number_input("PPE", value=values.get('ppe', 0.0), step=0.001)
    features = np.array([[fo, fhi, flo, jitter, rap, ppe]])

# ===== Predict & Visualize =====
if st.button("Predict"):
    res = model.predict(features)
    st.success("✅ Parkinson's Unlikely." if res[0] == 0 else "⚠️ Parkinson's Likely!")

    # Radar Chart
    labels = ['Fo', 'Fhi', 'Flo', 'Jitter(%)', 'RAP', 'PPE']
    vals = features.flatten().tolist()
    scaled = [v / m for v, m in zip(vals, [300, 400, 300, 1.0, 0.2, 1.0])]
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=scaled, theta=labels, fill='toself', name="Voice Features"))
    fig_radar.update_layout(title="Radar Chart of Voice Features", polar=dict(radialaxis=dict(range=[0, 1])))
    st.plotly_chart(fig_radar)

    # Bar Chart
    fig_bar = go.Figure([go.Bar(x=labels, y=vals)])
    fig_bar.update_layout(title="Feature Values")
    st.plotly_chart(fig_bar)
