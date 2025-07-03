# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pytesseract
from PIL import Image
import io
import plotly.graph_objects as go

# Load models
model_6 = pickle.load(open("parkinson_model_6.pkl", "rb"))
model_22 = pickle.load(open("parkinson_model_22.pkl", "rb"))

# Feature names
features_6 = ['fo', 'fhi', 'flo', 'jitter_percent', 'rap', 'ppe']
features_22 = features_6 + [f'feature_{i}' for i in range(7, 23)]

st.set_page_config(page_title="Parkinson's Predictor", layout="centered")
st.title("🧠 Parkinson's Disease Predictor")
mode = st.selectbox("Choose Input Method:", ["Upload CSV", "Upload Image (OCR)"])

if mode == "Upload CSV":
    upload = st.file_uploader("📄 Upload a CSV file", type=["csv"])
    if upload:
        df = pd.read_csv(upload)
        st.dataframe(df)

        usable_model = None
        prediction = None

        if all(f in df.columns for f in features_22):
            X = df[features_22].iloc[-1:].values
            prediction = model_22.predict(X)[0]
            usable_model = "22-feature model"
        elif all(f in df.columns for f in features_6):
            X = df[features_6].iloc[-1:].values
            prediction = model_6.predict(X)[0]
            usable_model = "6-feature model"

        if usable_model:
            st.success(f"✅ Used {usable_model}")
            st.write(f"🔍 Prediction: **{'Parkinson\'s Likely' if prediction == 1 else 'Parkinson\'s Unlikely'}**")

            # Visualization
            used_features = features_22 if "22" in usable_model else features_6
            raw_values = df[used_features].iloc[-1].values
            fig = go.Figure(data=[go.Bar(x=used_features, y=raw_values)])
            fig.update_layout(title="📊 Feature Values")
            st.plotly_chart(fig)
        else:
            st.error("❌ Required features not found in the uploaded file.")

elif mode == "Upload Image (OCR)":
    image = st.file_uploader("🖼️ Upload an image containing 6 feature values (in order)", type=["png", "jpg", "jpeg"])
    if image:
        img = Image.open(image)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # OCR processing
        text = pytesseract.image_to_string(img)
        st.text_area("📋 Extracted Text", text, height=150)

        # Extract numbers
        import re
        values = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        try:
            values = list(map(float, values))
        except:
            st.error("⚠️ OCR extraction failed. Please ensure values are clear in the image.")

        if len(values) >= 6:
            input_6 = np.array(values[:6]).reshape(1, -1)
            pred = model_6.predict(input_6)[0]
            st.success(f"🔍 Prediction: **{'Parkinson\'s Likely' if pred == 1 else 'Parkinson\'s Unlikely'}**")

            fig = go.Figure(data=[go.Bar(x=features_6, y=input_6.flatten())])
            fig.update_layout(title="📊 Feature Values (from OCR)")
            st.plotly_chart(fig)
        else:
            st.warning("❗ Please provide at least 6 numeric values in the image in order: fo, fhi, flo, jitter_percent, rap, ppe.")
