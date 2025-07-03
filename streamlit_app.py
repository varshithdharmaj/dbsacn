import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pytesseract
from PIL import Image
import re
import plotly.graph_objects as go

# Optional: Set this if tesseract is not found
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Load models
model_6 = pickle.load(open("parkinson_model_6.pkl", "rb"))
model_22 = pickle.load(open("parkinson_model_22.pkl", "rb"))

features_6 = ['fo', 'fhi', 'flo', 'jitter_percent', 'rap', 'ppe']
features_22 = features_6 + [f'feature_{i}' for i in range(7, 23)]

st.set_page_config(page_title="Parkinson's Predictor", layout="centered")
st.title("🧠 Parkinson's Disease Predictor")

# Model selector
model_type = st.radio("Select Model:", ["6-feature model", "22-feature model"])

# Input method selector
input_mode = st.selectbox("Choose Input Method:", [
    "Upload CSV",
    "Manual Entry Form",
    "Comma-Separated Text",
    "Upload Image (OCR)" if model_type == "6-feature model" else "Upload Image (OCR) (Disabled)"
])

required_features = features_6 if model_type == "6-feature model" else features_22
model = model_6 if model_type == "6-feature model" else model_22

# --- CSV Upload ---
if input_mode == "Upload CSV":
    uploaded = st.file_uploader("📄 Upload CSV file", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df)

        if all(f in df.columns for f in required_features):
            X = df[required_features].iloc[-1:].values
            prediction = model.predict(X)[0]
            st.success(f"✅ Prediction: **{'Parkinson\'s Likely' if prediction == 1 else 'Parkinson\'s Unlikely'}**")

            fig = go.Figure(data=[go.Bar(x=required_features, y=X.flatten())])
            fig.update_layout(title="📊 Feature Values")
            st.plotly_chart(fig)
        else:
            st.error("❌ Required features not found in uploaded file.")

# --- Manual Entry Form ---
elif input_mode == "Manual Entry Form":
    st.info("Enter each value in a separate field:")

    input_vals = []
    for f in required_features:
        val = st.text_input(f"{f}", key=f)
        try:
            input_vals.append(float(val))
        except:
            input_vals.append(None)

    if st.button("Predict"):
        if None in input_vals:
            st.error("⚠️ Please enter valid numeric values for all features.")
        else:
            X = np.array(input_vals).reshape(1, -1)
            prediction = model.predict(X)[0]
            st.success(f"🔍 Prediction: **{'Parkinson\'s Likely' if prediction == 1 else 'Parkinson\'s Unlikely'}**")

            fig = go.Figure(data=[go.Bar(x=required_features, y=X.flatten())])
            fig.update_layout(title="📊 Feature Values")
            st.plotly_chart(fig)

# --- Comma-Separated Text Input ---
elif input_mode == "Comma-Separated Text":
    st.info(f"Paste values in order ({len(required_features)} features), separated by commas:")

    user_input = st.text_area("Enter comma-separated values:")
    if st.button("Predict"):
        try:
            values = list(map(float, user_input.strip().split(',')))
            if len(values) != len(required_features):
                st.error(f"❌ Please provide exactly {len(required_features)} numeric values.")
            else:
                X = np.array(values).reshape(1, -1)
                prediction = model.predict(X)[0]
                st.success(f"🔍 Prediction: **{'Parkinson\'s Likely' if prediction == 1 else 'Parkinson\'s Unlikely'}**")

                fig = go.Figure(data=[go.Bar(x=required_features, y=X.flatten())])
                fig.update_layout(title="📊 Feature Values")
                st.plotly_chart(fig)
        except:
            st.error("⚠️ Invalid input format. Ensure values are numeric and separated by commas.")

# --- OCR Input for 6-Feature Model Only ---
elif input_mode.startswith("Upload Image") and model_type == "6-feature model":
    image = st.file_uploader("🖼️ Upload image with 6 feature values (in order)", type=["png", "jpg", "jpeg"])
    if image:
        img = Image.open(image)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        try:
            text = pytesseract.image_to_string(img)
            st.text_area("📋 Extracted Text from Image", text, height=150)

            values = re.findall(r"[-+]?\d*\.\d+|\d+", text)
            values = list(map(float, values))

            if len(values) >= 6:
                X = np.array(values[:6]).reshape(1, -1)
                prediction = model_6.predict(X)[0]
                st.success(f"🔍 Prediction: **{'Parkinson\'s Likely' if prediction == 1 else 'Parkinson\'s Unlikely'}**")

                fig = go.Figure(data=[go.Bar(x=features_6, y=X.flatten())])
                fig.update_layout(title="📊 Feature Values (from OCR)")
                st.plotly_chart(fig)
            else:
                st.warning("❗ OCR failed to extract enough numeric values (need 6).")
        except pytesseract.pytesseract.TesseractNotFoundError:
            st.error("❌ Tesseract-OCR not found. Please install it or set `tesseract_cmd` path.")
        except Exception as e:
            st.error(f"❌ OCR processing failed: {str(e)}")
