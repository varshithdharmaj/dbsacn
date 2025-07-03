import streamlit as st
import numpy as np
import pickle
import easyocr
import re
from PIL import Image
import plotly.graph_objects as go

# Load model
model = pickle.load(open('parkinson_model.pkl', 'rb'))

st.set_page_config(page_title="Parkinson’s Detection", layout="centered")

st.title("🧠 Parkinson’s Disease Detection")
st.write("Upload patient's voice report or enter values manually to detect Parkinson’s")

# --- OCR SECTION ---
st.markdown("## 📸 OCR: Upload Screenshot to Auto-Fill")
uploaded_file = st.file_uploader("🖼️ Upload a screenshot of your report", type=['png', 'jpg', 'jpeg'])

ocr_values = {}

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Screenshot", use_column_width=True)

    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(np.array(image), detail=0)
    extracted_text = ' '.join(result)
    
    # Extract features using regex
    def extract_value(pattern, text):
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1))
            except:
                return None
        return None

    patterns = {
        'fo': r'Fo\(Hz\)[^\d]*([\d.]+)',
        'fhi': r'Fhi\(Hz\)[^\d]*([\d.]+)',
        'flo': r'Flo\(Hz\)[^\d]*([\d.]+)',
        'jitter_percent': r'Jitter\(.*?%\)[^\d]*([\d.]+)',
        'rap': r'RAP[^\d]*([\d.]+)',
        'ppe': r'PPE[^\d]*([\d.]+)'
    }

    ocr_values = {key: extract_value(pat, extracted_text) for key, pat in patterns.items()}

    st.subheader("📊 Extracted Values")
    for key, value in ocr_values.items():
        st.write(f"**{key}**: {value if value is not None else 'Not found'}")

# --- MANUAL INPUT SECTION ---
st.markdown("## 📝 Manual Input (Optional or for Correction)")

fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0, step=0.1, value=ocr_values.get('fo', 0.0))
fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0, step=0.1, value=ocr_values.get('fhi', 0.0))
flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0, step=0.1, value=ocr_values.get('flo', 0.0))
jitter_percent = st.number_input("MDVP:Jitter(%)", min_value=0.0, step=0.001, value=ocr_values.get('jitter_percent', 0.0))
rap = st.number_input("MDVP:RAP", min_value=0.0, step=0.001, value=ocr_values.get('rap', 0.0))
ppe = st.number_input("PPE", min_value=0.0, step=0.001, value=ocr_values.get('ppe', 0.0))

input_data = np.array([[fo, fhi, flo, jitter_percent, rap, ppe]])

# --- PREDICTION ---
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("⚠️ The patient is likely to have Parkinson’s disease.")
    else:
        st.success("✅ The patient is unlikely to have Parkinson’s disease.")

    # --- VISUALIZATION ---
    st.markdown("### 📈 Feature Visualization")
    features = ["Fo", "Fhi", "Flo", "Jitter(%)", "RAP", "PPE"]
    values = input_data[0]

    fig = go.Figure(data=[
        go.Bar(name='Values', x=features, y=values, marker_color='lightskyblue')
    ])
    fig.update_layout(title="Patient's Feature Values", yaxis_title="Value", xaxis_title="Feature")
    st.plotly_chart(fig)
