import streamlit as st
import numpy as np
import pickle
import easyocr
import re
from PIL import Image
import plotly.graph_objects as go
import pandas as pd
from streamlit_theme_toggle import st_theme_toggle

# Load model
model = pickle.load(open('parkinson_model.pkl', 'rb'))

# OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Theme toggle
theme = st_theme_toggle()
if theme == "Dark":
    st.markdown("<style>body { background-color: #0E1117; color: white; }</style>", unsafe_allow_html=True)

st.set_page_config(page_title="Parkinson's Detection", layout="centered")
st.title("🧠 Parkinson’s Disease Detection")
st.markdown("Upload a voice report screenshot or enter parameters to detect Parkinson’s disease.")

# --- OCR Section ---
st.subheader("📸 OCR: Upload Screenshot for Auto-Fill")
uploaded_file = st.file_uploader("Upload Report Image (PNG/JPG)", type=["png", "jpg", "jpeg"])
ocr_values = {}

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Screenshot", use_column_width=True)

    result = reader.readtext(np.array(image), detail=0)
    extracted_text = ' '.join(result)

    def extract_value(pattern, text):
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

    ocr_values = {key: extract_value(pat, extracted_text) for key, pat in patterns.items()}

# --- Manual Input ---
st.subheader("✍️ Manual Input (Auto-filled if OCR used)")

# Original features
fo = st.number_input("MDVP:Fo(Hz)", value=ocr_values.get('fo', 0.0), min_value=0.0, step=0.1)
fhi = st.number_input("MDVP:Fhi(Hz)", value=ocr_values.get('fhi', 0.0), min_value=0.0, step=0.1)
flo = st.number_input("MDVP:Flo(Hz)", value=ocr_values.get('flo', 0.0), min_value=0.0, step=0.1)
jitter_percent = st.number_input("MDVP:Jitter(%)", value=ocr_values.get('jitter_percent', 0.0), min_value=0.0, step=0.001)
rap = st.number_input("MDVP:RAP", value=ocr_values.get('rap', 0.0), min_value=0.0, step=0.001)
ppe = st.number_input("PPE", value=ocr_values.get('ppe', 0.0), min_value=0.0, step=0.001)

# Additional 10 features
st.markdown("#### 🔧 Additional Parameters")
shimmer = st.number_input("Shimmer", min_value=0.0, step=0.001)
apq = st.number_input("APQ", min_value=0.0, step=0.001)
dfa = st.number_input("DFA", min_value=0.0, step=0.001)
spread1 = st.number_input("Spread1", min_value=-10.0, step=0.01)
spread2 = st.number_input("Spread2", min_value=-10.0, step=0.01)
mdvp_shimmer_db = st.number_input("MDVP:Shimmer(dB)", min_value=0.0, step=0.01)
hnr = st.number_input("HNR", min_value=0.0, step=0.1)
jitter_abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.0, step=0.00001)
d2 = st.number_input("D2", min_value=0.0, step=0.01)
ddp = st.number_input("DDP", min_value=0.0, step=0.001)

input_data = np.array([[fo, fhi, flo, jitter_percent, rap, ppe,
                        shimmer, apq, dfa, spread1, spread2, mdvp_shimmer_db,
                        hnr, jitter_abs, d2, ddp]])

# --- Prediction Section ---
if st.button("🔍 Detect Parkinson's"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]  # Probability of class 1

    if prediction[0] == 1:
        st.error(f"⚠️ Parkinson’s disease likely detected.\n🧪 Model Confidence: **{probability*100:.2f}%**")
    else:
        st.success(f"✅ No signs of Parkinson’s disease.\n🧪 Model Confidence: **{(1 - probability)*100:.2f}%**")

    # Radar Chart
    labels = ['Fo', 'Fhi', 'Flo', 'Jitter(%)', 'RAP', 'PPE']
    values = [fo, fhi, flo, jitter_percent, rap, ppe]
    max_vals = [300, 400, 300, 1.0, 0.2, 1.0]
    scaled_vals = [v / m if m else 0 for v, m in zip(values, max_vals)]

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
        go.Bar(x=labels, y=values, marker_color='lightskyblue')
    ])
    bar_fig.update_layout(title="📊 Feature Values", xaxis_title="Feature", yaxis_title="Value")
    st.plotly_chart(bar_fig)

    # CSV Export
    st.markdown("### ⬇️ Download Result")
    result_df = pd.DataFrame(input_data, columns=[
        'Fo', 'Fhi', 'Flo', 'Jitter(%)', 'RAP', 'PPE',
        'Shimmer', 'APQ', 'DFA', 'Spread1', 'Spread2',
        'MDVP:Shimmer(dB)', 'HNR', 'MDVP:Jitter(Abs)', 'D2', 'DDP'
    ])
    result_df['Prediction'] = ['Parkinson' if prediction[0] == 1 else 'Healthy']
    result_df['Confidence'] = f"{probability*100:.2f}%" if prediction[0] == 1 else f"{(1 - probability)*100:.2f}%"

    st.dataframe(result_df)
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download as CSV", data=csv, file_name="parkinsons_prediction.csv", mime="text/csv")
