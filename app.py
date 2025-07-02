import streamlit as st
import numpy as np
import pickle
import easyocr
import cv2
from PIL import Image
import plotly.graph_objects as go

# Load model
model = pickle.load(open('parkinson_model.pkl', 'rb'))

st.title("🧠 Parkinson’s Disease Detection")
st.write("Upload patient's voice test data to detect Parkinson’s")

# ----------- Manual Input Section ------------
st.header("📝 Manual Input")
fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0, step=0.1)
fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0, step=0.1)
flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0, step=0.1)
jitter_percent = st.number_input("MDVP:Jitter(%)", min_value=0.0, step=0.001)
rap = st.number_input("MDVP:RAP", min_value=0.0, step=0.001)
ppe = st.number_input("PPE", min_value=0.0, step=0.001)

manual_input = np.array([[fo, fhi, flo, jitter_percent, rap, ppe]])

# Predict Button for Manual Input
if st.button("🔍 Predict from Manual Input"):
    prediction = model.predict(manual_input)
    if prediction[0] == 1:
        st.error("⚠️ The patient is likely to have Parkinson’s disease.")
    else:
        st.success("✅ The patient is unlikely to have Parkinson’s disease.")

# ----------- OCR Screenshot Upload Section ------------
st.header("📸 OCR: Upload Screenshot to Auto-Fill")

uploaded_image = st.file_uploader("Upload a screenshot of test results", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    img_array = np.array(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    reader = easyocr.Reader(['en'])
    with st.spinner("🔍 Extracting text from image..."):
        result = reader.readtext(img_array, detail=0)
        st.success("✅ Text extracted!")

    st.write("📄 Detected Text:")
    st.code("\n".join(result))

    # Extract numerical values
    try:
        values = [float(x) for x in result if x.replace('.', '', 1).isdigit()][:6]

        if len(values) == 6:
            st.success("✅ All 6 required feature values extracted!")

            st.subheader("🧪 Auto-Filled Features")
            fields = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
                      "MDVP:Jitter(%)", "MDVP:RAP", "PPE"]
            table = {fields[i]: values[i] for i in range(6)}
            st.table(table)

            # Radar Chart Option
            if st.checkbox("📈 Show Radar Chart"):
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=fields,
                    fill='toself',
                    name='Voice Features'
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True)),
                    showlegend=False,
                    title="📊 Radar Chart of Extracted Features"
                )
                st.plotly_chart(fig)

            # Predict using extracted values
            if st.button("🤖 Predict from Screenshot Data"):
                prediction = model.predict(np.array([values]))
                if prediction[0] == 1:
                    st.error("⚠️ The patient is likely to have Parkinson’s disease.")
                else:
                    st.success("✅ The patient is unlikely to have Parkinson’s disease.")
        else:
            st.warning(f"⚠️ Only {len(values)} valid numbers found. 6 required for prediction.")
    except Exception as e:
        st.error(f"❌ Error: {e}")
