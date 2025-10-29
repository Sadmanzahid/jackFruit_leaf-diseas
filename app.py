import streamlit as st
import onnxruntime as ort
import numpy as np
import pandas as pd
from PIL import Image

st.set_page_config(
    page_title="Jackfruit Leaf Disease Detector",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS styles
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #f7fff7 0%, #ffffff 100%); color: #0b3d2e; }
    .card { background: #ffffff; border-radius: 16px; padding: 18px; box-shadow: 0 6px 18px rgba(11,61,46,0.06); }
    .header { font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; padding: 8px 0 20px 0; }
    .prediction-badge { display:inline-block; padding:8px 14px; border-radius:999px; color:#fff; font-weight:600; }
    .healthy { background: #29c97e; }
    .burn { background: #f97373; }
    .red\\ rust { background: #f59e0b; }
    .spot { background: #6366f1; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load ONNX model
MODEL_PATH = "best_leaf_model_int8.onnx"
try:
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    model_load_error = None
except Exception as e:
    session = None
    model_load_error = str(e)

CLASS_NAMES = ["burn", "healthy", "red rust", "spot"]

# Sidebar
with st.sidebar:
    st.markdown("## About")
    st.write("Modernized UI for jackfruit leaf disease detection.")
    st.markdown("---")
    st.markdown("### Model")
    if model_load_error:
        st.error("Model failed to load.")
        st.caption(model_load_error)
    else:
        st.success("ONNX Model loaded ✅")
    st.markdown("---")
    st.markdown("### Tips")
    st.write("- Use clear images of the leaf.")
    st.write("- Prefer full-leaf photos with good lighting.")

# Header
st.markdown(
    """
    <div class="header">
      <h1>🌿 Jackfruit Leaf Disease Detection</h1>
      <p style="margin-top:-10px;color:#3b3b3b">
        Upload a photo of a jackfruit leaf and get a quick diagnosis.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded_img = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded_img is None:
        st.info("No image uploaded yet.")
        st.image("https://via.placeholder.com/400x300.png?text=Upload+an+image", use_column_width=True)
    else:
        image = Image.open(uploaded_img).convert("RGB")
        st.image(image, caption="Preview", use_column_width=True)
        st.markdown("**Preprocessing**")
        st.write("Resizing and normalizing")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Prediction")
    if session is None:
        st.warning("Model not available — cannot predict.")
    else:
        if uploaded_img is None:
            st.write("Upload an image to enable prediction.")
        else:
            if st.button("Predict", key="predict_btn"):
                # Preprocess image
                img_size = session.get_inputs()[0].shape[2]
                img = image.resize((img_size, img_size))
                img_array = np.array(img, dtype=np.float32) / 255.0

                # Convert NHWC -> NCHW
                img_array = np.transpose(img_array, (2, 0, 1))
                img_array = np.expand_dims(img_array, axis=0)

                # Run ONNX inference
                preds = session.run(None, {input_name: img_array})[0][0]

                idx = int(np.argmax(preds))
                conf = float(np.max(preds) * 100)
                label = CLASS_NAMES[idx]
                badge_class = label.replace(" ", "\\ ")

                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:12px;">'
                    f'<span class="prediction-badge {badge_class}">{label.upper()}</span>'
                    f'<div><strong style="font-size:18px">{conf:.2f}%</strong><div style="color:#6b7280">confidence</div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                prob_df = pd.DataFrame({"class": CLASS_NAMES, "probability": preds}).sort_values("probability")
                st.write("Class probabilities")
                st.bar_chart(data=prob_df.set_index("class")["probability"])

                if label == "healthy":
                    st.balloons()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Built with Streamlit • Model: ONNX format")
