"""
Project 4 - Path 2: Object Detection with MobileNet-SSD
DecodeLabs Industrial Training Kit

Streamlit web app: lets you upload an image and see the object detection
pipeline (from detector.py) run on it, with labeled bounding boxes and
confidence scores (>= 80% threshold only).

Run with:
    streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

from detector import (
    load_model,
    detect_objects,
    CONFIDENCE_THRESHOLD,
    PROTOTXT_PATH,
    MODEL_PATH,
)

st.set_page_config(page_title="Project 4 - Object Detection", layout="centered")

st.title("🔎 Project 4: Object Detection (MobileNet-SSD)")
st.caption("DecodeLabs Industrial Training Kit — Path 2: Object Detection")

st.markdown(
    """
    This app uses a **pre-trained** MobileNet-SSD model (Transfer Learning —
    no dataset or training required) to detect objects in an image.
    Only detections with confidence **≥ 80%** are shown, per project requirements.
    """
)

# ---------------------------------------------------------------------------
# Check that the model files exist before doing anything else
# ---------------------------------------------------------------------------
if not (os.path.exists(PROTOTXT_PATH) and os.path.exists(MODEL_PATH)):
    st.error(
        f"Model files not found.\n\n"
        f"Please place these two files in the `models/` folder:\n"
        f"- `{PROTOTXT_PATH}`\n"
        f"- `{MODEL_PATH}`\n\n"
        f"See README.md for download instructions."
    )
    st.stop()


@st.cache_resource
def get_model():
    return load_model()


net = get_model()

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.5,
    max_value=1.0,
    value=CONFIDENCE_THRESHOLD,
    step=0.05,
    help="Project 4 requires a minimum of 0.80 (80%)."
)

# ---------------------------------------------------------------------------
# Image upload
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload a sample image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    st.subheader("Original Image")
    st.image(pil_image, use_container_width=True)

    with st.spinner("Running detection..."):
        output_image, results = detect_objects(
            image_bgr, net=net, confidence_threshold=confidence_threshold
        )

    output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    st.subheader("Detection Output")
    st.image(output_rgb, use_container_width=True)

    st.subheader("Detected Objects")
    if results:
        for det in results:
            st.write(
                f"**{det['label']}** — confidence: {det['confidence']*100:.2f}% "
                f"— box: {det['box']}"
            )
    else:
        st.info("No objects detected above the confidence threshold.")
else:
    st.info("Upload an image to run object detection.")
