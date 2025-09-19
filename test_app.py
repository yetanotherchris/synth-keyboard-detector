"""Streamlit playground for the Piano Key Detector."""
from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np
import streamlit as st

from piano_key_detector import PianoKeyDetector
from piano_key_detector.utils import bgr_to_rgb

st.set_page_config(page_title="Piano Key Detector", layout="wide")


def _load_uploaded_image(uploaded_file) -> Optional[np.ndarray]:
    if uploaded_file is None:
        return None
    data = uploaded_file.read()
    if not data:
        return None
    np_data = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    uploaded_file.seek(0)
    return image


def _draw_detection_overlay(image: np.ndarray, result: Dict[str, object]) -> np.ndarray:
    overlay = image.copy()
    white_color = (0, 255, 0)
    black_color = (255, 0, 0)
    bbox_color = (0, 165, 255)

    for key in result.get("white_keys", []):
        x, y, w, h = key["bbox"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), white_color, 2)

    for key in result.get("black_keys", []):
        x, y, w, h = key["bbox"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), black_color, 2)

    if result.get("bounding_box"):
        x, y, w, h = result["bounding_box"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), bbox_color, 2)

    return overlay


def _display_debug_images(result: Dict[str, object], show_edges: bool, show_binary: bool) -> None:
    debug = result.get("debug") or {}
    if show_edges and "edges" in debug:
        st.image(debug["edges"], caption="Canny edges", clamp=True, channels="GRAY")
    if show_binary and "binary" in debug:
        st.image(debug["binary"], caption="Binary mask", clamp=True, channels="GRAY")
    if "column_projection" in debug:
        projection = debug["column_projection"]
        if projection.size:
            st.line_chart(projection)


st.title("Piano Key Detector - Test Interface")
st.markdown("Upload a top-down image of a piano or synth keyboard to inspect detections.")

uploaded_file = st.file_uploader("Upload keyboard image", type=["jpg", "jpeg", "png"])

st.sidebar.header("Detection Parameters")
keyboard_type_option = st.sidebar.selectbox(
    "Force Keyboard Type", ["Auto", "25", "37", "49", "61", "72"], index=0
)
blur_kernel = st.sidebar.slider("Gaussian Blur Kernel", 3, 21, 5, step=2)
canny_low = st.sidebar.slider("Canny Low Threshold", 20, 200, 80)
canny_high = st.sidebar.slider("Canny High Threshold", 50, 300, 180)
adaptive_block = st.sidebar.slider("Adaptive Threshold Block Size", 3, 75, 35, step=2)
adaptive_c = st.sidebar.slider("Adaptive Threshold C", -20, 20, 5)
projection_peak = st.sidebar.slider("Projection Peak Ratio", 0.1, 0.9, 0.45, 0.01)
min_white_ratio = st.sidebar.slider("Min White Key Width Ratio", 0.005, 0.05, 0.01, 0.001)
max_white_ratio = st.sidebar.slider("Max White Key Width Ratio", 0.05, 0.4, 0.2, 0.01)
black_darkness = st.sidebar.slider("Black Key Darkness Threshold", 0.1, 0.8, 0.35, 0.01)
show_edges = st.sidebar.checkbox("Show Edge Detection", False)
show_binary = st.sidebar.checkbox("Show Binary Mask", False)

force_keyboard_type = None if keyboard_type_option == "Auto" else keyboard_type_option

if uploaded_file is None:
    st.info("Upload an image to begin.")
else:
    image = _load_uploaded_image(uploaded_file)
    if image is None:
        st.error("Unable to read the uploaded image.")
    else:
        detector = PianoKeyDetector()
        result = detector.detect_keys(
            image,
            force_keyboard_type=force_keyboard_type,
            blur_kernel_size=blur_kernel,
            canny_low=canny_low,
            canny_high=canny_high,
            adaptive_threshold_block_size=adaptive_block,
            adaptive_threshold_c=adaptive_c,
            projection_peak_ratio=projection_peak,
            min_white_key_width_ratio=min_white_ratio,
            max_white_key_width_ratio=max_white_ratio,
            black_key_darkness_threshold=black_darkness,
        )

        debug = result.get("debug") or {}
        original = debug.get("original", image)
        overlay = _draw_detection_overlay(original, result)

        col1, col2 = st.columns(2)
        col1.image(bgr_to_rgb(original), caption="Original", use_column_width=True)
        col2.image(bgr_to_rgb(overlay), caption="Detections", use_column_width=True)

        st.subheader("Detection Summary")
        summary = {
            "Keyboard Type": result.get("keyboard_type"),
            "Total Keys": result.get("total_keys"),
            "White Keys Detected": len(result.get("white_keys", [])),
            "Black Keys Detected": len(result.get("black_keys", [])),
            "Confidence": result.get("confidence"),
        }
        st.json(summary)

        with st.expander("White key data"):
            if result.get("white_keys"):
                st.dataframe(result["white_keys"])
            else:
                st.write("No white keys detected.")

        with st.expander("Black key data"):
            if result.get("black_keys"):
                st.dataframe(result["black_keys"])
            else:
                st.write("No black keys detected.")

        _display_debug_images(result, show_edges, show_binary)
