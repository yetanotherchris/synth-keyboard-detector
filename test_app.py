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


def _display_processing_stages(result: Dict[str, object]) -> None:
    """Display all 8 processing stages from first-steps.md"""
    debug = result.get("debug") or {}
    
    st.subheader("8 Processing Stages from first-steps.md")
    
    # Create columns for stage display
    col1, col2 = st.columns(2)
    
    # Stage 1: Crop the bottom 50% of the uploaded image
    if "original" in debug:
        with col1:
            st.write("**Stage 1: Original Image**")
            st.image(bgr_to_rgb(debug["original"]), caption="Original uploaded image", use_column_width=True)
    
    # Stage 1b: Bottom 50% crop
    if "stage1_crop" in debug:
        with col2:
            st.write("**Stage 1: Bottom 50% Crop**")
            st.image(bgr_to_rgb(debug["stage1_crop"]), caption="Cropped to bottom 50%", use_column_width=True)
    
    # Stage 3: Keyboard height crop
    if "stage3_crop" in debug:
        with col1:
            st.write("**Stage 3: Keyboard Height**")
            st.image(bgr_to_rgb(debug["stage3_crop"]), caption="Cropped to keyboard height", use_column_width=True)
    
    # Stage 3 Highlighted: Keyboard height with first key highlighted
    if "stage3_highlighted" in debug and debug["stage3_highlighted"] is not None:
        with col2:
            st.write("**Stage 3: Keyboard Height (Highlighted)**")
            st.image(bgr_to_rgb(debug["stage3_highlighted"]), caption="First white key highlighted using width from Stage 4", use_column_width=True)
    
    # Stage 4: White keys only (bottom 30%)
    if "stage4_crop" in debug:
        with col1:
            st.write("**Stage 4: White Keys Only**")
            st.image(bgr_to_rgb(debug["stage4_crop"]), caption="Bottom 30% - white keys only", use_column_width=True)
    
    # Stage 6: Green solid overlay
    if "stage6_highlighted" in debug and debug["stage6_highlighted"] is not None:
        with col2:
            st.write("**Stage 6: Green Solid Overlay**")
            st.image(bgr_to_rgb(debug["stage6_highlighted"]), caption="First white key highlighted with solid green (50% opacity)", use_column_width=True)


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


st.title("Piano Key Detector - First Steps Implementation")
st.markdown("Upload a top-down image of a piano or synth keyboard to see the 8-step detection process from first-steps.md.")

uploaded_file = st.file_uploader("Upload keyboard image", type=["jpg", "jpeg", "png"])

st.sidebar.header("Detection Parameters")
blur_kernel = st.sidebar.slider("Gaussian Blur Kernel", 3, 21, 5, step=2)
canny_low = st.sidebar.slider("Canny Low Threshold", 20, 200, 80)
canny_high = st.sidebar.slider("Canny High Threshold", 50, 300, 180)
adaptive_block = st.sidebar.slider("Adaptive Threshold Block Size", 3, 75, 35, step=2)
adaptive_c = st.sidebar.slider("Adaptive Threshold C", -20, 20, 5)
projection_peak = st.sidebar.slider("Projection Peak Ratio", 0.1, 0.9, 0.55, 0.01)
min_white_ratio = st.sidebar.slider("Min White Key Width Ratio", 0.005, 0.05, 0.01, 0.001)
max_white_ratio = st.sidebar.slider("Max White Key Width Ratio", 0.05, 0.4, 0.2, 0.01)

if uploaded_file is None:
    st.info("Upload an image to begin the 8-step detection process.")
    st.markdown("""
    ### The 8-Step Process:
    1. **Crop the bottom 50%** of the uploaded image
    2. **Find the white keys** in this image
    3. **Crop the image to the height** of these keys, the area they are inside
    4. **Further crop** the image produced in stage 3, to the bottom 30% of this image
    5. **You should have the white keys only** now
    6. **Find the first white key** and highlight it in green, using 50% opacity solid green fill
    7. **Highlight the right side boundary** of this key using a dotted black line, 50% opacity
    8. **Display each of these stages** as images in streamlit
    """)
else:
    image = _load_uploaded_image(uploaded_file)
    if image is None:
        st.error("Unable to read the uploaded image.")
    else:
        detector = PianoKeyDetector()
        result = detector.detect_keys(
            image,
            blur_kernel_size=blur_kernel,
            canny_low=canny_low,
            canny_high=canny_high,
            adaptive_threshold_block_size=adaptive_block,
            adaptive_threshold_c=adaptive_c,
            projection_peak_ratio=projection_peak,
            min_white_key_width_ratio=min_white_ratio,
            max_white_key_width_ratio=max_white_ratio,
        )

        # Display the 8 processing stages
        _display_processing_stages(result)
        
        # Detection summary
        st.subheader("Detection Summary")
        summary = {
            "First White Key Found": result.get("found", False),
            "Total Keys": result.get("total_keys"),
            "Confidence": result.get("confidence"),
        }
        st.json(summary)

        # First white key details
        if result.get("first_white_key"):
            with st.expander("First White Key Details"):
                key_data = result["first_white_key"]
                st.write(f"**Bounding Box (x, y, width, height):** {key_data['bbox']}")
                st.write(f"**Center:** {key_data['center']}")
                st.write(f"**Confidence:** {key_data['confidence']:.3f}")
                st.write(f"**Aspect Ratio:** {key_data.get('aspect_ratio', 'N/A')}")
                st.write(f"**Area Ratio:** {key_data.get('area_ratio', 'N/A')}")
        else:
            st.warning("No white key detected. Try adjusting the parameters.")
