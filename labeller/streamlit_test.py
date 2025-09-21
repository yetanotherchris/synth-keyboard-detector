#!/usr/bin/env python3
"""
Streamlit Interface for YOLOv8 Piano Keyboard Detection Testing

Web interface for testing YOLOv8 piano keyboard detection models using
functions from the existing test.py script for consistency and maintainability.

Features:
- Model file upload or local selection  
- Single image testing with inline results
- Batch image processing with statistics
- Adjustable detection parameters
- Results visualization and download
"""

import streamlit as st
import os
import shutil
import tempfile
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
from ultralytics import YOLO
import sys

# Import functions from existing test.py script
from test import (
    load_model, 
    process_single_image, 
    process_batch_images
)

# Set page configuration
st.set_page_config(
    page_title="Piano Keyboard Detection - Testing",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None
if 'uploaded_model' not in st.session_state:
    st.session_state.uploaded_model = None


def create_temp_directory():
    """Create a temporary directory for uploaded files"""
    if st.session_state.temp_dir is None or not os.path.exists(st.session_state.temp_dir):
        temp_base = os.path.join(os.path.dirname(__file__), '..', 'temp')
        os.makedirs(temp_base, exist_ok=True)
        st.session_state.temp_dir = tempfile.mkdtemp(dir=temp_base)
    return st.session_state.temp_dir


def cleanup_temp_directory():
    """Clean up temporary directory"""
    if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
        try:
            shutil.rmtree(st.session_state.temp_dir)
            st.session_state.temp_dir = None
            st.success("🧹 Temporary files cleaned up")
        except Exception as e:
            st.error(f"Error cleaning up temporary files: {e}")


def save_uploaded_image(uploaded_file, temp_dir):
    """Save uploaded image to temp directory and return path"""
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def convert_result_for_display(result_data):
    """Convert result from test.py format to display format"""
    detections = []
    
    for detection in result_data.get('detections', []):
        bbox = detection['bbox']
        detections.append({
            'confidence': detection['confidence'],
            'bbox': bbox,
            'width': int(bbox[2] - bbox[0]),
            'height': int(bbox[3] - bbox[1])
        })
    
    return detections


def display_single_image_results(image_path, result_data):
    """Display results for single image testing using full width"""
    # Load and display annotated image
    if os.path.exists(image_path):
        # Read the processed image (with annotations)
        img = cv2.imread(image_path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Display results in full width
            st.subheader("🎯 Detection Results")
            st.image(img_rgb, caption="Detection Results", use_column_width=True)
            
            # Show statistics
            st.subheader("📊 Detection Statistics")
            if result_data['detection_count'] > 0:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Detections Found", result_data['detection_count'])
                with col2:
                    st.metric("Max Confidence", f"{result_data['max_confidence']:.3f}")
                with col3:
                    st.metric("Avg Confidence", f"{result_data['avg_confidence']:.3f}")
                with col4:
                    # Calculate additional stats
                    confidences = [d['confidence'] for d in result_data['detections']]
                    if len(confidences) > 1:
                        st.metric("Min Confidence", f"{min(confidences):.3f}")
                    else:
                        st.metric("Min Confidence", f"{confidences[0]:.3f}")
                
                # Detection details in expandable sections
                with st.expander("🔍 Detailed Detection Information", expanded=False):
                    for i, detection in enumerate(result_data['detections']):
                        st.subheader(f"Detection {i+1}")
                        col_a, col_b = st.columns(2)
                        
                        bbox = detection['bbox']
                        width = int(bbox[2] - bbox[0])
                        height = int(bbox[3] - bbox[1])
                        
                        with col_a:
                            st.write(f"**Confidence:** {detection['confidence']:.3f}")
                            st.write(f"**Class:** {detection['class_name']}")
                            st.write(f"**Width:** {width}px")
                            st.write(f"**Height:** {height}px")
                        with col_b:
                            st.write(f"**Bounding box:** [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                            st.write(f"**Area:** {width * height}px²")
                            st.write(f"**Aspect ratio:** {width/height:.2f}")
                        
                        if i < len(result_data['detections']) - 1:
                            st.divider()
            else:
                st.warning("❌ No detections found")
                st.info("💡 Try adjusting the confidence threshold or using a different image")


def display_batch_results(all_results, output_dir):
    """Display batch processing results with statistics"""
    st.subheader("📊 Batch Processing Results")
    
    # Summary statistics
    total_images = len(all_results)
    images_with_detections = len([r for r in all_results if r['detection_count'] > 0])
    total_detections = sum([r['detection_count'] for r in all_results])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Images", total_images)
    with col2:
        st.metric("Images with Detections", images_with_detections)
    with col3:
        st.metric("Detection Rate", f"{images_with_detections/total_images*100:.1f}%")
    with col4:
        st.metric("Total Detections", total_detections)
    
    # Results table
    st.subheader("📋 Results Summary")
    results_data = []
    for result in all_results:
        results_data.append({
            'Filename': result['image_name'],
            'Detections': result['detection_count'],
            'Max Confidence': f"{result['max_confidence']:.3f}" if result['max_confidence'] > 0 else "N/A",
            'Avg Confidence': f"{result['avg_confidence']:.3f}" if result['avg_confidence'] > 0 else "N/A"
        })
    
    df = pd.DataFrame(results_data)
    st.dataframe(df, use_container_width=True)
    
    # Confidence distribution chart
    if any(r['max_confidence'] > 0 for r in all_results):
        st.subheader("📈 Confidence Distribution")
        
        all_confidences = []
        for result in all_results:
            for detection in result['detections']:
                all_confidences.append(detection['confidence'])
        
        if all_confidences:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(all_confidences, bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax.set_xlabel('Confidence')
            ax.set_ylabel('Count')
            ax.set_title('Detection Confidence Distribution')
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    # Image gallery with results
    st.subheader("🖼️ Results Gallery")
    
    # Grid view of results (show annotated images if they exist)
    cols_per_row = 2
    for i in range(0, len(all_results), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            if i + j < len(all_results):
                result = all_results[i + j]
                
                with col:
                    # Try to load and display the annotated image
                    output_path = result.get('output_path')
                    if output_path and os.path.exists(output_path):
                        annotated_img = cv2.imread(output_path)
                        if annotated_img is not None:
                            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                            st.image(
                                annotated_img_rgb, 
                                caption=f"{result['image_name']} ({result['detection_count']} detections)",
                                use_column_width=True
                            )
                    else:
                        st.write(f"📄 {result['image_name']}")
                    
                    if result['detection_count'] > 0:
                        st.info(f"🎯 Max confidence: {result['max_confidence']:.3f}")
                    else:
                        st.warning("❌ No detections")


def model_selection_section():
    """Model selection interface"""
    st.subheader("🤖 Model Selection")
    
    model_source = st.radio(
        "Choose model source:",
        ["Upload model file", "Use existing local file"],
        horizontal=True
    )
    
    model_path = None
    
    if model_source == "Upload model file":
        uploaded_model = st.file_uploader(
            "Upload YOLOv8 model (.pt file)",
            type=['pt'],
            help="Upload a trained YOLOv8 model file"
        )
        
        if uploaded_model:
            temp_dir = create_temp_directory()
            model_path = os.path.join(temp_dir, uploaded_model.name)
            with open(model_path, 'wb') as f:
                f.write(uploaded_model.getbuffer())
            st.success(f"✅ Model uploaded: {uploaded_model.name}")
            st.session_state.uploaded_model = model_path
            
    else:
        local_model_path = st.text_input(
            "Model file path:",
            value="./training_output/default/weights/best.pt",
            help="Path to your local YOLOv8 model file (.pt)"
        )
        
        if local_model_path and os.path.exists(local_model_path):
            model_path = local_model_path
            st.success(f"✅ Using local model: {model_path}")
        elif local_model_path:
            st.error(f"❌ File does not exist: {local_model_path}")
    
    return model_path


def detection_parameters_section():
    """Detection parameters configuration"""
    st.subheader("⚙️ Detection Parameters")
    
    conf_threshold = st.slider("Confidence threshold:", 0.0, 1.0, 0.25, 0.05)
    iou_threshold = st.slider("IoU threshold:", 0.0, 1.0, 0.45, 0.05)
    
    output_dir = st.text_input(
        "Output directory:",
        value="./test_results",
        help="Directory to save test results"
    )
    
    return conf_threshold, iou_threshold, output_dir


def main():
    """Main application"""
    st.title("🎹 Piano Keyboard Detection - Testing")
    st.markdown("Web interface for testing YOLOv8 piano keyboard detection models")
    st.markdown("*Uses functions from the original test.py script for consistency*")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ Tools")
        
        if st.button("🧹 Clean Temp Files"):
            cleanup_temp_directory()
        
        st.markdown("---")
        
        st.header("ℹ️ About")
        st.markdown("""
        This application uses the existing test.py functions for consistency:
        
        **Core Functions:**
        - `load_model()` - Model loading
        - `process_single_image()` - Single image processing
        - `process_batch_images()` - Batch processing
        
        **Features:**
        - Upload model files or use local models
        - Single image and batch testing with full-width display
        - Real-time results with inline images
        - Detailed statistics and visualizations
        """)
    
    # Main content
    with st.container():
        # Model selection
        model_path = model_selection_section()
        
        if model_path:
            # Detection parameters
            conf_threshold, iou_threshold, output_dir = detection_parameters_section()
            
            st.markdown("---")
            
            # Testing mode selection
            test_mode = st.radio(
                "Choose testing mode:",
                ["Single Image", "Batch Processing"],
                horizontal=True
            )
            
            if test_mode == "Single Image":
                st.subheader("🖼️ Single Image Testing")
                
                uploaded_image = st.file_uploader(
                    "Upload test image",
                    type=['jpg', 'jpeg', 'png', 'bmp'],
                    help="Upload an image to test the model"
                )
                
                if uploaded_image:
                    # Display original image in full width
                    image = Image.open(uploaded_image)
                    st.image(image, caption="Original Image", use_column_width=True)
                    
                    # Center the run button
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("🔍 Run Detection", type="primary", use_container_width=True):
                            try:
                                # Save uploaded image to temp directory
                                temp_dir = create_temp_directory()
                                temp_image_path = save_uploaded_image(uploaded_image, temp_dir)
                                
                                # Load model using original function
                                with st.spinner("Loading model..."):
                                    model = load_model(model_path)
                                
                                # Process image using original function
                                with st.spinner("Running detection..."):
                                    result_data = process_single_image(
                                        model=model,
                                        image_path=temp_image_path,
                                        conf_threshold=conf_threshold,
                                        iou_threshold=iou_threshold,
                                        output_dir=output_dir,
                                        show_image=False
                                    )
                                
                                # Display results
                                if result_data.get('output_path') and os.path.exists(result_data['output_path']):
                                    display_single_image_results(result_data['output_path'], result_data)
                                else:
                                    st.error("Failed to process image or save results")
                                
                            except Exception as e:
                                st.error(f"❌ Detection failed: {str(e)}")
            
            else:  # Batch Processing
                st.subheader("📁 Batch Image Testing")
                
                uploaded_images = st.file_uploader(
                    "Upload test images",
                    type=['jpg', 'jpeg', 'png', 'bmp'],
                    accept_multiple_files=True,
                    help="Upload multiple images for batch processing"
                )
                
                if uploaded_images:
                    st.info(f"📸 {len(uploaded_images)} images uploaded")
                    
                    # Show preview
                    if len(uploaded_images) > 0:
                        st.subheader("📷 Image Preview")
                        preview_cols = st.columns(min(4, len(uploaded_images)))
                        for i, img_file in enumerate(uploaded_images[:4]):
                            with preview_cols[i]:
                                image = Image.open(img_file)
                                st.image(image, caption=img_file.name, use_column_width=True)
                        
                        if len(uploaded_images) > 4:
                            st.info(f"... and {len(uploaded_images) - 4} more images")
                    
                    if st.button("🚀 Run Batch Detection", type="primary"):
                        try:
                            # Create temp directory for images
                            temp_dir = create_temp_directory()
                            temp_images_dir = os.path.join(temp_dir, 'batch_images')
                            os.makedirs(temp_images_dir, exist_ok=True)
                            
                            # Save all uploaded images
                            for uploaded_image in uploaded_images:
                                save_uploaded_image(uploaded_image, temp_images_dir)
                            
                            # Load model
                            with st.spinner("Loading model..."):
                                model = load_model(model_path)
                            
                            # Process batch using original function
                            with st.spinner("Processing batch..."):
                                all_results = process_batch_images(
                                    model=model,
                                    images_dir=temp_images_dir,
                                    conf_threshold=conf_threshold,
                                    iou_threshold=iou_threshold,
                                    output_dir=output_dir
                                )
                            
                            # Display results
                            if all_results:
                                display_batch_results(all_results, output_dir)
                            else:
                                st.error("Batch processing failed")
                                
                        except Exception as e:
                            st.error(f"❌ Batch processing failed: {str(e)}")


if __name__ == "__main__":
    main()