#!/usr/bin/env python3
"""
Streamlit Interface for YOLOv8 Piano Keyboard Detection Training

Web interface for training YOLOv8 piano keyboard detection models using
functions from the existing train.py script for consistency and maintainability.

Features:
- Dataset upload (ZIP files) or local folder selection
- Automatic dataset validation using original validation function
- Configurable training parameters
- Real-time training progress monitoring
- Training metrics visualization
"""

import streamlit as st
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import yaml
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import sys

# Import functions from existing train.py script
from train import (
    validate_dataset,
    create_dataset_yaml,
    get_model_path,
    train_model
)

# Set page configuration
st.set_page_config(
    page_title="Piano Keyboard Detection - Training",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None
if 'uploaded_dataset' not in st.session_state:
    st.session_state.uploaded_dataset = None
if 'training_results' not in st.session_state:
    st.session_state.training_results = []


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


def extract_zip_file(uploaded_file, extract_to):
    """Extract uploaded zip file to specified directory"""
    try:
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    except Exception as e:
        st.error(f"Error extracting zip file: {e}")
        return False


def get_dataset_stats(dataset_path):
    """Get statistics about the dataset"""
    stats = {
        'train_images': 0,
        'val_images': 0,
        'train_labels': 0,
        'val_labels': 0,
        'classes': []
    }
    
    # Count images
    train_img_dir = os.path.join(dataset_path, 'images', 'train')
    val_img_dir = os.path.join(dataset_path, 'images', 'val')
    
    if os.path.exists(train_img_dir):
        stats['train_images'] = len([f for f in os.listdir(train_img_dir) 
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    
    if os.path.exists(val_img_dir):
        stats['val_images'] = len([f for f in os.listdir(val_img_dir) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    
    # Count labels
    train_label_dir = os.path.join(dataset_path, 'labels', 'train')
    val_label_dir = os.path.join(dataset_path, 'labels', 'val')
    
    if os.path.exists(train_label_dir):
        stats['train_labels'] = len([f for f in os.listdir(train_label_dir) 
                                   if f.endswith('.txt')])
    
    if os.path.exists(val_label_dir):
        stats['val_labels'] = len([f for f in os.listdir(val_label_dir) 
                                 if f.endswith('.txt')])
    
    # Read classes
    classes_file = os.path.join(dataset_path, 'classes.txt')
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            stats['classes'] = [line.strip() for line in f.readlines()]
    
    return stats


def display_training_results():
    """Display training results and metrics"""
    if not st.session_state.training_results:
        return
    
    # Results table
    results_data = []
    for i, result in enumerate(st.session_state.training_results):
        results_data.append({
            'ID': i + 1,
            'Timestamp': result['timestamp'][:19],
            'Model': result['model_variant'],
            'Epochs': result['epochs'],
            'Batch Size': result['batch_size'],
            'Image Size': result['img_size'],
            'Status': '✅ Completed' if os.path.exists(result['best_model_path']) else '❌ Failed'
        })
    
    df = pd.DataFrame(results_data)
    st.dataframe(df, use_container_width=True)
    
    # Model selection for detailed view
    if len(st.session_state.training_results) > 0:
        selected_idx = st.selectbox(
            "Select training run for details:",
            range(len(st.session_state.training_results)),
            format_func=lambda x: f"Run {x+1} - {st.session_state.training_results[x]['model_variant']} ({st.session_state.training_results[x]['timestamp'][:19]})"
        )
        
        selected_result = st.session_state.training_results[selected_idx]
        
        # Display training plots if available
        results_dir = selected_result['output_path']
        if os.path.exists(results_dir):
            plot_files = ['results.png', 'confusion_matrix.png', 'F1_curve.png', 'PR_curve.png']
            
            cols = st.columns(2)
            for i, plot_file in enumerate(plot_files):
                plot_path = os.path.join(results_dir, plot_file)
                if os.path.exists(plot_path):
                    with cols[i % 2]:
                        st.image(plot_path, caption=plot_file.replace('.png', '').replace('_', ' ').title())


def dataset_configuration_section():
    """Dataset configuration interface"""
    st.subheader("📂 Dataset Configuration")
    
    # Dataset source selection
    dataset_source = st.radio(
        "Choose dataset source:",
        ["Upload ZIP file", "Use existing local folder"],
        horizontal=True
    )
    
    dataset_path = None
    
    if dataset_source == "Upload ZIP file":
        uploaded_zip = st.file_uploader(
            "Upload YOLO dataset (ZIP file)",
            type=['zip'],
            help="Upload a ZIP file containing your YOLO dataset with the required structure"
        )
        
        if uploaded_zip:
            temp_dir = create_temp_directory()
            dataset_extract_path = os.path.join(temp_dir, 'dataset')
            os.makedirs(dataset_extract_path, exist_ok=True)
            
            if extract_zip_file(uploaded_zip, dataset_extract_path):
                # Look for dataset in extracted files
                extracted_items = os.listdir(dataset_extract_path)
                if len(extracted_items) == 1 and os.path.isdir(os.path.join(dataset_extract_path, extracted_items[0])):
                    dataset_path = os.path.join(dataset_extract_path, extracted_items[0])
                else:
                    dataset_path = dataset_extract_path
                
                st.success(f"✅ Dataset extracted to: {dataset_path}")
                st.session_state.uploaded_dataset = dataset_path
        
    else:
        # Local folder selection
        local_dataset_path = st.text_input(
            "Dataset folder path:",
            value="./yolo_dataset",
            help="Path to your local YOLO dataset folder"
        )
        
        if os.path.exists(local_dataset_path):
            dataset_path = local_dataset_path
            st.success(f"✅ Using local dataset: {dataset_path}")
        elif local_dataset_path:
            st.error(f"❌ Path does not exist: {local_dataset_path}")
    
    return dataset_path


def training_parameters_section():
    """Training parameters configuration"""
    st.subheader("⚙️ Training Parameters")
    
    # Model selection
    model_variant = st.selectbox(
        "Model variant:",
        ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
        index=2,
        help="Larger models are more accurate but slower"
    )
    
    # Training parameters
    epochs = st.number_input("Epochs:", min_value=1, max_value=1000, value=50)
    batch_size = st.number_input("Batch size:", min_value=1, max_value=64, value=8)
    img_size = st.selectbox("Image size:", [320, 640, 1280], index=1)
    
    # Output directory
    output_dir = st.text_input(
        "Output directory:",
        value="./training_output",
        help="Directory to save training results"
    )
    
    return model_variant, epochs, batch_size, img_size, output_dir


def start_training_with_original_functions(dataset_path, output_dir, epochs, batch_size, img_size, model_variant):
    """Start training using the original train.py functions"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔄 Preparing training...")
        progress_bar.progress(10)
        
        # Create dataset YAML using original function
        dataset_yaml = create_dataset_yaml(dataset_path)
        progress_bar.progress(20)
        
        status_text.text("🚀 Starting model training...")
        
        # Get script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Train model using original function
        progress_bar.progress(30)
        
        # Create custom project name
        project_name = f"piano_keyboard_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Call the original training function
        model_path = train_model(
            dataset_yaml=dataset_yaml,
            output_dir=output_dir,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            model_name=model_variant,
            script_dir=script_dir
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Training completed!")
        
        # Store results
        result_info = {
            'timestamp': datetime.now().isoformat(),
            'model_variant': model_variant,
            'epochs': epochs,
            'batch_size': batch_size,
            'img_size': img_size,
            'dataset_path': dataset_path,
            'output_path': os.path.dirname(model_path),
            'best_model_path': model_path,
            'project_name': project_name
        }
        
        st.session_state.training_results.append(result_info)
        st.success(f"🎉 Training completed! Best model saved to: {model_path}")
        
        # Auto-refresh to show results
        st.rerun()
        
    except Exception as e:
        progress_bar.progress(0)
        status_text.text("❌ Training failed!")
        st.error(f"❌ Training failed: {str(e)}")


def main():
    """Main application"""
    st.title("🎹 Piano Keyboard Detection - Training")
    st.markdown("Web interface for training YOLOv8 piano keyboard detection models")
    st.markdown("*Uses functions from the original train.py script for consistency*")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ Tools")
        
        if st.button("🧹 Clean Temp Files"):
            cleanup_temp_directory()
        
        st.markdown("---")
        
        st.header("ℹ️ About")
        st.markdown("""
        This application uses the existing train.py functions for consistency:
        
        **Core Functions:**
        - `validate_dataset()` - Dataset validation
        - `create_dataset_yaml()` - YAML creation
        - `get_model_path()` - Model path resolution
        - `train_model()` - Model training
        
        **Features:**
        - Dataset upload and validation
        - Parameter configuration
        - Training progress monitoring
        - Results visualization
        """)
    
    # Main content
    with st.container():
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Dataset configuration
            dataset_path = dataset_configuration_section()
            
            # Validate dataset if path is available
            if dataset_path:
                st.markdown("---")
                st.subheader("🔍 Dataset Validation")
                
                # Use original validation function
                is_valid = validate_dataset(dataset_path)
                
                if is_valid:
                    st.success("✅ Dataset structure is valid!")
                    
                    # Show dataset statistics
                    stats = get_dataset_stats(dataset_path)
                    
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("Train Images", stats['train_images'])
                    with col_b:
                        st.metric("Val Images", stats['val_images'])
                    with col_c:
                        st.metric("Train Labels", stats['train_labels'])
                    with col_d:
                        st.metric("Val Labels", stats['val_labels'])
                    
                    if stats['classes']:
                        st.info(f"📝 Classes: {', '.join(stats['classes'])}")
                else:
                    st.error("❌ Dataset validation failed! Check the console output for details.")
                    
        with col2:
            # Training parameters
            model_variant, epochs, batch_size, img_size, output_dir = training_parameters_section()
            
            # Start training button
            st.markdown("---")
            if st.button("🚀 Start Training", type="primary", disabled=not (dataset_path and is_valid), use_container_width=True):
                if dataset_path and is_valid:
                    start_training_with_original_functions(dataset_path, output_dir, epochs, batch_size, img_size, model_variant)
    
    # Training progress and results section
    if st.session_state.training_results:
        st.markdown("---")
        st.subheader("📊 Training Results")
        display_training_results()


if __name__ == "__main__":
    main()