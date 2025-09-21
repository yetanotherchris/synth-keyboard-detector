# YOLOv8 Piano Keyboard Detection

This directory contains a streamlined training and testing pipeline for piano keyboard detection using YOLOv8.
https://github.com/ultralytics/ultralytics

## Files Overview

- `train.py` - Main training script with comprehensive validation and configuration options
- `test.py` - Unified testing script supporting both single image and batch processing
- `models/` - Directory containing pre-trained YOLOv8 models (yolov8l.pt, yolov8m.pt, yolo11n.pt)
- `training_output/` - Training results and trained models
- `yolo_dataset/` - YOLO format dataset directory

## Quick Start

You can use the small model and 50 epochs, the dataset is small and the piano shape is very simple.

```
streamlit run .\streamlit_train.py
streamlit run .\streamlit_test.py

# tiny model
python train.py --dataset ./yolo_dataset --model yolo11n.pt --output ./training_output --epochs 50

# Recommended starting point
python train.py --model yolov8l.pt --epochs 150 --batch 12

# If you have time/compute budget
python train.py --model yolov8x.pt --epochs 200 --batch 8

# 6 hours+
python train.py --dataset ./yolo_dataset --model yolov8l.pt --output ./training_output --epochs 200

# test the trained model
python test.py --model "training_output\piano_keyboard_detection\weights\best.pt" --image ./test-piano.jpg
```

### 1. Prepare Your Dataset

#### Option A: Export from Label Studio
In Label Studio:
1. Go to your project
2. Click **Export**
3. Choose **"YOLO with Images"** format
4. Download and extract to `labeller/yolo_dataset/`

#### Option B: Use Existing YOLO Dataset
Ensure your dataset follows the YOLO format structure:

```
labeller/yolo_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── classes.txt
```

### 2. Install Dependencies

Navigate to the project root and install dependencies:

```powershell
cd ..
pip install -r requirements.txt
cd labeller
```

### 3. Train Your Model

From the `labeller/` directory:

```powershell
# Basic training with default settings
python train.py

# Advanced training with custom options
python train.py --epochs 100 --batch 16 --model yolov8l.pt
```

### 4. Test Your Trained Model

#### Single Image Testing
```powershell
python test.py --model training_output/piano_keyboard_detection/weights/best.pt --image path/to/test/image.jpg
```

#### Batch Testing with Analytics
```powershell
python test.py --model training_output/piano_keyboard_detection/weights/best.pt --images path/to/test/images/
```

#### Display Results Interactively
```powershell
python test.py --model training_output/piano_keyboard_detection/weights/best.pt --image test.jpg --show
```

## Training Configuration Options

The training script supports multiple configuration options:

### Model Variants
- **yolov8n.pt**: Nano (fastest, least accurate)
- **yolov8s.pt**: Small (fast, good for mobile)
- **yolov8m.pt**: Medium (balanced speed/accuracy) [Default]
- **yolov8l.pt**: Large (slower, more accurate)
- **yolov8x.pt**: Extra Large (slowest, most accurate)

### Training Parameters
- **--epochs**: Number of training epochs (default: 50)
- **--batch**: Batch size (default: 8, adjust based on GPU memory)
- **--imgsz**: Image size for training (default: 640)
- **--dataset**: Path to YOLO dataset directory
- **--output**: Output directory for trained model

### Example Commands
```powershell
# Quick training for testing
python train.py --epochs 25 --batch 4

# Production training with large model
python train.py --model yolov8l.pt --epochs 200 --batch 16

# Custom dataset location
python train.py --dataset ./my_dataset --output ./my_models
```

## Model Management

### Pre-trained Models

The training script automatically handles YOLOv8 model loading:

1. **Local Models**: First checks the `models/` directory for pre-trained models
2. **Auto-download**: If not found locally, automatically downloads from Ultralytics servers
3. **Model Cache**: Downloaded models are cached for future use

### Model Download Process

When you specify a model (e.g., `--model yolov8m.pt`):

- **Step 1**: Script checks `labeller/models/yolov8m.pt`
- **Step 2**: If not found, YOLOv8 downloads from GitHub (https://github.com/ultralytics/assets)
- **Step 3**: Model is cached locally and used for training

### Model Sizes
- **yolov8n.pt**: ~6MB (Nano)
- **yolov8s.pt**: ~22MB (Small)  
- **yolov8m.pt**: ~52MB (Medium) [Default]
- **yolov8l.pt**: ~87MB (Large)
- **yolov8x.pt**: ~136MB (Extra Large)

*Note: All .pt model files are ignored by git to prevent committing large binaries*

## Expected Training Time

Training time varies based on hardware and configuration:
- **GPU Training**: 20-60 minutes (50 epochs, ~100 images)
- **CPU Training**: 2-6 hours (not recommended)

## Testing Options

The unified test script provides comprehensive testing capabilities:

### Single Image Testing
```powershell
# Basic single image test
python test.py --model training_output/piano_keyboard_detection/weights/best.pt --image test.jpg

# Show results interactively
python test.py --model best.pt --image test.jpg --show

# Custom confidence threshold
python test.py --model best.pt --image test.jpg --conf 0.5 --output results/
```

### Batch Processing
```powershell
# Process all images in a directory
python test.py --model best.pt --images test_images/

# Batch processing with custom settings
python test.py --model best.pt --images test_images/ --conf 0.3 --output batch_results/
```

### Test Script Features
- **Confidence threshold control**: Adjust detection sensitivity
- **IoU threshold**: Control overlap detection
- **Detailed analytics**: Batch processing generates JSON reports with statistics
- **Visual output**: Annotated images with bounding boxes and confidence scores
- **Interactive display**: View results immediately (single image mode)

## Performance Optimization

### For Better Accuracy:
- Increase epochs: `python train.py --epochs 200`
- Use larger model: `python train.py --model yolov8l.pt`
- Adjust confidence threshold during inference: `--conf 0.3`

### For Faster Training:
- Reduce batch size if memory issues: `--batch 4`
- Use smaller model: `python train.py --model yolov8s.pt`
- Reduce image size: `--imgsz 512`

### For Faster Inference:
- Use smaller model for testing
- Increase confidence threshold: `--conf 0.5`
- Reduce image resolution

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in training:
```powershell
python train.py --batch 4
```

### Low Detection Accuracy
1. Check if annotations are correct in your dataset
2. Increase training epochs: `--epochs 200`
3. Use a larger model: `--model yolov8l.pt`
4. Adjust confidence threshold during testing: `--conf 0.3`

### Model Not Found Error
Make sure training completed successfully and check:
```
training_output/piano_keyboard_detection/weights/best.pt
```

### No Images Found Error
Verify your dataset structure matches the expected YOLO format with proper file extensions (.jpg, .png, etc.)

## Quick Start Summary

1. **Prepare dataset**: Ensure YOLO format structure in `yolo_dataset/`
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Train model**: `python train.py`
4. **Test model**: `python test.py --model training_output/piano_keyboard_detection/weights/best.pt --image test.jpg`

The training will automatically detect and use your GPU if available, with fallback to CPU if necessary.