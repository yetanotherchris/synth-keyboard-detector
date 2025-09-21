#!/usr/bin/env python3
"""
YOLOv8 Training Script for Piano Keyboard Detection

This script trains a YOLOv8 model for piano keyboard detection using
data exported from Label Studio in YOLO format.

Usage:
    python train.py [OPTIONS]

Options:
    --dataset PATH      Path to YOLO dataset directory (default: ./yolo_dataset)
    --output PATH       Output directory for trained model (default: ./training_output)
    --epochs INT        Number of training epochs (default: 50)
    --batch INT         Batch size (default: 8)
    --imgsz INT         Image size for training (default: 640)
    --model STR         YOLOv8 model variant (default: yolov8m.pt)
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO


def validate_dataset(dataset_path: str) -> bool:
    """Validate YOLO dataset structure and contents"""
    required_dirs = [
        'images/train',
        'images/val',
        'labels/train',
        'labels/val'
    ]
    
    classes_file = os.path.join(dataset_path, 'classes.txt')
    
    # Check directory structure
    for dir_path in required_dirs:
        full_path = os.path.join(dataset_path, dir_path)
        if not os.path.exists(full_path):
            print(f"❌ Missing directory: {full_path}")
            return False
    
    # Check classes file
    if not os.path.exists(classes_file):
        print(f"❌ Missing classes.txt file: {classes_file}")
        return False
    
    # Count files
    train_images_path = os.path.join(dataset_path, 'images', 'train')
    val_images_path = os.path.join(dataset_path, 'images', 'val')
    train_labels_path = os.path.join(dataset_path, 'labels', 'train')
    val_labels_path = os.path.join(dataset_path, 'labels', 'val')
    
    train_images = len([f for f in os.listdir(train_images_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    val_images = len([f for f in os.listdir(val_images_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    train_labels = len([f for f in os.listdir(train_labels_path) if f.endswith('.txt')])
    val_labels = len([f for f in os.listdir(val_labels_path) if f.endswith('.txt')])
    
    print(f"📊 Dataset Summary:")
    print(f"   Training: {train_images} images, {train_labels} labels")
    print(f"   Validation: {val_images} images, {val_labels} labels")
    print(f"   Total: {train_images + val_images} images")
    
    if train_images == 0:
        print("❌ No training images found!")
        return False
    
    if train_labels != train_images:
        print(f"⚠️  Warning: Mismatch between training images ({train_images}) and labels ({train_labels})")
    
    if val_labels != val_images:
        print(f"⚠️  Warning: Mismatch between validation images ({val_images}) and labels ({val_labels})")
    
    return True


def create_dataset_yaml(dataset_path: str) -> str:
    """Create YOLO dataset configuration file"""
    classes_file = os.path.join(dataset_path, 'classes.txt')
    
    # Read class names
    with open(classes_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    
    # Create dataset configuration
    dataset_config = {
        'path': os.path.abspath(dataset_path),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    # Save YAML file
    yaml_path = os.path.join(dataset_path, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"✅ Created dataset config: {yaml_path}")
    print(f"📝 Classes: {class_names}")
    
    return yaml_path


def get_model_path(model_name: str, script_dir: str) -> str:
    """
    Get the full path to a model file, checking local models directory first.
    
    Args:
        model_name: Name of the model (e.g., 'yolov8m.pt')
        script_dir: Directory where the script is located
        
    Returns:
        Path to model file (local if exists, otherwise returns model_name for download)
    """
    # Check if it's already a full path
    if os.path.isabs(model_name) and os.path.exists(model_name):
        return model_name
    
    # Check local models directory
    models_dir = os.path.join(script_dir, 'models')
    local_model_path = os.path.join(models_dir, model_name)
    
    if os.path.exists(local_model_path):
        print(f"📁 Using local model: {local_model_path}")
        return local_model_path
    else:
        print(f"🌐 Local model not found, will download: {model_name}")
        return model_name


def train_model(dataset_yaml: str, output_dir: str, epochs: int, batch: int, 
                imgsz: int, model_name: str, script_dir: str) -> str:
    """Train YOLOv8 model"""
    
    # Get the appropriate model path (local or download)
    model_path = get_model_path(model_name, script_dir)
    
    # Initialize model
    print(f"🚀 Loading {model_name}...")
    model = YOLO(model_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🎯 Starting training...")
    print(f"   Dataset: {dataset_yaml}")
    print(f"   Output: {output_dir}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch}")
    print(f"   Image size: {imgsz}")
    
    try:
        # Train the model
        results = model.train(
            data=dataset_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=0,  # Use GPU if available
            project=output_dir,
            name='default',
            save=True,
            save_period=10,
            val=True,
            plots=True,
            cache='disk',
            amp=False,  # Disable AMP to avoid potential CUDA errors
            workers=2,
            verbose=True,
            patience=15,
            exist_ok=True
        )
        
        # Get model path
        model_path = os.path.join(output_dir, 'default', 'weights', 'best.pt')
        
        print(f"✅ Training completed successfully!")
        print(f"🎯 Best model saved to: {model_path}")
        
        return model_path
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(
        description='Train YOLOv8 model for piano keyboard detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train.py
    python train.py --epochs 100 --batch 16
    python train.py --dataset ./my_dataset --output ./my_training_output
    python train.py --model yolov8s.pt --epochs 200
        """
    )
    
    parser.add_argument('--dataset', type=str, default='./yolo_dataset',
                       help='Path to YOLO dataset directory')
    parser.add_argument('--output', type=str, default='./training_output',
                       help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for training')
    parser.add_argument('--model', type=str, default='yolov8m.pt',
                       choices=['yolo11n.pt', 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                       help='YOLOv8 model variant to use (checks models/ directory first, downloads if not found)')
    
    args = parser.parse_args()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("🎹 YOLOv8 Piano Keyboard Detection Training")
    print("=" * 50)
    
    # Validate dataset
    print("🔍 Validating dataset...")
    if not validate_dataset(args.dataset):
        print("❌ Dataset validation failed. Please check your dataset structure:")
        print("   yolo_dataset/")
        print("   ├── images/")
        print("   │   ├── train/")
        print("   │   └── val/")
        print("   ├── labels/")
        print("   │   ├── train/")
        print("   │   └── val/")
        print("   └── classes.txt")
        sys.exit(1)
    
    print("✅ Dataset validation passed!")
    
    try:
        # Create dataset configuration
        print("\n📝 Creating dataset configuration...")
        dataset_yaml = create_dataset_yaml(args.dataset)
        
        # Train model
        print("\n🚀 Starting model training...")
        model_path = train_model(
            dataset_yaml=dataset_yaml,
            output_dir=args.output,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            model_name=args.model,
            script_dir=script_dir
        )
        
        print("\n🎉 Training completed successfully!")
        print(f"📁 Results saved in: {args.output}/piano_keyboard_detection/")
        print(f"🏆 Best model: {model_path}")
        print("\nNext steps:")
        print("1. Test your model: python test.py --model path/to/best.pt --image path/to/test/image.jpg")
        print("2. Run batch inference: python test.py --model path/to/best.pt --images path/to/test/folder/")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()