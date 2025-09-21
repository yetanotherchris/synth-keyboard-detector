#!/usr/bin/env python3
"""
YOLOv8 Testing Script for Piano Keyboard Detection

This script runs inference using a trained YOLOv8 model for piano keyboard detection.
Supports both single image testing and batch processing with detailed analytics.

Usage:
    # Single image testing
    python test.py --model path/to/model.pt --image path/to/image.jpg
    
    # Batch processing
    python test.py --model path/to/model.pt --images path/to/folder/
    
    # With custom output directory
    python test.py --model path/to/model.pt --image test.jpg --output results/

Options:
    --model PATH        Path to trained model (.pt file) [REQUIRED]
    --image PATH        Path to single test image
    --images PATH       Path to directory containing test images
    --output PATH       Output directory for results (default: ./test_results)
    --conf FLOAT        Confidence threshold (default: 0.25)
    --iou FLOAT         IoU threshold for NMS (default: 0.45)
    --save-images       Save annotated images (default: True)
    --show              Display images (single image mode only)
"""

import os
import sys
import json
import cv2
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from ultralytics import YOLO


def load_model(model_path: str) -> YOLO:
    """Load the trained YOLOv8 model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"🔧 Loading model: {model_path}")
    model = YOLO(model_path)
    return model


def process_single_image(model: YOLO, image_path: str, conf_threshold: float, 
                        iou_threshold: float, output_dir: str = None, 
                        show_image: bool = False) -> Dict[str, Any]:
    """Process a single image and return detection results"""
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    print(f"🔍 Processing: {os.path.basename(image_path)}")
    
    # Run inference
    results = model(image_path, conf=conf_threshold, iou=iou_threshold, verbose=False)
    
    # Process results
    result_data = {
        'image_path': image_path,
        'image_name': os.path.basename(image_path),
        'detections': [],
        'detection_count': 0,
        'max_confidence': 0.0,
        'avg_confidence': 0.0
    }
    
    # Load original image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    confidences = []
    
    # Process detections
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                # Extract box data
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                
                # Store detection data
                detection = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': conf,
                    'class': cls,
                    'class_name': 'piano-keyboard'
                }
                result_data['detections'].append(detection)
                confidences.append(conf)
                
                # Draw bounding box
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Add label
                label = f"piano-keyboard: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(img, (int(x1), int(y1-label_size[1]-10)), 
                             (int(x1+label_size[0]), int(y1)), (0, 255, 0), -1)
                cv2.putText(img, label, (int(x1), int(y1-5)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Calculate statistics
    if confidences:
        result_data['detection_count'] = len(confidences)
        result_data['max_confidence'] = max(confidences)
        result_data['avg_confidence'] = sum(confidences) / len(confidences)
        print(f"   ✅ Found {len(confidences)} detection(s), max confidence: {max(confidences):.3f}")
    else:
        print(f"   ❌ No detections found")
    
    # Save or show image
    if output_dir and not show_image:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"result_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, img)
        print(f"   💾 Saved to: {output_path}")
        result_data['output_path'] = output_path
    
    if show_image:
        cv2.imshow(f'Detection: {os.path.basename(image_path)}', img)
        print("   👁️  Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result_data


def process_batch_images(model: YOLO, images_dir: str, conf_threshold: float,
                        iou_threshold: float, output_dir: str) -> List[Dict[str, Any]]:
    """Process multiple images and generate batch results"""
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(images_dir).glob(f'*{ext}'))
        image_files.extend(Path(images_dir).glob(f'*{ext.upper()}'))
    
    if not image_files:
        raise ValueError(f"No images found in directory: {images_dir}")
    
    print(f"📁 Found {len(image_files)} images to process")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    all_results = []
    total_detections = 0
    images_with_detections = 0
    all_confidences = []
    
    for i, img_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] ", end="")
        
        try:
            result = process_single_image(
                model=model,
                image_path=str(img_path),
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                output_dir=output_dir,
                show_image=False
            )
            all_results.append(result)
            
            if result['detection_count'] > 0:
                images_with_detections += 1
                total_detections += result['detection_count']
                all_confidences.extend([d['confidence'] for d in result['detections']])
                
        except Exception as e:
            print(f"   ❌ Error processing {img_path}: {str(e)}")
            continue
    
    # Generate summary statistics
    summary = {
        'total_images': len(image_files),
        'images_processed': len(all_results),
        'images_with_detections': images_with_detections,
        'total_detections': total_detections,
        'detection_rate': images_with_detections / len(all_results) if all_results else 0,
        'avg_detections_per_image': total_detections / len(all_results) if all_results else 0,
        'confidence_stats': {}
    }
    
    if all_confidences:
        summary['confidence_stats'] = {
            'min': min(all_confidences),
            'max': max(all_confidences),
            'mean': sum(all_confidences) / len(all_confidences),
            'median': sorted(all_confidences)[len(all_confidences)//2]
        }
    
    # Save detailed results
    results_file = os.path.join(output_dir, 'batch_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'summary': summary,
            'results': all_results,
            'processed_at': datetime.now().isoformat()
        }, f, indent=2)
    
    # Print summary
    print("\n📊 Batch Processing Summary:")
    print(f"   Total images: {summary['total_images']}")
    print(f"   Successfully processed: {summary['images_processed']}")
    print(f"   Images with detections: {summary['images_with_detections']} ({summary['detection_rate']*100:.1f}%)")
    print(f"   Total detections: {summary['total_detections']}")
    print(f"   Average detections per image: {summary['avg_detections_per_image']:.2f}")
    
    if all_confidences:
        stats = summary['confidence_stats']
        print(f"   Confidence scores - Min: {stats['min']:.3f}, Max: {stats['max']:.3f}, Mean: {stats['mean']:.3f}")
    
    print(f"   Results saved to: {results_file}")
    
    return all_results


def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(
        description='Test YOLOv8 model for piano keyboard detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test single image
    python test.py --model training_output/default/weights/best.pt --image test.jpg
    
    # Test single image with display
    python test.py --model training_output/default/weights/best.pt --image test.jpg --show
    
    # Batch process images
    python test.py --model training_output/default/weights/best.pt --images test_images/
    
    # Custom confidence threshold
    python test.py --model best.pt --image test.jpg --conf 0.5 --output results/
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pt file)')
    parser.add_argument('--image', type=str,
                       help='Path to single test image')
    parser.add_argument('--images', type=str,
                       help='Path to directory containing test images')
    parser.add_argument('--output', type=str, default='./test_results',
                       help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS (0.0-1.0)')
    parser.add_argument('--show', action='store_true',
                       help='Display images (single image mode only)')
    parser.add_argument('--no-save', action='store_true',
                       help='Don\'t save annotated images')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image and not args.images:
        parser.error("Either --image or --images must be specified")
    
    if args.image and args.images:
        parser.error("Cannot specify both --image and --images")
    
    if args.show and args.images:
        parser.error("--show can only be used with --image (single image mode)")
    
    print("🎹 YOLOv8 Piano Keyboard Detection Testing")
    print("=" * 50)
    
    try:
        # Load model
        model = load_model(args.model)
        
        # Single image mode
        if args.image:
            print(f"🖼️  Single image mode")
            output_dir = None if args.no_save else args.output
            
            result = process_single_image(
                model=model,
                image_path=args.image,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                output_dir=output_dir,
                show_image=args.show
            )
            
            print(f"\n✅ Processing completed!")
            if result['detection_count'] > 0:
                print(f"🎯 Found {result['detection_count']} detection(s)")
                print(f"🏆 Max confidence: {result['max_confidence']:.3f}")
            else:
                print("❌ No detections found")
        
        # Batch processing mode
        elif args.images:
            print(f"📁 Batch processing mode")
            
            results = process_batch_images(
                model=model,
                images_dir=args.images,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                output_dir=args.output
            )
            
            print(f"\n✅ Batch processing completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Testing failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()