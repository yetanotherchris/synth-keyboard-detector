#!/usr/bin/env python3
"""Debug script to inspect the detection process."""

import cv2
import numpy as np
import os
from piano_key_detector import PianoKeyDetector

def debug_detection():
    """Debug the detection process step by step."""
    print("Debugging piano key detection...")
    
    # Load the example image
    example_path = "example-1.jpg"
    if not os.path.exists(example_path):
        print(f"Example image {example_path} not found!")
        return
    
    # Load and inspect the original image
    image = cv2.imread(example_path)
    if image is None:
        print("Failed to load image!")
        return
    
    height, width = image.shape[:2]
    print(f"Original image size: {width}x{height}")
    
    # Create detector with more verbose output
    detector = PianoKeyDetector()
    
    # Run detection and get debug info
    result = detector.detect_keys(example_path, debug=True)
    
    # Inspect debug images
    debug_images = {
        "stage1": "debug_stage1_bottom_50_percent.jpg",
        "stage3": "debug_stage3_keyboard_height.jpg", 
        "stage4": "debug_stage4_white_keys_only.jpg"
    }
    
    for stage, filename in debug_images.items():
        if os.path.exists(filename):
            debug_img = cv2.imread(filename)
            if debug_img is not None:
                h, w = debug_img.shape[:2]
                print(f"{stage} image size: {w}x{h}")
                
                # Check if image is mostly black (no content)
                gray = cv2.cvtColor(debug_img, cv2.COLOR_BGR2GRAY)
                mean_brightness = np.mean(gray)
                print(f"{stage} mean brightness: {mean_brightness:.1f}")
                
                if mean_brightness < 10:
                    print(f"WARNING: {stage} image appears to be mostly black!")
    
    # Test with more relaxed parameters
    print("\nTrying with more relaxed detection parameters...")
    relaxed_detector = PianoKeyDetector(
        projection_peak_ratio=0.3,  # Lower threshold
        min_white_key_width_ratio=0.005,  # Smaller minimum width
        max_white_key_width_ratio=0.3,  # Larger maximum width
        min_white_key_aspect=1.0,  # Lower aspect ratio requirement
        min_white_key_area_ratio=0.001  # Smaller minimum area
    )
    
    relaxed_result = relaxed_detector.detect_keys(example_path, debug=False)
    print(f"Relaxed detection found {relaxed_result['total_keys']} keys")
    print(f"Relaxed detection confidence: {relaxed_result['confidence']:.2f}")

if __name__ == "__main__":
    debug_detection()