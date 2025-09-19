#!/usr/bin/env python3
"""Test script to verify the Stage 3 improvements."""

import cv2
import os
from piano_key_detector import PianoKeyDetector

def test_detector():
    """Test the detector with example image."""
    print("Testing piano key detector with Stage 3 improvements...")
    
    # Create detector
    detector = PianoKeyDetector()
    
    # Test with example image
    example_path = "example-1.jpg"
    if not os.path.exists(example_path):
        print(f"Example image {example_path} not found!")
        return
    
    print(f"Processing {example_path}...")
    
    # Run detection
    result = detector.detect_keys(example_path, debug=True)
    
    print(f"Detection results:")
    print(f"  Found {result['total_keys']} white keys")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  First white key found: {result['found']}")
    
    if result['found']:
        first_key = result['first_white_key']
        bbox = first_key['bbox']
        print(f"  First key bounding box: {bbox}")
        print(f"  First key aspect ratio: {first_key.get('aspect_ratio', 'N/A'):.2f}")
        print(f"  First key area ratio: {first_key.get('area_ratio', 'N/A'):.4f}")
    
    # Check if debug images were created
    debug_files = [
        "debug_stage1_bottom_50_percent.jpg",
        "debug_stage3_keyboard_height.jpg", 
        "debug_stage3_highlighted.jpg",
        "debug_stage4_white_keys_only.jpg",
        "debug_stage6_green_solid.jpg"
    ]
    
    print("\nDebug images:")
    for debug_file in debug_files:
        if os.path.exists(debug_file):
            print(f"  ✓ {debug_file}")
        else:
            print(f"  ✗ {debug_file} (missing)")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_detector()