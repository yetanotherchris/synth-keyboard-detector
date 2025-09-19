#!/usr/bin/env python3
"""Direct test of the green overlay functionality."""

import cv2
import numpy as np
import os
from piano_key_detector import PianoKeyDetector

def test_green_overlay():
    """Test the green overlay functionality directly."""
    print("Testing green overlay functionality...")
    
    # Load stage4 debug image
    stage4_path = "debug_stage4_white_keys_only.jpg"
    if not os.path.exists(stage4_path):
        print(f"Stage4 debug image {stage4_path} not found!")
        return
    
    stage4_img = cv2.imread(stage4_path)
    if stage4_img is None:
        print("Failed to load stage4 image!")
        return
    
    height, width = stage4_img.shape[:2]
    print(f"Stage4 image size: {width}x{height}")
    
    # Create a fake key detection for testing
    fake_key = {
        "bbox": (0, 0, width // 4, height),  # First quarter of the image
        "confidence": 1.0,
        "aspect_ratio": height / (width // 4),
        "area_ratio": 0.25
    }
    
    print(f"Testing with fake key: {fake_key['bbox']}")
    
    # Create detector and test the overlay
    detector = PianoKeyDetector()
    
    # Test the new solid green overlay
    green_overlay = detector._create_green_solid_overlay(stage4_img, fake_key)
    
    # Save the test overlay
    test_output_path = "test_green_solid_overlay.jpg"
    cv2.imwrite(test_output_path, green_overlay)
    print(f"Test green overlay saved to: {test_output_path}")
    
    # Verify the overlay was applied
    if os.path.exists(test_output_path):
        print("✓ Green solid overlay test completed successfully!")
        
        # Check if there's actually green in the overlay
        overlay_bgr = cv2.imread(test_output_path)
        green_channel = overlay_bgr[:, :, 1]  # Green channel
        max_green = np.max(green_channel)
        print(f"Max green value in overlay: {max_green}")
        
        if max_green > 200:
            print("✓ Green overlay appears to be working correctly!")
        else:
            print("⚠ Green overlay may not be bright enough")
    else:
        print("✗ Failed to create test overlay")

if __name__ == "__main__":
    test_green_overlay()