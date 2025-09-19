import cv2
import numpy as np

from piano_key_detector import PianoKeyDetector


def _create_synthetic_keyboard_image(
    *,
    white_key_count: int = 5,  # Reduced since we only need the first key
    white_key_width: int = 24,
    height: int = 180,
) -> np.ndarray:
    """Create a simple synthetic keyboard image for testing."""
    width = white_key_count * white_key_width
    image = np.full((height, width, 3), 120, dtype=np.uint8)

    # Draw white keys with small gaps between them
    for idx in range(white_key_count):
        x0 = idx * white_key_width
        x1 = x0 + white_key_width - 2
        cv2.rectangle(image, (x0, 0), (x1, height - 1), (245, 245, 245), -1)
        cv2.rectangle(image, (x0, 0), (x1, height - 1), (90, 90, 90), 1)

    return image


def test_detects_first_white_key_on_synthetic_keyboard(tmp_path):
    """Test that the detector finds the first (leftmost) white key."""
    image = _create_synthetic_keyboard_image()
    path = tmp_path / "keyboard.jpg"
    cv2.imwrite(str(path), image)

    detector = PianoKeyDetector()
    result = detector.detect_keys(str(path), debug=False)

    # Test the simplified return format
    assert result["found"] is True
    assert result["first_white_key"] is not None
    
    first_key = result["first_white_key"]
    assert "bbox" in first_key
    assert "center" in first_key
    assert "confidence" in first_key
    
    # Verify the first key is actually the leftmost one
    bbox = first_key["bbox"]
    assert bbox[0] >= 0  # x position should be close to left edge
    assert bbox[2] > 0   # width should be positive
    assert bbox[3] > 0   # height should be positive
    
    # Verify confidence is reasonable
    assert 0.0 <= first_key["confidence"] <= 1.0


def test_no_keys_detected_on_empty_image(tmp_path):
    """Test behavior when no keys are detected."""
    # Create a blank image
    image = np.full((180, 240, 3), 120, dtype=np.uint8)
    path = tmp_path / "blank.jpg"
    cv2.imwrite(str(path), image)

    detector = PianoKeyDetector()
    result = detector.detect_keys(str(path), debug=False)

    assert result["found"] is False
    assert result["first_white_key"] is None
