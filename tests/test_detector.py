import cv2
import numpy as np

from piano_key_detector import PianoKeyDetector
from piano_key_detector.keyboard_models import KEYBOARD_MODELS


def _create_synthetic_keyboard_image(
    *,
    white_key_count: int = 15,
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

    model = KEYBOARD_MODELS["25"]
    black_pattern = model.black_key_pattern
    black_width = int(white_key_width * 0.6)
    black_height = int(height * 0.6)
    for idx, has_black in enumerate(black_pattern[: white_key_count - 1]):
        if not has_black:
            continue
        center = int((idx + 1) * white_key_width)
        x0 = max(0, center - black_width // 2)
        x1 = min(width - 1, center + black_width // 2)
        cv2.rectangle(image, (x0, 0), (x1, black_height), (35, 35, 35), -1)

    return image


def test_detects_keys_on_synthetic_keyboard(tmp_path):
    image = _create_synthetic_keyboard_image()
    path = tmp_path / "keyboard.jpg"
    cv2.imwrite(str(path), image)

    detector = PianoKeyDetector()
    result = detector.detect_keys(str(path), force_keyboard_type="25", debug=False)

    assert result["keyboard_type"] == "25_key"
    assert result["total_keys"] == 25
    # Allow a small tolerance for detection noise
    assert 13 <= len(result["white_keys"]) <= 16
    assert len(result["black_keys"]) >= 8
    assert result["confidence"] > 0

    centers = [key["normalized_center"][0] for key in result["white_keys"]]
    assert centers == sorted(centers)

    bbox = result["bounding_box"]
    assert bbox[2] > 0 and bbox[3] > 0
