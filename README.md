# First White Key Detector

A simplified Python package that detects the first (leftmost) white key from top-down keyboard images.

AI/LLMs should always follow the instructions in AGENTS.MD and .github/copilot-instructions.md.

## Quick Start

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit test interface:
```bash
streamlit run test_app.py
```

---

## Package Details

`piano-key-detector` has been simplified to detect only the first (leftmost) white key on a top-down photograph of a piano or synthesizer keyboard. It provides a simple API for offline processing and ships with a Streamlit playground for experimenting with detection parameters on custom images.

### Installation Options

- **Development**: `pip install -e .`
- **Requirements**: `pip install -r requirements.txt`

### Usage

```python
from piano_key_detector import PianoKeyDetector

# Create the detector with default parameters
detector = PianoKeyDetector()

# Detect the first white key from an image path
result = detector.detect_keys("path/to/keyboard.jpg")

if result["found"]:
    first_key = result["first_white_key"]
    print(f"First white key at: {first_key['bbox']}")
    print(f"Center position: {first_key['center']}")
    print(f"Confidence: {first_key['confidence']}")
else:
    print("No white key detected")
```

The detector returns a dictionary containing:
- `found`: Boolean indicating if a white key was detected
- `first_white_key`: Dictionary with key information (if found):
  - `bbox`: Bounding box (x, y, width, height)
  - `center`: Center coordinates (x, y)
  - `confidence`: Detection confidence (0.0 - 1.0)
  - `aspect_ratio`: Height/width ratio
  - `area_ratio`: Key area relative to image size

### Simplified Functionality

This version has been dramatically simplified from the original multi-key detector:

**What it does:**
- Detects the first (leftmost) white key in an image
- Returns precise bounding box and center coordinates
- Provides confidence scoring

**What it no longer does:**
- ❌ Keyboard type identification (25-key, 49-key, etc.)
- ❌ Multiple white key detection
- ❌ Black key detection
- ❌ Full keyboard layout analysis
- ❌ Bottom-half cropping optimization

### Requirements

- Python >= 3.8
- Core dependencies: opencv-python, numpy, pillow, streamlit

> **Note:** When running in a headless environment (such as CI), you may prefer installing `opencv-python-headless` instead of `opencv-python` to avoid dependency issues with OpenGL libraries.

### Testing

Run the test suite:
```bash
pytest
```
