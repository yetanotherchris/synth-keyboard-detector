# Piano Key Detector

`piano-key-detector` is a Python package that detects white and black keys on a top-down photograph of a piano or synthesizer keyboard. It provides a simple API for offline processing and ships with a Streamlit playground for experimenting with detection parameters on custom images.

## Installation

```bash
Installation

- Option A (editable/development): `pip install -e .`
- Option B (requirements file): `pip install -r requirements.txt`

Usage

- After installation, import the package in Python:
  - `from piano_key_detector import detector`
  - See `tests/test_detector.py` for usage examples.

Notes

- Requires Python >= 3.8.
- Core dependencies: opencv-python, numpy, pillow, streamlit (see `requirements.txt`).
```

> **Note:** When running in a headless environment (such as CI), you may prefer
> installing `opencv-python-headless` instead of `opencv-python` to avoid
> dependency issues with OpenGL libraries.

## Usage

```python
from piano_key_detector import PianoKeyDetector

# Create the detector with default parameters
_detector = PianoKeyDetector()

# Detect keys from an image path
result = _detector.detect_keys("path/to/keyboard.jpg")

print(result["keyboard_type"])  # e.g. "49_key"
print(len(result["white_keys"]))
print(len(result["black_keys"]))
```

The detector returns a dictionary containing the keyboard type, bounding box, key locations, and detection confidence. Each key entry includes pixel and normalized positions which makes it easy to map keys back onto the original image.

## Streamlit playground

A basic Streamlit interface is provided in `test_app.py`. Launch it with:

```bash
streamlit run test_app.py
```

The app lets you upload an image, tweak detection parameters (Canny thresholds, blur size, etc.), and inspect the intermediate processing steps such as edge detection and binary masks.

## Running tests

```bash
pytest
```
