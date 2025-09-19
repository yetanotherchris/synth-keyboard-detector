# Piano Keyboard Recognition Python Package - Copilot Prompt

Create a Python pip package called `piano-key-detector` that recognizes piano keyboard keys from top-down (bird's eye view) JPG images. The package should handle various keyboard sizes and layouts.

## Requirements:

### Core Functionality:
- **Input**: JPG image of piano keyboard from top-down perspective
- **Output**: Dictionary containing detected key information (positions, types, count)
- **Keyboard Types**: Support 25, 37, 49, 61, and 72-key keyboards
- **Key Detection**: Identify both white and black keys with their approximate positions

### Technical Specifications:
- Use OpenCV for computer vision tasks
- Handle keyboards that may be partially cut off at edges (left/right boundaries not always distinct)
- Return standardized key positions regardless of keyboard size
- Include confidence scores for detections

### Package Structure:
```
piano-key-detector/
├── setup.py
├── README.md
├── piano_key_detector/
│   ├── __init__.py
│   ├── detector.py          # Main detection class
│   ├── keyboard_models.py   # Keyboard type definitions
│   └── utils.py            # Helper functions
└── tests/
    ├── __init__.py
    └── test_detector.py
```

### Main API Design:
```python
from piano_key_detector import PianoKeyDetector

detector = PianoKeyDetector()
result = detector.detect_keys("keyboard_image.jpg")

# Expected output format:
{
    "keyboard_type": "49_key",  # or 25, 37, 61, 72
    "total_keys": 49,
    "white_keys": [...],  # List of white key positions/info
    "black_keys": [...],  # List of black key positions/info
    "confidence": 0.85,
    "bounding_box": (x, y, width, height)
}
```

### Computer Vision Approach:
1. **Preprocessing**: Convert to grayscale, apply Gaussian blur
2. **Edge Detection**: Use Canny edge detection to find key boundaries
3. **Line Detection**: HoughLines to detect regular patterns of white keys
4. **Contour Analysis**: Find rectangular regions that match key proportions
5. **Pattern Matching**: Use known keyboard layouts to validate detections
6. **Black Key Detection**: Identify black keys based on position between white keys

### Keyboard Layout Knowledge:
- 25-key: 2 octaves (C to C)
- 37-key: 3+ octaves 
- 49-key: 4+ octaves (C to C)
- 61-key: 5+ octaves (C to C)
- 72-key: 6 octaves

Include standard piano key patterns (groups of 2 and 3 black keys) for validation.

### Robustness Requirements:
- Handle keyboards cut off at edges
- Work with different lighting conditions
- Accommodate slight perspective variations
- Filter out false positives (non-key objects)

### Dependencies:
- opencv-python
- numpy
- pillow
- streamlit (for testing interface)

### Testing Interface:
Create a Streamlit test application (`test_app.py`) that provides:
- **Image Upload**: File uploader for JPG/PNG keyboard images
- **Interactive Parameter Tuning**: 
  - Sliders for OpenCV parameters (Canny thresholds, contour area limits, blur kernel size)
  - Dropdown to force-test specific keyboard types (Auto, 25, 37, 49, 61, 72)
  - Checkboxes to toggle visualization modes
- **Visual Debugging**:
  - Side-by-side display of original image and detection results
  - Overlay detected key boundaries with different colors for white/black keys
  - Show intermediate processing steps (edges, contours, filtered regions)
  - Display detection metadata (key count, keyboard type, confidence score)
- **Real-time Results**: Updates automatically when parameters change

Example Streamlit structure:
```python
import streamlit as st
from piano_key_detector import PianoKeyDetector

st.title("Piano Key Detector - Test Interface")

# Upload section
uploaded_file = st.file_uploader("Upload keyboard image", type=['jpg', 'jpeg', 'png'])

# Parameter controls in sidebar
st.sidebar.header("Detection Parameters")
canny_low = st.sidebar.slider("Canny Low Threshold", 50, 150, 100)
canny_high = st.sidebar.slider("Canny High Threshold", 100, 300, 200)
keyboard_type = st.sidebar.selectbox("Force Keyboard Type", ["Auto", "25", "37", "49", "61", "72"])
show_edges = st.sidebar.checkbox("Show Edge Detection", False)
show_contours = st.sidebar.checkbox("Show Contours", False)

# Results display with columns for before/after
```

Please generate the initial package structure with a working prototype that can detect basic key regions from a top-down keyboard image, plus the Streamlit testing interface. Focus on getting the core detection working first, then use the Streamlit app to iteratively improve the algorithm with visual feedback.