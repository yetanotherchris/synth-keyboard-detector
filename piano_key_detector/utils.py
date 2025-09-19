"""Utility helpers for the piano key detector."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np


def ensure_odd(value: int, minimum: int = 3) -> int:
    """Return ``value`` adjusted to the nearest odd integer greater than or equal to ``minimum``."""
    if value < minimum:
        value = minimum
    if value % 2 == 0:
        value += 1
    return value


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` to the inclusive range ``[low, high]``."""
    return max(low, min(high, value))


def load_image(image_source) -> np.ndarray:
    """Load an image as a BGR ``numpy`` array.

    Parameters
    ----------
    image_source:
        Either a path, a file-like object, bytes, or a numpy array. If a numpy array
        is provided it is returned unchanged when already BGR, otherwise converted.
    """

    if isinstance(image_source, np.ndarray):
        image = image_source.copy()
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    if isinstance(image_source, (str, Path)):
        image = cv2.imread(str(image_source))
        if image is None:
            raise ValueError(f"Could not read image from path: {image_source}")
        return image

    # File-like or bytes
    if hasattr(image_source, "read"):
        data = image_source.read()
        if hasattr(image_source, "seek"):
            try:
                image_source.seek(0)
            except Exception:
                pass
    else:
        data = image_source

    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("Unsupported image source type")

    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image from bytes")
    return image


def normalize_point(x: float, y: float, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Return coordinates normalized to the provided bounding box."""
    bx, by, bw, bh = bbox
    if bw <= 0 or bh <= 0:
        return 0.0, 0.0
    return (x - bx) / float(bw), (y - by) / float(bh)


def bbox_union(boxes: Iterable[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    """Return the union bounding box covering ``boxes``."""
    boxes = list(boxes)
    if not boxes:
        return (0, 0, 0, 0)
    xs = [b[0] for b in boxes]
    ys = [b[1] for b in boxes]
    x2s = [b[0] + b[2] for b in boxes]
    y2s = [b[1] + b[3] for b in boxes]
    x = int(min(xs))
    y = int(min(ys))
    w = int(max(x2s) - x)
    h = int(max(y2s) - y)
    return (x, y, w, h)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to RGB."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_debug_image(image: np.ndarray, filename: str, prefix: str = "debug_") -> str:
    """Save a debug image to disk for visualization.
    
    Parameters
    ----------
    image : np.ndarray
        The image to save (BGR format)
    filename : str
        Base filename without extension
    prefix : str
        Prefix to add to filename
        
    Returns
    -------
    str
        The full path where the image was saved
    """
    full_filename = f"{prefix}{filename}.jpg"
    success = cv2.imwrite(full_filename, image)
    if success:
        print(f"Debug image saved: {full_filename}")
        return full_filename
    else:
        print(f"Failed to save debug image: {full_filename}")
        return ""
