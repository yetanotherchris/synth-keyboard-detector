"""Piano key detection implementation following first-steps.md exactly."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .utils import (
    ensure_odd,
    load_image,
    save_debug_image,
)


@dataclass
class DetectionParameters:
    """Configuration for the detector."""

    blur_kernel_size: int = 5
    canny_low: int = 80
    canny_high: int = 180
    adaptive_threshold_block_size: int = 35
    adaptive_threshold_c: int = 5
    projection_peak_ratio: float = 0.55
    min_white_key_width_ratio: float = 0.01
    max_white_key_width_ratio: float = 0.2
    min_white_key_aspect: float = 2.0
    min_white_key_area_ratio: float = 0.003

    def to_dict(self) -> Dict[str, float]:
        return self.__dict__.copy()


class PianoKeyDetector:
    """Detect piano keys following the exact process from first-steps.md."""

    def __init__(self, **kwargs):
        self.params = DetectionParameters(**kwargs)

    def update_params(self, **kwargs) -> None:
        """Update default parameters in-place."""
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)

    def detect_keys(
        self,
        image,
        debug: bool = True,
        **overrides,
    ) -> Dict[str, object]:
        """
        Detect piano keys following the exact 8-step process from first-steps.md:
        1. Crop the bottom 50% of the uploaded image.
        2. Find the white keys in this image.
        3. Crop the image to the height of these keys, the area they are inside.
        4. Further crop the image produced in stage 3, to the bottom 30% of this image.
        5. You should have the white keys only now.
        6. Find the first white key and highlight it in green, using 50% opacity green and shaded as dots.
        7. Highlight the right side boundary of this key using a dotted black line, 50% opacity.
        8. Display each of these stages as images in streamlit.
        """

        params = self._merge_params(overrides)
        image_bgr = load_image(image)
        if image_bgr is None:
            raise ValueError("Could not load image")

        height, width = image_bgr.shape[:2]
        
        # Stage 1: Crop the bottom 50% of the uploaded image
        stage1_crop_y = int(height * 0.5)  # Start from 50% down (keep bottom 50%)
        stage1_crop = image_bgr[stage1_crop_y:, :]
        stage1_height, stage1_width = stage1_crop.shape[:2]
        
        # Stage 2: Find the white keys in this image
        white_keys_region = self._find_white_keys_region(stage1_crop, params)
        
        # Stage 3: Crop the image to the height of these keys
        if white_keys_region is None:
            # Fallback if no white keys detected
            stage3_crop = stage1_crop
            stage3_crop_y_offset = 0
        else:
            top_y, bottom_y = white_keys_region
            stage3_crop = stage1_crop[top_y:bottom_y, :]
            stage3_crop_y_offset = top_y
        
        stage3_height, stage3_width = stage3_crop.shape[:2]
        
        # Stage 4: Further crop to the bottom 30% of this image
        stage4_crop_y = int(stage3_height * 0.7)  # Start from 70% down (keep bottom 30%)
        stage4_crop = stage3_crop[stage4_crop_y:, :]
        stage4_height, stage4_width = stage4_crop.shape[:2]
        
        # Stage 5: You should have the white keys only now
        # This is our final white keys image
        
        # Stage 6 & 7: Find the first white key and create overlays
        first_white_key = self._find_first_white_key_exact(stage4_crop, params)
        
        # Create highlighted images (stages 6 & 7)
        stage6_highlighted = None
        stage7_highlighted = None
        
        if first_white_key is not None:
            # Stage 6: Highlight first key in green with 50% opacity dots
            stage6_highlighted = self._create_green_dots_overlay(stage4_crop, first_white_key)
            
            # Stage 7: Add dotted black line for right boundary at 50% opacity
            stage7_highlighted = self._create_dotted_boundary_overlay(stage6_highlighted, first_white_key)
        
        # Convert coordinates back to original image space for API compatibility
        white_keys = []
        if first_white_key is not None:
            # Calculate total offset from original image
            total_y_offset = stage1_crop_y + stage3_crop_y_offset + stage4_crop_y
            
            bbox = first_white_key["bbox"]
            x, y, w, h = bbox
            
            # Transform to original image coordinates
            original_bbox = (x, y + total_y_offset, w, h)
            
            white_key_entry = {
                "bbox": original_bbox,
                "center": (x + w/2, y + total_y_offset + h/2),
                "confidence": first_white_key["confidence"],
                "aspect_ratio": first_white_key.get("aspect_ratio", 0),
                "area_ratio": first_white_key.get("area_ratio", 0),
            }
            white_keys.append(white_key_entry)

        result: Dict[str, object] = {
            "white_keys": white_keys,
            "black_keys": [],  # Not implemented in this version
            "total_keys": len(white_keys),
            "confidence": white_keys[0]["confidence"] if white_keys else 0.0,
            "keyboard_type": "unknown",
            "bounding_box": None,
            "first_white_key": first_white_key,
            "found": first_white_key is not None,
            "parameters": params.to_dict(),
        }

        if debug:
            # Save all processing stages
            save_debug_image(stage1_crop, "stage1_bottom_50_percent")
            save_debug_image(stage3_crop, "stage3_keyboard_height")
            save_debug_image(stage4_crop, "stage4_white_keys_only")
            
            if stage6_highlighted is not None:
                save_debug_image(stage6_highlighted, "stage6_green_dots")
            if stage7_highlighted is not None:
                save_debug_image(stage7_highlighted, "stage7_dotted_boundary")
            
            result["debug"] = {
                "original": image_bgr,
                "stage1_crop": stage1_crop,  # Bottom 50%
                "stage3_crop": stage3_crop,  # Keyboard height  
                "stage4_crop": stage4_crop,  # White keys only (bottom 30%)
                "stage6_highlighted": stage6_highlighted,  # Green dots overlay
                "stage7_highlighted": stage7_highlighted,  # Dotted boundary
                "white_keys_region": white_keys_region,
            }

        return result

    def _merge_params(self, overrides: Dict[str, object]) -> DetectionParameters:
        params_dict = self.params.to_dict()
        params_dict.update(overrides)
        return DetectionParameters(**params_dict)

    def _find_white_keys_region(self, image: np.ndarray, params: DetectionParameters) -> Optional[Tuple[int, int]]:
        """
        Find the vertical region containing white keys.
        Returns (top_y, bottom_y) tuple or None if not found.
        """
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply basic preprocessing to highlight white keys
        blur_kernel = ensure_odd(int(params.blur_kernel_size))
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        
        # Threshold to find white regions (white keys)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find horizontal projection to detect keyboard boundaries
        row_projection = binary.sum(axis=1).astype(np.float32)
        
        if row_projection.max() <= 0:
            return None
        
        # Find the top and bottom of the keyboard by detecting white key regions
        threshold = row_projection.max() * 0.3  # Lower threshold for white key detection
        
        # Find top boundary (first significant white region from top)
        keyboard_top = 0
        for i, value in enumerate(row_projection):
            if value >= threshold:
                keyboard_top = i
                break
        
        # Find bottom boundary (last significant white region from bottom)
        keyboard_bottom = height - 1
        for i in range(height - 1, -1, -1):
            if row_projection[i] >= threshold:
                keyboard_bottom = i
                break
        
        # Ensure we have a reasonable keyboard height
        keyboard_height = keyboard_bottom - keyboard_top + 1
        min_keyboard_height = int(height * 0.1)  # At least 10% of input height
        
        if keyboard_height < min_keyboard_height:
            return None
        
        # Add small padding to include key edges
        padding = int(keyboard_height * 0.05)  # 5% padding
        keyboard_top = max(0, keyboard_top - padding)
        keyboard_bottom = min(height - 1, keyboard_bottom + padding)
        
        return (keyboard_top, keyboard_bottom)

    def _find_first_white_key_exact(self, white_keys_image: np.ndarray, params: DetectionParameters) -> Optional[Dict[str, object]]:
        """
        Find the first (leftmost) white key in the white keys only image.
        Enhanced to handle cases where keys blend together.
        """
        height, width = white_keys_image.shape[:2]
        gray = cv2.cvtColor(white_keys_image, cv2.COLOR_BGR2GRAY)
        
        # Apply preprocessing
        blur_kernel = ensure_odd(int(params.blur_kernel_size))
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        
        # Use edge detection to find key boundaries
        edges = cv2.Canny(blurred, int(params.canny_low), int(params.canny_high))
        
        # Also use binary threshold for backup
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Method 1: Look for vertical edges that indicate key boundaries
        key_boundary = self._find_first_key_by_edges(edges, width)
        if key_boundary is not None:
            x_start, x_end = key_boundary
            return self._extract_key_from_region(white_keys_image, x_start, x_end, height, params)
        
        # Method 2: Use column projection with lower thresholds
        column_projection = binary.sum(axis=0).astype(np.float32)
        if column_projection.max() > 0:
            key_boundary = self._find_first_key_by_projection(column_projection, width, params)
            if key_boundary is not None:
                x_start, x_end = key_boundary
                return self._extract_key_from_region(white_keys_image, x_start, x_end, height, params)
        
        # Method 3: Estimate key width only if there's significant brightness variation
        # Check for brightness variation that suggests actual content
        brightness_std = np.std(gray)
        if (column_projection.max() > column_projection.min() * 2 and brightness_std > 5):  # Some variation in brightness and structure
            estimated_key_width = width // 7  # Assume about 7 white keys in view (typical)
            if estimated_key_width > 10:  # Reasonable minimum key width
                x_start = 0
                x_end = min(estimated_key_width, width)
                
                # Validate that this region actually looks like a key
                region = white_keys_image[:, x_start:x_end]
                region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                
                # Check if region has reasonable brightness (should be relatively bright for white keys)
                mean_brightness = np.mean(region_gray)
                if mean_brightness > 140:  # Should be reasonably bright for white keys
                    return self._extract_key_from_region(white_keys_image, x_start, x_end, height, params)
        
        return None

    def _find_first_key_by_edges(self, edges: np.ndarray, width: int) -> Optional[Tuple[int, int]]:
        """Find the first key by detecting vertical edges that separate keys."""
        height = edges.shape[0]
        
        # First check if there are any edges at all
        total_edges = np.sum(edges > 0)
        if total_edges < height * 0.1:  # Need at least some edges to be meaningful
            return None
        
        # Look for strong vertical lines that could be key separators
        min_key_width = max(width // 20, 10)  # Minimum reasonable key width
        max_key_width = width // 3  # Maximum reasonable key width
        
        # Scan for the first strong vertical edge from left to right
        for x in range(min_key_width, min(max_key_width, width - 1)):
            # Count vertical edge pixels in this column
            edge_count = np.sum(edges[:, x] > 0)
            
            # If we find a strong vertical edge, this could be the right boundary of first key
            if edge_count > height * 0.3:  # At least 30% of column has edges
                return (0, x)
        
        return None

    def _find_first_key_by_projection(self, column_projection: np.ndarray, width: int, params: DetectionParameters) -> Optional[Tuple[int, int]]:
        """Find the first key using column projection with enhanced logic."""
        
        # Use much lower threshold since all columns might be white
        threshold = column_projection.max() * 0.1  # Very low threshold
        min_width = max(int(params.min_white_key_width_ratio * width), 5)
        max_width = int(params.max_white_key_width_ratio * width)
        
        # If almost everything is white, look for subtle variations
        if np.sum(column_projection >= threshold) > width * 0.8:
            # Look for the first dip in the projection that could indicate a key boundary
            smoothed = np.convolve(column_projection, np.ones(5)/5, mode='same')
            
            # Find the first local minimum that could be a key separator
            for i in range(min_width, min(max_width, len(smoothed) - 1)):
                if (i > 0 and i < len(smoothed) - 1 and
                    smoothed[i] < smoothed[i-1] and smoothed[i] < smoothed[i+1] and
                    smoothed[i] < smoothed.max() * 0.9):  # At least 10% drop
                    return (0, i)
            
            # Fallback: estimate first key width
            estimated_width = min(width // 7, max_width)  # Assume 7 keys, or use max
            if estimated_width >= min_width:
                return (0, estimated_width)
        
        # Original logic for cases with clear white/dark boundaries
        in_region = False
        start = 0
        
        for idx, value in enumerate(column_projection):
            if value >= threshold:
                if not in_region:
                    in_region = True
                    start = idx
            else:
                if in_region:
                    end = idx
                    region_width = end - start
                    if min_width <= region_width <= max_width:
                        return (start, end)
                    in_region = False
        
        # Handle case where region extends to the end
        if in_region:
            end = len(column_projection) - 1
            region_width = end - start
            if min_width <= region_width <= max_width:
                return (start, end)
        
        return None

    def _extract_key_from_region(
        self,
        image: np.ndarray,
        x_start: int,
        x_end: int,
        height: int,
        params: DetectionParameters,
    ) -> Optional[Dict[str, object]]:
        """Extract key information from a detected white region."""
        
        # Validate input parameters
        if x_end <= x_start or x_start < 0 or x_end > image.shape[1]:
            return None
        
        # Find the true right boundary of the white key
        true_right_x = self._find_true_right_boundary(image, x_start, x_end)
        
        # Calculate bounding box
        x = x_start
        y = 0
        w = true_right_x - x_start
        h = height
        
        # Validate minimum size
        if w <= 0 or h <= 0:
            return None
        
        # Calculate center
        cx = x + w / 2.0
        cy = y + h / 2.0
        
        # Basic validation
        aspect = h / float(max(w, 1))
        area_ratio = (w * h) / float(image.shape[1] * image.shape[0])
        
        confidence = 1.0
        if aspect < params.min_white_key_aspect:
            confidence *= 0.5
        if area_ratio < params.min_white_key_area_ratio:
            confidence *= 0.5
        
        # Additional validation for very small keys
        if w < 5 or area_ratio < 0.001:  # Reject tiny regions
            return None
        
        return {
            "bbox": (x, y, w, h),
            "center": (cx, cy),
            "confidence": confidence,
            "aspect_ratio": aspect,
            "area_ratio": area_ratio,
        }

    def _find_true_right_boundary(self, image: np.ndarray, x_start: int, x_end: int) -> int:
        """Find the true right boundary of the white key."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height = gray.shape[0]
        
        true_right = x_start  # Fallback to left edge if nothing found
        max_gradient = 0
        
        # Sample several rows
        sample_rows = range(0, height, max(1, height // 5))
        
        for y in sample_rows:
            if y >= gray.shape[0]:
                continue
                
            # Scan from right to left within the detected region
            for x in range(min(x_end - 1, gray.shape[1] - 1), x_start, -1):
                if x >= gray.shape[1]:
                    continue
                    
                pixel_val = gray[y, x]
                
                # Look for white/light pixels (key surface)
                if pixel_val > 180:  # White key surface
                    # Check if the next pixel to the right is significantly darker (boundary)
                    if x + 1 < gray.shape[1]:
                        next_pixel = int(gray[y, x + 1])
                        gradient = int(pixel_val) - next_pixel
                        
                        # If there's a significant drop in brightness (white key edge)
                        if gradient > 20 and gradient > max_gradient:
                            max_gradient = gradient
                            true_right = max(true_right, x + 1)
                            break
                    else:
                        # At image edge
                        true_right = max(true_right, x + 1)
                        break
                
                # Look for light grey pixels (key surface with slight shading)
                elif pixel_val > 140:  # Light grey key surface
                    if x + 1 < gray.shape[1]:
                        next_pixel = int(gray[y, x + 1])
                        gradient = int(pixel_val) - next_pixel
                        
                        if gradient > 12 and gradient > max_gradient:
                            max_gradient = gradient
                            true_right = max(true_right, x + 1)
                            break
                    else:
                        true_right = max(true_right, x + 1)
                        break
        
        # Ensure we found a reasonable boundary
        if true_right <= x_start:
            # Fallback: look for the rightmost bright pixel
            for y in sample_rows:
                if y >= gray.shape[0]:
                    continue
                for x in range(min(x_end - 1, gray.shape[1] - 1), x_start, -1):
                    if x >= gray.shape[1]:
                        continue
                    if gray[y, x] > 160:  # Any reasonably bright pixel
                        true_right = max(true_right, x + 1)
                        break
        
        return min(true_right, x_end)  # Don't exceed original detection

    def _create_green_dots_overlay(self, image: np.ndarray, key_info: Dict[str, object]) -> np.ndarray:
        """
        Create green dots overlay with 50% opacity on the first white key.
        """
        overlay = image.copy()
        bbox = key_info["bbox"]
        x, y, w, h = bbox
        
        # Create a mask for the key area
        mask = np.zeros(overlay.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        
        # Create dots pattern
        dot_size = max(2, min(w, h) // 10)  # Dot size based on key size
        dot_spacing = dot_size * 3  # Space between dots
        
        # Create green overlay with dots
        green_overlay = overlay.copy()
        
        # Fill the key area with dots
        for dy in range(y, y + h, dot_spacing):
            for dx in range(x, x + w, dot_spacing):
                if dy < overlay.shape[0] and dx < overlay.shape[1]:
                    cv2.circle(green_overlay, (dx, dy), dot_size, (0, 255, 0), -1)
        
        # Apply 50% opacity blending
        alpha = 0.5
        result = cv2.addWeighted(overlay, 1 - alpha, green_overlay, alpha, 0)
        
        return result

    def _create_dotted_boundary_overlay(self, image: np.ndarray, key_info: Dict[str, object]) -> np.ndarray:
        """
        Add dotted black line for right boundary at 50% opacity.
        """
        overlay = image.copy()
        bbox = key_info["bbox"]
        x, y, w, h = bbox
        
        # Right boundary line
        right_x = x + w
        
        # Create dotted line
        dot_size = 2
        dot_spacing = 6
        
        # Draw dotted line along the right boundary
        for dy in range(y, y + h, dot_spacing):
            if dy < overlay.shape[0] and right_x < overlay.shape[1]:
                cv2.circle(overlay, (right_x, dy), dot_size, (0, 0, 0), -1)
        
        # Apply 50% opacity blending for the boundary line area
        alpha = 0.5
        line_overlay = image.copy()
        
        # Draw the boundary line area on the line overlay
        for dy in range(y, y + h, dot_spacing):
            if dy < line_overlay.shape[0] and right_x < line_overlay.shape[1]:
                cv2.circle(line_overlay, (right_x, dy), dot_size, (0, 0, 0), -1)
        
        # Blend only the line area
        result = cv2.addWeighted(image, 1 - alpha, line_overlay, alpha, 0)
        
        return result