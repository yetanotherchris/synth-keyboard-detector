"""Simplified key detection implementation - detects only the first white key."""
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
    """Detect the first (leftmost) white key from top-down keyboard images."""

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
        """Detect the first white key in the image and return metadata."""

        params = self._merge_params(overrides)
        image_bgr = load_image(image)
        if image_bgr is None:
            raise ValueError("Could not load image")

        height, width = image_bgr.shape[:2]
        
        # Stage 1: Crop to bottom 30% of the image to remove background
        stage1_crop_y = int(height * 0.7)  # Start from 70% down (keep bottom 30%)
        stage1_crop = image_bgr[stage1_crop_y:, :]
        stage1_height, stage1_width = stage1_crop.shape[:2]
        
        # Stage 2: Detect white key height and crop to just the keyboard
        keyboard_crop, keyboard_crop_y_offset = self._crop_to_keyboard_area(
            stage1_crop, params
        )
        keyboard_height, keyboard_width = keyboard_crop.shape[:2]
        
        # Process the final cropped keyboard image
        gray = cv2.cvtColor(keyboard_crop, cv2.COLOR_BGR2GRAY)
        blur_kernel = ensure_odd(int(params.blur_kernel_size))
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

        edges = cv2.Canny(blurred, int(params.canny_low), int(params.canny_high))

        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if params.adaptive_threshold_block_size > 0:
            block_size = ensure_odd(int(params.adaptive_threshold_block_size))
            adaptive = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                int(params.adaptive_threshold_c),
            )
            binary = cv2.bitwise_and(binary, adaptive)

        white_mask = binary

        # Find the first (leftmost) white key using the keyboard crop
        first_white_key = self._find_first_white_key(
            white_mask, 
            keyboard_crop, 
            keyboard_width, 
            keyboard_height, 
            params,
            y_offset=0  # Keep coordinates relative to keyboard crop
        )

        result: Dict[str, object] = {
            "first_white_key": first_white_key,
            "found": first_white_key is not None,
            "parameters": params.to_dict(),
        }

        if debug:
            # Save cropped image for debugging
            save_debug_image(stage1_crop, "30_percent_crop")
            save_debug_image(keyboard_crop, "keyboard_crop")
            
            result["debug"] = {
                "original": image_bgr,
                "stage1_crop": stage1_crop,
                "keyboard_crop": keyboard_crop,
                "gray": gray,
                "blurred": blurred,
                "edges": edges,
                "binary": white_mask,
            }

        return result

    def _crop_to_keyboard_area(
        self, 
        stage1_crop: np.ndarray, 
        params: DetectionParameters
    ) -> Tuple[np.ndarray, int]:
        """
        Detect white key height and crop to just the keyboard area.
        Returns the cropped keyboard image and the y-offset from stage1_crop.
        """
        height, width = stage1_crop.shape[:2]
        gray = cv2.cvtColor(stage1_crop, cv2.COLOR_BGR2GRAY)
        
        # Apply basic preprocessing to highlight white keys
        blur_kernel = ensure_odd(int(params.blur_kernel_size))
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        
        # Threshold to find white regions (white keys)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find horizontal projection to detect keyboard boundaries
        row_projection = binary.sum(axis=1).astype(np.float32)
        
        if row_projection.max() <= 0:
            # Fallback: return the original stage1_crop
            return stage1_crop, 0
        
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
        min_keyboard_height = int(height * 0.1)  # At least 10% of stage1 height
        
        if keyboard_height < min_keyboard_height:
            # Fallback: return the original stage1_crop
            return stage1_crop, 0
        
        # Add small padding to include key edges
        padding = int(keyboard_height * 0.05)  # 5% padding
        keyboard_top = max(0, keyboard_top - padding)
        keyboard_bottom = min(height - 1, keyboard_bottom + padding)
        
        # Crop to the detected keyboard area
        keyboard_crop = stage1_crop[keyboard_top:keyboard_bottom + 1, :]
        
        return keyboard_crop, keyboard_top

    def detect_all_white_key_boundaries(
        self,
        image,
        debug: bool = True,
        **overrides,
    ) -> Dict[str, object]:
        """Detect all white key boundaries in the image and return metadata."""

        params = self._merge_params(overrides)
        image_bgr = load_image(image)
        if image_bgr is None:
            raise ValueError("Could not load image")

        height, width = image_bgr.shape[:2]
        
        # Stage 1: Crop to bottom 30% of the image to remove background
        stage1_crop_y = int(height * 0.7)  # Start from 70% down (keep bottom 30%)
        stage1_crop = image_bgr[stage1_crop_y:, :]
        stage1_height, stage1_width = stage1_crop.shape[:2]
        
        # Stage 2: Detect white key height and crop to just the keyboard
        keyboard_crop, keyboard_crop_y_offset = self._crop_to_keyboard_area(
            stage1_crop, params
        )
        keyboard_height, keyboard_width = keyboard_crop.shape[:2]
        
        # Process the final cropped keyboard image
        gray = cv2.cvtColor(keyboard_crop, cv2.COLOR_BGR2GRAY)
        blur_kernel = ensure_odd(int(params.blur_kernel_size))
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

        edges = cv2.Canny(blurred, int(params.canny_low), int(params.canny_high))

        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if params.adaptive_threshold_block_size > 0:
            block_size = ensure_odd(int(params.adaptive_threshold_block_size))
            adaptive = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                int(params.adaptive_threshold_c),
            )
            binary = cv2.bitwise_and(binary, adaptive)

        white_mask = binary

        # Find all white key boundaries
        all_white_keys = self._find_all_white_key_boundaries(
            white_mask, 
            keyboard_crop, 
            keyboard_width, 
            keyboard_height, 
            params
        )

        result: Dict[str, object] = {
            "all_white_keys": all_white_keys,
            "num_keys_found": len(all_white_keys),
            "parameters": params.to_dict(),
        }

        if debug:
            # Save cropped image for debugging
            save_debug_image(stage1_crop, "30_percent_crop_all_keys")
            
            # Create visualization with all key boundaries
            visualization = self._create_key_boundaries_visualization(
                stage1_crop, all_white_keys, stage1_crop_y, keyboard_crop_y_offset
            )
            save_debug_image(visualization, "all_white_key_boundaries")
            
            result["debug"] = {
                "original": image_bgr,
                "stage1_crop": stage1_crop,
                "keyboard_crop": keyboard_crop,
                "gray": gray,
                "blurred": blurred,
                "edges": edges,
                "binary": white_mask,
                "visualization": visualization,
            }

        return result

    def keyboard_crop_with_key_detection(
        self,
        image,
        debug: bool = True,
        **overrides,
    ) -> Dict[str, object]:
        """
        3-stage cropping and detection:
        1. Crop to bottom 30% of image
        2. Detect keyboard area and crop further  
        3. Create additional crop for better white key detection
        4. Use stage 3 to detect key width, then apply to stage 2 image
        """
        
        params = self._merge_params(overrides)
        image_bgr = load_image(image)
        if image_bgr is None:
            raise ValueError("Could not load image")

        height, width = image_bgr.shape[:2]
        
        # Stage 1: Crop to bottom 30% of the image to remove background
        stage1_crop_y = int(height * 0.7)  # Start from 70% down (keep bottom 30%)
        stage1_crop = image_bgr[stage1_crop_y:, :]
        stage1_height, stage1_width = stage1_crop.shape[:2]
        
        # Stage 2: Detect white key height and crop to just the keyboard
        stage2_crop, stage2_crop_y_offset = self._crop_to_keyboard_area(
            stage1_crop, params
        )
        stage2_height, stage2_width = stage2_crop.shape[:2]
        
        # Stage 3: Create additional crop for better white key detection
        # Take middle portion of stage2 to focus on white key area
        stage3_crop, stage3_crop_y_offset = self._create_stage3_crop(stage2_crop)
        stage3_height, stage3_width = stage3_crop.shape[:2]
        
        # Detect white keys on stage 3 to determine key width
        stage3_white_keys = self._detect_white_keys_stage3(stage3_crop, params)
        
        # Extract key width information from stage 3 detection
        detected_key_width = self._extract_key_width_from_stage3(stage3_white_keys, stage3_width)
        
        # Use detected key width to find all white keys on stage 2 image
        stage2_white_keys = self._apply_detection_to_stage2(
            stage2_crop, detected_key_width, params
        )
        
        result: Dict[str, object] = {
            "stage3_white_keys": stage3_white_keys,
            "stage2_white_keys": stage2_white_keys, 
            "detected_key_width": detected_key_width,
            "num_stage3_keys": len(stage3_white_keys),
            "num_stage2_keys": len(stage2_white_keys),
            "parameters": params.to_dict(),
        }

        if debug:
            # Save all crops for debugging
            save_debug_image(stage1_crop, "stage1_30_percent_crop")
            save_debug_image(stage2_crop, "stage2_keyboard_crop") 
            save_debug_image(stage3_crop, "stage3_detection_crop")
            
            # Create visualization showing boundaries on 30% crop only
            visualization = self._create_stage1_visualization(
                stage1_crop, stage3_white_keys, stage2_crop_y_offset, stage3_crop_y_offset
            )
            save_debug_image(visualization, "stage1_with_boundaries")
            
            result["debug"] = {
                "original": image_bgr,
                "stage1_crop": stage1_crop,
                "stage2_crop": stage2_crop,
                "stage3_crop": stage3_crop,
                "visualization": visualization,
            }

        return result

    def _merge_params(self, overrides: Dict[str, object]) -> DetectionParameters:
        params_dict = self.params.to_dict()
        params_dict.update(overrides)
        return DetectionParameters(**params_dict)

    def _find_first_white_key(
        self,
        white_mask: np.ndarray,
        cropped_image: np.ndarray,
        width: int,
        height: int,
        params: DetectionParameters,
        y_offset: int = 0,
    ) -> Optional[Dict[str, object]]:
        """Find the leftmost white key using column projection in the cropped region."""
        
        # Get column projection to find white regions
        column_projection = white_mask.sum(axis=0).astype(np.float32)
        if column_projection.max() <= 0:
            return None
        
        # Find threshold for detecting white key regions
        threshold = column_projection.max() * params.projection_peak_ratio
        min_width = max(int(params.min_white_key_width_ratio * width), 3)
        max_width = int(params.max_white_key_width_ratio * width)
        
        # Find the first significant white region from left to right
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
                        # Found the first valid white key region
                        return self._extract_key_from_region(
                            cropped_image, start, end, height, params, y_offset
                        )
                    in_region = False
        
        # Handle case where region extends to the end
        if in_region:
            end = len(column_projection) - 1
            region_width = end - start
            if min_width <= region_width <= max_width:
                return self._extract_key_from_region(
                    cropped_image, start, end, height, params, y_offset
                )
        
        return None

    def _extract_key_from_region(
        self,
        cropped_image: np.ndarray,
        x_start: int,
        x_end: int,
        height: int,
        params: DetectionParameters,
        y_offset: int = 0,
    ) -> Dict[str, object]:
        """Extract key information from a detected white region in the cropped image."""
        
        # Convert to grayscale for analysis
        if len(cropped_image.shape) == 3:
            gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cropped_image
        
        # Since we're working with the bottom 30% crop, use the full height
        y_start = 0
        y_end = height - 1
        
        # Find the true right boundary of the white key in the cropped region
        true_right_x = self._find_true_right_boundary_cropped(gray, x_start, x_end, y_start, y_end)
        
        # Calculate bounding box in keyboard crop coordinates
        x = x_start
        y = y_start  # Keep relative to keyboard crop
        w = true_right_x - x_start
        h = y_end - y_start + 1
        
        # Calculate center in keyboard crop coordinates
        cx = x + w / 2.0
        cy = y + h / 2.0
        
        # Basic validation
        aspect = h / float(max(w, 1))
        # Use original image dimensions for area calculation
        area_ratio = (w * h) / float(cropped_image.shape[1] * cropped_image.shape[0])
        
        confidence = 1.0
        if aspect < params.min_white_key_aspect:
            confidence *= 0.5
        if area_ratio < params.min_white_key_area_ratio:
            confidence *= 0.5
        
        return {
            "bbox": (x, y, w, h),
            "center": (cx, cy),
            "confidence": confidence,
            "aspect_ratio": aspect,
            "area_ratio": area_ratio,
        }

    def _find_true_right_boundary_cropped(
        self,
        gray: np.ndarray,
        x_start: int,
        x_end: int,
        y_start: int,
        y_end: int,
    ) -> int:
        """
        Find the true right boundary of the white key in the cropped region.
        Since we're working with the bottom 30% crop where black keys don't interfere,
        we can use simpler boundary detection.
        """
        true_right = x_start  # Fallback to left edge if nothing found
        max_gradient = 0
        
        # Sample several rows in the cropped region
        sample_rows = range(y_start, min(y_end + 1, gray.shape[0]), max(1, (y_end - y_start) // 5))
        
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

    def _find_all_white_key_boundaries(
        self,
        white_mask: np.ndarray,
        cropped_image: np.ndarray,
        width: int,
        height: int,
        params: DetectionParameters,
    ) -> list:
        """Find all white key boundaries using column projection in the cropped region."""
        
        # Get column projection to find white regions
        column_projection = white_mask.sum(axis=0).astype(np.float32)
        if column_projection.max() <= 0:
            return []
        
        # Find threshold for detecting white key regions
        threshold = column_projection.max() * params.projection_peak_ratio
        min_width = max(int(params.min_white_key_width_ratio * width), 3)
        max_width = int(params.max_white_key_width_ratio * width)
        
        all_keys = []
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
                        # Found a valid white key region
                        key_info = self._extract_key_from_region(
                            cropped_image, start, end, height, params, y_offset=0
                        )
                        if key_info:
                            all_keys.append(key_info)
                    in_region = False
        
        # Handle case where region extends to the end
        if in_region:
            end = len(column_projection) - 1
            region_width = end - start
            if min_width <= region_width <= max_width:
                key_info = self._extract_key_from_region(
                    cropped_image, start, end, height, params, y_offset=0
                )
                if key_info:
                    all_keys.append(key_info)
        
        return all_keys

    def _create_key_boundaries_visualization(
        self,
        stage1_crop: np.ndarray,
        all_white_keys: list,
        stage1_crop_y: int,
        keyboard_crop_y_offset: int,
    ) -> np.ndarray:
        """Create a visualization showing detected white key boundaries on the original crop."""
        
        # Create a copy of the stage1_crop for visualization
        vis_image = stage1_crop.copy()
        
        # Draw rectangles for each detected white key
        for i, key in enumerate(all_white_keys):
            bbox = key['bbox']
            x, y, w, h = bbox
            
            # Adjust coordinates to stage1_crop space
            # The bbox is relative to keyboard_crop, so we need to add the keyboard_crop_y_offset
            adjusted_y = y + keyboard_crop_y_offset
            
            # Draw rectangle (green color)
            cv2.rectangle(vis_image, (x, adjusted_y), (x + w, adjusted_y + h), (0, 255, 0), 2)
            
            # Add key number label
            label = f"Key {i+1}"
            label_pos = (x + 5, adjusted_y + 20)
            cv2.putText(vis_image, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return vis_image

    def _create_stage3_crop(self, stage2_crop: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Create stage 3 crop for better white key detection.
        Takes the middle portion of stage2 to focus on white key area.
        """
        height, width = stage2_crop.shape[:2]
        
        # Take middle 60% of the height to focus on white keys
        crop_start_ratio = 0.2  # Start at 20% from top
        crop_end_ratio = 0.8    # End at 80% from top
        
        start_y = int(height * crop_start_ratio)
        end_y = int(height * crop_end_ratio)
        
        # Ensure we have at least some height
        if end_y - start_y < 20:
            start_y = 0
            end_y = height
        
        stage3_crop = stage2_crop[start_y:end_y, :]
        
        return stage3_crop, start_y

    def _detect_white_keys_stage3(self, stage3_crop: np.ndarray, params: DetectionParameters) -> list:
        """
        Detect white keys on stage 3 crop using enhanced processing.
        """
        height, width = stage3_crop.shape[:2]
        
        # Process the stage3 crop
        gray = cv2.cvtColor(stage3_crop, cv2.COLOR_BGR2GRAY)
        blur_kernel = ensure_odd(int(params.blur_kernel_size))
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

        # Use binary threshold to find white regions
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply adaptive threshold for better edge detection
        if params.adaptive_threshold_block_size > 0:
            block_size = ensure_odd(int(params.adaptive_threshold_block_size))
            adaptive = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                int(params.adaptive_threshold_c),
            )
            binary = cv2.bitwise_and(binary, adaptive)

        # Find white key boundaries using column projection
        return self._find_all_white_key_boundaries(
            binary, stage3_crop, width, height, params
        )

    def _extract_key_width_from_stage3(self, stage3_white_keys: list, stage3_width: int) -> Optional[int]:
        """
        Extract typical white key width from stage 3 detection results.
        """
        if not stage3_white_keys:
            return None
        
        # Calculate widths of detected keys
        widths = []
        for key in stage3_white_keys:
            bbox = key['bbox']
            w = bbox[2]  # width is the 3rd element
            widths.append(w)
        
        if not widths:
            return None
        
        # Use median width as the typical key width
        widths.sort()
        median_width = widths[len(widths) // 2]
        
        return median_width

    def _apply_detection_to_stage2(
        self, 
        stage2_crop: np.ndarray, 
        detected_key_width: Optional[int], 
        params: DetectionParameters
    ) -> list:
        """
        Apply detection to stage 2 image using the key width detected from stage 3.
        """
        if detected_key_width is None:
            # Fallback to regular detection
            height, width = stage2_crop.shape[:2]
            gray = cv2.cvtColor(stage2_crop, cv2.COLOR_BGR2GRAY)
            blur_kernel = ensure_odd(int(params.blur_kernel_size))
            blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return self._find_all_white_key_boundaries(
                binary, stage2_crop, width, height, params
            )
        
        # Use detected key width to find keys across the full stage2 width
        height, width = stage2_crop.shape[:2]
        
        # Process stage2 crop
        gray = cv2.cvtColor(stage2_crop, cv2.COLOR_BGR2GRAY)
        blur_kernel = ensure_odd(int(params.blur_kernel_size))
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Use the detected key width to find regular pattern of white keys
        keys = []
        num_expected_keys = max(1, width // detected_key_width)
        
        for i in range(num_expected_keys):
            x_start = i * detected_key_width
            x_end = min(x_start + detected_key_width, width)
            
            if x_end - x_start < detected_key_width * 0.5:  # Skip partial keys at the end
                break
            
            # Check if this region contains enough white pixels
            region_mask = binary[:, x_start:x_end]
            white_ratio = np.sum(region_mask > 128) / (region_mask.shape[0] * region_mask.shape[1])
            
            if white_ratio > 0.3:  # At least 30% white pixels
                key_info = {
                    'bbox': (x_start, 0, x_end - x_start, height),
                    'center': (x_start + (x_end - x_start) // 2, height // 2),
                    'confidence': white_ratio,
                    'area': (x_end - x_start) * height
                }
                keys.append(key_info)
        
        return keys

    def _create_stage1_visualization(
        self,
        stage1_crop: np.ndarray,
        stage3_white_keys: list,
        stage2_crop_y_offset: int,
        stage3_crop_y_offset: int,
    ) -> np.ndarray:
        """
        Create visualization showing detected white key boundaries on the 30% crop (stage1).
        Boundaries are from stage3 detection but displayed on stage1 image.
        """
        # Create a copy of the stage1_crop for visualization
        vis_image = stage1_crop.copy()
        
        # Draw rectangles for each detected white key from stage3
        for i, key in enumerate(stage3_white_keys):
            bbox = key['bbox']
            x, y, w, h = bbox
            
            # Adjust coordinates to stage1_crop space
            # The bbox is relative to stage3_crop, so we need to add both offsets
            adjusted_y = y + stage2_crop_y_offset + stage3_crop_y_offset
            
            # Draw rectangle (green color)
            cv2.rectangle(vis_image, (x, adjusted_y), (x + w, adjusted_y + h), (0, 255, 0), 2)
            
            # Add key number label
            label = f"Key {i+1}"
            label_pos = (x + 5, adjusted_y + 20)
            cv2.putText(vis_image, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return vis_image

