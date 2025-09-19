"""Key detection implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from . import keyboard_models
from .utils import (
    bbox_union,
    clamp,
    ensure_odd,
    load_image,
    normalize_point,
)


@dataclass
class DetectionParameters:
    """Configuration for the detector."""

    blur_kernel_size: int = 5
    canny_low: int = 80
    canny_high: int = 180
    adaptive_threshold_block_size: int = 35
    adaptive_threshold_c: int = 5
    projection_smooth_size: int = 9
    projection_peak_ratio: float = 0.45
    min_white_key_width_ratio: float = 0.01
    max_white_key_width_ratio: float = 0.2
    min_white_key_aspect: float = 2.0
    min_white_key_area_ratio: float = 0.003
    white_key_width_lower_factor: float = 0.45
    white_key_width_upper_factor: float = 1.7
    white_vertical_margin_ratio: float = 0.02
    black_key_height_ratio: float = 0.55
    black_key_relative_width: float = 0.65
    black_key_darkness_threshold: float = 0.35
    black_key_vertical_offset: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return self.__dict__.copy()


class PianoKeyDetector:
    """Detect piano keys from top-down keyboard images."""

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
        force_keyboard_type: Optional[str] = None,
        debug: bool = True,
        **overrides,
    ) -> Dict[str, object]:
        """Detect keys within ``image`` and return metadata."""

        params = self._merge_params(overrides)
        image_bgr = load_image(image)
        if image_bgr is None:
            raise ValueError("Could not load image")

        height, width = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
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

        white_boxes = self._locate_white_keys(white_mask, width, height, params)

        if white_boxes:
            white_boxes = sorted(white_boxes, key=lambda b: b[0])

        model = self._resolve_keyboard_model(force_keyboard_type, len(white_boxes))

        white_infos = self._build_white_key_info(white_boxes, params)
        bounding_box = bbox_union([info["bbox"] for info in white_infos])

        black_infos: List[Dict[str, object]] = []
        if white_infos:
            pattern = self._black_pattern_for_model(model, len(white_infos))
            black_infos = self._detect_black_keys(
                gray,
                white_infos,
                bounding_box,
                pattern,
                params,
            )

        confidence = self._compute_confidence(model, white_infos, black_infos)

        result: Dict[str, object] = {
            "keyboard_type": model.name if model else "unknown",
            "total_keys": model.total_keys if model else len(white_infos) + len(black_infos),
            "white_keys": white_infos,
            "black_keys": black_infos,
            "confidence": confidence,
            "bounding_box": bounding_box,
            "parameters": params.to_dict(),
        }

        if debug:
            result["debug"] = {
                "original": image_bgr,
                "gray": gray,
                "blurred": blurred,
                "edges": edges,
                "binary": white_mask,
                "column_projection": self._column_projection(white_mask),
            }

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _merge_params(self, overrides: Dict[str, object]) -> DetectionParameters:
        params_dict = self.params.to_dict()
        params_dict.update(overrides)
        return DetectionParameters(**params_dict)

    def _column_projection(self, white_mask: np.ndarray) -> np.ndarray:
        projection = white_mask.sum(axis=0).astype(np.float32)
        return projection

    def _locate_white_keys(
        self,
        white_mask: np.ndarray,
        width: int,
        height: int,
        params: DetectionParameters,
    ) -> List[tuple]:
        contour_boxes = self._locate_white_keys_contours(white_mask, width, height, params)
        projection_boxes = self._locate_white_keys_projection(white_mask, width, height, params)

        if contour_boxes and projection_boxes:
            if len(contour_boxes) >= len(projection_boxes):
                return contour_boxes
            return projection_boxes
        if contour_boxes:
            return contour_boxes
        return projection_boxes

    def _locate_white_keys_projection(
        self,
        white_mask: np.ndarray,
        width: int,
        height: int,
        params: DetectionParameters,
    ) -> List[tuple]:
        column_projection = white_mask.sum(axis=0).astype(np.float32)
        kernel = int(max(3, params.projection_smooth_size))
        if kernel % 2 == 0:
            kernel += 1
        smooth_kernel = np.ones(kernel, dtype=np.float32) / float(kernel)
        smooth_projection = np.convolve(column_projection, smooth_kernel, mode="same")
        max_val = smooth_projection.max() if smooth_projection.size else 0.0
        if max_val <= 0:
            return []
        threshold = max_val * float(params.projection_peak_ratio)

        min_width = max(int(params.min_white_key_width_ratio * width), 2)
        max_width = max(int(params.max_white_key_width_ratio * width), min_width)

        regions: List[tuple] = []
        in_region = False
        start = 0
        for idx, value in enumerate(smooth_projection):
            if value >= threshold:
                if not in_region:
                    in_region = True
                    start = idx
            else:
                if in_region:
                    end = idx
                    width_region = end - start
                    if width_region >= min_width:
                        regions.append((start, end))
                    in_region = False
        if in_region:
            end = len(smooth_projection) - 1
            if end - start >= min_width:
                regions.append((start, end))

        boxes: List[tuple] = []
        if not regions:
            return boxes

        refined_regions: List[tuple] = []
        for start, end in regions:
            width_region = end - start
            if width_region < min_width:
                continue
            if width_region > max_width:
                # Split wide regions to avoid merged keys
                pieces = int(round(width_region / float(max_width)))
                if pieces < 1:
                    pieces = 1
                step = width_region / pieces
                for i in range(pieces):
                    seg_start = int(start + i * step)
                    seg_end = int(min(end, start + (i + 1) * step))
                    if seg_end - seg_start >= min_width:
                        refined_regions.append((seg_start, seg_end))
            else:
                refined_regions.append((start, end))

        if not refined_regions:
            return boxes

        # Derive vertical bounds using row projection
        row_projection = white_mask.sum(axis=1).astype(np.float32)
        row_max = row_projection.max() if row_projection.size else 0
        if row_max > 0:
            row_threshold = row_max * 0.05
            rows = np.where(row_projection >= row_threshold)[0]
            if rows.size:
                row_top = int(rows.min())
                row_bottom = int(rows.max())
            else:
                row_top = 0
                row_bottom = height - 1
        else:
            row_top = 0
            row_bottom = height - 1

        margin = int(params.white_vertical_margin_ratio * height)
        row_top = max(0, row_top - margin)
        row_bottom = min(height - 1, row_bottom + margin)
        key_height = max(1, row_bottom - row_top + 1)

        for start, end in refined_regions:
            x0 = max(0, int(start))
            x1 = min(width - 1, int(end))
            if x1 <= x0:
                continue
            roi = white_mask[row_top:row_top + key_height, x0:x1]
            ys, xs = np.where(roi > 0)
            if ys.size:
                top = int(row_top + ys.min())
                bottom = int(row_top + ys.max())
            else:
                top = row_top
                bottom = row_top + key_height - 1
            y = max(0, top - margin)
            h = min(height - y, (bottom + margin) - y + 1)
            w = x1 - x0
            if w <= 0 or h <= 0:
                continue
            aspect = h / float(w)
            if aspect < params.min_white_key_aspect:
                continue
            area_ratio = (w * h) / float(width * height)
            if area_ratio < params.min_white_key_area_ratio:
                continue
            boxes.append((x0, y, w, h))

        if not boxes:
            return boxes

        return self._filter_white_key_width_outliers(boxes, params)

    def _locate_white_keys_contours(
        self,
        white_mask: np.ndarray,
        width: int,
        height: int,
        params: DetectionParameters,
    ) -> List[tuple]:
        min_width = max(int(params.min_white_key_width_ratio * width), 2)
        vertical_kernel_height = max(int(height * 0.25), 5)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_height))
        processed = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, vertical_kernel, iterations=1)
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[tuple] = []
        margin = int(params.white_vertical_margin_ratio * height)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < min_width:
                continue
            if h < height * 0.3:
                continue
            aspect = h / float(max(w, 1))
            if aspect < params.min_white_key_aspect:
                continue
            area_ratio = (w * h) / float(width * height)
            if area_ratio < params.min_white_key_area_ratio:
                continue
            y = max(0, y - margin)
            h = min(height - y, h + 2 * margin)
            boxes.append((x, y, w, h))

        if not boxes:
            return []

        boxes = sorted(boxes, key=lambda b: b[0])
        return self._filter_white_key_width_outliers(boxes, params)

    def _filter_white_key_width_outliers(
        self,
        boxes: Sequence[tuple],
        params: DetectionParameters,
    ) -> List[tuple]:
        if not boxes:
            return []
        widths = np.array([box[2] for box in boxes], dtype=np.float32)
        if widths.size == 0:
            return list(boxes)
        median_width = float(np.median(widths))
        lower = median_width * params.white_key_width_lower_factor
        upper = median_width * params.white_key_width_upper_factor
        filtered = [box for box in boxes if lower <= box[2] <= upper]
        if filtered:
            return filtered
        return list(boxes)

    def _resolve_keyboard_model(
        self,
        force_keyboard_type: Optional[str],
        detected_white_keys: int,
    ) -> Optional[keyboard_models.KeyboardModel]:
        if force_keyboard_type:
            key = str(force_keyboard_type).lower().replace("_key", "")
            if key in {"auto", "automatic", "none", ""}:
                forced = None
            else:
                forced = keyboard_models.get_model(key) or keyboard_models.get_model(f"{key}_key")
            if forced:
                return forced
        if detected_white_keys <= 0:
            return None
        return keyboard_models.find_closest_model(detected_white_keys)

    def _build_white_key_info(
        self,
        white_boxes: Sequence[tuple],
        params: DetectionParameters,
    ) -> List[Dict[str, object]]:
        if not white_boxes:
            return []
        white_boxes = sorted(white_boxes, key=lambda b: b[0])
        widths = np.array([box[2] for box in white_boxes], dtype=np.float32)
        median_width = float(np.median(widths)) if widths.size else 0.0
        boxes = []
        for index, box in enumerate(white_boxes):
            x, y, w, h = map(int, box)
            cx = x + w / 2.0
            cy = y + h / 2.0
            boxes.append(
                {
                    "index": index,
                    "bbox": (x, y, w, h),
                    "center": (cx, cy),
                    "normalized_center": (0.0, 0.0),  # Placeholder, filled later
                    "confidence": 1.0 if median_width == 0 else clamp(1.0 - abs(w - median_width) / median_width, 0.0, 1.0),
                }
            )
        bounding_box = bbox_union([b["bbox"] for b in boxes])
        for entry in boxes:
            cx, cy = entry["center"]
            entry["normalized_center"] = normalize_point(cx, cy, bounding_box)
        return boxes

    def _black_pattern_for_model(
        self,
        model: Optional[keyboard_models.KeyboardModel],
        detected_white_keys: int,
    ) -> List[bool]:
        if model:
            pattern = model.black_key_pattern
        else:
            pattern = keyboard_models.generate_black_key_pattern(detected_white_keys)
        if detected_white_keys <= 1:
            return []
        if len(pattern) >= detected_white_keys - 1:
            return pattern[: detected_white_keys - 1]
        # Pad if pattern shorter
        extras = keyboard_models.generate_black_key_pattern(detected_white_keys)
        return extras[: detected_white_keys - 1]

    def _detect_black_keys(
        self,
        gray: np.ndarray,
        white_infos: Sequence[Dict[str, object]],
        bounding_box: tuple,
        black_pattern: Sequence[bool],
        params: DetectionParameters,
    ) -> List[Dict[str, object]]:
        if not white_infos or not black_pattern:
            return []
        height, width = gray.shape[:2]
        bbx, bby, bbw, bbh = bounding_box
        if bbw <= 0 or bbh <= 0:
            bbx, bby, bbw, bbh = 0, 0, width, height
        median_white_width = np.median([info["bbox"][2] for info in white_infos]) if white_infos else 0
        black_infos: List[Dict[str, object]] = []
        for idx in range(min(len(white_infos) - 1, len(black_pattern))):
            if not black_pattern[idx]:
                continue
            left = white_infos[idx]
            right = white_infos[idx + 1]
            left_bbox = left["bbox"]
            right_bbox = right["bbox"]
            left_center = left["center"][0]
            right_center = right["center"][0]
            center_x = (left_center + right_center) / 2.0
            gap = max(right_center - left_center, 1.0)
            key_width = gap * float(params.black_key_relative_width)
            if median_white_width > 0:
                key_width = min(key_width, median_white_width * float(params.black_key_relative_width))
            half_width = max(key_width / 2.0, 1.0)
            x_start = int(max(0, center_x - half_width))
            x_end = int(min(width - 1, center_x + half_width))
            if x_end <= x_start:
                continue
            y_start = int(max(0, bby + params.black_key_vertical_offset * bbh))
            y_end = int(min(height - 1, bby + bbh * params.black_key_height_ratio))
            if y_end <= y_start:
                continue
            roi = gray[y_start:y_end, x_start:x_end]
            if roi.size == 0:
                continue
            darkness = 1.0 - float(np.mean(roi)) / 255.0
            if darkness < params.black_key_darkness_threshold:
                continue
            _, roi_binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(roi_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(contour)
                x += x_start
                y += y_start
            else:
                x = x_start
                y = y_start
                w = x_end - x_start
                h = y_end - y_start
            cx = x + w / 2.0
            cy = y + h / 2.0
            confidence = clamp(darkness, 0.0, 1.0)
            black_infos.append(
                {
                    "index": len(black_infos),
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "center": (cx, cy),
                    "normalized_center": normalize_point(cx, cy, bounding_box),
                    "confidence": confidence,
                    "source_gap": (left["index"], right["index"]),
                }
            )
        return black_infos

    def _compute_confidence(
        self,
        model: Optional[keyboard_models.KeyboardModel],
        white_infos: Sequence[Dict[str, object]],
        black_infos: Sequence[Dict[str, object]],
    ) -> float:
        if model:
            expected_white = max(model.white_keys, 1)
            expected_black = max(model.black_keys, 1)
            white_ratio = clamp(len(white_infos) / expected_white, 0.0, 1.0)
            black_ratio = clamp(len(black_infos) / expected_black, 0.0, 1.0) if model.black_keys else 1.0
            diff = abs(len(white_infos) - model.white_keys) / expected_white
            keyboard_conf = clamp(1.0 - diff, 0.0, 1.0)
            return round((white_ratio * 0.4 + black_ratio * 0.4 + keyboard_conf * 0.2), 4)
        if not white_infos:
            return 0.0
        return round(clamp(0.5 + 0.5 * min(1.0, len(black_infos) / max(len(white_infos) - 1, 1))), 4)
