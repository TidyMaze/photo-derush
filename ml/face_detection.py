"""Minimal face detection using MediaPipe.

Function: detect_faces(image)
  * image: numpy ndarray (BGR, as returned by cv2.imread / standard OpenCV pipeline)
  * returns: list of dicts [{"x": int, "y": int, "w": int, "h": int, "score": float}, ...]

This version is strict: missing dependencies or invalid images raise errors instead of
silently returning empty results. No internal fallback logic.
"""
from __future__ import annotations

from typing import List, Dict
import logging
import cv2  # type: ignore
import mediapipe as mp  # type: ignore

_logger = logging.getLogger(__name__)

_FACE_DETECTOR = None


def _load_detector(min_conf: float, model_selection: int):
    global _FACE_DETECTOR  # noqa: PLW0603
    if _FACE_DETECTOR is not None:
        return _FACE_DETECTOR
    _FACE_DETECTOR = mp.solutions.face_detection.FaceDetection(
        model_selection=model_selection,
        min_detection_confidence=min_conf,
    )
    return _FACE_DETECTOR


def _mediapipe_detect(image, min_conf: float, model_selection: int) -> List[Dict[str, float]]:
    det = _load_detector(min_conf=min_conf, model_selection=model_selection)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = det.process(rgb)
    if not results or not getattr(results, "detections", None):
        return []
    h, w = image.shape[:2]
    faces: List[Dict[str, float]] = []
    for detection in results.detections:  # type: ignore[attr-defined]
        loc = detection.location_data.relative_bounding_box  # type: ignore[attr-defined]
        xmin = max(0.0, loc.xmin)
        ymin = max(0.0, loc.ymin)
        box_w = max(0.0, loc.width)
        box_h = max(0.0, loc.height)
        x = int(xmin * w)
        y = int(ymin * h)
        bw = int(box_w * w)
        bh = int(box_h * h)
        score_vals = []
        if hasattr(detection, "score"):
            score_vals = [float(s) for s in detection.score if s is not None]
        score = max(score_vals) if score_vals else 0.0
        score = 0.0 if score < 0.0 else (1.0 if score > 1.0 else score)
        faces.append({"x": x, "y": y, "w": bw, "h": bh, "score": score})
    return faces


def detect_faces(image, min_conf: float = 0.5, model_selection: int = 0) -> List[Dict[str, float]]:
    """Detect faces strictly.

    Raises:
      ValueError: if image is None or not a BGR uint8 3-channel array.
      ImportError: if required libs missing (raised at import time).
    """
    if image is None:
        raise ValueError("image is None")
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f"expected BGR 3-channel image, got shape={image.shape}")
    return _mediapipe_detect(image, min_conf=min_conf, model_selection=model_selection)
