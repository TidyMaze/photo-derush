"""Minimal face detection using MediaPipe.

Function: detect_faces(image)
  * image: numpy ndarray (BGR, as returned by cv2.imread / standard OpenCV pipeline)
  * returns: list of dicts [{"x": int, "y": int, "w": int, "h": int, "score": float}, ...]
  * returns [] on any error / no faces.

Design goals:
  - KISS / SRP: a single pure function (no I/O, no global state beyond lazy model creation).
  - Safe to call even if mediapipe isn't installed (gracefully returns []).
  - No side effects (no file writes, prints); uses logging at DEBUG level only.
"""
from __future__ import annotations

from typing import List, Dict
import logging

try:  # Optional dependency
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

_logger = logging.getLogger(__name__)

# Lazy holder for MediaPipe detector
_FACE_DETECTOR = None


def _load_detector():
    """Lazy-load MediaPipe face detector (may return None)."""
    global _FACE_DETECTOR  # noqa: PLW0603
    if _FACE_DETECTOR is not None:
        return _FACE_DETECTOR
    try:  # Import mediapipe lazily so project doesn't hard-depend at import time
        import mediapipe as mp  # type: ignore
        _FACE_DETECTOR = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5,
        )
    except Exception as e:  # pragma: no cover - environment without mediapipe
        _logger.debug("[FaceDetect] Mediapipe unavailable: %s", e)
        _FACE_DETECTOR = None
    return _FACE_DETECTOR


def _mediapipe_detect(image) -> List[Dict[str, float]]:
    det = _load_detector()
    if det is None:
        return []
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = det.process(rgb)
        if not results or not getattr(results, "detections", None):
            return []
        h, w = image.shape[:2]
        faces: List[Dict[str, float]] = []
        for detection in results.detections:  # type: ignore[attr-defined]
            try:
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
                try:
                    if hasattr(detection, "score"):
                        score_vals = [float(s) for s in detection.score if s is not None]
                except Exception:  # pragma: no cover
                    pass
                score = max(score_vals) if score_vals else 0.0
                if score < 0.0:
                    score = 0.0
                if score > 1.0:
                    score = 1.0
                faces.append({"x": x, "y": y, "w": bw, "h": bh, "score": score})
            except Exception as e:  # pragma: no cover
                _logger.debug("[FaceDetect] Skipping malformed mediapipe detection: %s", e)
        return faces
    except Exception as e:  # pragma: no cover
        _logger.debug("[FaceDetect] Mediapipe detection failed: %s", e)
        return []


def detect_faces(image) -> List[Dict[str, float]]:
    """Detect faces in a BGR OpenCV image using MediaPipe only.

    Returns an empty list on any error or if no faces are detected.
    """
    if cv2 is None or image is None:
        return []
    try:
        if len(image.shape) != 3 or image.shape[2] != 3:
            return []
    except Exception:
        return []
    return _mediapipe_detect(image)
