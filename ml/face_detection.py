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
_DETECTOR_CFG = {"model_selection": None, "min_conf": None}


def _load_detector(min_conf: float, model_selection: int):
    """Lazy-load MediaPipe face detector for given config (may return None)."""
    global _FACE_DETECTOR, _DETECTOR_CFG  # noqa: PLW0603
    if (
        _FACE_DETECTOR is not None
        and _DETECTOR_CFG["model_selection"] == model_selection
        and _DETECTOR_CFG["min_conf"] == min_conf
    ):
        return _FACE_DETECTOR
    try:  # Import mediapipe lazily so project doesn't hard-depend at import time
        import mediapipe as mp  # type: ignore
        _FACE_DETECTOR = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_conf,
        )
        _DETECTOR_CFG = {"model_selection": model_selection, "min_conf": min_conf}
    except Exception as e:  # pragma: no cover - environment without mediapipe
        _logger.debug("[FaceDetect] Mediapipe unavailable: %s", e)
        _FACE_DETECTOR = None
        _DETECTOR_CFG = {"model_selection": None, "min_conf": None}
    return _FACE_DETECTOR


def _mediapipe_detect(image, min_conf: float, model_selection: int) -> List[Dict[str, float]]:
    det = _load_detector(min_conf=min_conf, model_selection=model_selection)
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


def detect_faces(image, min_conf: float = 0.5, model_selection: int = 0) -> List[Dict[str, float]]:
    """Detect faces in a BGR OpenCV image using MediaPipe only.

    Parameters:
      image: np.ndarray (BGR)
      min_conf: minimum detection confidence (default 0.5)
      model_selection: 0 = short-range, 1 = full-range (per mediapipe docs)
    Returns list of face dicts or [] on error / none.
    """
    if cv2 is None or image is None:
        return []
    try:
        if len(image.shape) != 3 or image.shape[2] != 3:
            return []
    except Exception:
        return []
    return _mediapipe_detect(image, min_conf=min_conf, model_selection=model_selection)
