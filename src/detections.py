"""Canonical detection types and normalization helpers.

Provides a small Detection dataclass and `normalize_detections()` which
converts a variety of backend outputs (dicts with tensors, lists/tuples,
simple name/score tuples) into a stable list of Detection objects for UI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np


@dataclass
class Detection:
    name: str
    confidence: float = 0.0
    bbox: Optional[List[float]] = None  # [x1,y1,x2,y2]
    det_w: Optional[int] = None
    det_h: Optional[int] = None
    raw: Optional[Any] = None


def _to_list(x):
    """Convert various array-like types to Python list. Returns None if conversion fails."""
    if x is None:
        return None

    # Try torch tensor with .cpu() method
    if hasattr(x, "cpu") and callable(getattr(x, "cpu")):
        try:
            return x.cpu().numpy().tolist()
        except Exception:
            pass  # Fall through to next attempt

    # Try numpy conversion
    try:
        return np.asarray(x).tolist()
    except Exception:
        pass  # Fall through to last attempt

    # Last resort: direct list coercion
    try:
        return list(x)
    except Exception:
        return None


def normalize_detections(objects) -> List[Detection]:
    """Normalize various detection representations into List[Detection].

    Cases handled:
    - None -> []
    - list of dicts with keys like 'bbox'/'boxes', 'confidence'/'score'/'scores', 'class'/'name'
    - list of tuples/lists like (name, score)
    - list of strings -> name with zero score
    """
    if not objects:
        return []

    dets: List[Detection] = []

    # If a single dict was passed (maybe worker returned dict), try to extract its 'detections'
    if isinstance(objects, dict) and "detections" in objects:
        objects = objects.get("detections") or []

    if not isinstance(objects, (list, tuple)):
        # single object
        objects = [objects]

    for o in objects:
        if o is None:
            continue
        # Rich dict with bbox/labels/scores
        if isinstance(o, dict):
            # bbox may be under 'bbox' or under 'boxes' (tensor/array)
            bbox = None
            if "bbox" in o and o.get("bbox") is not None:
                bbox_val = o.get("bbox")
                bbox = _to_list(bbox_val)
            elif "boxes" in o and o.get("boxes") is not None:
                # boxes could be array/tensor of shape (N,4); take first row if present
                boxes_val = o.get("boxes")
                bl = _to_list(getattr(boxes_val, "xyxy", boxes_val) if hasattr(boxes_val, "xyxy") else boxes_val)
                # bl might be nested; if so pick first
                if bl:
                    try:
                        if isinstance(bl[0], (list, tuple)):
                            bbox = list(map(float, bl[0]))
                        else:
                            bbox = list(map(float, bl))
                    except (ValueError, TypeError, IndexError):
                        bbox = None

            # name/class
            name = None
            for k in ("name", "class", "label", "class_name", "label_name"):
                if k in o and o.get(k) is not None:
                    name = str(o.get(k))
                    break

            # confidence/score
            conf = None
            for k in ("confidence", "score", "prob", "scores"):
                if k in o and o.get(k) is not None:
                    v = o.get(k)
                    # If it's an array, pick first element
                    try:
                        if isinstance(v, (list, tuple)):
                            conf = float(v[0])
                        else:
                            conf = float(v)
                    except (ValueError, TypeError, IndexError):
                        conf = None
                    break

            det_w = int(o.get("det_w")) if o.get("det_w") is not None else None
            det_h = int(o.get("det_h")) if o.get("det_h") is not None else None

            det = Detection(name=(name or ""), confidence=(conf or 0.0), bbox=bbox, det_w=det_w, det_h=det_h, raw=o)
            dets.append(det)
            continue

        # Simple tuple/list-like -> (name, score)
        if isinstance(o, (list, tuple)):
            if len(o) >= 2:
                try:
                    name = str(o[0])
                    conf = float(o[1])
                except Exception:
                    name = str(o[0])
                    conf = 0.0
            elif len(o) == 1:
                name = str(o[0])
                conf = 0.0
            else:
                continue
            dets.append(Detection(name=name, confidence=conf, raw=o))
            continue

        # Plain string
        if isinstance(o, str):
            dets.append(Detection(name=o, confidence=0.0, raw=o))
            continue

        # Fallback: attempt to stringify
        try:
            dets.append(Detection(name=str(o), confidence=0.0, raw=o))
        except Exception:
            logging.exception("detections.normalize_detections: failed stringify fallback for object")
            continue

    return dets
