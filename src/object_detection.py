import logging
import os
import platform
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
from PIL import Image

from src.constants import DETECTION_BACKEND, MIN_CONFIDENCE_KEEP_FRCNN, MIN_CONFIDENCE_KEEP_YOLO, YOLO_MODEL_NAME
from src.detection_config import DetectionConfig


class _CocoClassesDict(dict):
    """Custom dict for backward compat: accessing COCO_CLASSES[0] returns name, not tuple."""

    def __getitem__(self, key):
        val = super().__getitem__(key)
        # If accessing by int key (class_id), return just the name for backward compat
        if isinstance(key, int) and isinstance(val, tuple):
            return val[0]
        return val

    def __contains__(self, key):
        # Support both int keys (class_ids) and string keys (class names)
        if super().__contains__(key):
            return True
        # Check if key is a class name in any tuple value
        if isinstance(key, str):
            for val in self.values():
                if isinstance(val, tuple) and val[0] == key:
                    return True
        return False


# Structure: {class_id: (name, is_interesting)}
COCO_CLASSES = _CocoClassesDict(
    {
        0: ("__background__", False),
        1: ("person", True),  # people/faces
        2: ("bicycle", True),
        3: ("car", True),
        4: ("motorcycle", True),
        5: ("airplane", True),
        6: ("bus", True),
        7: ("train", True),
        8: ("truck", True),
        9: ("boat", True),
        10: ("traffic light", False),
        11: ("fire hydrant", False),
        13: ("stop sign", False),
        14: ("parking meter", False),
        15: ("bench", False),
        16: ("bird", True),
        17: ("cat", True),
        18: ("dog", True),
        19: ("horse", True),
        20: ("sheep", True),
        21: ("cow", True),
        22: ("elephant", True),
        23: ("bear", True),
        24: ("zebra", True),
        25: ("giraffe", True),
        27: ("backpack", False),
        28: ("umbrella", False),
        31: ("handbag", False),
        32: ("tie", False),
        33: ("suitcase", False),
        34: ("frisbee", False),
        35: ("skis", False),
        36: ("snowboard", False),
        37: ("sports ball", False),
        38: ("kite", False),
        39: ("baseball bat", False),
        40: ("baseball glove", False),
        41: ("skateboard", False),
        42: ("surfboard", False),
        43: ("tennis racket", False),
        44: ("bottle", True),
        46: ("wine glass", True),
        47: ("cup", True),
        48: ("fork", False),
        49: ("knife", True),
        50: ("spoon", True),
        51: ("bowl", True),
        52: ("banana", True),
        53: ("apple", True),
        54: ("sandwich", False),
        55: ("orange", True),
        56: ("broccoli", True),
        57: ("carrot", True),
        58: ("hot dog", False),
        59: ("pizza", False),
        60: ("pizza", True),
        61: ("donut", False),
        62: ("cake", True),
        63: ("chair", False),
        64: ("potted plant", True),
        65: ("bed", False),
        67: ("dining table", True),
        70: ("toilet", False),
        72: ("tv", False),
        73: ("book", True),
        74: ("clock", False),
        75: ("vase", False),
        76: ("scissors", False),
        77: ("cell phone", True),
        78: ("microwave", False),
        79: ("oven", False),
        80: ("toaster", False),
        81: ("sink", True),
        82: ("refrigerator", True),
        84: ("book", False),
        85: ("clock", False),
        86: ("vase", False),
        87: ("scissors", False),
        88: ("teddy bear", False),
        89: ("hair drier", False),
        90: ("toothbrush", False),
    }
)

# Derived: only classes marked as interesting for photo management
INTERESTING_CLASSES = {k: v[0] for k, v in COCO_CLASSES.items() if v[1]}

# Legacy list format for backward compatibility
# Note: This list doesn't perfectly match COCO_CLASSES dict (has extra entries like "couch", "laptop")
# Keeping as-is for backward compatibility with existing code that indexes by position
COCO_CLASSES_LIST = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


class DetectionContext:
    """Encapsulates global state for object detection operations.

    Manages model caching, device mapping, and thread synchronization
    to ensure safe concurrent detection calls.
    """

    def __init__(self):
        # Cache models by (backend, device) to ensure each backend/device
        # combination is only loaded once per process. This prevents repeated
        # reloads when multiple callers request detection with different
        # device hints and makes the logging of model loads one-per-process
        # per backend/device.
        self.model_cache: Dict[Tuple[str, str], Any] = {}
        self.weights_cache: Dict[Tuple[str, str], Any] = {}
        self.device_map: Dict[str, str] = {}  # maps backend->device actually used
        self.model_name_map: Dict[str, str] = {}  # maps backend->model name actually used

        # Serialize native detector calls to avoid concurrent invocations of
        # libtorch/ultralytics which may interact poorly with OpenMP thread pools
        # when called from multiple application threads (observed as EXC_BAD_ACCESS).
        # Use RLock to allow re-entrant acquisition within the same thread.
        self.detection_lock = threading.RLock()

        # A lightweight semaphore to limit the number of concurrent callers that may
        # attempt to initialize or call the heavy detection backend. If the app is
        # under heavy thumbnail-refresh load, callers that cannot acquire this
        # semaphore quickly will fall back to cached results to avoid triggering
        # repeated model loads which have caused crashes on some platforms.
        self.detection_semaphore = threading.Semaphore(1)



# Global singleton instance
_detection_ctx = DetectionContext()
# Track if MPS warning has been logged to avoid spam
_mps_warning_logged = False


def _to_numpy(x, name: str):
    """Convert torch tensor / numpy-like / python list to numpy array with basic checks.

    Returns a numpy array. Raises RuntimeError on unsupported types.
    """
    import numpy as _np

    # None => empty
    if x is None:
        if name == "boxes":
            return _np.zeros((0, 4))
        return _np.zeros((0,))
    # Torch tensor
    if hasattr(x, "cpu"):
        try:
            return x.cpu().numpy()
        except Exception as exc:
            raise RuntimeError(f"Failed to convert torch tensor '{name}' to numpy: {exc}") from exc
    # Numpy-like or list
    try:
        return _np.asarray(x)
    except Exception as exc:
        raise RuntimeError(f"Unsupported {name} type: {type(x)}; conversion failed: {exc}") from exc


def _normalize_model_predictions(predictions):
    """Normalize model-specific prediction objects into (boxes, labels, scores) numpy arrays.

    Accepts common backends (torchvision dicts with tensors, ultralytics Results-wrappers,
    or plain python dicts/lists returned by a worker). Returns (boxes_np, labels_np, scores_np).
    Raises RuntimeError on unexpected formats.
    """

    if not (predictions and isinstance(predictions, (list, tuple)) and len(predictions) > 0):
        raise RuntimeError(f"Unexpected prediction format: {type(predictions)}")

    raw0 = predictions[0]
    # If backend returned a mapping-like object with 'boxes' etc.
    if isinstance(raw0, dict):
        boxes = raw0.get("boxes")
        labels = raw0.get("labels")
        scores = raw0.get("scores")
        boxes = _to_numpy(boxes, "boxes")
        labels = _to_numpy(labels, "labels")
        scores = _to_numpy(scores, "scores")
        return boxes, labels, scores

    # Some backends might return objects (ultralytics Results) directly in lists;
    # callers wrapping this into dicts should be preferred. Fail clearly otherwise.
    raise RuntimeError(f"Unsupported raw prediction element type: {type(raw0)}")


# Backend selection: force to centralized constant
DETECTION_BACKEND = DETECTION_BACKEND
DETECT_VERBOSE = os.environ.get("DETECT_VERBOSE", "0") in ("1", "true", "True")


class TorchvisionAdapter:
    def __init__(self, model, weights=None):
        self.model = model
        self.weights = weights

    def to(self, device: str):
        # Propagate errors when moving model to device so failures are visible
        self.model.to(device)
        return self

    def eval(self):
        # Allow exceptions to surface during eval() to fail fast in tests
        self.model.eval()
        return self

    def preprocess(self, image: Image.Image, device: str = "cpu", max_size: int = 800):
        # Use weights transforms if available, otherwise ToTensor
        if self.weights is not None and hasattr(self.weights, "transforms"):
            transform = self.weights.transforms()
            return transform(image).unsqueeze(0).to(device)
        else:
            from torchvision.transforms import ToTensor

            transform = ToTensor()
            return transform(image).unsqueeze(0).to(device)

    def predict(self, input_tensor, conf: float = 0.5):
        # torchvision models accept a tensor batch
        import torch

        with torch.no_grad():
            return self.model(input_tensor)


class YOLOv8Adapter:
    def __init__(self, model_path_or_name=YOLO_MODEL_NAME):
        # Try common import paths for ultralytics to satisfy different package versions
        try:
            from ultralytics import YOLO
        except Exception:
            try:
                from ultralytics.yolo import YOLO  # fallback path some distributions use
            except Exception:
                raise
        self._yolo = YOLO(model_path_or_name)

    def to(self, device: str):
        # ultralytics handles device internally; no-op
        return self

    def eval(self):
        return self

    def preprocess(self, image: Image.Image, device: str = "cpu", max_size: int = 800):
        # YOLO accepts PIL images; ensure image is resized to max_size already by caller
        return [image]

    def predict(self, imgs, conf: float = 0.001):
        # imgs is a list of PIL images (we only use first)
        pil_img = imgs[0]
        # Serialize native YOLO calls. If the lock is held by another
        # thread, log that we had to wait so this contention is visible.
        waited = False
        if not _detection_ctx.detection_lock.acquire(blocking=False):
            waited = True
            logging.warning("Waiting for detection lock in YOLOv8Adapter.predict (another detection in-flight).")
            _detection_ctx.detection_lock.acquire()
        try:
            if waited:
                logging.info("Acquired detection lock after waiting in YOLOv8Adapter.predict; proceeding.")
            results = self._yolo.predict(pil_img, imgsz=max(pil_img.size), conf=conf, verbose=False)
        finally:
            try:
                _detection_ctx.detection_lock.release()
            except Exception:
                logging.exception("Failed to release detection lock in YOLOv8Adapter.predict")
        # Convert ultralytics Results into torchvision-like list[dict] with tensors
        import torch as _torch
        
        if not results:
            return [
                {
                    "boxes": _torch.zeros((0, 4)),
                    "labels": _torch.zeros((0,), dtype=_torch.int64),
                    "scores": _torch.zeros((0,)),
                }
            ]
        r = results[0]
        boxes_attr = getattr(r, "boxes", None)
        if boxes_attr is None:
            return [
                {
                    "boxes": _torch.zeros((0, 4)),
                    "labels": _torch.zeros((0,), dtype=_torch.int64),
                    "scores": _torch.zeros((0,)),
                }
            ]

        # Helper to extract attribute and convert to numpy safely
        import numpy as _np

        def _to_numpy(attr):
            if attr is None:
                return _np.zeros((0,))
            # Many ultralytics builds return tensors with .cpu(), others return numpy arrays
            try:
                if hasattr(attr, "cpu"):
                    return attr.cpu().numpy()
            except Exception as exc:
                raise RuntimeError(f"Failed to convert ultralytics attribute via .cpu(): {exc}") from exc
            try:
                return _np.asarray(attr)
            except Exception as exc:
                raise RuntimeError(f"Failed to convert ultralytics attribute to numpy: {exc}") from exc

        xyxy = _to_numpy(getattr(boxes_attr, "xyxy", None))
        confs = _to_numpy(getattr(boxes_attr, "conf", None))
        cls = (
            _to_numpy(getattr(boxes_attr, "cls", None)).astype(int)
            if _to_numpy(getattr(boxes_attr, "cls", None)).size
            else _np.zeros((0,)).astype(int)
        )

        import numpy as _np
        import torch as _torch

        # shift labels to match COCO_CLASSES where background is 0
        labels = (cls.astype(int) + 1).astype(int) if len(cls) else _np.zeros((0,)).astype(int)

        boxes_t = _torch.from_numpy(_np.asarray(xyxy)) if xyxy.size else _torch.zeros((0, 4))
        labels_t = (
            _torch.from_numpy(_np.asarray(labels)).to(_torch.int64)
            if labels.size
            else _torch.zeros((0,), dtype=_torch.int64)
        )
        scores_t = _torch.from_numpy(_np.asarray(confs)) if confs.size else _torch.zeros((0,))

        return [{"boxes": boxes_t, "labels": labels_t, "scores": scores_t}]


def _load_model(device: str = "auto"):
    """Load the object detection model and cache it per (backend, device).

    Returns (model, weights). Multiple calls with the same backend/device
    in the same process will return the cached instance without re-loading.
    The function is protected by detection_lock to prevent concurrent
    initialization attempts.
    """

    # Ensure only one thread attempts to load the model at a time.
    _detection_ctx.detection_lock.acquire()
    try:
        # Import heavy ML libs lazily so importing this module doesn't
        # immediately trigger lengthy native initialization (torch/ultralytics).
        try:
            import torch  # type: ignore[import-untyped]
        except Exception:
            torch = None  # type: ignore[assignment]

        # Allow forcing device via environment variable for easier debugging
        env_device = os.environ.get("DETECTION_DEVICE")
        if env_device:
            device = env_device

        # Auto-detect best available device
        if device == "auto":
            if torch and torch.cuda.is_available():  # type: ignore[attr-defined]
                device = "cuda"
            elif torch and torch.backends.mps.is_available():  # type: ignore[attr-defined]
                device = "mps"
            else:
                device = "cpu"

        effective_backend = DETECTION_BACKEND

        # On macOS MPS, prefer CPU for known unstable backends
        global _mps_warning_logged
        if platform.system() == "Darwin" and device == "mps":
            if effective_backend == "yolov8":
                if not _mps_warning_logged:
                    logging.debug("YOLOv8 on MPS on macOS can be unstable; forcing device='cpu' for this backend")
                    _mps_warning_logged = True
                device = "cpu"

        key = (effective_backend, device)
        # Return cached model if present
        if key in _detection_ctx.model_cache:
            return _detection_ctx.model_cache[key], _detection_ctx.weights_cache.get(key)

        msg = f"[pid={os.getpid()}] Loading detection backend '{effective_backend}' on {device}"
        if DETECT_VERBOSE:
            logging.info(msg)
        else:
            logging.debug(msg)

        # Load the backend
        try:
            from ultralytics import YOLO

            class YOLOv8Wrapper:
                def __init__(self, model_path_or_name="yolov8n"):
                    self._yolo = YOLO(model_path_or_name)

                def eval(self):
                    return self

                def to(self, device):
                    return self

                def __call__(self, imgs, conf=0.001, **kwargs):
                    pil_img = imgs[0]
                    waited = False
                    if not _detection_ctx.detection_lock.acquire(blocking=False):
                        waited = True
                        logging.warning(
                            "Waiting for detection lock in YOLOv8Wrapper.__call__ (another detection in-flight)."
                        )
                        _detection_ctx.detection_lock.acquire()
                    try:
                        if waited:
                            logging.info("Acquired detection lock after waiting in YOLOv8Wrapper.__call__; proceeding.")
                        results = self._yolo.predict(pil_img, imgsz=max(pil_img.size), conf=conf, verbose=False)
                    finally:
                        try:
                            _detection_ctx.detection_lock.release()
                        except Exception:
                            logging.exception("Failed to release detection lock in YOLOv8Wrapper.__call__")

                    out_boxes = []
                    out_labels = []
                    out_scores = []
                    if results and len(results) > 0:
                        r = results[0]
                        boxes_attr = getattr(r, "boxes", None)
                        if boxes_attr is not None:
                            try:
                                xyxy = boxes_attr.xyxy.cpu().numpy()
                                confs = boxes_attr.conf.cpu().numpy()
                                cls = boxes_attr.cls.cpu().numpy().astype(int)
                            except Exception:
                                xyxy = boxes_attr.xyxy.numpy()
                                confs = boxes_attr.conf.numpy()
                                cls = boxes_attr.cls.numpy().astype(int)
                            out_boxes = xyxy
                            out_scores = confs
                            out_labels = cls

                    import torch as _torch

                    return [
                        {
                            "boxes": _torch.from_numpy(out_boxes) if len(out_boxes) else _torch.zeros((0, 4)),
                            "labels": _torch.from_numpy(out_labels).to(_torch.int64)
                            if len(out_labels)
                            else _torch.zeros((0,), dtype=_torch.int64),
                            "scores": _torch.from_numpy(out_scores) if len(out_scores) else _torch.zeros((0,)),
                        }
                    ]

            weights = None
            model = YOLOv8Wrapper(YOLO_MODEL_NAME)
        except Exception as e:
            raise RuntimeError("Requested DETECTION_BACKEND=yolov8 but ultralytics is not available") from e

        model.eval()
        try:
            model.to(device)
        except Exception:
            # Some wrappers (like YOLO wrapper) may ignore .to(); log and continue
            logging.exception("Model .to(device) failed or was ignored")

        # Cache and record actual device and model name used for this backend
        _detection_ctx.model_cache[key] = model
        _detection_ctx.weights_cache[key] = weights
        _detection_ctx.device_map[effective_backend] = device
        _detection_ctx.model_name_map[effective_backend] = YOLO_MODEL_NAME

        return model, weights
    finally:
        try:
            _detection_ctx.detection_lock.release()
        except Exception:
            logging.exception("Failed to release _detection_ctx.detection_lock in _load_model")


def detect_objects(
    image_path: str, config: DetectionConfig | None = None, confidence_threshold: float | None = None, **kwargs
) -> List[Dict]:
    """Detect objects in a single image.

    Args:
        image_path: Path to the image file
        config: Detection configuration (uses defaults if None)
        confidence_threshold: [DEPRECATED] Use config.min_confidence instead
        **kwargs: Other backward-compat params (device, max_size) to add to config

    Returns:
        List of dicts with keys: 'class', 'confidence', 'bbox'
    """
    if config is None:
        config = DetectionConfig()

    # Backward compat: apply old signature params to config
    if confidence_threshold is not None:
        config.confidence_threshold = confidence_threshold
    if "device" in kwargs:
        config.device = kwargs["device"]
    if "max_size" in kwargs:
        config.max_size = kwargs["max_size"]

    # In pytest runs, avoid initializing heavy native detection backends
    # (ultralytics/torch/OpenMP interactions can deadlock on some macOS setups).
    # Return no detections instead of loading a model to keep unit tests fast
    # and deterministic. If you need integration tests for detection, run
    # them in a separate integration test job without the pytest env var.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        logging.info("[detect] pytest detected: skipping heavy detection backend and returning no detections")
        return []

    # Load (or get cached) model for the requested device. _load_model will
    # return the cached model if already loaded for the backend/device.
    model, weights = _load_model(config.device)

    # If caller requested 'auto' we may now query which device the backend
    # was actually loaded on in this process.
    try:
        if config.device == "auto":
            loaded = _detection_ctx.device_map.get(DETECTION_BACKEND)
            if loaded:
                config.device = loaded
    except Exception:
        pass

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")

    # Resize image to ensure the largest side equals max_size while
    # preserving aspect ratio. This upsamples small images and downsamples
    # large images so overlays appear consistent across different source
    # resolutions.
    if config.max_size is not None and config.max_size > 0:
        current_max = max(image.size)
        if current_max != config.max_size:
            ratio = config.max_size / float(current_max)
            new_size = (max(1, int(image.size[0] * ratio)), max(1, int(image.size[1] * ratio)))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Record detection input size (after potential resize)
    det_w, det_h = image.size

    # Allow override of minimum area ratio via env var so testers/users can
    # lower the area threshold without editing source. If the env var is set
    # and parses to a float, it will override the explicit function parameter.
    try:
        env_min_area = os.environ.get("DETECTION_MIN_AREA_RATIO")
        if env_min_area is not None:
            config.min_area_ratio = float(env_min_area)
    except Exception:
        logging.warning(
            "Invalid DETECTION_MIN_AREA_RATIO env var; using provided min_area_ratio=%s", config.min_area_ratio
        )

    # Hard lower bound for keeping detections. Use a stricter default for
    # YOLOv8 because we prefer higher precision on returned boxes.
    if DETECTION_BACKEND == "yolov8":
        MIN_CONFIDENCE_KEEP = MIN_CONFIDENCE_KEEP_YOLO
    else:
        MIN_CONFIDENCE_KEEP = MIN_CONFIDENCE_KEEP_FRCNN
    keep_confidence = max(config.confidence_threshold, MIN_CONFIDENCE_KEEP)

    # Transform using weights' transforms if available
    # Use weights.transforms() when weights object provides transforms (torchvision)
    # For YOLO, pass PIL image directly
    image_tensor = [image]  # list of PIL

    # Try worker process first if enabled
    params = {
        "confidence_threshold": config.confidence_threshold,
        "device": config.device,
        "max_size": config.max_size,
        "classes_filter": config.classes_filter,
        "min_area_ratio": config.min_area_ratio,
    }
    # In-process detection
    if not _detection_ctx.detection_lock.acquire(blocking=False):
        logging.warning("Waiting for detection lock in detect_objects (another detection in-flight).")
        _detection_ctx.detection_lock.acquire()

    try:
        try:
            import torch  # type: ignore[import-untyped]
            with torch.no_grad():  # type: ignore[attr-defined]
                predictions = model(image_tensor, conf=keep_confidence)
        except ImportError:
            predictions = model(image_tensor, conf=keep_confidence)
    finally:
            try:
                _detection_ctx.detection_lock.release()
            except RuntimeError:
                pass

    # Normalize prediction objects into numpy arrays (boxes, labels, scores)
    try:
        boxes, labels, scores = _normalize_model_predictions(predictions)
    except Exception as e:
        raise RuntimeError(f"Unexpected prediction format for image {image_path}: {e}") from e

    # Normalize types: boxes as float, labels as int, scores as float
    try:
        boxes = boxes.astype(float)
    except Exception as exc:
        raise RuntimeError(f"Failed to cast boxes to float: {exc}") from exc
    try:
        labels = labels.astype(int)
    except Exception as exc:
        raise RuntimeError(f"Failed to cast labels to int: {exc}") from exc
    try:
        scores = scores.astype(float)
    except Exception as exc:
        raise RuntimeError(f"Failed to cast scores to float: {exc}") from exc

    # For YOLO, labels are 0-based COCO, but our COCO_CLASSES has background at 0, so shift
    if DETECTION_BACKEND == "yolov8" and labels.size:
        labels = labels + 1

    # Filter by (enforced) confidence threshold and optionally by classes.
    # Additionally discard very small boxes whose area is a negligible
    # fraction of the detection input size (prevents tiny false positives).
    detections = []
    img_area = float(det_w * det_h) if det_w and det_h else 0.0
    for box, label, score in zip(boxes, labels, scores):
        # Enforce the keep_confidence lower bound
        if float(score) < keep_confidence:
            continue

        # Skip if filtering classes and this class is not in the filter
        if config.classes_filter is not None and int(label) not in config.classes_filter:
            continue

        # Discard boxes that take a very small fraction of the image area
        try:
            x1, y1, x2, y2 = box
            w = max(0.0, float(x2) - float(x1))
            h = max(0.0, float(y2) - float(y1))
            area = w * h
            area_ratio = (area / img_area) if img_area > 0 else 0.0
        except Exception:
            area_ratio = 0.0

        if area_ratio < float(config.min_area_ratio):
            # tiny box; skip
            continue

        idx = int(label)
        class_name = COCO_CLASSES_LIST[idx] if idx < len(COCO_CLASSES_LIST) else f"class_{idx}"
        detections.append(
            {
                "class": class_name,
                "confidence": float(score),
                "bbox": box.tolist(),  # [x1, y1, x2, y2]
                "det_w": det_w,
                "det_h": det_h,
            }
        )

    return detections


def get_cache_path() -> str:
    """Get the path for object detection cache file."""
    return os.path.join(".cache", "object_detections.joblib")


def load_object_cache() -> Dict[str, List[Dict]]:
    """Load cached object detections."""
    cache_path = get_cache_path()
    if os.path.exists(cache_path):
        try:
            data = joblib.load(cache_path)
            # If the cache was produced by a different detection backend, invalidate it
            cached_backend = data.get("backend")
            if cached_backend is not None and cached_backend != DETECTION_BACKEND:
                logging.info(
                    f"Detection backend changed (cache backend={cached_backend} != current={DETECTION_BACKEND}), invalidating cache"
                )
                return {}
            return data.get("detections", {})  # type: ignore[return-value, no-any-return]
        except Exception as e:
            logging.warning(f"Failed to load object cache: {e}")
    return {}


def save_object_cache(detections: Dict[str, List[Dict]]):
    """Save object detections to cache."""
    cache_path = get_cache_path()
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        cache_data = {
            "detections": detections,
            "model": "object_detections",
            "backend": DETECTION_BACKEND,
            "classes": COCO_CLASSES_LIST[1:],  # All COCO classes except background
        }
        joblib.dump(cache_data, cache_path)
        logging.info(f"Saved object detections cache to {cache_path}")
    except Exception as e:
        logging.warning(f"Failed to save object cache: {e}")


# Module-level cache for object detection cache to avoid repeated loads
_object_cache_cache: Dict[str, List[Dict]] | None = None
_object_cache_mtime: float = 0

def get_objects_for_image(
    image_path: str, config: DetectionConfig | None = None, confidence_threshold: float | None = None, **kwargs
) -> List[tuple[str, float]]:
    """Get list of detected object classes with confidence scores for an image.

    Args:
        image_path: Path to the image file
        config: Detection configuration (uses defaults if None)
        confidence_threshold: [DEPRECATED] Use config.confidence_threshold instead
        **kwargs: Other backward-compat params

    Returns unique object class names with confidence scores, sorted by confidence (highest first).
    Each class appears only once, even if multiple instances are detected.
    """
    global _object_cache_cache, _object_cache_mtime
    
    if config is None:
        config = DetectionConfig()

    # Backward compat: apply old signature params to config
    if confidence_threshold is not None:
        config.confidence_threshold = confidence_threshold

    basename = os.path.basename(image_path)

    # Try cache first - use module-level cache if available and still valid
    cache_path = get_cache_path()
    cache_mtime = os.path.getmtime(cache_path) if os.path.exists(cache_path) else 0
    
    if _object_cache_cache is None or cache_mtime != _object_cache_mtime:
        cache = load_object_cache()
        _object_cache_cache = cache
        _object_cache_mtime = cache_mtime
    else:
        cache = _object_cache_cache
    
    if basename in cache:
        detections = cache[basename]
        # Check if cached detections match current parameters
        # For simplicity, we'll re-detect if parameters don't match cache
        # In production, you might want more sophisticated cache invalidation

        # Filter by confidence threshold
        filtered = [d for d in detections if d["confidence"] >= config.confidence_threshold]
        # Sort by confidence descending and get unique classes with their highest confidence
        filtered.sort(key=lambda x: x["confidence"], reverse=True)
        unique_classes = []
        seen_classes = set()
        for d in filtered:
            if d["class"] not in seen_classes:
                unique_classes.append((d["class"], d["confidence"]))
                seen_classes.add(d["class"])
        return unique_classes

    # Detect objects
    detections = detect_objects(image_path, config)

    # Update cache
    cache[basename] = detections
    save_object_cache(cache)
    # Update module-level cache to reflect new detection (don't invalidate - keep in memory)
    _object_cache_cache = cache
    _object_cache_mtime = os.path.getmtime(cache_path) if os.path.exists(cache_path) else 0

    # Return unique class names with confidence scores, sorted by confidence
    detections.sort(key=lambda x: x["confidence"], reverse=True)
    unique_classes = []
    seen_classes = set()
    for d in detections:
        if d["class"] not in seen_classes:
            unique_classes.append((d["class"], d["confidence"]))
            seen_classes.add(d["class"])
    return unique_classes


def get_objects_for_images(
    image_paths: List[str], config: DetectionConfig | None = None, progress_callback=None
) -> Dict[str, List[Tuple[str, float]]]:
    """Get detected objects with confidence scores for multiple images.

    Args:
        image_paths: List of image file paths
        config: Detection configuration (uses defaults if None)
        progress_callback: Optional callback for progress updates

    Returns dict mapping basename to list of unique object classes with confidence scores.
    Each class appears only once per image, even if multiple instances are detected.
    """
    global _object_cache_cache, _object_cache_mtime
    
    if config is None:
        config = DetectionConfig()

    results = {}

    # Load existing cache - use module-level cache if available and still valid
    cache_path = get_cache_path()
    cache_mtime = os.path.getmtime(cache_path) if os.path.exists(cache_path) else 0
    
    if _object_cache_cache is None or cache_mtime != _object_cache_mtime:
        cache = load_object_cache()
        _object_cache_cache = cache
        _object_cache_mtime = cache_mtime
    else:
        cache = _object_cache_cache

    # Check cache hits
    to_process = []
    for path in image_paths:
        basename = os.path.basename(path)
        if basename in cache:
            detections = cache[basename]
            filtered = [d for d in detections if d["confidence"] >= config.confidence_threshold]
            filtered.sort(key=lambda x: x["confidence"], reverse=True)
            unique_classes = []
            seen_classes = set()
            for d in filtered:
                if d["class"] not in seen_classes:
                    unique_classes.append((d["class"], d["confidence"]))
                    seen_classes.add(d["class"])
            results[basename] = unique_classes
        else:
            to_process.append(path)

    # Process missing images
    if to_process:
        logging.info(f"Detecting objects in {len(to_process)} images")

        for i, path in enumerate(to_process):
            if progress_callback:
                progress_callback(i, len(to_process), f"Detecting objects in {os.path.basename(path)}")

            basename = os.path.basename(path)
            detections = detect_objects(path, config)
            cache[basename] = detections

            # Save cache immediately after each detection to prevent data loss
            try:
                save_object_cache(cache)
            except Exception as e:
                logging.warning(f"Failed to save cache after detection for {basename}: {e}")

            # Filter and sort, then get unique classes with confidence
            detections.sort(key=lambda x: x["confidence"], reverse=True)
            unique_classes = []
            seen_classes = set()
            for d in detections:
                if d["class"] not in seen_classes:
                    unique_classes.append((d["class"], d["confidence"]))
                    seen_classes.add(d["class"])
            results[basename] = unique_classes
        # Update module-level cache to reflect new detections (don't invalidate - keep in memory)
        cache_path = get_cache_path()
        _object_cache_cache = cache
        _object_cache_mtime = os.path.getmtime(cache_path) if os.path.exists(cache_path) else 0

    return results


def get_available_classes() -> List[str]:
    """Get list of all detectable object classes."""
    return COCO_CLASSES_LIST[1:]  # Skip '__background__'


__all__ = [
    "detect_objects",
    "get_objects_for_image",
    "get_objects_for_images",
    "get_available_classes",
    "load_object_cache",
    "save_object_cache",
    "get_cache_path",
    "get_loaded_device",
    "get_loaded_model_name",
]


def _detect_objects_compat(
    image_path: str,
    confidence_threshold: float = 0.6,
    device: str = "auto",
    max_size: int = 800,
    classes_filter: Optional[Set[int]] = None,
    min_area_ratio: float = 0.0,
) -> List[Dict]:
    """Backward compatibility wrapper for old detect_objects signature.

    Converts legacy parameters to DetectionConfig for new implementation.
    """
    config = DetectionConfig(
        confidence_threshold=confidence_threshold,
        device=device,
        max_size=max_size,
        classes_filter=classes_filter,
        min_area_ratio=min_area_ratio,
    )
    return detect_objects(image_path, config)


# Legacy alias for backward compatibility with scripts
_detect_objects = _detect_objects_compat


def save_overlay(
    image_path: str,
    detections: Optional[List[Dict]] = None,
    out_path: Optional[str] = None,
    label_font: Optional[str] = None,
) -> str:
    """Draw bounding boxes and text labels on an image and save overlay.

    If `detections` is None, runs `detect_objects` with default parameters.
    Returns saved overlay path.
    """
    from pathlib import Path

    from PIL import Image, ImageDraw, ImageFont

    if detections is None:
        detections = detect_objects(image_path)

    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    if detections and isinstance(detections[0], dict) and "det_w" in detections[0]:
        det_w = detections[0].get("det_w", orig_w)
        det_h = detections[0].get("det_h", orig_h)
    else:
        det_w, det_h = orig_w, orig_h

    scale_x = orig_w / det_w if det_w else 1.0
    scale_y = orig_h / det_h if det_h else 1.0

    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    # Try to load a readable truetype font. If not available, fall back to a
    # default bitmap font but log detailed diagnostics so missing fonts are
    # visible in logs (no silent swallow).
    font = None
    candidates = []
    if label_font:
        candidates.append(label_font)
    # Common fonts on many systems
    candidates.extend(
        [
            "DejaVuSans.ttf",
            "LiberationSans-Regular.ttf",
            "Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/SFNS.ttf",
        ]
    )

    for c in candidates:
        try:
            font = ImageFont.truetype(c, 48)
            if DETECT_VERBOSE:
                logging.info(f"Loaded overlay font: {c}")
            break
        except Exception as exc:
            logging.debug(f"Overlay font candidate failed: {c} -> {exc}")

    if font is None:
        # Last-resort: use PIL's default bitmap font. This may look worse, but
        # ensures overlays are produced instead of raising an OSError.
        try:
            font = ImageFont.load_default()  # type: ignore[assignment]
            logging.warning("Falling back to PIL default font for overlays; truetype fonts unavailable")
        except Exception:
            logging.exception("Failed to load any overlay font (tried candidates and default)")
            raise

    for d in detections:
        box = d.get("bbox")
        if not box:
            continue
        x1, y1, x2, y2 = box
        x1 *= scale_x
        x2 *= scale_x
        y1 *= scale_y
        y2 *= scale_y
        label = f"{d.get('class', 'obj')} {d.get('confidence', 0.0):.2f}"

        # Draw a much thicker rectangle for improved visibility (use tuple for typing)
        draw.rectangle((x1, y1, x2, y2), outline=(255, 165, 0), width=20)
        if hasattr(font, "getbbox"):
            bbox = font.getbbox(label)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        elif hasattr(font, "getsize"):
            try:
                text_w, text_h = font.getsize(label)
            except Exception:
                logging.exception("Failed to compute text size using font.getsize")
                raise
        else:
            # Fall back to a conservative estimate but still report if this happens
            logging.error("Font object lacks getbbox/getsize; using fallback text size estimation")
            text_w, text_h = (len(label) * 8, 16)
        # Increase padding for much bigger font
        padding_x = 32
        padding_y = 24
        draw.rectangle((x1, y1 - text_h - padding_y, x1 + text_w + padding_x, y1), fill=(255, 165, 0))
        draw.text((x1 + 16, y1 - text_h - (padding_y - 8)), label, fill="black", font=font)

    if out_path is None:
        out_dir = Path(".cache/overlays")
        out_dir.mkdir(parents=True, exist_ok=True)
        final_path = out_dir / (Path(image_path).name + ".overlay.png")
    else:
        final_path = Path(out_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)

    overlay.save(final_path)
    return str(final_path)


__all__.append("save_overlay")


def get_loaded_device() -> Optional[str]:
    """Return the device string used for the loaded detection backend in this process, if any.

    This maps the current `DETECTION_BACKEND` to the device that was actually used
    when the backend was first loaded in this process. Returns None if no model
    has been loaded yet.
    """
    try:
        return _detection_ctx.device_map.get(DETECTION_BACKEND)
    except Exception:
        return None


def get_loaded_model_name() -> Optional[str]:
    """Return the model name used for the loaded detection backend in this process, if any.

    This maps the current `DETECTION_BACKEND` to the model name that was actually used
    when the backend was first loaded in this process. Returns None if no model
    has been loaded yet.
    """
    try:
        return _detection_ctx.model_name_map.get(DETECTION_BACKEND)
    except Exception:
        return None


def sanitize_detection(item):
    """Return canonical detection dict for a single detection item.

    Accepts either:
    - a tuple/list of (name, score)
    - a dict with keys 'class' and 'confidence' and optional 'bbox', 'det_w', 'det_h'

    Raises ValueError for any other shape (fail-fast).
    """
    # tuple/list form
    if isinstance(item, (list, tuple)) and len(item) == 2:
        name, score = item
        return {"class": name, "confidence": float(score), "bbox": None}

    # dict form
    if isinstance(item, dict):
        result = {}
        if "class" in item and "confidence" in item:
            result["class"] = item["class"]
            result["confidence"] = float(item["confidence"])
            result["bbox"] = item.get("bbox") if "bbox" in item else None
            # Preserve det_w and det_h if present (critical for bbox scaling)
            if "det_w" in item:
                result["det_w"] = item["det_w"]
            if "det_h" in item:
                result["det_h"] = item["det_h"]
            return result
        # tolerate legacy keys used in detection outputs
        if "name" in item and "score" in item:
            result = {"class": item["name"], "confidence": float(item["score"]), "bbox": item.get("bbox")}
            # Preserve det_w and det_h if present
            if "det_w" in item:
                result["det_w"] = item["det_w"]
            if "det_h" in item:
                result["det_h"] = item["det_h"]
            return result

    raise ValueError(f"Unexpected detection item shape: {type(item)}")


