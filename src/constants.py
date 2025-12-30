"""Project-wide constants for photo-derush application.

Centralizes magic numbers and configuration values to avoid duplication
and improve maintainability.
"""

# Detection Backend Configuration
DETECTION_BACKEND = "yolov8"
"""Object detection backend to use"""

YOLO_MODEL_NAME = "yolov8n"
"""YOLOv8 model variant (nano, small, medium, large, xlarge)"""

MIN_CONFIDENCE_KEEP_YOLO = 0.6
"""Hard lower bound for YOLOv8 detections (higher precision preferred)"""

MIN_CONFIDENCE_KEEP_FRCNN = 0.5
"""Hard lower bound for Faster R-CNN detections"""

# Detection Configuration
CONFIDENCE_THRESHOLD = 0.6
"""Default confidence threshold for object detection"""

HIGH_CONFIDENCE_THRESHOLD = 0.8
"""High confidence threshold for filtering prominent detections"""

MAX_DETECTION_SIZE = 800
"""Maximum image dimension for detection processing (scales down larger images)"""

MIN_AREA_RATIO = 0.0
"""Minimum bounding box area ratio (fraction of image) to include in results"""

HIGH_CONF_CAP = 5
"""Maximum number of high-confidence detections to show in overlay"""

# Cache Configuration
PIXMAP_CACHE_SIZE = 256
"""Maximum number of pixmaps to cache in memory"""

OVERLAY_CACHE_SIZE = 512
"""Maximum number of overlay pixmaps to cache"""

THUMBNAIL_CACHE_SIZE = 256
"""Maximum number of thumbnails to cache"""

# UI Configuration
DEFAULT_THUMB_SIZE = 128
"""Default thumbnail size in pixels"""

MIN_THUMB_SIZE = 64
"""Minimum thumbnail size"""

MAX_THUMB_SIZE = 512
"""Maximum thumbnail size"""

ZOOM_DEBOUNCE_MS = 250
"""Debounce delay for zoom slider changes (milliseconds)"""

# Model Configuration
DEFAULT_MODEL_PATH = "~/.photo-derush-keep-trash-model.joblib"
"""Default path for trained classification model"""

BEST_PARAMS_PATH = "~/.photo-derush-best-params.json"
"""Default path for hyperparameter tuning results"""

# Badge Colors
BADGE_COLORS = {
    "keep_manual": "#008000",  # green
    "keep_auto": "#90EE90",  # light green
    "trash_manual": "#FF0000",  # red
    "trash_auto": "#FFB6C1",  # light pink
}
"""Color scheme for keep/trash badges (manual vs auto-labeled)"""

# Badge Icons
BADGE_ICONS = {
    "keep": "\u2713",  # checkmark
    "trash": "\u2717",  # X mark
}
"""Unicode symbols for keep/trash badges"""
