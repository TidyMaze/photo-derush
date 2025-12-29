"""Configuration objects for object detection operations.

Provides structured config classes to simplify function signatures
with many parameters.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DetectionConfig:
    """Configuration for object detection operations."""

    confidence_threshold: float = 0.6
    """Minimum confidence score to include detection"""

    device: str = "auto"
    """Device to run on ('cpu', 'cuda', or 'auto')"""

    max_size: int = 800
    """Maximum dimension to resize images to"""

    classes_filter: Optional[set[int]] = None
    """Set of class indices to detect (None for all classes)"""

    min_area_ratio: float = 0.0
    """Minimum detection box area as ratio of image area"""

    backend: Optional[str] = None
    """Detection backend ('yolov8', 'faster-rcnn', None=use env default)"""


@dataclass
class OverlayConfig:
    """Configuration for detection overlay rendering."""

    confidence_threshold: float = 0.6
    """Minimum confidence to include detection in overlay"""

    high_confidence_threshold: float = 0.8
    """Threshold for high-confidence badge"""

    high_confidence_cap: int = 5
    """Maximum number of high-confidence detections to show"""

    show_labels: bool = True
    """Whether to show class labels on bounding boxes"""

    show_scores: bool = True
    """Whether to show confidence scores"""

    line_width: int = 2
    """Bounding box line width"""
