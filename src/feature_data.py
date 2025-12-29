"""Data structures for feature extraction to replace 16-element tuple."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ImagePreprocessResult:
    """Result of image preprocessing containing all computed values."""

    # Core image data
    img: Image.Image
    gray_arr: np.ndarray
    w: int
    h: int

    # RGB statistics
    mean_r: float
    mean_g: float
    mean_b: float
    std_r: float
    std_g: float
    std_b: float

    # Brightness statistics
    mean_brightness: float
    std_brightness: float

    # Histogram features
    hist_feat: np.ndarray

    # File metadata
    file_size: int
    aspect: float
    exif_data: dict
