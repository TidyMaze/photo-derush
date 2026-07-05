import os
import pytest
from PySide6.QtWidgets import QApplication
from src.overlay_widget import OverlayWidget, _overlay_cache
from src.badge_overlay_widget import BadgeOverlayWidget
from src.dpr_helper import get_dpr

def test_overlay_pixmap_sharpness_high_dpr():
    # Setup QApplication if not exists
    app = QApplication.instance() or QApplication([])

    # Set environment variable to force Retina scaling (DPR = 2.0)
    os.environ["FORCE_DPR"] = "2.0"
    try:
        assert get_dpr() == 2.0

        # Test OverlayWidget cached pixmap DPR
        overlay = OverlayWidget()
        overlay.resize(200, 150)
        overlay.set_overlay(label_text="keep", is_auto=False, pred_prob=0.9, objects=[{"class": "person", "bbox": [0.1, 0.1, 0.5, 0.5]}])
        
        # Trigger paintEvent manually since widget is offscreen
        overlay.paintEvent(None)

        # Find key in cache
        cached_pixmaps = list(_overlay_cache.values())
        assert len(cached_pixmaps) > 0
        latest_pixmap = cached_pixmaps[-1]
        
        # This will fail on the current implementation because devicePixelRatio is 1.0
        assert latest_pixmap.devicePixelRatio() == 2.0

        # Test BadgeOverlayWidget cached pixmap DPR
        badge = BadgeOverlayWidget()
        badge.resize(200, 150)
        badge.set_badge(label_text="keep", label_source="manual", probability=0.9)
        
        # Trigger paintEvent manually since widget is offscreen
        badge.paintEvent(None)
        
        # This will also fail on the current implementation
        assert badge._cached_pixmap is not None
        assert badge._cached_pixmap.devicePixelRatio() == 2.0

    finally:
        if "FORCE_DPR" in os.environ:
            del os.environ["FORCE_DPR"]
