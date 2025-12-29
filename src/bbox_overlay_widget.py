"""Reusable Qt widget component for displaying bounding boxes on images."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from src.view_helpers import paint_bboxes


class BoundingBoxOverlayWidget(QWidget):
    """Reusable transparent overlay widget for drawing bounding boxes on images.
    
    This widget can be placed on top of any image widget (QLabel, etc.) to display
    detected object bounding boxes with labels.
    
    Usage:
        overlay = BoundingBoxOverlayWidget(image_label)
        overlay.set_detections(detections, original_image_size=(width, height))
        overlay.set_geometry(x, y, display_width, display_height)
        overlay.show()
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.detections = []  # List of detection dicts with 'bbox', 'class', 'confidence', optionally 'det_w', 'det_h'
        self.original_image_size = None  # (width, height) of original image used for detection
    
    def set_detections(self, detections, original_image_size=None):
        """Set the detections to display.
        
        Args:
            detections: List of detection dicts with keys:
                - 'bbox': [x1, y1, x2, y2] in original image coordinates
                - 'class' or 'name': class name
                - 'confidence': confidence score (0.0-1.0)
                - 'det_w', 'det_h': optional, detection image size (if different from original)
            original_image_size: Optional (width, height) tuple of original image size.
                                If None, uses det_w/det_h from first detection.
        """
        self.detections = list(detections) if detections else []
        if original_image_size:
            self.original_image_size = original_image_size
        elif self.detections:
            # Try to infer from first detection
            first = self.detections[0]
            if first.get("det_w") and first.get("det_h"):
                self.original_image_size = (first.get("det_w"), first.get("det_h"))
        self.update()
    
    def paintEvent(self, event):
        """Draw bounding boxes using the reusable paint_bboxes function."""
        from PySide6.QtGui import QPainter
        import logging
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        if not self.detections or not self.original_image_size:
            logging.debug(f"[BBOX-WIDGET] paintEvent: no detections or size - detections={len(self.detections) if self.detections else 0}, size={self.original_image_size}")
            return
        
        # Get displayed image size (overlay widget matches image size)
        img_w = self.width()
        img_h = self.height()
        
        if img_w <= 0 or img_h <= 0:
            logging.debug(f"[BBOX-WIDGET] paintEvent: invalid size {img_w}x{img_h}")
            return
        
        # Ensure det_w/det_h are set for each detection
        det_w_orig, det_h_orig = self.original_image_size
        for det in self.detections:
            if not det.get("det_w"):
                det["det_w"] = det_w_orig
            if not det.get("det_h"):
                det["det_h"] = det_h_orig
        
        # Use reusable paint_bboxes function
        # offset_x=0, offset_y=0 because overlay widget is positioned to match image exactly
        logging.debug(f"[BBOX-WIDGET] paintEvent: drawing {len(self.detections)} bboxes on {img_w}x{img_h} widget")
        paint_bboxes(painter, self.detections, 0, 0, img_w, img_h, thumb_label=None)

