"""Reusable Qt widget component for displaying badge overlays at display resolution."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget


class BadgeOverlayWidget(QWidget):
    """Reusable transparent overlay widget for drawing badges at display resolution.
    
    This widget renders badges (keep/trash with score) at physical pixel resolution
    for sharp rendering, similar to BoundingBoxOverlayWidget.
    
    Usage:
        badge_overlay = BadgeOverlayWidget(image_label)
        badge_overlay.set_badge(label_text="keep", label_source="manual", probability=0.85)
        badge_overlay.set_geometry(x, y, width, height)
        badge_overlay.show()
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.label_text = ""
        self.label_source = ""  # "manual", "auto", or "predicted"
        self.probability = None
        self._updating = False  # Guard against recursive updates
    
    def set_badge(self, label_text="", label_source="", probability=None):
        """Set the badge to display.
        
        Args:
            label_text: "keep" or "trash"
            label_source: "manual", "auto", or "predicted"
            probability: Optional float (0.0-1.0) to show as percentage
        """
        if self._updating:
            return
        # Only update if badge data actually changed
        new_label_text = label_text or ""
        new_label_source = label_source or ""
        if (self.label_text == new_label_text and 
            self.label_source == new_label_source and 
            self.probability == probability):
            return
        
        self._updating = True
        try:
            self.label_text = new_label_text
            self.label_source = new_label_source
            self.probability = probability
            self.update()
        finally:
            self._updating = False
    
    def paintEvent(self, event):
        """Draw badge at display resolution."""
        if not self.label_text:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        
        # Get device pixel ratio for sharp rendering
        # Use primary screen to avoid potential recursion from self.screen()
        try:
            from PySide6.QtGui import QGuiApplication
            screen = QGuiApplication.primaryScreen()
            dpr = float(screen.devicePixelRatio() or 1.0) if screen else 1.0
        except Exception:
            dpr = 1.0
        
        # Badge dimensions in logical pixels (will be scaled by DPR automatically by Qt)
        # Badge spans 100% width of widget, positioned at bottom
        widget_w = self.width()
        widget_h = self.height()
        if widget_w <= 0 or widget_h <= 0:
            return
        
        # Badge spans full width, height based on font size
        badge_w_logical = widget_w
        badge_h_logical = max(14, int(widget_h * 0.08))  # ~8% of height, minimum 14px
        
        # Position in logical pixels (bottom, full width)
        x = 0
        y = widget_h - badge_h_logical
        
        # Determine badge content
        is_keep = self.label_text.lower() == "keep"
        # Show prediction probability if available (even for manual/auto labels)
        # This allows users to see model confidence even after manual override
        # Only show 100% if no probability is available (manual label without prediction)
        if self.probability is not None and self.probability == self.probability:  # Check for NaN
            # Show probability aligned with the label (keep or trash)
            pct = self.probability * 100 if is_keep else (1 - self.probability) * 100
        elif self.label_source in ("manual", "auto"):
            # Manual/auto label but no prediction available - show 100% confidence in the label
            pct = 100.0
        else:
            # Predicted label but no probability (shouldn't happen, but handle gracefully)
            pct = 0.0
        letter = "K" if is_keep else "T"
        
        # Emoji indicator for source
        emoji = ""
        if self.label_source == "manual":
            emoji = "✓ "
        elif self.label_source == "auto":
            emoji = "🤖 "
        
        badge_text = f"{emoji}{pct:.1f}% {letter}"
        
        # Background color based on source
        if self.label_source == "manual":
            bg_color = QColor(27, 140, 77, 240) if is_keep else QColor(200, 50, 40, 240)
        elif self.label_source == "auto":
            bg_color = QColor(60, 200, 120, 200) if is_keep else QColor(250, 120, 100, 200)
        else:
            bg_color = QColor(39, 174, 96, 220) if is_keep else QColor(231, 76, 60, 220)
        
        # Text with proper DPR scaling for sharp rendering
        font = QFont()
        base_font_size_logical = 6.5  # Reduced from 8.0 for smaller text
        font_pixel_size = max(8, int(base_font_size_logical * dpr))
        font.setPixelSize(font_pixel_size)
        font.setBold(True)
        painter.setFont(font)
        
        # Drop shadow
        shadow_offset = 1
        painter.fillRect(x + shadow_offset, y + shadow_offset, badge_w_logical, badge_h_logical, QColor(0, 0, 0, 100))
        
        # Background
        painter.fillRect(x, y, badge_w_logical, badge_h_logical, bg_color)
        
        # Text outline (dark) then fill (white)
        painter.setPen(QPen(QColor(0, 0, 0, 200), 1))
        painter.drawText(x, y, badge_w_logical, badge_h_logical, Qt.AlignmentFlag.AlignCenter, badge_text)
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(x, y, badge_w_logical, badge_h_logical, Qt.AlignmentFlag.AlignCenter, badge_text)

