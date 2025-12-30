"""Reusable Qt widget component for displaying badge overlays at display resolution."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
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
        self._cached_pixmap = None  # Cache rendered badge to avoid repainting
        self._cache_key = None  # Key for cache invalidation
        # Ensure widget fills parent when parent is resized
        if parent:
            # Use thumb_size if parent not yet sized (labels are setFixedSize after creation)
            initial_size = parent.width() or 200
            self.setGeometry(0, 0, initial_size, initial_size)
    
    def resizeEvent(self, event):
        """Resize widget to match parent size."""
        super().resizeEvent(event)
        # Don't call setGeometry here - it's handled by badge refresh when needed
        # This just ensures widget updates its cached pixmap when resized
    
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
            # Invalidate cache when badge data changes
            self._cached_pixmap = None
            self._cache_key = None
            # Ensure widget is visible and has geometry before updating
            if not self.isVisible():
                self.show()
            if self.width() <= 0 or self.height() <= 0:
                parent = self.parent()
                if parent:
                    self.setGeometry(0, 0, parent.width(), parent.height())
            self.update()
            self.raise_()  # Ensure on top
        finally:
            self._updating = False
    
    def paintEvent(self, event):
        """Draw badge at display resolution."""
        # Always paint if widget is visible and has valid geometry
        if not self.isVisible():
            return
        if not self.label_text and self.probability is None:
            return  # Nothing to draw
        
        # Badge dimensions in logical pixels (will be scaled by DPR automatically by Qt)
        # Badge spans 100% width of widget, positioned at bottom
        widget_w = self.width()
        widget_h = self.height()
        if widget_w <= 0 or widget_h <= 0:
            return
        
        # Get device pixel ratio for sharp rendering
        # Use primary screen to avoid potential recursion from self.screen()
        try:
            from PySide6.QtGui import QGuiApplication
            screen = QGuiApplication.primaryScreen()
            dpr = float(screen.devicePixelRatio() or 1.0) if screen else 1.0
        except Exception:
            dpr = 1.0
        
        # OPTIMIZATION: Cache rendered badge pixmap to avoid expensive repaints
        # Cache key includes widget size, DPR, and badge content
        cache_key = (widget_w, widget_h, round(dpr, 2), self.label_text, self.label_source, self.probability)
        if self._cached_pixmap is not None and self._cache_key == cache_key:
            # Use cached pixmap
            painter = QPainter(self)
            painter.drawPixmap(0, 0, self._cached_pixmap)
            return
        
        # Render badge to pixmap
        pixmap = QPixmap(widget_w, widget_h)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        
        # Badge spans full width, height based on font size
        badge_w_logical = widget_w
        badge_h_logical = max(14, int(widget_h * 0.08))  # ~8% of height, minimum 14px
        
        # Position in logical pixels (bottom, full width)
        x = 0
        y = widget_h - badge_h_logical
        
        # Determine badge content
        is_keep = self.label_text.lower() == "keep"
        letter = "K" if is_keep else "T"
        
        # Emoji indicator for source
        emoji = ""
        if self.label_source == "manual":
            emoji = "✓ "
        elif self.label_source == "auto":
            emoji = "🤖 "
        
        # Only show percentage if we have a valid prediction probability
        # If no probability, just show the label without percentage (e.g., "✓ K" or "🤖 T")
        if self.probability is not None and self.probability == self.probability:  # Check for NaN
            # Show probability aligned with the label (keep or trash)
            pct = self.probability * 100 if is_keep else (1 - self.probability) * 100
            badge_text = f"{emoji}{pct:.1f}% {letter}"
        else:
            # No prediction available - show label only without percentage
            badge_text = f"{emoji}{letter}"
        
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
        
        painter.end()
        
        # Cache the rendered pixmap
        self._cached_pixmap = pixmap
        self._cache_key = cache_key
        
        # Draw cached pixmap to widget
        widget_painter = QPainter(self)
        widget_painter.drawPixmap(0, 0, pixmap)

