"""Widget for displaying group badges (BEST pick, group size)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QWidget


class GroupBadgeWidget(QWidget):
    """Transparent overlay widget for displaying group badges (BEST, ×N).

    Usage:
        group_badge = GroupBadgeWidget(image_label)
        group_badge.set_group_info(is_best=True, group_size=3)
        group_badge.set_geometry(x, y, width, height)
        group_badge.show()
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.is_best = False
        self.group_size = 1
        self.group_id = None
        self._updating = False

    def set_group_info(self, is_best: bool = False, group_size: int = 1, group_id: int | None = None):
        """Set group badge information.

        Args:
            is_best: True if this is the best pick in the group
            group_size: Number of photos in the group
            group_id: Group ID to display
        """
        if self._updating:
            return

        # Always update to ensure visibility (don't skip if values are the same)
        # This ensures badges appear even if they were hidden before
        self._updating = True
        try:
            self.is_best = is_best
            self.group_size = group_size
            self.group_id = group_id
            self.update()  # Trigger repaint
        finally:
            self._updating = False

    def paintEvent(self, event):
        """Draw group badges at display resolution."""
        # Show badges if we have group info (group_id, BEST, or ×N)
        # Show if: group_id exists, OR (is_best AND group_size >= 2), OR group_size > 1
        has_group_id = self.group_id is not None
        has_best_badge = self.is_best and self.group_size >= 2
        has_size_badge = self.group_size > 1
        
        if not self.isVisible():
            return
        
        if not (has_group_id or has_best_badge or has_size_badge):
            return  # Nothing to show

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # Get device pixel ratio
        try:
            from PySide6.QtGui import QGuiApplication
            screen = QGuiApplication.primaryScreen()
            dpr = float(screen.devicePixelRatio() or 1.0) if screen else 1.0
        except Exception:
            dpr = 1.0

        widget_w = self.width()
        widget_h = self.height()
        if widget_w <= 0 or widget_h <= 0:
            return

        # Font setup - consistent smaller font for all badges
        font = QFont()
        base_font_size = 5.5  # Smaller font for all badges
        font_pixel_size = max(7, int(base_font_size * dpr))  # Minimum 7px
        font.setPixelSize(font_pixel_size)
        font.setBold(True)
        painter.setFont(font)
        
        # Use QFontMetrics to calculate text width for proper badge sizing
        font_metrics = QFontMetrics(font)
        
        # Calculate required width for each badge text
        best_text = "⭐ BEST"
        best_text_width = font_metrics.horizontalAdvance(best_text)
        
        # Calculate max group size text width (e.g., "×999" for very large groups)
        max_size_text = f"×{self.group_size}"
        max_size_text_width = font_metrics.horizontalAdvance(max_size_text)
        
        # Calculate group_id text width (e.g., "#999" for large group IDs)
        group_id_text = f"#{self.group_id}" if self.group_id is not None else ""
        group_id_text_width = font_metrics.horizontalAdvance(group_id_text) if group_id_text else 0
        
        # Badge dimensions - use text width + padding, but respect widget bounds
        padding = 8  # Horizontal padding (4px on each side)
        min_badge_w = 35
        max_badge_w = int(widget_w * 0.4)  # Max 40% of widget width
        
        # Calculate required width for all badges that will be shown
        required_widths = []
        if self.is_best and self.group_size >= 2:
            required_widths.append(best_text_width + padding)
        if self.group_size > 1:
            required_widths.append(max_size_text_width + padding)
        if self.group_id is not None:
            required_widths.append(group_id_text_width + padding)
        
        # Use the maximum required width, but respect min/max bounds
        badge_w = max(min_badge_w, min(max(required_widths) if required_widths else min_badge_w, max_badge_w))
        badge_h = max(14, int(widget_h * 0.10))  # ~10% of height, minimum 14px

        # Position badges at top-left
        x = 2
        y = 2

        badges_drawn = 0

        # Draw "BEST" badge only if group has at least 2 images
        if self.is_best and self.group_size >= 2:
            best_x = x
            best_y = y + badges_drawn * (badge_h + 2)

            # Background (gold/yellow)
            bg_color = QColor(255, 215, 0, 240)  # Gold with transparency
            shadow_color = QColor(0, 0, 0, 100)

            # Drop shadow
            painter.fillRect(best_x + 1, best_y + 1, badge_w, badge_h, shadow_color)

            # Background
            painter.fillRect(best_x, best_y, badge_w, badge_h, bg_color)

            # Text
            painter.setPen(QPen(QColor(0, 0, 0, 200), 1))
            painter.drawText(best_x, best_y, badge_w, badge_h, Qt.AlignmentFlag.AlignCenter, "⭐ BEST")
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(best_x, best_y, badge_w, badge_h, Qt.AlignmentFlag.AlignCenter, "⭐ BEST")

            badges_drawn += 1

        # Draw group size badge if group_size > 1
        if self.group_size > 1:
            size_x = x
            size_y = y + badges_drawn * (badge_h + 2)

            # Background (blue/cyan)
            bg_color = QColor(70, 130, 180, 220)  # Steel blue with transparency
            shadow_color = QColor(0, 0, 0, 100)

            # Drop shadow
            painter.fillRect(size_x + 1, size_y + 1, badge_w, badge_h, shadow_color)

            # Background
            painter.fillRect(size_x, size_y, badge_w, badge_h, bg_color)

            # Text
            text = f"×{self.group_size}"
            painter.setPen(QPen(QColor(0, 0, 0, 200), 1))
            painter.drawText(size_x, size_y, badge_w, badge_h, Qt.AlignmentFlag.AlignCenter, text)
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(size_x, size_y, badge_w, badge_h, Qt.AlignmentFlag.AlignCenter, text)

            badges_drawn += 1

        # Draw group_id badge (always show if group_id is not None)
        if self.group_id is not None:
            group_x = x
            group_y = y + badges_drawn * (badge_h + 2)

            # Background (gray/dark)
            bg_color = QColor(100, 100, 100, 200)  # Gray with transparency
            shadow_color = QColor(0, 0, 0, 100)

            # Drop shadow
            painter.fillRect(group_x + 1, group_y + 1, badge_w, badge_h, shadow_color)

            # Background
            painter.fillRect(group_x, group_y, badge_w, badge_h, bg_color)

            # Text
            text = f"#{self.group_id}"
            painter.setPen(QPen(QColor(0, 0, 0, 200), 1))
            painter.drawText(group_x, group_y, badge_w, badge_h, Qt.AlignmentFlag.AlignCenter, text)
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(group_x, group_y, badge_w, badge_h, Qt.AlignmentFlag.AlignCenter, text)

