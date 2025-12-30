from __future__ import annotations

from typing import Any

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import QWidget

from src.cache_config import is_cache_disabled
from src.constants import BADGE_COLORS, BADGE_ICONS, OVERLAY_CACHE_SIZE
from src.view_helpers import overlay_state_hash, paint_bboxes

# Simple cache for overlay pixmaps. Keyed by (w,h,dpr,overlay_hash)
_overlay_cache: dict[str, Any] = {}
_overlay_cache_max = OVERLAY_CACHE_SIZE


class OverlayWidget(QWidget):
    """Transparent overlay widget placed on top of a thumbnail QLabel.

    Draws bounding boxes, a keep/trash badge, and optional debug markers.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.objects = []
        self.label_text = ""
        self.is_auto = False
        self.pred_prob = None
        # overlay image offset and image size in logical coords
        self._overlay_image_offset = (0, 0, parent.width() if parent else 0, parent.height() if parent else 0)
        self._cached_key = None
        self._cached_pixmap = None

    def set_overlay(self, label_text, is_auto, pred_prob, objects, overlay_image_offset=None):
        self.label_text = label_text or ""
        self.is_auto = bool(is_auto)
        self.pred_prob = pred_prob
        self.objects = list(objects) if objects else []
        if overlay_image_offset is not None:
            self._overlay_image_offset = overlay_image_offset
        self.update()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if not self._overlay_image_offset or self._overlay_image_offset[2] == 0:
            self._overlay_image_offset = (0, 0, self.width(), self.height())

    def paintEvent(self, ev):
        """Paint overlay with caching to avoid expensive repeated draws."""
        w = max(1, self.width())
        h = max(1, self.height())

        # Compute deterministic state hash
        state_hash = overlay_state_hash(self.objects, self.label_text, self.is_auto, self.pred_prob)

        # Get device pixel ratio
        from PySide6.QtGui import QGuiApplication
        dpr = float(QGuiApplication.primaryScreen().devicePixelRatio() or 1.0)

        key = (w, h, round(dpr, 2), state_hash)

        # Try cached pixmap first
        pm = None
        if not is_cache_disabled():
            pm = _overlay_cache.get(key)
        if pm is not None:
            qp = QPainter(self)
            qp.drawPixmap(0, 0, pm)
            qp.end()
            return

        # No cached pixmap; draw into temporary QPixmap and cache it
        temp = QPixmap(self.width() or 1, self.height() or 1)
        temp.fill(Qt.GlobalColor.transparent)
        p = QPainter(temp)

        # Draw badge icon if present
        icon = BADGE_ICONS.get(self.label_text)
        if icon:
            color_key = f'{self.label_text}_{"auto" if self.is_auto else "manual"}'
            color = QColor(BADGE_COLORS.get(color_key, "white"))
            p.setPen(color)
            f = p.font()
            f.setPointSize(14)
            f.setBold(True)
            p.setFont(f)
            r = QRect(0, 0, temp.width(), temp.height())
            p.drawText(r, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight, icon)

        # Paint bounding boxes into temp pixmap
        if self.objects:
            offset_x, offset_y, img_w, img_h = self._overlay_image_offset
            paint_bboxes(p, self.objects, offset_x, offset_y, img_w, img_h, thumb_label=None)

        p.end()

        # Cache temp pixmap (FIFO eviction)
        if not is_cache_disabled():
            _overlay_cache[key] = temp
            if len(_overlay_cache) > _overlay_cache_max:
                _overlay_cache.pop(next(iter(_overlay_cache)))

        # Blit to screen
        qp = QPainter(self)
        qp.drawPixmap(0, 0, temp)
        qp.end()
