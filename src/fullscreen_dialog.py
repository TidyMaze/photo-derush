from __future__ import annotations

"""Fullscreen/Compare dialog (SRP): multi-image display, zoom, navigation, overlay."""
import logging
import os

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QDialog, QLabel, QScrollArea, QVBoxLayout, QWidget

from .settings import get_setting


class FullscreenDialog(QDialog):
    def __init__(self, image_paths: list[str], parent=None, viewmodel=None):
        super().__init__(parent)
        self.viewmodel = viewmodel
        if self.viewmodel:
            for sig in [
                self.viewmodel.exif_changed,
                self.viewmodel.rating_changed,
                self.viewmodel.tags_changed,
                self.viewmodel.primary_selection_changed,
            ]:
                sig.connect(lambda *a: self._update_overlay())
            self.viewmodel.label_changed.connect(lambda *a: self._update_overlay())
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)
        self.setWindowState(Qt.WindowState.WindowFullScreen)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._current_theme = None
        self._zoom_factor = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 6.0
        self._focus_index = 0
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        outer_layout.addWidget(self.scroll_area)
        container = QWidget()
        self._container_layout = QVBoxLayout(container)
        self._container_layout.setContentsMargins(20, 20, 20, 20)
        self._container_layout.setSpacing(12)
        self._container_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(container)
        self.scroll_area.viewport().installEventFilter(self)
        container.installEventFilter(self)
        self.installEventFilter(self)
        self._labels: list[QLabel] = []
        self._image_paths = list(image_paths)
        self._overlay_visible = True
        self._overlay_label = QLabel(self)
        self._overlay_label.setObjectName("metadataOverlay")
        self._overlay_label.setStyleSheet(
            "#metadataOverlay { background: rgba(0,0,0,140); color: #eee; padding: 8px 12px; border-radius: 6px; font-family: monospace; }"
        )
        self._overlay_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._overlay_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        for path in self._image_paths:
            if isinstance(path, list):
                logging.error(f"Path list instead of str: {path}")
                path = path[0] if path else ""
            elif not isinstance(path, (str, os.PathLike)):
                logging.error(f"Invalid path type {type(path)}: {path}")
                path = str(path)
            label = QLabel()
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setScaledContents(False)
            label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            if os.path.exists(path):
                try:
                    from PIL import Image

                    img = Image.open(path).convert("RGBA")
                    data = img.tobytes()
                    w, h = img.size
                    img_format = getattr(QImage, "Format_RGBA8888", QImage.Format.Format_ARGB32)
                    qimg = QImage(data, w, h, img_format)
                    pixmap = QPixmap.fromImage(qimg)
                    label.original_pixmap = pixmap  # type: ignore[attr-defined]
                    label.setPixmap(pixmap)
                except Exception as e:
                    logging.error(f"Failed to load fullscreen image {path}: {e}")
                    label.setText("Failed to load image")
            else:
                label.setText("File not found")
            self._container_layout.addWidget(label)
            self._labels.append(label)
        QTimer.singleShot(0, self._layout_images)
        QTimer.singleShot(50, self._update_overlay)
        QTimer.singleShot(0, self._apply_theme)
        QTimer.singleShot(0, self.setFocus)
        QTimer.singleShot(0, self.grabKeyboard)
        QTimer.singleShot(0, self._init_focus_index)

    def showEvent(self, event):
        super().showEvent(event)
        self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
        self.grabKeyboard()

    def closeEvent(self, event):
        try:
            self.releaseKeyboard()
        except Exception:
            logging.exception("Error releasing keyboard in FullscreenDialog.closeEvent")
        super().closeEvent(event)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress:
            self.keyPressEvent(event)
            return True
        return super().eventFilter(obj, event)

    def _apply_theme(self):
        theme = get_setting("fullscreen_theme", "dark") or "dark"
        if theme == self._current_theme:
            return
        self._current_theme = theme
        target = self.scroll_area.viewport()
        if theme == "dark":
            target.setStyleSheet("background: #111;")
        elif theme == "neutral":
            target.setStyleSheet("background: #2b2b2b;")
        else:
            target.setStyleSheet("background: #111;")

    def _cycle_theme(self):
        order = ["dark", "neutral"]
        curr = self._current_theme or (get_setting("fullscreen_theme", "dark") or "dark")
        if curr not in order:
            curr = "dark"
        nxt = order[(order.index(curr) + 1) % len(order)]
        from .settings import set_setting

        set_setting("fullscreen_theme", nxt)
        self._apply_theme()

    def _init_focus_index(self):
        if not self._image_paths:
            return
        primary = None
        if self.viewmodel and self.viewmodel.selection_model:
            primary = self.viewmodel.selection_model.primary()
        if primary and primary in self._image_paths:
            self._focus_index = self._image_paths.index(primary)
        else:
            self._focus_index = 0
        self._apply_focus_styles()
        self._ensure_focus_visible()

    def _apply_focus_styles(self):
        if not self._labels:
            return
        focused_style = "border: 4px solid #2d84ff; border-radius:6px;"
        for idx, lbl in enumerate(self._labels):
            if idx == self._focus_index and len(self._labels) > 1:
                lbl.setStyleSheet(focused_style)
            elif idx != self._focus_index and lbl.styleSheet():
                lbl.setStyleSheet("")

    def _ensure_focus_visible(self):
        if 0 <= self._focus_index < len(self._labels):
            try:
                self.scroll_area.ensureWidgetVisible(self._labels[self._focus_index], 24, 24)
            except Exception:
                logging.exception("Error in fullscreen dialog render")
                raise

    def _layout_images(self):
        if not self._labels:
            return
        screen_obj = self.screen() or (self.parent().screen() if self.parent() else None)
        if screen_obj is None:
            avail_w, avail_h = 1920, 1080
        else:
            geo = screen_obj.availableGeometry()
            avail_w, avail_h = geo.width(), geo.height()
        try:
            frac = float(get_setting("fullscreen_single_fraction", 0.92) or 0.92)
        except Exception:
            frac = 0.92
        frac = max(0.5, min(1.0, frac))
        max_w = int(avail_w * frac)
        max_h = int(avail_h * frac)
        for label in self._labels:
            pm = getattr(label, "original_pixmap", None)
            if not pm or pm.isNull():
                continue
            ow, oh = pm.width(), pm.height()
            if ow <= 0 or oh <= 0:
                continue
            base_scale = min(max_w / ow, max_h / oh)
            eff = min(base_scale * self._zoom_factor, self._max_zoom * base_scale)
            tw = max(1, int(ow * eff))
            th = max(1, int(oh * eff))
            scaled = pm.scaled(tw, th, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            label.setPixmap(scaled)
            label.setFixedSize(scaled.width(), scaled.height())
        self._position_overlay()
        self._apply_focus_styles()
        self._ensure_focus_visible()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        mods = event.modifiers()
        if delta and self._mods_has_ctrl_like(mods):
            factor = 1.1 if delta > 0 else 0.9
            self._set_zoom(self._zoom_factor * factor)
            event.accept()
            return
        super().wheelEvent(event)

    def _mods_has_ctrl_like(self, mods):
        return bool(mods & Qt.KeyboardModifier.ControlModifier) or bool(mods & Qt.KeyboardModifier.MetaModifier)

    def _set_zoom(self, factor: float):
        factor = max(self._min_zoom, min(self._max_zoom, factor))
        if abs(factor - self._zoom_factor) < 1e-4:
            return
        self._zoom_factor = factor
        self._layout_images()

    def _position_overlay(self):
        if not self._overlay_visible:
            self._overlay_label.hide()
            return
        margin = 16
        self._overlay_label.adjustSize()
        self._overlay_label.move(margin, margin)
        self._overlay_label.show()

    def _current_primary_path(self):
        if not self._image_paths:
            return None
        if self.viewmodel and self.viewmodel.selection_model.primary() in self._image_paths:
            return self.viewmodel.selection_model.primary()
        return self._image_paths[0]

    def _update_overlay(self):
        if not self._overlay_visible:
            self._overlay_label.hide()
            return
        path = self._current_primary_path()
        if not path:
            self._overlay_label.setText("No image")
            self._position_overlay()
            return
        filename = os.path.basename(path)
        label = exif_short = ""
        dims = ""
        if self.viewmodel and self.viewmodel.selected_image == path:
            label = self.viewmodel.label or ""
            if isinstance(self.viewmodel.exif, dict) and self.viewmodel.exif:
                fields = []
                for k in ["Model", "LensModel", "FNumber", "ExposureTime", "ISO", "FocalLength"]:
                    v = self.viewmodel.exif.get(k)
                    if v is not None:
                        fields.append(f"{k}={v}")
                exif_short = " | ".join(fields)
        for lbl in self._labels:
            has_valid_pixmap = (
                hasattr(lbl, "original_pixmap") and lbl.original_pixmap and not lbl.original_pixmap.isNull()
            )
            if has_valid_pixmap:
                dims = f"{lbl.original_pixmap.width()}x{lbl.original_pixmap.height()}"
                break
        parts = [filename]
        if len(self._image_paths) > 1:
            parts.append(f"{self._focus_index+1}/{len(self._image_paths)}")
        if dims:
            parts.append(dims)
        if label:
            parts.append(f"[{label}]")
        if exif_short:
            parts.append(exif_short)
        self._overlay_label.setText("  ".join(parts))
        self._position_overlay()

    def _navigate(self, direction: int):
        if len(self._image_paths) > 1:
            count = len(self._image_paths)
            if count == 0:
                return
            self._focus_index = (self._focus_index + direction) % count
            new_path = self._image_paths[self._focus_index]
            if self.viewmodel and self.viewmodel.selection_model:
                try:
                    current_sel = set(self.viewmodel.selection_model.selected())
                    current_sel.add(new_path)
                    self.viewmodel.selection_model.set(list(current_sel), new_path)
                except Exception:
                    logging.exception("Failed to update selection in FullscreenDialog._navigate")
            self._apply_focus_styles()
            self._ensure_focus_visible()
            self._update_overlay()
            return
        if not self.viewmodel:
            return
        current = self.viewmodel.selection_model.primary()
        if not current:
            return
        try:
            ordered_names = self.viewmodel.current_filtered_images()
        except Exception:
            ordered_names = []
        if not ordered_names:
            ordered_names = list(self.viewmodel.images)
        full_paths = [
            self.viewmodel.model.get_image_path(name)
            for name in ordered_names
            if self.viewmodel.model.get_image_path(name)
        ]
        if current not in full_paths:
            return
        idx = full_paths.index(current)
        new_idx = (idx + direction) % len(full_paths)
        new_path = full_paths[new_idx]
        self.viewmodel.selection_model.replace(new_path)
        if len(self._image_paths) == 1:
            self._image_paths[0] = new_path
            if os.path.exists(new_path):
                try:
                    from PIL import Image

                    img = Image.open(new_path).convert("RGBA")
                    data = img.tobytes()
                    w, h = img.size
                    img_format = getattr(QImage, "Format_RGBA8888", QImage.Format.Format_ARGB32)
                    qimg = QImage(data, w, h, img_format)
                    pix = QPixmap.fromImage(qimg)
                    lbl = self._labels[0]
                    lbl.original_pixmap = pix  # type: ignore[attr-defined]
                    lbl.setPixmap(pix)
                    self._layout_images()
                except Exception as e:
                    logging.error(f"Navigation load failed: {e}")
        self._update_overlay()

    def _apply_rating(self, rating: int):
        if self.viewmodel:
            self.viewmodel.set_rating(rating)
            self._update_overlay()

    def _apply_label(self, label: str):
        if self.viewmodel:
            self.viewmodel.set_label(label)
            self._update_overlay()

    def keyPressEvent(self, event):
        key = event.key()
        mods = event.modifiers()
        if key in (Qt.Key.Key_Escape, Qt.Key.Key_F, Qt.Key.Key_Space):
            self.close()
            return
        if key == Qt.Key.Key_B:
            self._cycle_theme()
            return
        if key in (Qt.Key.Key_Left, Qt.Key.Key_A):
            self._navigate(-1)
            return
        if key in (Qt.Key.Key_Right, Qt.Key.Key_D):
            self._navigate(1)
            return
        if key == Qt.Key.Key_K:
            self._apply_label("keep")
            return
        if key == Qt.Key.Key_T:
            self._apply_label("trash")
            return
        if key == Qt.Key.Key_I:
            self._overlay_visible = not self._overlay_visible
            self._update_overlay()
            return
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self._set_zoom(self._zoom_factor * 1.1)
            return
        if key in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            self._set_zoom(self._zoom_factor / 1.1)
            return
        if key == Qt.Key.Key_0 and self._mods_has_ctrl_like(mods):
            self._set_zoom(1.0)
            return
        super().keyPressEvent(event)


__all__ = ["FullscreenDialog"]
