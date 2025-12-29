"""Helper functions for view rendering (SRP)."""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen

from src.constants import HIGH_CONF_CAP, HIGH_CONFIDENCE_THRESHOLD
from src.detections import Detection, normalize_detections


def overlay_state_hash(objects, label_text, is_auto, prediction_prob) -> int:
    """Compute a simple deterministic hash for overlay state."""
    prob_int = 0 if prediction_prob is None else int(round(prediction_prob * 100))
    obj_count = len(objects) if objects else 0
    return hash((label_text or "", bool(is_auto), prob_int, obj_count))


def select_overlay_detections(
    objects, high_conf_threshold: float = HIGH_CONFIDENCE_THRESHOLD, high_conf_cap: int = HIGH_CONF_CAP
):
    """Pure selection helper for thumbnail overlays.

    Args:
        objects: list of detection dicts (with 'confidence') or simple tuples/strings.
        high_conf_threshold: confidences strictly greater than this are considered high.
        high_conf_cap: maximum number of high-confidence detections to show.

    Returns:
        A tuple (mode, items) where mode is either 'bbox' or 'simple' and items
        is the list of items to display. For 'simple' items are dicts with
        keys 'name' and 'score'. For 'bbox' items, original detection dicts are
        returned (possibly truncated).
    """
    if not objects:
        return "none", []

    dets = normalize_detections(objects)
    if not dets:
        return "none", []

    # Separate bbox detections vs simple (no bbox)
    bbox_dets = [d for d in dets if d.bbox is not None]
    if bbox_dets:
        # Convert Detection objects back to plain dicts for bbox mode so
        # downstream consumers (and tests) that expect mapping-like
        # detections continue to work. Preserve original fields when
        # available.
        def _to_dict(d):
            if isinstance(d, Detection):
                return {
                    "name": d.name,
                    "confidence": float(d.confidence),
                    "bbox": list(d.bbox) if d.bbox is not None else None,
                    "det_w": d.det_w,
                    "det_h": d.det_h,
                    "raw": d.raw,
                }
            # assume it's already a dict-like
            return d

    # simple named detections
    simple = [{"name": d.name, "score": float(d.confidence)} for d in dets]
    high_conf_float = float(high_conf_threshold) if isinstance(high_conf_threshold, (int, float, str)) else 0.0
    high = [s for s in simple if float(s.get("score", 0.0) or 0.0) > high_conf_float]  # type: ignore[arg-type]
    high.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)  # type: ignore[arg-type]
    if len(high) > high_conf_cap:
        return "simple", high[:high_conf_cap]
    return "simple", simple


def update_label_icon(thumb_label, label, filename=None, is_auto=False, prediction_prob=None, objects=None):
    """Paint overlays directly onto existing base pixmap. KISS."""
    import os

    from PySide6.QtGui import QPainter

    # Get base pixmap - use _logical_pixmap which is the raw PNG-loaded version before DPR scaling
    base = getattr(thumb_label, "_logical_pixmap", None)
    if base is None or base.isNull():
        # Fallback to physical versions
        base = getattr(thumb_label, "base_pixmap", None) or getattr(thumb_label, "original_pixmap", None)
        if base is None or base.isNull():
            logging.warning(f"update_label_icon: no base pixmap for {filename}")
            return

    # Skip if pixmap is transparent placeholder (worker hasn't loaded real image yet)
    try:
        from PySide6.QtGui import QImage

        qi = base.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        import numpy as np

        arr = np.frombuffer(bytes(qi.bits()), dtype=np.uint8).reshape((qi.height(), qi.width(), 4))
        opaque_pixels = (arr[:, :, 3] == 255).sum()
        if opaque_pixels == 0:
            logging.debug(f"update_label_icon: skipping transparent placeholder for {filename}")
            return
    except Exception:
        pass  # If check fails, proceed anyway

    # Copy and paint
    pixmap = base.copy()
    painter = QPainter(pixmap)

    # Draw badge
    icon = None
    color = QColor("white")
    if label == "keep":
        icon = "\u2713"
        color = QColor("green") if not is_auto else QColor("#90EE90")
    elif label == "trash":
        icon = "\u2717"
        color = QColor("red") if not is_auto else QColor("#FFB6C1")
    if icon:
        painter.setPen(color)
        f = painter.font()
        f.setPointSize(24)
        f.setBold(True)
        painter.setFont(f)
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight, icon)

    # Draw bboxes
    if objects:
        offset_x, offset_y, img_w, img_h = getattr(
            thumb_label, "_overlay_image_offset", (0, 0, pixmap.width(), pixmap.height())
        )
        paint_bboxes(painter, objects[:5], offset_x, offset_y, img_w, img_h, thumb_label)

    painter.end()
    thumb_label.setPixmap(pixmap)

    # Dump if debugging
    if os.environ.get("PHOTO_DERUSH_DUMP_PIXMAPS"):
        try:
            dump_dir = os.path.join(".cache", "final_thumbs")
            os.makedirs(dump_dir, exist_ok=True)
            dump_path = os.path.join(dump_dir, f"{os.path.basename(filename) if filename else 'unknown'}_final.png")
            pixmap.save(dump_path)
        except Exception:
            pass


def paint_bboxes(painter: QPainter, objects, offset_x: int, offset_y: int, img_w: int, img_h: int, thumb_label=None):
    """Paint bounding boxes onto an active QPainter using unified styling.

    - Uses a thin, consistent outline scaled modestly with thumbnail size.
    - Draws class name labels in a high-contrast style.
    - Expects `painter` to be already begun on the target pixmap.
    """
    if painter is None:
        raise ValueError("paint_bboxes requires a QPainter")
    
    if not objects:
        logging.debug("paint_bboxes: No objects to draw")
        return
    
    logging.debug(f"paint_bboxes: Drawing {len(objects)} objects, img_w={img_w}, img_h={img_h}, offset=({offset_x}, {offset_y})")
    
    
    # Determine pen width in logical (device-independent) pixels so the
    # visual thickness appears constant regardless of thumbnail size or DPR.
    try:
        # nominal logical pen width (in CSS/logical pixels) - thinner and less attention-grabbing
        logical_pen_width = 1.0
        # Try to detect painter/device scale. Use deviceTransform when available
        device_scale = 1.0
        try:
            if hasattr(painter, "deviceTransform") and callable(painter.deviceTransform):
                dt = painter.deviceTransform()
                sx = getattr(dt, "m11", lambda: 1.0)()
                sy = getattr(dt, "m22", lambda: 1.0)()
                device_scale = max(1.0, (sx + sy) / 2.0)
        except Exception:
            device_scale = 1.0

        pen_w = max(1, int(round(logical_pen_width * device_scale)))
    except Exception:
        pen_w = 1

    outline_color = QColor(150, 200, 150)  # muted green for less attention
    text_bg = QColor(0, 0, 0, 160)
    text_fg = QColor(255, 255, 255)

    # Get device pixel ratio for sharp text rendering at display resolution
    text_dpr = 1.0
    try:
        from PySide6.QtGui import QGuiApplication
        device = painter.device()
        # Try to get DPR from the device's associated window/screen
        if device:
            try:
                if hasattr(device, 'window') and callable(device.window):
                    win = device.window()
                    if win:
                        screen = win.screen()
                        if screen:
                            text_dpr = float(screen.devicePixelRatio() or 1.0)
            except Exception:
                pass
        # Fallback to primary screen if DPR still 1.0
        if text_dpr == 1.0:
            screen = QGuiApplication.primaryScreen()
            if screen:
                text_dpr = float(screen.devicePixelRatio() or 1.0)
    except Exception:
        text_dpr = 1.0

    # Prepare shared drawing resources once to avoid repeated allocations
    # Use pixel size scaled by DPR for sharp rendering at physical pixel resolution
    try:
        font_shared = painter.font()
        # Base font size in logical pixels - scale by DPR for physical pixel rendering
        # This ensures text is drawn at display resolution, not scaled
        base_font_size_logical = 5
        font_pixel_size = max(4, int(base_font_size_logical * text_dpr))
        font_shared.setPixelSize(font_pixel_size)
        font_shared.setBold(False)
        # Enable text antialiasing for sharp rendering
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setFont(font_shared)
        try:
            from PySide6.QtGui import QFontMetrics

            fm_shared = QFontMetrics(font_shared)
        except Exception:
            fm_shared = None
    except Exception:
        font_shared = None
        fm_shared = None

    # Create pen and set it on painter immediately
    pen = QPen(outline_color, pen_w)
    pen.setStyle(Qt.PenStyle.SolidLine)
    pen.setCapStyle(Qt.PenCapStyle.SquareCap)
    pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
    painter.setPen(pen)

    for det in objects:
        try:
            bbox = det.get("bbox")
            if not bbox:
                continue
            det_w = det.get("det_w")
            det_h = det.get("det_h")
            
            # map and draw
            rx1, ry1, rx2, ry2 = map_bbox_to_thumbnail(
                bbox, det_w or img_w or 1, det_h or img_h or 1, img_w, img_h, offset_x, offset_y
            )
            w = max(1, rx2 - rx1)
            h = max(1, ry2 - ry1)

            # Clip coordinates to image bounds FIRST
            device = painter.device()
            if device:
                max_x = device.width() - 1
                max_y = device.height() - 1
                rx1_clipped = max(0, min(int(rx1), max_x))
                ry1_clipped = max(0, min(int(ry1), max_y))
                rx2_clipped = max(0, min(int(rx2), max_x))
                ry2_clipped = max(0, min(int(ry2), max_y))
                w_clipped = max(1, rx2_clipped - rx1_clipped)
                h_clipped = max(1, ry2_clipped - ry1_clipped)
            else:
                rx1_clipped = int(rx1)
                ry1_clipped = int(ry1)
                w_clipped = int(w)
                h_clipped = int(h)
            
            # Draw bbox outline - pen is already set above
            painter.setBrush(Qt.BrushStyle.NoBrush)
            # Verify pen is set correctly before drawing
            actual_pen = painter.pen()
            logging.debug(f"Drawing bbox: ({rx1_clipped}, {ry1_clipped}, {w_clipped}, {h_clipped}), pen_w={actual_pen.width()}, pen_color=({actual_pen.color().red()},{actual_pen.color().green()},{actual_pen.color().blue()}), device_size=({device.width() if device else 'N/A'}, {device.height() if device else 'N/A'})")
            painter.drawRect(rx1_clipped, ry1_clipped, w_clipped, h_clipped)

            # draw class label background and text
            try:
                clsname = det.get("class") or det.get("name") or ""
                if clsname:
                    # Use shared font/metrics to reduce allocations
                    conf = det.get("confidence")
                    if conf is not None:
                        txt = f"{clsname} {conf:.2f}"
                    else:
                        txt = str(clsname)
                    fm = fm_shared
                    if fm is not None:
                        try:
                            tw = fm.horizontalAdvance(txt)
                            th = fm.height()
                        except Exception:
                            tw = len(txt) * 6
                            th = 12
                    else:
                        tw = len(txt) * 6
                        th = 12
                    # position label slightly inset from bbox top-left
                    tx = rx1 + 2
                    ty = max(th, ry1 + th) - 4
                    # background rect (small padding)
                    painter.setBrush(text_bg)
                    painter.setPen(QColor(0, 0, 0, 160))
                    painter.drawRect(tx - 2, ty - th + 2, tw + 6, th)
                    # text
                    painter.setPen(text_fg)
                    painter.drawText(tx, ty - 2, txt)
                    # restore pen
                    painter.setPen(pen)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
            except Exception:
                logging.exception("paint_bboxes: failed to draw class label")
                # Continue with other bboxes even if one label fails
        except Exception:
            logging.exception("paint_bboxes: failed drawing bbox")
            # Continue with other bboxes even if one fails


def map_bbox_to_thumbnail(bbox, det_w, det_h, img_w, img_h, offset_x=0, offset_y=0):
    """Map detection bbox coordinates (absolute pixels in detection space) to thumbnail coordinates.

    Args:
        bbox: tuple/list of (x1, y1, x2, y2) in detection pixel coordinates.
        det_w, det_h: original detection image width/height.
        img_w, img_h: displayed image size inside thumbnail.
        offset_x, offset_y: top-left offset of the displayed image inside the thumbnail.

    Returns:
        (rx1, ry1, rx2, ry2) integer coordinates for drawing on the thumbnail pixmap.
    """
    x1, y1, x2, y2 = bbox
    sx = float(img_w) / float(det_w) if det_w else 1.0
    sy = float(img_h) / float(det_h) if det_h else 1.0
    logging.debug(
        "view_helpers: map_bbox_to_thumbnail bbox=%s det_w=%s det_h=%s img_w=%s img_h=%s offset=(%s,%s) sx=%s sy=%s",
        bbox,
        det_w,
        det_h,
        img_w,
        img_h,
        offset_x,
        offset_y,
        sx,
        sy,
    )
    rx1 = int(x1 * sx) + int(offset_x)
    ry1 = int(y1 * sy) + int(offset_y)
    rx2 = int(x2 * sx) + int(offset_x)
    ry2 = int(y2 * sy) + int(offset_y)
    
    logging.debug(
        f"map_bbox_to_thumbnail: bbox[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}] -> [{rx1},{ry1},{rx2},{ry2}] "
        f"(sx={sx:.3f}, sy={sy:.3f}, offset=({offset_x},{offset_y}))"
    )
    
    return rx1, ry1, rx2, ry2


__all__ = ["update_label_icon", "overlay_state_hash"]
