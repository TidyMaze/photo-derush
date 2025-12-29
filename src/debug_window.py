from __future__ import annotations

import json
import logging
import os

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

try:
    import pandas as pd
except Exception:
    logging.exception("DebugWindow: failed to import pandas; falling back to minimal table model")
    pd = None

from src.detections import normalize_detections
from src.view_helpers import paint_bboxes


class PandasModel(QAbstractTableModel):
    def __init__(self, df):
        super().__init__()
        self._df = df if df is not None else None

    def rowCount(self, parent=QModelIndex()):
        return 0 if self._df is None else len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return 0 if self._df is None else len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or self._df is None:
            return None
        if role in (Qt.DisplayRole, Qt.EditRole):
            val = self._df.iat[index.row(), index.column()]
            return str(val)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if self._df is None or role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        return str(self._df.index[section])


class DebugWindow(QWidget):
    def __init__(self, viewmodel, parent=None):
        super().__init__(parent)
        logging.info("DebugWindow: __init__ start")
        self.setWindowTitle("Debug: Detection Data")
        self.viewmodel = viewmodel
        # store latest browser snapshot when emitted by viewmodel
        self._last_browser_state = None
        try:
            if hasattr(self.viewmodel, "browser_state_changed"):
                self.viewmodel.browser_state_changed.connect(self._on_browser_state_changed)
        except Exception:
            logging.exception("DebugWindow: failed to connect browser_state_changed")
        # Make the window obvious during debug runs
        try:
            from PySide6.QtCore import Qt as _Qt

            self.setWindowFlags(self.windowFlags() | _Qt.WindowStaysOnTopHint)
        except Exception:
            logging.exception("DebugWindow: failed to set WindowStaysOnTopHint")
        # Make window modal and try to raise/activate so it is visible during debug runs
        try:
            self.setWindowModality(Qt.ApplicationModal)
            self.show()
            logging.info("DebugWindow: show() called")
            self.raise_()
            logging.info("DebugWindow: raise_() called")
            try:
                self.activateWindow()
                logging.info("DebugWindow: activateWindow() called")
            except Exception:
                logging.exception("DebugWindow: activateWindow() failed")
        except Exception:
            logging.exception("DebugWindow: failed during initial show/activate")

        self.split = QSplitter()

        self.table = QTableView()
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.SingleSelection)

        right = QWidget()
        rv = QVBoxLayout(right)
        self.preview = QLabel()
        self.preview.setMinimumSize(320, 240)
        self.preview.setStyleSheet("background: #222;")
        self.preview.setScaledContents(True)
        rv.addWidget(self.preview)

        self.raw = QPlainTextEdit()
        self.raw.setReadOnly(True)
        rv.addWidget(self.raw)

        btns = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_data)
        btns.addWidget(self.refresh_btn)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        btns.addWidget(self.close_btn)
        rv.addLayout(btns)

        self.split.addWidget(self.table)
        self.split.addWidget(right)

        layout = QVBoxLayout(self)
        layout.addWidget(self.split)

        self._df = None
        self._model = None
        # Keep track of the selectionModel we've connected to so we don't
        # repeatedly attach the handler on every refresh (QTimer runs).
        self._connected_selection_model = None

        # Connect selectionChanged after model is assigned in refresh_data()
        # (connecting here can fail because selectionModel() is None until a model is set)

        # Prefer signal-driven refresh to avoid polling and repeated model swaps.
        # Initial population will happen when viewmodel emits the first browser_state_changed
        # snapshot; call refresh_data once now to populate immediately.
        try:
            self.refresh_data()
        except Exception:
            logging.exception("DebugWindow: initial refresh_data failed")
        logging.info("DebugWindow: __init__ done")

    def closeEvent(self, event):
        logging.info("DebugWindow: closeEvent")
        # Clean up selectionModel connection to avoid dangling handlers
        try:
            prev = getattr(self, "_connected_selection_model", None)
            if prev is not None:
                try:
                    prev.selectionChanged.disconnect(self._on_selection_changed)
                except Exception:
                    logging.exception("DebugWindow: failed to disconnect previous selectionModel")
                self._connected_selection_model = None
        except Exception:
            logging.exception("DebugWindow: closeEvent cleanup failed")
        return super().closeEvent(event)

    def refresh_data(self):
        try:
            # Use the latest browser state snapshot supplied by the viewmodel.
            # If the DebugWindow didn't receive the browser_state_changed signal
            # (signal connect may fail in some environments), fall back to the
            # viewmodel's internal cache so UI still shows detections.
            detected = {}
            if getattr(self, "_last_browser_state", None):
                detected = getattr(self._last_browser_state, "detected_objects", {}) or {}
            else:
                detected = getattr(self.viewmodel, "_detected_objects", {}) or {}
            imgs = list(self.viewmodel.images) if hasattr(self.viewmodel, "images") else []
            rows = []
            for fname in imgs:
                base = os.path.basename(fname)
                dets = detected.get(base, []) or []
                det_objs = normalize_detections(dets) if dets else []
                bbox_count = sum(1 for d in det_objs if getattr(d, "bbox", None) is not None)
                # Compute largest bbox (width x height) among detections for quick inspection
                max_w = 0
                max_h = 0
                for d in det_objs:
                    try:
                        bbox = getattr(d, "bbox", None)
                        if not bbox:
                            continue
                        x1, y1, x2, y2 = bbox
                        w = abs(float(x2) - float(x1))
                        h = abs(float(y2) - float(y1))
                        if w * h > max_w * max_h:
                            max_w, max_h = int(w), int(h)
                    except Exception:
                        logging.exception(f"DebugWindow: failed computing largest bbox for {base}")
                        continue
                largest = f"{max_w}x{max_h}" if max_w and max_h else ""
                rows.append(
                    {
                        "filename": base,
                        "detected_objects": len(det_objs),
                        "bbox_count": bbox_count,
                        "largest_bbox": largest,
                        "raw": json.dumps(dets),
                    }
                )

            if pd is not None:
                df = pd.DataFrame(rows)
            else:
                # fallback: build a minimal object that PandasModel can handle
                try:
                    import pandas as _pd

                    df = _pd.DataFrame(rows)
                except Exception:
                    # simple list-of-dicts to DataFrame-like structure
                    class SimpleIat:
                        def __init__(self, rows, columns):
                            self._rows = rows
                            self._cols = columns

                        def __getitem__(self, idx):
                            r, c = idx
                            return self._rows[r][self._cols[c]]

                    class SimpleDF:
                        def __init__(self, rows):
                            self._rows = rows
                            self.columns = ["filename", "detected_objects", "bbox_count", "largest_bbox", "raw"]
                            self.iat = SimpleIat(self._rows, self.columns)

                        def __len__(self):
                            return len(self._rows)

                        def __getitem__(self, key):
                            return [r[key] for r in self._rows]

                    df = SimpleDF(rows)

            self._df = df
            self._model = PandasModel(df)
            self.table.setModel(self._model)
            self.table.resizeColumnsToContents()
            # connect selection change handler safely after model is set
            try:
                sm = self.table.selectionModel()
                if sm is not None:
                    # If we've already connected to this selection model, do nothing.
                    prev = getattr(self, "_connected_selection_model", None)
                    if prev is sm:
                        # already connected
                        pass
                    else:
                        # disconnect handler from previous selectionModel if possible
                        try:
                            if prev is not None:
                                prev.selectionChanged.disconnect(self._on_selection_changed)
                        except Exception:
                            logging.exception("DebugWindow: failed to disconnect previous selectionModel (ignored)")
                        # connect to the current selectionModel and remember it
                        try:
                            sm.selectionChanged.connect(self._on_selection_changed)
                            self._connected_selection_model = sm
                            logging.info("DebugWindow: connected table.selectionModel.selectionChanged")
                        except Exception:
                            logging.exception("Failed to connect selectionChanged on new selectionModel")
            except Exception:
                logging.exception("Failed to connect table selectionModel")
        except Exception:
            logging.exception("DebugWindow.refresh_data failed")

    def _on_browser_state_changed(self, state):
        try:
            self._last_browser_state = state
            logging.info("DebugWindow: received browser_state_changed")
            # Signal-driven update: refresh the table/view based on latest snapshot.
            try:
                self.refresh_data()
            except Exception:
                logging.exception("DebugWindow: refresh_data failed during browser_state_changed")
        except Exception:
            logging.exception("DebugWindow: failed handling browser_state_changed")

    def _on_selection_changed(self, selected, deselected):
        try:
            logging.info("DebugWindow: _on_selection_changed fired")
            idxs = self.table.selectionModel().selectedRows()
            if not idxs:
                return
            row = idxs[0].row()
            # read filename from model
            if self._df is None:
                return
            try:
                fname = self._df.iat[row, 0] if hasattr(self._df, "iat") else self._df._rows[row]["filename"]
            except Exception:
                logging.exception("DebugWindow: failed to read filename via iat, attempting column access")
                # try column access
                try:
                    fname = self._df["filename"][row]
                except Exception:
                    logging.exception("DebugWindow: failed to read filename via column access")
                    return

            # Show raw detections (prefer latest browser snapshot, fall back to
            # the viewmodel's internal detected_objects cache if needed).
            detected = {}
            if getattr(self, "_last_browser_state", None):
                detected = getattr(self._last_browser_state, "detected_objects", {}) or {}
            else:
                detected = getattr(self.viewmodel, "_detected_objects", {}) or {}
            raw = detected.get(fname, [])
            try:
                logging.info(f"DebugWindow: selection {fname!r} raw_count={len(raw)}")
            except Exception:
                logging.info("DebugWindow: selection (logging failed to format)")
            self.raw.setPlainText(json.dumps(raw, indent=2))

            # Show preview with boxes
            path = self.viewmodel.model.get_image_path(fname)
            if isinstance(path, (list, tuple)):
                path = path[0] if path else None
            if not path or not os.path.exists(path):
                self.preview.setPixmap(QPixmap())
                return
            pix = QPixmap(path)
            if pix.isNull():
                self.preview.setPixmap(QPixmap())
                return
            # keep the original pix dimensions for correct scaling
            orig_w = pix.width()
            orig_h = pix.height()
            disp = pix.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            try:
                logging.info(f"DebugWindow: preview sizes orig={orig_w}x{orig_h} disp={disp.width()}x{disp.height()}")
            except Exception:
                logging.info("DebugWindow: preview sizes (logging failed)")
            # draw boxes onto the scaled pixmap using shared painter helper
            painter = QPainter(disp)
            try:
                dets = raw or []
                det_objs = normalize_detections(dets) if dets else []
                # convert normalized objects back to mapping-like dicts expected by paint_bboxes
                det_dicts = []
                for d in det_objs:
                    if getattr(d, "bbox", None) is None:
                        continue
                    # Use original image dimensions for det_w/det_h if not set in detection
                    # This ensures bbox coordinates are correctly mapped from original to scaled display
                    det_w = d.det_w if d.det_w is not None else orig_w
                    det_h = d.det_h if d.det_h is not None else orig_h
                    det_dicts.append(
                        {
                            "bbox": list(d.bbox),
                            "det_w": det_w,
                            "det_h": det_h,
                            "class": getattr(d, "name", None) or getattr(d, "class", None) or None,
                        }
                    )
                try:
                    # When preview is scaled, img_w/img_h are disp.width/height (scaled display size)
                    # and offset is 0. det_w/det_h in each dict should be orig_w/orig_h (original image size)
                    paint_bboxes(painter, det_dicts, 0, 0, disp.width(), disp.height(), thumb_label=None)
                except Exception:
                    logging.exception("DebugWindow: paint_bboxes failed")
            finally:
                painter.end()
            self.preview.setPixmap(disp)
        except Exception:
            logging.exception("DebugWindow selection handler failed")


__all__ = ["DebugWindow"]
