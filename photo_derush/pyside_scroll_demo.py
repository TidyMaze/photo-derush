"""PySide6 scroll pane demo.

Provides a simple vertically scrollable list of placeholder labels.
Run this module directly to see the UI, or import `create_scroll_window`.
"""
from __future__ import annotations
import os
import sys
import argparse
import logging
from typing import List, Optional, Tuple, Iterable
from PySide6.QtWidgets import (
    QApplication, QWidget, QScrollArea, QVBoxLayout, QLabel, QMainWindow, QStatusBar
)
from PySide6.QtCore import QTimer, Qt

logger = logging.getLogger(__name__)

# ----------------------------- Core Factory ----------------------------------

def create_scroll_window(items: Optional[List[str]] = None, num_items: int = 50,
                         auto_close_ms: Optional[int] = None,
                         incremental: bool = False,
                         batch_size: int = 250,
                         batch_interval_ms: int = 10) -> Tuple[QApplication, QMainWindow]:
    """Create a main window containing a scroll area filled with labels.

    Parameters
    ----------
    items: list[str] | None
        If None, dummy items are generated (num_items).
    num_items: int
        Count of dummy items if items not provided.
    auto_close_ms: int | None
        If set, window closes automatically after this many ms.
    incremental: bool
        If True, populate labels in timed batches (keeps UI responsive for huge lists).
    batch_size: int
        Number of labels added per batch when incremental.
    batch_interval_ms: int
        Delay between batches when incremental.

    Returns
    -------
    (app, window)
    """
    # Only force offscreen in automated contexts (tests/CI) to allow visible window in normal usage
    if auto_close_ms is not None or os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    # Lightweight logging setup (idempotent)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    app = QApplication.instance() or QApplication([])

    window = QMainWindow()
    window.setWindowTitle("PySide6 Scroll Pane Demo")
    status = QStatusBar()
    window.setStatusBar(status)
    status.showMessage("Initializing…")

    central = QWidget()
    window.setCentralWidget(central)

    layout = QVBoxLayout(central)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    layout.addWidget(scroll)

    content = QWidget()
    content_layout = QVBoxLayout(content)

    if items is None:
        items = [f"Item {i+1}" for i in range(num_items)]

    def _add_label(text: str):
        lbl = QLabel(text)
        lbl.setMinimumHeight(24)
        # Allow selecting text for debugging / copy
        lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        content_layout.addWidget(lbl)

    if incremental and len(items) > batch_size:
        logger.info("Using incremental population: %d items in batches of %d", len(items), batch_size)
        iterator: Iterable[str] = iter(items)
        remaining = len(items)

        def add_batch():
            nonlocal remaining
            added = 0
            while added < batch_size:
                try:
                    _add_label(next(iterator))
                except StopIteration:
                    remaining = 0
                    break
                added += 1
                remaining -= 1
            status.showMessage(f"Loaded {len(items)-remaining}/{len(items)} items")
            if remaining > 0:
                QTimer.singleShot(batch_interval_ms, add_batch)
            else:
                status.showMessage(f"Loaded all {len(items)} items")
        QTimer.singleShot(0, add_batch)
    else:
        logger.info("Populating %d items synchronously", len(items))
        for text in items:
            _add_label(text)
        status.showMessage(f"Loaded {len(items)} items")

    scroll.setWidget(content)

    # Keyboard shortcuts for user-friendly exit
    def _key_close(event):
        if event.key() in (Qt.Key_Escape, Qt.Key_Q):
            window.close()
        else:
            return QWidget.keyPressEvent(central, event)  # type: ignore
    central.keyPressEvent = _key_close  # type: ignore

    if auto_close_ms is not None:
        logger.info("Auto-close in %d ms", auto_close_ms)
        QTimer.singleShot(auto_close_ms, window.close)
    else:
        # Safety fallback for automated runs: environment variable
        fallback = os.environ.get("AUTO_CLOSE_FALLBACK_MS")
        if fallback and fallback.isdigit():
            ms = int(fallback)
            logger.info("Safety fallback auto-close in %d ms (AUTO_CLOSE_FALLBACK_MS)", ms)
            QTimer.singleShot(ms, window.close)

    window.resize(500, 400)
    return app, window

# ----------------------------- CLI Runner ------------------------------------

def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PySide6 Scroll Area Demo")
    p.add_argument('-n', '--num-items', type=int, default=200, help='Number of dummy items')
    p.add_argument('--incremental', action='store_true', help='Populate items in batches')
    p.add_argument('--batch-size', type=int, default=500, help='Batch size for incremental mode')
    p.add_argument('--batch-interval', type=int, default=10, help='Interval (ms) between batches')
    p.add_argument('--auto-close', type=int, default=None, help='Auto close after N ms')
    p.add_argument('--offscreen', action='store_true', help='Force offscreen platform')
    return p.parse_args(argv)

def run_demo(argv: Optional[List[str]] = None):  # pragma: no cover - manual invocation
    args = _parse_args(argv or sys.argv[1:])
    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app, window = create_scroll_window(num_items=args.num_items,
                                       auto_close_ms=args.auto_close,
                                       incremental=args.incremental,
                                       batch_size=args.batch_size,
                                       batch_interval_ms=args.batch_interval)
    window.show()
    logger.info("Demo running. Press ESC or Q to quit.")
    sys.exit(app.exec())

# ------------------------------ Main Guard -----------------------------------
if __name__ == "__main__":  # pragma: no cover
    run_demo()
