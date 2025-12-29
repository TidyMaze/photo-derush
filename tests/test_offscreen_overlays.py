import os
import time
from pathlib import Path
from PIL import Image

import pytest


def test_offscreen_overlays(tmp_path):
    # Run app code in offscreen mode to produce thumbnail debug images
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    from src.viewmodel import PhotoViewModel
    from src.view import PhotoView

    app = QApplication.instance() or QApplication([])
    # use repo root as directory (contains test images)
    repo_root = Path(__file__).resolve().parents[1]
    vm = PhotoViewModel(str(repo_root))
    view = PhotoView(vm, thumb_size=106, images_per_row=6)

    # load images and give event loop a bit of time
    vm.load_images()
    # process events and allow thumbnails to load
    for _ in range(30):
        app.processEvents()
        time.sleep(0.05)

    outdir = Path('.cache') / 'debug_overlays'
    outdir.mkdir(parents=True, exist_ok=True)

    saved = []
    # grab first few label pixmaps and save
    for idx, ((row, col), lbl) in enumerate(view.label_refs.items()):
        if idx >= 3:
            break
        try:
            base = getattr(lbl, 'base_pixmap', None)
            orig = getattr(lbl, 'original_pixmap', None)
            logical = getattr(lbl, '_logical_pixmap', None)
            if orig:
                orig.save(str(outdir / f"thumb_test_{idx}_orig.png"))
                saved.append(outdir / f"thumb_test_{idx}_orig.png")
            if base:
                base.save(str(outdir / f"thumb_test_{idx}_base.png"))
                saved.append(outdir / f"thumb_test_{idx}_base.png")
            if logical:
                logical.save(str(outdir / f"thumb_test_{idx}_logical.png"))
                saved.append(outdir / f"thumb_test_{idx}_logical.png")
        except Exception:
            # test must continue even if saving fails for one
            continue

    assert saved, "No thumbnail images were saved during offscreen run"

    # Quick pixel checks: ensure at least one saved PNG has non-transparent pixels
    def has_nontransparent(p: Path):
        im = Image.open(p).convert('RGBA')
        px = im.getdata()
        return any(a > 0 for (_, _, _, a) in px)

    assert any(has_nontransparent(p) for p in saved), "All saved thumbnails are fully transparent"
