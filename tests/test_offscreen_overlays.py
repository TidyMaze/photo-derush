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
    # Create mock images in tmp_path
    for i in range(3):
        img_path = tmp_path / f"test_{i}.jpg"
        img = Image.new('RGB', (100, 100), color='red')
        img.save(img_path)

    vm = PhotoViewModel(str(tmp_path))
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

    def has_nontransparent(p: Path):
        im = Image.open(p).convert('RGBA')
        import numpy as np
        arr = np.array(im)
        return np.any(arr[:, :, 3] > 0)

    assert any(has_nontransparent(p) for p in saved), "All saved thumbnails are fully transparent"


def test_show_hide_detected_objects(tmp_path):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    from src.viewmodel import PhotoViewModel
    from src.view import PhotoView

    app = QApplication.instance() or QApplication([])
    vm = PhotoViewModel(str(tmp_path))
    # mock some images
    vm.images = ["test0.jpg", "test1.jpg"]
    # Mock detected objects
    vm._detected_objects = {
        "test0.jpg": [{"class": "person", "confidence": 0.8, "bbox": [0.1, 0.1, 0.5, 0.5]}],
        "test1.jpg": []
    }
    
    view = PhotoView(vm, thumb_size=106, images_per_row=6)
    
    # Assert checkbox exists
    assert hasattr(view, "show_detected_objects_checkbox")
    assert view.show_detected_objects_checkbox.isChecked() is True
    
    # Trigger refresh
    view._refresh_thumbnail_badges()
    app.processEvents()
    
    # Toggle off
    view.show_detected_objects_checkbox.setChecked(False)
    view._refresh_thumbnail_badges()
    app.processEvents()
    
    # Check that isChecked is False
    assert view.show_detected_objects_checkbox.isChecked() is False
