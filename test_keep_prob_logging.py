import logging
import os
from PIL import Image
import pytest

from PySide6.QtWidgets import QApplication

from photo_derush.image_grid import ImageGrid


class DummyInfoPanel:
    def update_info(self, *args, **kwargs):
        pass

class DummyStatusBar:
    def showMessage(self, msg):
        pass


def _ensure_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_keep_prob_logging_once(tmp_path, caplog):
    # Create two dummy images
    for name in ["a.jpg", "b.jpg"]:
        img = Image.new('RGB', (10, 10), color='red')
        img.save(tmp_path / name)

    def get_sorted_images():
        return ["a.jpg", "b.jpg"]

    app = _ensure_app()
    grid = ImageGrid(
        image_paths=["a.jpg", "b.jpg"],
        directory=str(tmp_path),
        info_panel=DummyInfoPanel(),
        status_bar=DummyStatusBar(),
        get_sorted_images=get_sorted_images,
        image_info={},
        on_open_fullscreen=None,
        on_select=None,
        labels_map={},
    )

    prob_map = {"a.jpg": 0.42, "b.jpg": 0.91}
    with caplog.at_level(logging.INFO):
        grid.update_keep_probabilities(prob_map)
        # Second update should not add new log lines for same images
        grid.update_keep_probabilities(prob_map)

    msgs = [r.message for r in caplog.records if r.message.startswith('[ImageGrid] Updated keep probability for')]
    # Expect exactly one log per image
    assert len(msgs) == 2, f"Expected 2 log messages (one per image), got {len(msgs)}: {msgs}"
    assert any('a.jpg' in m for m in msgs)
    assert any('b.jpg' in m for m in msgs)

    # Change probabilities and call again: still should not log duplicates
    with caplog.at_level(logging.INFO):
        grid.update_keep_probabilities({"a.jpg": 0.55, "b.jpg": 0.33})
    new_msgs = [r.message for r in caplog.records if r.message.startswith('[ImageGrid] Updated keep probability for')]
    assert len(new_msgs) == 2, f"Still should have only original 2 log messages, got {len(new_msgs)}"

    # Basic UI cycle to avoid warnings
    for _ in range(5):
        app.processEvents()

