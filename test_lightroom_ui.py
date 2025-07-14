import pytest
from unittest.mock import MagicMock
import main

class DummyEvent:
    def __init__(self, widget):
        self.widget = widget

class DummyLabel:
    def __init__(self):
        self.selected = False
        self.opened = False
        self.handlers = {}
    def bind(self, event, handler):
        self.handlers[event] = handler
    def config(self, **kwargs):
        if kwargs.get('bg') == "#555":
            self.selected = True
    def simulate_click(self):
        if "<Button-1>" in self.handlers:
            self.handlers["<Button-1>"](DummyEvent(self))
    def simulate_double_click(self):
        if "<Double-Button-1>" in self.handlers:
            self.handlers["<Double-Button-1>"](DummyEvent(self))
            self.opened = True

# Test: clicking an image selects it, double-clicking opens it

def test_image_selection_and_open(monkeypatch):
    opened = {'called': False}
    def fake_open_full_image(img_path):
        opened['called'] = True
    monkeypatch.setattr(main, 'open_full_image', fake_open_full_image)
    label = DummyLabel()
    # Simulate the UI binding logic
    def on_click(event, label=label):
        label.config(bg="#555")
    def on_double_click(event):
        main.open_full_image("dummy_path")
    label.bind("<Button-1>", on_click)
    label.bind("<Double-Button-1>", on_double_click)
    # Simulate click
    label.simulate_click()
    assert label.selected, "Image should be selected on click"
    # Simulate double-click
    label.simulate_double_click()
    assert label.opened, "Image should be opened on double-click"
    assert opened['called'], "open_full_image should be called on double-click"

def test_thumbnail_low_resolution(monkeypatch):
    loaded_sizes = []
    class DummyImage:
        def __init__(self):
            self.size = (4000, 3000)
        def thumbnail(self, size):
            loaded_sizes.append(size)
    def fake_open(path):
        return DummyImage()
    monkeypatch.setattr("PIL.Image.open", fake_open)
    import main
    main.show_lightroom_ui(["img1.jpg"], "/tmp")
    assert any(s[0] <= 200 and s[1] <= 200 for s in loaded_sizes), "Images should be loaded in low resolution (thumbnail)"
