import os
from src.app import FileScanner, ImageLoader
from PySide6.QtCore import QEventLoop, QCoreApplication
from PySide6.QtGui import QImage, QPixmap

def test_file_scanner(tmp_path):
    # Create dummy image files
    (tmp_path / "a.jpg").write_bytes(b"1")
    (tmp_path / "b.png").write_bytes(b"2")
    (tmp_path / "c.txt").write_bytes(b"3")
    found = []
    app = QCoreApplication([])
    loop = QEventLoop()
    def on_found(files):
        found.extend(files)
        loop.quit()
    scanner = FileScanner(str(tmp_path))
    scanner.files_found.connect(on_found)
    scanner.start()
    loop.exec()
    assert set(found) == {"a.jpg", "b.png"}

def test_image_loader(tmp_path):
    # Create a dummy image file
    img_path = tmp_path / "test.jpg"
    image = QImage(800, 600, QImage.Format_RGB32)
    image.fill(0xFF0000)  # Red
    image.save(str(img_path))
    app = QCoreApplication.instance() or QCoreApplication([])
    loop = QEventLoop()
    result = []
    def on_loaded(pixmap):
        result.append(pixmap)
        loop.quit()
    loader = ImageLoader(str(img_path))
    loader.image_loaded.connect(on_loaded)
    loader.start()
    loop.exec()
    assert result
    pixmap = result[0]
    assert isinstance(pixmap, QPixmap)
    assert not pixmap.isNull()
    # Should be scaled to 256x256 or less
    assert pixmap.width() <= 256 and pixmap.height() <= 256
