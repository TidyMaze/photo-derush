import os
from src.app import FileScanner, ImageLoader
from PySide6.QtCore import QEventLoop, QCoreApplication
from PySide6.QtGui import QImage, QPixmap
from PIL import Image as PILImage
import piexif

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
    def on_loaded(pixmap, info):
        result.append((pixmap, info))
        loop.quit()
    loader = ImageLoader(str(img_path))
    loader.image_loaded.connect(on_loaded)
    loader.start()
    loop.exec()
    assert result
    pixmap, info = result[0]
    assert isinstance(pixmap, QPixmap)
    assert not pixmap.isNull()
    # Should be scaled to 256x256 or less
    assert pixmap.width() <= 256 and pixmap.height() <= 256
    # Check file info
    assert info['width'] == 800
    assert info['height'] == 600
    assert info['size'] == os.stat(str(img_path)).st_size
    assert abs(info['mtime'] - os.stat(str(img_path)).st_mtime) < 2  # Allow 2s drift

def test_image_loader_with_exif(tmp_path):
    # Create a dummy JPEG with EXIF
    img_path = tmp_path / "exif.jpg"
    pil_img = PILImage.new("RGB", (100, 100), color="red")
    exif_dict = {
        "0th": {
            piexif.ImageIFD.Make: u"TestMake",
            piexif.ImageIFD.Model: u"TestModel",
        },
        "Exif": {
            piexif.ExifIFD.DateTimeOriginal: u"2025:08:23 12:34:56",
            piexif.ExifIFD.LensModel: u"TestLens",
            piexif.ExifIFD.ExposureTime: (1, 100),
            piexif.ExifIFD.FNumber: (28, 10),
            piexif.ExifIFD.ISOSpeedRatings: 200,
            piexif.ExifIFD.FocalLength: (50, 1),
        },
    }
    exif_bytes = piexif.dump(exif_dict)
    pil_img.save(str(img_path), exif=exif_bytes)
    app = QCoreApplication.instance() or QCoreApplication([])
    loop = QEventLoop()
    result = []
    def on_loaded(pixmap, info, exif):
        result.append((pixmap, info, exif))
        loop.quit()
    loader = ImageLoader(str(img_path))
    loader.image_loaded.connect(on_loaded)
    loader.start()
    loop.exec()
    assert result
    pixmap, info, exif = result[0]
    assert isinstance(pixmap, QPixmap)
    assert not pixmap.isNull()
    # Check EXIF fields
    assert exif.get('Make') == 'TestMake'
    assert exif.get('Model') == 'TestModel'
    assert exif.get('DateTimeOriginal') == '2025:08:23 12:34:56'
    assert exif.get('LensModel') == 'TestLens'
