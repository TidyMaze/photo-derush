import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.app import ImageLoader, ExifLoaderWorker
from PySide6.QtCore import QEventLoop, QCoreApplication, QTimer
from PySide6.QtGui import QImage, QPixmap
from PIL import Image as PILImage
import piexif
import numpy as np
from PySide6.QtWidgets import QApplication

# def test_file_scanner(tmp_path):
#     # Create dummy image files
#     (tmp_path / "a.jpg").write_bytes(b"1")
#     (tmp_path / "b.png").write_bytes(b"2")
#     (tmp_path / "c.txt").write_bytes(b"3")
#     found = []
#     app = QCoreApplication([])
#     loop = QEventLoop()
#     def on_found(files):
#         found.extend(files)
#         loop.quit()
#     scanner = FileScanner(str(tmp_path))
#     scanner.files_found.connect(on_found)
#     scanner.start()
#     loop.exec()
#     assert set(found) == {"a.jpg", "b.png"}

def test_image_loader(tmp_path):
    # Create a dummy image file
    img_path = tmp_path / "test.jpg"
    image = QImage(800, 600, QImage.Format_RGB32)
    image.fill(0xFF0000)  # Red
    image.save(str(img_path))
    app = QApplication.instance() or QApplication([])
    loop = QEventLoop()
    result = []
    def on_loaded(pixmap, info, exif):
        result.append((pixmap, info, exif))
        loop.quit()
    loader = ImageLoader(str(img_path))
    loader.image_loaded.connect(on_loaded)
    QTimer.singleShot(0, loader.start)
    loop.exec()
    assert result
    pixmap, info, exif = result[0]
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
    app = QApplication.instance() or QApplication([])
    loop = QEventLoop()
    result = []
    def on_loaded(pixmap, info, exif):
        result.append((pixmap, info, exif))
        loop.quit()
    loader = ImageLoader(str(img_path))
    loader.image_loaded.connect(on_loaded)
    QTimer.singleShot(0, loader.start)
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

def test_exif_loader_worker(tmp_path):
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication
    from PIL import Image as PILImage
    import piexif
    import numpy as np
    # Create a dummy image with EXIF
    img_path = tmp_path / "exif_test.jpg"
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    pil_img = PILImage.fromarray(arr)
    exif_dict = {"0th": {piexif.ImageIFD.Make: u"TestMake"}}
    exif_bytes = piexif.dump(exif_dict)
    pil_img.save(str(img_path), exif=exif_bytes)
    app = QApplication.instance() or QApplication([])
    loop = QEventLoop()
    result = {}
    def on_loaded(path, exif):
        result["path"] = path
        result["exif"] = exif
        print(f"on_loaded called: {path}, exif: {exif}")
        loop.quit()
    def on_timeout():
        print("Timeout reached in test_exif_loader_worker")
        loop.quit()
    worker = ExifLoaderWorker(str(img_path))
    worker.exif_loaded.connect(on_loaded)
    QTimer.singleShot(0, worker.load_exif)
    QTimer.singleShot(3000, on_timeout)  # 3 second timeout
    loop.exec()
    assert "path" in result, "Signal was not emitted in time"
    assert result["path"] == str(img_path)
    assert "Make" in result["exif"]
    assert result["exif"]["Make"] == "TestMake"
