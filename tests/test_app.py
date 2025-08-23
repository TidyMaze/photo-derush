import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PySide6.QtCore import QEventLoop, QCoreApplication, QTimer
from PySide6.QtGui import QImage, QPixmap
from PIL import Image as PILImage
import piexif
import numpy as np
from PySide6.QtWidgets import QApplication

# NOTE: The following tests referenced obsolete classes (ImageLoader, ExifLoaderWorker) and have been removed.
# To test the current architecture, add tests for PhotoViewModel, ImageLoaderWorker, or other active classes.
