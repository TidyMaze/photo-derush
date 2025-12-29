#!/usr/bin/env python3
"""Debug window - clean start."""
import os
import sys
import logging

# Set up environment before importing Qt
os.environ.setdefault('QT_LOGGING_RULES', 'qt.*=false')
os.environ.setdefault('DETECTION_WORKER', '0')

from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PySide6.QtGui import QColor, QPixmap, QPainter, QPen
from PySide6.QtCore import Qt
from PIL import Image
from PIL.ImageQt import ImageQt
from src.bbox_overlay_widget import BoundingBoxOverlayWidget

LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s %(levelname)s:%(name)s:%(message)s'
)


class RedBorderOverlayWidget(QWidget):
    """Overlay widget that draws a red border rectangle around the image."""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.hide()  # Hide initially, will be shown when positioned
    
    def paintEvent(self, event):
        """Draw red border rectangle."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = self.rect()
        
        # Red border wrapping entire image, transparent fill
        red_pen = QPen(QColor(255, 0, 0), 2)
        painter.setPen(red_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect.adjusted(1, 1, -1, -1))


class DebugCellWindow(QMainWindow):
    """Clean debug window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Debug Cell - Clean")
        
        # Allow window to be resized smaller
        self.setMinimumSize(100, 100)
        
        # Create central widget with blue background
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Blue panel taking entire space
        blue_panel = QWidget()
        blue_panel.setStyleSheet("background-color: blue;")
        blue_layout = QVBoxLayout(blue_panel)
        blue_layout.setContentsMargins(0, 0, 0, 0)
        
        # Red panel centered (will contain image)
        self.red_panel = QWidget()
        self.red_panel.setStyleSheet("background-color: red;")
        red_layout = QVBoxLayout(self.red_panel)
        red_layout.setContentsMargins(0, 0, 0, 0)
        
        # Image label inside red panel
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setScaledContents(False)  # Don't scale, maintain aspect ratio
        red_layout.addWidget(self.image_label)
        
        # Red border overlay
        self.red_border_overlay = RedBorderOverlayWidget(self.image_label)
        
        # Bounding box overlay (reusable component)
        self.bbox_overlay = BoundingBoxOverlayWidget(self.image_label)
        
        blue_layout.addWidget(self.red_panel, alignment=Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(blue_panel)
        
        # Set initial size (non-square window)
        self.resize(800, 600)
        
        # Image aspect ratio (will be set when image loads)
        self.image_aspect_ratio = None
        
        # Load image from dataset
        self._load_image()
    
    def resizeEvent(self, event):
        """Update red panel size to maintain 4:3 aspect ratio."""
        super().resizeEvent(event)
        self._update_red_panel_size()
    
    def _update_red_panel_size(self):
        """Calculate and set red panel size to match image aspect ratio."""
        if self.image_aspect_ratio is None:
            return
        
        window_w = self.width()
        window_h = self.height()
        
        # Calculate size if we use full width
        width_from_width = window_w
        height_from_width = int(window_w / self.image_aspect_ratio)
        
        # Calculate size if we use full height
        width_from_height = int(window_h * self.image_aspect_ratio)
        height_from_height = window_h
        
        # Use whichever fits within the window
        if height_from_width <= window_h:
            # Use full width
            panel_w = width_from_width
            panel_h = height_from_width
        else:
            # Use full height
            panel_w = width_from_height
            panel_h = height_from_height
        
        self.red_panel.setFixedSize(panel_w, panel_h)
        # Update image size when panel resizes
        self._update_image_size()
    
    def _load_image(self):
        """Load an image from the dataset that has objects."""
        try:
            # Load dataset folder from config
            CONFIG_PATH = os.path.expanduser('~/.photo_app_config.json')
            dataset_dir = None
            try:
                import json
                if os.path.exists(CONFIG_PATH):
                    with open(CONFIG_PATH) as f:
                        data = json.load(f)
                        dataset_dir = data.get('last_dir')
            except Exception:
                pass
            
            if not dataset_dir or not os.path.isdir(dataset_dir):
                dataset_dir = os.path.expanduser('~')
            
            # Find images with objects
            from src.model import ImageModel
            from src import object_detection
            
            model = ImageModel(dataset_dir)
            images = model.get_image_files()
            cache = object_detection.load_object_cache()
            
            # Find an image with person detections
            image_path = None
            person_classes = {"person"}
            candidates_with_persons = []
            candidates_with_objects = []
            
            for fname in images:
                full_path = model.get_image_path(fname)
                basename = os.path.basename(full_path) if full_path else fname
                
                if basename in cache and cache[basename]:
                    detections = cache[basename]
                    classes = [d.get("class", "").lower() for d in detections]
                    has_person = any(c in person_classes for c in classes)
                    
                    if has_person:
                        candidates_with_persons.append((full_path or os.path.join(dataset_dir, fname), basename, detections, classes))
                    else:
                        candidates_with_objects.append((full_path or os.path.join(dataset_dir, fname), basename, detections, classes))
            
            # Prefer images with persons, but skip the first one (likely the tower)
            if candidates_with_persons:
                if len(candidates_with_persons) > 1:
                    # Skip first, use second
                    image_path, basename, detections, classes = candidates_with_persons[1]
                else:
                    image_path, basename, detections, classes = candidates_with_persons[0]
                person_count = sum(1 for c in classes if c in person_classes)
                logging.info(f"Selected image with {person_count} person(s) and {len(detections)} total objects: {basename}")
            elif candidates_with_objects:
                # Skip first image (likely tower), use second or later
                if len(candidates_with_objects) > 1:
                    image_path, basename, detections, classes = candidates_with_objects[1]
                    logging.info(f"Selected image with {len(detections)} objects (no person, skipped first): {basename}")
                elif len(candidates_with_objects) == 1:
                    # Only one candidate, but try to find a different image from the list
                    # Look for an image that's not in cache but exists
                    for fname in images[1:]:  # Skip first image
                        full_path = model.get_image_path(fname)
                        test_path = full_path or os.path.join(dataset_dir, fname)
                        if os.path.exists(test_path):
                            image_path = test_path
                            logging.info(f"Selected different image (not in cache): {os.path.basename(image_path)}")
                            break
                    if not image_path:
                        # Fallback to the only candidate
                        image_path, basename, detections, classes = candidates_with_objects[0]
                        logging.info(f"Selected image with {len(detections)} objects (no person, only candidate): {basename}")
            
            # If still no image, try images 4 or 5 which have detections
            if not image_path:
                # Try image 4 (index 3) which has person detections
                if len(images) > 3:
                    fname = images[3]
                    image_path = model.get_image_path(fname) or os.path.join(dataset_dir, fname)
                    logging.info(f"Using image 4 (has detections): {os.path.basename(image_path)}")
                elif len(images) > 1:
                    fname = images[1]
                    image_path = model.get_image_path(fname) or os.path.join(dataset_dir, fname)
                    logging.info(f"Using second image: {os.path.basename(image_path)}")
                elif images:
                    fname = images[0]
                    image_path = model.get_image_path(fname) or os.path.join(dataset_dir, fname)
                    logging.info(f"Using first image (only option): {os.path.basename(image_path)}")
            
            if image_path and os.path.exists(image_path):
                self._display_image(image_path)
        except Exception:
            logging.exception("Failed to load image")
    
    def _display_image(self, image_path):
        """Display image in the red panel, maintaining aspect ratio."""
        try:
            # Load image with PIL
            pil_img = Image.open(image_path)
            pil_img = pil_img.convert("RGBA")
            
            # Store original image and calculate aspect ratio
            self.original_image = pil_img
            img_w, img_h = pil_img.size
            self.image_aspect_ratio = img_w / img_h
            
            # Load detections for this image
            from src import object_detection
            basename = os.path.basename(image_path)
            cache = object_detection.load_object_cache()
            detections = cache.get(basename, [])
            
            # If not in cache or detections don't have bboxes, run detection
            if not detections or not any(d.get("bbox") for d in detections):
                logging.info(f"Running detection for {basename} (not in cache or missing bboxes)")
                detections = object_detection.detect_objects(image_path)
            
            # Store detections in reusable bounding box overlay component
            self.bbox_overlay.set_detections(detections, original_image_size=(img_w, img_h))
            
            logging.info(f"Loaded {len(detections)} detections for {basename}")
            if detections:
                for det in detections:
                    logging.info(f"  Detection: {det.get('class')} {det.get('confidence'):.2f} bbox={det.get('bbox')}")
            
            # Update red panel size to match image ratio
            self._update_red_panel_size()
            
            # Update image display
            self._update_image_size()
        except Exception:
            logging.exception(f"Failed to display image: {image_path}")
    
    def _update_image_size(self):
        """Update image size to fit in red panel while maintaining aspect ratio."""
        if not hasattr(self, 'original_image'):
            return
        
        try:
            panel_w = self.red_panel.width()
            panel_h = self.red_panel.height()
            
            if panel_w <= 0 or panel_h <= 0:
                return
            
            img_w, img_h = self.original_image.size
            img_ratio = img_w / img_h
            panel_ratio = panel_w / panel_h
            
            # Calculate size to fit in panel while maintaining aspect ratio
            if img_ratio > panel_ratio:
                # Image is wider - fit to width
                display_w = panel_w
                display_h = int(panel_w / img_ratio)
            else:
                # Image is taller - fit to height
                display_h = panel_h
                display_w = int(panel_h * img_ratio)
            
            # Resize image - use BILINEAR for faster resizing (good enough for display)
            resized = self.original_image.resize((display_w, display_h), Image.Resampling.BILINEAR)
            
            # Convert to QPixmap
            qimg = ImageQt(resized)
            pixmap = QPixmap.fromImage(qimg)
            
            # Set on label
            self.image_label.setPixmap(pixmap)
            
            # Update overlay to match image size
            self._update_overlay()
        except Exception:
            logging.exception("Failed to update image size")
    
    def _update_overlay(self):
        """Update overlay widget to match image size and position."""
        if not hasattr(self, 'image_label') or not self.image_label.pixmap():
            return
        
        try:
            pixmap = self.image_label.pixmap()
            if pixmap.isNull():
                return
            
            # Get image size
            img_w = pixmap.width()
            img_h = pixmap.height()
            
            # Get label geometry
            label_rect = self.image_label.geometry()
            
            # Calculate image position (centered in label)
            img_x = (label_rect.width() - img_w) // 2
            img_y = (label_rect.height() - img_h) // 2
            
            # Position red border overlay to match image
            self.red_border_overlay.setGeometry(img_x, img_y, img_w, img_h)
            self.red_border_overlay.show()
            self.red_border_overlay.raise_()
            
            # Position bounding box overlay to match image (reusable component)
            self.bbox_overlay.setGeometry(img_x, img_y, img_w, img_h)
            self.bbox_overlay.show()
            self.bbox_overlay.raise_()
        except Exception:
            logging.exception("Failed to update overlay")
    
    def showEvent(self, event):
        """Center window when shown."""
        super().showEvent(event)
        from PySide6.QtGui import QGuiApplication
        screen = QGuiApplication.primaryScreen()
        if screen:
            screen_geometry = screen.availableGeometry()
            window_geometry = self.frameGeometry()
            window_geometry.moveCenter(screen_geometry.center())
            self.move(window_geometry.topLeft())
        # Set initial red panel size with 4:3 ratio
        self._update_red_panel_size()


def main():
    app = QApplication(sys.argv)
    window = DebugCellWindow()
    
    # Ensure window is visible and on top
    window.setVisible(True)
    window.show()
    window.raise_()
    window.activateWindow()
    
    # Force window to front and ensure it's not minimized
    if window.isMinimized():
        window.showNormal()
    window.setWindowState(window.windowState() & ~Qt.WindowState.WindowMinimized)
    window.raise_()
    window.activateWindow()
    
    logging.info(f"Window shown: visible={window.isVisible()}, minimized={window.isMinimized()}")
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
