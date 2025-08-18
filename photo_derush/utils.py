# Common helpers

import cv2
from PIL import Image, ExifTags
import imagehash
import logging
from .image_manager import image_manager

def pil2pixmap(img: Image.Image):
    from PySide6.QtGui import QPixmap, QImage
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    data = img.tobytes("raw", "RGBA")
    qimg = QImage(data, img.width, img.height, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg)

def compute_blur_score(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.Laplacian(img, cv2.CV_64F).var()

def compute_sharpness_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    features = {}
    features['variance_laplacian'] = cv2.Laplacian(img, cv2.CV_64F).var()
    features['tenengrad'] = cv2.Laplacian(img, cv2.CV_64F).var()  # Placeholder for Tenengrad
    features['brenner'] = cv2.Laplacian(img, cv2.CV_64F).var()  # Placeholder for Brenner
    features['wavelet_energy'] = cv2.Laplacian(img, cv2.CV_64F).var()  # Placeholder for Wavelet energy
    return features

def compute_perceptual_hash(img_path):
    try:
        img = image_manager.get_image(img_path)
        if img is None:
            logging.warning(f"Could not compute perceptual hash for {img_path}: image not loadable")
            return None
        return imagehash.phash(img)
    except Exception as e:
        logging.warning(f"Could not compute perceptual hash for {img_path}: {e}")
        return None

def extract_exif(img_path):
    """Extract EXIF metadata from an image.
    Tries multiple Pillow APIs to maximize chance of retrieving tags.
    Falls back gracefully to empty dict on failure.
    """
    try:
        # First attempt: open directly to preserve raw EXIF block
        try:
            with Image.open(img_path) as im:
                # Prefer modern getexif() (returns Exif object) then legacy _getexif()
                exif_block = None
                try:
                    exif_block = im.getexif()  # Pillow Exif object or empty
                except Exception:  # noqa: PERF203
                    exif_block = None
                if exif_block and len(exif_block):  # Exif object behaves like dict
                    exif = {}
                    for tag_id, value in exif_block.items():
                        tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                        # Attempt simple byte -> utf8 decode
                        if isinstance(value, bytes):
                            try:
                                value = value.decode('utf-8', 'ignore')
                            except Exception:  # noqa: PERF203
                                pass
                        exif[tag_name] = value
                    return exif
                # Legacy path
                legacy = getattr(im, '_getexif', lambda: None)()
                if legacy:
                    exif = {}
                    for tag, value in legacy.items():
                        decoded = ExifTags.TAGS.get(tag, tag)
                        if isinstance(value, bytes):
                            try:
                                value = value.decode('utf-8', 'ignore')
                            except Exception:  # noqa: PERF203
                                pass
                        exif[decoded] = value
                    return exif
        except Exception:  # noqa: PERF203
            pass
        # Fallback: use ImageManager (may have stripped EXIF in some cases)
        img = image_manager.get_image(img_path)
        if img is None:
            logging.warning(f"Could not extract EXIF for {img_path}: image not loadable")
            return {}
        exif_data = None
        # Try modern API
        try:
            exif_data = img.getexif()
            if exif_data and len(exif_data):
                exif = {}
                for tag_id, value in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8', 'ignore')
                        except Exception:  # noqa: PERF203
                            pass
                    exif[tag_name] = value
                return exif
        except Exception:  # noqa: PERF203
            exif_data = None
        if not exif_data:
            exif_data = getattr(img, '_getexif', lambda: None)()
        if not exif_data:
            return {}
        exif = {}
        for tag, value in exif_data.items():
            decoded = ExifTags.TAGS.get(tag, tag)
            if isinstance(value, bytes):
                try:
                    value = value.decode('utf-8', 'ignore')
                except Exception:  # noqa: PERF203
                    pass
            exif[decoded] = value
        return exif
    except Exception as e:  # noqa: PERF203
        logging.warning(f"Could not extract EXIF for {img_path}: {e}")
        return {}

def format_gps_info(gps_info):
    def _to_float(val):
        # Handles tuple, int, float, or IFDRational
        try:
            if hasattr(val, 'numerator') and hasattr(val, 'denominator'):
                return float(val.numerator) / float(val.denominator)
            elif isinstance(val, (tuple, list)) and len(val) == 2:
                return float(val[0]) / float(val[1])
            else:
                return float(val)
        except Exception:
            return float(val)
    def _convert_to_degrees(value):
        d, m, s = value
        return _to_float(d) + _to_float(m) / 60 + _to_float(s) / 3600
    try:
        gps_tags = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_info.items()}
        lat = lon = None
        if 'GPSLatitude' in gps_tags and 'GPSLatitudeRef' in gps_tags:
            lat = _convert_to_degrees(gps_tags['GPSLatitude'])
            if gps_tags['GPSLatitudeRef'] in ['S', b'S']:
                lat = -lat
        if 'GPSLongitude' in gps_tags and 'GPSLongitudeRef' in gps_tags:
            lon = _convert_to_degrees(gps_tags['GPSLongitude'])
            if gps_tags['GPSLongitudeRef'] in ['W', b'W']:
                lon = -lon
        if lat is not None and lon is not None:
            return f"Latitude: {lat:.6f}, Longitude: {lon:.6f}"
        return str(gps_tags)
    except Exception as e:
        return f"[Invalid GPSInfo: {e}]"
