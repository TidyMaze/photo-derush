# Common helpers

import cv2
from PIL import Image, ExifTags
import imagehash
import logging

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
        img = Image.open(img_path)
        return imagehash.phash(img)
    except Exception as e:
        logging.warning(f"Could not compute perceptual hash for {img_path}: {e}")
        return None

def extract_exif(img_path):
    try:
        img = Image.open(img_path)
        exif_data = img._getexif()
        if not exif_data:
            return {}
        exif = {}
        for tag, value in exif_data.items():
            decoded = ExifTags.TAGS.get(tag, tag)
            exif[decoded] = value
        return exif
    except Exception as e:
        logging.warning(f"Could not extract EXIF for {img_path}: {e}")
        return {}

def format_gps_info(gps_info):
    def _convert_to_degrees(value):
        d, m, s = value
        return float(d[0]) / float(d[1]) + \
               float(m[0]) / float(m[1]) / 60 + \
               float(s[0]) / float(s[1]) / 3600
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
