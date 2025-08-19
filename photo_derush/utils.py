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
    """Extract EXIF metadata directly from the original image file.
    Combines modern getexif() + legacy _getexif() and expands GPS / Exif sub-IFDs.
    Returns mapping tag_name -> value with GPSInfo as nested dict when available.
    """
    try:
        with Image.open(img_path) as im:
            merged = {}
            exif_obj = None
            try:
                exif_obj = im.getexif()
            except Exception:  # noqa: PERF203
                exif_obj = None
            # Helper to record a tag
            def _store(tag_id, value):
                name = ExifTags.TAGS.get(tag_id, tag_id)
                if isinstance(value, bytes):
                    try:
                        value = value.decode('utf-8', 'ignore')
                    except Exception:  # noqa: PERF203
                        pass
                merged[name] = value
            # Modern pass
            if exif_obj and len(exif_obj):
                for tid, val in exif_obj.items():
                    _store(tid, val)
                # Expand Exif and GPS sub-IFDs if accessible (Pillow exposes get_ifd)
                try:
                    from PIL import ExifTags as _ET
                    # Exif SubIFD
                    if hasattr(exif_obj, 'get_ifd') and hasattr(_ET, 'IFD'):
                        for ifd_member in ('Exif', 'GPSInfo', 'Interop'):
                            if hasattr(_ET.IFD, ifd_member):
                                ifd_id = getattr(_ET.IFD, ifd_member)
                                try:
                                    sub = exif_obj.get_ifd(ifd_id)
                                except Exception:  # noqa: PERF203
                                    sub = None
                                if sub:
                                    if ifd_member == 'GPSInfo':
                                        gps_map = {}
                                        for stid, sval in sub.items():
                                            gps_name = _ET.GPSTAGS.get(stid, stid)
                                            if isinstance(sval, bytes):
                                                try:
                                                    sval = sval.decode('utf-8', 'ignore')
                                                except Exception:  # noqa: PERF203
                                                    pass
                                            gps_map[gps_name] = sval
                                        if gps_map:
                                            merged['GPSInfo'] = gps_map
                                    else:
                                        for stid, sval in sub.items():
                                            _store(stid, sval)
                except Exception:  # noqa: PERF203
                    pass
            # Legacy flatten (may contain more entries) merge without overwriting existing unless new
            legacy = getattr(im, '_getexif', lambda: None)()
            if legacy:
                for tid, val in legacy.items():
                    name = ExifTags.TAGS.get(tid, tid)
                    if name not in merged:
                        if isinstance(val, bytes):
                            try:
                                val = val.decode('utf-8', 'ignore')
                            except Exception:  # noqa: PERF203
                                pass
                        merged[name] = val
            # If GPSInfo still just an int offset, drop or leave? Replace int with empty dict for consistency
            gps_val = merged.get('GPSInfo')
            if isinstance(gps_val, int):
                # Attempt second-chance expansion via exif_obj if possible
                try:
                    from PIL import ExifTags as _ET
                    if exif_obj and hasattr(exif_obj, 'get_ifd') and hasattr(_ET, 'IFD') and hasattr(_ET, 'GPSTAGS'):
                        sub = exif_obj.get_ifd(_ET.IFD.GPSInfo)
                        if sub:
                            gps_map = {}
                            for stid, sval in sub.items():
                                gps_name = _ET.GPSTAGS.get(stid, stid)
                                if isinstance(sval, bytes):
                                    try:
                                        sval = sval.decode('utf-8', 'ignore')
                                    except Exception:  # noqa: PERF203
                                        pass
                                gps_map[gps_name] = sval
                            if gps_map:
                                merged['GPSInfo'] = gps_map
                except Exception:  # noqa: PERF203
                    pass
            return merged
    except Exception as e:  # noqa: PERF203
        logging.warning(f"[EXIF] Failed reading EXIF from {img_path}: {e}")
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
