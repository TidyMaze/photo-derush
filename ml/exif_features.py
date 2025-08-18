import logging
from pathlib import Path
from datetime import datetime
from math import log, log1p, sin, cos, pi
from typing import Dict, Any
try:
    from PIL import Image, ExifTags
except Exception:  # pragma: no cover
    Image = None
    ExifTags = None

logger = logging.getLogger(__name__)

# List of EXIF-derived feature keys (strings allowed for some)
EXIF_FEATURE_KEYS = [
    # Exposure & lighting (numeric)
    'iso','iso_log1p','exposure_time_s','exposure_time_log','fnumber','aperture_value',
    'exposure_bias','brightness_value','ev100','low_light_flag','ev_vs_iso',
    # Optics
    'focal_length_mm','focal_length_35mm','subject_distance_m','digital_zoom_ratio','hand_shake_risk','dof_proxy',
    # Device & processing (categorical / int flags)
    'make','model','lens_make','lens_model','software','custom_rendered','composite_image','exposure_mode',
    'exposure_program','metering_mode','white_balance',
    # Flash & scene
    'flash_fired','scene_capture_type','saturation','contrast','sharpness',
    # Time
    'hour_of_day','hour_sin','hour_cos','day_of_week','night_flag',
    # GPS
    'latitude','longitude','urban_flag',
    # Image geometry (from EXIF, not pixel processing resize)
    'exif_image_width','exif_image_height','megapixels_exif','orientation_exif',
]

# Categorical string keys for default handling
_STRING_DEFAULT = "unknown"

# Helper mapping numeric tag ids
_TAG_MAP = None
_GPS_TAG_MAP = None
if ExifTags is not None:
    _TAG_MAP = {v: k for k, v in ExifTags.TAGS.items()}
    _GPS_TAG_MAP = ExifTags.GPSTAGS


def _safe_float(v, default=0.0):
    try:
        if v is None:
            return float(default)
        if isinstance(v, (tuple, list)) and len(v) == 2 and v[1] != 0:  # rational
            return float(v[0]) / float(v[1])
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v, default=0):
    try:
        if v is None:
            return int(default)
        if isinstance(v, (tuple, list)) and len(v) == 2 and v[1] != 0:
            return int(round(float(v[0]) / float(v[1])))
        return int(v)
    except Exception:
        return int(default)


def _compute_ev100(fnumber, exposure_s):
    try:
        if fnumber > 0 and exposure_s > 0:
            return (fnumber * fnumber) and (log((fnumber ** 2) / exposure_s, 2))
    except Exception:
        pass
    return 0.0


def _gps_to_deg(value):
    # value like [(num,den), ...]
    try:
        d, m, s = value
        def _rat(r):
            return r[0] / r[1] if isinstance(r, tuple) and r[1] else float(r)
        deg = _rat(d) + _rat(m) / 60.0 + _rat(s) / 3600.0
        return deg
    except Exception:
        return None


def _truncate(v, decimals=2):
    try:
        factor = 10 ** decimals
        return int(v * factor) / factor
    except Exception:
        return 0.0


def extract_exif_features(path: Path) -> Dict[str, Any]:
    feats: Dict[str, Any] = {k: 0.0 for k in EXIF_FEATURE_KEYS}
    # Initialize string categorical defaults
    for k in ['make','model','lens_make','lens_model','software']:
        feats[k] = _STRING_DEFAULT
    if Image is None:
        return feats
    try:
        with Image.open(path) as img:
            exif = img._getexif() if hasattr(img, '_getexif') else None
            if not exif:
                return feats
            # Reverse map tag id -> name
            tag_map = ExifTags.TAGS
            # Simple lookups
            rev = {tag_map.get(t, str(t)): v for t, v in exif.items()}
            iso = _safe_int(rev.get('ISOSpeedRatings') or rev.get('PhotographicSensitivity'))
            feats['iso'] = iso
            feats['iso_log1p'] = log1p(iso) if iso > 0 else 0.0
            # Exposure time / shutter speed
            exposure_time_s = _safe_float(rev.get('ExposureTime'))
            feats['exposure_time_s'] = exposure_time_s
            feats['exposure_time_log'] = log(exposure_time_s) if exposure_time_s > 0 else 0.0
            fnumber = _safe_float(rev.get('FNumber'))
            feats['fnumber'] = fnumber
            feats['aperture_value'] = _safe_float(rev.get('ApertureValue'))
            feats['exposure_bias'] = _safe_float(rev.get('ExposureBiasValue'))
            feats['brightness_value'] = _safe_float(rev.get('BrightnessValue'))
            ev100 = _compute_ev100(fnumber, exposure_time_s)
            feats['ev100'] = ev100
            feats['low_light_flag'] = 1 if (iso > 1600 or exposure_time_s > (1/15)) else 0
            feats['ev_vs_iso'] = ev100 - (log(iso/100.0,2) if iso>0 else 0.0)
            focal_len = _safe_float(rev.get('FocalLength'))
            feats['focal_length_mm'] = focal_len
            feats['focal_length_35mm'] = _safe_float(rev.get('FocalLengthIn35mmFilm'))
            feats['subject_distance_m'] = _safe_float(rev.get('SubjectDistance'))
            feats['digital_zoom_ratio'] = _safe_float(rev.get('DigitalZoomRatio'), 1.0)
            feats['hand_shake_risk'] = exposure_time_s * (feats['focal_length_35mm'] or focal_len)
            feats['dof_proxy'] = (fnumber*fnumber / focal_len) if (fnumber>0 and focal_len>0) else 0.0
            # Device strings
            feats['make'] = str(rev.get('Make', _STRING_DEFAULT)) or _STRING_DEFAULT
            feats['model'] = str(rev.get('Model', _STRING_DEFAULT)) or _STRING_DEFAULT
            feats['lens_make'] = str(rev.get('LensMake', _STRING_DEFAULT)) or _STRING_DEFAULT
            feats['lens_model'] = str(rev.get('LensModel', _STRING_DEFAULT)) or _STRING_DEFAULT
            feats['software'] = str(rev.get('Software', _STRING_DEFAULT)) or _STRING_DEFAULT
            feats['custom_rendered'] = _safe_int(rev.get('CustomRendered'))
            feats['composite_image'] = _safe_int(rev.get('CompositeImage'))
            feats['exposure_mode'] = _safe_int(rev.get('ExposureMode'))
            feats['exposure_program'] = _safe_int(rev.get('ExposureProgram'))
            feats['metering_mode'] = _safe_int(rev.get('MeteringMode'))
            feats['white_balance'] = _safe_int(rev.get('WhiteBalance'))
            flash_val = _safe_int(rev.get('Flash'))
            feats['flash_fired'] = 1 if (flash_val & 0x1) else 0
            feats['scene_capture_type'] = _safe_int(rev.get('SceneCaptureType'))
            feats['saturation'] = _safe_int(rev.get('Saturation'))
            feats['contrast'] = _safe_int(rev.get('Contrast'))
            feats['sharpness'] = _safe_int(rev.get('Sharpness'))
            # Time
            dt_raw = rev.get('DateTimeOriginal') or rev.get('DateTime')
            if isinstance(dt_raw, bytes):
                try: dt_raw = dt_raw.decode('utf-8', 'ignore')
                except Exception: dt_raw = None
            hour = 0
            dow = 0
            if isinstance(dt_raw, str):
                try:
                    dt = datetime.strptime(dt_raw, '%Y:%m:%d %H:%M:%S')
                    hour = dt.hour
                    dow = dt.weekday()
                except Exception:
                    pass
            feats['hour_of_day'] = hour
            feats['hour_sin'] = sin(2*pi*hour/24.0)
            feats['hour_cos'] = cos(2*pi*hour/24.0)
            feats['day_of_week'] = dow
            feats['night_flag'] = 1 if (hour >= 20 or hour <= 6) else 0
            # Geometry
            feats['exif_image_width'] = _safe_int(rev.get('ExifImageWidth'))
            feats['exif_image_height'] = _safe_int(rev.get('ExifImageHeight'))
            w = feats['exif_image_width']; h = feats['exif_image_height']
            feats['megapixels_exif'] = (w*h)/1e6 if (w>0 and h>0) else 0.0
            feats['orientation_exif'] = 1 if h> w else 0
            # GPS
            gps = rev.get('GPSInfo')
            lat = lon = None
            if isinstance(gps, dict):
                gps_map = gps
                try:
                    lat = _gps_to_deg(gps_map.get(2))
                    if gps_map.get(1,'N') == 'S' and lat is not None:
                        lat = -lat
                    lon = _gps_to_deg(gps_map.get(4))
                    if gps_map.get(3,'E') == 'W' and lon is not None:
                        lon = -lon
                except Exception:
                    lat = lon = None
            feats['latitude'] = _truncate(lat) if lat is not None else 0.0
            feats['longitude'] = _truncate(lon) if lon is not None else 0.0
            feats['urban_flag'] = 0  # placeholder
    except Exception:  # pragma: no cover
        logger.info('[EXIF] Failed extraction for %s', path)
    return feats

