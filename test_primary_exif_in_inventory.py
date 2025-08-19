import json
import os
import pytest

PRIMARY_FIELDS = [
    'DateTimeOriginal','DateTime','Make','Model','LensMake','LensModel','Software',
    'ExposureTime','ShutterSpeedValue','FNumber','ApertureValue','ISOSpeedRatings',
    'ExposureBiasValue','BrightnessValue','MeteringMode','WhiteBalance','Flash',
    'FocalLength','FocalLengthIn35mmFilm','SubjectDistance','DigitalZoomRatio',
    'SceneCaptureType','Contrast','Saturation','Sharpness','CompositeImage',
    'GPSInfo','ExifImageWidth','ExifImageHeight'
]

def load_inventory(path='exif_inventory_run.json'):
    if not os.path.exists(path):
        pytest.skip(f"Inventory file {path} not found; run exif inventory first")
    with open(path,'r') as f:
        return json.load(f)

def test_primary_exif_fields_present_in_inventory():
    inv = load_inventory()
    freq_names = {name for name,_cnt in inv.get('tag_frequency', [])}
    missing = [k for k in PRIMARY_FIELDS if k not in freq_names]
    assert not missing, f"Primary EXIF fields missing from inventory: {missing}"
