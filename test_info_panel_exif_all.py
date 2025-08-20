import os
import pytest
from PySide6.QtWidgets import QApplication
from photo_derush.info_panel import InfoPanel

# Synthetic comprehensive EXIF sample (mix of primary + other fields)
SYN_EXIF = {
    'DateTimeOriginal': '2025:07:04 12:34:56',
    'DateTime': '2025:07:04 12:34:56',
    'Make': 'TestMake',
    'Model': 'TestModel',
    'LensMake': 'LensCorp',
    'LensModel': 'LensCorp Zoom 10-20mm',
    'Software': 'UnitTest 1.0',
    'ExposureTime': '0.01',
    'ShutterSpeedValue': '6.64',
    'FNumber': '2.8',
    'ApertureValue': '2.8',
    'ISOSpeedRatings': 800,
    'ExposureBiasValue': '0.0',
    'BrightnessValue': '1.23',
    'MeteringMode': 5,
    'WhiteBalance': 0,
    'Flash': 16,
    'FocalLength': '35.0',
    'FocalLengthIn35mmFilm': 52,
    'SubjectDistance': '4.0',
    'DigitalZoomRatio': '1.0',
    'SceneCaptureType': 0,
    'Contrast': 0,
    'Saturation': 0,
    'Sharpness': 0,
    'CompositeImage': 3,
    'GPSInfo': {1: 'N', 2: ((47,1),(30,1),(1000,100)), 3: 'E', 4: ((2,1),(15,1),(500,100))},
    'ExifImageWidth': 4000,
    'ExifImageHeight': 3000,
    'OffsetTimeOriginal': '+02:00',
    'SubsecTimeDigitized': '123',
    'ResolutionUnit': 2,
    'XResolution': '72.0',
    'YResolution': '72.0',
    'Orientation': 1,
    'CustomRendered': 1,
    'ExposureProgram': 2,
    'ColorSpace': 65535,
}

@pytest.fixture(scope='module')
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app

def get_real_image_path():
    image_dir = '/Users/yannrolland/Pictures/photo-dataset'
    image_paths = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
    if not image_paths:
        import pytest
        pytest.skip(f"At least one .jpg image is required in {image_dir}.")
    return image_dir, image_paths[0]

def test_all_exif_keys_rendered(qapp):
    image_dir, image_name = get_real_image_path()
    image_path = os.path.join(image_dir, image_name)
    from photo_derush.info_panel import extract_exif
    exif_data = extract_exif(image_path)
    panel = InfoPanel()
    panel.update_info(image_name, image_path, '-', '-', '-', metrics=None, keep_prob=None)
    html = panel.text_edit.toHtml()
    missing = []
    for k in exif_data.keys():
        if k == 'GPSInfo':
            continue
        if str(k) not in html:
            missing.append(k)
    assert not missing, f"Missing EXIF keys in HTML: {missing}"
