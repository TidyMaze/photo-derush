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

def test_all_exif_keys_rendered(qapp, monkeypatch):
    # Monkeypatch extract_exif to return synthetic dict
    from photo_derush import info_panel as ip_mod
    monkeypatch.setattr(ip_mod, 'extract_exif', lambda p: SYN_EXIF)
    panel = InfoPanel()
    panel.update_info('img.jpg', '/tmp/img.jpg', '-', '-', '-', metrics=None, keep_prob=None)
    html = panel.text_edit.toHtml()
    # Each key (except GPSInfo handled specially) must appear in HTML
    missing = []
    for k in SYN_EXIF.keys():
        # GPSInfo displayed as 'GPSInfo' line; skip dict contents
        if k == 'GPSInfo':
            if 'GPSInfo' not in html:
                missing.append(k)
            continue
        if k not in html:
            missing.append(k)
    assert not missing, f"Missing EXIF keys in rendered HTML: {missing}"
