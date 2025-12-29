import pytest

from src.object_detection import sanitize_detection as _sanitize_detection


def test_sanitize_tuple_input():
    out = _sanitize_detection(('person', 0.9))
    assert out == {'class': 'person', 'confidence': 0.9, 'bbox': None}


def test_sanitize_dict_with_bbox():
    out = _sanitize_detection({'class': 'dog', 'confidence': 0.75, 'bbox': [10, 20, 30, 40]})
    assert out == {'class': 'dog', 'confidence': 0.75, 'bbox': [10, 20, 30, 40]}


def test_sanitize_invalid_raises():
    with pytest.raises(ValueError):
        _sanitize_detection(123)
