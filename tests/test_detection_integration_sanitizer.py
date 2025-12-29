import pytest
from unittest.mock import patch

from src.viewmodel import PhotoViewModel


def test_viewmodel_fails_fast_on_malformed_batch(tmp_path):
    # Prepare a small directory with one image filename
    d = tmp_path
    # create the VM
    vm = PhotoViewModel(str(d))
    # set images (basename expected by VM)
    vm.images = ['a.jpg']

    # Patch object_detection.get_objects_for_images to return malformed items
    malformed = {'a.jpg': ['not-a-tuple-or-dict']}

    import src.object_detection as od

    with patch.object(od, 'get_objects_for_images', return_value=malformed):
        with pytest.raises(ValueError):
            vm._load_object_detections()
