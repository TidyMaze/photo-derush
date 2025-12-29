from PIL import Image
from unittest.mock import patch

from src.viewmodel import PhotoViewModel


def make_img(path):
    img = Image.new('RGB', (16, 16), (200, 50, 50))
    img.save(path, 'PNG')


def test_manual_label_protection(tmp_path):
    with patch('src.viewmodel.PhotoViewModel._load_object_detections', lambda self: None):
        # Setup directory with two images
        p1 = tmp_path / 'a.png'; p2 = tmp_path / 'b.png'
        make_img(p1); make_img(p2)

        vm = PhotoViewModel(str(tmp_path))
        vm.load_images()

        # Manually label first image as keep
        vm.select_image('a.png')
        vm.set_label('keep')
        path1 = vm.model.get_image_path('a.png')
        assert vm.model.get_state(path1) == 'keep'

        # Simulate auto attempt to overwrite manual label
        vm.model.set_state(path1, 'trash', source='auto')
        # Should remain keep
        assert vm.model.get_state(path1) == 'keep'

        # Simulate auto attempt to clear manual label
        vm.model.set_state(path1, '', source='auto')
        assert vm.model.get_state(path1) == 'keep'

        # Manual change should work
        vm.set_label('trash')
        assert vm.model.get_state(path1) == 'trash'

        # Auto can set label on second unlabeled image
        path2 = vm.model.get_image_path('b.png')
        vm.model.set_state(path2, 'keep', source='auto')
        assert vm.model.get_state(path2) == 'keep'


