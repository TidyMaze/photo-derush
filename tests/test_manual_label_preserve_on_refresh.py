from PIL import Image

from src.viewmodel import PhotoViewModel


def make_img(path, color):
    Image.new('RGB', (20, 20), color).save(path)


def test_manual_label_not_removed_on_refresh(tmp_path):
    from unittest.mock import patch
    with patch('src.viewmodel.PhotoViewModel._load_object_detections', lambda self: None):
        # create images
        a = tmp_path / 'a.png'; b = tmp_path / 'b.png'
        make_img(a, (250,20,20)); make_img(b, (20,250,20))
        vm = PhotoViewModel(str(tmp_path))
        vm.load_images()
        vm.select_image('a.png'); vm.set_label('keep')  # manual
        vm.select_image('b.png'); vm.set_label('trash')  # manual
        # simulate predictions below thresholds (empty) after retrain
        vm._auto.predicted_labels = {'a.png':'', 'b.png':''}
        vm._refresh_auto_labels()
        assert vm.model.get_state(str(a)) == 'keep'
        assert vm.model.get_state(str(b)) == 'trash'

