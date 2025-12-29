from PIL import Image
from unittest.mock import patch

from src.viewmodel import PhotoViewModel


def img(path, color):
    Image.new('RGB',(24,24),color).save(path)


def test_manual_labels_persist_after_other_manual_changes(tmp_path):
    with patch('src.viewmodel.PhotoViewModel._load_object_detections', lambda self: None):
        names=['a.png','b.png','c.png']
        colors=[(250,10,10),(10,250,10),(200,200,40)]
        for n,c in zip(names,colors): img(tmp_path/n, c)
        vm=PhotoViewModel(str(tmp_path)); vm.load_images()
        vm.select_image('a.png'); vm.set_label('keep')
        vm.select_image('b.png'); vm.set_label('trash')
        # Simulate predictions clearing all
        vm._auto.predicted_labels={n:'' for n in names}
        vm._refresh_auto_labels()
        # Both manual labels should remain
        a_path=vm.model.get_image_path('a.png'); b_path=vm.model.get_image_path('b.png')
        assert vm.model.get_state(a_path)=='keep'
        assert vm.model.get_state(b_path)=='trash'
        # Add manual label to c
        vm.select_image('c.png'); vm.set_label('keep')
        # Refresh again with empty predictions
        vm._auto.predicted_labels={n:'' for n in names}
        vm._refresh_auto_labels()
        c_path=vm.model.get_image_path('c.png')
        assert vm.model.get_state(a_path)=='keep'
        assert vm.model.get_state(b_path)=='trash'
        assert vm.model.get_state(c_path)=='keep'

