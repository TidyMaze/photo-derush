import os
import tempfile

from PySide6.QtCore import QCoreApplication

from src.viewmodel import PhotoViewModel


def test_command_stack_rating_tags_label_undo_redo():
    app = QCoreApplication.instance() or QCoreApplication([])
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, 'a.jpg')
        open(img_path, 'a').close()
        vm = PhotoViewModel(tmpdir, max_images=10)
        vm.load_images()
        assert 'a.jpg' in vm.images
        vm.select_image('a.jpg')

        # Initial defaults
        assert vm.rating == 0
        assert vm.tags == []
        assert vm.label is None

        # Rating command
        vm.set_rating(4)
        assert vm.rating == 4
        vm.undo()
        assert vm.rating == 0
        vm.redo()
        assert vm.rating == 4

        # Tags command
        vm.set_tags(['x', 'y'])
        assert vm.tags == ['x', 'y']
        vm.undo()
        assert vm.tags == []
        vm.redo()
        assert vm.tags == ['x', 'y']

        # Label command
        vm.set_label('keep')
        assert vm.label == 'keep'
        vm.undo()
        # After undo label back to previous (None or '')
        assert vm.label in (None, '')
        vm.redo()
        assert vm.label == 'keep'

        # Multiple undos
        vm.undo()  # label
        vm.undo()  # tags
        vm.undo()  # rating
        assert vm.label in (None, '')
        assert vm.tags == []
        assert vm.rating == 0
        # Redo chain
        vm.redo(); vm.redo(); vm.redo()
        assert vm.rating == 4
        assert vm.tags == ['x', 'y']
        assert vm.label == 'keep'
