import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from src.viewmodel import PhotoViewModel
import src.object_detection


class TestViewModelObjectDetection:
    """Test object detection integration in PhotoViewModel."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_images(self, temp_dir):
        """Create sample image files for testing."""
        from PIL import Image
        images = []
        for i in range(3):
            img_path = os.path.join(temp_dir, f'image_{i}.jpg')
            img = Image.new('RGB', (100, 100), color='red')
            img.save(img_path)
            images.append(f'image_{i}.jpg')
        return images

    def test_load_object_detections_empty_images(self, temp_dir):
        """Test _load_object_detections with no images."""
        vm = PhotoViewModel(temp_dir, max_images=10)
        vm.images = []

        vm._load_object_detections()

        assert vm._detected_objects == {}

    def test_load_object_detections_with_images(self, temp_dir, sample_images):
        """Test _load_object_detections with images."""
        vm = PhotoViewModel(temp_dir, max_images=10)
        vm.images = sample_images

        # Mock the object detection function to return tuples (class, confidence)
        mock_detections = {
            'image_0.jpg': [('person', 0.95)],
            'image_1.jpg': [('car', 0.87)],
            'image_2.jpg': []  # No objects detected
        }

        # Patch the actual module object used at runtime to avoid import-time mismatches.
        def _mock_batch(paths):
            # viewmodel calls with full paths; return dict keyed by basename
            return {os.path.basename(p): v for p, v in mock_detections.items()}

        with patch.object(src.object_detection, 'load_object_cache', return_value={}), \
             patch.object(src.object_detection, 'save_object_cache', return_value=None), \
             patch.object(src.object_detection, 'get_objects_for_images', return_value=mock_detections), \
             patch.object(vm._tasks, 'run', side_effect=lambda name, fn: fn()):
            vm._load_object_detections()

            expected = {
                'image_0.jpg': [{'class': 'person', 'confidence': 0.95, 'bbox': None}],
                'image_1.jpg': [{'class': 'car', 'confidence': 0.87, 'bbox': None}],
                'image_2.jpg': []
            }
            assert vm._detected_objects == expected

    def test_load_object_detections_handles_exceptions(self, temp_dir, sample_images):
        """Test _load_object_detections handles exceptions gracefully."""
        vm = PhotoViewModel(temp_dir, max_images=10)
        vm.images = sample_images
        # Mock object detection to raise an exception
        with patch.object(src.object_detection, 'load_object_cache', return_value={}), \
             patch.object(src.object_detection, 'save_object_cache', return_value=None), \
             patch.object(src.object_detection, 'get_objects_for_images', side_effect=Exception("Test error")), \
             patch.object(vm._tasks, 'run', side_effect=lambda name, fn: fn()):
            vm._load_object_detections()

            # ViewModel adopts an empty mapping when the batch API raises
            assert vm._detected_objects == {}

    def test_emit_state_snapshot_includes_detected_objects(self, temp_dir, sample_images):
        """Test that browser state snapshot includes detected objects."""
        vm = PhotoViewModel(temp_dir, max_images=10)
        vm.images = sample_images

        # Mock detected objects
        mock_detections = {
            'image_0.jpg': [{'class': 'person', 'confidence': 0.95, 'bbox': None}],
            'image_1.jpg': [{'class': 'car', 'confidence': 0.87, 'bbox': None}],
        }
        vm._detected_objects = mock_detections

        # Mock other required attributes
        vm._rating = 0
        vm._tags = []
        vm._label = None
        vm._has_selected_image = False
        vm._progress_current = 3
        vm._progress_total = 3
        vm._active_tasks = set()
        vm._filtered_images = sample_images
        vm._cmd_stack = MagicMock()
        vm._cmd_stack.can_undo = False
        vm._cmd_stack.can_redo = False

        # Mock selection model
        vm.selection_model = MagicMock()
        vm.selection_model.selected.return_value = []
        vm.selection_model.primary.return_value = None

        # Mock auto label manager
        vm._auto = MagicMock()
        vm._auto.predicted_labels = {}
        vm._auto.predicted_probabilities = {}
        vm._auto.auto_assigned = set()

        # Mock filter controller
        vm._filter_ctrl = MagicMock()
        vm._filter_ctrl.active.return_value = False

        # Capture emitted state
        emitted_states = []
        vm.browser_state_changed.connect(lambda state: emitted_states.append(state))

        with patch.object(vm, '_load_object_detections'):  # Prevent actual model loading
            vm._emit_state_snapshot()

        assert len(emitted_states) == 1
        state = emitted_states[0]
        assert hasattr(state, 'detected_objects')
        assert state.detected_objects == mock_detections

    def test_object_detections_loaded_on_image_loading(self, temp_dir, sample_images):
        """Test that object detections are loaded when images are loaded."""
        vm = PhotoViewModel(temp_dir, max_images=10)

        mock_detections = {
            'image_0.jpg': [('person', 0.95)],
            'image_1.jpg': [('car', 0.87)],
            'image_2.jpg': []
        }

        with patch.object(src.object_detection, 'load_object_cache', return_value={}), \
             patch.object(src.object_detection, 'save_object_cache', return_value=None), \
             patch.object(src.object_detection, 'get_objects_for_images', return_value=mock_detections), \
             patch.object(vm._tasks, 'run', side_effect=lambda name, fn: fn()):
            # Simulate loading images
            vm.images = sample_images
            vm._emit_state_snapshot()  # This should call _load_object_detections

            expected = {
                'image_0.jpg': [{'class': 'person', 'confidence': 0.95, 'bbox': None}],
                'image_1.jpg': [{'class': 'car', 'confidence': 0.87, 'bbox': None}],
                'image_2.jpg': []
            }
            assert vm._detected_objects == expected

    def test_object_detections_updated_on_state_changes(self, temp_dir, sample_images):
        """Test that object detections are refreshed when browser state changes."""
        vm = PhotoViewModel(temp_dir, max_images=10)
        vm.images = sample_images

        # Initial load
        mock_detections_1 = {'image_0.jpg': [('person', 0.95)]}
        expected_1 = {'image_0.jpg': [{'class': 'person', 'confidence': 0.95, 'bbox': None}]}
        with patch.object(src.object_detection, 'load_object_cache', return_value={}), \
             patch.object(src.object_detection, 'save_object_cache', return_value=None), \
             patch.object(src.object_detection, 'get_objects_for_images', return_value=mock_detections_1), \
             patch.object(vm._tasks, 'run', side_effect=lambda name, fn: fn()):
            vm._emit_state_snapshot()
            assert vm._detected_objects == expected_1

        # Simulate state change that triggers re-detection
        mock_detections_2 = {'image_0.jpg': [('person', 0.95), ('dog', 0.82)]}
        expected_2 = {'image_0.jpg': [{'class': 'person', 'confidence': 0.95, 'bbox': None}, {'class': 'dog', 'confidence': 0.82, 'bbox': None}]}
        with patch.object(src.object_detection, 'load_object_cache', return_value={}), \
             patch.object(src.object_detection, 'save_object_cache', return_value=None), \
             patch.object(src.object_detection, 'get_objects_for_images', return_value=mock_detections_2), \
             patch.object(vm._tasks, 'run', side_effect=lambda name, fn: fn()):
            vm._emit_state_snapshot()  # Should reload detections
            assert vm._detected_objects == expected_2