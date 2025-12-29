"""Test that manual labeling is blocked during background tasks."""
import os
from unittest.mock import patch

import pytest

from src.viewmodel import PhotoViewModel


@pytest.fixture
def temp_images_dir(tmp_path):
    """Create temporary directory with test images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # Create dummy image files
    for i in range(3):
        img_path = img_dir / f"test{i}.jpg"
        # Create minimal valid JPEG
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='red')
        img.save(img_path, 'JPEG')

    return str(img_dir)


@pytest.fixture(autouse=True)
def disable_object_detection():
    """Disable object detection for all tests in this module."""
    with patch('src.viewmodel.PhotoViewModel._load_object_detections', lambda self: None):
        yield


@pytest.fixture
def vm(temp_images_dir):
    """Create PhotoViewModel with test directory."""
    with patch('src.viewmodel.ThumbnailService'), \
         patch('src.viewmodel.ExifService'):
        vm = PhotoViewModel(temp_images_dir, max_images=100)
        # Load images synchronously for testing
        vm.load_images()
        return vm


def test_label_blocked_during_retrain(vm):
    """Manual labeling should be blocked for manually-labeled images during retrain."""
    # Select an image and give it a manual label first
    if vm.images:
        vm.select_image(vm.images[0])
        vm.set_label('keep')  # Set manual label

    initial_label = vm.model.get_state(vm.selected_image)

    # Simulate retrain task starting
    vm._on_task_started('retrain')

    # Verify blocking check works
    assert vm._is_blocking_task_running()
    assert 'retrain' in vm._active_tasks

    # Try to change the manually-labeled image - should be blocked
    vm.set_label('trash')

    # Label should not have changed
    current_label = vm.model.get_state(vm.selected_image)
    assert current_label == initial_label

    # Simulate task finishing
    vm._on_task_finished('retrain', True)

    # Now labeling should work
    assert not vm._is_blocking_task_running()
    vm.set_label('trash')
    assert vm.model.get_state(vm.selected_image) == 'trash'


def test_auto_label_not_blocked_during_retrain(vm):
    """Auto-labeled images CAN be manually changed during retrain (safe override)."""
    if vm.images:
        vm.select_image(vm.images[0])
        # Set an auto label directly (bypass set_label to simulate auto-labeling)
        path = vm.selected_image
        vm.model.set_state(path, 'keep', source='auto')
        vm._auto.auto_assigned.add(path)

    # Simulate retrain task starting
    vm._on_task_started('retrain')
    assert vm._is_blocking_task_running()

    # Try to manually override the auto-label - should be ALLOWED
    vm.set_label('trash')

    # Label should have changed to trash with manual source
    assert vm.model.get_state(vm.selected_image) == 'trash'
    filename = os.path.basename(vm.selected_image)
    source = vm.model._repo.get_label_source(filename)
    assert source == 'manual'

    # Should be removed from auto-assigned tracking
    assert vm.selected_image not in vm._auto.auto_assigned

    vm._on_task_finished('retrain', True)


def test_label_blocked_during_predict(vm):
    """Manually-labeled images should be blocked during predict task."""
    if vm.images:
        vm.select_image(vm.images[0])
        vm.set_label('keep')  # Set manual label first

    initial_label = vm.model.get_state(vm.selected_image)

    vm._on_task_started('predict')
    assert vm._is_blocking_task_running()

    vm.set_label('trash')

    # Label should not have changed
    assert vm.model.get_state(vm.selected_image) == initial_label

    vm._on_task_finished('predict', True)
    assert not vm._is_blocking_task_running()


def test_label_blocked_during_auto_label(vm):
    """Manually-labeled images should be blocked during auto-label task."""
    if vm.images:
        vm.select_image(vm.images[0])
        vm.set_label('trash')  # Set manual label first

    initial_label = vm.model.get_state(vm.selected_image)

    vm._on_task_started('auto-label')
    assert vm._is_blocking_task_running()

    vm.set_label('keep')

    assert vm.model.get_state(vm.selected_image) == initial_label

    vm._on_task_finished('auto-label', True)
    assert not vm._is_blocking_task_running()


def test_label_not_blocked_during_filter(vm):
    """Manual labeling should NOT be blocked during filter task (fast UI op)."""
    if vm.images:
        vm.select_image(vm.images[0])

    vm._on_task_started('filter')
    # Filter should NOT block labeling
    assert not vm._is_blocking_task_running()

    vm.set_label('keep')
    assert vm.model.get_state(vm.selected_image) == 'keep'

    vm._on_task_finished('filter', True)


def test_unlabeled_image_allowed_during_tasks(vm):
    """Unlabeled images (no source) CAN be labeled during tasks."""
    if vm.images:
        vm.select_image(vm.images[0])
        # Ensure image has no label
        assert vm.model.get_state(vm.selected_image) == ''

    vm._on_task_started('retrain')
    assert vm._is_blocking_task_running()

    # Labeling an unlabeled image should be ALLOWED
    vm.set_label('keep')
    assert vm.model.get_state(vm.selected_image) == 'keep'

    vm._on_task_finished('retrain', True)


def test_label_blocked_during_multiple_tasks(vm):
    """Manually-labeled images should be blocked when multiple blocking tasks run."""
    if vm.images:
        vm.select_image(vm.images[0])
        vm.set_label('keep')  # Set manual label first

    initial_label = vm.model.get_state(vm.selected_image)

    # Start multiple tasks
    vm._on_task_started('retrain')
    vm._on_task_started('predict')

    assert vm._is_blocking_task_running()
    assert 'retrain' in vm._active_tasks
    assert 'predict' in vm._active_tasks

    vm.set_label('trash')
    assert vm.model.get_state(vm.selected_image) == initial_label

    # Finish one task - still blocked
    vm._on_task_finished('retrain', True)
    assert vm._is_blocking_task_running()

    vm.set_label('trash')
    assert vm.model.get_state(vm.selected_image) == initial_label

    # Finish second task - now unblocked
    vm._on_task_finished('predict', True)
    assert not vm._is_blocking_task_running()

    vm.set_label('trash')
    assert vm.model.get_state(vm.selected_image) == 'trash'


def test_task_tracking_lifecycle(vm):
    """Task tracking should correctly add/remove tasks from active set."""
    assert len(vm._active_tasks) == 0

    vm._on_task_started('retrain')
    assert len(vm._active_tasks) == 1
    assert 'retrain' in vm._active_tasks

    vm._on_task_started('predict')
    assert len(vm._active_tasks) == 2

    vm._on_task_finished('retrain', True)
    assert len(vm._active_tasks) == 1
    assert 'predict' in vm._active_tasks

    vm._on_task_finished('predict', False)  # failed
    assert len(vm._active_tasks) == 0


def test_label_works_when_no_tasks_running(vm):
    """Manual labeling should work normally when no tasks are running."""
    if vm.images:
        vm.select_image(vm.images[0])

    assert not vm._is_blocking_task_running()
    assert len(vm._active_tasks) == 0

    vm.set_label('keep')
    assert vm.model.get_state(vm.selected_image) == 'keep'

    vm.set_label('trash')
    assert vm.model.get_state(vm.selected_image) == 'trash'


def test_auto_label_updates_when_prediction_changes(vm):
    """Auto-labeled images should update when new predictions differ."""
    if not vm.images or len(vm.images) < 2:
        return

    # Setup: Auto-label two images
    path1 = vm.model.get_image_path(vm.images[0])
    path2 = vm.model.get_image_path(vm.images[1])

    vm.model.set_state(path1, 'keep', source='auto')
    vm._auto.auto_assigned.add(path1)

    vm.model.set_state(path2, 'trash', source='auto')
    vm._auto.auto_assigned.add(path2)

    # Verify initial state
    assert vm.model.get_state(path1) == 'keep'
    assert vm.model.get_state(path2) == 'trash'
    assert vm.model._repo.get_label_source(vm.images[0]) == 'auto'
    assert vm.model._repo.get_label_source(vm.images[1]) == 'auto'

    # Simulate new predictions after retraining (reversed)
    vm._auto.predicted_labels = {
        vm.images[0]: 'trash',  # Was 'keep', now predicts 'trash'
        vm.images[1]: 'keep',   # Was 'trash', now predicts 'keep'
    }

    # Refresh auto-labels (simulates post-retrain workflow)
    vm._refresh_auto_labels()

    # Verify labels were updated to match new predictions
    assert vm.model.get_state(path1) == 'trash', "Auto-label should update from 'keep' to 'trash'"
    assert vm.model.get_state(path2) == 'keep', "Auto-label should update from 'trash' to 'keep'"

    # Verify they're still marked as auto-labeled
    assert vm.model._repo.get_label_source(vm.images[0]) == 'auto'
    assert vm.model._repo.get_label_source(vm.images[1]) == 'auto'


def test_auto_label_tracking_rebuilds_from_repository(vm):
    """Auto-assigned tracking should rebuild from repository on refresh."""
    if not vm.images or len(vm.images) < 1:
        return

    path1 = vm.model.get_image_path(vm.images[0])

    # Setup: Auto-label an image directly in repository
    vm.model.set_state(path1, 'keep', source='auto')

    # Simulate app restart - clear the in-memory tracking
    vm._auto.auto_assigned.clear()
    assert len(vm._auto.auto_assigned) == 0

    # Setup new predictions
    vm._auto.predicted_labels = {
        vm.images[0]: 'trash',  # Prediction changed
    }

    # Refresh should rebuild tracking from repository
    vm._refresh_auto_labels()

    # Verify tracking was rebuilt and label was updated
    assert path1 in vm._auto.auto_assigned, "Should rebuild tracking from repository"
    assert vm.model.get_state(path1) == 'trash', "Should update label based on new prediction"
    assert vm.model._repo.get_label_source(vm.images[0]) == 'auto'


def test_auto_label_removed_when_prediction_below_threshold(vm):
    """Auto-labels should be removed when new predictions fall below threshold."""
    if not vm.images or len(vm.images) < 2:
        return

    path1 = vm.model.get_image_path(vm.images[0])
    path2 = vm.model.get_image_path(vm.images[1])

    # Setup: Auto-label two images
    vm.model.set_state(path1, 'keep', source='auto')
    vm._auto.auto_assigned.add(path1)

    vm.model.set_state(path2, 'trash', source='auto')
    vm._auto.auto_assigned.add(path2)

    # Verify initial state
    assert vm.model.get_state(path1) == 'keep'
    assert vm.model.get_state(path2) == 'trash'

    # Simulate new predictions where probabilities are below threshold (neutral)
    vm._auto.predicted_labels = {
        vm.images[0]: '',  # Below threshold - should be removed
        vm.images[1]: '',  # Below threshold - should be removed
    }

    # Refresh auto-labels
    vm._refresh_auto_labels()

    # Verify labels were removed (cleared to empty string)
    assert vm.model.get_state(path1) == '', "Auto-label should be removed when prediction below threshold"
    assert vm.model.get_state(path2) == '', "Auto-label should be removed when prediction below threshold"

    # Verify they're removed from tracking
    assert path1 not in vm._auto.auto_assigned, "Should remove from tracking when label cleared"
    assert path2 not in vm._auto.auto_assigned, "Should remove from tracking when label cleared"


def test_auto_label_mixed_update_and_removal(vm):
    """Test mixed scenario: some predictions change, some removed, some unchanged."""
    if not vm.images or len(vm.images) < 3:
        return

    path1 = vm.model.get_image_path(vm.images[0])
    path2 = vm.model.get_image_path(vm.images[1])
    path3 = vm.model.get_image_path(vm.images[2])

    # Setup: Auto-label three images
    vm.model.set_state(path1, 'keep', source='auto')
    vm._auto.auto_assigned.add(path1)

    vm.model.set_state(path2, 'trash', source='auto')
    vm._auto.auto_assigned.add(path2)

    vm.model.set_state(path3, 'keep', source='auto')
    vm._auto.auto_assigned.add(path3)

    # New predictions: change, remove, keep
    vm._auto.predicted_labels = {
        vm.images[0]: 'trash',  # Change: keep → trash
        vm.images[1]: '',       # Remove: trash → (empty)
        vm.images[2]: 'keep',   # Unchanged: keep → keep
    }

    # Refresh
    vm._refresh_auto_labels()

    # Verify outcomes
    assert vm.model.get_state(path1) == 'trash', "Should update to new prediction"
    assert vm.model.get_state(path2) == '', "Should remove label when below threshold"
    assert vm.model.get_state(path3) == 'keep', "Should keep unchanged label"

    # Verify tracking
    assert path1 in vm._auto.auto_assigned, "Changed label still auto-labeled"
    assert path2 not in vm._auto.auto_assigned, "Removed label not in tracking"
    assert path3 in vm._auto.auto_assigned, "Unchanged label still auto-labeled"


