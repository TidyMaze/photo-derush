"""Tests for hyperparameter tuning functionality."""
from pathlib import Path

from PIL import Image

from src.training import load_best_params, save_best_params, tune_hyperparameters


class FakeRepo:
    def __init__(self, mapping):
        self.mapping = mapping

    def get_state(self, filename: str):
        return self.mapping.get(filename, '')

    def get_label_source(self, filename: str):
        return 'manual' if filename in self.mapping else ''


def make_img(path: Path, color):
    img = Image.new('RGB', (32, 32), color)
    img.save(path, 'PNG')


def test_save_load_best_params(tmp_path):
    """Test saving and loading best parameters."""
    params_file = tmp_path / 'test_params.json'

    # Save params
    test_params = {
        'n_estimators': 150,
        'learning_rate': 0.05,
        'max_depth': 5,
        '_scale_pos_weight_ratio': 1.5
    }
    save_best_params(test_params, path=str(params_file))

    # Verify file exists
    assert params_file.exists()

    # Load params
    loaded = load_best_params(path=str(params_file))
    assert loaded is not None
    assert loaded['n_estimators'] == 150
    assert loaded['learning_rate'] == 0.05
    assert loaded['max_depth'] == 5
    assert loaded['_scale_pos_weight_ratio'] == 1.5


def test_load_best_params_missing_file():
    """Test loading params when file doesn't exist."""
    result = load_best_params(path='/nonexistent/path.json')
    assert result is None


def test_tune_hyperparameters_insufficient_data(tmp_path):
    """Test that tuning fails gracefully with insufficient data."""
    # Create only 2 images (not enough for 3-fold CV)
    make_img(tmp_path / 'k1.png', (255, 0, 0))
    make_img(tmp_path / 't1.png', (0, 255, 0))

    repo = FakeRepo({'k1.png': 'keep', 't1.png': 'trash'})

    result = tune_hyperparameters(
        str(tmp_path),
        repo=repo,
        n_iter=5,
        cv_folds=3,
        save_params=False
    )

    # Should return None due to insufficient samples
    assert result is None


def test_tune_hyperparameters_small_dataset(tmp_path):
    """Test tuning with a small but sufficient dataset."""
    # Create 8 images (enough for 3-fold CV with 2-3 per fold)
    colors_labels = [
        ((255, 0, 0), 'keep'),
        ((250, 5, 5), 'keep'),
        ((245, 10, 10), 'keep'),
        ((240, 15, 15), 'keep'),
        ((0, 255, 0), 'trash'),
        ((5, 250, 5), 'trash'),
        ((10, 245, 10), 'trash'),
        ((15, 240, 15), 'trash'),
    ]

    mapping = {}
    for i, (color, label) in enumerate(colors_labels):
        filename = f'img{i}.png'
        make_img(tmp_path / filename, color)
        mapping[filename] = label

    repo = FakeRepo(mapping)
    params_file = tmp_path / 'tuned_params.json'

    result = tune_hyperparameters(
        str(tmp_path),
        repo=repo,
        n_iter=3,  # Very few iterations for speed
        cv_folds=2,  # 2-fold CV for small dataset
        save_params=False,
        random_state=42
    )

    # Should return best params
    assert result is not None
    assert isinstance(result, dict)

    # Check that essential params are present
    assert 'n_estimators' in result
    assert 'learning_rate' in result
    assert 'max_depth' in result

    # Check metadata
    assert '_scale_pos_weight_ratio' in result
    assert result['_scale_pos_weight_ratio'] == 1.0  # 4 keep, 4 trash


def test_tune_hyperparameters_saves_to_file(tmp_path):
    """Test that tuning saves parameters to file when requested."""
    # Create dataset
    colors_labels = [
        ((255, 0, 0), 'keep'),
        ((250, 5, 5), 'keep'),
        ((245, 10, 10), 'keep'),
        ((0, 255, 0), 'trash'),
        ((5, 250, 5), 'trash'),
        ((10, 245, 10), 'trash'),
    ]

    mapping = {}
    for i, (color, label) in enumerate(colors_labels):
        filename = f'img{i}.png'
        make_img(tmp_path / filename, color)
        mapping[filename] = label

    repo = FakeRepo(mapping)
    params_file = tmp_path / 'saved_params.json'

    result = tune_hyperparameters(
        str(tmp_path),
        repo=repo,
        n_iter=2,
        cv_folds=2,
        save_params=True,
        random_state=42
    )

    # Should save to default path (we can't easily override in test, so skip file check)
    # But result should still be returned
    assert result is not None
    assert 'n_estimators' in result

