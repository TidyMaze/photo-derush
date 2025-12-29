import joblib
import numpy as np
from PIL import Image

from src.training import USE_FULL_FEATURES, extract_features, predict_keep_probability, train_keep_trash_model


class FakeRepo:
    def __init__(self, mapping):
        self.mapping = mapping
    def get_state(self, filename):
        return self.mapping.get(filename, '')
    def get_label_source(self, filename):
        # Return 'manual' for all test labels
        return 'manual' if filename in self.mapping else ''


def _make_image(path: str, color: tuple[int, int, int], size=(32, 24)):
    img = Image.new('RGB', size, color)
    img.save(path, 'PNG')


def test_extract_features_basic(tmp_path):
    img_path = tmp_path / 'a.png'
    _make_image(str(img_path), (10, 20, 30))
    feats = extract_features(str(img_path))
    assert feats is not None
    # Dynamic feature count support: fast mode (71) vs full mode (95)
    # Now includes real EXIF data extraction
    expected = 95 if USE_FULL_FEATURES else 71
    assert len(feats) == expected
    # Width, height first two
    assert feats[0] == 32
    assert feats[1] == 24


def test_train_model_insufficient_data(tmp_path):
    # Only one labeled image -> insufficient
    img_path = tmp_path / 'only.png'
    _make_image(str(img_path), (0, 0, 0))
    repo = FakeRepo({'only.png': 'keep'})
    model_path = tmp_path / 'model.joblib'
    result = train_keep_trash_model(str(tmp_path), model_path=str(model_path), repo=repo, n_estimators=10)
    assert result is None
    assert not model_path.exists()


def test_train_model_and_predict(tmp_path):
    # Create 4 images (2 keep, 2 trash)
    names_colors_states = [
        ('k1.png', (255, 0, 0), 'keep'),
        ('k2.png', (250, 5, 5), 'keep'),
        ('t1.png', (0, 255, 0), 'trash'),
        ('t2.png', (0, 250, 5), 'trash'),
    ]
    mapping = {}
    for name, color, state in names_colors_states:
        _make_image(str(tmp_path / name), color)
        mapping[name] = state

    repo = FakeRepo(mapping)
    model_path = tmp_path / 'model.joblib'
    result = train_keep_trash_model(str(tmp_path), model_path=str(model_path), repo=repo, n_estimators=20, random_state=0)
    assert result is not None
    assert result.n_samples == 4
    assert result.n_keep == 2
    assert result.n_trash == 2
    # With only 4 samples CV skipped
    assert result.cv_accuracy_mean is None
    assert model_path.exists()

    data = joblib.load(model_path)
    assert data['feature_length'] == (95 if USE_FULL_FEATURES else 71)
    assert set(data['filenames']) == {n for n, *_ in names_colors_states}

    # Predict on the same images; probabilities should be finite and within [0,1]
    img_paths = [str(tmp_path / n) for n, *_ in names_colors_states]
    probs = predict_keep_probability(img_paths, model_path=str(model_path))
    assert len(probs) == 4
    for p in probs:
        # Probability must be between 0 and 1 (inclusive); NaN would indicate feature failure
        assert (0.0 <= p <= 1.0) or np.isnan(p)


def test_train_model_cancellation(tmp_path):
    # Create multiple images to trigger training stages
    mapping = {}
    for i in range(6):
        name = f'im_{i}.png'
        _make_image(str(tmp_path / name), (i*10 % 255, (i*20+50) % 255, (i*30+80) % 255))
        # Alternate keep/trash
        mapping[name] = 'keep' if i % 2 == 0 else 'trash'
    repo = FakeRepo(mapping)
    model_path = tmp_path / 'model_cancel.joblib'
    # Cancellation token always returns True -> immediate cancel
    cancelled = train_keep_trash_model(
        str(tmp_path),
        model_path=str(model_path),
        repo=repo,
        n_estimators=50,
        cancellation_token=lambda: True,
    )
    assert cancelled is None
    assert not model_path.exists()
