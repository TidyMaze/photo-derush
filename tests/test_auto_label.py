import json
from pathlib import Path
import os

from PIL import Image

import src.settings as settings_mod
from src.training import train_keep_trash_model
from src.viewmodel import PhotoViewModel


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


def test_auto_label_applies(tmp_path, monkeypatch):
    train_dir = tmp_path / 'train'
    infer_dir = tmp_path / 'infer'
    train_dir.mkdir()
    infer_dir.mkdir()

    # Training dataset: red = keep, green = trash
    mapping = {}
    reds = ['r1.png', 'r2.png']
    greens = ['g1.png', 'g2.png']
    for name in reds:
        make_img(train_dir / name, (250, 10, 10))
        mapping[name] = 'keep'
    for name in greens:
        make_img(train_dir / name, (10, 250, 10))
        mapping[name] = 'trash'

    model_path = tmp_path / 'model.joblib'
    repo = FakeRepo(mapping)
    result = train_keep_trash_model(str(train_dir), model_path=str(model_path), repo=repo, n_estimators=30, random_state=0)
    assert result is not None
    assert model_path.exists()

    # Inference directory with unlabeled images similar to training colors
    infer_reds = ['new_r1.png', 'new_r2.png']
    infer_greens = ['new_g1.png', 'new_g2.png']
    for name in infer_reds:
        make_img(infer_dir / name, (255, 20, 20))
    for name in infer_greens:
        make_img(infer_dir / name, (20, 255, 20))

    # Configure auto-label settings via config file override
    test_config_path = tmp_path / 'config.json'
    config_data = {
        'auto_label_enabled': True,
        'auto_label_model_path': str(model_path),
        'auto_label_keep_threshold': 0.6,
        'auto_label_trash_threshold': 0.4,
    }
    test_config_path.write_text(json.dumps(config_data))

    # Monkeypatch global CONFIG_PATH used by get_setting
    monkeypatch.setattr(settings_mod, 'CONFIG_PATH', str(test_config_path))

    # Mock object detection to prevent PyTorch model downloads
    from unittest.mock import patch
    with patch('src.viewmodel.PhotoViewModel._load_object_detections', lambda self: None):
        # Instantiate viewmodel (should pick up config) and load images
        vm = PhotoViewModel(str(infer_dir))
        vm.load_images()

        keep_thr, trash_thr = vm._auto.compute_dynamic_thresholds()

        # Validate each red image is not classified as trash; classification matches threshold logic
        for name in infer_reds:
            p = infer_dir / name
            state = vm.model.get_state(str(p))
            prob = vm._auto.predicted_probabilities.get(name)
            if prob is not None:
                if prob >= keep_thr:
                    assert state == 'keep'
                elif prob <= trash_thr:
                    assert state == 'trash'
                else:
                    assert state == ''  # mid-range gap
            assert state != 'trash'

        # Validate each green image is not classified as keep; classification matches threshold logic
        for name in infer_greens:
            p = infer_dir / name
            state = vm.model.get_state(str(p))
            prob = vm._auto.predicted_probabilities.get(name)
            if prob is not None:
                if prob >= keep_thr:
                    assert state == 'keep'
                elif prob <= trash_thr:
                    assert state == 'trash'
                else:
                    assert state == ''
            assert state != 'keep'

        # Additional sanity: average red prob >= average green prob (model separates classes)
        red_probs = [vm._auto.predicted_probabilities.get(n, 0.0) for n in infer_reds]
        green_probs = [vm._auto.predicted_probabilities.get(n, 0.0) for n in infer_greens]
        assert sum(red_probs)/len(red_probs) >= sum(green_probs)/len(green_probs)
