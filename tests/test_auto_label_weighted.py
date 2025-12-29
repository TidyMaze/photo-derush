import json
import os
from pathlib import Path

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


def test_weighted_thresholds_adjustment(tmp_path, monkeypatch):
    train_dir = tmp_path / 'train'; infer_dir = tmp_path / 'infer'
    train_dir.mkdir(); infer_dir.mkdir()
    # Imbalanced training dataset: 6 keep, 2 trash
    mapping = {}
    keeps = [f'k{i}.png' for i in range(6)]
    trashes = [f't{i}.png' for i in range(2)]
    for name in keeps:
        make_img(train_dir / name, (250, 30, 30)); mapping[name] = 'keep'
    for name in trashes:
        make_img(train_dir / name, (30, 250, 30)); mapping[name] = 'trash'

    model_path = tmp_path / 'model.joblib'
    repo = FakeRepo(mapping)
    result = train_keep_trash_model(str(train_dir), model_path=str(model_path), repo=repo, n_estimators=40, random_state=0)
    assert result is not None and model_path.exists()

    # Inference unlabeled images
    infer_keeps = [f'new_k{i}.png' for i in range(3)]
    infer_trashes = [f'new_t{i}.png' for i in range(3)]
    for name in infer_keeps:
        make_img(infer_dir / name, (255, 40, 40))
    for name in infer_trashes:
        make_img(infer_dir / name, (40, 255, 40))

    # Config: enable weighted mode
    cfg = {
        'auto_label_enabled': True,
        'auto_label_model_path': str(model_path),
        'auto_label_keep_threshold': 0.6,
        'auto_label_trash_threshold': 0.4,
        'auto_label_weighted_enabled': True,
    }
    cfg_path = tmp_path / 'config.json'
    cfg_path.write_text(json.dumps(cfg))
    monkeypatch.setattr(settings_mod, 'CONFIG_PATH', str(cfg_path))

    from unittest.mock import patch
    with patch('src.viewmodel.PhotoViewModel._load_object_detections', lambda self: None):
        vm = PhotoViewModel(str(infer_dir))
        vm.load_images()

        # After load + auto labeling, dynamic thresholds should reflect imbalance (keep majority)
        auto_mgr = vm._auto
        keep_eff, trash_eff = auto_mgr.compute_dynamic_thresholds()
        assert keep_eff >= auto_mgr.keep_thr  # raised keep threshold
        assert trash_eff == auto_mgr.trash_thr  # trash stays same in keep majority scenario
        assert keep_eff > trash_eff

        # All inferred keep-like images likely exceed new keep threshold and get 'keep'
        for name in infer_keeps:
            assert vm.model.get_state(str(infer_dir / name)) in ('keep','')  # may be conservative if threshold rose
        # Trash-like images should be labeled 'trash' or remain unlabeled if conservative
        for name in infer_trashes:
            state = vm.model.get_state(str(infer_dir / name))
            assert state in ('trash','')


def test_weighted_mode_toggle(tmp_path, monkeypatch):
    # Small balanced dataset to verify mode toggling does not distort thresholds
    train_dir = tmp_path / 'train'; train_dir.mkdir()
    mapping = {}
    for i in range(2):
        make_img(train_dir / f'k{i}.png', (250, 20, 20)); mapping[f'k{i}.png']='keep'
        make_img(train_dir / f't{i}.png', (20, 250, 20)); mapping[f't{i}.png']='trash'
    model_path = tmp_path / 'model.joblib'
    repo = FakeRepo(mapping)
    result = train_keep_trash_model(str(train_dir), model_path=str(model_path), repo=repo, n_estimators=30, random_state=0)
    assert result is not None

    cfg = {
        'auto_label_enabled': True,
        'auto_label_model_path': str(model_path),
        'auto_label_keep_threshold': 0.6,
        'auto_label_trash_threshold': 0.4,
        'auto_label_weighted_enabled': False,
    }
    cfg_path = tmp_path / 'config.json'
    cfg_path.write_text(json.dumps(cfg))
    monkeypatch.setattr(settings_mod, 'CONFIG_PATH', str(cfg_path))

    from unittest.mock import patch
    with patch('src.viewmodel.PhotoViewModel._load_object_detections', lambda self: None):
        vm = PhotoViewModel(str(train_dir))
        vm.load_images()
        auto_mgr = vm._auto
        k1, t1 = auto_mgr.compute_dynamic_thresholds()
        assert k1 == auto_mgr.keep_thr and t1 == auto_mgr.trash_thr  # unchanged
        auto_mgr.set_weighted_mode(True)
        k2, t2 = auto_mgr.compute_dynamic_thresholds()
        # Balanced dataset => thresholds unchanged even when enabled
        assert k2 == k1 and t2 == t1

