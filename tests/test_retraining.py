import json
import time
from pathlib import Path

import joblib
from PIL import Image

import src.settings as settings_mod
import src.training as training_mod
from src.training import TrainingResult
from src.viewmodel import PhotoViewModel
import logging

log = logging.getLogger(__name__)


def _mkimg(path: Path, color):
    img = Image.new('RGB', (24, 24), color)
    img.save(path, 'PNG')


def test_retraining_on_label_changes(tmp_path, monkeypatch):
    # Prepare directory with 4 images
    img_dir = tmp_path / 'imgs'
    img_dir.mkdir()
    names = ['a.png', 'b.png', 'c.png', 'd.png']
    colors = [(255,0,0),(254,1,1),(0,255,0),(1,254,1)]
    for n,c in zip(names, colors):
        _mkimg(img_dir / n, c)

    model_path = tmp_path / 'model.joblib'

    # Config with model path so viewmodel picks it up
    config_path = tmp_path / 'config.json'
    cfg = {
        'auto_label_model_path': str(model_path),
        'auto_label_enabled': False,  # auto-label not required for retrain
        'auto_label_keep_threshold': 0.8,
        'auto_label_trash_threshold': 0.2,
    }
    config_path.write_text(json.dumps(cfg))
    monkeypatch.setattr(settings_mod, 'CONFIG_PATH', str(config_path))

    # Stub train_keep_trash_model to speed up and observe calls
    call_counter = {'n':0}
    def fast_train(directory, model_path, random_state=42, repo=None, n_estimators=120, progress_callback=None):
        call_counter['n'] += 1
        log.debug("fast_train called; call_counter=%s", call_counter['n'])
        from src.training import FEATURE_COUNT
        data = {
            'model': None,
            'feature_length': FEATURE_COUNT,  # use current count to avoid auto-delete
            'n_samples': 4,
            'n_keep': 2,
            'n_trash': 2,
            'filenames': names,
        }
        joblib.dump(data, model_path)
        return TrainingResult(model_path=str(model_path), n_samples=4, n_keep=2, n_trash=2, cv_accuracy_mean=None, cv_accuracy_std=None)
    monkeypatch.setattr(training_mod, 'train_keep_trash_model', fast_train)

    vm = PhotoViewModel(str(img_dir))
    log.debug("PhotoViewModel initialized; images=%s", vm.images)
    # Speed up throttle
    vm._retrain_min_interval = 0.05

    # Select and label two images quickly to test pending logic
    for fname,label in [('a.png','keep'),('b.png','trash')]:
        vm.select_image(fname)
        log.debug("selected %s", fname)
        vm.set_label(label)
        log.debug("set_label %s", label)

    # Wait for retrain threads
    deadline = time.time() + 2
    while time.time() < deadline and call_counter['n'] < 1:
        time.sleep(0.02)
        log.debug("waiting for first retrain; so far=%s", call_counter['n'])
    assert call_counter['n'] >= 1, 'Expected at least one retrain'
    assert model_path.exists()

    # Trigger another label change while training may have completed
    vm.select_image('c.png')
    log.debug("selected c.png")
    vm.set_label('keep')
    log.debug("set_label keep for c.png")
    # Ensure second retrain eventually happens (pending or new)
    target_calls = call_counter['n'] + 1
    deadline = time.time() + 2
    while time.time() < deadline and call_counter['n'] < target_calls:
        time.sleep(0.02)
    assert call_counter['n'] >= target_calls, 'Expected second retrain after new label'
