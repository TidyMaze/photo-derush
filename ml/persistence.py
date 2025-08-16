import os
import json
import joblib
from threading import Lock
from typing import List, Tuple, Dict, Any

MODEL_PATH = 'ml/personal_model.joblib'
LOG_PATH = 'ml/event_log.jsonl'

_save_lock = Lock()

def save_model(model):
    with _save_lock:
        joblib.dump(model, MODEL_PATH)

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def append_event(event):
    with _save_lock:
        with open(LOG_PATH, 'a') as f:
            f.write(json.dumps(event) + '\n')

def iter_events():
    if not os.path.exists(LOG_PATH):
        return
    with open(LOG_PATH, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # Expect well-formed single-line JSON object written by append_event
            if not (s.startswith('{') and s.endswith('}')):
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj
            # else skip non-dict (numbers, arrays, etc.)

def latest_labeled_events() -> Dict[str, dict]:
    """Return dict mapping image basename -> last event dict (latest label wins)."""
    latest = {}
    for ev in iter_events() or []:
        img = ev.get('image') or ev.get('path')
        if img:
            latest[os.path.basename(img)] = ev
    return latest

def load_latest_labeled_samples() -> Tuple[List[List[float]], List[int], List[str]]:
    """Return (X, y, images) using only the final label for each image (definitive 0/1)."""
    latest = latest_labeled_events()
    X, y, images = [], [], []
    for img, ev in latest.items():
        lbl = ev.get('label')
        feats = ev.get('features')
        if lbl in (0,1) and isinstance(feats, list):
            X.append(feats)
            y.append(lbl)
            images.append(img)
    return X, y, images

def clear_model_and_log(delete_log: bool = True):
    with _save_lock:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        if delete_log and os.path.exists(LOG_PATH):
            os.remove(LOG_PATH)

def rebuild_model_from_log(learner):
    # Use only latest label per image
    X, y, _ = load_latest_labeled_samples()
    if X and y:
        learner.partial_fit(X, y)
        save_model(learner)
    return learner
