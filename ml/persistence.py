import os
import json
import joblib
from threading import Lock

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
            yield json.loads(line)

def clear_model_and_log():
    with _save_lock:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        # Do not delete log by default (for rebuild)

def rebuild_model_from_log(learner):
    X, y = [], []
    for event in iter_events():
        if 'features' in event and 'label' in event:
            label = event['label']
            # Only use definitive labels (0 trash, 1 keep) for model training
            if label in (0, 1):
                X.append(event['features'])
                y.append(label)
    if X and y:
        learner.partial_fit(X, y)
        save_model(learner)
    return learner
