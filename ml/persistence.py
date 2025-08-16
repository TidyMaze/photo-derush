import os
import json
import joblib
from threading import Lock
from typing import List, Tuple, Dict, Any
import logging

MODEL_PATH = 'ml/personal_model.joblib'
LOG_PATH = 'ml/event_log.jsonl'

logger = logging.getLogger(__name__)

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
    """Robust iterator over event log.
    Reconstructs JSON objects that may span multiple lines (legacy/corrupted writes) by counting braces.
    Skips malformed fragments and non-dict JSON values.
    """
    if not os.path.exists(LOG_PATH):
        return
    with open(LOG_PATH, 'r') as f:
        buf_lines = []
        brace_depth = 0
        for raw in f:
            line = raw.rstrip('\n')
            stripped = line.strip()
            if not buf_lines:
                # Start of a potential JSON object
                if stripped.startswith('{'):
                    buf_lines.append(line)
                    brace_depth = line.count('{') - line.count('}')
                    if brace_depth == 0:  # single-line object
                        text = ''.join(buf_lines)
                        try:
                            obj = json.loads(text)
                            if isinstance(obj, dict):
                                yield obj
                        except Exception:
                            logger.debug("[iter_events] Skipping malformed single-line JSON")
                        buf_lines.clear()
                        brace_depth = 0
                else:
                    # Ignore stray lines (e.g., orphaned numbers from interrupted writes)
                    continue
            else:
                buf_lines.append(line)
                brace_depth += line.count('{') - line.count('}')
                if brace_depth <= 0:  # object closed
                    text = '\n'.join(buf_lines)
                    try:
                        obj = json.loads(text)
                        if isinstance(obj, dict):
                            yield obj
                        else:
                            logger.debug("[iter_events] Non-dict JSON skipped")
                    except Exception:
                        logger.debug("[iter_events] Skipping malformed multi-line JSON block")
                    buf_lines.clear()
                    brace_depth = 0
        # If file ends mid-object, drop buffer silently

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

def compact_event_log(keep_unlabeled: bool = True):
    """Rewrite event log keeping only the latest event per image (and optionally unlabeled entries).
    Useful to clean up corrupted partial lines or reduce file size. Atomic rewrite via temp file.
    """
    if not os.path.exists(LOG_PATH):
        return
    latest = latest_labeled_events()
    # Optionally include unlabeled events (those without a definitive label) by scanning original
    unlabeled = {}
    if keep_unlabeled:
        for ev in iter_events() or []:
            img = ev.get('image') or ev.get('path')
            if not img:
                continue
            if ev.get('label') in (0,1):
                continue  # already covered by latest
            # keep last occurrence
            unlabeled[os.path.basename(img)] = ev
    combined = {**unlabeled, **latest}  # latest labels override unlabeled
    tmp_path = LOG_PATH + '.tmp'
    with open(tmp_path, 'w') as out:
        for ev in combined.values():
            out.write(json.dumps(ev) + '\n')
    os.replace(tmp_path, LOG_PATH)
    logger.info('[Log] Compacted event log: %d entries (latest + %s unlabeled)', len(combined), 'with' if keep_unlabeled else 'without')
