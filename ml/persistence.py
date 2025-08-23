import os
import json
import joblib
from threading import Lock
from typing import List, Tuple, Dict
import logging

MODEL_PATH = 'ml/personal_model.joblib'
LOG_PATH = 'ml/event_log.jsonl'
FEATURE_CACHE_PATH = 'ml/feature_cache.json'

logger = logging.getLogger(__name__)

_save_lock = Lock()
_feature_cache_lock = Lock()

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
    """Iterate over the event log treating each line as a standalone JSON object.
    No attempt is made to repair or reconstruct multi-line / partial entries.
    Invalid JSON lines or non-dict JSON values are skipped silently (with debug log).
    """
    if not os.path.exists(LOG_PATH):
        return
    try:
        with open(LOG_PATH, 'r') as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        yield obj
                    else:
                        logger.info('[iter_events] Skipping non-dict JSON line')
                except Exception:
                    logger.info('[iter_events] Skipping malformed JSON line')
    except FileNotFoundError:
        return

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

def rebuild_model_from_log(learner, expected_n_features: int | None = None):
    """Replay latest labeled samples into learner.
    If expected_n_features provided, filter samples to that length to avoid schema mismatch.
    """
    X, y, _ = load_latest_labeled_samples()
    if expected_n_features is not None:
        filtered = [(x, yy) for x, yy in zip(X, y) if isinstance(x, list) and len(x) == expected_n_features]
        if filtered and len(filtered) != len(X):
            logger.info('[Rebuild] Filtered %d/%d samples due to feature length mismatch (expected=%d)', len(X)-len(filtered), len(X), expected_n_features)
        if not filtered:
            logger.info('[Rebuild] No matching samples after filtering; skipping rebuild')
            return learner
        X, y = zip(*filtered)
        X, y = list(X), list(y)
    if X and y:
        try:
            learner.train_and_explain(X, y)
        except AttributeError:
            # Fallback for legacy learner without full_retrain
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

def load_feature_cache():
    """Load persisted feature cache from disk.
    Returns dict[path] = (mtime, (fv, keys)) compatible with in-memory format.
    Stale entries (mtime mismatch or missing file) are skipped silently.
    """
    cache = {}
    if not os.path.exists(FEATURE_CACHE_PATH):
        return cache
    try:
        with open(FEATURE_CACHE_PATH, 'r') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return cache
        for path, rec in data.items():
            try:
                mtime = rec.get('mtime')
                fv = rec.get('fv')
                keys = rec.get('keys')
                if mtime is None or fv is None:
                    continue
                if not os.path.exists(path):
                    continue
                current_mtime = os.path.getmtime(path)
                if abs(current_mtime - mtime) > 1e-6:
                    continue  # file changed
                cache[path] = (mtime, (fv, keys))
            except Exception:
                continue
    except Exception:
        return {}
    return cache

def persist_feature_cache_entry(path: str, mtime: float, fv, keys):
    """Persist/append a single feature cache entry (atomic replace). Thread-safe."""
    try:
        with _feature_cache_lock:
            existing = {}
            if os.path.exists(FEATURE_CACHE_PATH):
                try:
                    with open(FEATURE_CACHE_PATH, 'r') as f:
                        existing = json.load(f) or {}
                except Exception:
                    existing = {}
            # Normalize vector to list
            try:
                import numpy as _np
                if hasattr(fv, 'tolist'):
                    fv_ser = fv.tolist()
                elif isinstance(fv, _np.ndarray):
                    fv_ser = fv.astype(float).tolist()
                else:
                    fv_ser = list(fv)
            except Exception:
                fv_ser = fv if isinstance(fv, list) else []
            existing[path] = {'mtime': mtime, 'fv': fv_ser, 'keys': keys}
            logger.info('[FeatureCache] Added/updated cache entry for %s (len=%d, keys=%s)', path, len(fv_ser) if hasattr(fv_ser, '__len__') else -1, keys[:5] if keys else None)
            tmp = FEATURE_CACHE_PATH + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(existing, f)
            os.replace(tmp, FEATURE_CACHE_PATH)
    except Exception:
        logger.info('[FeatureCache] Failed to persist entry for %s', path)

def upgrade_event_log_to_current_schema(force: bool = False) -> bool:
    """Upgrade legacy feature vectors (CV-only) in the event log to the current combined schema.
    Uses the feature vector cache for efficient extraction.
    Returns True if an upgrade was performed, False otherwise.
    """
    from ml.features import all_feature_names
    from photo_derush.feature_cache import FeatureVectorCache
    from ml.features import feature_vector
    target_len = len(all_feature_names(include_strings=False))
    latest = latest_labeled_events()
    if not latest:
        return False
    labeled = [ev for ev in latest.values() if ev.get('label') in (0,1) and isinstance(ev.get('features'), list)]
    if not labeled:
        return False
    legacy = [ev for ev in labeled if len(ev.get('features', [])) != target_len]
    if not legacy and not force:
        logger.info('[Upgrade] Event log already at current schema (len=%d)', target_len)
        return False

    # Use feature vector cache for efficient extraction
    cache = FeatureVectorCache(feature_vector)
    upgraded = 0
    for ev in labeled:
        path = ev.get('path') or ev.get('image')
        if not path or not os.path.exists(path):
            continue
        cached = cache.get(path)
        if cached and isinstance(cached, tuple) and len(cached) == 2:
            vec, keys = cached
        else:
            vec, keys = feature_vector(path, include_strings=False)
            cache.set(path, (vec, keys))
        if len(vec) == target_len:
            ev['features'] = vec.tolist() if hasattr(vec, 'tolist') else list(vec)
            upgraded += 1
    if not upgraded:
        logger.info('[Upgrade] No events re-extracted (maybe already current)')
        return False
    # Rebuild combined mapping: include unlabeled events (latest occurrence) similar to compact logic
    unlabeled = {}
    for ev in iter_events() or []:
        img = ev.get('image') or ev.get('path')
        if not img:
            continue
        base = os.path.basename(img)
        if ev.get('label') in (0,1):
            continue
        unlabeled[base] = ev
    combined = {**unlabeled, **{os.path.basename(ev.get('image') or ev.get('path')): ev for ev in latest.values()}}
    tmp_path = LOG_PATH + '.upgrade.tmp'
    with open(tmp_path, 'w') as f:
        for ev in combined.values():
            f.write(json.dumps(ev) + '\n')
    os.replace(tmp_path, LOG_PATH)
    logger.info('[Upgrade] Upgraded %d labeled events to new schema len=%d', upgraded, target_len)
    return True
