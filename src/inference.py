"""Inference utilities: load model & predict probabilities."""

from __future__ import annotations

import logging
import os
import pickle
import threading
from dataclasses import dataclass
from typing import Any, Optional

import joblib
import numpy as np

from .features import (  # added extract_features
    FEATURE_COUNT,
    USE_FULL_FEATURES,
    batch_extract_features,
    extract_features,
)
from .model_version import validate_model_metadata
# Lazy import: tuning imports sklearn which is heavy (~1.6s startup time)

# Module-level cache for loaded models
_model_bundle_cache: dict[str, ModelBundle] = {}
_model_bundle_lock = threading.Lock()


@dataclass
class ModelBundle:
    model: Any
    meta: dict
    calibrator: Optional[Any] = None


def invalidate_model_cache(model_path: str | None = None):
    """Invalidate the model cache for a specific path or all paths.

    Call this after retraining to ensure the new model is loaded.
    """
    global _model_bundle_cache, _model_bundle_lock
    with _model_bundle_lock:
        if model_path:
            if model_path in _model_bundle_cache:
                del _model_bundle_cache[model_path]
                logging.info("[inference] Invalidated cache for %s", model_path)
        else:
            _model_bundle_cache.clear()
            logging.info("[inference] Invalidated all model caches")


def check_model_health(model_path: str | None = None) -> bool:
    """Check if model is healthy (has non-zero feature importances).
    
    Returns True if model is healthy, False otherwise.
    Supports both XGBoost and CatBoost models.
    """
    # Lazy import to avoid loading xgboost at module level
    if model_path is None:
        from .training_core import DEFAULT_MODEL_PATH
        model_path = DEFAULT_MODEL_PATH
    try:
        loaded = load_model(model_path)
        if loaded is None:
            return False
        model = loaded.model
        if hasattr(model, 'named_steps'):
            classifier = model.named_steps.get('xgb') or model.named_steps.get('cat')
            if classifier and hasattr(classifier, 'feature_importances_'):
                importances_sum = classifier.feature_importances_.sum()
                if importances_sum == 0:
                    logging.warning("[inference] Model has zero feature importances - model is broken")
                    return False
        return True
    except Exception as e:
        logging.warning("[inference] Failed to check model health: %s", e)
        return False


def load_model(model_path: str | None = None) -> Optional[ModelBundle]:
    # Lazy import to avoid loading xgboost at module level
    if model_path is None:
        from .training_core import DEFAULT_MODEL_PATH
        model_path = DEFAULT_MODEL_PATH
    # Cache loaded ModelBundle per-model-path to avoid repeated heavy loads
    global _model_bundle_cache, _model_bundle_lock
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Return cached bundle if already loaded in this process
    with _model_bundle_lock:
        if model_path in _model_bundle_cache:
            logging.debug("[inference] Returning cached model %s", model_path)
            return _model_bundle_cache[model_path]

    logging.info("[inference] Loading model %s", model_path)

    try:
        data = joblib.load(model_path)
    except (EOFError, KeyError, pickle.UnpicklingError, ValueError) as e:
        logging.error("[inference] Corrupted model file %s: %s", model_path, e)
        return None
    except Exception as e:
        logging.error("[inference] Failed to load model %s: %s", model_path, e)
        return None

    # Check for metadata (v2 format) and validate
    metadata = data.get("__metadata__")
    if metadata:
        # Lazy import: tuning imports sklearn which is heavy
        from .tuning import load_best_params
        current_params = load_best_params() or {}
        is_valid, mismatches = validate_model_metadata(
            metadata,
            current_feature_count=FEATURE_COUNT,
            current_mode="FULL" if USE_FULL_FEATURES else "FAST",
            current_params=current_params,
        )
        if not is_valid:
            for mismatch in mismatches:
                logging.warning("[inference] %s", mismatch)
            if "Feature count" in str(mismatches):
                logging.error("[inference] Feature count mismatch detected; model may be stale")

    model_feature_count = data.get("feature_length", 0)
    if model_feature_count != FEATURE_COUNT:
        # If model expects different feature length but contains PCA/embeddings metadata,
        # allow loading so combined models can be used (they will require image embeddings at predict time).
        if not USE_FULL_FEATURES and model_feature_count == 71 and FEATURE_COUNT == 71:
            logging.info("[inference] Loaded legacy fast model (71 features)")
        else:
            # Allow models with embeddings, PCA, or feature interactions
            has_embeddings = data.get("pca") is not None or data.get("pca_dim") is not None or metadata.get("has_embeddings")
            has_interactions = metadata.get("has_feature_interactions") or data.get("feature_transformer") is not None
            if has_embeddings or has_interactions or data.get("model_type") == "combined":
                logging.info(
                    "[inference] Loading model with differing feature length (%d vs %d) as combined model (embeddings=%s, interactions=%s)",
                    model_feature_count,
                    FEATURE_COUNT,
                    has_embeddings,
                    has_interactions,
                )
            else:
                logging.warning(
                    "[inference] Outdated model (model=%d expected=%d) not suitable for current features: %s",
                    model_feature_count,
                    FEATURE_COUNT,
                    model_path,
                )
                # Do not delete automatically; return None to signal incompatible model
                return None
    logging.info(
        "[inference] Metadata: %s",
        {k: v for k, v in data.items() if k not in ("model", "filenames", "feature_importances")},
    )
    # Attempt to load an associated calibrator (model_path + '.calib.joblib') if present
    calibrator = None
    calib_path = f"{model_path}.calib.joblib"
    if os.path.isfile(calib_path):
        try:
            calibrator = joblib.load(calib_path)
            logging.info("[inference] Loaded calibrator %s", calib_path)
        except Exception as e:
            logging.warning("[inference] Failed to load calibrator %s: %s", calib_path, e)

    bundle = ModelBundle(model=data["model"], meta=data, calibrator=calibrator)
    try:
        with _model_bundle_lock:
            _model_bundle_cache[model_path] = bundle
    except Exception:
        pass
    return bundle


def load_ensemble_models():
    """Load both baseline and combined models if available for ensemble prediction.

    Returns a tuple of ModelBundle or None: (baseline_bundle|None, combined_bundle|None)
    """
    # Lazy import to avoid loading xgboost at module level
    from .training_core import DEFAULT_MODEL_PATH
    baseline_path = os.path.join(os.path.dirname(DEFAULT_MODEL_PATH), "baseline_current.joblib")
    combined_path = os.path.join(os.path.dirname(DEFAULT_MODEL_PATH), "combined_current.joblib")

    baseline = None
    combined = None

    if os.path.isfile(baseline_path):
        try:
            baseline = load_model(baseline_path)
            logging.info("[inference] Loaded baseline model for ensemble")
        except Exception as e:
            logging.warning("[inference] Failed to load baseline model for ensemble: %s", e)

    if os.path.isfile(combined_path):
        try:
            combined = load_model(combined_path)
            logging.info("[inference] Loaded combined model for ensemble")
        except Exception as e:
            logging.warning("[inference] Failed to load combined model for ensemble: %s", e)

    return baseline, combined


def predict_keep_probability(
    image_paths: list[str], model_path: str | None = None, progress_callback=None
) -> list[float]:
    # Lazy import to avoid loading xgboost at module level
    if model_path is None:
        from .training_core import DEFAULT_MODEL_PATH
        model_path = DEFAULT_MODEL_PATH
    try:
        # Try to load ensemble models first
        baseline_bundle, combined_bundle = load_ensemble_models()
        if baseline_bundle and combined_bundle:
            logging.info("[inference] Using ensemble of baseline + combined models")
            # Predict with both models and average
            probs_baseline = _predict_with_model(
                image_paths, baseline_bundle.model, baseline_bundle.meta, baseline_bundle.calibrator, progress_callback
            )
            probs_combined = _predict_with_model(
                image_paths, combined_bundle.model, combined_bundle.meta, combined_bundle.calibrator, progress_callback
            )

            # Average probabilities
            probs = []
            for b, c in zip(probs_baseline, probs_combined):
                if b == b and c == c:  # both not nan
                    probs.append((b + c) / 2)
                elif b == b:  # baseline valid
                    probs.append(b)
                elif c == c:  # combined valid
                    probs.append(c)
                else:
                    probs.append(float("nan"))
            return probs
        else:
            # Fall back to single model
            loaded = load_model(model_path)
            if loaded is None:
                return [float("nan")] * len(image_paths)
            model, meta, calibrator = loaded.model, loaded.meta, loaded.calibrator
            return _predict_with_model(image_paths, model, meta, calibrator, progress_callback)
    except Exception as e:
        logging.warning("[inference] Failed to load model %s: %s", model_path, e)
        return [float("nan")] * len(image_paths)


def _predict_with_model(image_paths: list[str], model, meta, calibrator, progress_callback=None) -> list[float]:
    """Predict keep probabilities using a single model."""
    # Check if model is broken (zero feature importances) - supports both XGBoost and CatBoost
    if hasattr(model, 'named_steps'):
        classifier = model.named_steps.get('xgb') or model.named_steps.get('cat')
        if classifier and hasattr(classifier, 'feature_importances_'):
            if classifier.feature_importances_.sum() == 0:
                logging.warning("[inference] Model has zero feature importances - returning NaN predictions")
                return [float("nan")] * len(image_paths)
    
    # Also check if calibrator exists but model is broken - don't use calibrator on broken model
    if calibrator is not None:
        if hasattr(model, 'named_steps'):
            classifier = model.named_steps.get('xgb') or model.named_steps.get('cat')
            if classifier and hasattr(classifier, 'feature_importances_'):
                if classifier.feature_importances_.sum() == 0:
                    logging.warning("[inference] Model broken - ignoring calibrator and returning NaN")
                    calibrator = None  # Don't use broken calibrator
    
    model_feature_len = int(meta.get("feature_length", FEATURE_COUNT) or FEATURE_COUNT)
    # If model expects extra embedding dims (combined model), we'll compute embeddings below and concat
    is_combined_model = model_feature_len != FEATURE_COUNT
    if is_combined_model:
        # allow legacy fast mapping
        if not USE_FULL_FEATURES and model_feature_len == 71 and FEATURE_COUNT == 71:
            logging.info("[inference] Using legacy fast model (71 features)")
            is_combined_model = False
    total = len(image_paths)
    if total == 0:
        return []
    logging.info("[inference] Predict %d images", total)
    internal_logging = progress_callback is None and total > 0
    if internal_logging:
        logging.info("[inference] start feature-extraction total=%d", total)

        # Wrap an internal callback to reuse batch extraction progress
        def _internal_cb(current, tot, detail):
            if tot:  # log every ~10% or final
                interval = max(1, tot // 10)
                if current % interval == 0 or current == tot:
                    logging.info("[inference] progress %d/%d %s", current, tot, detail)

        all_features = batch_extract_features(image_paths, progress_callback=_internal_cb)
    else:
        all_features = batch_extract_features(image_paths, progress_callback=progress_callback)
    valid_idx = []
    matrix = []
    for idx, feats in enumerate(all_features):
        if feats and len(feats) == FEATURE_COUNT:
            valid_idx.append(idx)
            matrix.append(feats)
        elif feats and len(feats) != FEATURE_COUNT:
            logging.warning(
                "[inference] Length mismatch %s got %d expected %d", image_paths[idx], len(feats), FEATURE_COUNT
            )
    probs = [float("nan")] * total
    if matrix and hasattr(model, "predict_proba"):
        try:
            X = np.array(matrix, dtype=float)
            # Apply feature subset selection if model uses it
            feature_indices = meta.get("feature_indices")
            if feature_indices is not None and len(feature_indices) > 0:
                logging.info("[inference] Using feature subset: %d features", len(feature_indices))
                X = X[:, feature_indices]
            elif feature_indices is not None and len(feature_indices) == 0:
                logging.warning("[inference] Empty feature_indices detected, using all %d features", FEATURE_COUNT)
            # If combined model (expects embeddings), compute embeddings and concat
            if is_combined_model:
                try:
                    # Load/compute embeddings for the valid subset of images
                    img_paths = [image_paths[i] for i in valid_idx]
                    emb = _compute_image_embeddings(img_paths)
                    # If model stored a PCA transformer, apply it
                    pca = meta.get("pca")
                    if pca is not None:
                        try:
                            emb = pca.transform(emb)
                        except Exception:
                            logging.exception("Error in inference postprocessing")
                            raise
                    # If embedding dims don't match, try to pad/truncate
                    expected_emb_len = model_feature_len - FEATURE_COUNT
                    if emb.shape[1] != expected_emb_len:
                        if emb.shape[1] > expected_emb_len:
                            emb = emb[:, :expected_emb_len]
                        else:
                            pad = np.zeros((emb.shape[0], expected_emb_len - emb.shape[1]), dtype=float)
                            emb = np.hstack([emb, pad])
                    X = np.hstack([X, emb])
                except Exception as e:
                    logging.warning("[inference] Failed computing/aligning embeddings for combined model: %s", e)
            
            # Apply feature interactions if model has them
            feature_transformer = meta.get("feature_transformer")
            if feature_transformer is not None:
                try:
                    # Check if we have base features (before interactions)
                    n_base = meta.get("n_base_features")
                    if n_base and X.shape[1] == n_base:
                        X = feature_transformer.transform(X)
                        logging.debug("[inference] Applied feature interactions: %d → %d features", n_base, X.shape[1])
                except Exception as e:
                    logging.warning("[inference] Failed applying feature interactions: %s", e)
            
            if calibrator is not None and hasattr(calibrator, "predict_proba"):
                # Use calibrator on scaled data if calibrator expects already-scaled features
                try:
                    # If calibrator was built with cv='prefit' it expects raw xgb input; try predict_proba directly
                    preds = calibrator.predict_proba(X)[:, 1]
                except Exception:
                    # Fall back to model
                    preds = model.predict_proba(X)[:, 1]
            else:
                preds = model.predict_proba(X)[:, 1]
            for vi, p in zip(valid_idx, preds):
                probs[vi] = float(p)
            logging.info("[inference] Batch prediction done: %d/%d", len(valid_idx), total)
        except Exception as e:
            logging.error("[inference] Batch prediction failed: %s", e)
    elif not hasattr(model, "predict_proba"):
        logging.warning("[inference] Model lacks predict_proba")
    return probs


# --- Streaming incremental prediction ---
def predict_keep_probability_stream(
    image_paths: list[str], model_path: str | None = None, per_prediction_callback=None, progress_callback=None
) -> list[float]:
    """Stream keep probabilities image-by-image, invoking per_prediction_callback(fname, prob, idx, total).
    Falls back to batch function if model load fails. Returns full list of probabilities.
    """
    # Lazy import to avoid loading xgboost at module level
    if model_path is None:
        from .training_core import DEFAULT_MODEL_PATH
        model_path = DEFAULT_MODEL_PATH
    try:
        loaded = load_model(model_path)
        if loaded is None:
            return [float("nan")] * len(image_paths)
        model, meta, calibrator = loaded.model, loaded.meta, loaded.calibrator
        
        # Check if model is broken (zero feature importances) - supports both XGBoost and CatBoost
        if hasattr(model, 'named_steps'):
            classifier = model.named_steps.get('xgb') or model.named_steps.get('cat')
            if classifier and hasattr(classifier, 'feature_importances_'):
                if classifier.feature_importances_.sum() == 0:
                    logging.warning("[inference-stream] Model has zero feature importances - returning NaN predictions")
                    return [float("nan")] * len(image_paths)
        
        # Don't use calibrator on broken model
        if calibrator is not None:
            if hasattr(model, 'named_steps'):
                classifier = model.named_steps.get('xgb') or model.named_steps.get('cat')
                if classifier and hasattr(classifier, 'feature_importances_'):
                    if classifier.feature_importances_.sum() == 0:
                        logging.warning("[inference-stream] Model broken - ignoring calibrator")
                        calibrator = None
    except Exception as e:
        logging.warning("[inference] Failed to load model %s: %s", model_path, e)
        return [float("nan")] * len(image_paths)
    model_feature_len = int(meta.get("feature_length", FEATURE_COUNT) or FEATURE_COUNT)
    if model_feature_len != FEATURE_COUNT:
        if not USE_FULL_FEATURES and model_feature_len == 71 and FEATURE_COUNT == 71:
            logging.info("[inference] Using legacy fast model (71 features)")
        else:
            logging.error("[inference] Feature count mismatch model=%d expected=%d", model_feature_len, FEATURE_COUNT)
            return [float("nan")] * len(image_paths)
    total = len(image_paths)
    logging.info("[inference-stream] Predict streaming %d images", total)
    import time
    stream_start = time.perf_counter()
    
    # Pre-extract features in batches to avoid blocking on slow uncached images
    # This uses parallel extraction for >10 images, which is much faster
    from .features import batch_extract_features
    feat_extract_start = time.perf_counter()
    if progress_callback:
        progress_callback(0, total, "extracting features")
    all_feats = batch_extract_features(image_paths, progress_callback=progress_callback)
    feat_extract_end = time.perf_counter()
    
    probs: list[float] = [float("nan")] * total
    for idx, path in enumerate(image_paths):
        pred_start = time.perf_counter()
        if progress_callback:
            progress_callback(idx, total, f"predicting {idx+1}/{total}")
        feats = all_feats[idx]  # Use pre-extracted features
        feat_time = time.perf_counter()
        prob = float("nan")
        if feats and len(feats) == FEATURE_COUNT and hasattr(model, "predict_proba"):
            try:
                import numpy as np

                arr = np.array([feats], dtype=float)
                # Apply feature subset selection if model uses it
                feature_indices = meta.get("feature_indices")
                if feature_indices is not None and len(feature_indices) > 0:
                    arr = arr[:, feature_indices]
                elif feature_indices is not None and len(feature_indices) == 0:
                    logging.debug("[inference-stream] Empty feature_indices, using all features")
                if calibrator is not None and hasattr(calibrator, "predict_proba"):
                    try:
                        pred = calibrator.predict_proba(arr)[0, 1]
                    except Exception:
                        pred = model.predict_proba(arr)[0, 1]
                else:
                    pred = model.predict_proba(arr)[0, 1]
                prob = float(pred)
            except Exception as e:
                logging.debug("[inference-stream] predict failed %s: %s", path, e)
        pred_time = time.perf_counter()
        probs[idx] = prob
        if per_prediction_callback:
            try:
                from os import path as osp

                per_prediction_callback(osp.basename(path), prob, idx + 1, total)
            except Exception as e:
                logging.debug("[inference-stream] per_prediction_callback failed: %s", e)
        # Log every 10th prediction or first/last
        if idx % 10 == 0 or idx == 0 or idx == total - 1:
            logging.debug("[inference-stream] Predicted %d/%d: %s -> %.3f", idx + 1, total, path, prob)
    
    stream_end = time.perf_counter()
    if progress_callback:
        progress_callback(total, total, "done")
    logging.info("[inference-stream] Done streaming predictions %d/%d", sum(p == p for p in probs), total)
    return probs


__all__ = ["load_model", "predict_keep_probability", "predict_keep_probability_stream"]


# Module-level cache for embedding model and device
_embedding_model_cache: dict[str, Any] = {}
_embedding_device: str | None = None
_embedding_model_lock = threading.Lock()


def _get_embedding_device(device: str = "auto") -> str:
    """Auto-detect best available device for embeddings (similar to object detection).
    
    Args:
        device: Device string ("auto", "cpu", "cuda", "mps")
    
    Returns:
        Device string ("cpu", "cuda", or "mps")
    """
    if device != "auto":
        return device
    
    # Allow forcing device via environment variable
    env_device = os.environ.get("EMBEDDING_DEVICE")
    if env_device:
        return env_device
    
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    
    return "cpu"


def _get_embedding_device_used() -> str:
    """Get the device currently used for embeddings."""
    global _embedding_device
    return _embedding_device or "cpu"


def _compute_image_embeddings(image_paths: list[str], device: str = "auto") -> np.ndarray:
    """Compute or fetch embeddings for a list of image paths.

    Strategy:
    - If a cached embeddings file exists at `.cache/embeddings_resnet18_full.joblib` or similar, load and align.
    - Otherwise, attempt to import torch/torchvision and build embeddings on-the-fly (best-effort).
    - If neither is available, return zero vectors.
    
    Args:
        image_paths: List of image file paths
        device: Device to use ("auto", "cpu", "cuda", "mps"). Defaults to "auto" (auto-detect).
    
    Returns:
        numpy array of embeddings (n_images, embedding_dim)
    """
    global _embedding_model_cache, _embedding_device, _embedding_model_lock
    
    # Try cached embeddings in common path
    possible = [
        os.path.join(".cache", "embeddings_resnet18_full.joblib"),
        os.path.join(".cache", "embeddings_resnet18.joblib"),
    ]
    for p in possible:
        if os.path.isfile(p):
            try:
                ed = joblib.load(p)
                emb = ed.get("embeddings") or ed.get("emb")
                ef = ed.get("filenames")
                if emb is None or ef is None:
                    continue
                # Align by basename
                fmap = {os.path.basename(f): i for i, f in enumerate(ef)}
                rows = []
                for ip in image_paths:
                    b = os.path.basename(ip)
                    if b in fmap:
                        rows.append(emb[fmap[b]])
                    else:
                        rows.append(np.zeros((emb.shape[1],), dtype=float))
                return np.vstack(rows)
            except Exception:
                continue

    # Try to compute on the fly using torch if available
    # During pytest avoid importing torch / heavy computation
    if os.environ.get("PYTEST_CURRENT_TEST"):
        logging.info("[inference] PYTEST detected: skipping torch embeddings and returning zeros")
        return np.zeros((len(image_paths), 128), dtype=float)

    try:
        import torch
        from PIL import Image
        from torchvision import models, transforms

        # Auto-detect device
        effective_device = _get_embedding_device(device)
        _embedding_device = effective_device
        
        # Cache model per device to avoid reloading
        with _embedding_model_lock:
            if effective_device not in _embedding_model_cache:
                logging.info(f"[inference] Loading ResNet18 for embeddings on {effective_device}")
                model = models.resnet18(pretrained=True)
                # Remove final classifier
                model = torch.nn.Sequential(*list(model.children())[:-1])
                model.eval()
                # Move model to device
                model = model.to(effective_device)
                _embedding_model_cache[effective_device] = model
            else:
                model = _embedding_model_cache[effective_device]

        def _embed(p):
            # OPTIMIZATION: Use shared image cache to avoid repeated file opens
            from .image_cache import get_cached_image
            cached_img = get_cached_image(p)
            if cached_img is None:
                return None
            img = cached_img.convert("RGB")
            tf = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            # Apply transforms (returns tensor after ToTensor)
            tensor_result = tf(img)
            # Type assertion: ToTensor() in Compose ensures this is a tensor
            if not isinstance(tensor_result, torch.Tensor):
                raise RuntimeError("Expected tensor from transforms")
            # Add batch dimension and move to device
            t = tensor_result.unsqueeze(0).to(effective_device)
            with torch.no_grad():
                out = model(t)
            # Move result back to CPU for numpy conversion
            return out.cpu().squeeze().numpy().reshape(1, -1)

        mats = []
        for p in image_paths:
            try:
                mats.append(_embed(p))
            except Exception:
                mats.append(np.zeros((512,), dtype=float).reshape(1, -1))
        return np.vstack(mats)
    except Exception:
        # Fallback: zeros with 128 dims (reasonable default)
        return np.zeros((len(image_paths), 128), dtype=float)
