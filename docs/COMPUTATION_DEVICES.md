# Heavy Computations and Device Usage

## Summary

This document lists all heavy computations in the project, which device they run on, and their current UI visibility.

## 1. Object Detection (YOLOv8)

**Location:** `src/object_detection.py`

**Device Selection:**
- **Auto-detection logic** (lines 479-497):
  1. CUDA (NVIDIA GPU) - if available
  2. MPS (Apple Silicon GPU) - if available
  3. CPU - fallback
- **macOS MPS override:** YOLOv8 on MPS is unstable, so it's forced to CPU on macOS (line 492-497)

**Current Device:**
- Detected automatically on first model load
- Stored in `_detection_ctx.device_map`
- Can be overridden via `DETECTION_DEVICE` environment variable

**UI Display:**
- ✅ **Currently shown** in status bar: "Device: {device}"
- Location: `src/view.py` line 656, 813-830
- Updates from `viewmodel.py` line 1038-1069

**Code Reference:**
```479:497:src/object_detection.py
# Auto-detect best available device
if device == "auto":
    if torch and torch.cuda.is_available():  # type: ignore[attr-defined]
        device = "cuda"
    elif torch and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        device = "mps"
    else:
        device = "cpu"

effective_backend = DETECTION_BACKEND

# On macOS MPS, prefer CPU for known unstable backends
global _mps_warning_logged
if platform.system() == "Darwin" and device == "mps":
    if effective_backend == "yolov8":
        if not _mps_warning_logged:
            logging.debug("YOLOv8 on MPS on macOS can be unstable; forcing device='cpu' for this backend")
            _mps_warning_logged = True
        device = "cpu"
```

---

## 2. Image Embeddings (ResNet18)

**Location:** `src/inference.py` `_compute_image_embeddings()`

**Device Selection:**
- ✅ **Auto-detection implemented** - similar to object detection
- Auto-detects: CUDA → MPS → CPU
- Model and tensors moved to detected device
- Can be overridden via `EMBEDDING_DEVICE` environment variable

**Current Device:**
- Detected automatically on first model load
- Stored in module-level `_embedding_device` variable
- Model cached per device to avoid reloading

**UI Display:**
- ✅ **Shown in UI** - Status bar shows "Embed: ResNet18 ({device})"

**Code Reference:**
```python
def _get_embedding_device(device: str = "auto") -> str:
    """Auto-detect best available device for embeddings."""
    if device != "auto":
        return device
    
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
```

**Implementation:**
- Model loaded and moved to device: `model = model.to(effective_device)`
- Tensors moved to device before inference: `t = tensor.unsqueeze(0).to(effective_device)`
- Results moved back to CPU for numpy conversion: `out.cpu().squeeze().numpy()`

---

## 3. Classification Training (XGBoost/CatBoost)

**Location:** `src/training_core.py`

**Device Selection:**
- **CPU only** - XGBoost and CatBoost are CPU-based libraries
- No GPU support in current implementation
- Uses multi-threading on CPU (line 257: `thread_count=-1` for CatBoost)

**Current Behavior:**
- Runs on CPU with multi-threading
- Thread count controlled by environment (`PYTEST_CURRENT_TEST` disables threading)

**UI Display:**
- ❌ **Not shown** in UI (training happens in background)

**Code Reference:**
```257:257:src/training_core.py
"thread_count": 1 if os.environ.get("PYTEST_CURRENT_TEST") else -1,
```

**Note:** XGBoost/CatBoost can use GPU with special builds, but current implementation doesn't support it.

---

## 4. Classification Prediction (XGBoost/CatBoost)

**Location:** `src/inference.py` `predict_keep_probability()`

**Device Selection:**
- **CPU only** - same as training
- Uses pre-trained XGBoost/CatBoost models

**Current Behavior:**
- Runs on CPU
- Fast inference (typically <10ms per image)

**UI Display:**
- ❌ **Not shown** in UI

---

## 5. Feature Extraction

**Location:** `src/features.py`

**Device Selection:**
- **CPU only** - uses PIL, numpy, opencv
- No GPU acceleration

**Current Behavior:**
- Image processing on CPU
- EXIF extraction, color histograms, quality metrics

**UI Display:**
- ❌ **Not shown** in UI

---

## Current UI Status

### ✅ Shown in UI:
- **Object Detection** - Status bar shows:
  - "Detector: {backend}" (e.g., "yolov8")
  - "Device: {device}" (e.g., "cpu", "cuda", "mps")
  - "Model: {model}" (e.g., "yolov8n")
- **Classification** - Status bar shows:
  - "Classify: {type}" (e.g., "CatBoost" or "XGBoost")
- **Embeddings** - Status bar shows:
  - "Embed: {model} ({device})" (e.g., "ResNet18 (CPU)")
- **Feature Extraction** - Status bar shows:
  - "Features: {backend}" (e.g., "PIL/OpenCV/NumPy")

---

## Recommendations

### ✅ 1. Add Device Selection for Image Embeddings - COMPLETED
- ✅ Implemented auto-detection similar to object detection
- ✅ ResNet18 model and tensors moved to detected device
- ✅ Device info shown in UI

### ✅ 2. Show All Computation Devices in UI - COMPLETED
- ✅ All computation devices shown in status bar:
  - Object Detection: {backend}, {device}, {model}
  - Classification: {type}
  - Image Embeddings: {model} ({device})
  - Feature Extraction: {backend}

### 3. Add Device Selection to Settings - TODO
- Allow users to override device selection via settings dialog
- Show available devices (CUDA, MPS, CPU)
- Warn about known issues (e.g., YOLOv8 on MPS)
- Add environment variable documentation

---

## Device Detection Helper

Current device detection pattern (from `object_detection.py`):

```python
def get_device(device: str = "auto") -> str:
    """Auto-detect best available device."""
    if device != "auto":
        return device
    
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    
    return "cpu"
```

This pattern should be reused for image embeddings.

