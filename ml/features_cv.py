import cv2
import numpy as np
import time
import math
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Ordered list of feature names (stable contract)
FEATURE_NAMES = [
    # Sharpness / Focus
    'laplacian_var',
    'sobel_mean_abs',
    'edge_density',
    'fft_highfreq_ratio',
    # Exposure & Contrast
    'gray_mean',
    'gray_std',
    'gray_p01',
    'gray_p50',
    'gray_p99',
    'clip_low_pct',
    'clip_high_pct',
    'rms_contrast',
    # Noise proxy
    'noise_hp_std',
    # Color / Saturation
    'saturation_mean',
    'colorfulness',
    # Entropy
    'gray_entropy',
]

# Simple latency tracker (nice-to-have)
_latency = []


def _safe_stat(x):
    if not math.isfinite(x):
        return 0.0
    return float(x)


def _ensure_gray(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _compute_entropy(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = hist.sum()
    if total <= 0:
        return 0.0
    p = hist / total
    # Avoid log(0)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return -float(np.sum(p * np.log2(p)))


def _compute_fft_highfreq_ratio(gray):
    # Use centered FFT and a square low-freq mask (10% of min dim)
    g = gray.astype(np.float32)
    f = np.fft.fft2(g)
    fshift = np.fft.fftshift(f)
    mag2 = np.abs(fshift) ** 2
    h, w = gray.shape
    r = int(0.1 * min(h, w))
    cy, cx = h // 2, w // 2
    y0, y1 = max(0, cy - r), min(h, cy + r)
    x0, x1 = max(0, cx - r), min(w, cx + r)
    low = mag2[y0:y1, x0:x1].sum()
    total = mag2.sum() + 1e-12
    high = total - low
    return float(high / total)


def _compute_colorfulness(img):
    # Hasler & Süsstrunk (simplified)
    b, g, r = cv2.split(img.astype(np.float32))
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)
    return float(np.sqrt(std_rg ** 2 + std_yb ** 2) + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2))


def compute_quality_features_from_path(path: Path, resize_max: int = 1600):
    """Compute deterministic, fast technical quality features using only OpenCV + NumPy.

    Returns dict containing every key in FEATURE_NAMES.
    Any non-finite numeric result is replaced with 0.0.
    """
    start = time.perf_counter()
    p = Path(path)
    img = cv2.imread(str(p))
    feats = {k: 0.0 for k in FEATURE_NAMES}
    if img is None:
        return feats
    h, w = img.shape[:2]
    max_side = max(h, w)
    if max_side > resize_max:
        scale = resize_max / float(max_side)
        new_size = (int(w * scale), int(h * scale))
        if new_size[0] <= 0 or new_size[1] <= 0:
            new_size = (w, h)
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    gray = _ensure_gray(img)
    if gray is None:
        return feats

    gray_f = gray.astype(np.float32)

    # Sharpness / Focus
    # Laplacian variance (focus) – use CV_32F to be compatible with minimal builds
    try:
        lap_var = cv2.Laplacian(gray, cv2.CV_32F).var()
    except Exception:
        try:
            lap_var = cv2.Laplacian(gray, cv2.CV_16S).var()
        except Exception:
            lap_var = 0.0
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mean_abs = float(np.mean(np.abs(gx) + np.abs(gy)))
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))
    fft_ratio = _compute_fft_highfreq_ratio(gray)

    # Exposure & Contrast
    gray_mean = float(np.mean(gray_f))
    gray_std = float(np.std(gray_f))
    p01, p50, p99 = np.percentile(gray_f, [1, 50, 99])
    clip_low_pct = float(np.mean(gray_f <= 2))
    clip_high_pct = float(np.mean(gray_f >= 253))
    rms_contrast = float(np.std(gray_f / 255.0))

    # Noise proxy
    blur1 = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=1)
    hp = gray_f - blur1
    noise_hp_std = float(np.std(hp))

    # Color / Saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat_mean = float(np.mean(hsv[:, :, 1]) / 255.0)
    colorfulness = _compute_colorfulness(img)

    # Entropy
    entropy = _compute_entropy(gray)

    # Assign
    feats.update({
        'laplacian_var': _safe_stat(lap_var),
        'sobel_mean_abs': _safe_stat(sobel_mean_abs),
        'edge_density': _safe_stat(edge_density),
        'fft_highfreq_ratio': _safe_stat(fft_ratio),
        'gray_mean': _safe_stat(gray_mean),
        'gray_std': _safe_stat(gray_std),
        'gray_p01': _safe_stat(p01),
        'gray_p50': _safe_stat(p50),
        'gray_p99': _safe_stat(p99),
        'clip_low_pct': _safe_stat(clip_low_pct),
        'clip_high_pct': _safe_stat(clip_high_pct),
        'rms_contrast': _safe_stat(rms_contrast),
        'noise_hp_std': _safe_stat(noise_hp_std),
        'saturation_mean': _safe_stat(sat_mean),
        'colorfulness': _safe_stat(colorfulness),
        'gray_entropy': _safe_stat(entropy),
    })

    # Sanitize (avoid lingering NaN/inf)
    for k, v in list(feats.items()):
        if not math.isfinite(v):
            feats[k] = 0.0
    dur = (time.perf_counter() - start) * 1000.0
    _latency.append(dur)
    if len(_latency) % 50 == 0:
        arr = np.array(_latency)
        logger.info('[FeaturesCV] latency_ms mean=%.2f p95=%.2f (n=%d)', arr.mean(), np.percentile(arr, 95), len(arr))
    return feats


def compute_feature_vector(path: str | Path):
    """Helper returning (np.array(vector), FEATURE_NAMES)."""
    d = compute_quality_features_from_path(Path(path))
    vec = np.array([d[k] for k in FEATURE_NAMES], dtype=np.float32)
    return vec, FEATURE_NAMES
