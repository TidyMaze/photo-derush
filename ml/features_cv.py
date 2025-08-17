import cv2
import numpy as np
import time
import math
from pathlib import Path
import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Ordered list of feature names (stable contract)
FEATURE_NAMES = [
    # --- Existing base features (keep order for backward compatibility/migration) ---
    'laplacian_var', 'sobel_mean_abs', 'edge_density', 'fft_highfreq_ratio',
    'gray_mean', 'gray_std', 'gray_p01', 'gray_p50', 'gray_p99', 'clip_low_pct', 'clip_high_pct',
    'rms_contrast', 'noise_hp_std', 'saturation_mean', 'colorfulness', 'gray_entropy',
    # --- Added Sharpness (multi-scale & robust) ---
    'tenengrad_var', 'brenner_gradient', 'laplacian_tile_mean', 'laplacian_tile_std', 'grad_hist_p90',
    # --- Exposure / Histogram shape ---
    'gray_skewness', 'gray_kurtosis', 'dynamic_range', 'midtone_offset', 'shadow_area', 'highlight_area',
    # --- Local contrast & texture ---
    'local_rms_mean', 'local_rms_std', 'lbp_uniformity',
    # --- Noise (extended) ---
    'noise_hp_std_ms_s1', 'noise_hp_std_ms_s2', 'noise_hp_std_ms_ratio', 'chroma_noise_std', 'chroma_to_luma_noise_ratio',
    # --- Color balance & saturation shape ---
    'grayworld_deviation', 'rg_ratio', 'bg_ratio', 'saturation_p90', 'saturation_std',
    # --- Local information / entropy ---
    'local_entropy_mean', 'local_entropy_std',
    # --- Composition lite ---
    'energy_centroid_x', 'energy_centroid_y', 'thirds_alignment_score', 'edge_orientation_entropy',
    # --- Geometry / misc ---
    'aspect_ratio', 'orientation_flag', 'phash'
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

    # Assign base (existing) features
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

    # --- New feature groups ---
    sharp = _sharpness_features(gray_f)
    expos = _exposure_hist_features(gray_f, p01, p50, p99)
    lct = _local_contrast_texture(gray_f)
    noise_ext = _noise_features(gray_f, img, noise_hp_std)
    color_extra = _color_balance_features(img, hsv)
    local_ent = _tile_entropy(gray)
    compo = _composition_features(gray_f)
    geom = _geometry_misc(img, gray)
    feats.update(sharp)
    feats.update(expos)
    feats.update(lct)
    feats.update(noise_ext)
    feats.update(color_extra)
    feats.update(local_ent)
    feats.update(compo)
    feats.update(geom)

    # Normalize heavy-tailed metrics with log1p heuristic
    for k in list(feats.keys()):
        feats[k] = _log1p_if_heavy(k, feats[k])

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


def _log1p_if_heavy(name: str, value: float) -> float:
    """Apply log1p to heavy-tailed metrics. Heuristic by name pattern."""
    if value <= 0:
        return float(value)
    patterns = (
        '_var', '_std', 'brenner_gradient', 'laplacian_tile_', 'tenengrad', 'grad_hist_p90', 'dynamic_range',
        'noise_hp_std_ms', 'chroma_noise_std', 'grayworld_deviation', 'saturation_p90', 'local_rms_', 'local_entropy_',
        'edge_orientation_entropy'
    )
    for p in patterns:
        if p in name:
            return float(math.log1p(value))
    return float(value)


def _sharpness_features(gray_f: np.ndarray) -> Dict[str, float]:
    h, w = gray_f.shape
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    tenengrad_var = float(np.var(mag)) if mag.size else 0.0
    # Brenner gradient (stride=2) – mean of squared diffs both axes where valid
    bg_x = (gray_f[:, 2:] - gray_f[:, :-2]) ** 2
    bg_y = (gray_f[2:, :] - gray_f[:-2, :]) ** 2
    brenner = float(np.mean(bg_x)) if bg_x.size else 0.0
    if bg_y.size:
        brenner = 0.5 * (brenner + float(np.mean(bg_y)))
    # Laplacian tiles
    try:
        lap_full = cv2.Laplacian(gray_f, cv2.CV_32F)
    except Exception:
        lap_full = np.zeros_like(gray_f)
    tile_size = max(16, min(h, w) // 8)  # adaptive small tiles
    if tile_size <= 0:
        tile_size = min(h, w)
    laps_abs_means = []
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = lap_full[y:y+tile_size, x:x+tile_size]
            if tile.size:
                laps_abs_means.append(float(np.mean(np.abs(tile))))
    if laps_abs_means:
        laps_abs_means_arr = np.array(laps_abs_means, dtype=np.float32)
        lap_tile_mean = float(np.mean(laps_abs_means_arr))
        lap_tile_std = float(np.std(laps_abs_means_arr))
    else:
        lap_tile_mean = 0.0
        lap_tile_std = 0.0
    grad_hist_p90 = float(np.percentile(mag, 90)) if mag.size else 0.0
    return {
        'tenengrad_var': _safe_stat(tenengrad_var),
        'brenner_gradient': _safe_stat(brenner),
        'laplacian_tile_mean': _safe_stat(lap_tile_mean),
        'laplacian_tile_std': _safe_stat(lap_tile_std),
        'grad_hist_p90': _safe_stat(grad_hist_p90),
    }


def _exposure_hist_features(gray_f: np.ndarray, p01: float, p50: float, p99: float) -> Dict[str, float]:
    # Skewness & Kurtosis (Fisher) with protection
    g = gray_f.ravel().astype(np.float32)
    if g.size == 0:
        return {k: 0.0 for k in ['gray_skewness','gray_kurtosis','dynamic_range','midtone_offset','shadow_area','highlight_area']}
    mean = g.mean()
    std = g.std()
    if std < 1e-6:
        skew = 0.0
        kurt = 0.0
    else:
        z = (g - mean) / std
        skew = float(np.mean(z**3))
        kurt = float(np.mean(z**4) - 3.0)
    dynamic_range = float(p99 - p01)
    midtone_offset = float(p50/255.0 - 0.18)
    shadow_area = float(np.mean(g < 10.0))
    highlight_area = float(np.mean(g > 245.0))
    return {
        'gray_skewness': _safe_stat(skew),
        'gray_kurtosis': _safe_stat(kurt),
        'dynamic_range': _safe_stat(dynamic_range),
        'midtone_offset': _safe_stat(midtone_offset),
        'shadow_area': _safe_stat(shadow_area),
        'highlight_area': _safe_stat(highlight_area),
    }


def _local_contrast_texture(gray_f: np.ndarray) -> Dict[str, float]:
    h, w = gray_f.shape
    g_norm = gray_f / 255.0
    tile = max(16, min(h, w) // 8)
    if tile <= 0:
        tile = min(h, w)
    rms_vals = []
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            t = g_norm[y:y+tile, x:x+tile]
            if t.size:
                rms_vals.append(float(np.sqrt(np.mean((t - t.mean())**2))))
    if rms_vals:
        arr = np.array(rms_vals, dtype=np.float32)
        local_rms_mean = float(arr.mean())
        local_rms_std = float(arr.std())
    else:
        local_rms_mean = 0.0
        local_rms_std = 0.0
    # Simple LBP uniformity
    lbp_uniformity = 0.0
    if h >= 3 and w >= 3:
        center = gray_f[1:-1,1:-1]
        neighbors = [
            gray_f[0:-2,0:-2], gray_f[0:-2,1:-1], gray_f[0:-2,2:],
            gray_f[1:-1,2:], gray_f[2:,2:], gray_f[2:,1:-1],
            gray_f[2:,0:-2], gray_f[1:-1,0:-2]
        ]
        codes = np.zeros_like(center, dtype=np.uint8)
        for i, nb in enumerate(neighbors):
            codes |= ((nb >= center) << i).astype(np.uint8)
        # count bit transitions in circular binary string
        # precompute transitions for 256 possible codes
        trans_cache = np.zeros(256, dtype=np.uint8)
        for v in range(256):
            b = ((v >> np.arange(8)) & 1).astype(np.int8)
            trans = np.sum(b != np.roll(b, -1))
            trans_cache[v] = trans
        trans_vals = trans_cache[codes]
        uniform = np.mean(trans_vals <= 2)
        lbp_uniformity = float(uniform)
    return {
        'local_rms_mean': _safe_stat(local_rms_mean),
        'local_rms_std': _safe_stat(local_rms_std),
        'lbp_uniformity': _safe_stat(lbp_uniformity),
    }


def _noise_features(gray_f: np.ndarray, img: np.ndarray, base_noise_std: float) -> Dict[str, float]:
    g32 = gray_f.astype(np.float32)
    blur1 = cv2.GaussianBlur(g32, (0,0), 1)
    blur2 = cv2.GaussianBlur(g32, (0,0), 2)
    hp1 = g32 - blur1
    hp2 = g32 - blur2
    std1 = float(np.std(hp1))
    std2 = float(np.std(hp2))
    ratio = float(std2/std1) if std1 > 1e-9 else 0.0
    # Chroma noise (LAB high-pass std of a/b)
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        a = lab[:,:,1].astype(np.float32)
        b = lab[:,:,2].astype(np.float32)
        a_hp = a - cv2.GaussianBlur(a,(0,0),1)
        b_hp = b - cv2.GaussianBlur(b,(0,0),1)
        chroma_noise = float(0.5*(np.std(a_hp)+np.std(b_hp)))
    except Exception:
        chroma_noise = 0.0
    chroma_to_luma = float(chroma_noise / base_noise_std) if base_noise_std > 1e-9 else 0.0
    return {
        'noise_hp_std_ms_s1': _safe_stat(std1),
        'noise_hp_std_ms_s2': _safe_stat(std2),
        'noise_hp_std_ms_ratio': _safe_stat(ratio),
        'chroma_noise_std': _safe_stat(chroma_noise),
        'chroma_to_luma_noise_ratio': _safe_stat(chroma_to_luma),
    }


def _color_balance_features(img: np.ndarray, hsv: np.ndarray) -> Dict[str, float]:
    b, g, r = cv2.split(img.astype(np.float32))
    mean_r, mean_g, mean_b = float(r.mean()), float(g.mean()), float(b.mean())
    mean_vec = np.array([mean_r, mean_g, mean_b], dtype=np.float32)
    m = float(mean_vec.mean())
    grayworld_deviation = float(np.sqrt(np.sum((mean_vec - m)**2)))
    rg_ratio = float(mean_r/mean_g) if mean_g > 1e-6 else 0.0
    bg_ratio = float(mean_b/mean_g) if mean_g > 1e-6 else 0.0
    sat = hsv[:,:,1].astype(np.float32)/255.0
    saturation_p90 = float(np.percentile(sat, 90)) if sat.size else 0.0
    saturation_std = float(np.std(sat)) if sat.size else 0.0
    return {
        'grayworld_deviation': _safe_stat(grayworld_deviation),
        'rg_ratio': _safe_stat(rg_ratio),
        'bg_ratio': _safe_stat(bg_ratio),
        'saturation_p90': _safe_stat(saturation_p90),
        'saturation_std': _safe_stat(saturation_std),
    }


def _tile_entropy(gray: np.ndarray) -> Dict[str, float]:
    h, w = gray.shape
    tile = max(16, min(h, w)//8)
    if tile <= 0:
        tile = min(h, w)
    ent = []
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            t = gray[y:y+tile, x:x+tile]
            if t.size:
                ent.append(_compute_entropy(t))
    if ent:
        arr = np.array(ent, dtype=np.float32)
        return {
            'local_entropy_mean': _safe_stat(float(arr.mean())),
            'local_entropy_std': _safe_stat(float(arr.std())),
        }
    return {'local_entropy_mean': 0.0, 'local_entropy_std': 0.0}


def _composition_features(gray_f: np.ndarray) -> Dict[str, float]:
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    h, w = gray_f.shape
    if mag.sum() <= 1e-6:
        ecx = ecy = 0.5
    else:
        ys = np.arange(h, dtype=np.float32)[:, None]
        xs = np.arange(w, dtype=np.float32)[None, :]
        ecx = float((mag * xs).sum() / (mag.sum() * (w-1))) if w > 1 else 0.5
        ecy = float((mag * ys).sum() / (mag.sum() * (h-1))) if h > 1 else 0.5
    thirds_pts = [(1/3,1/3),(1/3,2/3),(2/3,1/3),(2/3,2/3)]
    dists = [math.hypot(ecx-px, ecy-py) for px,py in thirds_pts]
    min_d = min(dists) if dists else 0.0
    max_possible = math.hypot(0.5,0.5)
    thirds_alignment_score = 1.0 - float(min_d / max_possible) if max_possible>0 else 0.0
    # Orientation entropy
    if mag.size == 0:
        edge_entropy = 0.0
    else:
        ang = cv2.phase(gx, gy, angleInDegrees=False)  # 0..2pi
        bins = 8
        hist, _ = np.histogram(ang, bins=bins, range=(0, 2*math.pi), weights=mag)
        total = hist.sum()
        if total <= 0:
            edge_entropy = 0.0
        else:
            p = hist/total
            nz = p[p>0]
            edge_entropy = float(-(nz*np.log2(nz)).sum())
    return {
        'energy_centroid_x': _safe_stat(ecx),
        'energy_centroid_y': _safe_stat(ecy),
        'thirds_alignment_score': _safe_stat(thirds_alignment_score),
        'edge_orientation_entropy': _safe_stat(edge_entropy),
    }


def _geometry_misc(img: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
    h, w = gray.shape
    aspect_ratio = float(w / h) if h > 0 else 0.0
    if h > w:
        orientation_flag = -1.0
    elif w > h:
        orientation_flag = 1.0
    else:
        orientation_flag = 0.0
    phash_val = _compute_phash(gray)
    return {
        'aspect_ratio': _safe_stat(aspect_ratio),
        'orientation_flag': orientation_flag,
        'phash': _safe_stat(phash_val),
    }


def _compute_phash(gray: np.ndarray) -> float:
    try:
        # Resize to 32x32, DCT, take top-left 8x8 excluding DC, threshold by median
        g = cv2.resize(gray, (32,32), interpolation=cv2.INTER_AREA).astype(np.float32)
        dct = cv2.dct(g)
        block = dct[:8,:8]
        median = np.median(block[1:].ravel())  # exclude DC element
        bits = (block >= median).astype(np.uint8)
        # pack into 64-bit integer
        flat = bits.ravel()
        val = 0
        for b in flat:
            val = (val << 1) | int(b)
        # scale to [0,1] by dividing by 2^64-1 to keep magnitude manageable
        val_norm = val / ((1<<64)-1)
        return float(val_norm)
    except Exception:
        return 0.0


def compute_feature_vector(path: str | Path):
    """Helper returning (np.array(vector), FEATURE_NAMES)."""
    d = compute_quality_features_from_path(Path(path))
    vec = np.array([d[k] for k in FEATURE_NAMES], dtype=np.float32)
    return vec, FEATURE_NAMES
