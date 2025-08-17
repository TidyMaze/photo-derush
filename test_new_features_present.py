import os, numpy as np
from ml.features_cv import compute_quality_features_from_path, FEATURE_NAMES
from pathlib import Path
from PIL import Image

def test_new_features_present(tmp_path):
    # Create synthetic gradient image to exercise feature code paths
    w,h = 64,48
    arr = np.zeros((h,w,3), dtype=np.uint8)
    for y in range(h):
        arr[y,:,0] = np.linspace(0,255,w,dtype=np.uint8)
        arr[y,:,1] = y*255//h
        arr[y,:,2] = 255 - arr[y,:,0]
    img_path = tmp_path/"grad.png"
    Image.fromarray(arr).save(img_path)
    feats = compute_quality_features_from_path(Path(img_path))
    # All feature names returned and finite
    missing = [k for k in FEATURE_NAMES if k not in feats]
    assert not missing, f"Missing features: {missing}"
    bad = {k:v for k,v in feats.items() if not np.isfinite(v)}
    assert not bad, f"Non-finite feature values: {bad}"
    # Spot check a few newly added keys exist
    for k in [
        'tenengrad_var','brenner_gradient','laplacian_tile_mean','grad_hist_p90','gray_skewness',
        'local_rms_mean','noise_hp_std_ms_s1','chroma_noise_std','grayworld_deviation','saturation_p90',
        'local_entropy_mean','thirds_alignment_score','aspect_ratio','phash']:
        assert k in feats, f"Expected new feature {k} missing"

