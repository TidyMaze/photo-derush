import cv2
import numpy as np

# Technical quality features for an image (OpenCV)
def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = {}
    # Mean and std of grayscale
    features['gray_mean'] = float(np.mean(img_gray))
    features['gray_std'] = float(np.std(img_gray))
    # Blur (variance of Laplacian)
    features['blur'] = float(cv2.Laplacian(img_gray, cv2.CV_64F).var())
    # Edge density (Canny)
    edges = cv2.Canny(img_gray, 100, 200)
    features['edge_density'] = float(np.mean(edges > 0))
    # Color histogram (flattened, 8 bins per channel)
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    for i, v in enumerate(hist):
        features[f'colorhist_{i}'] = float(v)
    return features

def feature_vector(image_path):
    logging.info("[Predict] Extracting feature vector for image=%s", image_path)
    features = extract_features(image_path)
    if features is None:
        return None
    # Return as numpy array (sorted by key for consistency)
    keys = sorted(features.keys())
    return np.array([features[k] for k in keys], dtype=np.float32), keys

