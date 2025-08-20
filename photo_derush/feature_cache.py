import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple

class FeatureVectorCache:
    def __init__(self, feature_vector_fn, max_workers=4, cache_path="feature_cache.pkl"):
        self.feature_vector_fn = feature_vector_fn
        self.cache: Dict[str, Tuple] = {}
        self.lock = threading.Lock()
        self.max_workers = max_workers
        self.cache_path = cache_path
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    self.cache = pickle.load(f)
            except Exception as e:
                # Log error and start with empty cache
                import logging
                logging.warning(f"Failed to load feature cache: {e}")
                self.cache = {}

    def _save_cache(self):
        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            import logging
            logging.error(f"Failed to save feature cache: {e}")

    def get(self, image_path):
        with self.lock:
            return self.cache.get(image_path)

    def set(self, image_path, value):
        with self.lock:
            self.cache[image_path] = value
            self._save_cache()

    def batch_extract(self, image_paths, progress_callback=None):
        # Only extract for images not already cached
        to_extract = [p for p in image_paths if p not in self.cache]
        total = len(to_extract)
        completed = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.feature_vector_fn, p): p for p in to_extract}
            for future in as_completed(futures):
                p = futures[future]
                try:
                    result = future.result()
                    self.set(p, result)
                except Exception as e:
                    self.set(p, None)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)
        # Return all cached results in order
        return [self.get(p) for p in image_paths]
