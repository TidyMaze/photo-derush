import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple

class FeatureVectorCache:
    def __init__(self, feature_vector_fn, max_workers=4):
        self.feature_vector_fn = feature_vector_fn
        self.cache: Dict[str, Tuple] = {}
        self.lock = threading.Lock()
        self.max_workers = max_workers

    def get(self, image_path):
        with self.lock:
            return self.cache.get(image_path)

    def set(self, image_path, value):
        with self.lock:
            self.cache[image_path] = value

    def batch_extract(self, image_paths):
        # Only extract for images not already cached
        to_extract = [p for p in image_paths if p not in self.cache]
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.feature_vector_fn, p): p for p in to_extract}
            for future in as_completed(futures):
                p = futures[future]
                try:
                    result = future.result()
                    self.set(p, result)
                except Exception as e:
                    self.set(p, None)
        # Return all cached results in order
        return [self.get(p) for p in image_paths]

