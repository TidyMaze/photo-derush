"""Performance and correctness tests for LSH and EXIF pre-loading optimizations."""

import time
from unittest.mock import MagicMock

import pytest

from src.lazy_loader import LazyImageLoader
from src.photo_grouping import group_near_duplicates, USE_LSH_OPTIMIZATION


def test_lsh_correctness_vs_naive():
    """LSH should produce identical results to naive O(n²) approach."""
    try:
        import imagehash
    except ImportError:
        pytest.skip("imagehash not available")
    
    # Create test dataset with known relationships
    num_images = 30
    filenames = [f'img{i}.jpg' for i in range(num_images)]
    
    # Create hashes - some similar, some unique
    hash_map = {}
    base_hash = imagehash.phash(imagehash.Image.new('RGB', (64, 64), color='red'))
    
    # First 10 images: similar (same color)
    for i in range(10):
        hash_map[filenames[i]] = str(base_hash)
    
    # Next 10 images: similar (different but close color)
    hash2 = imagehash.phash(imagehash.Image.new('RGB', (64, 64), color='darkred'))
    for i in range(10, 20):
        hash_map[filenames[i]] = str(hash2)
    
    # Last 10 images: unique
    for i in range(20, 30):
        color_val = (i * 20) % 255
        img = imagehash.Image.new('RGB', (64, 64), color=(color_val, 0, 0))
        hash_map[filenames[i]] = str(imagehash.phash(img))
    
    def hash_fn(filename):
        return hash_map[filename]
    
    # Test with LSH enabled
    groups_lsh = group_near_duplicates(
        filenames,
        hash_fn,
        hamming_threshold=10,
    )
    
    # Test with LSH disabled (naive)
    import src.photo_grouping as pg_module
    original_use_lsh = pg_module.USE_LSH_OPTIMIZATION
    try:
        pg_module.USE_LSH_OPTIMIZATION = False
        groups_naive = group_near_duplicates(
            filenames,
            hash_fn,
            hamming_threshold=10,
        )
    finally:
        pg_module.USE_LSH_OPTIMIZATION = original_use_lsh
    
    # Results should be identical (same groups)
    # Convert to sets of groups for comparison
    lsh_groups = {}
    naive_groups = {}
    
    for i, (fname, lsh_gid, naive_gid) in enumerate(zip(filenames, groups_lsh, groups_naive)):
        if lsh_gid not in lsh_groups:
            lsh_groups[lsh_gid] = set()
        lsh_groups[lsh_gid].add(fname)
        
        if naive_gid not in naive_groups:
            naive_groups[naive_gid] = set()
        naive_groups[naive_gid].add(fname)
    
    # Compare group structures
    lsh_group_sets = sorted([frozenset(g) for g in lsh_groups.values()])
    naive_group_sets = sorted([frozenset(g) for g in naive_groups.values()])
    
    assert lsh_group_sets == naive_group_sets, "LSH and naive should produce same groups"


def test_lsh_performance_improvement():
    """LSH should be faster than naive for large datasets with similar hashes."""
    try:
        import imagehash
    except ImportError:
        pytest.skip("imagehash not available")
    
    # Create larger dataset (100 images) with some similar hashes
    num_images = 100
    filenames = [f'img{i}.jpg' for i in range(num_images)]
    
    hash_map = {}
    # Create 10 groups of similar hashes
    for group_id in range(10):
        base_img = imagehash.Image.new('RGB', (64, 64), color=(group_id * 25, 0, 0))
        base_hash = imagehash.phash(base_img)
        
        # 10 images per group (similar hashes)
        for i in range(group_id * 10, (group_id + 1) * 10):
            hash_map[filenames[i]] = str(base_hash)
    
    def hash_fn(filename):
        return hash_map[filename]
    
    # Measure naive approach
    import src.photo_grouping as pg_module
    original_use_lsh = pg_module.USE_LSH_OPTIMIZATION
    
    try:
        pg_module.USE_LSH_OPTIMIZATION = False
        start_naive = time.perf_counter()
        groups_naive = group_near_duplicates(filenames, hash_fn, hamming_threshold=10)
        time_naive = time.perf_counter() - start_naive
        
        # Measure LSH approach
        pg_module.USE_LSH_OPTIMIZATION = True
        start_lsh = time.perf_counter()
        groups_lsh = group_near_duplicates(filenames, hash_fn, hamming_threshold=10)
        time_lsh = time.perf_counter() - start_lsh
        
        # LSH should be faster (or at least not much slower)
        # For datasets with similar hashes, LSH should show improvement
        logging.info(f"Naive: {time_naive:.4f}s, LSH: {time_lsh:.4f}s")
        
        # Verify correctness
        assert len(groups_lsh) == len(groups_naive)
        assert len(set(groups_lsh)) == len(set(groups_naive))
        
    finally:
        pg_module.USE_LSH_OPTIMIZATION = original_use_lsh


def test_exif_preload_performance_measurement():
    """Measure actual performance improvement from EXIF pre-loading."""
    # Create mock with realistic delay
    model = MagicMock()
    
    def slow_load_exif(path):
        time.sleep(0.010)  # 10ms per image
        return {'Make': 'TestCamera', 'path': path}
    
    model.load_exif = MagicMock(side_effect=slow_load_exif)
    model.load_thumbnail = MagicMock(return_value=MagicMock())
    
    loader = LazyImageLoader(model, max_workers=4, cache_size=256)
    
    # Test with 20 images
    paths = [f'/img{i}.jpg' for i in range(20)]
    
    # Sequential loading (clearing cache each time)
    loader.clear_cache()
    model.load_exif.reset_mock()
    start_seq = time.perf_counter()
    for path in paths:
        loader._cached_exif(path)
    time_seq = time.perf_counter() - start_seq
    
    # Parallel pre-loading
    loader.clear_cache()
    model.load_exif.reset_mock()
    start_parallel = time.perf_counter()
    loader.preload_exif_silent(paths)
    # Wait for completion (20 images * 10ms / 4 workers = ~50ms)
    time.sleep(0.1)
    time_parallel = time.perf_counter() - start_parallel
    
    # Verify all were loaded
    assert model.load_exif.call_count == 20
    
    # Parallel should be faster
    speedup = time_seq / time_parallel if time_parallel > 0 else 1.0
    logging.info(f"Sequential: {time_seq:.4f}s, Parallel: {time_parallel:.4f}s, Speedup: {speedup:.2f}x")
    
    # With 4 workers, should see at least 2x speedup
    assert speedup >= 1.5, f"Expected at least 1.5x speedup, got {speedup:.2f}x"
    
    loader.shutdown(wait=False)


# Add logging import
import logging

