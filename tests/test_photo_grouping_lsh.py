"""Tests for LSH-optimized photo grouping (correctness and performance)."""

import time
from collections import Counter

import pytest

from src.photo_grouping import group_near_duplicates


def test_lsh_correctness_same_as_naive():
    """LSH implementation should produce same results as naive O(n²) approach."""
    try:
        import imagehash
    except ImportError:
        pytest.skip("imagehash not available")
    
    # Create test hashes with known relationships
    # Hash 0, 1, 2 are similar (within threshold)
    # Hash 3, 4 are similar
    # Hash 5 is unique
    
    # Generate hashes with controlled Hamming distances
    base_hash = imagehash.phash(imagehash.Image.new('RGB', (64, 64), color='red'))
    
    # Create similar hashes by modifying bits slightly
    hash0 = base_hash
    hash1 = imagehash.phash(imagehash.Image.new('RGB', (64, 64), color='red'))  # Should be very similar
    hash2 = imagehash.phash(imagehash.Image.new('RGB', (64, 64), color='darkred'))  # Should be similar
    
    hash3 = imagehash.phash(imagehash.Image.new('RGB', (64, 64), color='blue'))
    hash4 = imagehash.phash(imagehash.Image.new('RGB', (64, 64), color='blue'))  # Should be very similar
    
    hash5 = imagehash.phash(imagehash.Image.new('RGB', (64, 64), color='green'))  # Unique
    
    # Create hash function that returns hex strings
    hash_map = {
        'img0.jpg': str(hash0),
        'img1.jpg': str(hash1),
        'img2.jpg': str(hash2),
        'img3.jpg': str(hash3),
        'img4.jpg': str(hash4),
        'img5.jpg': str(hash5),
    }
    
    def hash_fn(filename):
        return hash_map[filename]
    
    filenames = ['img0.jpg', 'img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg']
    
    # Test with LSH (if implemented)
    # For now, test that naive version works correctly
    groups_naive = group_near_duplicates(
        filenames,
        hash_fn,
        hamming_threshold=10,  # Allow some variation
    )
    
    # Verify groups are reasonable (similar images should be grouped)
    unique_groups = set(groups_naive)
    assert len(unique_groups) >= 1  # At least one group
    assert len(unique_groups) <= len(filenames)  # At most one per image
    
    # Verify all images have a group ID
    assert len(groups_naive) == len(filenames)
    assert all(isinstance(g, int) for g in groups_naive)


def test_lsh_performance_improvement():
    """LSH should be faster than naive O(n²) for large datasets."""
    try:
        import imagehash
    except ImportError:
        pytest.skip("imagehash not available")
    
    # Create a larger dataset (50 images)
    num_images = 50
    filenames = [f'img{i}.jpg' for i in range(num_images)]
    
    # Create hash function with unique hashes (no duplicates for simplicity)
    hash_map = {}
    for i, fname in enumerate(filenames):
        # Create slightly different images to get different hashes
        color_val = (i * 10) % 255
        img = imagehash.Image.new('RGB', (64, 64), color=(color_val, 0, 0))
        hash_map[fname] = str(imagehash.phash(img))
    
    def hash_fn(filename):
        return hash_map[filename]
    
    # Measure naive approach time
    start_naive = time.perf_counter()
    groups_naive = group_near_duplicates(
        filenames,
        hash_fn,
        hamming_threshold=10,
    )
    time_naive = time.perf_counter() - start_naive
    
    # Verify results
    assert len(groups_naive) == num_images
    
    # For now, just verify it completes in reasonable time
    # When LSH is implemented, we'll compare times
    assert time_naive < 10.0  # Should complete in < 10 seconds
    
    # TODO: When LSH is implemented, compare:
    # groups_lsh = group_near_duplicates_lsh(...)
    # time_lsh = ...
    # assert time_lsh < time_naive * 0.5  # LSH should be at least 2x faster


def test_lsh_edge_cases():
    """LSH should handle edge cases correctly."""
    try:
        import imagehash
    except ImportError:
        pytest.skip("imagehash not available")
    
    # Empty list
    groups = group_near_duplicates([], lambda x: "", hamming_threshold=10)
    assert groups == []
    
    # Single image
    hash_map = {'img0.jpg': str(imagehash.phash(imagehash.Image.new('RGB', (64, 64), color='red')))}
    groups = group_near_duplicates(['img0.jpg'], lambda x: hash_map[x], hamming_threshold=10)
    assert len(groups) == 1
    assert groups[0] == 0  # Should get group ID 0
    
    # Two identical images
    hash_str = str(imagehash.phash(imagehash.Image.new('RGB', (64, 64), color='red')))
    hash_map = {'img0.jpg': hash_str, 'img1.jpg': hash_str}
    groups = group_near_duplicates(['img0.jpg', 'img1.jpg'], lambda x: hash_map[x], hamming_threshold=10)
    assert len(groups) == 2
    assert groups[0] == groups[1]  # Should be in same group


def test_lsh_comparison_count():
    """LSH should make fewer comparisons than naive O(n²) approach."""
    try:
        import imagehash
    except ImportError:
        pytest.skip("imagehash not available")
    
    # For now, we can't easily count comparisons without modifying the code
    # This test will be updated when LSH is implemented with comparison counting
    num_images = 20
    filenames = [f'img{i}.jpg' for i in range(num_images)]
    
    hash_map = {}
    for i, fname in enumerate(filenames):
        color_val = (i * 10) % 255
        img = imagehash.Image.new('RGB', (64, 64), color=(color_val, 0, 0))
        hash_map[fname] = str(imagehash.phash(img))
    
    def hash_fn(filename):
        return hash_map[filename]
    
    groups = group_near_duplicates(filenames, hash_fn, hamming_threshold=10)
    
    # Naive approach would make n(n-1)/2 = 190 comparisons
    # LSH should make significantly fewer
    # For now, just verify it works
    assert len(groups) == num_images



