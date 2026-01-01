# Optimization: imagehash.__sub__ - Locality-Sensitive Hashing (LSH)

## Current Problem

**Hotspot**: `imagehash.__sub__` - 3.932s (85,191 calls)
- **Location**: `src/photo_grouping.py:217` - `group_near_duplicates()`
- **Complexity**: O(n²) - comparing every hash against every other hash
- **Current**: For 591 images, performs ~174,000 comparisons

## Current Implementation

```python
# src/photo_grouping.py:197-218
for i, fname1 in enumerate(filenames):
    h1 = filename_to_hash_obj.get(fname1)
    if h1 is None:
        continue
    
    for fname2 in filenames[i + 1:]:
        h2 = filename_to_hash_obj.get(fname2)
        if h2 is None:
            continue
        
        if h1 - h2 <= hamming_threshold:  # O(n²) comparison
            G.add_edge(fname1, fname2)
```

**Problem**: For n images, this requires n(n-1)/2 comparisons. For 591 images = 174,405 comparisons.

## Solution: Locality-Sensitive Hashing (LSH)

### Concept

LSH groups similar items into "buckets" using hash functions that map similar items to the same bucket with high probability. Instead of comparing all pairs, we only compare items within the same bucket.

**Key Insight**: If two hashes have Hamming distance ≤ threshold, they likely share some common bits. We can use these bits as a "signature" to group similar hashes.

### Algorithm: Multi-Index Hashing

For perceptual hashes (pHash), we can use **Multi-Index Hashing**:

1. **Split hash into segments**: Divide the hash into k segments
2. **Create index tables**: For each segment position, create an index mapping segment value → list of hashes
3. **Query**: For each hash, check all segments - if any segment matches, compare full hash
4. **Early termination**: Stop after finding enough matches

**Complexity**: O(n × k × m) where:
- n = number of images
- k = number of segments (typically 4-8)
- m = average bucket size (much smaller than n)

For typical datasets: **O(n log n)** instead of **O(n²)**

## Implementation Plan

### Option 1: Simple Multi-Index Hashing (Recommended)

**Effort**: Medium (2-3 hours)
**Impact**: 70-90% reduction in comparisons (3.932s → 0.4-1.2s)

```python
def group_near_duplicates_lsh(
    filenames: list[str],
    hash_fn: Callable[[str], str],
    hamming_threshold: int = 8,
    progress_reporter=None,
) -> list[int]:
    """
    Group near-duplicates using Locality-Sensitive Hashing.
    
    Uses multi-index hashing to reduce O(n²) comparisons to O(n log n).
    """
    import imagehash
    import networkx as nx
    
    # Step 1: Compute all hashes (same as before)
    filename_to_hash: dict[str, str] = {}
    for fname in filenames:
        filename_to_hash[fname] = hash_fn(fname)
    
    # Convert to hash objects
    filename_to_hash_obj: dict[str, imagehash.ImageHash] = {}
    for fname, h_str in filename_to_hash.items():
        if not h_str.startswith("error_"):
            try:
                filename_to_hash_obj[fname] = imagehash.hex_to_hash(h_str)
            except Exception:
                pass
    
    # Step 2: Build LSH index
    # Split hash into segments (e.g., 4 segments of 16 bits each for 64-bit hash)
    hash_bits = 64  # pHash is 64 bits
    num_segments = 4
    segment_size = hash_bits // num_segments  # 16 bits per segment
    
    # Create index: segment_index -> segment_value -> list of (filename, hash_obj)
    lsh_index: dict[int, dict[int, list[tuple[str, imagehash.ImageHash]]]] = {}
    for idx in range(num_segments):
        lsh_index[idx] = {}
    
    # Index all hashes by each segment
    for fname, hash_obj in filename_to_hash_obj.items():
        hash_int = int(str(hash_obj), 16)  # Convert to integer
        
        for seg_idx in range(num_segments):
            # Extract segment (16 bits)
            shift = seg_idx * segment_size
            mask = (1 << segment_size) - 1
            segment_value = (hash_int >> shift) & mask
            
            # Add to index
            if segment_value not in lsh_index[seg_idx]:
                lsh_index[seg_idx][segment_value] = []
            lsh_index[seg_idx][segment_value].append((fname, hash_obj))
    
    # Step 3: Build graph using LSH (only compare within same buckets)
    G = nx.Graph()
    G.add_nodes_from(filenames)
    
    compared_pairs = set()  # Track comparisons to avoid duplicates
    comparisons_made = 0
    
    for fname1, hash1 in filename_to_hash_obj.items():
        # Find candidate matches using LSH
        candidates = set()
        
        hash1_int = int(str(hash1), 16)
        
        # Check each segment - if any segment matches, add to candidates
        for seg_idx in range(num_segments):
            shift = seg_idx * segment_size
            mask = (1 << segment_size) - 1
            segment_value = (hash1_int >> shift) & mask
            
            # Get all hashes in this bucket
            bucket = lsh_index[seg_idx].get(segment_value, [])
            for fname2, hash2 in bucket:
                if fname2 != fname1:
                    candidates.add((fname2, hash2))
        
        # Compare only with candidates (much smaller set)
        for fname2, hash2 in candidates:
            # Avoid duplicate comparisons
            pair = tuple(sorted([fname1, fname2]))
            if pair in compared_pairs:
                continue
            compared_pairs.add(pair)
            
            comparisons_made += 1
            if hash1 - hash2 <= hamming_threshold:
                G.add_edge(fname1, fname2)
    
    logging.info(f"[photo_grouping] LSH: {comparisons_made} comparisons (vs {len(filenames)*(len(filenames)-1)//2} naive)")
    
    # Step 4: Find connected components (same as before)
    groups = {}
    group_id = 0
    for component in nx.connected_components(G):
        for fname in component:
            groups[fname] = group_id
        group_id += 1
    
    # Assign group IDs
    result = []
    for fname in filenames:
        result.append(groups.get(fname, group_id))
        if fname not in groups:
            group_id += 1
    
    return result
```

### Option 2: Use Annoy Library (Alternative)

**Effort**: Low (1 hour)
**Impact**: 80-95% reduction, but requires external dependency

```python
from annoy import AnnoyIndex

def group_near_duplicates_annoy(
    filenames: list[str],
    hash_fn: Callable[[str], str],
    hamming_threshold: int = 8,
) -> list[int]:
    """
    Use Annoy (Approximate Nearest Neighbors) for fast similarity search.
    """
    import imagehash
    
    # Build Annoy index
    hash_dim = 64  # pHash dimension
    index = AnnoyIndex(hash_dim, 'hamming')  # Hamming distance metric
    
    filename_to_hash_obj = {}
    for i, fname in enumerate(filenames):
        h_str = hash_fn(fname)
        if not h_str.startswith("error_"):
            hash_obj = imagehash.hex_to_hash(h_str)
            # Convert hash to vector
            hash_vector = [int(bit) for bit in str(hash_obj)]
            index.add_item(i, hash_vector)
            filename_to_hash_obj[i] = (fname, hash_obj)
    
    index.build(10)  # Build index with 10 trees
    
    # Query for similar hashes
    G = nx.Graph()
    for i, (fname, hash_obj) in filename_to_hash_obj.items():
        hash_vector = [int(bit) for bit in str(hash_obj)]
        # Find nearest neighbors within threshold
        neighbors = index.get_nns_by_vector(
            hash_vector, 
            n=100,  # Max neighbors to check
            include_distances=True
        )
        
        for neighbor_idx, distance in zip(neighbors[0], neighbors[1]):
            if distance <= hamming_threshold:
                neighbor_fname = filename_to_hash_obj[neighbor_idx][0]
                G.add_edge(fname, neighbor_fname)
    
    # Find connected components (same as before)
    # ...
```

## Recommended Approach

**Use Option 1 (Multi-Index Hashing)** because:
- ✅ No external dependencies
- ✅ Simple to implement
- ✅ Good performance (70-90% reduction)
- ✅ Maintainable code

**Consider Option 2 (Annoy)** if:
- You need even better performance (95%+ reduction)
- You're okay with external dependency (`pip install annoy`)
- Dataset is very large (>10,000 images)

## Implementation Steps

1. **Create new function** `group_near_duplicates_lsh()` in `src/photo_grouping.py`
2. **Add configuration** to enable/disable LSH (for testing)
3. **Update** `grouping_service.py` to use LSH version
4. **Test** with existing dataset to verify correctness
5. **Benchmark** to measure performance improvement
6. **Replace** old implementation once verified

## Expected Results

### Before (Current)
- **Comparisons**: 174,405 (for 591 images)
- **Time**: 3.932s
- **Complexity**: O(n²)

### After (LSH)
- **Comparisons**: ~5,000-20,000 (70-90% reduction)
- **Time**: 0.4-1.2s (70-90% reduction)
- **Complexity**: O(n log n)

## Testing Strategy

1. **Correctness**: Verify LSH produces same groups as naive approach
2. **Performance**: Measure actual time reduction
3. **Edge cases**: Test with small datasets (<20 images), large datasets (>1000 images)
4. **Threshold sensitivity**: Test with different hamming thresholds

## Rollout Plan

1. **Phase 1**: Implement LSH version alongside existing code
2. **Phase 2**: Add feature flag to switch between implementations
3. **Phase 3**: Test with real datasets
4. **Phase 4**: Enable by default if performance is better
5. **Phase 5**: Remove old implementation after validation



