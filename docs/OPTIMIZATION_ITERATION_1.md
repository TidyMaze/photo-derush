# Optimization Iteration 1

## Optimizations Implemented

### 1. Duplicate Grouping O(n²) Loop Optimization
**File**: `src/duplicate_grouping.py`

**Change**: Use separate dict for ImageHash objects to avoid isinstance checks in hot loop.

**Before**:
```python
for existing_hash_obj, gid in hash_obj_to_group.items():
    if isinstance(existing_hash_obj, str) and existing_hash_obj.startswith("fallback_"):
        continue
    if isinstance(existing_hash_obj, imagehash.ImageHash):
        if phash - existing_hash_obj <= hamming_threshold:
            ...
```

**After**:
```python
# Separate dict for ImageHash objects only (no isinstance checks)
imagehash_to_group: dict[imagehash.ImageHash, int] = {}
for existing_hash_obj, gid in imagehash_to_group.items():
    # Direct comparison without isinstance check
    if phash - existing_hash_obj <= hamming_threshold:
        ...
```

**Expected Impact**: 10-20% reduction in comparison overhead (removes isinstance checks from hot loop

### 2. Badge Refresh - Visible Thumbnails Only
**File**: `src/view.py`

**Change**: Only refresh badges for thumbnails visible in viewport.

**Before**:
```python
for (row, col), label in self.label_refs.items():
    # Process all labels (could be 1000+)
    ...
```

**After**:
```python
# Get visible labels from viewport
visible_labels = filter_visible_labels(self.scroll_area, self.label_refs)
for (row, col), label in visible_labels:
    # Process only visible labels (typically 20-50)
    ...
```

**Expected Impact**: 80-90% reduction in badge refresh work (from all labels to only visible ones)

## Next Steps

1. Run profile to measure actual improvement
2. Identify next hotspot
3. Implement next optimization
4. Repeat

