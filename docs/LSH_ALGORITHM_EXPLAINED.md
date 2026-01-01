# LSH Algorithm Explained Simply

## The Problem

**Naive approach**: Compare every image hash with every other image hash.
- For 591 images: 591 × 590 / 2 = **174,405 comparisons**
- Most comparisons are wasted (comparing very different images)

**LSH approach**: Only compare images that are likely to be similar.
- For 591 images: ~5,000-20,000 comparisons (70-90% reduction)

## The Key Idea

**If two images are similar, their hashes share some common bits.**

Instead of comparing all pairs, we:
1. Split each hash into segments
2. Group images by segments they share
3. Only compare images within the same groups

## Step-by-Step Example

### Step 1: We Have 4 Images

Let's say we have 4 images with these 64-bit hashes (shown in hex for readability):

```
Image A: 0x1234567890ABCDEF  (similar to B)
Image B: 0x1234567890ABCCEF  (similar to A, differs by 1 bit)
Image C: 0xABCDEF1234567890  (very different)
Image D: 0xABCDEF1234567891  (similar to C, differs by 1 bit)
```

### Step 2: Split Hash into Segments

We split the 64-bit hash into 4 segments of 16 bits each:

```
Image A: [0x1234] [0x5678] [0x90AB] [0xCDEF]
Image B: [0x1234] [0x5678] [0x90AB] [0xCCEF]  ← Segment 4 differs
Image C: [0xABCD] [0xEF12] [0x3456] [0x7890]
Image D: [0xABCD] [0xEF12] [0x3456] [0x7891]  ← Segment 4 differs
```

### Step 3: Build Index Tables

For each segment position, we create an index:

**Segment 0 Index:**
```
0x1234 → [Image A, Image B]  ← A and B share segment 0
0xABCD → [Image C, Image D]  ← C and D share segment 0
```

**Segment 1 Index:**
```
0x5678 → [Image A, Image B]  ← A and B share segment 1
0xEF12 → [Image C, Image D]  ← C and D share segment 1
```

**Segment 2 Index:**
```
0x90AB → [Image A, Image B]  ← A and B share segment 2
0x3456 → [Image C, Image D]  ← C and D share segment 2
```

**Segment 3 Index:**
```
0xCDEF → [Image A]           ← Only A has this
0xCCEF → [Image B]           ← Only B has this
0x7890 → [Image C]           ← Only C has this
0x7891 → [Image D]           ← Only D has this
```

### Step 4: Find Candidates for Each Image

For **Image A**:
- Check Segment 0 (0x1234) → finds [A, B]
- Check Segment 1 (0x5678) → finds [A, B]
- Check Segment 2 (0x90AB) → finds [A, B]
- Check Segment 3 (0xCDEF) → finds [A]

**Candidates for A**: {B} (from segments 0, 1, 2)

For **Image B**:
- Check Segment 0 (0x1234) → finds [A, B]
- Check Segment 1 (0x5678) → finds [A, B]
- Check Segment 2 (0x90AB) → finds [A, B]
- Check Segment 3 (0xCCEF) → finds [B]

**Candidates for B**: {A} (from segments 0, 1, 2)

For **Image C**:
- Check Segment 0 (0xABCD) → finds [C, D]
- Check Segment 1 (0xEF12) → finds [C, D]
- Check Segment 2 (0x3456) → finds [C, D]
- Check Segment 3 (0x7890) → finds [C]

**Candidates for C**: {D} (from segments 0, 1, 2)

For **Image D**:
- Check Segment 0 (0xABCD) → finds [C, D]
- Check Segment 1 (0xEF12) → finds [C, D]
- Check Segment 2 (0x3456) → finds [C, D]
- Check Segment 3 (0x7891) → finds [D]

**Candidates for D**: {C} (from segments 0, 1, 2)

### Step 5: Compare Only Candidates

**Naive approach**: 4 × 3 / 2 = **6 comparisons**
- A vs B, A vs C, A vs D, B vs C, B vs D, C vs D

**LSH approach**: **2 comparisons**
- A vs B (they're candidates)
- C vs D (they're candidates)
- Skip A vs C, A vs D, B vs C, B vs D (not candidates)

**Result**: 67% reduction (6 → 2 comparisons)

## Why It Works

### Similar Images Share Segments

If two images are similar:
- Their hashes are similar (small Hamming distance)
- They likely share at least one segment
- LSH finds them as candidates

### Different Images Don't Share Segments

If two images are very different:
- Their hashes are very different (large Hamming distance)
- They don't share segments
- LSH doesn't compare them (saves time)

## Real Example with 591 Images

### Naive Approach
```
For each image (591 images):
  For each other image (590 comparisons):
    Compare hashes
Total: 591 × 590 / 2 = 174,405 comparisons
```

### LSH Approach
```
1. Build 4 index tables (one per segment)
2. For each image:
   - Check 4 segments
   - Find candidates (images sharing at least one segment)
   - Compare only with candidates
Total: ~5,000-20,000 comparisons (70-90% reduction)
```

## Why LSH is Faster

**Key insight**: Most images are unique, so most comparisons are wasted.

**Example**: Out of 591 images:
- Maybe 50 are similar (near-duplicates)
- 541 are unique

**Naive**: Compares all 591 × 590 / 2 = 174,405 pairs
**LSH**: Only compares ~5,000-20,000 pairs (the ones that share segments)

**Time saved**: 70-90% reduction in comparisons = 70-90% faster

## Trade-offs

### Advantages
- ✅ Much faster for large datasets
- ✅ Same results as naive approach (correctness preserved)
- ✅ Scales well (O(n log n) vs O(n²))

### Limitations
- ⚠️ Small overhead for building index (negligible for large datasets)
- ⚠️ Less benefit when all images are very similar (rare case)
- ⚠️ Best for datasets with many unique images

## When LSH Helps Most

**Best case**: Large dataset (500+ images) with many unique images
- **Example**: 591 vacation photos, most are unique
- **Benefit**: 70-90% reduction in comparisons

**Worst case**: Small dataset (< 20 images) or all images very similar
- **Example**: 10 images, all nearly identical
- **Benefit**: Minimal (overhead may outweigh benefit)
- **Solution**: LSH only enabled for datasets > 20 images

## Summary

**LSH = Smart Filtering**

Instead of comparing everything with everything:
1. **Group** images by shared hash segments
2. **Compare** only images in the same groups
3. **Skip** comparisons that are unlikely to match

**Result**: Same accuracy, much faster for large datasets.


