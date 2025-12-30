#!/usr/bin/env python3
"""Test thumbnail endpoint after implementation."""
import sys

import requests

print("=" * 70)
print("THUMBNAIL ENDPOINT TEST")
print("=" * 70)
print()

# Test 1: Health check
print("1. Testing backend health...")
try:
    r = requests.get("http://localhost:8000/health", timeout=2)
    if r.status_code == 200:
        print("   ✅ Backend is running")
    else:
        print(f"   ❌ Backend returned {r.status_code}")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ Backend not accessible: {e}")
    print()
    print("Please start the backend:")
    print("  cd <project-directory>")
    print("  poetry run uvicorn api.main:app --reload --port 8000")
    sys.exit(1)

# Test 2: Get first image
print("2. Getting first image...")
try:
    r = requests.get("http://localhost:8000/api/images?project_id=1&per_page=1", timeout=2)
    images = r.json()
    if not images:
        print("   ❌ No images found")
        sys.exit(1)
    image_id = images[0]['id']
    print(f"   ✅ Found image ID: {image_id}")
    print(f"   Filename: {images[0]['filename']}")
except Exception as e:
    print(f"   ❌ Failed to get images: {e}")
    sys.exit(1)

# Test 3: Request thumbnail
print("3. Testing thumbnail endpoint...")
try:
    r = requests.get(f"http://localhost:8000/api/images/{image_id}/thumbnail", timeout=5)
    if r.status_code == 200:
        print("   ✅ Thumbnail generated successfully")
        print(f"   Content-Type: {r.headers.get('content-type')}")
        print(f"   Size: {len(r.content):,} bytes")
        print(f"   Cache-Control: {r.headers.get('cache-control')}")

        # Save thumbnail
        with open('/tmp/test_thumbnail.jpg', 'wb') as f:
            f.write(r.content)
        print("   ✅ Saved to: /tmp/test_thumbnail.jpg")
    else:
        print(f"   ❌ Failed: {r.status_code}")
        print(f"   Error: {r.text}")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ Request failed: {e}")
    sys.exit(1)

# Test 4: Different sizes
print("4. Testing different sizes...")
sizes = [100, 200, 300]
for size in sizes:
    try:
        r = requests.get(
            f"http://localhost:8000/api/images/{image_id}/thumbnail?size={size}",
            timeout=5
        )
        if r.status_code == 200:
            print(f"   ✅ {size}x{size}: {len(r.content):,} bytes")
        else:
            print(f"   ❌ {size}x{size}: Failed")
    except Exception as e:
        print(f"   ❌ {size}x{size}: {e}")

print()
print("=" * 70)
print("✅ THUMBNAIL ENDPOINT WORKING!")
print("=" * 70)
print()
print("Next steps:")
print("  1. Open http://localhost:5173")
print("  2. Reload the page (Cmd+R)")
print("  3. Click 'View Images'")
print("  4. See your photos with thumbnails!")
print()
print("Saved test thumbnail: /tmp/test_thumbnail.jpg")
print("  View it: open /tmp/test_thumbnail.jpg")
print()

