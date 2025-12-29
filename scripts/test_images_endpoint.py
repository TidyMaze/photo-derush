#!/usr/bin/env python3
"""Test image listing endpoint."""

import requests

BASE_URL = "http://localhost:8000"

print("=" * 60)
print("Testing Image Listing Endpoint")
print("=" * 60)

# Test 1: List all images for project 1 (first page)
print("\n1. GET /api/images?project_id=1&page=1&per_page=10")
response = requests.get(f"{BASE_URL}/api/images", params={
    "project_id": 1,
    "page": 1,
    "per_page": 10
})
print(f"Status: {response.status_code}")
if response.status_code == 200:
    images = response.json()
    print(f"Images returned: {len(images)}")
    if images:
        print(f"First image: {images[0]['filename']}")
        print(f"  - Rating: {images[0]['rating']}")
        print(f"  - Label: {images[0]['label']}")
        print(f"  - Tags: {images[0]['tags']}")
else:
    print(f"Error: {response.text}")

# Test 2: Get detailed image info with EXIF
print("\n2. GET /api/images/{id} (with EXIF)")
if response.status_code == 200 and images:
    image_id = images[0]['id']
    response = requests.get(f"{BASE_URL}/api/images/{image_id}")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        detail = response.json()
        print(f"Image: {detail['filename']}")
        if detail.get('exif'):
            exif = detail['exif']
            print("  EXIF:")
            print(f"    Camera: {exif.get('camera_make')} {exif.get('camera_model')}")
            print(f"    Settings: {exif.get('focal_length')}mm f/{exif.get('aperture')} ISO{exif.get('iso')}")
            print(f"    Dimensions: {exif.get('width')}x{exif.get('height')}")
        else:
            print("  No EXIF data")

# Test 3: Filter by rating
print("\n3. GET /api/images?project_id=1&rating_min=3")
response = requests.get(f"{BASE_URL}/api/images", params={
    "project_id": 1,
    "rating_min": 3,
    "per_page": 5
})
print(f"Status: {response.status_code}")
if response.status_code == 200:
    images = response.json()
    print(f"Images with rating >= 3: {len(images)}")

# Test 4: Pagination
print("\n4. Pagination test (page 2)")
response = requests.get(f"{BASE_URL}/api/images", params={
    "project_id": 1,
    "page": 2,
    "per_page": 10
})
print(f"Status: {response.status_code}")
if response.status_code == 200:
    images = response.json()
    print(f"Page 2 images: {len(images)}")

print("\n" + "=" * 60)
print("✅ Image listing endpoint tests complete!")
print("=" * 60)

