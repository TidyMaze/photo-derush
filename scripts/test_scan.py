#!/usr/bin/env python3
"""Test the scan endpoint."""
import json

import requests

# Test scan endpoint
print("Testing POST /api/projects/1/scan...")
response = requests.post("http://localhost:8000/api/projects/1/scan")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Check updated image count
print("\nGetting project details...")
response = requests.get("http://localhost:8000/api/projects/1")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

