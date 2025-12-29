"""Quick test to trace label update flow"""
import logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(message)s')

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Monkey-patch to trace calls
original_refresh = None
original_get_state = None

def trace_refresh(self):
    print(f"\n>>> _refresh_thumbnail_badges called")
    return original_refresh(self)

def trace_get_state(self, path):
    result = original_get_state(self, path)
    fname = Path(path).name
    print(f"    model.get_state({fname}) = {result}")
    return result

from src import view, model
original_refresh = view.PhotoView._refresh_thumbnail_badges
original_get_state = model.Model.get_state
view.PhotoView._refresh_thumbnail_badges = trace_refresh
model.Model.get_state = trace_get_state

# Now run a short test
import pytest
sys.exit(pytest.main([
    'tests/test_auto_label.py::test_auto_label_applies',
    '-xvs', '--tb=short', '--capture=no'
]))
