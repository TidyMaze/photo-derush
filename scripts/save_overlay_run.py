#!/usr/bin/env python3
import os
from src.object_detection import save_overlay
from pathlib import Path

p = os.path.join(os.path.expanduser('~/Pictures/photo-dataset'), 'PXL_20250526_190857345.jpg')
print('Generating overlay for', p)
try:
    out = save_overlay(p)
    print('Saved overlay to', out)
    # Print a marker so caller can grep for it
    print('OVERLAY_PATH:' + str(out))
except Exception as e:
    import traceback
    traceback.print_exc()
    raise
