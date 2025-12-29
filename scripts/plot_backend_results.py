#!/usr/bin/env python3
"""Run backend worker for each detection backend (10 runs) and plot results.

Creates a PNG report at `.cache/backend_inference_stats.png` showing inference
duration distributions and load times per backend.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parent
WORKER = str(ROOT / 'check_backend_worker.py')
IMAGE = os.environ.get('CHECK_IMAGE', os.path.join(os.path.expanduser('~/Pictures/photo-dataset'), 'PXL_20250712_183833561.PORTRAIT.jpg'))
BACKENDS = ['fasterrcnn', 'ssdlite', 'yolov8']
MODES = ['mps', 'cpu']
# Number of inference repetitions per image
N = int(os.environ.get('BENCH_REPEATS', '10'))
# Dataset folder to sample images from
DATA_DIR = Path(os.environ.get('BENCH_DIR', str(Path(__file__).resolve().parent.parent / 'test_images')))
# How many random images to sample for the benchmark
NUM_IMAGES = int(os.environ.get('BENCH_IMAGES', '20'))

results = []
out_dir = Path('.cache')
out_dir.mkdir(parents=True, exist_ok=True)

# Per-backend timeout in seconds
WORKER_TIMEOUT = 120

import random
import shutil

# Collect images to benchmark
images = []
if DATA_DIR.exists() and DATA_DIR.is_dir():
    all_imgs = [str(p) for p in DATA_DIR.glob('**/*') if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    if not all_imgs:
        print(f'No images found in {DATA_DIR}; falling back to single image {IMAGE}')
        images = [IMAGE]
    else:
        random.seed(0)
        images = random.sample(all_imgs, min(NUM_IMAGES, len(all_imgs)))
else:
    print(f'Data directory {DATA_DIR} not found; falling back to single image {IMAGE}')
    images = [IMAGE]

print(f'Benchmark will run on {len(images)} images')

def print_progress(prefix, idx, total, width=None):
    if width is None:
        width = shutil.get_terminal_size((80, 20)).columns
    bar_len = max(20, min(60, width - 40))
    filled = int(bar_len * (idx / float(total)))
    bar = '[' + '#' * filled + '-' * (bar_len - filled) + ']'
    pct = f'{(idx/float(total))*100:5.1f}%'
    print(f"{prefix} {bar} {pct} ({idx}/{total})", end='\r', flush=True)

for b in BACKENDS:
    for mode in MODES:
        print(f"Running backend {b} in mode {mode} on {len(images)} images (N={N} repeats each)")
        per_image_results = []
        total = len(images)
        for i, img in enumerate(images, start=1):
            print_progress(f'{b}/{mode}', i, total)
            cmd = [PY, WORKER, b, img, str(N), mode]
            try:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=WORKER_TIMEOUT, text=True)
                raw = proc.stdout.strip()
            except subprocess.TimeoutExpired:
                raw = json.dumps({'backend': b, 'mode': mode, 'image': img, 'error': 'timeout'})

            # Save raw output per image
            safe_name = Path(img).stem.replace(' ', '_')
            res_path = out_dir / f'res_{b}_{mode}_{safe_name}.json'
            with open(res_path, 'w') as fh:
                fh.write(raw + '\n')

            # Parse last JSON object
            last_line = raw.splitlines()[-1] if raw else ''
            try:
                rec = json.loads(last_line)
            except Exception:
                rec = {'backend': b, 'mode': mode, 'image': img, 'error': 'parse_failure', 'raw': raw}
            rec['mode'] = mode
            rec['image'] = img
            per_image_results.append(rec)

            print(f"Saved raw output to {res_path}")

        # Aggregate per-backend-mode
        print_progress(f'{b}/{mode}', total, total)
        print()
        agg_path = out_dir / f'res_{b}_{mode}.json'
        with open(agg_path, 'w') as fh:
            fh.write(json.dumps(per_image_results) + '\n')
        print(f'Wrote aggregate results to {agg_path}')
        results.extend(per_image_results)

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception as e:
    print("Missing plotting dependencies: pandas, matplotlib, seaborn. Install in venv to generate report.")
    sys.exit(1)

# Prepare DataFrame for inference durations
rows = []
load_rows = []
for r in results:
    b = r.get('backend')
    load_rows.append({'backend': b, 'load_duration': r.get('load_duration')})
    durs = r.get('inference_durations') or []
    for d in durs:
        rows.append({'backend': b, 'inference_duration': d})

df = pd.DataFrame(rows)
df_load = pd.DataFrame(load_rows)

out_dir = Path('.cache')
out_dir.mkdir(exist_ok=True)
out_file = out_dir / 'backend_inference_stats.png'

plt.figure(figsize=(10,6))
if not df.empty:
    sns.set(style='whitegrid')
    ax = plt.subplot(1,2,1)
    sns.boxplot(x='backend', y='inference_duration', data=df, ax=ax)
    sns.swarmplot(x='backend', y='inference_duration', data=df, color='.25', ax=ax)
    ax.set_title('Inference durations (s)')
else:
    plt.subplot(1,2,1)
    plt.text(0.5, 0.5, 'No inference data', ha='center')

ax2 = plt.subplot(1,2,2)
if not df_load.empty:
    sns.barplot(x='backend', y='load_duration', data=df_load, ax=ax2)
    ax2.set_title('Model load duration (s)')
else:
    ax2.text(0.5, 0.5, 'No load data', ha='center')

plt.tight_layout()
plt.savefig(out_file, dpi=150)
print(f"Saved report to {out_file}")

print('\nJSON results:')
print(json.dumps(results, indent=2))
