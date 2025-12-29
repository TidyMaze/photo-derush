#!/usr/bin/env python3
"""Promote a model artifact to the app's default model path.

Usage:
  poetry run python scripts/promote_model.py /tmp/photo_combined_pca128.joblib

This will copy the model file to the path used by the app (DEFAULT_MODEL_PATH in src/inference.py)
and also copy any calibrator file (model_path + '.calib.joblib') if present.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import os
import joblib

from pathlib import Path

# Default model location used by the app (match src/inference.py conventions)
DEFAULT_MODEL_PATH = os.path.join('models', 'current_model.joblib')


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('source', help='Path to source model joblib to promote')
    parser.add_argument('--dest', default=DEFAULT_MODEL_PATH, help='Destination path for promoted model')
    parser.add_argument('--pca-dim', type=int, default=None, help='If promoting a combined model, specify embedding/PCA dim (e.g. 128)')
    args = parser.parse_args(argv or sys.argv[1:])

    src = Path(args.source)
    dest = Path(args.dest)

    if not src.exists():
        print('Source model not found:', src)
        return 2

    dest.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src, dest)
    print(f'Copied {src} -> {dest}')

    # Copy calibrator if exists
    calib_src = src.with_name(src.name + '.calib.joblib')
    if calib_src.exists():
        calib_dest = dest.with_name(dest.name + '.calib.joblib')
        shutil.copy2(calib_src, calib_dest)
        print(f'Copied calibrator {calib_src} -> {calib_dest}')

    # If the model contains a 'pca' object, also write a small metadata file alongside the model
    try:
        data = joblib.load(str(src))
        if isinstance(data, dict) and 'pca' in data and data['pca'] is not None:
            meta = {'pca_dim': data.get('pca_dim')}
            meta_path = dest.with_name(dest.name + '.meta.json')
            import json

            with open(meta_path, 'w') as f:
                json.dump(meta, f)
            print('Wrote metadata', meta_path)
    except Exception:
        pass

    # If the promoted model lacks __metadata__ or feature_length, try to attach metadata so src.inference can accept it.
    try:
        data = joblib.load(str(dest))
        needs_meta = not isinstance(data, dict) or data.get('__metadata__') is None or data.get('feature_length') is None
        if needs_meta:
            try:
                from src.model_version import create_model_metadata
                from src.features import FEATURE_COUNT, USE_FULL_FEATURES
                # Determine embedding dim
                emb_dim = None
                if args.pca_dim:
                    emb_dim = int(args.pca_dim)
                elif isinstance(data, dict) and data.get('pca_dim'):
                    emb_dim = int(data.get('pca_dim'))
                else:
                    # best-effort: try to find a cached embeddings file
                    try:
                        embfile = os.path.join('.cache', 'embeddings_resnet18_full.joblib')
                        if os.path.isfile(embfile):
                            ed = joblib.load(embfile)
                            emb_dim = int(ed.get('embeddings').shape[1])
                    except Exception:
                        emb_dim = None

                feature_length = FEATURE_COUNT + (emb_dim or 0)
                metadata = create_model_metadata(feature_count=FEATURE_COUNT, feature_mode='FULL' if USE_FULL_FEATURES else 'FAST', params={}, n_samples=data.get('n_samples') or 0)
                if not isinstance(data, dict):
                    data = {'model': data}
                data['__metadata__'] = metadata
                data['feature_length'] = int(feature_length)
                if emb_dim:
                    data['pca_dim'] = int(emb_dim)
                    data['model_type'] = 'combined'
                data.setdefault('n_samples', data.get('n_samples') or 0)
                data.setdefault('n_keep', data.get('n_keep') or 0)
                data.setdefault('n_trash', data.get('n_trash') or 0)
                data.setdefault('filenames', data.get('filenames') or [])
                joblib.dump(data, str(dest))
                print('Attached metadata to promoted model (feature_length=%d)' % feature_length)
            except Exception as e:
                print('Failed to attach metadata to promoted model:', e)
    except Exception:
        pass

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
