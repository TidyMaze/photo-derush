import json
import math
from pathlib import Path
from typing import Dict, List
import sys
import argparse

# Ensure project root on path when executing as a script from within the ml/ directory
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from ml.features import all_feature_names
except Exception:
    all_feature_names = None  # type: ignore

# Correct path (always root/ml/feature_cache.json)
FEATURE_CACHE_PATH = ROOT / 'ml' / 'feature_cache.json'


def load_cache() -> Dict[str, dict]:
    if not FEATURE_CACHE_PATH.exists():
        raise SystemExit(f'feature_cache.json not found at {FEATURE_CACHE_PATH}')
    try:
        return json.loads(FEATURE_CACHE_PATH.read_text())
    except Exception as e:  # noqa: PERF203
        raise SystemExit(f'Failed reading feature cache: {e}')


def feature_variation_report(cache: Dict[str, dict]):
    # Build list of per-image mappings from keys->value
    per_image_maps: List[Dict[str, float]] = []
    union_keys = set()
    for rec in cache.values():
        keys = rec.get('keys')
        fv = rec.get('fv')
        if not keys or not fv:
            continue
        mapping = dict(zip(keys, fv))
        per_image_maps.append(mapping)
        union_keys.update(keys)
    if all_feature_names:
        all_names = all_feature_names(include_strings=False)
    else:
        all_names = sorted(union_keys)
    zero_only = []
    non_zero = []
    first_non_zero_example = {}
    eps = 1e-9
    for name in all_names:
        example_val = None
        for m in per_image_maps:
            v = m.get(name)
            if isinstance(v, (int, float)) and not math.isnan(v) and abs(v) > eps:
                example_val = v
                break
        if example_val is None:
            zero_only.append(name)
        else:
            non_zero.append(name)
            first_non_zero_example[name] = example_val
    return zero_only, non_zero, first_non_zero_example, len(per_image_maps)


def main():
    parser = argparse.ArgumentParser(description='Check which features have at least one non-zero value across cached images.')
    parser.add_argument('--json', action='store_true', help='Emit JSON summary')
    parser.add_argument('--flat-only', action='store_true', help='Print only features stuck at zero across all images')
    args = parser.parse_args()
    cache = load_cache()
    zero_only, non_zero, examples, n = feature_variation_report(cache)
    if args.json:
        out = {
            'images_analyzed': n,
            'feature_count': len(zero_only) + len(non_zero),
            'non_zero_feature_count': len(non_zero),
            'zero_only_feature_count': len(zero_only),
            'zero_only_features': zero_only,
            'non_zero_features': [{'name': k, 'example_value': examples.get(k)} for k in non_zero]
        }
        print(json.dumps(out, indent=2))
        return
    if args.flat_only:
        print(f'Images analyzed: {n}')
        print(f'Flat (all-zero) features count: {len(zero_only)}')
        for z in zero_only:
            print(z)
        return
    print(f'Images analyzed: {n}')
    print(f'Total features (numeric): {len(zero_only) + len(non_zero)}')
    print(f'Features with at least one non-zero value: {len(non_zero)}')
    print(', '.join(non_zero))
    print('\nFeatures stuck at zero across all images: {cnt}'.format(cnt=len(zero_only)))
    for z in zero_only:
        print(z)


if __name__ == '__main__':
    main()
