import json
import os
import tempfile
from pathlib import Path

from PIL import Image

from src.param_sweep import _parse_values, plot_sweep, sweep_hyperparameter


def _make_dataset(n_pairs=3):
    tmp = Path(tempfile.mkdtemp(prefix='sweep_ds_'))
    ratings = {}
    for i in range(n_pairs):
        k = tmp / f'keep_{i}.jpg'
        t = tmp / f'trash_{i}.jpg'
        Image.new('RGB', (32,32), (250,10+i,10)).save(k, 'JPEG')
        Image.new('RGB', (32,32), (10,250,10+i)).save(t, 'JPEG')
        ratings[k.name] = {'state':'keep','label_source':'manual'}
        ratings[t.name] = {'state':'trash','label_source':'manual'}
    with open(tmp/'.ratings_tags.json','w') as f: json.dump(ratings,f)
    return str(tmp)


def test_parse_values_range_and_csv():
    assert _parse_values('1:3') == [1,2,3]
    assert _parse_values('0:0.5:0.25') == [0.0,0.25,0.5]
    assert _parse_values('2,4,6') == [2,4,6]
    assert _parse_values('fast,2') == ['fast',2]


def test_sweep_and_plot(tmp_path):
    ds = _make_dataset(n_pairs=4)
    # sweep max_depth small values
    results = sweep_hyperparameter(ds, 'max_depth', [2,3,4], cv_folds=2)
    assert len(results) == 3
    # At least one result should have a mean value
    assert any(r['mean'] is not None for r in results)
    out = plot_sweep(results, 'max_depth')
    assert out and os.path.exists(out)

