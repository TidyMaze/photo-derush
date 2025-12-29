#!/usr/bin/env python3
"""Generate a detailed benchmark report from `.cache/res_*.json` outputs.

Creates:
- .cache/benchmark_summary.csv
- .cache/report_plots/*.png
- .cache/benchmark_report.html (dashboard embedding plots; interactive if mpld3 available)
"""
import json
from pathlib import Path
import base64
import io
import sys

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / '.cache'
PLOTS_DIR = CACHE / 'report_plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def load_results():
    recs = []
    for p in sorted(CACHE.glob('res_*.json')):
        try:
            txt = p.read_text()
            data = json.loads(txt)
        except Exception:
            continue
        # aggregated file: list of per-image records
        if isinstance(data, list):
            for r in data:
                r['_source_file'] = str(p.name)
                recs.append(r)
        # single record
        elif isinstance(data, dict):
            data['_source_file'] = str(p.name)
            recs.append(data)
    return recs

def explode_records(recs):
    import pandas as pd
    rows = []
    for r in recs:
        backend = r.get('backend')
        mode = r.get('mode')
        image = r.get('image')
        load = r.get('load_duration')
        counts = r.get('inference_counts') or []
        durs = r.get('inference_durations') or []
        for idx, d in enumerate(durs):
            rows.append({'backend': backend, 'mode': mode, 'image': image, 'load_duration': load, 'inference_duration': d, 'detection_count': counts[idx] if idx < len(counts) else None})
    df = pd.DataFrame(rows)
    return df

def summary_stats(df):
    import pandas as pd
    grp = df.groupby(['backend','mode'])
    agg = grp.agg(inference_mean=('inference_duration','mean'), inference_median=('inference_duration','median'), inference_std=('inference_duration','std'), load_mean=('load_duration','mean'), load_median=('load_duration','median'), detection_mean=('detection_count','mean'))
    agg = agg.reset_index()
    return agg

def save_csv(agg):
    out = CACHE / 'benchmark_summary.csv'
    agg.to_csv(out, index=False)
    print('Saved CSV summary to', out)

def plot_all(df, agg):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style='whitegrid')

    # Boxplot by backend and mode
    plt.figure(figsize=(10,6))
    ax = sns.boxplot(x='backend', y='inference_duration', hue='mode', data=df)
    ax.set_title('Inference durations by backend and mode')
    f1 = PLOTS_DIR / 'inference_boxplot.png'
    plt.savefig(f1, dpi=150)
    plt.close()

    # Load duration barplot (agg has load_mean column)
    plt.figure(figsize=(8,4))
    if 'load_mean' in agg.columns:
        ax = sns.barplot(x='backend', y='load_mean', hue='mode', data=agg)
    else:
        ax = sns.barplot(x='backend', y='load_duration', hue='mode', data=agg)
    ax.set_title('Model load duration (mean)')
    f2 = PLOTS_DIR / 'load_duration_bar.png'
    plt.savefig(f2, dpi=150)
    plt.close()

    # Detection count distribution
    plt.figure(figsize=(8,4))
    ax = sns.boxplot(x='backend', y='detection_count', hue='mode', data=df)
    ax.set_title('Detection counts per inference')
    f3 = PLOTS_DIR / 'detection_counts_box.png'
    plt.savefig(f3, dpi=150)
    plt.close()

    # Per-image comparison heatmap (median inference per backend-mode)
    import pandas as pd
    med = df.groupby(['image','backend','mode']).inference_duration.median().reset_index()
    # pivot to backend|mode columns
    med['col'] = med['backend'] + ':' + med['mode']
    pivot = med.pivot(index='image', columns='col', values='inference_duration')
    if not pivot.empty:
        plt.figure(figsize=(12, max(4, min(40, pivot.shape[0]*0.2))))
        sns.heatmap(pivot.fillna(0), cmap='viridis')
        plt.title('Median inference duration per image (s)')
        f4 = PLOTS_DIR / 'per_image_heatmap.png'
        plt.savefig(f4, dpi=150)
        plt.close()
    else:
        f4 = None

    return [f1, f2, f3, f4]

def make_html(plot_files, agg):
    html = ['<html><head><meta charset="utf-8"><title>Benchmark Report</title></head><body>']
    html.append('<h1>Benchmark Report</h1>')
    html.append('<h2>Summary statistics</h2>')
    html.append('<pre>')
    html.append(agg.to_string(index=False))
    html.append('</pre>')

    html.append('<h2>Plots</h2>')
    for p in plot_files:
        if p is None:
            continue
        with open(p, 'rb') as fh:
            b = fh.read()
        data = base64.b64encode(b).decode('ascii')
        html.append(f'<h3>{p.name}</h3>')
        html.append(f'<img src="data:image/png;base64,{data}" style="max-width:100%;height:auto;"/>')

    html.append('</body></html>')
    out = CACHE / 'benchmark_report.html'
    out.write_text('\n'.join(html))
    print('Saved HTML report to', out)

def try_mpld3_interactive(df):
    try:
        import mpld3
    except Exception:
        return False
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(10,6))
    sns.boxplot(x='backend', y='inference_duration', hue='mode', data=df, ax=ax)
    html = mpld3.fig_to_html(fig)
    out = CACHE / 'benchmark_report_interactive.html'
    out.write_text(html)
    print('Saved interactive report to', out)
    return True

def main():
    recs = load_results()
    if not recs:
        print('No result files found in .cache. Run benchmark first.')
        sys.exit(1)
    df = explode_records(recs)
    if df.empty:
        print('No inference records found in results.')
        sys.exit(1)
    agg = summary_stats(df)
    save_csv(agg)
    plot_files = plot_all(df, agg)
    make_html(plot_files, agg)
    if not try_mpld3_interactive(df):
        print('mpld3 not available; interactive HTML not generated (static HTML created).')

if __name__ == '__main__':
    main()
