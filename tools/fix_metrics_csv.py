#!/usr/bin/env python3
import json
import csv
import sys
from pathlib import Path
import numpy as np

FIELDNAMES = [
    'round',
    'train_loss', 'val_loss', 'test_loss',
    'rouge1', 'rouge2', 'rougeL',
    'bleu1', 'bleu2', 'bleu3', 'bleu4',
    'gini_coefficient', 'mean_contribution', 'min_contribution',
    'max_contribution', 'cv_contribution', 'num_samples'
]

def load_metrics_json(run_dir: Path):
    metrics_path = run_dir / 'metrics.json'
    with open(metrics_path, 'r') as f:
        return json.load(f)


def align_and_write_csv(run_dir: Path, metrics: dict):
    # Determine max length across lists
    lists = [v for v in metrics.values() if isinstance(v, list)]
    if not lists:
        raise ValueError('No list metrics found in metrics.json')
    max_len = max(len(v) for v in lists)

    # Align test_loss to last round
    if 'test_loss' in metrics and isinstance(metrics['test_loss'], list):
        non_none = [v for v in metrics['test_loss'] if v is not None]
        aligned = [None] * max_len
        if non_none:
            aligned[-1] = non_none[-1]
        metrics['test_loss'] = aligned

    # Pad all other arrays
    for k, v in metrics.items():
        if isinstance(v, list) and len(v) < max_len:
            metrics[k] = v + [None] * (max_len - len(v))

    # Build CSV rows
    rows = []
    for i in range(max_len):
        row = {'round': i + 1}
        for key in FIELDNAMES[1:]:
            val = metrics.get(key, [None] * max_len)[i]
            # Convert numpy types
            if isinstance(val, (np.generic,)):
                val = val.item()
            row[key] = val
        rows.append(row)

    # Write CSV
    out_path = run_dir / 'metrics.csv'
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f'Wrote fixed CSV: {out_path}')


def main():
    if len(sys.argv) != 2:
        print('Usage: fix_metrics_csv.py <run_dir>')
        sys.exit(1)
    run_dir = Path(sys.argv[1]).resolve()
    metrics = load_metrics_json(run_dir)
    align_and_write_csv(run_dir, metrics)

if __name__ == '__main__':
    main()
