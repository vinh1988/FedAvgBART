#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re

# Target schema
FIELDNAMES = [
    'num_clients', 'dirichlet_alpha', 'round', 'loss',
    'rouge1_f1', 'rouge2_f1', 'rougeL_f1',
    'bleu1', 'bleu2', 'bleu3', 'bleu4',
    'gini_coefficient', 'mean_contribution', 'min_contribution', 'max_contribution', 'cv_contribution'
]


def read_config_num_clients(run_dir: Path) -> int:
    cfg = run_dir / 'config.yaml'
    if not cfg.exists():
        return -1
    try:
        import yaml
    except Exception:
        # Fallback: simple parse for 'num_clients:'
        for line in cfg.read_text().splitlines():
            if line.strip().startswith('num_clients:'):
                try:
                    return int(line.split(':', 1)[1].strip())
                except Exception:
                    return -1
        return -1
    with open(cfg, 'r') as f:
        data = yaml.safe_load(f)
    return int(data.get('num_clients', -1))

def read_config_alpha(run_dir: Path) -> float | None:
    cfg = run_dir / 'config.yaml'
    if not cfg.exists():
        return None
    try:
        import yaml
        with open(cfg, 'r') as f:
            data = yaml.safe_load(f)
        alpha = data.get('dirichlet_alpha', None)
        return float(alpha) if alpha is not None else None
    except Exception:
        # Fallback: simple parse
        for line in cfg.read_text().splitlines():
            if line.strip().startswith('dirichlet_alpha:'):
                try:
                    return float(line.split(':', 1)[1].strip())
                except Exception:
                    return None
        return None


def load_metrics_csv(run_dir: Path) -> List[Dict[str, Any]]:
    metrics_csv = run_dir / 'metrics.csv'
    if not metrics_csv.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(metrics_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def convert_row(num_clients: int, dirichlet_alpha: float | None, row: Dict[str, Any]) -> Dict[str, Any]:
    # Map DistilBART metrics.csv columns into the target schema
    # Use val_loss as consolidated 'loss' (as in BART consolidation)
    def to_float(x):
        try:
            return float(x) if x not in (None, '', 'None') else None
        except Exception:
            return None

    return {
        'num_clients': num_clients,
        'dirichlet_alpha': dirichlet_alpha,
        'round': int(row.get('round', 0) or 0),
        'loss': to_float(row.get('val_loss')),
        'rouge1_f1': to_float(row.get('rouge1')),
        'rouge2_f1': to_float(row.get('rouge2')),
        'rougeL_f1': to_float(row.get('rougeL')),
        'bleu1': to_float(row.get('bleu1')),
        'bleu2': to_float(row.get('bleu2')),
        'bleu3': to_float(row.get('bleu3')),
        'bleu4': to_float(row.get('bleu4')),
        'gini_coefficient': to_float(row.get('gini_coefficient')),
        'mean_contribution': to_float(row.get('mean_contribution')),
        'min_contribution': to_float(row.get('min_contribution')),
        'max_contribution': to_float(row.get('max_contribution')),
        'cv_contribution': to_float(row.get('cv_contribution')),
    }


def parse_run_timestamp(name: str) -> Tuple[int, int]:
    # name like run_YYYYMMDD_HHMMSS
    m = re.match(r"run_(\d{8})_(\d{6})", name)
    if not m:
        return (0, 0)
    return (int(m.group(1)), int(m.group(2)))


def consolidate(base_dir: Path) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if not base_dir.exists():
        return results

    # Collect candidate (nc, alpha, run_dir) tuples from both new and legacy layouts
    candidates: List[Tuple[int, float | None, Path]] = []

    # New layout: base_dir/nc_{N}/run_*
    for nc_dir in base_dir.glob('nc_*'):
        if not nc_dir.is_dir():
            continue
        try:
            nc = int(str(nc_dir.name).split('_', 1)[1])
        except Exception:
            continue
        for run_dir in nc_dir.glob('run_*'):
            if run_dir.is_dir():
                candidates.append((nc, read_config_alpha(run_dir), run_dir))

    # Legacy layout: base_dir/run_* (infer num_clients from config.yaml)
    for run_dir in base_dir.glob('run_*'):
        if not run_dir.is_dir():
            continue
        nc = read_config_num_clients(run_dir)
        if nc >= 0:
            candidates.append((nc, read_config_alpha(run_dir), run_dir))

    # Group by (num_clients, dirichlet_alpha) and pick latest by timestamp
    runs_by_key: Dict[Tuple[int, float | None], Path] = {}
    for nc, alpha, run_dir in candidates:
        ts = parse_run_timestamp(run_dir.name)
        key = (nc, alpha)
        if key not in runs_by_key or parse_run_timestamp(runs_by_key[key].name) < ts:
            runs_by_key[key] = run_dir

    # Consolidate only the latest run per (num_clients, dirichlet_alpha)
    for (nc, alpha), run_dir in sorted(runs_by_key.items(), key=lambda kv: (kv[0][0], float('inf') if kv[0][1] is None else kv[0][1])):
        rows = load_metrics_csv(run_dir)
        for r in rows:
            out_row = convert_row(nc, alpha, r)
            results.append(out_row)

    # Sort by num_clients, then round
    results.sort(key=lambda x: (x['num_clients'], (float('inf') if x['dirichlet_alpha'] is None else x['dirichlet_alpha']), x['round']))
    return results


def write_consolidated(rows: List[Dict[str, Any]], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    # Usage: consolidate_distilbart_metrics.py [base_results_dir] [output_csv]
    cwd = Path.cwd()
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else cwd / 'results_distilbart_cnndm_federated'
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else base / 'consolidated_metrics_distil_gen_fed.csv'
    rows = consolidate(base)
    write_consolidated(rows, out)
    print(f'Wrote consolidated CSV: {out} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
