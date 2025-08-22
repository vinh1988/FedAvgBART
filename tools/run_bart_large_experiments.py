#!/usr/bin/env python3
import argparse
import subprocess
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROOT / 'run_distilbart_experiment.py'
FIX_SCRIPT = ROOT / 'tools' / 'fix_metrics_csv.py'
RESULTS_BASE = ROOT / 'results_distilbart_cnndm_federated'


def run_once(config_path: str, num_clients: int, num_rounds: int | None, extra_args: list[str] | str | None = None):
    cmd = [
        sys.executable,
        str(RUN_SCRIPT),
        '--config', config_path,
        '--num_clients', str(num_clients),
    ]
    if num_rounds is not None:
        cmd += ['--num_rounds', str(num_rounds)]
    if extra_args:
        if isinstance(extra_args, list):
            cmd.extend(extra_args)
        else:
            cmd.extend(shlex.split(extra_args))
    print('Running:', ' '.join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def find_latest_run_dir_for_nc(num_clients: int) -> Path | None:
    nc_dir = RESULTS_BASE / f'nc_{num_clients}'
    if not nc_dir.exists():
        return None
    runs = sorted(nc_dir.glob('run_*'), key=lambda p: p.name)
    return runs[-1] if runs else None


def fix_metrics_for_last_run(num_clients: int):
    last = find_latest_run_dir_for_nc(num_clients)
    if not last:
        print(f'No run_* directory found to fix for nc_{num_clients}.')
        return
    cmd = [sys.executable, str(FIX_SCRIPT), str(last)]
    print('Fixing metrics for:', last)
    subprocess.check_call(cmd, cwd=ROOT)


def parse_args():
    p = argparse.ArgumentParser(description='Run DistilBART federated experiments across client counts')
    p.add_argument('--config', required=True, help='Path to YAML config')
    p.add_argument('--min-clients', type=int, required=True)
    p.add_argument('--max-clients', type=int, required=True)
    p.add_argument('--num-rounds', type=int, default=None, help='Override rounds')
    p.add_argument('--repeats', type=int, default=1, help='Runs per client count')
    p.add_argument('--extra', nargs=argparse.REMAINDER, default=[], help='Additional args passed to run_distilbart_experiment.py')
    return p.parse_args()


def main():
    args = parse_args()
    for nc in range(args.min_clients, args.max_clients + 1):
        for r in range(args.repeats):
            print(f'=== num_clients={nc} repeat={r+1}/{args.repeats} ===')
            run_once(args.config, nc, args.num_rounds, args.extra)
            # Post-run fix for metrics.csv alignment
            fix_metrics_for_last_run(nc)

    print('All experiments completed.')
    print('Consolidating to experiment_results/analysis/consolidated_metrics_distil_gen_fed.csv ...')
    consolidate = [
        sys.executable,
        str(ROOT / 'tools' / 'consolidate_distilbart_metrics.py'),
        str(RESULTS_BASE),
        str(ROOT / 'experiment_results' / 'analysis' / 'consolidated_metrics_distil_gen_fed.csv')
    ]
    subprocess.check_call(consolidate, cwd=ROOT)


if __name__ == '__main__':
    main()
