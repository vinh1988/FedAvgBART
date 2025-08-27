#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path
import os


def parse_args():
    p = argparse.ArgumentParser(
        description="Run multiple 20news federated runs by sweeping num_clients"
    )
    p.add_argument("--min-clients", type=int, required=True, help="Minimum number of clients (inclusive)")
    p.add_argument("--max-clients", type=int, required=True, help="Maximum number of clients (inclusive)")
    p.add_argument("--num-rounds", type=int, required=True, help="Number of federated rounds per run")
    # Forwardable optional controls (train script supports these now)
    p.add_argument("--participation-rate", dest="participation_rate", type=float, default=None,
                   help="Fraction of clients per round (0 < r <= 1), forwarded to train script")
    p.add_argument("--min-clients-per-round", dest="min_clients_per_round", type=int, default=None,
                   help="Minimum clients per round, forwarded to train script")
    p.add_argument("--dirichlet-alpha", dest="dirichlet_alpha", type=float, default=None,
                   help="Dirichlet concentration for non-IID split; None means IID (forwarded)")
    p.add_argument("--dirichlet-min-size", dest="dirichlet_min_size", type=int, default=None,
                   help="Minimum samples per client for Dirichlet split (forwarded)")
    # Additional placeholders (not forwarded currently)
    p.add_argument("--local-epochs", type=int, default=1, help="(unused) Local epochs per client per round")
    p.add_argument("--batch-size", type=int, default=8, help="(unused) Batch size for training")
    p.add_argument("--seed", type=int, default=42, help="(unused) Random seed")
    p.add_argument("--wandb-mode", type=str, default=None, choices=["offline", "online"], help="Set WANDB_MODE for runs")
    p.add_argument("--python", type=str, default=sys.executable, help="Python interpreter to use (defaults to current)")
    p.add_argument(
        "--train-script",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "train_distilbart_20news.py"),
        help="Path to train_distilbart_20news.py",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Base directory to store run artifacts; forwarded to train script as --output_dir",
    )
    return p.parse_args()


def main():
    args = parse_args()
    assert args.min_clients >= 1 and args.max_clients >= args.min_clients

    train_script = Path(args.train_script).resolve()
    if not train_script.exists():
        raise FileNotFoundError(f"Train script not found: {train_script}")

    env = os.environ.copy()
    if args.wandb_mode:
        env["WANDB_MODE"] = args.wandb_mode

    for n in range(args.min_clients, args.max_clients + 1):
        # Pass only the arguments supported by train_distilbart_20news.py
        cmd = [
            args.python,
            str(train_script),
            "--num_clients", str(n),
            "--num_rounds", str(args.num_rounds),
        ]
        if args.participation_rate is not None:
            cmd += ["--participation_rate", str(args.participation_rate)]
        if args.min_clients_per_round is not None:
            cmd += ["--min_clients_per_round", str(args.min_clients_per_round)]
        if args.dirichlet_alpha is not None:
            cmd += ["--dirichlet_alpha", str(args.dirichlet_alpha)]
        if args.dirichlet_min_size is not None:
            cmd += ["--dirichlet_min_size", str(args.dirichlet_min_size)]
        if args.output_dir is not None:
            cmd += ["--output_dir", str(args.output_dir)]

        print("==================================================")
        print(f"Running: num_clients={n}, num_rounds={args.num_rounds}")
        print("Command:", " ".join(cmd))
        print("==================================================")
        proc = subprocess.run(cmd, env=env)
        if proc.returncode != 0:
            print(f"Run failed for num_clients={n} (exit {proc.returncode}). Aborting sweep.")
            sys.exit(proc.returncode)

    print("All runs completed successfully.")


if __name__ == "__main__":
    main()
