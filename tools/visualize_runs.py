#!/usr/bin/env python3
import argparse
import os
import glob
import warnings
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def safe_read_csv(path: Path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        warnings.warn(f"Failed to read {path}: {e}")
        return None


essential_metric_cols = [
    'round', 'phase', 'loss', 'accuracy', 'precision', 'recall', 'f1'
]


def plot_metrics_over_rounds(df: pd.DataFrame, out_dir: Path, title_prefix: str = ""):
    if df is None or df.empty:
        return
    # Expect aggregated per-round validation rows or raw rows with phase == validation
    val = df.copy()
    if 'phase' in val.columns:
        val = val[val['phase'].str.contains('val', case=False, na=False)]
    # Try to ensure numeric types
    for c in ['round', 'loss', 'accuracy', 'precision', 'recall', 'f1']:
        if c in val.columns:
            val[c] = pd.to_numeric(val[c], errors='coerce')
    val = val.dropna(subset=['round'])
    if val.empty:
        return

    metrics = [c for c in ['accuracy', 'f1', 'precision', 'recall', 'loss'] if c in val.columns]
    if not metrics:
        return

    # Line plot over rounds for each metric
    for m in metrics:
        plt.figure(figsize=(7, 4))
        sns.lineplot(data=val.sort_values('round'), x='round', y=m, marker='o')
        plt.title(f"{title_prefix}Validation {m} over rounds")
        plt.tight_layout()
        out_path = out_dir / f"val_{m}_over_rounds.png"
        plt.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")
        plt.close()


def plot_client_contributions(df: pd.DataFrame, out_dir: Path):
    if df is None or df.empty:
        return
    # Expect columns like: client_id, split, count or similar
    # Try to infer
    cols = {c.lower(): c for c in df.columns}
    client_col = cols.get('client_id') or cols.get('client') or list(df.columns)[0]
    count_col = cols.get('count') or cols.get('size') or None
    if count_col is None:
        # try to compute from any numeric columns
        numeric_cols = df.select_dtypes(include='number').columns
        if len(numeric_cols) >= 1:
            count_col = numeric_cols[0]
        else:
            return

    plt.figure(figsize=(8, 4))
    order = df.groupby(client_col)[count_col].sum().sort_values(ascending=False).index
    sns.barplot(data=df, x=client_col, y=count_col, order=order, color="#4C72B0")
    plt.title("Samples per client")
    plt.tight_layout()
    out_path = out_dir / "samples_per_client.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


def plot_label_distribution_heatmap(df: pd.DataFrame, out_dir: Path, title: str):
    if df is None or df.empty:
        return
    # Expect wide-format: client rows x class columns or long with client_id, label, count
    # Try to detect long format
    cols = {c.lower(): c for c in df.columns}
    if {'client_id', 'label', 'count'}.issubset(set(cols.keys())):
        client = cols['client_id']
        label = cols['label']
        count = cols['count']
        pivot = df.pivot_table(index=client, columns=label, values=count, aggfunc='sum', fill_value=0)
    else:
        # if first column is client and others are labels
        pivot = df.set_index(df.columns[0])
        # keep only numeric
        pivot = pivot.select_dtypes(include='number')

    if pivot.empty:
        return

    plt.figure(figsize=(10, max(4, pivot.shape[0] * 0.4)))
    sns.heatmap(pivot, annot=False, cmap="Blues")
    plt.title(title)
    plt.xlabel("Label")
    plt.ylabel("Client")
    plt.tight_layout()
    fname = "label_distribution_train_heatmap.png" if "train" in title.lower() else "label_distribution_test_heatmap.png"
    out_path = out_dir / fname
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


def plot_participation(df: pd.DataFrame, out_dir: Path):
    if df is None or df.empty:
        return
    # Expect columns: round, client_id, selected (1/0) or similar
    cols = {c.lower(): c for c in df.columns}
    round_col = cols.get('round')
    client_col = cols.get('client_id') or cols.get('client')
    selected_col = cols.get('selected') or cols.get('participated') or None

    if not (round_col and client_col):
        return

    if selected_col is None:
        # Assume presence indicates participation
        df['__selected__'] = 1
        selected_col = '__selected__'
    
    # Pivot to rounds x clients (0/1)
    table = df.pivot_table(index=round_col, columns=client_col, values=selected_col, fill_value=0, aggfunc='max')
    plt.figure(figsize=(10, max(3, table.shape[1] * 0.3)))
    sns.heatmap(table.T, cmap='Greens', cbar=False)
    plt.title('Client Participation by Round')
    plt.xlabel('Round')
    plt.ylabel('Client')
    plt.tight_layout()
    out_path = out_dir / 'participation_heatmap.png'
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


def find_first(path: Path, patterns):
    if isinstance(patterns, str):
        patterns = [patterns]
    for pat in patterns:
        found = list(path.glob(pat))
        if found:
            return found[0]
    return None


def process_run_dir(run_dir: Path):
    charts_dir = run_dir / 'charts'
    charts_dir.mkdir(parents=True, exist_ok=True)

    # Try to find metrics csv
    metrics_csv = find_first(run_dir, ["training_metrics_*.csv", "metrics*.csv"]) 
    metrics_df = safe_read_csv(metrics_csv) if metrics_csv else None

    # Contributions and label distributions
    contrib_csv = find_first(run_dir, ["data_contribution.csv", "client_sizes.csv"]) 
    contrib_df = safe_read_csv(contrib_csv) if contrib_csv else None

    label_train_csv = find_first(run_dir, ["label_distribution_train.csv", "label_dist_train.csv"]) 
    label_test_csv = find_first(run_dir, ["label_distribution_test.csv", "label_dist_test.csv"]) 
    label_train_df = safe_read_csv(label_train_csv) if label_train_csv else None
    label_test_df = safe_read_csv(label_test_csv) if label_test_csv else None

    participation_csv = find_first(run_dir, ["participation.csv"]) 
    participation_df = safe_read_csv(participation_csv) if participation_csv else None

    # Plotters
    plot_metrics_over_rounds(metrics_df, charts_dir)
    plot_client_contributions(contrib_df, charts_dir)
    plot_label_distribution_heatmap(label_train_df, charts_dir, "Train label distribution")
    plot_label_distribution_heatmap(label_test_df, charts_dir, "Test label distribution")
    plot_participation(participation_df, charts_dir)


def main():
    parser = argparse.ArgumentParser(description="Visualize federated run outputs")
    parser.add_argument("--base-dir", type=str, required=True,
                        help="Base directory that contains clients_X_rounds_Y/<timestamp>/ run folders")
    args = parser.parse_args()

    base = Path(args.base_dir)
    if not base.exists():
        print(f"Base directory not found: {base}")
        return 1

    # Expect structure: base/clients_*/TIMESTAMP/
    client_groups = sorted([p for p in base.iterdir() if p.is_dir()])
    if not client_groups:
        print(f"No subdirectories found in {base}")
        return 0

    run_count = 0
    for group in client_groups:
        timestamps = sorted([p for p in group.iterdir() if p.is_dir()])
        if not timestamps:
            continue
        for run_dir in timestamps:
            process_run_dir(run_dir)
            run_count += 1

    print(f"Visualization complete. Processed {run_count} run(s).")


if __name__ == "__main__":
    raise SystemExit(main())
