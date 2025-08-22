#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")


def load_client_metrics(base_dir: Path) -> pd.DataFrame:
    rows = []
    # Expect structure: base_dir/nc_{N}/run_*/client_contributions/client_metrics.csv
    for nc_dir in sorted(base_dir.glob('nc_*')):
        try:
            nc = int(str(nc_dir.name).split('_', 1)[1])
        except Exception:
            continue
        # Pick latest run for that nc
        run_dirs = sorted([p for p in nc_dir.glob('run_*') if p.is_dir()], key=lambda p: p.name)
        if not run_dirs:
            continue
        run_dir = run_dirs[-1]
        cm_path = run_dir / 'client_contributions' / 'client_metrics.csv'
        if not cm_path.exists():
            continue
        df = pd.read_csv(cm_path)
        # Expect columns like: client_id, contribution, num_samples (adjust as available)
        # Normalize to percentage contribution per nc
        if 'contribution' in df.columns:
            total = df['contribution'].sum()
            if total > 0:
                df['percentage'] = df['contribution'] / total
            else:
                df['percentage'] = 0.0
        elif 'num_samples' in df.columns:
            total = df['num_samples'].sum()
            if total > 0:
                df['percentage'] = df['num_samples'] / total
            else:
                df['percentage'] = 0.0
        else:
            # fallback: equal contribution
            df['percentage'] = 1.0 / len(df)
        # Ensure client_id exists
        if 'client_id' not in df.columns:
            df = df.reset_index().rename(columns={'index': 'client_id'})
        df['num_clients'] = nc
        rows.append(df[['num_clients', 'client_id', 'percentage']])
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame(columns=['num_clients','client_id','percentage'])


def plot_participation_heatmap(df: pd.DataFrame, out_dir: Path):
    if df.empty:
        print('No client metrics found to plot heatmap.')
        return
    # Pivot to heatmap: rows clients (0..9), columns num_clients (2..10)
    pivot = df.pivot_table(index='client_id', columns='num_clients', values='percentage', aggfunc='mean')
    # Sort axes
    pivot = pivot.reindex(sorted(pivot.index))
    pivot = pivot[sorted(pivot.columns)]
    plt.figure(figsize=(10,6))
    sns.heatmap(pivot, annot=False, cmap='viridis', cbar_kws={'label': 'Participation (%)'}, vmin=0, vmax=1)
    plt.xlabel('Number of Clients (2-10)')
    plt.ylabel('Client Index')
    plt.title('Client Participation Heatmap (percentage across client configurations)')
    plt.tight_layout()
    plt.savefig(out_dir / 'participation_heatmap.png', dpi=200)
    plt.close()


def plot_contribution_distribution(df: pd.DataFrame, out_dir: Path):
    if df.empty:
        print('No client metrics found to plot distribution.')
        return
    # Use boxplot + jittered points on a 0-100% scale for readability
    df_plot = df.copy()
    df_plot['percentage_pct'] = df_plot['percentage'] * 100.0
    plt.figure(figsize=(10,5))
    sns.boxplot(data=df_plot, x='num_clients', y='percentage_pct', showfliers=True, width=0.6)
    sns.stripplot(data=df_plot, x='num_clients', y='percentage_pct', color='black', size=3, alpha=0.5, jitter=0.25)
    plt.xlabel('Number of Clients')
    plt.ylabel('Participation (%)')
    plt.title('Distribution of Client Participation by Client Count')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(out_dir / 'contribution_distribution.png', dpi=200)
    plt.close()


def plot_gini_vs_clients(df: pd.DataFrame, out_dir: Path):
    if df.empty:
        print('No client metrics found to plot Gini.')
        return
    # Compute Gini per num_clients
    def gini(x: np.ndarray) -> float:
        if len(x) == 0:
            return 0.0
        x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(x)
        if cumx[-1] == 0:
            return 0.0
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    stats = df.groupby('num_clients')['percentage'].apply(lambda s: gini(s.values)).reset_index(name='gini')
    plt.figure(figsize=(8,5))
    sns.lineplot(data=stats, x='num_clients', y='gini', marker='o')
    plt.xlabel('Number of Clients')
    plt.ylabel('Gini Coefficient (Participation)')
    plt.title('Inequality of Participation vs Number of Clients')
    plt.tight_layout()
    plt.savefig(out_dir / 'gini_vs_clients.png', dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description='Visualize data distribution across client configurations')
    ap.add_argument('--base', default='results_distilbart_cnndm_federated', help='Base results dir with nc_*/run_*')
    ap.add_argument('--out-dir', default='results_distilbart_cnndm_federated', help='Output directory')
    args = ap.parse_args()

    base = Path(args.base)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_client_metrics(base)
    if df.empty:
        print('No client metrics loaded. Ensure client_contributions/client_metrics.csv exists for each nc_*/run_*')
    else:
        # Convert to percentage (0-1) already; for display we may keep 0-1 scale
        plot_participation_heatmap(df, out_dir)
        plot_contribution_distribution(df, out_dir)
        plot_gini_vs_clients(df, out_dir)
        print(f'Wrote data distribution plots to {out_dir}')


if __name__ == '__main__':
    main()
