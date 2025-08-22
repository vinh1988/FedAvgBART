#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_df(path: Path, model_name: str) -> pd.DataFrame:
    # Try to auto-detect delimiter (handles comma and tab)
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, sep='\t')
    # If columns look collapsed (single column with tabs), force tab sep
    if df.shape[1] == 1 and '\t' in df.columns[0]:
        df = pd.read_csv(path, sep='\t')
    # If header was read but expected columns missing, try python engine sniffing
    if 'num_clients' not in df.columns and 'round' not in df.columns:
        try:
            df = pd.read_csv(path, sep=None, engine='python')
        except Exception:
            df = pd.read_csv(path, sep='\t')
    # normalize columns
    if 'num_clients' not in df.columns:
        raise ValueError(f"Missing num_clients in {path}")
    if 'round' not in df.columns:
        raise ValueError(f"Missing round in {path}")
    # Ensure numeric types
    df['num_clients'] = pd.to_numeric(df['num_clients'], errors='coerce').astype(int)
    df['round'] = pd.to_numeric(df['round'], errors='coerce').astype(int)
    # Add model column
    df['model'] = model_name
    # Ensure expected metric columns exist (fill if missing)
    for c in ['rouge1_f1','rouge2_f1','rougeL_f1','bleu1','bleu2','bleu3','bleu4','loss']:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def plot_last_round_vs_clients(df: pd.DataFrame, out_dir: Path):
    last = df.sort_values('round').groupby(['model','num_clients']).tail(1)
    # ROUGE
    plt.figure(figsize=(8,5))
    for col, label in [("rouge1_f1","ROUGE-1"),("rouge2_f1","ROUGE-2"),("rougeL_f1","ROUGE-L")]:
        sns.lineplot(data=last, x='num_clients', y=col, hue='model', marker='o')
    plt.xlabel('Number of Clients')
    plt.ylabel('ROUGE F1')
    plt.title('ROUGE vs Number of Clients (last round): DistilBART vs BART-Large')
    plt.savefig(out_dir / 'rouge_compare.png', bbox_inches='tight')
    plt.close()
    # BLEU
    plt.figure(figsize=(8,5))
    for col, label in [("bleu1","BLEU-1"),("bleu2","BLEU-2"),("bleu3","BLEU-3"),("bleu4","BLEU-4")]:
        if col in last.columns:
            sns.lineplot(data=last, x='num_clients', y=col, hue='model', marker='o')
    plt.xlabel('Number of Clients')
    plt.ylabel('BLEU')
    plt.title('BLEU vs Number of Clients (last round): DistilBART vs BART-Large')
    plt.savefig(out_dir / 'bleu_compare.png', bbox_inches='tight')
    plt.close()


def plot_convergence_nc(df: pd.DataFrame, out_dir: Path, nc: int = 10):
    sub = df[df['num_clients'] == nc].copy()
    if sub.empty:
        return
    plt.figure(figsize=(8,5))
    sns.lineplot(data=sub, x='round', y='rouge1_f1', hue='model', marker='o')
    plt.xlabel('Round')
    plt.ylabel('ROUGE-1 F1')
    plt.title(f'ROUGE-1 Convergence (nc={nc}): DistilBART vs BART-Large')
    plt.savefig(out_dir / f'rouge1_convergence_compare_nc{nc}.png', bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser(description='Compare DistilBART and BART-Large federated results')
    ap.add_argument('--distil-csv', required=True, help='Path to DistilBART consolidated CSV')
    ap.add_argument('--bart-csv', required=True, help='Path to BART-Large consolidated CSV')
    ap.add_argument('--out-dir', default='experiment_results/analysis/compare_bart_distil', help='Output directory for comparison plots and CSV')
    ap.add_argument('--conv-nc', type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    distil_df = load_df(Path(args.distil_csv), 'DistilBART')
    bart_df = load_df(Path(args.bart_csv), 'BART-Large')
    merged = pd.concat([distil_df, bart_df], ignore_index=True)

    # Write merged CSV for record
    merged_csv = out_dir / 'consolidated_metrics_compare_bart_vs_distil.csv'
    merged.to_csv(merged_csv, index=False)

    # Plots
    plot_last_round_vs_clients(merged, out_dir)
    plot_convergence_nc(merged, out_dir, nc=args.conv_nc)

    print(f'Wrote comparison CSV to {merged_csv}')
    print(f'Wrote comparison plots to {out_dir}')


if __name__ == '__main__':
    main()
