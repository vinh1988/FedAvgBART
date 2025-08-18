#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Ensure types
    df['num_clients'] = df['num_clients'].astype(int)
    df['round'] = df['round'].astype(int)
    if 'dirichlet_alpha' in df.columns:
        # allow None/NaN for IID
        df['dirichlet_alpha'] = pd.to_numeric(df['dirichlet_alpha'], errors='coerce')
    return df


def plot_rouge_across_clients_last_round(df: pd.DataFrame, out_dir: Path, suffix=""):
    # take last round per (num_clients, dirichlet_alpha)
    last = df.sort_values('round').groupby(['num_clients', 'dirichlet_alpha'] if 'dirichlet_alpha' in df.columns else ['num_clients']).tail(1)
    plt.figure(figsize=(8,5))
    hue = 'dirichlet_alpha' if 'dirichlet_alpha' in last.columns else None
    for col, label in [("rouge1_f1", "ROUGE-1"), ("rouge2_f1", "ROUGE-2"), ("rougeL_f1", "ROUGE-L")]:
        if hue:
            sns.lineplot(data=last, x='num_clients', y=col, marker='o', hue=hue)
        else:
            sns.lineplot(data=last, x='num_clients', y=col, marker='o', label=label)
    plt.xlabel('Number of Clients')
    plt.ylabel('ROUGE F1')
    plt.title('ROUGE vs Number of Clients (last round)' + (" by Dirichlet α" if hue else ""))
    fname = 'rouge_metrics{}.png'.format(suffix)
    plt.savefig(Path(out_dir) / fname, bbox_inches='tight')
    plt.close()


def plot_bleu_across_clients_last_round(df, out_dir, suffix=""):
    last = df.sort_values('round').groupby(['num_clients', 'dirichlet_alpha'] if 'dirichlet_alpha' in df.columns else ['num_clients']).tail(1)
    plt.figure(figsize=(8,5))
    hue = 'dirichlet_alpha' if 'dirichlet_alpha' in last.columns else None
    for col, label in [("bleu1", "BLEU-1"), ("bleu2", "BLEU-2"), ("bleu3", "BLEU-3"), ("bleu4", "BLEU-4")]:
        if col in last.columns:
            if hue:
                sns.lineplot(data=last, x='num_clients', y=col, marker='o', hue=hue)
            else:
                sns.lineplot(data=last, x='num_clients', y=col, marker='o', label=label)
    plt.xlabel('Number of Clients')
    plt.ylabel('BLEU')
    plt.title('BLEU vs Number of Clients (last round)' + (" by Dirichlet α" if hue else ""))
    fname = 'bleu_metrics{}.png'.format(suffix)
    plt.savefig(Path(out_dir) / fname, bbox_inches='tight')
    plt.close()


def plot_training_loss(df, out_dir, suffix=""):
    plt.figure(figsize=(8,5))
    hue = 'dirichlet_alpha' if 'dirichlet_alpha' in df.columns else None
    style = 'num_clients' if 'num_clients' in df.columns else None
    sns.lineplot(data=df, x='round', y='loss', hue=hue, style=style, markers=True)
    plt.xlabel('Round')
    plt.ylabel('Training Loss')
    plt.title('Training Loss over Rounds' + (" by Dirichlet α" if hue else ""))
    fname = 'training_loss{}.png'.format(suffix)
    plt.savefig(Path(out_dir) / fname, bbox_inches='tight')
    plt.close()


def plot_rouge1_convergence(df, out_dir, conv_nc=None, conv_alpha=None, suffix=""):
    if conv_nc is not None:
        df = df[df['num_clients'] == conv_nc]
    if conv_alpha is not None and 'dirichlet_alpha' in df.columns:
        df = df[df['dirichlet_alpha'] == conv_alpha]
    if df.empty:
        return
    plt.figure(figsize=(8,5))
    hue = 'dirichlet_alpha' if (conv_alpha is None and 'dirichlet_alpha' in df.columns) else None
    style = 'num_clients' if (conv_nc is None and 'num_clients' in df.columns) else None
    sns.lineplot(data=df, x='round', y='rouge1_f1', hue=hue, style=style, markers=True)
    title_parts = ["ROUGE-1 Convergence"]
    if conv_nc is not None:
        title_parts.append(f"nc={conv_nc}")
    if conv_alpha is not None:
        title_parts.append(f"α={conv_alpha}")
    plt.title(' '.join(title_parts))
    plt.xlabel('Round')
    plt.ylabel('ROUGE-1 F1')
    fname = 'rouge1_convergence{}.png'.format(suffix)
    plt.savefig(Path(out_dir) / fname, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser(description='Visualize consolidated metrics')
    ap.add_argument('--csv', default='experiment_results/analysis/consolidated_metrics_distil_gen_fed.csv')
    ap.add_argument('--out-dir', default='results_distilbart_cnndm_federated')
    ap.add_argument('--conv-nc', type=int, default=None, help='num_clients to use for convergence plot')
    ap.add_argument('--conv-alpha', type=float, default=None, help='Dirichlet alpha to filter for convergence plot (omit to overlay all)')
    ap.add_argument('--split-by-alpha', action='store_true', help='Generate separate plots per dirichlet_alpha value')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(Path(args.csv))

    # Overlay (default) plots
    plot_rouge_across_clients_last_round(df, out_dir)
    plot_bleu_across_clients_last_round(df, out_dir)
    plot_training_loss(df, out_dir)
    plot_rouge1_convergence(df, out_dir, conv_nc=args.conv_nc, conv_alpha=args.conv_alpha)

    # Per-alpha plots (BART-style per-config)
    if args.split_by_alpha and 'dirichlet_alpha' in df.columns:
        for a in sorted(df['dirichlet_alpha'].dropna().unique()):
            df_a = df[df['dirichlet_alpha'] == a].copy()
            # Remove hue for per-alpha plots to match per-config styling
            df_a = df_a.drop(columns=['dirichlet_alpha'])
            suf = f"_alpha_{a}"
            plot_rouge_across_clients_last_round(df_a, out_dir, suffix=suf)
            plot_bleu_across_clients_last_round(df_a, out_dir, suffix=suf)
            plot_training_loss(df_a, out_dir, suffix=suf)
            # Convergence for this alpha (respect conv-nc if provided)
            plot_rouge1_convergence(df, out_dir, conv_nc=args.conv_nc, conv_alpha=a, suffix=suf)

    print(f'Wrote plots to {out_dir}')


if __name__ == '__main__':
    main()
