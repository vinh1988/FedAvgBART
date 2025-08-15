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


def plot_rouge_across_clients_last_round(df: pd.DataFrame, out_dir: Path):
    # take last round per (num_clients, dirichlet_alpha)
    group_cols = ['num_clients'] + (['dirichlet_alpha'] if 'dirichlet_alpha' in df.columns else [])
    last = df.sort_values(group_cols + ['round']).groupby(group_cols).tail(1)
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
    plt.tight_layout()
    plt.savefig(out_dir / 'rouge_metrics.png', dpi=200)
    plt.close()


def plot_bleu_across_clients_last_round(df: pd.DataFrame, out_dir: Path):
    group_cols = ['num_clients'] + (['dirichlet_alpha'] if 'dirichlet_alpha' in df.columns else [])
    last = df.sort_values(group_cols + ['round']).groupby(group_cols).tail(1)
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
    plt.tight_layout()
    plt.savefig(out_dir / 'bleu_metrics.png', dpi=200)
    plt.close()


def plot_training_loss_over_rounds(df: pd.DataFrame, out_dir: Path):
    plt.figure(figsize=(9,5))
    if 'dirichlet_alpha' in df.columns:
        sns.lineplot(data=df, x='round', y='loss', hue='dirichlet_alpha', style='num_clients', marker='o')
    else:
        sns.lineplot(data=df, x='round', y='loss', hue='num_clients', marker='o')
    plt.xlabel('Round')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss over Rounds' + (" by Dirichlet α and #Clients" if 'dirichlet_alpha' in df.columns else " by Number of Clients"))
    plt.legend(title=('Dirichlet α' if 'dirichlet_alpha' in df.columns else 'num_clients'))
    plt.tight_layout()
    plt.savefig(out_dir / 'training_loss.png', dpi=200)
    plt.close()


def plot_rouge1_convergence(df: pd.DataFrame, out_dir: Path, num_clients: int | None, alpha: float | None):
    if num_clients is None:
        num_clients = sorted(df['num_clients'].unique())[-1]
    sub = df[df['num_clients'] == num_clients].sort_values('round')
    plt.figure(figsize=(8,5))
    if 'dirichlet_alpha' in sub.columns and alpha is None:
        sns.lineplot(data=sub, x='round', y='rouge1_f1', hue='dirichlet_alpha', marker='o')
    elif 'dirichlet_alpha' in sub.columns and alpha is not None:
        sub = sub[sub['dirichlet_alpha'] == alpha]
        sns.lineplot(data=sub, x='round', y='rouge1_f1', marker='o')
    else:
        sns.lineplot(data=sub, x='round', y='rouge1_f1', marker='o')
    plt.xlabel('Round')
    plt.ylabel('ROUGE-1 F1')
    suffix = f" (num_clients={num_clients}" + (f", alpha={alpha}" if alpha is not None else ")")
    plt.title('ROUGE-1 Convergence' + suffix)
    plt.tight_layout()
    plt.savefig(out_dir / 'rouge1_convergence.png', dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description='Visualize consolidated metrics')
    ap.add_argument('--csv', default='results_distilbart_cnndm_federated/consolidated_metrics.csv')
    ap.add_argument('--out-dir', default='results_distilbart_cnndm_federated')
    ap.add_argument('--conv-nc', type=int, default=None, help='num_clients to use for convergence plot')
    ap.add_argument('--conv-alpha', type=float, default=None, help='Dirichlet alpha to filter for convergence plot (omit to overlay all)')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(Path(args.csv))

    plot_rouge_across_clients_last_round(df, out_dir)
    plot_bleu_across_clients_last_round(df, out_dir)
    plot_training_loss_over_rounds(df, out_dir)
    plot_rouge1_convergence(df, out_dir, args.conv_nc, args.conv_alpha)

    print(f'Wrote plots to {out_dir}')


if __name__ == '__main__':
    main()
