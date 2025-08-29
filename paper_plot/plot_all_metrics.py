#!/usr/bin/env python3
"""
Generate comprehensive classification and generation plots from analysis_results
and save them to paper_plot/plots/{classification,generation}.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path('/mnt/sda1/Projects/jsl/vp_gitlab/FED/FED-OPT-BERT/commit/FED-OPT-BERT-PYTORCH/paper_plot')
ANALYSIS = BASE / 'analysis_results'
PLOTS_CLS = BASE / 'plots' / 'classification'
PLOTS_GEN = BASE / 'plots' / 'generation'
PLOTS_CLS.mkdir(parents=True, exist_ok=True)
PLOTS_GEN.mkdir(parents=True, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.figsize': (10, 5),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

def savefig(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


# ------------------------ Classification ------------------------

def load_classification():
    data = {}
    cen = ANALYSIS / 'classification' / 'centralized' / 'training_metrics_combined.csv'
    fed = ANALYSIS / 'classification' / 'federated' / 'training_metrics_combined.csv'
    best = ANALYSIS / 'classification' / 'federated' / 'best_rounds.csv'
    data['centralized'] = pd.read_csv(cen) if cen.exists() else None
    data['federated'] = pd.read_csv(fed) if fed.exists() else None
    data['best'] = pd.read_csv(best) if best.exists() else None
    return data


def plot_cls_comparisons(data):
    cen, fed = data['centralized'], data['federated']
    if cen is None or fed is None:
        return
    # Build replicate observations so seaborn can compute 95% CI
    # Centralized: use per-row observations (already multiple rows); tag type
    cen_long = cen.copy()
    cen_long['type'] = 'Centralized'
    # Federated: average per run_id within model/alpha to form replicates
    fed_runs = (
        fed.groupby(['model', 'alpha', 'run_id'], as_index=False)[['accuracy', 'f1', 'precision', 'recall', 'loss']]
        .mean()
    )
    fed_long = fed_runs.copy()
    fed_long['type'] = 'Federated'
    both = pd.concat([cen_long, fed_long], ignore_index=True, sort=False)
    for metric in ['accuracy', 'f1', 'precision', 'recall', 'loss']:
        fig, ax = plt.subplots()
        sns.barplot(data=both, x='model', y=metric, hue='type', ci=95, ax=ax)
        ax.set_title(f'{metric.upper()} Centralized vs Federated (95% CI)')
        ax.set_ylabel(metric.upper())
        savefig(PLOTS_CLS / f'performance_comparison_{metric}.png')


def plot_cls_learning_curves(data):
    fed = data['federated']
    if fed is None:
        return
    # Accuracy over rounds per model/alpha
    for model in fed['model'].unique():
        for alpha in sorted(fed['alpha'].dropna().unique()):
            sub = fed[(fed['model'] == model) & (fed['alpha'] == alpha)]
            if sub.empty:
                continue
            fig, ax = plt.subplots()
            sns.lineplot(data=sub, x='round', y='accuracy', hue='num_clients', style='phase', ax=ax)
            ax.set_title(f'Learning Curve Accuracy - {model} (α={alpha})')
            ax.set_ylabel('Accuracy')
            savefig(PLOTS_CLS / f'learning_curve_{model}_alpha_{alpha}.png')
    # Client scaling: average accuracy by num_clients
    agg = fed.groupby(['model', 'alpha', 'num_clients'])['accuracy'].mean().reset_index()
    for model in agg['model'].unique():
        fig, ax = plt.subplots()
        sns.lineplot(data=agg[agg['model'] == model], x='num_clients', y='accuracy', hue='alpha', marker='o', ax=ax)
        ax.set_title(f'Client Scaling Analysis - {model}')
        ax.set_ylabel('Avg Accuracy')
        savefig(PLOTS_CLS / f'client_scaling_{model}.png')


def plot_cls_fed_vs_central_progress(data):
    """Create a side-by-side figure comparing centralized vs federated validation
    accuracy progress over epochs/rounds.
    Left: Centralized (validation accuracy vs epoch) per model.
    Right: Federated (validation accuracy vs round) per model averaged across clients and runs, with alpha shown in style.
    Saves to plots/classification/fed_vs_central_progress.png
    """
    cen = data.get('centralized')
    fed = data.get('federated')
    if cen is None or fed is None or cen.empty or fed.empty:
        return
    # Centralized: validation only, aggregate by model, epoch
    cen_val = cen[cen['phase'] == 'validation'].copy()
    cen_agg = cen_val.groupby(['model', 'epoch'], as_index=False)['accuracy'].mean()

    # Federated: validation only, aggregate by model, alpha, round across runs/clients
    fed_val = fed[fed['phase'] == 'validation'].copy()
    have_alpha = 'alpha' in fed_val.columns
    grp_cols = ['model', 'round'] + (['alpha'] if have_alpha else [])
    fed_agg = fed_val.groupby(grp_cols, as_index=False)['accuracy'].mean()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # Left: centralized
    ax = axes[0]
    sns.lineplot(data=cen_agg, x='epoch', y='accuracy', hue='model', marker='o', ax=ax)
    ax.set_title('Centralized: Validation Accuracy vs Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')

    # Right: federated
    ax = axes[1]
    if have_alpha:
        sns.lineplot(data=fed_agg, x='round', y='accuracy', hue='model', style='alpha', marker='o', ax=ax)
    else:
        sns.lineplot(data=fed_agg, x='round', y='accuracy', hue='model', marker='o', ax=ax)
    ax.set_title('Federated: Validation Accuracy vs Round')
    ax.set_xlabel('Communication Round')

    savefig(PLOTS_CLS / 'fed_vs_central_progress.png')


def plot_cls_best_rounds(data):
    best = data['best']
    if best is None:
        return
    # For each model/alpha, plot best accuracy and f1 vs num_clients
    for metric in ['accuracy', 'f1']:
        fig, ax = plt.subplots()
        sns.barplot(data=best, x='num_clients', y=metric, hue='model', ax=ax)
        ax.set_title(f'Best-Round {metric.upper()} by Clients')
        savefig(PLOTS_CLS / f'best_rounds_{metric}_by_clients.png')


# ------------------------ Generation ------------------------

def load_generation():
    data = {}
    # mean metrics per round (rouge and bleu)
    gen_mean_rouge = ANALYSIS / 'generation' / 'federated_mean_metrics_rouge.csv'
    gen_mean_bleu = ANALYSIS / 'generation' / 'federated_mean_metrics_bleu.csv'
    gen_mean_train = ANALYSIS / 'generation' / 'federated_mean_metrics_training.csv'
    gen_combined = ANALYSIS / 'generation' / 'combined_results.csv'
    gen_central = ANALYSIS / 'generation' / 'centralized_summary.csv'
    data['mean_rouge'] = pd.read_csv(gen_mean_rouge) if gen_mean_rouge.exists() else None
    data['mean_bleu'] = pd.read_csv(gen_mean_bleu) if gen_mean_bleu.exists() else None
    data['mean_train'] = pd.read_csv(gen_mean_train) if gen_mean_train.exists() else None
    data['combined'] = pd.read_csv(gen_combined) if gen_combined.exists() else None
    data['centralized'] = pd.read_csv(gen_central) if gen_central.exists() else None
    return data


def plot_gen_curves(data):
    rouge = data['mean_rouge']
    bleu = data['mean_bleu']
    if rouge is not None:
        for model in rouge['model'].unique():
            for alpha in sorted(rouge['alpha'].dropna().unique()):
                sub = rouge[(rouge['model'] == model) & (rouge['alpha'] == alpha)]
                if sub.empty:
                    continue
                # ROUGE-1/2/L
                fig, ax = plt.subplots()
                ax.plot(sub['round'], sub['rouge1'], label='ROUGE-1 F1')
                ax.plot(sub['round'], sub['rouge2'], label='ROUGE-2 F1')
                ax.plot(sub['round'], sub['rougeL'], label='ROUGE-L F1')
                ax.set_title(f'ROUGE vs Round - {model} (α={alpha})')
                ax.set_xlabel('Round')
                ax.set_ylabel('Score')
                ax.legend()
                savefig(PLOTS_GEN / f'rouge_vs_round_{model}_alpha_{alpha}.png')
    if bleu is not None:
        for model in bleu['model'].unique():
            for alpha in sorted(bleu['alpha'].dropna().unique()):
                sub = bleu[(bleu['model'] == model) & (bleu['alpha'] == alpha)]
                if sub.empty:
                    continue
                # BLEU-4
                fig, ax = plt.subplots()
                ax.plot(sub['round'], sub['bleu4'], label='BLEU-4')
                ax.set_title(f'BLEU-4 vs Round - {model} (α={alpha})')
                ax.set_xlabel('Round')
                ax.set_ylabel('Score')
                ax.legend()
                savefig(PLOTS_GEN / f'bleu4_vs_round_{model}_alpha_{alpha}.png')


def plot_gen_clients_vs_metrics(data):
    # Use mean_rouge (has num_clients) to aggregate best/peak per num_clients
    rouge = data['mean_rouge']
    if rouge is None:
        return
    # Peak metrics per model/alpha/num_clients
    idx = rouge.groupby(['model', 'alpha', 'num_clients'])['rouge1'].idxmax()
    peak = rouge.loc[idx]
    for metric in ['rouge1', 'bleu4']:
        fig, ax = plt.subplots()
        sns.lineplot(data=peak, x='num_clients', y=metric, hue='model', style='alpha', marker='o', ax=ax)
        ax.set_title(f'Peak {metric.upper()} vs Clients')
        savefig(PLOTS_GEN / f'clients_vs_{metric}.png')


def plot_gen_model_comparison(data):
    rouge = data['mean_rouge']
    if rouge is None:
        return
    # Peak per model across all alphas/clients
    idx = rouge.groupby(['model'])['rouge1'].idxmax()
    peak = rouge.loc[idx]
    fig, ax = plt.subplots()
    sns.barplot(data=peak, x='model', y='rouge1', ax=ax)
    ax.set_title('Model Comparison: Peak ROUGE-1 F1')
    savefig(PLOTS_GEN / 'model_comparison_rouge1_f1.png')

    # BLEU-4 peak using same file if available
    if 'bleu4' in rouge.columns:
        idx2 = rouge.groupby(['model'])['bleu4'].idxmax()
        peak2 = rouge.loc[idx2]
        fig, ax = plt.subplots()
        sns.barplot(data=peak2, x='model', y='bleu4', ax=ax)
        ax.set_title('Model Comparison: Peak BLEU-4')
        savefig(PLOTS_GEN / 'model_comparison_bleu4.png')


def plot_gen_centralized_vs_federated(data):
    """Bar comparisons for Centralized vs Federated on ROUGE-1 and BLEU-4 per model with 95% CI.
    We align scales by converting centralized ROUGE values (fractions) to percentages to match federated files.
    """
    cen = data.get('centralized')
    rouge = data.get('mean_rouge')
    if cen is None or rouge is None:
        return
    # Prepare centralized: convert to percentage scale to match federated mean files
    cen2 = cen.copy()
    # centralized_summary.csv columns: Model, rouge1, rouge2, rougeL, bleu, ... (bleu likely already in same scale)
    cen2.rename(columns={'Model': 'model'}, inplace=True)
    cen2['rouge1_pct'] = cen2['rouge1'] * 100.0
    # BLEU-4 field name in centralized is 'bleu'
    cen2['type'] = 'Centralized'

    # Prepare federated: use per (model, alpha, round) as replicates for CI
    fedr = rouge.copy()
    fedr['rouge1_pct'] = fedr['rouge1']  # already in percentage scale
    # Try to join BLEU-4 from same file if present, else skip BLEU-4 comparison
    has_bleu4 = 'bleu4' in fedr.columns
    fedr['type'] = 'Federated'

    # ROUGE-1 comparison
    r_cols = ['model', 'rouge1_pct', 'type']
    r_df = pd.concat([
        cen2[['model', 'rouge1_pct', 'type']],
        fedr[r_cols]
    ], ignore_index=True, sort=False)
    fig, ax = plt.subplots()
    sns.barplot(data=r_df, x='model', y='rouge1_pct', hue='type', ci=95, ax=ax)
    ax.set_title('ROUGE-1 F1 Centralized vs Federated (95% CI)')
    ax.set_ylabel('ROUGE-1 F1 (%)')
    savefig(PLOTS_GEN / 'performance_comparison_rouge1.png')

    # BLEU-4 comparison (if available)
    if has_bleu4:
        # Federated BLEU-4 is in the same file (already correct scale)
        f_bleu = fedr[['model', 'bleu4', 'type']].rename(columns={'bleu4': 'bleu4_score'})
        c_bleu = cen2[['model', 'bleu']].rename(columns={'bleu': 'bleu4_score'})
        c_bleu['type'] = 'Centralized'
        b_df = pd.concat([c_bleu[['model', 'bleu4_score', 'type']], f_bleu[['model', 'bleu4_score', 'type']]], ignore_index=True)
        fig, ax = plt.subplots()
        sns.barplot(data=b_df, x='model', y='bleu4_score', hue='type', ci=95, ax=ax)
        ax.set_title('BLEU-4 Centralized vs Federated (95% CI)')
        ax.set_ylabel('BLEU-4')
        savefig(PLOTS_GEN / 'performance_comparison_bleu4.png')


def main():
    # Classification
    cls = load_classification()
    plot_cls_comparisons(cls)
    plot_cls_learning_curves(cls)
    plot_cls_best_rounds(cls)
    plot_cls_fed_vs_central_progress(cls)

    # Generation
    gen = load_generation()
    plot_gen_curves(gen)
    plot_gen_clients_vs_metrics(gen)
    plot_gen_model_comparison(gen)
    plot_gen_centralized_vs_federated(gen)

    print(f'All plots saved to {PLOTS_CLS} and {PLOTS_GEN}')


if __name__ == '__main__':
    main()
