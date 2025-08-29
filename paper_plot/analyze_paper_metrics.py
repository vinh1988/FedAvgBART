#!/usr/bin/env python3
"""
Comprehensive analysis of classification and generation metrics for both
centralized and federated training setups.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import json
from collections import defaultdict

# Set up paths
base_dir = Path('/mnt/sda1/Projects/jsl/vp_gitlab/FED/FED-OPT-BERT/commit/FED-OPT-BERT-PYTORCH/paper_plot')
output_dir = base_dir / 'analysis_results/paper_metrics'
output_dir.mkdir(parents=True, exist_ok=True)

# Set plotting style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

class PaperMetricsAnalyzer:
    def __init__(self):
        self.metrics = {}
        self.load_data()
        
    def load_data(self):
        """Load all metrics data."""
        # Load centralized metrics
        centralized_path = base_dir / 'analysis_results/classification/centralized/training_metrics_combined.csv'
        if centralized_path.exists():
            self.metrics['centralized'] = pd.read_csv(centralized_path)
            print(f"Loaded centralized metrics with {len(self.metrics['centralized'])} records")
        
        # Load federated metrics
        fed_path = base_dir / 'analysis_results/classification/federated/training_metrics_combined.csv'
        if fed_path.exists():
            self.metrics['federated'] = pd.read_csv(fed_path)
            print(f"Loaded federated metrics with {len(self.metrics['federated'])} records")

        # Load generation federated metrics (ROUGE/BLEU)
        gen_fed_path = base_dir / 'analysis_results/generation/federated_metrics_rouge.csv'
        if gen_fed_path.exists():
            try:
                self.metrics['generation_federated'] = pd.read_csv(gen_fed_path)
                print(
                    f"Loaded generation federated metrics with {len(self.metrics['generation_federated'])} rows "
                    f"from {gen_fed_path}"
                )
            except Exception as e:
                print(f"Failed to read generation metrics at {gen_fed_path}: {e}")

        # Load best rounds
        best_rounds_path = base_dir / 'analysis_results/classification/federated/best_rounds.csv'
        if best_rounds_path.exists():
            self.best_rounds = pd.read_csv(best_rounds_path)
            print(f"Loaded best rounds with {len(self.best_rounds)} records")
    
    def analyze_classification_metrics(self):
        """Analyze classification metrics across all setups."""
        results = {}
        
        # Analyze centralized metrics
        if 'centralized' in self.metrics:
            df = self.metrics['centralized']
            results['centralized'] = {
                'models': df['model'].unique().tolist(),
                'metrics': df.groupby('model').agg({
                    'accuracy': ['mean', 'std'],
                    'f1': ['mean', 'std'],
                    'precision': ['mean', 'std'],
                    'recall': ['mean', 'std'],
                    'loss': ['mean', 'std']
                }).round(4).to_dict()
            }
        
        # Analyze federated metrics
        if 'federated' in self.metrics:
            df = self.metrics['federated']
            
            # Overall federated metrics
            results['federated'] = {
                'models': df['model'].unique().tolist(),
                'num_clients': sorted(df['num_clients'].unique().tolist()),
                'alphas': sorted(df['alpha'].unique().tolist()),
                'metrics_by_alpha': {}
            }
            
            # Metrics by alpha value
            for alpha in results['federated']['alphas']:
                alpha_df = df[df['alpha'] == alpha]
                results['federated']['metrics_by_alpha'][alpha] = {
                    'overall': alpha_df.groupby('model').agg({
                        'accuracy': ['mean', 'std'],
                        'f1': ['mean', 'std'],
                        'precision': ['mean', 'std'],
                        'recall': ['mean', 'std'],
                        'loss': ['mean', 'std']
                    }).round(4).to_dict(),
                    'by_clients': {}
                }
                
                # Metrics by number of clients
                for n_clients in results['federated']['num_clients']:
                    client_df = alpha_df[alpha_df['num_clients'] == n_clients]
                    if not client_df.empty:
                        results['federated']['metrics_by_alpha'][alpha]['by_clients'][n_clients] = \
                            client_df.groupby('model').agg({
                                'accuracy': ['mean', 'std'],
                                'f1': ['mean', 'std'],
                                'precision': ['mean', 'std'],
                                'recall': ['mean', 'std'],
                                'loss': ['mean', 'std']
                            }).round(4).to_dict()
        
        return results
    
    def plot_metric_comparison(self, metric='accuracy'):
        """Plot comparison of metrics between centralized and federated."""
        if 'federated' not in self.metrics or 'centralized' not in self.metrics:
            print("Missing data for comparison")
            return
            
        plt.figure(figsize=(14, 7))
        
        # Prepare data
        central = self.metrics['centralized'].groupby('model')[metric].mean().reset_index()
        central['type'] = 'Centralized'
        
        fed = self.metrics['federated'].groupby(['model', 'alpha'])[metric].mean().reset_index()
        fed = fed.rename(columns={'alpha': 'Alpha'})
        fed['type'] = 'Federated'
        
        # Plot
        ax = sns.barplot(
            x='model', 
            y=metric, 
            hue='type',
            data=pd.concat([
                central[['model', metric, 'type']],
                fed[['model', metric, 'type']]
            ])
        )
        
        # Add alpha information for federated
        for i, (_, row) in enumerate(fed.iterrows()):
            ax.text(i, row[metric], f"α={row['Alpha']}", 
                   ha='center', va='bottom', fontsize=9)
        
        plt.title(f'Comparison of {metric.upper()} between Centralized and Federated Training')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save
        plt.savefig(output_dir / f'comparison_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_learning_curves(self):
        """Plot learning curves for different configurations."""
        if 'federated' not in self.metrics:
            return
            
        df = self.metrics['federated']
        
        # Plot for each model and alpha
        for model in df['model'].unique():
            for alpha in df['alpha'].unique():
                plt.figure(figsize=(12, 6))
                
                # Filter data
                model_df = df[(df['model'] == model) & (df['alpha'] == alpha)]
                
                # Plot learning curves
                sns.lineplot(
                    data=model_df,
                    x='round',
                    y='accuracy',
                    hue='num_clients',
                    style='phase',
                    markers=True,
                    dashes=False
                )
                
                plt.title(f'Learning Curves - {model} (α={alpha})')
                plt.xlabel('Round')
                plt.ylabel('Accuracy')
                plt.legend(title='# Clients')
                plt.tight_layout()
                
                # Save
                plt.savefig(
                    output_dir / f'learning_curve_{model}_alpha{alpha}.png', 
                    dpi=300, 
                    bbox_inches='tight'
                )
                plt.close()
    
    def generate_latex_tables(self, results):
        """Generate LaTeX tables for the paper."""
        if 'centralized' in results:
            # Centralized results table
            df = pd.DataFrame(results['centralized']['metrics'])
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
            
            # Generate LaTeX
            latex = "\n\\begin{table*}[t]\n\\centering\n"
            latex += f"\\caption{{Centralized Training Results}}\\label{{tab:centralized_results}}\n"
            latex += df.to_latex(
                float_format="%.4f",
                bold_rows=True,
                caption="Centralized Training Results",
                label="tab:centralized_results"
            )
            latex += "\n\\end{table*}"
            
            with open(output_dir / 'centralized_results.tex', 'w') as f:
                f.write(latex)
        
        # Federated results table
        if 'federated' in results:
            # Create a summary table for federated results
            rows = []
            for alpha, alpha_data in results['federated']['metrics_by_alpha'].items():
                for n_clients, client_data in alpha_data['by_clients'].items():
                    for model in results['federated']['models']:
                        # Safely access metrics with proper error handling
                        try:
                            acc = client_data.get('accuracy', {}).get(model, {})
                            f1 = client_data.get('f1', {}).get(model, {})
                            prec = client_data.get('precision', {}).get(model, {})
                            rec = client_data.get('recall', {}).get(model, {})
                            
                            if acc and 'mean' in acc and 'std' in acc:
                                rows.append({
                                    'Model': model,
                                    'Alpha': alpha,
                                    'Clients': n_clients,
                                    'Accuracy': f"{acc['mean']:.4f} ± {acc['std']:.4f}",
                                    'F1': f"{f1.get('mean', 0):.4f} ± {f1.get('std', 0):.4f}",
                                    'Precision': f"{prec.get('mean', 0):.4f} ± {prec.get('std', 0):.4f}",
                                    'Recall': f"{rec.get('mean', 0):.4f} ± {rec.get('std', 0):.4f}"
                                })
                        except Exception as e:
                            print(f"Error processing model {model}, alpha {alpha}, clients {n_clients}: {e}")
            
            if rows:
                df = pd.DataFrame(rows)
                latex = "\n\\begin{table*}[t]\n\\centering\n"
                latex += f"\\caption{{Federated Training Results}}\\label{{tab:federated_results}}\n"
                latex += df.to_latex(
                    index=False,
                    float_format="%.4f",
                    bold_rows=True,
                    caption="Federated Training Results",
                    label="tab:federated_results"
                )
                latex += "\n\\end{table*}"
                
                with open(output_dir / 'federated_results.tex', 'w') as f:
                    f.write(latex)
    
    def run_analysis(self):
        """Run complete analysis pipeline."""
        print("Starting comprehensive analysis...")
        
        # Analyze metrics
        results = self.analyze_classification_metrics()
        
        # Generate plots
        for metric in ['accuracy', 'f1', 'precision', 'recall', 'loss']:
            self.plot_metric_comparison(metric)
        
        # Generate learning curves
        self.plot_learning_curves()
        
        # Generate LaTeX tables
        self.generate_latex_tables(results)
        
        # Generation plots (ROUGE-1 vs #clients with CIs)
        self.plot_generation_rouge1_vs_clients()
        # Generation plots (BLEU-4 vs #clients with CIs)
        self.plot_generation_bleu4_vs_clients()
        # Classification plots (F1 vs #clients with CIs)
        self.plot_classification_f1_vs_clients()
        
        # Save results to JSON with enhanced serialization
        def clean_json(obj):
            """Recursively clean objects for JSON serialization"""
            if obj is None or isinstance(obj, (str, int, float, bool)):
                return obj
            elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, (list, tuple)):
                return [clean_json(x) for x in obj]
            elif isinstance(obj, dict):
                return {str(k): clean_json(v) for k, v in obj.items()}
            elif hasattr(obj, '__dict__'):
                return {k: clean_json(v) for k, v in obj.__dict__.items()}
            else:
                return str(obj)
        
        # Clean the results before serialization
        clean_results = clean_json(results)
        
        # Write to file with proper encoding
        with open(output_dir / 'analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, ensure_ascii=False, indent=2)
        
        print(f"Analysis complete. Results saved to {output_dir}")
        return results

    def plot_generation_rouge1_vs_clients(self):
        """Plot ROUGE-1 (F1) vs number of clients with lines, markers, and 95% CI bands.

        Uses grouped statistics over rounds for each (model, alpha, num_clients).
        Saves per-alpha plots and a combined plot for LaTeX inclusion.
        """
        if 'generation_federated' not in self.metrics:
            print("Generation federated metrics not found; skipping ROUGE-1 vs clients plot.")
            return

        df = self.metrics['generation_federated'].copy()

        # Ensure required columns exist
        required_cols = {'model', 'alpha', 'num_clients', 'rouge1', 'round'}
        if not required_cols.issubset(df.columns):
            print(f"Missing columns for generation plot: {required_cols - set(df.columns)}")
            return

        # Clean and aggregate: mean and 95% CI over rounds
        def agg_ci(g):
            m = g['rouge1'].mean()
            s = g['rouge1'].std(ddof=1)
            n = g['rouge1'].count()
            sem = s / np.sqrt(n) if n > 0 and not np.isnan(s) else 0.0
            ci95 = 1.96 * sem
            return pd.Series({'mean': m, 'ci95': ci95})

        grouped = (
            df.groupby(['model', 'alpha', 'num_clients'], as_index=False)
              .apply(agg_ci)
              .reset_index()
        )

        # Output directories
        plots_dir = base_dir / 'plots/generation'
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Per-alpha plots
        alphas = sorted(grouped['alpha'].dropna().unique().tolist())
        for alpha in alphas:
            sub = grouped[grouped['alpha'] == alpha].copy()
            if sub.empty:
                continue

            plt.figure(figsize=(8, 5))
            ax = plt.gca()

            for model in sub['model'].unique():
                mdf = sub[sub['model'] == model].sort_values('num_clients')
                x = mdf['num_clients'].values
                y = mdf['mean'].values
                ci = mdf['ci95'].values

                # Line with markers
                ax.plot(x, y, marker='o', label=model)
                # Confidence band
                ax.fill_between(x, y - ci, y + ci, alpha=0.15)

            ax.set_title(f'ROUGE-1 F1 vs #Clients (alpha={alpha})')
            ax.set_xlabel('# Clients')
            ax.set_ylabel('ROUGE-1 F1')
            ax.set_xticks(sorted(sub['num_clients'].unique()))
            ax.legend(title='Model')
            plt.tight_layout()

            # Save to both analysis and LaTeX plots folder
            fname = f'rouge1_vs_clients_alpha_{alpha}.png'
            plt.savefig(output_dir / fname, dpi=300, bbox_inches='tight')
            plt.savefig(plots_dir / fname, dpi=300, bbox_inches='tight')
            plt.close()

    def plot_generation_bleu4_vs_clients(self):
        """Plot BLEU-4 vs number of clients with lines, markers, and 95% CI bands."""
        if 'generation_federated' not in self.metrics:
            return

        df = self.metrics['generation_federated'].copy()
        required_cols = {'model', 'alpha', 'num_clients', 'bleu4', 'round'}
        if not required_cols.issubset(df.columns):
            print(f"Missing columns for BLEU-4 plot: {required_cols - set(df.columns)}")
            return

        def agg_ci(g):
            m = g['bleu4'].mean()
            s = g['bleu4'].std(ddof=1)
            n = g['bleu4'].count()
            sem = s / np.sqrt(n) if n > 0 and not np.isnan(s) else 0.0
            ci95 = 1.96 * sem
            return pd.Series({'mean': m, 'ci95': ci95})

        grouped = (
            df.groupby(['model', 'alpha', 'num_clients'], as_index=False)
              .apply(agg_ci)
              .reset_index()
        )

        plots_dir = base_dir / 'plots/generation'
        plots_dir.mkdir(parents=True, exist_ok=True)

        alphas = sorted(grouped['alpha'].dropna().unique().tolist())
        for alpha in alphas:
            sub = grouped[grouped['alpha'] == alpha].copy()
            if sub.empty:
                continue
            plt.figure(figsize=(8, 5))
            ax = plt.gca()
            for model in sub['model'].unique():
                mdf = sub[sub['model'] == model].sort_values('num_clients')
                x = mdf['num_clients'].values
                y = mdf['mean'].values
                ci = mdf['ci95'].values
                ax.plot(x, y, marker='o', label=model)
                ax.fill_between(x, y - ci, y + ci, alpha=0.15)
            ax.set_title(f'BLEU-4 vs #Clients (alpha={alpha})')
            ax.set_xlabel('# Clients')
            ax.set_ylabel('BLEU-4')
            ax.set_xticks(sorted(sub['num_clients'].unique()))
            ax.legend(title='Model')
            plt.tight_layout()
            fname = f'bleu4_vs_clients_alpha_{alpha}.png'
            plt.savefig(output_dir / fname, dpi=300, bbox_inches='tight')
            plt.savefig(plots_dir / fname, dpi=300, bbox_inches='tight')
            plt.close()

        if len(alphas) > 1:
            plt.figure(figsize=(9, 5))
            ax = plt.gca()
            for model in grouped['model'].unique():
                for alpha in alphas:
                    mdf = grouped[(grouped['model'] == model) & (grouped['alpha'] == alpha)].sort_values('num_clients')
                    if mdf.empty:
                        continue
                    x = mdf['num_clients'].values
                    y = mdf['mean'].values
                    ci = mdf['ci95'].values
                    ax.plot(x, y, marker='o', label=f"{model}, α={alpha}")
                    ax.fill_between(x, y - ci, y + ci, alpha=0.08)
            ax.set_title('BLEU-4 vs #Clients (by alpha)')
            ax.set_xlabel('# Clients')
            ax.set_ylabel('BLEU-4')
            ax.set_xticks(sorted(grouped['num_clients'].unique()))
            ax.legend(title='Configuration', ncol=2)
            plt.tight_layout()
            fname = 'bleu4_vs_clients_combined.png'
            plt.savefig(output_dir / fname, dpi=300, bbox_inches='tight')
            plt.savefig(plots_dir / fname, dpi=300, bbox_inches='tight')
            plt.close()

    def plot_classification_f1_vs_clients(self):
        """Plot classification F1 vs #clients (lines + 95% CI) by model and alpha."""
        if 'federated' not in self.metrics:
            return

        df = self.metrics['federated'].copy()
        # Prefer validation/test f1 if phase exists; otherwise use f1 as-is
        if 'phase' in df.columns:
            df = df[df['phase'].isin(['validation', 'val', 'test'])].copy()
        required_cols = {'model', 'alpha', 'num_clients', 'f1'}
        if 'round' in df.columns:
            required_cols.add('round')
        if not required_cols.issubset(df.columns):
            print(f"Missing columns for classification F1 plot: {required_cols - set(df.columns)}")
            return

        group_keys = ['model', 'alpha', 'num_clients']
        if 'round' in df.columns:
            def agg_ci(g):
                m = g['f1'].mean()
                s = g['f1'].std(ddof=1)
                n = g['f1'].count()
                sem = s / np.sqrt(n) if n > 0 and not np.isnan(s) else 0.0
                ci95 = 1.96 * sem
                return pd.Series({'mean': m, 'ci95': ci95})
            grouped = df.groupby(group_keys, as_index=False).apply(agg_ci).reset_index()
        else:
            grouped = df.groupby(group_keys)['f1'].mean().reset_index().rename(columns={'f1':'mean'})
            grouped['ci95'] = 0.0

        plots_dir = base_dir / 'plots/classification'
        plots_dir.mkdir(parents=True, exist_ok=True)

        alphas = sorted(grouped['alpha'].dropna().unique().tolist())
        for alpha in alphas:
            sub = grouped[grouped['alpha'] == alpha].copy()
            if sub.empty:
                continue
            plt.figure(figsize=(8, 5))
            ax = plt.gca()
            for model in sub['model'].unique():
                mdf = sub[sub['model'] == model].sort_values('num_clients')
                x = mdf['num_clients'].values
                y = mdf['mean'].values
                ci = mdf['ci95'].values
                ax.plot(x, y, marker='o', label=model)
                ax.fill_between(x, y - ci, y + ci, alpha=0.15)
            ax.set_title(f'Classification F1 vs #Clients (alpha={alpha})')
            ax.set_xlabel('# Clients')
            ax.set_ylabel('F1')
            ax.set_xticks(sorted(sub['num_clients'].unique()))
            ax.legend(title='Model')
            plt.tight_layout()
            fname = f'cls_f1_vs_clients_alpha_{alpha}.png'
            plt.savefig(output_dir / fname, dpi=300, bbox_inches='tight')
            plt.savefig(plots_dir / fname, dpi=300, bbox_inches='tight')
            plt.close()

        if len(alphas) > 1:
            plt.figure(figsize=(9, 5))
            ax = plt.gca()
            for model in grouped['model'].unique():
                for alpha in alphas:
                    mdf = grouped[(grouped['model'] == model) & (grouped['alpha'] == alpha)].sort_values('num_clients')
                    if mdf.empty:
                        continue
                    x = mdf['num_clients'].values
                    y = mdf['mean'].values
                    ci = mdf['ci95'].values
                    ax.plot(x, y, marker='o', label=f"{model}, α={alpha}")
                    ax.fill_between(x, y - ci, y + ci, alpha=0.08)
            ax.set_title('Classification F1 vs #Clients (by alpha)')
            ax.set_xlabel('# Clients')
            ax.set_ylabel('F1')
            ax.set_xticks(sorted(grouped['num_clients'].unique()))
            ax.legend(title='Configuration', ncol=2)
            plt.tight_layout()
            fname = 'cls_f1_vs_clients_combined.png'
            plt.savefig(output_dir / fname, dpi=300, bbox_inches='tight')
            plt.savefig(plots_dir / fname, dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    analyzer = PaperMetricsAnalyzer()
    results = analyzer.run_analysis()
