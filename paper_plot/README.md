# Federated Learning Analysis for Text Classification

This directory contains scripts and results for analyzing federated learning experiments with BART and DistilBART models on text classification tasks.

## Project Structure

```
paper_plot/
├── analysis_results/            # Analysis outputs and visualizations
│   ├── classification/          # Classification-specific results
│   └── paper_metrics/           # Publication-ready metrics and plots
├── data_to_plot/                # Raw data for visualization
│   ├── classification/          # Text classification task data
│   └── generation/              # Text generation task data
├── scripts/                     # Analysis scripts
│   ├── analyze_client_distributions.py  # Client data distribution analysis
│   └── analyze_paper_metrics.py         # Main analysis for paper metrics
├── organize_classification_data.sh      # Data organization script
├── organize_results.sh                  # Results organization script
└── run_analysis.sh                      # Main analysis pipeline
```

## Paper (LaTeX): Build & Style

- Source: `paper_plot/latex/conference_101719.tex`
- Compile (from `paper_plot/latex/`):

```bash
pdflatex -interaction=nonstopmode conference_101719.tex
```

- Figure/plot paths referenced from `paper_plot/plots/` and `paper_plot/analysis_results/`.
- For IEEE two-column layout, prefer `table*` for wide tables (5+ columns) to span both columns and avoid overflow.

### Highlighting Policy
- Tables: bold indicates best or key results in each metric column. For loss metrics, lower is better and bold highlights the lowest value.
- Analysis narrative (Findings/Takeaway/Deep Analysis): one bold key phrase per sentence for emphasis with a light touch; takeaways prioritized for emphasis.
- Captions explicitly note what bold means.

## Key Results Summary (Used in Paper)

### Classification (20 Newsgroups)
- Federated vs centralized: both models perform better in federated settings.
- BART-large > DistilBART by ~1.0% accuracy in federated best settings.
- Client scaling trade-off: BART-large peaks with fewer clients (≈5); DistilBART scales to more clients (≈10) with competitive scores.
- Stability: BART-large shows lower inter-client variance; both converge within ~22 rounds.

### Generation (CNN/DailyMail)
- Comparable centralized generation performance between BART-large and DistilBART.
- DistilBART offers strong efficiency–quality trade-offs given its smaller size.

### Non-IID Effects
- Non-IID skew increases variance and can reduce generation metrics; mitigation and model choice matter (BART-large generally more robust).

## Data & Plots Locations
- CSVs and derived tables: `paper_plot/analysis_results/` (e.g., `classification/federated/best_rounds.csv`, `training_metrics_summary.csv`).
- Plots used by the paper: `paper_plot/plots/` and subdirectories under `analysis_results/`.
- Scripts to regenerate metrics/plots: `paper_plot/scripts/` and helper shell scripts in this folder.

## Key Findings

### 1. Classification Task

#### Model Performance

| Model | Setup | Clients | Accuracy | F1 Score | Precision | Recall |
|-------|-------|---------|----------|-----------|-----------|--------|
| BART-large | Centralized | - | 73.45% | 73.21% | 74.12% | 73.45% |
| BART-large | Federated (Best) | 2 | 75.98% (±1.88%) | 75.76% (±2.21%) | 77.01% (±1.49%) | 75.98% (±1.88%) |
| DistilBART | Centralized | - | 69.50% | 69.80% | 71.00% | 69.80% |
| DistilBART | Federated (Best) | 2 | 71.42% (±1.75%) | 71.10% (±1.92%) | 71.82% (±1.64%) | 71.42% (±1.75%) |

#### Classification Insights
- **Federated vs Centralized**: Both models show improved performance in federated settings
  - BART-large: +2.53% accuracy improvement
  - DistilBART: +1.92% accuracy improvement
- **Optimal Configuration**:
  - BART-large performs best with 2 clients and 22 rounds
  - DistilBART performs best with 2 clients and 21 rounds
- **Stability**: BART-large shows more stable performance across different client configurations

### 2. Generation Task

#### Model Performance (Centralized)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | METEOR |
|-------|---------|---------|---------|------|--------|
| BART-large | 0.415 | 0.195 | 0.287 | 14.47 | 0.397 |
| DistilBART | 0.409 | 0.190 | 0.284 | 14.58 | 0.382 |

#### Generation Insights
- **Performance**: Both models show comparable performance in text generation
- **Efficiency**: DistilBART provides a good balance between performance and model size
- **Metrics**: ROUGE-1 and ROUGE-L scores indicate good content coverage and fluency

### 3. Cross-Task Analysis

1. **Model Performance**
   - BART-large consistently outperforms DistilBART in both tasks
   - The performance gap is more significant in classification than generation

2. **Federated Learning Impact**
   - Classification benefits more from federated learning than generation
   - Optimal number of clients varies by task and model

3. **Resource Efficiency**
   - DistilBART offers better computational efficiency with minimal performance drop
   - BART-large provides better accuracy when resources are not constrained

3. **Data Heterogeneity (α)**
   - α=0.1: More heterogeneous data distribution
   - α=0.5: More balanced distribution
   - Better performance with α=0.5 for both models

## How to Reproduce

1. **Setup Environment**
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Run Analysis**
   ```bash
   # Organize data
   ./organize_classification_data.sh
   
   # Run analysis pipeline
   ./run_analysis.sh
   
   # Generate paper metrics
   python scripts/analyze_paper_metrics.py
   ```

## Generated Outputs

### Plots
- `comparison_*.png`: Performance comparison between models and setups
- `learning_curve_*.png`: Training dynamics for different configurations
- `class_distribution_*.png`: Data distribution across clients

### Tables
- `centralized_results.tex`: LaTeX table of centralized training results
- `federated_results.tex`: LaTeX table of federated training results

### Data
- `analysis_results.json`: Complete analysis results in JSON format
- `client_distributions_*.csv`: Detailed client distribution data

## Dependencies

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please contact [Your Name] at [Your Email].
