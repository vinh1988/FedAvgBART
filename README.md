
# Federated Learning for Text Summarization with BART

This repository contains an implementation of Federated Learning with DistilBART for abstractive text summarization on the CNN/DailyMail dataset. The implementation supports federated training of sequence-to-sequence models with non-IID data distribution across clients, along with comprehensive visualization and analysis tools.

## 📊 Latest Results (2025-08-09)

### Performance Across Different Client Counts
| Clients | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---------|---------|---------|---------|
| 2       | 33.36   | 9.82    | 20.15   |
| 3       | 34.28   | 13.82   | 22.73   |
| 4       | 37.95   | 14.50   | 23.43   |
| 5       | 35.87   | 14.24   | 26.05   |

### Model Configuration
- **Base Model**: BART (facebook/bart-base)
- **Federated Learning**: FedAvg with weighted aggregation
- **Client Selection**: Random sampling per round
- **Evaluation**: Performed on a held-out test set

### Key Observations
- The model achieves strong performance while maintaining data privacy
- Performance generally improves with more clients, indicating effective knowledge sharing
- The implementation handles non-IID data distribution effectively
- ROUGE-L scores show particular improvement with more clients

## ✨ Features

- **Federated Learning** with BART (Bidirectional and Auto-Regressive Transformers)
- **Multi-client Training** with configurable client counts and participation rates
- **Comprehensive Metrics**:
  - ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
  - Training and evaluation loss tracking
  - Client contribution analysis
- **Visualization Tools**:
  - Performance comparison across client counts
  - Client contribution heatmaps
  - ROUGE score trends
  - Data distribution analysis
  - **Configuration Analysis**: Comprehensive comparison of different client configurations
- **Efficient Training**:
  - Gradient accumulation
  - Mixed precision training
  - Memory optimization
- **Flexible Configuration**:
  - YAML-based configuration
  - Command-line overrides
  - Experiment tracking

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.12.0+
- Transformers 4.18.0+
- datasets (Hugging Face)
- rouge-score
- tqdm
- numpy

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fed-distilbart-cnndm.git
cd fed-distilbart-cnndm
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## 🚀 Experiment Workflow

### 1. Running Federated Learning Experiments

#### Single Configuration
To train with a specific number of clients:

```bash
python train_fed_bart_cnndm.py \
    --config configs/bart_cnndm.yaml \
    --num_clients 5 \
    --num_rounds 10 \
    --output_dir experiment_results/clients_5
```

#### Multiple Configurations
To run experiments with different numbers of clients (2-10):

```bash
python run_experiments.py --config configs/bart_cnndm.yaml \
    --min-clients 2 \
    --max-clients 10 \
    --num-rounds 10
```

### 2. Collecting and Consolidating Metrics

After running experiments, consolidate metrics from all client configurations:

```bash
python -m src.analysis.collect_metrics \
    --base-dir ./experiment_results \
    --output-file ./experiment_results/analysis/consolidated_metrics.csv \
    --min-clients 2 \
    --max-clients 10
```

### 3. Generating Visualizations

Generate comprehensive visualizations from the consolidated metrics:

```bash
python -m src.visualization.visualize_consolidated_metrics \
    --metrics-file ./experiment_results/analysis/consolidated_metrics.csv \
    --output-dir ./experiment_results/analysis/plots
```

### 4. Analyzing Configuration Performance

To compare different client configurations and find the optimal setup:

```bash
python -m src.visualization.analyze_config_performance \
    --metrics-file ./experiment_results/analysis/consolidated_metrics.csv \
    --output-dir ./experiment_results/analysis/plots
```

**Outputs**:
- `config_performance_comparison.png`: Comprehensive visualization of all metrics across configurations
  - ROUGE and BLEU scores by client count
  - Training loss and BERTScore comparisons
  - Client contribution metrics (Gini coefficient, CV)
  - Client participation statistics

### 5. Understanding the Outputs

#### Generated Files
- `consolidated_metrics.csv`: All metrics in a single CSV file
- `plots/rouge_metrics.png`: ROUGE scores across client configurations
- `plots/bleu_metrics.png`: BLEU scores across client configurations
- `plots/bertscore_and_loss.png`: BERTScore and training loss
- `plots/contribution_metrics.png`: Client contribution analysis
- `plots/config_performance_comparison.png`: Comprehensive configuration comparison
- `plots/metrics_summary.csv`: Tabular summary of all metrics

## 📊 Metrics Explanation

### Text Generation Metrics

#### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- **ROUGE-1**: Measures unigram (single word) overlap between generated and reference summaries
- **ROUGE-2**: Measures bigram overlap between generated and reference summaries
- **ROUGE-L**: Measures the longest common subsequence between generated and reference summaries
  - Higher ROUGE scores (0-100 scale) indicate better summary quality

#### BLEU (Bilingual Evaluation Understudy)
- **BLEU-1 to BLEU-4**: Measures n-gram precision between generated and reference summaries
  - BLEU-1: Unigram precision
  - BLEU-2: Bigram precision
  - BLEU-3: Trigram precision
  - BLEU-4: 4-gram precision
  - Higher BLEU scores (0-100 scale) indicate better n-gram matching

#### BERTScore
- **BERTScore F1**: Measures semantic similarity between generated and reference texts using contextual embeddings
  - Values closer to 100 indicate better semantic similarity
  - More robust to paraphrasing than n-gram based metrics

### Training Metrics
- **Loss**: Cross-entropy loss during training
  - Lower values indicate better model fitting
  - Should generally decrease and stabilize over training rounds

### Federated Learning Metrics
- **Gini Coefficient**: Measures inequality in client contributions
  - 0 = perfect equality, 1 = maximum inequality
- **CV (Coefficient of Variation)**: Measures relative variability in client contributions
  - Lower values indicate more balanced participation
- **Mean/Min/Max Contribution**: Statistics about client participation

## 🛠️ Configuration Options

Key parameters in `configs/bart_cnndm.yaml`:

```yaml
data:
  dataset_name: "cnn_dailymail"
  dataset_config: "3.0.0"
  max_source_length: 1024
  max_target_length: 142
  
model:
  model_name: "facebook/bart-large-cnn"
  learning_rate: 5e-5
  weight_decay: 0.01
  
federated:
  num_clients: 5
  clients_per_round: 3
  num_rounds: 10
  
training:
  batch_size: 4
  gradient_accumulation_steps: 4
  num_train_epochs: 1
  fp16: True
  
evaluation:
  eval_batch_size: 4
  max_eval_samples: 100
```

## 📈 Analysis and Interpretation

### Configuration Analysis

1. **Optimal Client Count**
   - Run the configuration analysis tool to identify the best performing setup:
   ```bash
   python -m src.visualization.analyze_config_performance \
       --metrics-file ./experiment_results/analysis/consolidated_metrics.csv \
       --output-dir ./experiment_results/analysis/plots
   ```
   - Look for the "elbow point" in the performance curves where adding more clients yields diminishing returns
   - Balance between model performance and computational resources

2. **Performance Trends**
   - Check if ROUGE-L and BERTScore improve with more clients
   - Look for consistent patterns across different metrics

2. **Model Stability**
   - Consistent performance across different client counts indicates robust learning
   - Large fluctuations in metrics might suggest unstable training

3. **Client Contribution**
   - Balanced Gini coefficient (closer to 0) indicates fair client participation
   - High CV might suggest some clients are contributing more than others

Edit `configs/bart_cnndm.yaml` to customize:

```yaml
training:
  num_rounds: 10
  clients_per_round: 3
  batch_size: 8
  learning_rate: 5e-5
  max_seq_length: 1024
  max_target_length: 142
  
model:
  model_name: "facebook/bart-large-cnn"
  tokenizer_name: "facebook/bart-large-cnn"
  
data:
  train_split: "train"
  val_split: "validation"
  test_split: "test"
  max_samples: null  # Set to number for debugging
```

## 📊 Understanding the Metrics

### ROUGE Scores
- **ROUGE-1**: Measures unigram (single word) overlap between generated and reference summaries
- **ROUGE-2**: Measures bigram (two consecutive words) overlap
- **ROUGE-L**: Measures the longest common subsequence, capturing sentence-level structure

### Client Contribution Metrics
- **Data Volume**: Amount of training data per client
- **Gradient Updates**: Magnitude and direction of model updates
- **ROUGE Impact**: Contribution to model performance improvement

### Training Metrics
- **Loss**: Cross-entropy loss on the training data
- **Validation Loss**: Model performance on held-out validation data
- **Round Time**: Time taken per federated round

## 📈 Visualization Examples

### Performance Comparison
![Performance Comparison](experiment_results/analysis/performance_comparison.png)

### Client Contribution Heatmap
![Contribution Heatmap](experiment_results/analysis/contribution_heatmap.png)

### ROUGE Score Trends
![ROUGE Trends](experiment_results/analysis/rouge_trends.png)

## 🔍 Implementation Details

### Model Architecture
- Based on BART (Bidirectional and Auto-Regressive Transformers) from Hugging Face
- Sequence-to-sequence model for abstractive summarization
- Tokenizer: BART tokenizer with a maximum sequence length of 1024 tokens

### Training Process
1. The global model is initialized with pre-trained BART weights
2. In each federated round:
   - 5 out of 10 clients are randomly selected (50% participation rate)
   - Each client trains the model on its local data for 1 epoch
   - Model updates are sent to the server
   - The server aggregates the updates using FedAvg
   - The global model is updated with the aggregated weights
   - Model is evaluated on the validation set

### Evaluation
- ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
- Training and validation loss tracking
- Model checkpoints are saved after each round

## 💾 Memory Management

The implementation includes several memory optimization techniques:
- Gradient accumulation for larger effective batch sizes
- Mixed precision training (FP16)
- Efficient data loading with multiple workers
- Gradient clipping to prevent exploding gradients
- Dynamic batching
- Gradient checkpointing
- Memory-efficient attention

## Federated Learning for Text Summarization

This repository contains an implementation of Federated Learning with BART for abstractive text summarization on the CNN/DailyMail dataset. The implementation supports federated training of sequence-to-sequence models with non-IID data distribution across clients.

### Features

- Federated Learning with BART
- Support for non-IID data partitioning
- Configurable number of clients and federated learning rounds
- Evaluation using ROUGE metrics
- Integration with HuggingFace Transformers and Datasets
- Support for both CPU and GPU training metrics (accuracy, F1, precision, recall)
- Progress tracking with tqdm
- GPU acceleration support

## Results

### Final Evaluation Metrics (After 13 Rounds)
- **Loss**: 1.7397
- **ROUGE-1**: 28.25
- **ROUGE-2**: 32.49
- **ROUGE-L**: 29.36

## Implementation Details
## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.12.0+
- Transformers 4.18.0+
- scikit-learn
- tqdm

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fed-bart-cnndm.git
cd fed-bart-cnndm
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

### Usage

#### Training

To train the federated BART model on the CNN/DailyMail dataset:

#### Arguments

- `--num_clients`: Number of clients in federated learning
- `--num_rounds`: Number of federated learning rounds
- `--local_epochs`: Number of local training epochs per round
- `--batch_size`: Training batch size
- `--max_length`: Maximum sequence length for BART
- `--learning_rate`: Learning rate for AdamW optimizer
- `--test_batch_size`: Batch size for evaluation
- `--alpha`: Dirichlet distribution parameter for non-IID split (smaller values create more heterogeneous data distribution)
- `--seed`: Random seed for reproducibility

## Implementation Details

### Datasets
* Supports all image classification datasets in `torchvision.datasets`.
* Supports all text classification datasets in `torchtext.datasets`.
* Supports all datasets in [LEAF benchmark](https://leaf.cmu.edu/) (*NO need to prepare raw data manually*)
* Supports additional image classification datasets ([`TinyImageNet`](https://www.kaggle.com/c/tiny-imagenet), [`CINIC10`](https://datashare.ed.ac.uk/handle/10283/3192)).
* Supports additional text classification datasets ([`BeerReviews`](https://snap.stanford.edu/data/web-BeerAdvocate.html)).
* Supports tabular datasets ([`Heart`, `Adult`, `Cover`](https://archive.ics.uci.edu/ml/index.php)).
* Supports temporal dataset ([`GLEAM`](http://www.skleinberg.org/data.html))
* __NOTE__: don't bother to search raw files of datasets; the dataset can automatically be downloaded to the designated path by just passing its name!
### Statistical Heterogeneity Simulations
* `IID` (i.e., statistical homogeneity)
* `Unbalanced` (i.e., sample counts heterogeneity)
* `Pathological Non-IID` ([McMahan et al., 2016](https://arxiv.org/abs/1602.05629))
* `Dirichlet distribution-based Non-IID` ([Hsu et al., 2019](https://arxiv.org/abs/1909.06335))
* `Pre-defined` (for datasets having natural semantic separation, including `LEAF` benchmark ([Caldas et al., 2018](https://arxiv.org/abs/1812.01097)))
### Models
* `LogReg` (logistic regression), `StackedTransformer` (TransformerEncoder-based classifier)
* `TwoNN`, `TwoCNN`, `SimpleCNN` ([McMahan et al., 2016](https://arxiv.org/abs/1602.05629))
* `FEMNISTCNN`, `Sent140LSTM` ([Caldas et al., 2018](https://arxiv.org/abs/1812.01097)))
* `FedProx` (Li et al., 2018) <a href='https://arxiv.org/abs/1812.06127'>Federated Optimization in Heterogeneous Networks</a>

## Evaluation

### Metrics
- **ROUGE Scores**: ROUGE-1, ROUGE-2, ROUGE-L
- **Training Loss**: Cross-entropy loss on training data
- **Validation Loss**: Cross-entropy loss on validation set

## Requirements

### Core Dependencies
- Python 3.8+
- PyTorch 1.12.0+
- Transformers 4.18.0+
- datasets (Hugging Face)
- rouge-score
- tqdm
- numpy

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fed-bart-cnndm.git
   cd fed-bart-cnndm
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Edit the `configs/bart_cnndm.yaml` file to adjust training parameters such as:
- Number of clients
- Clients per round
- Batch size
- Learning rate
- Sequence lengths
- Number of training rounds

## Running the Code

### Training

To train the federated BART model on the CNN/DailyMail dataset:

```bash
python train_fed_bart_cnndm.py \
    --num_clients 3 \
    --num_rounds 1 \
    --epochs_per_client 1 \
    --batch_size 2 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --data_dir "./data/cnndm" \
    --model_save_path "./saved_models/bart_cnndm"
```

### Visualization
To visualize training metrics:
```bash
python visualize_fed_bart_cnndm.py --results-dir ./results_bart_cnndm_federated
```

### Using Configuration File
Alternatively, you can use a YAML configuration file:
```bash
python train_fed_bart_cnndm.py --config configs/bart_cnndm.yaml
```

## Performance Optimization

### Training Efficiency
- The implementation includes several optimizations for efficient federated training
- Gradient accumulation and mixed precision training are supported for better memory utilization
- Consider implementing class weights in the loss function
- Potential improvements through oversampling minority classes or undersampling majority classes

### Hyperparameter Tuning
- Experiment with different learning rates and scheduling strategies
- Try different batch sizes based on available GPU memory
- Adjust the number of local epochs and federated rounds

### Model Architecture
- Current implementation uses DistilBART for efficient sequence-to-sequence learning
- Potential to experiment with different model variants
- Consider adding additional regularization techniques

## Future Work

- [ ] Implement class weighting for imbalanced datasets
- [ ] Add learning rate scheduling with warmup
- [ ] Support for more sequence-to-sequence model architectures
- [ ] Add differential privacy for enhanced privacy guarantees
- [ ] Implement client selection strategies based on data distribution
- [ ] Add support for more text classification datasets
- [ ] Implement model compression techniques for edge deployment
- [ ] Add support for cross-silo federated learning

## Acknowledgements

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## TODO
- [ ] Support more sequence-to-sequence models for cross-device FL setting
- [ ] Support another structured dataset including temporal and tabular data, along with datasets suitable for cross-silo FL setting. (e.g., [`MedMNIST`](https://github.com/MedMNIST/MedMNIST))
- [ ] Add other popular FL algorithms including personalized FL algorithms (e.g., [`SuPerFed`](https://arxiv.org/abs/2109.07628)).
- [ ] Attach benchmark results of sample commands.

## Contact
Should you have any feedback, please create a thread in __issue__ tab. Thank you :)
