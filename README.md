
# Federated Learning for Text Summarization with DistilBART

This repository contains an implementation of Federated Learning with DistilBART for abstractive text summarization on the CNN/DailyMail dataset. The implementation supports federated training of sequence-to-sequence models with non-IID data distribution across clients.

## 📊 Latest Results (2025-08-07)

### Final Evaluation Metrics (Round 13)
- **Loss**: 1.7397
- **ROUGE-1**: 28.25
- **ROUGE-2**: 32.49
- **ROUGE-L**: 29.36

### Training Configuration
- **Model**: DistilBART (facebook/bart-large-cnn)
- **Number of Clients**: 10
- **Clients per Round**: 5
- **Local Epochs**: 1
- **Batch Size**: 16
- **Learning Rate**: 5e-5
- **Max Sequence Length**: 1024 tokens
- **Max Target Length**: 142 tokens

### Performance Trends
- The model shows consistent improvement in ROUGE scores throughout training
- Best validation loss achieved: 1.7397
- Stable convergence with federated averaging

### Key Observations
- The model achieves competitive ROUGE scores while maintaining data privacy
- Federated training demonstrates effective knowledge sharing across clients
- The implementation handles the non-IID nature of client data effectively

## ✨ Features

- Federated Learning with DistilBART (distilled version of BART)
- Support for CNN/DailyMail dataset for text summarization
- Non-IID data partitioning
- Client-side model training with local updates
- Centralized model aggregation (FedAvg)
- ROUGE score evaluation
- Progress tracking with tqdm
- GPU acceleration support
- Memory-efficient training with gradient accumulation

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

### Usage

#### Training

To train the federated DistilBART model on the CNN/DailyMail dataset:

```bash
python train_distilbart_cnndm.py \
    --num_clients 3 \
    --num_rounds 1 \
    --epochs_per_client 1 \
    --batch_size 2 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --data_dir "./data/cnndm" \
    --model_save_path "./saved_models/distilbart_cnndm"
```

#### Arguments

- `--num_clients`: Number of clients in federated learning (default: 3)
- `--num_rounds`: Number of federated learning rounds (default: 1)
- `--epochs_per_client`: Number of local training epochs per client (default: 1)
- `--batch_size`: Training batch size (default: 2)
- `--learning_rate`: Learning rate for AdamW optimizer (default: 5e-5)
- `--max_grad_norm`: Maximum gradient norm for gradient clipping (default: 1.0)
- `--data_dir`: Directory to store/load the dataset (default: "./data/cnndm")
- `--model_save_path`: Path to save the trained model (default: "./saved_models/distilbart_cnndm")

## 🏗️ Implementation Details

### Model Architecture
- Based on DistilBART (distilled version of BART) from Hugging Face
- Sequence-to-sequence model for abstractive summarization
- Tokenizer: BART tokenizer with a maximum sequence length of 1024 tokens

### Training Process
1. The global model is initialized with pre-trained DistilBART weights
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

This repository contains an implementation of Federated Learning with DistilBART for abstractive text summarization on the CNN/DailyMail dataset. The implementation supports federated training of sequence-to-sequence models with non-IID data distribution across clients.

### Features

- Federated Learning with DistilBART
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
git clone https://github.com/yourusername/fed-distilbart-cnndm.git
cd fed-distilbart-cnndm
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

### Usage

#### Training

To train the federated DistilBART model on the CNN/DailyMail dataset:

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
   git clone https://github.com/yourusername/fed-distilbart-cnndm.git
   cd fed-distilbart-cnndm
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Edit the `configs/distilbart_cnndm.yaml` file to adjust training parameters such as:
- Number of clients
- Clients per round
- Batch size
- Learning rate
- Sequence lengths
- Number of training rounds

## Running the Code

### Training

To train the federated DistilBART model on the CNN/DailyMail dataset:

```bash
python train_distilbart_cnndm.py \
    --num_clients 3 \
    --num_rounds 1 \
    --epochs_per_client 1 \
    --batch_size 2 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --data_dir "./data/cnndm" \
    --model_save_path "./saved_models/distilbart_cnndm"
```

### Visualization
To visualize training metrics:
```bash
python visualize_fed_distilbart_cnndm.py --results-dir ./results_distilbart_cnndm_federated
```

### Using Configuration File
Alternatively, you can use a YAML configuration file:
```bash
python train_distilbart_cnndm.py --config configs/distilbart_cnndm.yaml
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
