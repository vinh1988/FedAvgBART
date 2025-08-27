
# Federated Learning in PyTorch

Implementations of various Federated Learning (FL) algorithms in PyTorch, especially for research purposes.

## Federated Text Classification with DistilBART

This repository contains an implementation of Federated Learning with DistilBART for text classification on the 20 Newsgroups dataset. The implementation includes support for non-IID data distribution across clients using Dirichlet distribution.

## Features

- Federated Learning with DistilBART (distilled version of BART)
- Support for 20 Newsgroups text classification (20 classes)
- Non-IID data partitioning using Dirichlet distribution
- Client-side model training with local updates
- Centralized model aggregation (FedAvg)
- Comprehensive evaluation metrics (accuracy, F1, precision, recall)
- Progress tracking with tqdm and Weights & Biases
- GPU acceleration support
- Experiment tracking and visualization
- Model checkpointing and versioning

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.12.0+
- Transformers 4.18.0+
- scikit-learn
- tqdm
- numpy
- pandas
- matplotlib
- Weights & Biases (`wandb`)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/FED-OPT-BERT.git
cd FED-OPT-BERT
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Log in to Weights & Biases (if you haven't already):
```bash
wandb login
```
   Follow the instructions to authenticate with your Weights & Biases account. If you don't have an account, you can create one at [wandb.ai](https://wandb.ai).

### Usage

#### Training

To train the federated DistilBART model on the 20 Newsgroups dataset:

```bash
python train_distilbart_20news.py \
    --num_clients 3 \
    --num_rounds 5 \
    --epochs_per_client 1 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --data_dir "./data/20news" \
    --model_save_path "./saved_models/distilbart_20news"
```

#### Arguments

- `--num_clients`: Number of clients in federated learning (default: 3)
- `--num_rounds`: Number of federated learning rounds (default: 5)
- `--epochs_per_client`: Number of local training epochs per client (default: 1)
- `--batch_size`: Training batch size (default: 16)
- `--learning_rate`: Learning rate for AdamW optimizer (default: 2e-5)
- `--max_grad_norm`: Maximum gradient norm for gradient clipping (default: 1.0)
- `--data_dir`: Directory to store/load the dataset (default: "./data/20news")
- `--model_save_path`: Path to save the trained model (default: "./saved_models/distilbart_20news")

## Experiment Tracking with Weights & Biases

This project uses Weights & Biases (wandb) for experiment tracking, visualization, and model management. Each training run is automatically logged to your wandb account, where you can:

- Track training and validation metrics in real-time
- Compare different runs and hyperparameters
- Monitor system resource usage (CPU/GPU/memory)
- Save and version model checkpoints
- Visualize model predictions

### Logged Metrics

- **Training Metrics** (per client, per epoch):
  - Loss
  - Accuracy
  - Precision (weighted)
  - Recall (weighted)
  - F1 Score (weighted)

- **Validation Metrics** (per round):
  - Loss
  - Accuracy
  - Precision (weighted)
  - Recall (weighted)
  - F1 Score (weighted)

### Viewing Results

1. During or after training, visit your [Weights & Biases dashboard](https://wandb.ai/)
2. Select your project (`federated-distilbart-20news` by default)
3. Explore the different tabs:
   - **Charts**: Interactive plots of all metrics
   - **System**: Resource utilization
   - **Models**: Saved model checkpoints
   - **Files**: Logs and artifacts

## Implementation Details

### Model Architecture
- Based on DistilBART (distilled version of BART) from Hugging Face
- Custom classification head for 20 Newsgroups classification
- Tokenizer: DistilBERT tokenizer with a maximum sequence length of 128 tokens

### Training Process
1. The global model is initialized with pre-trained DistilBART weights
2. In each federated round:
   - A subset of clients is selected
   - Each client trains the model on its local data
   - Model updates are sent to the server
   - The server aggregates the updates using FedAvg
   - The global model is updated with the aggregated weights

### Evaluation
- Accuracy, Precision, Recall, F1 Score
- Confusion matrix
- Per-class metrics
- Real-time tracking with Weights & Biases
- Automatic logging of all metrics and model checkpoints

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
* `LeNet` ([LeCun et al., 1998](https://ieeexplore.ieee.org/document/726791/)), `MobileNet` ([Howard et al., 2019](https://arxiv.org/abs/1905.02244)), `SqueezeNet` ([Iandola et al., 2016](https://arxiv.org/abs/1602.07360)), `VGG` ([Simonyan et al., 2014](https://arxiv.org/abs/1409.1556)), `ResNet` ([He et al., 2015](https://arxiv.org/abs/1512.03385))
* `MobileNeXt` ([Daquan et al., 2020](https://arxiv.org/abs/2007.02269)), `SqueezeNeXt` ([Gholami et al., 2016](https://arxiv.org/abs/1803.10615)), `MobileViT` ([Mehta et al., 2021](https://arxiv.org/abs/2110.02178))
* `DistilBERT` ([Sanh et al., 2019](https://arxiv.org/abs/1910.01108)), `SqueezeBERT` ([Iandola et al., 2020](https://arxiv.org/abs/2006.11316)), `MobileBERT` ([Sun et al., 2020](https://arxiv.org/abs/2004.02984))
* `M5` ([Dai et al., 2016](https://arxiv.org/abs/1610.00087))
### Algorithms
* `FedAvg` and `FedSGD` (McMahan et al., 2016) <a href='https://arxiv.org/abs/1602.05629'>Communication-Efficient Learning of Deep Networks from Decentralized Data</a>
* `FedAvgM` (Hsu et al., 2019) <a href='https://arxiv.org/abs/1909.06335'>Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification</a>
* `FedProx` (Li et al., 2018) <a href='https://arxiv.org/abs/1812.06127'>Federated Optimization in Heterogeneous Networks</a>
* `FedOpt` (`FedAdam`, `FedYogi`, `FedAdaGrad`) (Reddi et al., 2020) <a href='https://arxiv.org/abs/2003.00295'>Adaptive Federated Optimization</a>

### Evaluation schemes
* `local`: evaluate FL algorithm using holdout sets of (some/all) clients NOT participating in the current round. (i.e., evaluation of personalized federated learning setting)
* `global`: evaluate FL algorithm using global holdout set located at the server. (*ONLY available if the raw dataset supports pre-defined validation/test set*).
* `both`: evaluate FL algorithm using both `local` and `global` schemes.
### Metrics
* Top-1 Accuracy, Top-5 Accuracy, Precision, Recall, F1
* Area under ROC, Area under PRC, Youden's J
* Seq2Seq Accuracy
* MSE, RMSE, MAE, MAPE
* $R^2$, $D^2$

## Requirements
* See `requirements.txt`. (I recommend building an independent environment for this project, using e.g., `Docker` or `conda`)
* When you install `torchtext`, please check the version compatibility with `torch`. (See [official repository](https://github.com/pytorch/text#installation))
* Plus, please install `torch`-related packages using one command provided by the official guide (See [official installation guide](https://pytorch.org/get-started/locally/)); e.g., `conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 torchtext==0.13.0 cudatoolkit=11.6 -c pytorch -c conda-forge` 

## Configurations
* See `python3 main.py -h`.

## Example Commands
* See shell files prepared in `commands` directory.

### Background Dirichlet alpha sweep (nohup, using experiment runner)

Run a short sweep for Dirichlet α ∈ {0.1, 0.5} sequentially in the background, logging to `nohup_alpha_sweep.log`. Results are written under `--output_dir`.

```bash
nohup bash -lc '
for a in 0.1 0.5; do
  WANDB_MODE=offline /mnt/sda1/Projects/jsl/vp_gitlab/FED/FED-OPT-BERT/FED-OPT-BERT-main/.venv/bin/python \
    tools/run_20news_experiments.py \
    --min-clients 2 --max-clients 10 --num-rounds 22 \
    --participation-rate 1.0 --dirichlet-alpha "$a" --dirichlet-min-size 50 \
    --output_dir results_distilbart_fed_runs_20news
done
' > nohup_alpha_sweep.log 2>&1 &
```

Notes:
- `tools/run_20news_experiments.py` forwards flags to `train_distilbart_20news.py`.
- Omit `--output_dir` to use the default: `results_distilbart_fed_runs_20news`.

### BART-large Dirichlet alpha sweep (nohup, using experiment runner)

Run the same sweep for BART-large by pointing the experiment runner to the BART script with `--train-script`. Results are written under the specified `--output_dir`.

```bash
nohup bash -lc '
for a in 0.1 0.5; do
  WANDB_MODE=offline /mnt/sda1/Projects/jsl/vp_gitlab/FED/FED-OPT-BERT/FED-OPT-BERT-main/.venv/bin/python \
    tools/run_20news_experiments.py \
    --min-clients 2 --max-clients 10 --num-rounds 22 \
    --participation-rate 1.0 --dirichlet-alpha "$a" --dirichlet-min-size 50 \
    --train-script /mnt/sda1/Projects/jsl/vp_gitlab/FED/FED-OPT-BERT/20news/FED-OPT-BERT-PYTORCH/train_bart_large_20news.py \
    --output_dir results_bart_large_fed_runs_20news
done
' > nohup_alpha_sweep_bart_large.log 2>&1 &
```

Notes:
- `tools/run_20news_experiments.py` forwards flags to the script provided via `--train-script` (BART-large in this example).
- Omit `--output_dir` to use the default of the target train script.

## Experiment Results

### Latest Training Run (2025-03-08)
- **Model**: DistilBART-base
- **Dataset**: 20 Newsgroups
- **Configuration**:
  - Number of clients: 10
  - Federated rounds: 22
  - Epochs per client: 1
  - Batch size: 16
  - Learning rate: 2e-5
  - Max sequence length: 128 tokens

### Performance Metrics (Final Round)
| Metric | Training | Validation |
|--------|----------|------------|
| Loss   | 0.644    | 0.062      |
| Accuracy | 0.798  | 0.724      |
| Precision | 0.835 | 0.727     |
| Recall | 0.830    | 0.724      |
| F1 Score | 0.829  | 0.720      |

### Performance Trends
- The model shows consistent improvement over federated rounds
- Training metrics show good convergence
- Validation metrics indicate the model generalizes well
- The gap between training and validation metrics suggests some overfitting, which is expected with local training

## Performance Optimization

### Class Imbalance
- The 20 Newsgroups dataset has relatively balanced classes
- Consider implementing class weights if needed for specific non-IID scenarios

### Hyperparameter Tuning
- Experiment with different learning rates and scheduling strategies
- Try different batch sizes based on available GPU memory
- Adjust the number of local epochs and federated rounds
- Use Weights & Biases Sweeps for automated hyperparameter optimization

### Memory Management
- Gradient accumulation for large batch sizes
- Mixed precision training (FP16) support
- Gradient checkpointing for memory efficiency

## Future Work

- [ ] Implement learning rate scheduling with warmup
- [ ] Add support for more text classification datasets
- [ ] Implement model compression techniques for edge deployment
- [ ] Add support for cross-silo federated learning
- [ ] Add support for federated learning with differential privacy
- [ ] Implement model distillation for better client-side efficiency
- [ ] Add support for federated learning with secure aggregation

## Acknowledgements

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Weights & Biases](https://wandb.ai/)
- [FedML](https://fedml.ai/) for federated learning inspiration

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## TODO
- [ ] Support another model, especially lightweight ones for cross-device FL setting. (e.g., [`EdgeNeXt`](https://github.com/mmaaz60/EdgeNeXt))
- [ ] Support another structured dataset including temporal and tabular data, along with datasets suitable for cross-silo FL setting. (e.g., [`MedMNIST`](https://github.com/MedMNIST/MedMNIST))
- [ ] Add other popular FL algorithms including personalized FL algorithms (e.g., [`SuPerFed`](https://arxiv.org/abs/2109.07628)).
- [ ] Attach benchmark results of sample commands.

## Contact
Should you have any feedback, please create a thread in __issue__ tab. Thank you :)
