
# Federated Learning in PyTorch
Implementations of various Federated Learning (FL) algorithms in PyTorch, especially for research purposes.

# Federated BERT for Text Classification

This repository contains an implementation of Federated Learning with BERT for text classification tasks, specifically optimized for the 20 Newsgroups dataset. The implementation includes support for non-IID data distribution across clients using Dirichlet distribution.

## Features

- Federated Learning with BERT-base-uncased
- Support for 20 Newsgroups text classification
- Non-IID data partitioning using Dirichlet distribution
- Client-side model training with local updates
- Centralized model aggregation (FedAvg)
- Comprehensive evaluation metrics (accuracy, F1, precision, recall)
- Progress tracking with tqdm
- GPU acceleration support

## Results

### 20 Newsgroups Classification
- **Dataset**: 20 Newsgroups (20 classes)
- **Training Samples**: 11,314
- **Test Samples**: 7,532
- **Model**: BERT-base-uncased
- **Clients**: 3
- **Rounds**: 20
- **Local Epochs**: 2
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Alpha (Dirichlet)**: 0.5

### Final Evaluation Metrics (After 20 Rounds)
```
Test Loss: 1.9797, Accuracy: 69.32%
F1 Score: 0.7000, Precision: 0.7187, Recall: 0.6932
```

### Performance Improvement (5 Rounds → 20 Rounds)
| Metric     | 5 Rounds | 20 Rounds | Improvement |
|------------|----------|-----------|-------------|
| Accuracy   | 67.62%   | 69.32%    | +1.70%      |
| F1 Score   | 0.6836   | 0.7000    | +0.0164     |
| Precision  | 0.7058   | 0.7187    | +0.0129     |
| Recall     | 0.6762   | 0.6932    | +0.0170     |

### Performance by Round
| Round | Test Accuracy | F1 Score | Precision | Recall |
|-------|---------------|----------|-----------|--------|
| 1     | 8.27%         | 0.0377   | 0.1095    | 0.0827 |
| 2     | 9.63%         | 0.0596   | 0.1778    | 0.0963 |
| 3     | 65.29%        | 0.6578   | 0.6938    | 0.6529 |
| 4     | 67.64%        | 0.6832   | 0.7015    | 0.6764 |
| 5     | 67.62%        | 0.6836   | 0.7058    | 0.6762 |
| ...   | ...           | ...      | ...       | ...    |
| 20    | 69.32%        | 0.7000   | 0.7187    | 0.6932 |

### Per-Class Metrics (After 20 Rounds)
| Class | F1     | Precision | Recall   | Improvement (F1) |
|-------|--------|-----------|----------|------------------|
| 0     | 0.4717 | 0.5924    | 0.3918   | +0.0046          |
| 1     | 0.6899 | 0.6292    | 0.7635   | +0.0273          |
| 2     | 0.6262 | 0.6000    | 0.6548   | -0.0091          |
| 3     | 0.6716 | 0.6507    | 0.6939   | +0.0572          |
| 4     | 0.7292 | 0.7311    | 0.7273   | +0.1591          |
| 5     | 0.7945 | 0.8727    | 0.7291   | +0.0133          |
| 6     | 0.8295 | 0.8704    | 0.7923   | +0.0517          |
| 7     | 0.6257 | 0.5461    | 0.7323   | -0.1047          |
| 8     | 0.7423 | 0.7920    | 0.6985   | +0.0611          |
| 9     | 0.8595 | 0.9271    | 0.8010   | +0.0039          |
| 10    | 0.8721 | 0.8483    | 0.8972   | +0.0019          |
| 11    | 0.7208 | 0.7493    | 0.6944   | +0.0077          |
| 12    | 0.6138 | 0.6277    | 0.6005   | +0.0179          |
| 13    | 0.8360 | 0.8937    | 0.7854   | +0.0225          |
| 14    | 0.7808 | 0.8482    | 0.7234   | +0.0024          |
| 15    | 0.6705 | 0.7845    | 0.5854   | -0.0313          |
| 16    | 0.5921 | 0.5356    | 0.6621   | -0.0022          |
| 17    | 0.8248 | 0.8567    | 0.7952   | +0.0350          |
| 18    | 0.4734 | 0.5055    | 0.4452   | -0.0051          |
| 19    | 0.3509 | 0.2653    | 0.5179   | +0.0069          |

### Key Observations
- **Best Performing Classes (F1 > 0.8)**:
  - Class 10 (0.8721) - Best overall performance
  - Class 9 (0.8595)
  - Class 13 (0.8360)
  - Class 17 (0.8248)
  - Class 6 (0.8295) - New addition to high performers
  - Class 14 (0.7808)
- **Most Improved Classes (F1 increase > 0.05)**:
  - Class 4: +0.1591 (0.5701 → 0.7292)
  - Class 3: +0.0572 (0.6144 → 0.6716)
  - Class 8: +0.0611 (0.6812 → 0.7423)
  - Class 6: +0.0517 (0.7778 → 0.8295)
- **Classes Needing Attention (F1 < 0.5)**:
  - Class 0 (0.4717) - Slight improvement (+0.0046)
  - Class 18 (0.4734) - Minor regression (-0.0051)
  - Class 19 (0.3509) - Small improvement (+0.0069)
- **Areas for Improvement**:
  - Class 7 showed a significant drop in performance (-0.1047)
  - Class 15 also decreased (-0.0313)
  - Consider class imbalance and potential overfitting for these classes

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
git clone https://github.com/yourusername/federated-bert.git
cd federated-bert
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

### Usage

#### Training

To train the federated BERT model on the 20 Newsgroups dataset:

```bash
python run_bert_20news.py \
    --num_clients 3 \
    --num_rounds 5 \
    --local_epochs 2 \
    --batch_size 16 \
    --max_length 128 \
    --learning_rate 2e-5 \
    --seed 42 \
    --test_batch_size 32 \
    --alpha 0.5
```

#### Arguments

- `--num_clients`: Number of clients in federated learning
- `--num_rounds`: Number of federated learning rounds
- `--local_epochs`: Number of local training epochs per round
- `--batch_size`: Training batch size
- `--max_length`: Maximum sequence length for BERT
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

## Performance Optimization

### Class Imbalance
- The current implementation shows varying performance across different classes
- Consider implementing class weights in the loss function
- Potential improvements through oversampling minority classes or undersampling majority classes

### Hyperparameter Tuning
- Experiment with different learning rates and scheduling strategies
- Try different batch sizes based on available GPU memory
- Adjust the number of local epochs and federated rounds

### Model Architecture
- Current implementation uses BERT-base-uncased with a custom classifier head
- Potential to experiment with different BERT variants (e.g., DistilBERT, RoBERTa)
- Consider adding additional regularization techniques

## Future Work

- [ ] Implement class weighting for imbalanced datasets
- [ ] Add learning rate scheduling with warmup
- [ ] Support for more BERT variants and architectures
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
- [ ] Support another model, especially lightweight ones for cross-device FL setting. (e.g., [`EdgeNeXt`](https://github.com/mmaaz60/EdgeNeXt))
- [ ] Support another structured dataset including temporal and tabular data, along with datasets suitable for cross-silo FL setting. (e.g., [`MedMNIST`](https://github.com/MedMNIST/MedMNIST))
- [ ] Add other popular FL algorithms including personalized FL algorithms (e.g., [`SuPerFed`](https://arxiv.org/abs/2109.07628)).
- [ ] Attach benchmark results of sample commands.

## Contact
Should you have any feedback, please create a thread in __issue__ tab. Thank you :)
