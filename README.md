
# Federated Learning with BERT/TinyBERT and HuggingFace Datasets

This repository provides a streamlined framework for **federated learning** using **BERT** and **TinyBERT** models with datasets from the HuggingFace `datasets` library, with a focus on text classification tasks such as AG_NEWS. It supports both full BERT models for high accuracy and TinyBERT for resource-constrained and edge devices.

---

## 🚀 Quickstart: Federated Learning on AG_NEWS

### Option 1: BERT-base (Higher Accuracy)

Run federated learning with BERT-base on AG_NEWS:

```
python main.py \
  --exp_name fedavg_bert_agnews \
  --dataset AG_NEWS \
  --split_type iid \
  --model_name bert-base-uncased \
  --algorithm fedavg \
  --eval_type both \
  --eval_metrics precision recall f1 \
  --K 10 \
  --R 5 \
  --C 0.2 \
  --E 2 \
  --B 16 \
  --optimizer AdamW \
  --lr 2e-5 \
  --criterion CrossEntropyLoss \
  --device cuda \
  --use_model_tokenizer \
  --lr_decay_step 2
```

### Option 2: TinyBERT (Resource-Efficient)

### 2. Install Requirements

Install all dependencies:
```
pip install -r requirements.txt
```

Or, for just the essentials:
```
pip install torch torchvision torchtext transformers datasets matplotlib numpy
```

### 3. Train and Validate

Run federated learning with TinyBERT on AG_NEWS and log Precision, Recall, and F1-score:

```
python main.py \
  --exp_name fedavg_tinybert_agnews_metrics \
  --dataset AG_NEWS \
  --split_type iid \
  --model_name TinyBERT \
  --algorithm fedavg \
  --eval_type both \
  --eval_metrics precision recall f1 \
  --K 10 \
  --R 5 \
  --C 0.2 \
  --E 2 \
  --B 16 \
  --optimizer Adam \
  --lr 2e-5 \
  --criterion CrossEntropyLoss \
  --device cuda \
  --use_model_tokenizer \
  --lr_decay_step 2
```

- **AG_NEWS** is automatically downloaded from HuggingFace datasets.
- **TinyBERT** is loaded from HuggingFace transformers (`huawei-noah/TinyBERT_General_4L_312D`).
- All federated learning logic is handled for you.

### 4. Visualize Metrics

After training, visualize Precision, Recall, and F1-score over rounds:
```
python commands/plot_metrics.py
```
This will generate a plot at:
```
result/fedavg_tinybert_agnews_metrics_xxxxx/metrics_over_rounds.png
```
where `xxxxx` is a timestamp.

---

## Features
- **BERT/TinyBERT**: Choose between full BERT for higher accuracy or TinyBERT for efficient federated learning.
- **HuggingFace Datasets**: Seamless integration for AG_NEWS and other NLP datasets.
- **Metrics**: Precision, Recall, F1-score tracked and visualized.
- **Edge/Resource-Constrained Ready**: Designed for practical deployment scenarios.

---

## Requirements
- Python 3.8+
- torch, torchvision, torchtext
- transformers, datasets
- matplotlib, numpy

See `requirements.txt` for details.

---

## Contact
For questions or feedback, please open an issue on GitHub.
