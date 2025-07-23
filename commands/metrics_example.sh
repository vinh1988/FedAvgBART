#!/bin/bash
# Example script to run federated learning with Precision, Recall, and F1-score metrics

# python main.py \
#   --exp_name fedavg_tinybert_agnews_metrics \
#   --dataset AG_NEWS \
#   --split_type iid \
#   --model_name TinyBERT \
#   --algorithm fedavg \
#   --eval_type both \
#   --eval_metrics precision recall f1 \
#   --K 10 \
#   --R 5 \
#   --C 0.2 \
#   --E 2 \
#   --B 16 \
#   --optimizer Adam \
#   --lr 2e-5 \
#   --criterion CrossEntropyLoss \
#   --device cuda \
#   --use_model_tokenizer \
#   --lr_decay_step 2 

# python main.py \
#   --exp_name fedavg_bertbase_agnews \
#   --dataset AG_NEWS \
#   --split_type iid \
#   --model_name BertBase \
#   --algorithm fedavg \
#   --eval_type both \
#   --eval_metrics precision recall f1 \
#   --K 10 \
#   --R 5 \
#   --C 0.2 \
#   --E 2 \
#   --B 8 \
#   --optimizer Adam \
#   --lr 2e-5 \
#   --criterion CrossEntropyLoss \
#   --device cuda \
#   --use_model_tokenizer \
#   --lr_decay_step 2


  nohup python main.py \
  --exp_name fedavg_bertbase_agnews \
  --dataset AG_NEWS \
  --split_type iid \
  --model_name BertBase \
  --algorithm fedavg \
  --eval_type both \
  --eval_metrics precision recall f1 \
  --K 10 \
  --R 5 \
  --C 0.2 \
  --E 2 \
  --B 8 \
  --optimizer Adam \
  --lr 2e-5 \
  --criterion CrossEntropyLoss \
  --device cuda \
  --use_model_tokenizer \
  --lr_decay_step 2 > out.log 2>&1 &

# ps aux | grep main.py