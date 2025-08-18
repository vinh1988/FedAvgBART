#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Centralized fine-tuning on CNN/DailyMail with DistilBART and BART.
- Trains each model for 5 epochs (configurable)
- Computes comprehensive summarization metrics per epoch and on test set:
  ROUGE-1/2/L/Lsum, BLEU, METEOR, chrF, optional BERTScore
- Saves metrics to CSV files per model for paper-ready reporting

Usage example:
    python train_cnndm_centralized.py \
        --output_root ./results_cnndm_centralized \
        --epochs 5 --batch_size 4 --eval_batch_size 4 --use_bertscore

"""

import os
import sys
import math
import argparse
import random
import numpy as np
import pandas as pd
import torch
import tempfile

from dataclasses import dataclass
from typing import Dict, List, Optional

from datasets import load_dataset
import evaluate
import shutil
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    return v.lower() in {"1", "true", "t", "yes", "y"}


@dataclass
class ModelSpec:
    name: str
    hub_id: str
    csv_path: str
    output_dir: str


def prepare_dataset(
    tokenizer: AutoTokenizer,
    max_source_len: int,
    max_target_len: int,
    num_proc: Optional[int] = None,
    limit_train: Optional[int] = None,
    limit_eval: Optional[int] = None,
    limit_test: Optional[int] = None,
):
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    def preprocess(batch):
        inputs = batch["article"]
        targets = batch["highlights"]
        model_inputs = tokenizer(
            inputs,
            max_length=max_source_len,
            truncation=True,
            padding=False,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=max_target_len,
                truncation=True,
                padding=False,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    num_proc = num_proc or os.cpu_count() or 1
    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=num_proc,
    )

    # Optionally limit split sizes for quick smoke tests
    if limit_train is not None and limit_train > 0:
        limit = min(limit_train, len(tokenized["train"]))
        tokenized["train"] = tokenized["train"].select(range(limit))
    if "validation" in tokenized and limit_eval is not None and limit_eval > 0:
        limit = min(limit_eval, len(tokenized["validation"]))
        tokenized["validation"] = tokenized["validation"].select(range(limit))
    if "test" in tokenized and limit_test is not None and limit_test > 0:
        limit = min(limit_test, len(tokenized["test"]))
        tokenized["test"] = tokenized["test"].select(range(limit))

    return tokenized


def build_metrics(use_bertscore: bool):
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("sacrebleu")
    meteor = evaluate.load("meteor")
    chrf = evaluate.load("chrf")
    bertscore = evaluate.load("bertscore") if use_bertscore else None

    def compute(preds: List[str], labels: List[str]) -> Dict[str, float]:
        # ROUGE
        r = rouge.compute(predictions=preds, references=labels, use_aggregator=True)
        # BLEU (expects list of references per prediction)
        b = bleu.compute(predictions=preds, references=[[l] for l in labels])
        # METEOR
        m = meteor.compute(predictions=preds, references=labels)
        # chrF
        c = chrf.compute(predictions=preds, references=labels)
        out = {
            "rouge1": r.get("rouge1", float("nan")),
            "rouge2": r.get("rouge2", float("nan")),
            "rougeL": r.get("rougeL", float("nan")),
            "rougeLsum": r.get("rougeLsum", float("nan")),
            "bleu": b.get("score", float("nan")),
            "meteor": m.get("meteor", float("nan")),
            "chrf": c.get("score", float("nan")),
        }
        if bertscore is not None:
            bs = bertscore.compute(predictions=preds, references=labels, lang="en")
            out.update({
                "bertscore_precision": float(np.mean(bs["precision"])) if len(bs["precision"]) else float("nan"),
                "bertscore_recall": float(np.mean(bs["recall"])) if len(bs["recall"]) else float("nan"),
                "bertscore_f1": float(np.mean(bs["f1"])) if len(bs["f1"]) else float("nan"),
            })
        return out

    return compute


def run_one_model(spec: ModelSpec, args: argparse.Namespace, metrics_rows: List[Dict]):
    print(f"\n===== Training {spec.name}: {spec.hub_id} =====")
    tokenizer = AutoTokenizer.from_pretrained(spec.hub_id, use_fast=True)
    # Prefer safetensors and ensure CPU-friendly dtype; avoid torch.load vulnerability by using safetensors
    model = AutoModelForSeq2SeqLM.from_pretrained(
        spec.hub_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    if args.force_cpu:
        model.to("cpu")

    tokenized = prepare_dataset(
        tokenizer,
        args.max_source_length,
        args.max_target_length,
        args.num_proc,
        limit_train=args.limit_train_samples,
        limit_eval=args.limit_eval_samples,
        limit_test=args.limit_test_samples,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    metric_fn = build_metrics(args.use_bertscore)

    training_args = Seq2SeqTrainingArguments(
        output_dir=spec.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=args.grad_accum,
        # evaluation_strategy not supported in older transformers; we'll evaluate manually after training
        # Disable intermediate checkpoints on older transformers by pushing save_steps very high
        save_steps=1000000000,
        save_total_limit=0,
        logging_steps=args.logging_steps,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        generation_num_beams=args.num_beams,
        dataloader_num_workers=max(1, (os.cpu_count() or 2) // 2),
        fp16=(torch.cuda.is_available() and (not args.force_cpu)),
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        # Decode
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in labels as tokenizer pad token id
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Strip
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]
        return metric_fn(decoded_preds, decoded_labels)

    # Helper to build rows consistently (defined before training/eval loop)
    def pull_metrics(prefix: str, m: Dict[str, float], epoch: Optional[int]):
        row = {
            "model": spec.name,
            "hub_id": spec.hub_id,
            "epoch": epoch if epoch is not None else prefix,
            # core metrics
            "rouge1": m.get("eval_rouge1", m.get("rouge1", float("nan"))),
            "rouge2": m.get("eval_rouge2", m.get("rouge2", float("nan"))),
            "rougeL": m.get("eval_rougeL", m.get("rougeL", float("nan"))),
            "rougeLsum": m.get("eval_rougeLsum", m.get("rougeLsum", float("nan"))),
            "bleu": m.get("eval_bleu", m.get("bleu", float("nan"))),
            "meteor": m.get("eval_meteor", m.get("meteor", float("nan"))),
            "chrf": m.get("eval_chrf", m.get("chrf", float("nan"))),
            "bertscore_precision": m.get("eval_bertscore_precision", m.get("bertscore_precision", float("nan"))),
            "bertscore_recall": m.get("eval_bertscore_recall", m.get("bertscore_recall", float("nan"))),
            "bertscore_f1": m.get("eval_bertscore_f1", m.get("bertscore_f1", float("nan"))),
            # losses
            "loss": m.get("eval_loss", m.get("loss", float("nan"))),
        }
        # derive perplexity if loss is available and finite
        loss_val = row.get("loss")
        try:
            if loss_val is not None and not (math.isinf(loss_val) or math.isnan(loss_val)):
                row["perplexity"] = float(math.exp(loss_val))
            else:
                row["perplexity"] = float("nan")
        except Exception:
            row["perplexity"] = float("nan")
        return row

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Manual epoch-by-epoch training with evaluation at the end of each epoch
    eval_metrics = None
    for ep in range(1, int(args.epochs) + 1):
        # Increase target epoch and continue training
        trainer.args.num_train_epochs = ep
        trainer.train()
        # Evaluate at the end of this epoch
        eval_metrics = trainer.evaluate(eval_dataset=tokenized["validation"], max_length=args.max_target_length, num_beams=args.num_beams)
        metrics_rows.append(pull_metrics("val", eval_metrics, ep))

    # Test metrics after the final epoch
    test_metrics = trainer.evaluate(eval_dataset=tokenized["test"], max_length=args.max_target_length, num_beams=args.num_beams)

    # Build rows

    # Append a clearly labeled final validation row
    metrics_rows.append(pull_metrics("val", eval_metrics or {}, None))
    metrics_rows.append(pull_metrics("test", test_metrics, None))

    # Skip saving final model to avoid any checkpoint artifacts; focus on CSV outputs only
    # trainer.save_model(spec.output_dir)

    # Persist CSV
    df = pd.DataFrame(metrics_rows)
    os.makedirs(os.path.dirname(spec.csv_path), exist_ok=True)
    df.to_csv(spec.csv_path, index=False)
    print(f"Saved metrics CSV: {spec.csv_path}")

    # Optionally remove all training artifacts (events, state, etc.)
    if args.no_artifacts:
        try:
            shutil.rmtree(spec.output_dir, ignore_errors=True)
            print(f"Removed training artifacts under: {spec.output_dir}")
        except Exception as e:
            print(f"Warning: failed to remove artifacts at {spec.output_dir}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--max_source_length", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=142)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument("--use_bertscore", type=str2bool, default=False, nargs="?", const=True)
    parser.add_argument("--output_root", type=str, default="./results_cnndm_centralized")
    parser.add_argument("--force_cpu", action="store_true", default=False, help="Force CPU execution and disable fp16")
    parser.add_argument("--no_artifacts", action="store_true", default=False, help="Do not keep any trainer artifacts (only CSV outputs)")
    # Quick-run options: limit number of samples per split (useful for smoke tests)
    parser.add_argument("--limit_train_samples", type=int, default=None, help="Limit number of training samples (e.g., 100)")
    parser.add_argument("--limit_eval_samples", type=int, default=None, help="Limit number of validation samples")
    parser.add_argument("--limit_test_samples", type=int, default=None, help="Limit number of test samples")

    args = parser.parse_args()
    set_seed(args.seed)

    # Select device
    if args.force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Disable external loggers/artifacts if requested or by default preference
    # Prevent W&B from activating implicitly
    os.environ.setdefault("WANDB_DISABLED", "true")
    # Reduce HF telemetry noise
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    device = torch.device("cuda" if (torch.cuda.is_available() and (not args.force_cpu)) else "cpu")
    print(f"Using device: {device}")

    # Define model specs
    out_root = args.output_root
    temp_dirs = []
    if args.no_artifacts:
        # Route trainer output to a temporary directory to avoid creating project-local folders
        out_root = tempfile.mkdtemp(prefix="cnndm_centralized_")
        temp_dirs.append(out_root)
    specs = [
        ModelSpec(
            name="DistilBART",
            hub_id="sshleifer/distilbart-cnn-12-6",
            csv_path=os.path.join("centralize_data_gen_text", "results_distilbart_cnndm_centralized.csv"),
            output_dir=os.path.join(out_root, "distilbart")
        ),
        ModelSpec(
            name="BART",
            hub_id="facebook/bart-large-cnn",
            csv_path=os.path.join("centralize_data_gen_text", "results_bart_cnndm_centralized.csv"),
            output_dir=os.path.join(out_root, "bart")
        ),
    ]

    all_rows: List[Dict] = []
    for spec in specs:
        rows_before = len(all_rows)
        run_one_model(spec, args, all_rows)
        # Save per-model as well
        model_rows = all_rows[rows_before:]
        os.makedirs(os.path.dirname(spec.csv_path), exist_ok=True)
        pd.DataFrame(model_rows).to_csv(spec.csv_path, index=False)

    # Save combined CSV
    combined_csv = os.path.join("centralize_data_gen_text", "results_cnndm_centralized_all.csv")
    os.makedirs(os.path.dirname(combined_csv), exist_ok=True)
    pd.DataFrame(all_rows).to_csv(combined_csv, index=False)
    print(f"Saved combined metrics CSV: {combined_csv}")

    # Cleanup any temporary roots used for artifacts
    if args.no_artifacts:
        try:
            for d in temp_dirs:
                shutil.rmtree(d, ignore_errors=True)
                print(f"Removed temp artifact root: {d}")
        except Exception as e:
            print(f"Warning: failed to remove temp roots: {e}")


if __name__ == "__main__":
    main()
