import os
import sys
import argparse
from datetime import datetime
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
import wandb
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import BartForSequenceClassification
from src.datasets.news20 import load_20newsgroups


def parse_args():
    parser = argparse.ArgumentParser(description="Federated BART-Large on 20 Newsgroups")
    parser.add_argument("--num_clients", type=int, default=10, help="Number of clients")
    parser.add_argument("--num_rounds", type=int, default=22, help="Federated rounds")
    parser.add_argument("--participation_rate", type=float, default=0.5,
                        help="Fraction of clients selected per round (0 < r <= 1)")
    parser.add_argument("--min_clients_per_round", type=int, default=1,
                        help="Minimum clients selected per round (use 2 for small-N)")
    parser.add_argument("--dirichlet_alpha", type=float, default=None,
                        help="Dirichlet concentration for non-IID split; None means IID random split")
    parser.add_argument("--dirichlet_min_size", type=int, default=10,
                        help="Minimum samples per client in Dirichlet partition")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length for tokenization")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (BART-large is memory heavy)")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs_per_client", type=int, default=1, help="Local epochs per client per round")
    parser.add_argument("--data_dir", type=str, default="./data/20newsgroups",
                        help="Directory to store/load the dataset")
    parser.add_argument("--output_dir", type=str, default="results_bart_large_fed_runs_20news",
                        help="Base directory to store run artifacts (metrics, checkpoints, logs)")
    return parser.parse_args()


def train(args):
    # Configuration for wandb
    config = {
        'num_clients': args.num_clients,
        'num_rounds': args.num_rounds,
        'participation_rate': args.participation_rate,
        'min_clients_per_round': args.min_clients_per_round,
        'dirichlet_alpha': args.dirichlet_alpha,
        'dirichlet_min_size': args.dirichlet_min_size,
        'epochs_per_client': args.epochs_per_client,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_seq_length': args.max_length,
        'model_name': 'facebook/bart-large',
        'project_name': 'federated-bart-large-20news',
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
    }

    wandb.init(
        project=config['project_name'],
        name=f"fed-bart-large-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config=config,
    )
    config = wandb.config

    # Unpack
    num_clients = int(config['num_clients'])
    num_rounds = int(config['num_rounds'])
    participation_rate = float(config['participation_rate'])
    min_clients_per_round = int(config['min_clients_per_round'])
    dirichlet_alpha = config.get('dirichlet_alpha', None)
    dirichlet_min_size = int(config.get('dirichlet_min_size', 10))
    epochs_per_client = int(config['epochs_per_client'])
    batch_size = int(config['batch_size'])
    learning_rate = float(config['learning_rate'])
    max_seq_length = int(config['max_seq_length'])
    data_dir = config['data_dir']

    # Prepare output dirs
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"clients_{num_clients}_rounds_{num_rounds}", timestamp)
    os.makedirs(run_dir, exist_ok=True)
    results_file = os.path.join(run_dir, f"training_metrics_{timestamp}.csv")

    metrics_columns = ['round', 'client_id', 'epoch', 'phase', 'loss', 'accuracy', 'precision', 'recall', 'f1']
    metrics_data = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    print("Loading 20 Newsgroups dataset...")
    train_datasets, test_datasets, num_classes, _ = load_20newsgroups(
        data_dir,
        num_clients=num_clients,
        test_size=0.2,
        random_state=42,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_min_size=dirichlet_min_size,
        tokenizer_name='facebook/bart-large',
        max_length=max_seq_length,
    )

    # ------------------------------
    # Federated data distribution logging (train split)
    # ------------------------------
    # Compute per-client class counts and proportions
    cls_cols = [f'class_{i}' for i in range(num_classes)]
    prop_cols = [f'prop_{i}' for i in range(num_classes)]
    client_rows = []
    # Global distribution from concatenated train shards
    all_train_labels = []
    base_ds = train_datasets[0].dataset  # underlying News20Dataset
    for i in range(num_clients):
        idxs = train_datasets[i].indices
        lbs = [int(base_ds.labels[j]) for j in idxs]
        all_train_labels.extend(lbs)
    global_counts = np.bincount(all_train_labels, minlength=num_classes)
    global_props = (global_counts / max(1, global_counts.sum())).astype(float)

    for i in range(num_clients):
        idxs = train_datasets[i].indices
        lbs = [int(base_ds.labels[j]) for j in idxs]
        counts = np.bincount(lbs, minlength=num_classes)
        props = (counts / max(1, counts.sum())).astype(float)
        # KL(client || global)
        eps = 1e-8
        p = np.clip(props, eps, 1.0)
        q = np.clip(global_props, eps, 1.0)
        kl = float(np.sum(p * np.log(p / q)))
        row = {
            'client_id': i,
            'num_samples': int(counts.sum()),
            'kl_to_global': kl,
        }
        row.update({c: int(v) for c, v in zip(cls_cols, counts)})
        row.update({c: float(v) for c, v in zip(prop_cols, props)})
        client_rows.append(row)

    dist_df = pd.DataFrame(client_rows)
    dist_df['dirichlet_alpha'] = dirichlet_alpha if dirichlet_alpha is not None else 'IID'
    dist_csv = os.path.join(run_dir, 'data_distribution_train.csv')
    dist_df.to_csv(dist_csv, index=False)

    # Heatmap of proportions (clients x classes)
    heatmap_data = dist_df[[f'prop_{i}' for i in range(num_classes)]].values
    plt.figure(figsize=(min(18, 2 + num_classes * 0.6), min(12, 2 + num_clients * 0.5)))
    sns.heatmap(heatmap_data, annot=False, cmap='viridis')
    plt.xlabel('Class')
    plt.ylabel('Client')
    plt.title(f'Train distribution per client (Dirichlet alpha={dirichlet_alpha if dirichlet_alpha is not None else "IID"})')
    heatmap_path = os.path.join(run_dir, 'data_distribution_heatmap.png')
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=200)
    plt.close()
    wandb.log({'data_distribution/heatmap': wandb.Image(heatmap_path)})

    # Global model
    print("Initializing BART-large model...")
    model = BartForSequenceClassification.from_pretrained(
        'facebook/bart-large', num_labels=num_classes, use_safetensors=True
    ).to(device)

    criterion = CrossEntropyLoss()

    print("Starting federated training...")
    participation_rows = []
    # Precompute comms (bytes per full model sync, float32 up+down)
    def count_params_bytes(mdl: torch.nn.Module):
        n_params = sum(p.numel() for p in mdl.parameters())
        return n_params, int(n_params * 4 * 2)  # up+down

    for round_num in range(1, num_rounds + 1):
        round_start = time.time()
        print(f"\nRound {round_num}/{num_rounds}")

        # Sample clients
        k = int(np.ceil(num_clients * participation_rate))
        k = max(min_clients_per_round, min(k, num_clients))
        selected_clients = np.random.choice(num_clients, size=k, replace=False)
        for cid in range(num_clients):
            participation_rows.append({'round': round_num, 'client_id': cid, 'selected': int(cid in selected_clients)})

        # Local training per selected client
        client_states = []
        client_sizes = []
        for client_idx in selected_clients:
            print(f"\nTraining client {client_idx}...")

            local_model = BartForSequenceClassification.from_pretrained(
                'facebook/bart-large', num_labels=num_classes, use_safetensors=True
            ).to(device)
            local_model.load_state_dict(model.state_dict())

            train_loader = DataLoader(
                train_datasets[client_idx],
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )

            optimizer = AdamW(local_model.parameters(), lr=learning_rate)
            local_model.train()

            for epoch in range(epochs_per_client):
                total_loss = 0.0
                correct = 0
                total = 0
                epoch_seen_labels = []
                for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs_per_client}"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = local_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    _, predicted = torch.max(logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    epoch_seen_labels.extend(labels.detach().cpu().numpy().tolist())

                avg_loss = total_loss / max(1, len(train_loader))
                accuracy = correct / max(1, total)

                # Compute PRF1 on training set quickly (using same loader)
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for batch in train_loader:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)
                        out = local_model(input_ids=input_ids, attention_mask=attention_mask)
                        _, pred = torch.max(out.logits, 1)
                        all_preds.extend(pred.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)

                # Label distribution seen this epoch (actual training batches)
                if epoch_seen_labels:
                    counts = np.bincount(np.array(epoch_seen_labels), minlength=num_classes)
                    props = counts / max(1, counts.sum())
                    dist_row = {
                        'round': round_num,
                        'client_id': int(client_idx),
                        'epoch': int(epoch + 1),
                        'num_samples': int(counts.sum()),
                    }
                    dist_row.update({f'class_{i}': int(counts[i]) for i in range(num_classes)})
                    dist_row.update({f'prop_{i}': float(props[i]) for i in range(num_classes)})
                    dist_df_ep = pd.DataFrame([dist_row])
                    dist_ep_csv = os.path.join(run_dir, f'label_distribution_train_round_{round_num}_client_{client_idx}_epoch_{epoch+1}.csv')
                    dist_df_ep.to_csv(dist_ep_csv, index=False)

                    # Bar plot of proportions for this epoch
                    plt.figure(figsize=(10, 3))
                    plt.bar(range(num_classes), props)
                    plt.xlabel('Class'); plt.ylabel('Proportion')
                    plt.title(f'Client {client_idx} Epoch {epoch+1} label distribution (round {round_num})')
                    bar_png = os.path.join(run_dir, f'label_distribution_train_round_{round_num}_client_{client_idx}_epoch_{epoch+1}.png')
                    plt.tight_layout(); plt.savefig(bar_png, dpi=200); plt.close()
                    wandb.log({
                        'train/label_distribution_image': wandb.Image(bar_png),
                        'train/label_num_samples': int(counts.sum()),
                    }, step=round_num)

                wandb.log({
                    'round': round_num,
                    'client': int(client_idx),
                    'epoch': int(epoch + 1),
                    'train/loss': avg_loss,
                    'train/accuracy': accuracy,
                    'train/precision': precision,
                    'train/recall': recall,
                    'train/f1': f1,
                }, step=round_num)

                metrics_data.append({
                    'round': round_num,
                    'client_id': int(client_idx),
                    'epoch': int(epoch + 1),
                    'phase': 'train',
                    'loss': avg_loss,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                })
                print(f"Client {client_idx} - Epoch {epoch+1}: Loss {avg_loss:.4f} Acc {accuracy*100:.2f}%")

            client_states.append(local_model.state_dict())
            client_sizes.append(len(train_datasets[client_idx]))

        # FedAvg aggregation
        print("\nAggregating model updates...")
        global_state = model.state_dict()
        total_size = sum(client_sizes) if client_sizes else 1
        for kname in global_state.keys():
            global_state[kname] = torch.zeros_like(global_state[kname])
            for i, cstate in enumerate(client_states):
                weight = client_sizes[i] / total_size
                global_state[kname] += weight * cstate[kname]
        model.load_state_dict(global_state)

        # Evaluate global model across all clients' test shards
        print("\nEvaluating global model...")
        model.eval()
        all_preds, all_labels = [], []
        all_logits = []
        per_client_rows = []
        val_loss = 0.0
        with torch.no_grad():
            for client_idx in range(num_clients):
                test_loader = DataLoader(
                    test_datasets[client_idx],
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True,
                )
                c_preds, c_labels = [], []
                c_logits = []
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += out.loss.item()
                    logits = out.logits
                    _, pred = torch.max(logits, 1)
                    np_pred = pred.cpu().numpy()
                    np_labels = labels.cpu().numpy()
                    all_preds.extend(np_pred)
                    all_labels.extend(np_labels)
                    all_logits.append(logits.cpu().numpy())
                    c_preds.extend(np_pred)
                    c_labels.extend(np_labels)
                    c_logits.append(logits.cpu().numpy())

                # Per-client test metrics
                if c_labels:
                    c_acc = accuracy_score(c_labels, c_preds)
                    c_pr, c_rc, c_f1, _ = precision_recall_fscore_support(c_labels, c_preds, average='macro', zero_division=0)
                    per_client_rows.append({'round': round_num, 'client_id': client_idx, 'accuracy': c_acc, 'macro_precision': c_pr, 'macro_recall': c_rc, 'macro_f1': c_f1, 'num_samples': len(c_labels)})

        accuracy = accuracy_score(all_labels, all_preds) if all_preds else 0.0
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
        precision_mi, recall_mi, f1_mi, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro', zero_division=0)
        precision_ma, recall_ma, f1_ma, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)

        # Per-class metrics and confusion matrix
        per_class_pr, per_class_rc, per_class_f1, support = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
        cm_df = pd.DataFrame(cm, columns=[f'pred_{i}' for i in range(num_classes)], index=[f'true_{i}' for i in range(num_classes)])
        cm_csv = os.path.join(run_dir, f'confusion_matrix_round_{round_num}.csv')
        cm_df.to_csv(cm_csv)
        # Plot normalized confusion matrix heatmap
        cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, cmap='Blues', cbar=True)
        plt.title(f'Confusion Matrix (Normalized) - Round {round_num}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        cm_png = os.path.join(run_dir, f'confusion_matrix_round_{round_num}.png')
        plt.tight_layout(); plt.savefig(cm_png, dpi=200); plt.close()

        # ROC-AUC and PR-AUC (OvR). Need probabilities
        roc_auc_macro = roc_auc_micro = pr_auc_macro = pr_auc_micro = 0.0
        per_class_auc, per_class_pr_auc = [], []
        if all_logits:
            logits_all = np.concatenate(all_logits, axis=0)
            probs = torch.softmax(torch.from_numpy(logits_all), dim=1).numpy()
            y_true = np.array(all_labels)
            y_bin = label_binarize(y_true, classes=list(range(num_classes)))
            try:
                roc_auc_macro = roc_auc_score(y_bin, probs, average='macro', multi_class='ovr')
                roc_auc_micro = roc_auc_score(y_bin, probs, average='micro', multi_class='ovr')
            except Exception:
                pass
            try:
                pr_auc_macro = average_precision_score(y_bin, probs, average='macro')
                pr_auc_micro = average_precision_score(y_bin, probs, average='micro')
            except Exception:
                pass
            # Per-class AUCs
            for k in range(num_classes):
                try:
                    per_class_auc.append(roc_auc_score(y_bin[:, k], probs[:, k]))
                except Exception:
                    per_class_auc.append(float('nan'))
                try:
                    per_class_pr_auc.append(average_precision_score(y_bin[:, k], probs[:, k]))
                except Exception:
                    per_class_pr_auc.append(float('nan'))

            # Calibration: ECE/MCE and Brier score with reliability diagram
            def compute_ece_mce(p, y, n_bins=15):
                confidences = p.max(axis=1)
                preds = p.argmax(axis=1)
                accuracies = (preds == y).astype(float)
                bins = np.linspace(0.0, 1.0, n_bins + 1)
                ece = 0.0; mce = 0.0
                bin_centers = []
                bin_acc = []
                bin_conf = []
                for i in range(n_bins):
                    l, r = bins[i], bins[i + 1]
                    mask = (confidences > l) & (confidences <= r) if i > 0 else (confidences >= l) & (confidences <= r)
                    if mask.sum() == 0:
                        bin_centers.append((l + r) / 2); bin_acc.append(0.0); bin_conf.append(0.0)
                        continue
                    acc = float(accuracies[mask].mean())
                    conf = float(confidences[mask].mean())
                    gap = abs(acc - conf)
                    ece += gap * (mask.sum() / len(confidences))
                    mce = max(mce, gap)
                    bin_centers.append((l + r) / 2); bin_acc.append(acc); bin_conf.append(conf)
                return ece, mce, np.array(bin_centers), np.array(bin_acc), np.array(bin_conf)

            ece, mce, bc, ba, bcfn = compute_ece_mce(probs, y_true, n_bins=15)
            y_onehot = y_bin
            brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))

            # Reliability diagram
            plt.figure(figsize=(5, 5))
            plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
            plt.plot(bc, ba, marker='o', label='Empirical')
            plt.plot(bc, bcfn, marker='x', label='Confidence')
            plt.xlabel('Confidence'); plt.ylabel('Accuracy'); plt.title(f'Reliability (ECE={ece:.3f}) - Round {round_num}')
            plt.legend()
            rel_png = os.path.join(run_dir, f'reliability_round_{round_num}.png')
            plt.tight_layout(); plt.savefig(rel_png, dpi=200); plt.close()
            wandb.log({'calibration/reliability': wandb.Image(rel_png), 'calibration/ece': ece, 'calibration/mce': mce, 'calibration/brier': brier}, step=round_num)

        # Per-client fairness (Jain's index)
        fairness = None
        if per_client_rows:
            accs = np.array([r['accuracy'] for r in per_client_rows])
            fairness = float((accs.sum() ** 2) / (len(accs) * (accs ** 2).sum() + 1e-12)) if accs.size > 0 else None

        wandb.log({
            'round': round_num,
            'val/loss': val_loss / max(1, len(all_preds)),
            'val/accuracy': accuracy,
            'val/precision_weighted': precision_w,
            'val/recall_weighted': recall_w,
            'val/f1_weighted': f1_w,
            'val/precision_micro': precision_mi,
            'val/recall_micro': recall_mi,
            'val/f1_micro': f1_mi,
            'val/precision_macro': precision_ma,
            'val/recall_macro': recall_ma,
            'val/f1_macro': f1_ma,
            'auc/roc_macro': roc_auc_macro,
            'auc/roc_micro': roc_auc_micro,
            'auc/pr_macro': pr_auc_macro,
            'auc/pr_micro': pr_auc_micro,
            'fairness/jain_index': fairness if fairness is not None else 0.0,
        }, step=round_num)

        metrics_data.append({
            'round': round_num,
            'client_id': -1,
            'epoch': epochs_per_client,
            'phase': 'validation',
            'loss': val_loss / max(1, len(all_preds)),
            'accuracy': accuracy,
            'precision': precision_w,
            'recall': recall_w,
            'f1': f1_w,
        })

        # Save per-client metrics CSV for this round
        if per_client_rows:
            pd.DataFrame(per_client_rows).to_csv(os.path.join(run_dir, f'per_client_metrics_round_{round_num}.csv'), index=False)

        # Save per-class metrics CSV for this round
        pc_df = pd.DataFrame({
            'class': list(range(num_classes)),
            'precision': per_class_pr,
            'recall': per_class_rc,
            'f1': per_class_f1,
            'support': support,
            'roc_auc': per_class_auc if per_class_auc else [np.nan]*num_classes,
            'pr_auc': per_class_pr_auc if per_class_pr_auc else [np.nan]*num_classes,
        })
        pc_df.to_csv(os.path.join(run_dir, f'per_class_metrics_round_{round_num}.csv'), index=False)

        # Communication cost and timing
        n_params, bytes_per_sync = count_params_bytes(model)
        round_time_sec = time.time() - round_start
        wandb.log({'system/params': n_params, 'system/bytes_per_round_float32_updown': bytes_per_sync, 'system/round_time_sec': round_time_sec}, step=round_num)

        # Persist metrics CSV per round (no model checkpoint saving)
        pd.DataFrame(metrics_data, columns=metrics_columns).to_csv(results_file, index=False)

    # Final save
    metrics_df = pd.DataFrame(metrics_data)
    final_metrics_file = os.path.join(run_dir, f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    metrics_df.to_csv(final_metrics_file, index=False)
    if participation_rows:
        pd.DataFrame(participation_rows).to_csv(os.path.join(run_dir, 'participation.csv'), index=False)

    wandb.finish()
    print("Training completed. Metrics saved at:", results_file)


if __name__ == "__main__":
    args = parse_args()
    train(args)
