import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from importlib import import_module
from src import Range, set_logger, TensorBoardRunner, check_args, set_seed
from src.models.distilbart import DistilBART
from src.loaders.data import load_dataset
from transformers import BartForSequenceClassification, BartTokenizer
import argparse
import time

def map_bart_to_distilbart(bart_state, distilbart_model):
    """Map BART-base weights to DistilBART (first 6 layers)."""
    distilbart_state = distilbart_model.state_dict()
    for key in distilbart_state.keys():
        if key in bart_state and distilbart_state[key].shape == bart_state[key].shape:
            distilbart_state[key].copy_(bart_state[key])
    distilbart_model.load_state_dict(distilbart_state)
    return distilbart_model

def map_distilbart_to_bart(distilbart_state, bart_state):
    """Map DistilBART weights back to BART-base."""
    for key in distilbart_state.keys():
        if key in bart_state and bart_state[key].shape == distilbart_state[key].shape:
            bart_state[key].copy_(distilbart_state[key])
    return bart_state

def main(args):
    # Set seed
    set_seed(args.seed)

    # Initialize logger and TensorBoard
    curr_time = time.strftime("%y%m%d_%H%M%S", time.localtime())
    args.result_path = os.path.join(args.result_path, f'{args.exp_name}_{curr_time}')
    os.makedirs(args.result_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    set_logger(f'{args.log_path}/{args.exp_name}_{curr_time}.log', args)
    writer = SummaryWriter(log_dir=os.path.join(args.log_path, f'{args.exp_name}_{curr_time}'))
    tb = TensorBoardRunner(args.log_path, args.tb_host, args.tb_port) if args.use_tb else None

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load dataset
    if args.task == "classification":
        server_dataset, client_datasets = load_dataset(args)  # e.g., 20 Newsgroups
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    else:  # summarization
        client_datasets, server_dataset, tokenizer = load_dataset(
            args.data_path, num_clients=args.K, test_size=args.test_size, random_state=args.seed
        )

    # Initialize global BART-base model
    if args.task == "classification":
        global_model = BartForSequenceClassification.from_pretrained("facebook/bart-base", num_labels=20).to(device)
    else:  # summarization
        global_model = DistilBART(use_pt_model=True).to(device)  
    global_state = global_model.state_dict()

    # Federated learning loop
    for curr_round in range(1, args.R + 1):
        print(f"\nRound {curr_round}/{args.R}")
        selected_ids = np.random.choice(args.K, size=max(1, int(args.K * args.C)), replace=False)
        client_models = []
        client_sizes = []

        # Client update phase
        for client_idx in selected_ids:
            print(f"Training client {client_idx}...")
            local_model = DistilBART(use_pt_model=True).to(device)
            local_model = map_bart_to_distilbart(global_state, local_model)

            train_loader = DataLoader(
                client_datasets[client_idx], batch_size=args.B, shuffle=not args.no_shuffle,
                num_workers=2, pin_memory=False
            )
            optimizer = AdamW(local_model.parameters(), lr=args.lr)
            local_model.train()

            for epoch in range(args.E):
                total_loss = 0
                for batch in tqdm(train_loader, desc=f"Client {client_idx} Epoch {epoch+1}"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = local_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    optimizer.zero_grad()
                    loss.backward()
                    clip_grad_norm_(local_model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                print(f"Client {client_idx} - Epoch {epoch+1}: Loss: {avg_loss:.4f}")

            client_models.append(local_model.state_dict())
            client_sizes.append(len(client_datasets[client_idx]))

        # Aggregate updates (FedAvg)
        total_size = sum(client_sizes)
        for key in global_state.keys():
            global_state[key] = torch.zeros_like(global_state[key])
            for i, client_state in enumerate(client_models):
                weight = client_sizes[i] / total_size
                if key in client_state and global_state[key].shape == client_state[key].shape:
                    global_state[key] += weight * client_state[key]

        global_model.load_state_dict(global_state)

        # Evaluate
        if curr_round % args.eval_every == 0 or curr_round == args.R:
            global_model.eval()
            test_loader = DataLoader(server_dataset, batch_size=args.B, shuffle=False, num_workers=2, pin_memory=False)
            with torch.no_grad():
                if args.task == "classification":
                    correct, total = 0, 0
                    for batch in test_loader:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)
                        outputs = global_model(input_ids=input_ids, attention_mask=attention_mask)
                        _, predicted = torch.max(outputs.logits, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    accuracy = correct / total
                    print(f"Round {curr_round} - Test Accuracy: {accuracy:.4f}")
                else:  # summarization
                    for i, batch in enumerate(test_loader):
                        if i >= 2: break
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)
                        summary_ids = global_model.generate(
                            input_ids=input_ids, attention_mask=attention_mask, max_length=128, num_beams=4
                        )
                        article = tokenizer.decode(input_ids[0], skip_special_tokens=True)[:500]
                        reference = tokenizer.decode(labels[0][labels[0] != -100], skip_special_tokens=True)
                        generated = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                        print(f"\nExample {i+1}\nARTICLE: {article}...\nREFERENCE: {reference}\nGENERATED: {generated}")

            global_model.train()

        # Save checkpoint
        torch.save({
            'round': curr_round,
            'model_state_dict': global_state,
            'tokenizer': tokenizer
        }, os.path.join(args.result_path, f'model_round_{curr_round}.pt'))

    if args.use_tb:
        tb.finalize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--log_path", type=str, default="./log")
    parser.add_argument("--result_path", type=str, default="./result")
    parser.add_argument("--use_tb", action="store_true")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--task", type=str, default="classification")
    parser.add_argument("--split_type", type=str, default="diri")
    parser.add_argument("--cncntrtn", type=float, default=0.5)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--R", type=int, default=5)
    parser.add_argument("--C", type=float, default=0.5)
    parser.add_argument("--E", type=int, default=1)
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_type", type=str, default="global")
    parser.add_argument("--eval_metrics", nargs="+", default=["acc1", "f1", "precision", "recall"])
    parser.add_argument("--use_model_tokenizer", action="store_true", help="Use model's tokenizer")
    parser.add_argument("--use_pt_model", action="store_true", help="Use pretrained model")
    parser.add_argument("--tb_host", type=str, default="localhost", help="TensorBoard host")
    parser.add_argument("--tb_port", type=int, default=6006, help="TensorBoard port")
    parser.add_argument("--model_name", type=str, default="DistilBert", help="Model name (e.g., DistilBART, Bart)")
    args = parser.parse_args()
    main(args)