import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.distilbart_gen import DistilBARTGen
from src.datasets.cnndm import load_cnndm

def train():
    # Configuration
    num_clients = 3
    num_rounds = 1
    epochs_per_client = 1
    batch_size = 2  # Reduced batch size to save memory
    learning_rate = 5e-5
    max_grad_norm = 1.0
    data_dir = "./data/cnndm"
    model_save_path = "./saved_models/distilbart_cnndm"
    
    # Limit the number of samples for faster training and less memory usage
    max_train_samples_per_client = 1000  # Adjust based on your system's memory
    max_test_samples_per_client = 200
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data with limited samples
    print("Loading CNN/DailyMail dataset with limited samples...")
    train_datasets, test_datasets, tokenizer = load_cnndm(
        data_dir, 
        num_clients=num_clients, 
        test_size=0.1, 
        random_state=42,
        max_train_samples=max_train_samples_per_client * num_clients,
        max_test_samples=max_test_samples_per_client * num_clients
    )
    
    # Initialize global model
    print("Initializing DistilBART model...")
    model = DistilBARTGen(use_pt_model=True).to(device)
    
    # Federated training loop
    print("Starting federated training...")
    for round_num in range(1, num_rounds + 1):
        print(f"\nRound {round_num}/{num_rounds}")
        
        # Randomly select clients for this round (50% participation rate)
        selected_clients = np.random.choice(
            num_clients, 
            size=max(1, int(num_clients * 0.5)),
            replace=False
        )
        
        # Client update phase
        client_models = []
        client_sizes = []
        
        for client_idx in selected_clients:
            print(f"\nTraining client {client_idx}...")
            
            # Create local model copy
            local_model = DistilBARTGen(use_pt_model=True).to(device)
            local_model.load_state_dict(model.state_dict())
            
            # Get client data with reduced num_workers and pin_memory=False to save memory
            train_loader = DataLoader(
                train_datasets[client_idx],
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,  # Reduced number of workers
                pin_memory=False,  # Disable pin_memory to save GPU memory
                persistent_workers=False  # Avoid keeping worker processes alive
            )
            
            # Local training
            optimizer = AdamW(local_model.parameters(), lr=learning_rate)
            local_model.train()
            
            for epoch in range(epochs_per_client):
                total_loss = 0
                
                for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs_per_client}"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    # Forward pass
                    outputs = local_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    
                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    clip_grad_norm_(local_model.parameters(), max_grad_norm)
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                # Print epoch statistics
                avg_loss = total_loss / len(train_loader)
                print(f"Client {client_idx} - Epoch {epoch+1}: Loss: {avg_loss:.4f}")
            
            # Store updated model and data size
            client_models.append(local_model.state_dict())
            client_sizes.append(len(train_datasets[client_idx]))
        
        # Aggregate model updates (Federated Averaging)
        print("\nAggregating model updates...")
        global_state = model.state_dict()
        total_size = sum(client_sizes)
        
        # Initialize averaged model parameters
        for key in global_state.keys():
            global_state[key] = torch.zeros_like(global_state[key])
            
            # Weighted average of client models
            for i, client_state in enumerate(client_models):
                weight = client_sizes[i] / total_size
                global_state[key] += weight * client_state[key]
        
        # Update global model
        model.load_state_dict(global_state)
        
        # Evaluate global model on test set
        print("\nEvaluating global model...")
        model.eval()
        
        # Evaluate on a small subset of the test set for efficiency
        test_loader = DataLoader(
            test_datasets[0],  # Just use first test client for evaluation
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,  # Reduced number of workers
            pin_memory=False,  # Disable pin_memory to save GPU memory
            persistent_workers=False  # Avoid keeping worker processes alive
        )
        
        # Evaluate on a few examples
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= 2:  # Just show 2 examples
                    break
                    
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Generate summary
                summary_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True
                )
                
                # Decode and print results (with truncation for readability)
                article = tokenizer.decode(input_ids[0], skip_special_tokens=True)[:500]  # Truncate for display
                reference = tokenizer.decode(
                    labels[0][labels[0] != -100],  # Remove -100 labels
                    skip_special_tokens=True
                )
                generated = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                
                print("\nExample", i+1)
                print("-" * 50)
                print("ARTICLE:", article[:500] + "...")
                print("\nREFERENCE SUMMARY:", reference)
                print("\nGENERATED SUMMARY:", generated)
                print("-" * 50)
        
        # Save model checkpoint
        os.makedirs(model_save_path, exist_ok=True)
        torch.save({
            'round': round_num,
            'model_state_dict': model.state_dict(),
            'tokenizer': tokenizer,
        }, os.path.join(model_save_path, f'model_round_{round_num}.pt'))
        
        print(f"\nModel checkpoint saved to {model_save_path}")
        
        # Set back to training mode
        model.train()

if __name__ == "__main__":
    train()
