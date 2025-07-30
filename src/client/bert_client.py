import torch
import torch.nn as nn
from collections import Counter
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from src.models.bert_model import BERT
from src.datasets.agnews import AGNewsDataset
from src.datasets.twenty_news import TwentyNewsDataset

class BERTClient:
    def __init__(self, client_id, train_data, device, args, num_classes=20):
        self.client_id = client_id
        self.device = device
        self.args = args
        self.num_classes = num_classes
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Initialize model with the correct number of classes and dropout
        self.model = BERT(num_classes=num_classes, dropout_prob=0.1).to(device)
        
        # Initialize tokenizer
        from transformers import BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Calculate class weights for imbalanced data
        self.class_weights = self._calculate_class_weights(train_data['labels'])
        
        # Create dataset and dataloader
        if num_classes == 4:
            DatasetClass = AGNewsDataset
        else:
            DatasetClass = TwentyNewsDataset
            
        train_dataset = DatasetClass(
            train_data['texts'],
            train_data['labels'],
            self.tokenizer,
            max_length=args.max_length
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        # Optimizer and scheduler
        optimizer_args = {
            'lr': args.learning_rate,
            'eps': getattr(args, 'adam_epsilon', 1e-8)  # Default to 1e-8 if not provided
        }
        self.optimizer = AdamW(self.model.parameters(), **optimizer_args)
        
        total_steps = len(self.train_loader) * args.local_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
    
    def _calculate_class_weights(self, labels):
        """Calculate class weights based on label distribution"""
        # Count occurrences of each class
        class_counts = Counter(labels)
        total_samples = len(labels)
        
        # Calculate weights (inverse of class frequency)
        weights = [total_samples / (self.num_classes * count) for _, count in sorted(class_counts.items())]
        weights = torch.FloatTensor(weights).to(self.device)
        
        return weights
    
    def train(self, global_model):
        # Set model to training mode
        self.model.train()
        
        # Copy global model parameters to local model
        self.model.load_state_dict(global_model.state_dict())
        
        # Calculate class weights if not already done
        if not hasattr(self, 'class_weights'):
            self.class_weights = self._calculate_class_weights(
                [batch['labels'].item() for batch in self.train_loader]
            )
        
        # Define loss function with class weights
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        # Define optimizer with weight decay and learning rate scheduling
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=0.01,
            eps=1e-8,  # Epsilon for numerical stability
            betas=(0.9, 0.999)  # Beta parameters for AdamW
        )
        
        # Learning rate scheduling with warmup
        total_steps = len(self.train_loader) * self.args.local_epochs
        warmup_steps = int(0.1 * total_steps)  # 10% of training steps for warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=total_steps - warmup_steps,
            eta_min=1e-7  # Minimum learning rate
        )
        
        # Gradient accumulation steps (simulate larger batch size)
        grad_accum_steps = max(1, 32 // self.args.batch_size)
        
        # Learning rate scheduler with warmup
        total_steps = len(self.train_loader) * self.args.local_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=1e-6
        )
        
        # Training loop
        best_model_state = None
        best_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(self.args.local_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
            for i, batch in enumerate(progress_bar):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Scale loss for gradient accumulation
                loss = outputs['loss'] / grad_accum_steps
                logits = outputs['logits']
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Update weights and learning rate
                if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(self.train_loader):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Update statistics
                batch_loss = loss.item() * grad_accum_steps
                epoch_loss += batch_loss
                
                # Calculate accuracy
                _, preds = torch.max(logits, 1)
                correct = (preds == labels).sum().item()
                accuracy = correct / len(labels)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'acc': f'{accuracy*100:.2f}%',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })
                
                # Free up GPU memory
                del input_ids, attention_mask, labels, outputs, logits
                torch.cuda.empty_cache()
            
            # Calculate epoch metrics
            epoch_loss = epoch_loss / len(self.train_loader)
            
            # Early stopping check
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience and epoch >= self.args.local_epochs // 2:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
            
            # Print epoch summary
            print(f'Client {self.client_id} - Epoch {epoch+1}:')
            print(f'  Loss: {epoch_loss:.4f}, Accuracy: {accuracy*100:.2f}%')
            print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Load best model state
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Move model back to device
        self.model.to(self.device)
    
    def evaluate(self, test_loader):
        """Evaluate the model on test data."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(test_loader)
        
        return avg_loss, accuracy
