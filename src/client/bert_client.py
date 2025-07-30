import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from src.models.bert_model import BERT
from src.datasets.agnews import AGNewsDataset

class BERTClient:
    def __init__(self, client_id, train_data, device, args):
        self.client_id = client_id
        self.device = device
        self.args = args
        
        # Initialize model
        self.model = BERT(num_classes=4).to(device)
        
        # Initialize tokenizer
        from transformers import BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Create dataset and dataloader
        train_dataset = AGNewsDataset(
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
    
    def train(self, global_model):
        """Train the model for one round."""
        # Set model to training mode
        self.model.train()
        
        # Update local model with global weights
        self.model.load_state_dict(global_model.state_dict())
        
        # Training loop
        total_loss = 0.0
        for _ in range(self.args.local_epochs):
            for batch in self.train_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
        
        # Calculate average loss
        avg_loss = total_loss / (len(self.train_loader) * self.args.local_epochs)
        
        return avg_loss, self.model.get_weights()
    
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
