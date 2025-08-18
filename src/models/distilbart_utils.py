import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
from typing import Optional, Dict, Any, Tuple
import os

class DistilBARTWrapper:
    """Wrapper class for DistilBART model for federated learning."""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn", max_length: int = 1024, 
                 max_target_length: int = 128, device: str = None):
        """Initialize DistilBART model.
        
        Args:
            model_name: Name or path of the pre-trained model
            max_length: Maximum input sequence length
            max_target_length: Maximum target sequence length for generation
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.max_length = max_length
        self.max_target_length = max_target_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            output_attentions=False,
            output_hidden_states=False,
        ).to(self.device)
        
        # Set model to evaluation mode by default
        self.model.eval()
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get model parameters as a state dict."""
        return self.model.state_dict()
    
    def set_model_parameters(self, params: Dict[str, Any]):
        """Set model parameters from a state dict."""
        self.model.load_state_dict(params)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step.
        
        Args:
            batch: Dictionary containing 'input_ids', 'attention_mask', and 'labels'
            
        Returns:
            Dictionary containing loss and other metrics
        """
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        # Backward pass
        loss = outputs.loss
        loss.backward()
        
        return {
            'loss': loss.item(),
            'logits': outputs.logits.detach()
        }
    
    def evaluate(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Evaluate model on a batch of data.
        
        Args:
            batch: Dictionary containing 'input_ids', 'attention_mask', and 'labels'
            
        Returns:
            Dictionary containing loss and other metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
            
            # Generate summaries for ROUGE/BLEU calculation
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_target_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
            
            # Decode generated and reference summaries with validation
            preds = []
            for ids in generated_ids:
                # Ensure token IDs are valid before decoding
                valid_ids = [tid for tid in ids.tolist() if tid < self.tokenizer.vocab_size and tid >= 0]
                if not valid_ids:  # Skip if no valid tokens
                    continue
                try:
                    pred = self.tokenizer.decode(
                        valid_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    preds.append(pred)
                except Exception as e:
                    print(f"Error decoding prediction: {e}")
                    continue
            
            refs = []
            for label in labels:
                # Ensure token IDs are valid before decoding
                valid_labels = [tid for tid in label.tolist() if tid < self.tokenizer.vocab_size and tid >= 0]
                if not valid_labels:  # Skip if no valid tokens
                    continue
                try:
                    ref = self.tokenizer.decode(
                        valid_labels,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    refs.append(ref)
                except Exception as e:
                    print(f"Error decoding reference: {e}")
                    continue
            
            return {
                'loss': outputs.loss.item(),
                'predictions': preds,
                'references': refs
            }
    
    def save_pretrained(self, output_dir: str):
        """Save model and tokenizer to directory."""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load model and tokenizer from directory."""
        model = cls(model_name=model_path, **kwargs)
        return model
