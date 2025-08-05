import torch
import torch.nn as nn
from transformers import BartConfig, BartModel
from transformers.modeling_outputs import SequenceClassifierOutput

class BART(torch.nn.Module):
    def __init__(self, num_classes, num_embeddings=50265, embedding_size=1024, hidden_size=1024, 
                 dropout=0.1, use_pt_model=True, is_seq2seq=False):
        super(BART, self).__init__()
        self.is_seq2seq = is_seq2seq
        self.num_classes = num_classes
        self.print_once = True  # For debug prints
        
        print(f"\n=== Initializing BART Model ===")
        print(f"Number of classes: {num_classes}")
        print(f"Is seq2seq: {is_seq2seq}")
        
        try:
            # Load BART model
            print("Loading BART base model...")
            self.model = BartModel.from_pretrained('facebook/bart-base')
            self.dropout = nn.Dropout(dropout)
            
            # Add classification head
            hidden_size = self.model.config.hidden_size
            print(f"Adding classification head with input size {hidden_size} -> {num_classes} classes")
            self.classifier = nn.Linear(hidden_size, num_classes)
            
            # Initialize weights
            print("Initializing classifier weights...")
            self.classifier.weight.data.normal_(mean=0.0, std=0.02)
            self.classifier.bias.data.zero_()
            
            # Store config
            self.num_embeddings = self.model.config.vocab_size
            self.embedding_size = self.model.config.d_model
            self.hidden_size = hidden_size
            
            print(f"Model initialized with {sum(p.numel() for p in self.parameters() if p.requires_grad)} trainable parameters")
            
        except Exception as e:
            print(f"\n=== Error initializing BART model ===")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            raise

    def forward(self, input_ids, attention_mask=None, labels=None, decoder_input_ids=None):
        # Ensure we have attention mask if not provided
        if attention_mask is None:
            attention_mask = input_ids != self.model.config.pad_token_id
        
        # Debug: Print input shapes
        if hasattr(self, 'print_once'):
            print(f"\n=== Forward Pass ===")
            print(f"Input IDs shape: {input_ids.shape}")
            print(f"Attention mask shape: {attention_mask.shape if attention_mask is not None else 'None'}")
            print(f"Labels shape: {labels.shape if labels is not None else 'None'}")
            if labels is not None:
                print(f"Unique labels: {torch.unique(labels).tolist()}")
            self.print_once = False
        
        try:
            # Get BART outputs - use encoder outputs for classification
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            if self.is_seq2seq:
                return outputs  # Return raw outputs for seq2seq
            
            # For classification, use the encoder's last hidden state
            # BART returns encoder_last_hidden_state for encoder outputs
            if hasattr(outputs, 'encoder_last_hidden_state'):
                hidden_states = outputs.encoder_last_hidden_state
            else:
                hidden_states = outputs.last_hidden_state
                
            if hidden_states is None:
                raise ValueError("Hidden states are None. Check model output.")
                
            # Take the first token's representation (BART's <s> token)
            first_token_tensor = hidden_states[:, 0]  # (batch_size, hidden_size)
            
            # Apply dropout and classifier
            pooled_output = self.dropout(first_token_tensor)
            logits = self.classifier(pooled_output)  # (batch_size, num_classes)
            
            # Calculate loss if labels are provided
            loss = None
            if labels is not None:
                # Ensure labels are in the correct shape and type
                labels = labels.long()  # Ensure labels are long integers
                if labels.dim() > 1:
                    labels = labels.squeeze(-1)
                
                # Verify logits and labels shapes match
                if logits.shape[0] != labels.shape[0]:
                    raise ValueError(f"Logits and labels batch size mismatch: {logits.shape} vs {labels.shape}")
                
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            
            # Only include the parameters that SequenceClassifierOutput expects
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
                attentions=outputs.attentions if hasattr(outputs, 'attentions') else None
            )
            
        except Exception as e:
            print(f"\n=== Error in forward pass ===")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"Input shapes - input_ids: {input_ids.shape if input_ids is not None else 'None'}")
            print(f"attention_mask: {attention_mask.shape if attention_mask is not None else 'None'}")
            print(f"labels: {labels.shape if labels is not None else 'None'}")
            if 'outputs' in locals():
                print(f"Model outputs keys: {list(outputs.keys())}")
                if 'last_hidden_state' in outputs:
                    print(f"Last hidden state shape: {outputs.last_hidden_state.shape if outputs.last_hidden_state is not None else 'None'}")
            raise
    
    def get_embeddings(self):
        """Get the word embedding layer."""
        return self.model.get_input_embeddings()
    
    def save_pretrained(self, save_directory):
        """Save the model and its configuration."""
        self.model.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load a pretrained model from a directory or HF model hub."""
        model = BartForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs
        )
        return model
