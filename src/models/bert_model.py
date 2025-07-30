import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig
import torch.nn.functional as F

class BERT(nn.Module):
    def __init__(self, num_classes=20, model_name='bert-base-uncased', dropout_prob=0.1):
        super(BERT, self).__init__()
        
        # Load BERT configuration
        config = BertConfig.from_pretrained(
            model_name,
            num_labels=num_classes,
            output_attentions=False,
            output_hidden_states=True,
            hidden_dropout_prob=dropout_prob,
            attention_probs_dropout_prob=dropout_prob,
            classifier_dropout=dropout_prob
        )
        
        # Initialize BERT with custom config
        self.bert = BertForSequenceClassification.from_pretrained(
            model_name,
            config=config
        )
        
        # Additional dropout layer
        self.dropout = nn.Dropout(dropout_prob)
        
        # Custom classifier head
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, input_ids, attention_mask, labels=None):
        # Get BERT outputs
        outputs = self.bert.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get pooled output (CLS token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        # Pass through custom classifier
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.bert.config.num_labels), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states
        }
    
    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}
    
    def set_weights(self, weights):
        self.load_state_dict(weights)
        
    def get_gradients(self):
        grads = []
        for param in self.parameters():
            if param.grad is not None:
                grads.append(param.grad.data.cpu().numpy())
        return grads
    
    def set_gradients(self, gradients):
        for g, param in zip(gradients, self.parameters()):
            if g is not None:
                param.grad = torch.from_numpy(g)
