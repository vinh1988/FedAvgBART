import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig

class BERT(nn.Module):
    def __init__(self, num_classes=4, model_name='bert-base-uncased'):
        super(BERT, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            output_attentions=False,
            output_hidden_states=False
        )
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
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
