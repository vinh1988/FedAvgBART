import torch
from transformers import DistilBertModel, DistilBertConfig, DistilBertForSequenceClassification

class DistilBART(torch.nn.Module):
    def __init__(self, num_classes, num_embeddings, embedding_size, hidden_size, dropout, use_pt_model, is_seq2seq=False):
        super(DistilBART, self).__init__()
        self.is_seq2seq = is_seq2seq
        
        if use_pt_model:  # fine-tuning
            self.model = DistilBertForSequenceClassification.from_pretrained(
                'distilbert-base-uncased',
                num_labels=num_classes,
                output_attentions=False,
                output_hidden_states=False,
            )
            self.num_embeddings = self.model.config.vocab_size
            self.embedding_size = self.model.config.dim
            self.num_hiddens = self.model.config.hidden_size
            self.dropout = self.model.config.dropout
        else:  # from scratch
            self.num_classes = num_classes
            self.num_embeddings = num_embeddings
            self.embedding_size = embedding_size
            self.num_hiddens = hidden_size
            self.dropout = dropout
            
            config = DistilBertConfig(
                vocab_size=self.num_embeddings,
                dim=self.embedding_size,
                hidden_dim=4 * self.embedding_size,  # As per original BERT
                n_layers=6,  # DistilBERT has 6 layers vs BERT's 12
                n_heads=8,   # 8 attention heads
                max_position_embeddings=512,
                attention_dropout=self.dropout,
                dropout=self.dropout,
                num_labels=self.num_classes
            )
            self.model = DistilBertForSequenceClassification(config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs  # Returns (loss, logits) if labels provided, else just logits
    
    def get_embeddings(self):
        """Get the word embedding layer."""
        return self.model.distilbert.embeddings.word_embeddings
    
    def get_classifier(self):
        """Get the classifier head."""
        return self.model.classifier
    
    def save_pretrained(self, save_directory):
        """Save the model and tokenizer to a directory."""
        self.model.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load a pretrained model from a directory or HF model hub."""
        model = DistilBertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs
        )
        
        # Create a new instance and replace its model with the loaded one
        instance = cls(
            num_classes=model.num_labels,
            num_embeddings=model.config.vocab_size,
            embedding_size=model.config.dim,
            hidden_size=model.config.hidden_size,
            dropout=model.config.dropout,
            use_pt_model=True,
            is_seq2seq=False
        )
        instance.model = model
        return instance
