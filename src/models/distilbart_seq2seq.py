import torch
from transformers import AutoModelForSeq2SeqLM, AutoConfig, AutoTokenizer

class DistilBARTSeq2Seq(torch.nn.Module):
    def __init__(self, model_name='facebook/bart-large-cnn', use_pt_model=True, max_length=1024, max_target_length=128):
        super(DistilBARTSeq2Seq, self).__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.max_target_length = max_target_length
        
        if use_pt_model:  # fine-tuning
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                output_attentions=False,
                output_hidden_states=False,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_config(config)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return outputs
    
    def generate(self, input_ids, attention_mask=None, **kwargs):
        """Generate text using the model."""
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_target_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            **kwargs
        )
    
    def get_embeddings(self):
        """Get the word embedding layer."""
        return self.model.get_input_embeddings()
    
    def save_pretrained(self, save_directory):
        """Save the model and tokenizer to a directory."""
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load a pretrained model from a directory or HF model hub."""
        model = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        
        instance = cls(
            model_name=pretrained_model_name_or_path,
            use_pt_model=True
        )
        instance.model = model
        instance.tokenizer = tokenizer
        return instance
