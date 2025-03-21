import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Set memory allocation strategy
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class RestaurantChatbot(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        # Initialize the base model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-small')
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
        
        # Enable gradient checkpointing
        self.model.gradient_checkpointing_enable()
        
        # Add sentiment analysis head
        self.sentiment_head = nn.Linear(
            self.model.config.hidden_size,
            1  # Single scalar output for sentiment
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.device = device
        self.to(device)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get base model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            output_hidden_states=True  # Request hidden states
        )
        
        # Get the last hidden state
        hidden_states = outputs.hidden_states[-1] if outputs.hidden_states is not None else None
        
        # Calculate sentiment if hidden states are available
        sentiment_logits = None
        if hidden_states is not None:
            sentiment_logits = self.sentiment_head(hidden_states.mean(dim=1))
        
        # Return a dictionary with all outputs
        return {
            'loss': outputs.loss if labels is not None else None,
            'logits': outputs.logits,
            'sentiment_logits': sentiment_logits,
            'hidden_states': hidden_states
        }

    def generate_response(self, input_text, max_length=128, temperature=0.7, top_k=50, top_p=0.9, do_sample=True):
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Generate response
        outputs = self.model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        return response

    def get_sentiment(self, text):
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
        
        # Get sentiment prediction
        if outputs['sentiment_logits'] is not None:
            sentiment = torch.sigmoid(outputs['sentiment_logits'])
            return sentiment.item()
        return 0.5  # Neutral sentiment if no sentiment logits available 