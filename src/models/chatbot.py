"""
Restaurant recommendation chatbot implementation using GPT-2.
Includes response generation and sentiment analysis capabilities.
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Dict, List, Optional, Union
import os
import torch.nn.functional as F

# Configure memory allocation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class RestaurantChatbot(nn.Module):
    """
    Restaurant recommendation chatbot based on GPT-2.
    Implements response generation and sentiment analysis.
    """
    
    def __init__(self, model_name="gpt2", device=None):
        """Initialize the chatbot with specified model and device."""
        super().__init__()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # GPT-2 model
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.config.pad_token_id = self.model.config.eos_token_id

        # Sentiment head (⚠️ Don't move individually)
        self.sentiment_head = nn.Linear(self.model.config.n_embd, 1)

        # Move everything in one go
        self.to(self.device)


    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        Handles both training and inference modes.
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        # Forward pass through GPT-2
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True  # Always get hidden states for sentiment
        )

        # Get the last hidden state
        last_hidden = outputs.hidden_states[-1]

        # 🤯 This is probably still CPU — fix it
        if last_hidden.device != self.device:
            last_hidden = last_hidden.to(self.device)

        # Do mean manually to control the device
        pooled = torch.mean(last_hidden, dim=1)

        # Just in case (safety check)
        if pooled.device != self.device:
            pooled = pooled.to(self.device)

        # 🤬 Colab-safe double check (will print visibly if crashing)
        if pooled.device != self.sentiment_head.weight.device:
            raise RuntimeError(
                f"[DEVICE MISMATCH] pooled is on {pooled.device}, "
                f"sentiment_head is on {self.sentiment_head.weight.device}"
            )

        sentiment_logits = self.sentiment_head(pooled)

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "sentiment_logits": sentiment_logits
        }


    
    def generate_response(
        self,
        user_input: str,
        max_length: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True
    ) -> str:
        """
        Generate a response to user input.
        Controls generation parameters for response quality.
        """
        # Format input with User: and Assistant: as regular text
        formatted_input = f"User: {user_input.strip()}\nAssistant:"
        
        # Prepare input for generation
        inputs = self.tokenizer(
            formatted_input,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        # Generate response
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode and format response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant's response
        if "Assistant:" in response:
            response = response.split("Assistant:", 1)[1].strip()
        
        return response
    
    def get_sentiment(self, text: str) -> float:
        """Get sentiment score for a piece of text."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self(
                **inputs,
                output_hidden_states=True
            )
        
        return outputs["sentiment_logits"].item()
    
    def save_pretrained(self, path: str) -> None:
        """Save model and tokenizer to path."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save sentiment head separately
        torch.save(self.sentiment_head.state_dict(), f"{path}/sentiment_head.pt")
    
    @classmethod
    def from_pretrained(cls, path: str, device: str = None) -> "RestaurantChatbot":
        """Load model from path."""
        instance = cls(model_name=path, device=device)
        
        # Load sentiment head if available
        sentiment_path = f"{path}/sentiment_head.pt"
        if os.path.exists(sentiment_path):
            instance.sentiment_head.load_state_dict(
                torch.load(sentiment_path, map_location=instance.device)
            )
        
        return instance

    def generate_with_log_prob(self, user_input, max_length=200, temperature=0.7, top_k=50, top_p=0.9):
        """
        Generate response with log probability computation.
        Used for reinforcement learning training.
        """
        # Format input
        formatted_input = f"User: {user_input.strip()}\nAssistant:"
        input_ids = self.tokenizer.encode(formatted_input, return_tensors='pt').to(self.device)
        
        # Initialize sequence storage
        generated_ids = input_ids
        log_probs = torch.tensor(0.0, requires_grad=True, device=self.device)
        
        # Generate token by token
        for _ in range(max_length):
            # Get model outputs
            outputs = self.model(generated_ids)
            next_token_logits = outputs.logits[:, -1, :] / max(temperature, 1e-6)  # Prevent division by zero
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits[0, :] = float('-inf')
                next_token_logits[0, top_k_indices[0]] = top_k_logits[0]
            
            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Compute probabilities with numerical stability
            next_token_logits = F.log_softmax(next_token_logits, dim=-1)
            probs = torch.exp(next_token_logits)
            
            # Check for numerical issues
            if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                # Fallback to greedy sampling
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            else:
                # Sample from the distribution
                next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to log probability with clipping to prevent extreme values
            token_log_prob = next_token_logits[0, next_token[0]]
            log_probs = log_probs + torch.clamp(token_log_prob, min=-100, max=100)
            
            # Add the predicted token to the sequence
            generated_ids = torch.cat((generated_ids, next_token), dim=1)
            
            # Stop if we predict the end token
            if next_token[0].item() == self.tokenizer.eos_token_id:
                break
        
        # Decode the sequence
        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        if "Assistant:" in response:
            response = response.split("Assistant:", 1)[1].strip()
        
        return response, generated_ids[0], log_probs
    
    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        self.sentiment_head = self.sentiment_head.to(device)
        return super().to(device)

    
    def train(self):
        """Set the model to training mode."""
        self.model.train()
    
    def eval(self):
        """Set the model to evaluation mode."""
        self.model.eval()
    
    def parameters(self):
        """Return all model parameters for optimization."""
        return list(self.model.parameters()) + list(self.sentiment_head.parameters())
    
    def load_state_dict(self, state_dict):
        """Load model state dictionary."""
        if isinstance(state_dict, dict) and 'model' in state_dict and 'sentiment_head' in state_dict:
            self.model.load_state_dict(state_dict['model'])
            self.sentiment_head.load_state_dict(state_dict['sentiment_head'])
        else:
            # For backward compatibility
            self.model.load_state_dict(state_dict)
    
    def state_dict(self):
        """Get model state dictionary."""
        return {
            'model': self.model.state_dict(),
            'sentiment_head': self.sentiment_head.state_dict()
        } 
