"""
Training script for the baseline restaurant recommendation model.
Implements supervised learning using restaurant review data.
"""

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import os
import gc
import mlflow
import time
import numpy as np
from transformers import get_linear_schedule_with_warmup
from models.chatbot import RestaurantChatbot
from evaluation.evaluate_responses import compute_metrics, TEST_QUERIES

class RestaurantDataset(Dataset):
    """
    Dataset class for restaurant conversations.
    Processes and formats data for model training.
    """
    
    def __init__(self, data_path, tokenizer, max_length=128):
        """Initialize dataset with data path and tokenizer."""
        # Read the file and split into individual JSON objects
        with open(data_path, 'r') as f:
            content = f.read()
            # Split on closing braces and filter out empty strings
            json_strings = [s.strip() + '}' for s in content.split('}') if s.strip()]
            
        # Parse each JSON object
        self.data = []
        for json_str in json_strings:
            try:
                item = json.loads(json_str)
                # Extract question and response from input/output
                question = item['input'].replace('User: ', '').strip()
                response = item['output'].replace('Assistant:', '').strip()
                self.data.append({'question': question, 'response': response})
            except json.JSONDecodeError:
                continue  # Skip invalid JSON objects
                
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        response = item['response']
        
        # Combine question and response with special tokens
        text = f"User: {question.strip()}\nAssistant: {response.strip()}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Remove the batch dimension the tokenizer adds
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Mask padding

        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def evaluate_model(model, test_queries):
    """
    Evaluate model on test queries and compute metrics.
    Returns dictionary of averaged evaluation metrics.
    """
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        for query in test_queries:
            response = model.generate_response(
                query,
                max_length=200,
                temperature=0.7,
                top_k=50,
                top_p=0.9
            )
            metrics = compute_metrics(response, query)  
            all_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
    
    return avg_metrics

def train_baseline(
    train_path='data/train.json',
    val_path='data/val.json',
    output_dir='models/baseline',
    num_epochs=8,
    batch_size=8,
    gradient_accumulation_steps=3,
    learning_rate=5e-5,
    warmup_steps=500,
    max_length=128,
    max_train_samples=8000,
    experiment_name="restaurant_chatbot_baseline",
    run_name=None,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train the baseline restaurant recommendation model.
    Implements supervised training with validation and checkpointing.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Set up MLflow
    mlflow.set_experiment(experiment_name)
    if run_name is None:
        run_name = f"baseline_run_{int(time.time())}"
    
    with mlflow.start_run(run_name=run_name):
        print(f"Using device: {device}")
        print(f"Training on {max_train_samples} samples for {num_epochs} epochs")
        print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
        
        # Log parameters
        mlflow.log_params({
            "model_type": "gpt2",
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "max_length": max_length,
            "max_train_samples": max_train_samples
        })
        
        # Clear GPU cache and garbage collect
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize model
        model = RestaurantChatbot(device=device)
        model = model.to(device)
        model.train()

        # Create datasets and dataloaders
        train_dataset = RestaurantDataset(train_path, model.tokenizer, max_length)
        val_dataset = RestaurantDataset(val_path, model.tokenizer, max_length)

        # Configure DataLoader with generator for reproducibility
        g = torch.Generator()
        g.manual_seed(42)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for Colab
            pin_memory=True if device == 'cuda' else False,
            generator=g,
            persistent_workers=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if device == 'cuda' else False,
            persistent_workers=False
        )

        # Initialize optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Training loop
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0
            train_steps = 0
            optimizer.zero_grad()  # Zero gradients at start of epoch
            
            # Training
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            for batch_idx, batch in enumerate(progress_bar):
                # Clear GPU cache periodically
                if train_steps % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Scale loss by gradient accumulation steps
                loss = outputs['loss'] / gradient_accumulation_steps
                total_train_loss += loss.item() * gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update weights if we've accumulated enough gradients
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                train_steps += 1
                progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})
                
                if train_steps * batch_size >= max_train_samples:
                    break
                
                # Free up memory
                del outputs, loss
            
            avg_train_loss = total_train_loss / train_steps
            train_losses.append(avg_train_loss)
            
            # Validation
            model.eval()
            total_val_loss = 0
            val_steps = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc='Validation'):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs['loss']
                    total_val_loss += loss.item()
                    val_steps += 1
                    
                    # Free up memory
                    del outputs, loss
            
            avg_val_loss = total_val_loss / val_steps
            val_losses.append(avg_val_loss)
            
            # Evaluate on test queries
            test_metrics = evaluate_model(model, TEST_QUERIES)
            
            # Log metrics
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "epoch": epoch,
                **test_metrics
            })
            
            print(f'\nEpoch {epoch+1}:')
            print(f'Average training loss: {avg_train_loss:.4f}')
            print(f'Average validation loss: {avg_val_loss:.4f}')
            print("Test Metrics:", test_metrics)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_path = os.path.join(output_dir, 'best_model.pt')
                # Save complete state dict including both model and sentiment head
                torch.save({
                    'model': model.model.state_dict(),
                    'sentiment_head': model.sentiment_head.state_dict()
                }, model_path)
                print(f'New best model saved with validation loss: {best_val_loss:.4f}')
                mlflow.log_artifact(model_path)
            
            # Clear cache after each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Log final metrics and artifacts
        mlflow.log_metrics({
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "best_val_loss": best_val_loss
        })
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'final_metrics': test_metrics
        } 
