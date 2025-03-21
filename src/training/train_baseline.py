import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import os
import gc
from transformers import get_linear_schedule_with_warmup
from src.models.chatbot import RestaurantChatbot

class RestaurantDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=128):
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
        text = f"{question} [SEP] {response}"
        
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
        
        # Create labels (same as input_ids for causal LM)
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def train_baseline(
    train_path='data/train.json',
    val_path='data/val.json',
    output_dir='models/baseline',
    num_epochs=8,
    batch_size=16,  # Reduced batch size
    learning_rate=5e-5,
    warmup_steps=500,
    max_length=128,
    max_train_samples=8000,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    print(f"Using device: {device}")
    print(f"Training on {max_train_samples} samples for {num_epochs} epochs")
    
    # Clear GPU cache and garbage collect
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model
    model = RestaurantChatbot(device=device)
    model = model.to(device)

    # Create datasets
    train_dataset = RestaurantDataset(train_path, model.tokenizer, max_length)
    val_dataset = RestaurantDataset(val_path, model.tokenizer, max_length)

    # Create dataloaders with reduced num_workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,  # Reduced num_workers
        pin_memory=True  # Enable pin memory for faster data transfer
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,  # Reduced num_workers
        pin_memory=True  # Enable pin memory for faster data transfer
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
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        train_steps = 0
        
        # Training
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in progress_bar:
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
            
            loss = outputs['loss']
            total_train_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            train_steps += 1
            progress_bar.set_postfix({'loss': loss.item()})
            
            if train_steps * batch_size >= max_train_samples:
                break
            
            # Free up memory
            del outputs, loss
        
        avg_train_loss = total_train_loss / train_steps
        
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
        
        print(f'\nEpoch {epoch+1}:')
        print(f'Average training loss: {avg_train_loss:.4f}')
        print(f'Average validation loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
            print(f'New best model saved with validation loss: {best_val_loss:.4f}')
        
        # Clear cache after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    return {
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'best_val_loss': best_val_loss
    } 