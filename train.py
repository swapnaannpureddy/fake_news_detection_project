print("Training script started...")    
import sys
import os
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import numpy as np

from src.config import Config
from src.model_bert_attn import BertWithAttention
from src.data_loader import load_liar
from src.data_utils import LiarDataset  # Make sure dataset.py exists!

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# Load and preprocess data
print("🔄 Loading LIAR dataset...")
df = load_liar(Config.DATA_DIR, split='train')
print(f"✅ Dataset loaded with {len(df)} samples.")

texts = df['text'].tolist()
labels = df['label'].tolist()
print(f"📊 First sample: {texts[0]} | Label: {labels[0]}")

# Tokenizer
print("🔄 Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained(Config.BERT_NAME)
print(f"✅ Tokenizer loaded: {Config.BERT_NAME}")

# Split into train and val
print("🔄 Splitting data into train and validation sets...")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)
print(f"✅ Train size: {len(train_texts)}, Val size: {len(val_texts)}")

# Datasets
print("🔄 Creating Dataset objects...")
train_dataset = LiarDataset(train_texts, train_labels, tokenizer, Config.MAX_LEN)
val_dataset = LiarDataset(val_texts, val_labels, tokenizer, Config.MAX_LEN)
print("✅ Dataset objects created.")

# Dataloaders
print("🔄 Initializing DataLoaders...")
train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
print("✅ DataLoaders ready.")

# Model, optimizer, loss
print("🔄 Initializing model...")
model = BertWithAttention(Config.BERT_NAME, n_classes=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
print("✅ Model initialized.")

# Training loop
print("🚀 Starting training loop...", flush=True)
print(f"Total training batches: {len(train_loader)}", flush=True)
best_accuracy = 0.0  # Track best accuracy for saving

for epoch in range(Config.EPOCHS):
    print(f"\n🟢 Epoch {epoch + 1}/{Config.EPOCHS}", flush=True)
    model.train()
    total_loss, correct, total = 0, 0, 0


    for i, batch in enumerate(train_loader):
        print(f"🔄 Processing batch {i+1}/{len(train_loader)}...", flush=True)

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        logits, attn_weights = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    avg_loss = total_loss / len(train_loader)
    print(f"📊 Epoch {epoch+1}/{Config.EPOCHS} | Avg Loss: {avg_loss:.4f} | Train Accuracy: {train_acc:.4f}", flush=True)
# ✅ Save best model based on accuracy
    if train_acc > best_accuracy:
        best_accuracy = train_acc
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        torch.save(model.state_dict(), f"{Config.MODEL_DIR}/best_model.pt")
        print(f"💾 Best model updated and saved with accuracy {best_accuracy:.4f}", flush=True)

print("\n✅ Training complete. Best model saved.")