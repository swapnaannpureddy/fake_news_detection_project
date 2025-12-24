import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from transformers import AutoTokenizer
import re  # Needed for clean_text


from collections import Counter
import numpy as np

def build_vocab(texts, max_vocab_size=5000):
    all_words = ' '.join(texts).lower().split()
    most_common = Counter(all_words).most_common(max_vocab_size - 2)  # Reserve 0: <PAD>, 1: <UNK>
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for i, (word, _) in enumerate(most_common, start=2):
        vocab[word] = i
    return vocab

def encode_texts(texts, vocab, max_len=128):
    encoded_texts = []
    for text in texts:
        words = text.lower().split()
        encoded = [vocab.get(word, vocab['<UNK>']) for word in words]
        if len(encoded) < max_len:
            encoded += [vocab['<PAD>']] * (max_len - len(encoded))
        else:
            encoded = encoded[:max_len]
        encoded_texts.append(encoded)
    return np.array(encoded_texts)


# ⬇️ Add clean_text here
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)          # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)       # Remove punctuation and numbers
    text = re.sub(r"\s+", " ", text).strip()      # Remove extra whitespace
    return text

# Your LiarDataset class
class LiarDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            
        }

# Function to load train and val loaders
def get_dataloaders(config):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    #tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_df = pd.read_csv(os.path.join(config.DATA_DIR, "liar_train.csv"))
    val_df = pd.read_csv(os.path.join(config.DATA_DIR, "liar_valid.csv"))

    # Optional: Clean the text using clean_text function
    train_df['statement'] = train_df['statement'].apply(clean_text)
    val_df['statement'] = val_df['statement'].apply(clean_text)
    
    train_df['label'] = train_df['label'].astype(int)
    val_df['label'] = val_df['label'].astype(int)
    
    train_dataset = LiarDataset(
        texts=train_df['statement'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer,
        max_len=config.MAX_LEN
    )

    val_dataset = LiarDataset(
        texts=val_df['statement'].tolist(),
        labels=val_df['label'].tolist(),
        tokenizer=tokenizer,
        max_len=config.MAX_LEN
    )

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    

    return train_loader, val_loader
