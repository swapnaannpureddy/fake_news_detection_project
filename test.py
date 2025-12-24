import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from src.data_utils import LiarDataset
from src.model_bert_attn import BertWithAttention
from src.config import Config

# Load test dataset
base_dir = os.path.dirname(__file__)
test_path = os.path.join(base_dir, 'data', 'liar', 'liar_test.csv')  # ✅ Corrected path

test_df = pd.read_csv(test_path)

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(Config.BERT_NAME)
model = BertWithAttention(n_classes=2)
model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Prepare test dataset
test_dataset = LiarDataset(
    texts=test_df['statement'].tolist(),
    labels=test_df['label'].tolist(),
    tokenizer=tokenizer,
    max_len=Config.MAX_LEN
)
test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)

# Run inference
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs, _ = model(input_ids, attention_mask)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total * 100
print(f"Test Accuracy: {accuracy:.2f}%")
