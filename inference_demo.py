import torch
from transformers import BertTokenizer
from src.config import Config
from src.model_bert_attn import BertWithAttention

# ✅ Load tokenizer and model with the same config as used during training
tokenizer = BertTokenizer.from_pretrained(Config.BERT_NAME)
model = BertWithAttention(bert_name=Config.BERT_NAME, n_classes=2)
model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE))
model.to(Config.DEVICE)
model.eval()

def predict(text):
    # Preprocess input
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=Config.MAX_LEN,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(Config.DEVICE)
    attention_mask = encoding['attention_mask'].to(Config.DEVICE)

    # Forward pass
    with torch.no_grad():
        logits, attn_weights = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    label = "Real" if pred == 1 else "Fake"
    return label, probs.squeeze().tolist()

# 🔍 Test example
if __name__ == "__main__":
    example_text = "The president will visit Canada next month for a summit."
    label, confidence = predict(example_text)
    print(f"📰 Prediction: {label}")
    print(f"📊 Confidence: {confidence}")
