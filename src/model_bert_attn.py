import torch
import torch.nn as nn
from transformers import BertModel

class BertWithAttention(nn.Module):
    def __init__(self, bert_name='bert-base-uncased', n_classes=2):
        super(BertWithAttention, self).__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.attn = nn.Linear(self.bert.config.hidden_size, 1)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        attn_weights = torch.softmax(self.attn(last_hidden_state), dim=1)  # [batch_size, seq_len, 1]
        context = torch.sum(attn_weights * last_hidden_state, dim=1)  # [batch_size, hidden_size]
        logits = self.fc(context)  # [batch_size, n_classes]

        return logits, attn_weights
