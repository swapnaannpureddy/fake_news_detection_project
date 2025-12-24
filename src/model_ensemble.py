import torch
import torch.nn as nn

class EnsembleFakeNews(nn.Module):
    def __init__(self, bert_model, bilstm_model, n_classes=2):
        super().__init__()
        self.bert_model = bert_model
        self.bilstm_model = bilstm_model
        self.fc = nn.Linear(n_classes * 2, n_classes)  # For stacking

    def forward(self, input_ids, attention_mask, x_bilstm):
        bert_logits, _ = self.bert_model(input_ids, attention_mask)
        bilstm_logits, _ = self.bilstm_model(x_bilstm)
        out = torch.cat([bert_logits, bilstm_logits], dim=1)
        logits = self.fc(out)
        return logits