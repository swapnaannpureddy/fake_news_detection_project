import torch
import torch.nn as nn

class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, embeddings=None, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embeddings))
            self.embedding.weight.requires_grad = False  # freeze embeddings
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim*2, 1)
        self.fc = nn.Linear(hidden_dim*2, n_classes)

    def forward(self, x, mask=None):
        emb = self.embedding(x)
        lstm_out, _ = self.bilstm(emb)
        attn_weights = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=1)
        context = (lstm_out * attn_weights.unsqueeze(-1)).sum(dim=1)
        logits = self.fc(context)
        return logits, attn_weights