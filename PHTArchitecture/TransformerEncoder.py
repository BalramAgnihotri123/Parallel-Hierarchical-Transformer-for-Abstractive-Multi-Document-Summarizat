import torch
from torch import nn
from SelfAttention import SelfAttention

class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(TransformerEncoder, self).__init__()
        self.attention = SelfAttention(embed_size, heads)

        self.norm = nn.LayerNorm(embed_size)
        self.ffc = nn.Sequential(
            nn.Linear(embed_size,embed_size*forward_expansion),
            nn.ReLU(),
            nn.Linear(embed_size*forward_expansion, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, key, query, values, mask):
        attention = self.attention(key, query, values, mask)
        x = self.dropout(self.norm(query+attention))
        
        forward = self.ffc(x)
        out = self.dropout(self.norm(forward+x))
        return out