import torch 
from torch import nn
from SelfAttention import SelfAttention


class DecoderBlock1(nn.Module):
    def __init__(self, embed_size, heads, dropout):
        super(DecoderBlock1, self).__init__()
        self.attention = SelfAttention(embed_size, heads)

        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, key, query, values, mask):
        attention = self.attention(key, query, values, mask)
        x = self.dropout(self.norm(query+attention))

        return x