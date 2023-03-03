import torch
from torch import nn
import math

class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(CustomMultiheadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        self.linear_q = torch.nn.Linear(self.head_dim, self.head_dim)
        self.linear_k = torch.nn.Linear(self.head_dim, self.head_dim)
        self.linear_v = torch.nn.Linear(self.head_dim, self.head_dim)
        self.linear_out = torch.nn.Linear(embed_size, embed_size)
        
    def forward(self, query, paragraph_encoding, mask=None, return_weights=False):
        batch_size = query.size(0)
        tgt_seq_len = query.size(1)
        src_seq_len = paragraph_encoding.size(0)
        
        # Reshape the paragraph encoding to match expected shape.
        paragraph_encoding = paragraph_encoding.view(src_seq_len, 1, 1, self.embed_size)
        paragraph_encoding = torch.reshape(paragraph_encoding, (src_seq_len, 1, 1, self.num_heads, self.head_dim))
        # Transpose dimensions to get shape (src_seq_len, num_heads, heads_dim, 1, 1)
        paragraph_encoding = torch.transpose(paragraph_encoding, 1, 3)
        paragraph_encoding = torch.transpose(paragraph_encoding, 2, 3)

        # Remove the last two dimensions
        paragraph_encoding = torch.squeeze(paragraph_encoding, -1)
        paragraph_encoding = torch.squeeze(paragraph_encoding, -1)
        
        # Split the query, key, and value into multiple heads.
        query = query.view(batch_size, tgt_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        paragraph_encoding = paragraph_encoding.view(src_seq_len, self.num_heads, self.head_dim).transpose(0, 1)

        # Project the query, key, and value using linear layers.
        query = self.linear_q(query)
        key = self.linear_k(paragraph_encoding)
        value = self.linear_v(paragraph_encoding)
        
        # Compute the scaled dot-product attention.
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.embed_size)
        attention_weights = torch.softmax(scores, dim=-1)

        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask==0, float('-1e20'))
        
        context = torch.matmul(attention_weights, value)

        # Concatenate the context from multiple heads and project to output space.
        context = context.transpose(1, 2).contiguous().view(batch_size, tgt_seq_len, -1)
        output = self.linear_out(context)

        if return_weights == True:
            return output, torch.sum(attention_weights, dim=1, keepdim=False)
        else:
            return output