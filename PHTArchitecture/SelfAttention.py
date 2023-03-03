import torch
from torch import nn

class SelfAttention(nn.Module):
    def __init__(self,embed_size,heads):
        super(SelfAttention,self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.heads_dim = embed_size // heads

        assert(self.heads_dim*self.heads == embed_size), "heads not divisible with embed size"

        self.values = nn.Linear(self.heads_dim,self.heads_dim, bias=False)
        self.keys = nn.Linear(self.heads_dim,self.heads_dim,bias=False)
        self.queries = nn.Linear(self.heads_dim,self.heads_dim,bias=False)
        self.fc_out = nn.Linear(embed_size,embed_size)

    def forward(self, keys, query, values, mask = None, return_weights = False):
        N = query.shape[0]
        key_len, value_len, query_len = keys.shape[1], values.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.heads_dim)
        query = query.reshape(N, query_len, self.heads, self.heads_dim)
        keys = keys.reshape(N, key_len, self.heads, self.heads_dim)

        values = self.values(values)
        query = self.queries(query)
        keys = self.keys(keys)

        energy = torch.einsum("nqhd,nkhd->nhqk",[query,keys])

        if mask is not None:
            energy = energy.masked_fill(mask==0, float('-1e20'))

        attention = torch.softmax(energy / (self.embed_size**1/2),dim=3)
        attention_weights = torch.sum(attention, dim=1, keepdim = False)

        out = torch.einsum('nhql,nlhd->nqhd',[attention,values]).reshape(
            N, query_len, self.heads*self.heads_dim
        )

        out = self.fc_out(out)

        if return_weights==False:
            return out
        else:
            return out, attention_weights