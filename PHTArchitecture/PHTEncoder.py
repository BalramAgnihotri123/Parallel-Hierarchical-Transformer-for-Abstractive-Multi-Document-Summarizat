import torch
from torch import nn
from MultiHeadedAttentionPooling import MultiHeadedAttentionPooling
from TransformerEncoder import TransformerEncoder


class PHTEncoder(nn.Module):
    def __init__(self, 
                 src_vocab_size,
                 max_length,
                 embed_size, 
                 word_heads,
                 para_heads,
                 forward_expansion,
                 dropout,
                 num_layers, 
                 device
                 ):
        super(PHTEncoder,self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.word_embeddings = nn.Embedding(src_vocab_size, embed_size)
        self.positional_embeddings = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerEncoder(embed_size, word_heads, forward_expansion, dropout)
                for _ in range(num_layers)
            ]
        )
        
        self.attention_pooling = MultiHeadedAttentionPooling(embed_size, para_heads, dropout, forward_expansion)


    def forward(self, x, mask=None):
        x = x.to(self.device)
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device).to(self.device)

        embedded = self.dropout(self.word_embeddings(x) + self.positional_embeddings(positions))
        for layer in self.layers:
            word_embeddings = layer(embedded,embedded,embedded, mask)

        paragraph_embeddings = self.attention_pooling(word_embeddings)
        
        return word_embeddings, paragraph_embeddings


if __name__ == '__main__':
    a = PHTEncoder(4000,2000,100,4,10,512,0,3,'cpu').to("cpu")
    b = torch.randint(low=0, high=400, size=(32, 512), dtype=torch.long)
    out = a(b)
    print(out[0].shape)