import torch
from torch import nn
import math
from PHTEncoder import PHTEncoder
from CustomMultiheadAttention import CustomMultiheadAttention
from DecoderBlock1 import DecoderBlock1
from SelfAttention import SelfAttention

class PHTDecoder(nn.Module):
    def __init__(self,
                 tgt_vocab_size,
                 max_length,
                 embed_size,
                 device,
                 dropout,
                 forward_expansion,
                 word_heads,
                 para_heads,
                 num_layers):
        super(PHTDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.device = device

        self.word_embeddings = nn.Embedding(tgt_vocab_size, embed_size)
        self.positional_embeddings = nn.Embedding(max_length, embed_size)
        self.norm = nn.BatchNorm1d(embed_size)

        self.decoder1 = nn.ModuleList([
            DecoderBlock1(embed_size, word_heads, dropout) 
            for _ in range(num_layers)
        ])

        self.paragraph_attention = CustomMultiheadAttention(embed_size, para_heads)
        self.word_attention = SelfAttention(embed_size, word_heads)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion),
            nn.ReLU(),
            nn.Linear(forward_expansion, embed_size)
        )

        self.fc_out = nn.Linear(embed_size, tgt_vocab_size)

    def RankingEncoding(self, paragraph_encoding, device = 'cpu'):
        # Compute the positional encodings once in log space.
        max_len, d_model = paragraph_encoding.shape
        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe
        return pe

    def forward(self, x, word_embeddings, paragraph_embeddings, src_mask=None, tgt_mask=None):
        N, seq_len = x.shape

        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)  # [[0,1,2...,seq_len],[0,1,2...,seq_len],[0,1,2...,seq_len],...,[Nth]]
        tgt_embeddings = self.norm((self.word_embeddings(x) + self.positional_embeddings(positions)).permute(0,2,1))
        tgt_embeddings = self.dropout(tgt_embeddings.permute(0,2,1))

        for layer in self.decoder1:
            X1 = layer(tgt_embeddings, tgt_embeddings, tgt_embeddings, mask=None)

        # 2nd part of the decoder

        paragraph_embeddings += self.RankingEncoding(paragraph_embeddings, device=self.device)
        cross_para_attention, para_attention_weights = self.paragraph_attention(X1,
                                                                                paragraph_embeddings,
                                                                                tgt_mask,
                                                                                return_weights=True)

        if tgt_mask is not None:
            tgt_mask = tgt_mask.permute(0, 1, 3, 2)
        cross_word_attention, word_attention_weights = self.word_attention(word_embeddings,
                                                                             X1,
                                                                             word_embeddings,
                                                                             tgt_mask,
                                                                             return_weights=True)

        # perform einsum multiplication
        output = torch.einsum('xyz,xyx->xyz', cross_word_attention, para_attention_weights)
        X2 = self.norm((X1 + cross_para_attention + output).permute(0,2,1))
        X2 = self.dropout(X2.permute(0,2,1))

        forward = self.feed_forward(X2)

        X3 = (self.norm((forward + X2).permute(0,2,1))).permute(0,2,1)
        X3 = self.dropout(X3)
        X3 = self.fc_out(X3)
        X3 = nn.Softmax(dim=-1)(X3)

        return X3
    

if __name__ == '__main__':
    decoder = PHTDecoder(4000,2000,100,'cpu',0,0,4,10,4)
    b = torch.randint(low=0, high=400, size=(32, 512), dtype=torch.long)
    a = PHTEncoder(4000,2000,100,4,10,512,0,3,'cpu').to("cpu")
    b = torch.randint(low=0, high=400, size=(32, 512), dtype=torch.long)
    out = a(b)
    output_decoder = decoder(b, out[0], out[1])
    print(output_decoder.shape)
    