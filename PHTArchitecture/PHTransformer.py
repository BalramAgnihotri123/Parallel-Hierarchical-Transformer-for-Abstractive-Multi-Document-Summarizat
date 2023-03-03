import torch
from torch import nn

from PHTEncoder import PHTEncoder
from PHTDecoder import PHTDecoder

class PHTransformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 src_pad_idx,
                 tgt_pad_idx,
                 device='cpu',
                 dropout=0,
                 word_heads=4,
                 para_heads = 10,
                 num_layers=6,
                 forward_expansion=4,
                 max_length=2000,
                 embed_size=100,
                 ):
        super(PHTransformer,self).__init__()
        self.encoder = PHTEncoder(embed_size=embed_size, 
                               word_heads=word_heads,
                               para_heads = para_heads, 
                               device = device, 
                               forward_expansion = forward_expansion,
                               src_vocab_size=src_vocab_size,
                               num_layers=num_layers,
                               dropout = dropout,
                               max_length = max_length
                               )
        self.decoder = PHTDecoder(tgt_vocab_size,
                               max_length,
                               embed_size,
                               device,
                               dropout,
                               forward_expansion,
                               word_heads,
                               para_heads,
                               num_layers
                               )
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src!=self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_tgt_mask(self, tgt):
        N, tgt_len = tgt.shape
        tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len)).expand(
            N, 1, tgt_len, tgt_len
            )
        
        return tgt_mask.to(self.device)

    def forward(self, src, tgt, src_mask = None, tgt_mask = None):
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        if src_mask is None:
            src_mask = self.make_src_mask(src)

        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt)

        word_embeddings, paragraph_embeddings = self.encoder(src)
        out = self.decoder(tgt, word_embeddings, paragraph_embeddings)

        return out


if __name__ == '__main__':
    b = torch.randint(low=0, high=400, size=(32, 512), dtype=torch.long)
    phtransformer = PHTransformer(50265, 50265, 1, 1).to('cpu')
    out = phtransformer(b,b)
    print(out.shape)