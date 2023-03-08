import torch
from torch import nn

class MultiHeadedAttentionPooling(nn.Module):
    def __init__(self, embed_size, para_heads ,dropout, forward_expansion):
        super(MultiHeadedAttentionPooling,self).__init__()
        self.heads_dim = embed_size // para_heads
        self.heads = para_heads
        self.embed_size = embed_size

        assert(self.heads_dim*para_heads == embed_size), "Paragraph heads not a divisor of embed_size"

        # Multi-head attention pooling layer for computing paragraph embeddings
        self.W1 = nn.Linear(embed_size, para_heads*self.heads_dim, bias = False)
        self.W2 = nn.Linear(self.heads_dim, 1, bias = False)
        self.W3 = nn.Linear(self.embed_size, self.embed_size, bias = False)

        self.ffc = nn.Sequential(
            nn.Linear(embed_size,forward_expansion),
            nn.ReLU(),
            nn.Linear(forward_expansion,embed_size)
        )
        self.norm = nn.LayerNorm(embed_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, Cp):
        N, seq_len, embed_size = Cp.shape

        # First Linear layer and head split across last dimension
        CpW1 = self.W1(Cp)
        head_split = CpW1.reshape(N, self.heads, seq_len, self.heads_dim)

        # 2nd Linear Transformation converting the shape to N,L,D,1 and then applying softmax for standardization
        CpW2 = nn.Softmax(dim=-1)(self.W2(head_split)) # shape of CpW2 (N, seq_len, heads, 1)

        phi = torch.einsum( "nhld,nhlo->nhd" ,[head_split, CpW2])
        # converting the embeddings
        phi = phi.view(N, self.embed_size)

        W3_out = self.W3(phi)
        forward = self.ffc(W3_out)

        out = self.dropout(self.norm(W3_out+forward))
        
        return out