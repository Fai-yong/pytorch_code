import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size:int, heads:int):
        super().__init__()
        self.embedding_size = embedding_size
        self.heads = heads
        self.head_dim = embedding_size // heads

        assert(self.head_dim * heads  == embedding_size)

        self.W_q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.W_k = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.d_v = self.head_dim
        self.W_v = nn.Linear(self.head_dim, self.d_v, bias=False)

        self.out_layer = nn.Linear(self.heads * self.d_v, self.embedding_size)

    def forward(self, query:torch.Tensor, key:torch.Tensor, mask):

        batch_size = query.shape[0]
        q_len, k_len, v_len = query.shape[1], key.shape[1], key.shape[1]

        query = query.view(batch_size, q_len, self.heads, self.head_dim)
        key = key.view(batch_size, k_len, self.heads, self.head_dim)

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(key)
        scale = self.head_dim ** 0.5

        # einsum 路径:  (B,Lq,Hn,Hd) · (B,Lk,Hn,Hd) -> (B,Hn,Lq,Lk)
        score = torch.einsum("bqhd,bkhd->bhqk", Q, K) / scale

        if mask is not None:
            score = score.masked_fill(mask==0, float(-1e20))

        attention = F.softmax(score, dim=-1)

        # einsum: B,Hn,Lq,Lk * B,Lk,Hn,dv -> B,Lk,Hn,d_v
        out = torch.einsum("bhqk,bkhd->bkhd", attention, V)

        out = out.reshape(batch_size, k_len, self.heads*self.d_v)
        return self.out_layer(out)
    
# Example usage
embed_size = 256  # Embedding size
heads = 8  # Number of attention heads
q_length = 10  # Length of the input sequence
k_length = 20
self_attention = MultiHeadAttention(embed_size, heads)
query = torch.rand((1, q_length, embed_size))
keys = torch.rand((1, k_length, embed_size))
mask = None

output = self_attention(query, keys, mask)
print("Output :")
print(output)
print("Output.shape:")  # Should output: torch.Size([1, seq_length, embed_size])
print(output.shape)  # Should output: torch.Size([1, seq_length, embed_size])

        



