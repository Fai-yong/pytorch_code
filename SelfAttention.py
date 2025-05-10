import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size:int, heads:int):
        super(SelfAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert(self.heads * self.head_dim == embed_size)

        self.W_q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.W_v = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.W_k = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.out_layer = nn.Linear(self.head_dim * heads, embed_size)

    def forward(self, querys:torch.tensor, key:torch.Tensor, mask):
        N = querys.shape[0]
        value_len, key_len, query_len = key.shape[1], key.shape[1], querys.shape[1]

        values = key.reshape(N, value_len, self.heads, self.head_dim)
        keys = key.reshape(N, key_len, self.heads, self.head_dim) 
        querys = querys.reshape(N, value_len, self.heads, self.head_dim)

        querys = self.W_q(querys)
        keys = self.W_k(keys)
        values = self.W_v(keys)
        

        score = torch.einsum("nqhd, nkhd->nhqk", querys, keys)

        if mask is not None:
            score = score.masked_fill(mask==0, float(-1e20))
                                      
        attention = F.softmax(score / (self.embed_size ** 0.5), dim = 3)

        out = torch.einsum("nhql,nlhd->nqhd", attention, values).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.out_layer(out)
        return out

# Example usage
embed_size = 256  # Embedding size
heads = 8  # Number of attention heads
seq_length = 10  # Length of the input sequence

self_attention = SelfAttention(embed_size, heads)
values = torch.rand((1, seq_length, embed_size))
keys = torch.rand((1, seq_length, embed_size))
query = torch.rand((1, seq_length, embed_size))
mask = None

output = self_attention(query, keys, mask)
print("Output :")
print(output)
print("Output.shape:")  # Should output: torch.Size([1, seq_length, embed_size])
print(output.shape)  # Should output: torch.Size([1, seq_length, embed_size])



        