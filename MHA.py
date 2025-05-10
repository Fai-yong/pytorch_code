import math
from typing import Optional, List

import torch
from torch import nn
# from labml import tracker

# class PrepareForMultiHeadAttention(nn.Module):
#     def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
#         super().__init__()
#         self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
#         self.heads = heads
#         self.d_k = d_k

#     def forward(self, x: torch.Tensor):
#         head_shape = x.shape[:-1]
#         x = self.linear(x)
#         x = x.view(*head_shape, self.heads, self.d_k)
#         return x

# class MultiHeadAttention(nn.Module):
#     def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
#         super().__init__()
#         self.d_k = d_model // heads
#         self.heads = heads

#         self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
#         self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
#         self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)

#         self.softmax = nn.Softmax(dim=1)
#         self.output = nn.Linear(d_model, d_model)
#         self.dropout = nn.Dropout(dropout_prob)
#         self.scale = 1 / math.sqrt(self.d_k)

#         self.attn = None

#     def get_scores(self, query: torch.Tensor, key: torch.Tensor):
#         return torch.einsum('ibhd,jbhd->ijbh', query, key)

#     def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):
#         assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
#         assert mask.shape[1] == key_shape[0]
#         assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]
#         mask = mask.unsqueeze(-1)
#         return mask

#     def forward(self, *, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None):
#         seq_len, batch_size, _ = query.shape

#         if mask is not None:
#             mask = self.prepare_mask(mask, query.shape, key.shape)

#         query = self.query(query)
#         key = self.key(key)
#         value = self.value(value)

#         scores = self.get_scores(query, key)
#         scores *= self.scale

#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, float('-inf'))

#         attn = self.softmax(scores)
#         tracker.debug('attn', attn)
#         attn = self.dropout(attn)

#         x = torch.einsum("ijbh,jbhd->ibhd", attn, value)
#         self.attn = attn.detach()

#         x = x.reshape(seq_len, batch_size, -1)
#         return self.output(x)

import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self, heads, embedding_dim):
        super().__init__()

        self.heads = heads
        self.head_dim = embedding_dim // heads

        assert(self.head_dim * heads == embedding_dim)

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.o_linear = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, query:torch.Tensor, key:torch.Tensor):

        q_len, k_len = query.shape[1], query.shape[1]
        batch_size = query.shape[0]

        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(key)

        query = query.view(batch_size, -1, self.heads, self.head_dim).transpose(1,2)
        key = key.view(batch_size, -1, self.heads, self.head_dim).transpose(1,2)
        value = value.view(batch_size, -1, self.heads, self.head_dim).transpose(1,2)

        # [B, h, q, h_d], [B, h, k, h_d]
        scale = torch.tensor(self.head_dim, dtype=torch.float32)
        score = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(scale)
        attention = torch.softmax(score, dim=-1)

        # B,H,q,k * B,H,k,h_d -> B,H,q,h_d
        output = torch.matmul(attention, value)

        # B,H,q,h_d -> B,q,H,h_d -> B,q,d
        output = output.transpose(1,2)
        print(output.shape)
        output = output.reshape(batch_size,-1,self.head_dim * self.heads)

        return self.o_linear(output)







def test_MHA():
    batch_size = 16
    seq_len = 512
    hidden_size = 1024
    num_heads = 8
    
    # 随机生成输入数据
    hidden_state = torch.randn(batch_size, seq_len, hidden_size)  # (batch_size, seq_len, hidden_size)
    
    # 生成因果掩码（下三角矩阵），这里就不刻意生成 padding_mask
    # causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    
    # 创建多头注意力模块
    mha = Attention(num_heads, hidden_size)

    # 计算多头注意力输出
    output = mha(hidden_state, hidden_state)
    
    print("Input shape:", hidden_state.shape)
    print("Output shape:", output.shape)
    print("Input tensor:", hidden_state)
    print("Output tensor:", output)
    
if __name__ == "__main__":
	test_MHA()