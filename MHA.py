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

class MHA(nn.Module):
    def __init__(self, heads, dim, droup_out = 0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)
        self.drop_out = nn.Dropout(droup_out)

    def forward(self, x, casual_mask=None, padding_mask = None):
        # 联合proj
        # B, N, D = x.shape
        # qkv_proj = self.q_proj(x) + self.k_proj(x) + self.v_proj(x) # shape: (B, N, 3D)
        B, N, D = x.shape
        q = self.q_proj(x)  # shape: (B, N, D)
        k = self.k_proj(x)  # shape: (B, N, D)
        v = self.v_proj(x)  # shape: (B, N, D)

        query = q.view(B, N, self.heads, D // self.heads).permute(0, 2, 1, 3)  # shape: (B, heads, N, D//heads)
        key = k.view(B, N, self.heads, D // self.heads).permute(0, 2, 1, 3)  # shape: (B, heads, N, D//heads)
        value = v.view(B, N, self.heads, D // self.heads).permute(0, 2, 1, 3)  # shape: (B, heads, N, D//heads)


        # query: [B, heads, N, D//heads]
        # key.transpose(-2, -1): [B, heads, D//heads, N]
        attn = torch.matmul(query, key.transpose(-2, -1)) * self.scale  # shape: (B, heads, N, N)
        attn = attn.softmax(dim = -1)
        attn = self.drop_out(attn)
        # attn: (B, heads, N, N), value: (B, heads, N, D//heads)
        out = torch.matmul(attn, value).transpose(1,2).reshape(B, N, D) 
        # (B, heads, N, D//heads) -> (B, N, D)
        return self.out_proj(out)






class ViT_Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.to_qkv(x)  # (B, N, 3D)
        qkv = qkv.reshape(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        # qkv: [3, B, heads, N, D/heads]

        q, k, v = qkv[0], qkv[1], qkv[2]
        # q, k, v: [B, H, N, D/heads]

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, D)
        return self.out(out)




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
    mha = MHA(heads=num_heads, dim=hidden_size)
    vit_attn = ViT_Attention(heads=num_heads, dim=hidden_size)

    # 计算多头注意力输出
    output = mha(hidden_state)
    vit_attn_out = vit_attn(hidden_state)
    
    print("Input shape:", hidden_state.shape)
    print("Output shape:", output.shape)
    # print("Input tensor:", hidden_state)
    # print("Output tensor:", output)

    print("vit Output shape", vit_attn_out.shape)
    
if __name__ == "__main__":
	test_MHA()