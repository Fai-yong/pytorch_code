import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # 计算均方根(RMS)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # 归一化
        x_norm = x / rms
        # 应用缩放参数
        return self.weight * x_norm

class PreNorm(nn.Module):
    """
    Pre-normalization wrapper: 先归一化，再应用传入的模块
    """
    def __init__(self, dim: int, fn, norm_type="rms"):
        super().__init__()
        if norm_type == "rms":
            self.norm = RMSNorm(dim)
        else:
            self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit activation function
    """
    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
    
    def forward(self, x):
        # 实现SwiGLU激活函数
        swish = self.w1(x) * torch.sigmoid(self.w1(x) * 1.0)  # β=1.0
        gate = self.w2(x)
        x = swish * gate  # 门控机制
        x = self.w3(x)  # 投影到输出维度
        return x

class LLaMAFeedForward(nn.Module):
    """
    LLaMA模型中的前馈网络模块
    """
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256, dropout: float = 0.0):
        super().__init__()
        # 确保hidden_dim是multiple_of的整数倍
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.swiglu = SwiGLU(dim, hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.swiglu(x))