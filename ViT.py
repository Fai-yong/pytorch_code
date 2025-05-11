
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_num = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor): # x: B, C, H, W
        x = self.proj(x)  # x: (B, D, H/p, W/p)
        x = x.flatten(2)  # x: (B, D, N), N = H*W / p*p
        x = x.transpose(1,2) # x: (B, N, D)

        return x

class Attention(nn.Module):
    def __init__(self, dim, heads, drop__out):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** 0.5
        self.
