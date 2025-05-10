import timm
import torch.nn as nn

class VisionEncoder(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.reset_classifier(0)  # 去掉分类头部
        self.output_dim = self.model.num_features

    def forward(self, images):
        """
        images: (B, 3, H, W)
        return: (B, N, D)  其中 N 是 patch 数（包括 cls token）
        """
        return self.model.forward_features(images)  # (B, N, D)

from transformers import BertModel, BertTokenizer

class TextEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.model = BertModel.from_pretrained(model_name)
        self.output_dim = self.model.config.hidden_size

    def forward(self, input_ids, attention_mask):
        """
        input_ids: (B, L)
        attention_mask: (B, L)
        return: (B, L, D)
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state  # 每个 token 的表征 (B, L, D)
    
import torch.nn as nn

class CrossModalEncoder(nn.Module):
    def __init__(self, hidden_dim=768, n_heads=12, num_layers=2):
        super().__init__()
        # 图像 token 自身的 self-attn
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim,
                                       nhead=n_heads,
                                       dim_feedforward=2048,
                                       batch_first=True)
            for _ in range(num_layers)
        ])
        # 图像 token attend 文本 token
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, 
                                                num_heads=n_heads,
                                                batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, vision_feats, text_feats, vision_mask=None, text_mask=None):
        """
        vision_feats: (B, V, D)
        text_feats: (B, L, D)
        return: 融合语义的视觉表示 (B, V, D)
        """
        x = vision_feats
        for layer in self.self_attn_layers:
            x = layer(x, src_key_padding_mask=vision_mask)  # vision 自注意力
        
        # 关键：图像 token 作为 Query 去 attend 文本
        cross_out, _ = self.cross_attn(
            query=x,
            key=text_feats,
            value=text_feats,
            key_padding_mask=text_mask
        )
        x = self.norm(x + cross_out)
        return x  # 输出为融合了语义的视觉 token

    
class CrossModalDecoderLayer(nn.Module):
    def __init__(self, hidden_dim=768, n_heads=12, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, text_feats, tgt_mask=None, text_mask=None):
        # 1. Causal Self-Attention
        x2, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(x2))

        # 2. Cross Attention: vision_feats作为Query
        x2, _ = self.cross_attn(x, text_feats, text_feats, key_padding_mask=text_mask)
        x = self.norm2(x + self.dropout(x2))

        # 3. FFN
        x2 = self.ffn(x)
        x = self.norm3(x + self.dropout(x2))
        return x


class CrossModalDecoder(nn.Module):
    def __init__(self, vocab_size=30522, hidden_dim=768, n_heads=12, num_layers=6, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            CrossModalDecoderLayer(hidden_dim, n_heads, dropout)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, decoder_input_ids, text_feats, tgt_mask=None, text_mask=None):
        """
        decoder_input_ids: (B, T)
        text_feats: 来自 TextEncoder 的文本 token (B, L, D)
        """
        x = self.token_embed(decoder_input_ids)  # (B, T, D)

        for layer in self.layers:
            x = layer(x, text_feats, tgt_mask, text_mask)

        return self.output_proj(x)  # (B, T, vocab_size)


