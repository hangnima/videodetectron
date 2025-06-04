import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from einops import rearrange

class TemporalCrossAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 投影层定义
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        self.query_proj_2 = nn.Linear(d_model, d_model)
        self.key_proj_2 = nn.Linear(d_model, d_model)
        self.value_proj_2 = nn.Linear(d_model, d_model)

        # 相对位置编码
        self.rel_pos_bias = nn.Parameter(torch.randn(num_heads, 1, 1, 2 * num_heads - 1))

        # 输出层
        self.out_proj = nn.Linear(d_model, d_model)
        self.out_proj_2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_face, x_skeleton):
        """输入维度: [batch*group, time, feat]"""
        B_T, T, D = x_face.shape  # B_T = batch * group

        # 生成Q/K/V投影
        q_face = rearrange(self.query_proj(x_face), "b t (h d) -> b h t d", h=self.num_heads)
        k_face = rearrange(self.key_proj(x_face), "b t (h d) -> b h t d", h=self.num_heads)
        v_face = rearrange(self.value_proj(x_face), "b t (h d) -> b h t d", h=self.num_heads)

        q_skeleton = rearrange(self.query_proj_2(x_skeleton), "b t (h d) -> b h t d", h=self.num_heads)
        k_skeleton = rearrange(self.key_proj_2(x_skeleton), "b t (h d) -> b h t d", h=self.num_heads)
        v_skeleton = rearrange(self.value_proj_2(x_skeleton), "b t (h d) -> b h t d", h=self.num_heads)

        q, k, v = torch.cat([q_face, q_skeleton], dim=3), torch.cat([k_face, k_skeleton], dim=3), torch.cat([v_face, v_skeleton], dim=3)
        # 计算注意力分数
        scores = torch.einsum("b h i d, b h j d -> b h i j", q, k) / (self.head_dim ** 0.5)
        # torch.nn.functional.scaled_dot_product_attention(q, k, v)

        # 注意力权重计算
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 特征聚合
        output = torch.einsum("b h i j, b h j d -> b h i d", attn_weights, v)
        output = rearrange(output, "b h t d -> b t (h d)")
        x1, x2 = torch.chunk(output, chunks=2, dim=-1)

        return self.out_proj(x1), self.out_proj_2(x2)

class VideoTransformer2(nn.Module):
    def __init__(self,
                 args,
                 num_classes=5,
                 img_size=224,
                 patch_size=16,
                 num_frames=16,
                 dim=768,
                 depth=12,
                 heads=8):
        super().__init__()

        # 输入嵌入层
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.frame_pos_embed = nn.Parameter(torch.randn(1, num_patches, dim))
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, num_frames, dim))

        # Transformer层堆叠
        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, num_frames, args)
            for _ in range(depth)
        ])

        # 分类头
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x_face, x_skeleton):
        # x: (B, C, T, H, W)

        B, C, T, H, W = x_face.shape

        # 空间嵌入
        spatial_tokens_face = []
        spatial_tokens_skeleton = []
        for t in range(T):
            frame_face = x_face[:, :, t]  # (B,C,H,W)
            tokens_face = self.patch_embed(frame_face)  # (B, D, h, w)
            tokens_face = rearrange(tokens_face, 'b d h w -> b (h w) d')
            tokens_face += self.frame_pos_embed
            spatial_tokens_face.append(tokens_face)

            frame_skeleton = x_skeleton[:, :, t]  # (B,C,H,W)
            tokens_skeleton = self.patch_embed(frame_skeleton)  # (B, D, h, w)
            tokens_skeleton = rearrange(tokens_skeleton, 'b d h w -> b (h w) d')
            tokens_skeleton += self.frame_pos_embed
            spatial_tokens_skeleton.append(tokens_skeleton)

        # 时间维度整合
        x_skeleton = torch.stack(spatial_tokens_skeleton, dim=1)  # (B, T, N, D)
        x_skeleton += self.temporal_pos_embed.unsqueeze(2)

        x_face = torch.stack(spatial_tokens_face, dim=1)  # (B, T, N, D)
        x_face += self.temporal_pos_embed.unsqueeze(2)

        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, T, -1, -1)
        x_face = torch.cat([cls_tokens, x_face], dim=2)  # (B, T, N+1, D)
        x_skeleton = torch.cat([cls_tokens, x_skeleton], dim=2)  # (B, T, N+1, D)

        # 时空Transformer处理
        for layer in self.layers:
            x_face, x_skeleton = layer(x_face, x_skeleton)

        # 分类特征提取
        cls_feat = x_face[:, :, 0] + x_skeleton[:, :, 0]  # (B, T, D)
        # cls_feat = cls_feat.mean(dim=1)  # (B, D)

        return self.mlp_head(cls_feat)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, num_frames, args):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.temporal_attn = TemporalCrossAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
        self.norm1_2 = nn.LayerNorm(dim)
        self.norm2_2 = nn.LayerNorm(dim)
        self.mlp_2 = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x_face, x_skeleton):
        # x: (B, T, N, D)
        B, T, N, D = x_face.shape

        # 时间交叉注意力
        residual_face = x_face
        residual_skeleton = x_skeleton
        x_face = self.norm1(x_face)
        x_skeleton = self.norm1_2(x_skeleton)
        x_face = rearrange(x_face, 'b t n d -> (b n) t d')  # 空间展开
        x_skeleton = rearrange(x_skeleton, 'b t n d -> (b n) t d')  # 空间展开
        x_face, x_skeleton = self.temporal_attn(x_face, x_skeleton)  # 时间维度处理
        x_face = rearrange(x_face, '(b n) t d -> b t n d', b=B, n=N)
        x_skeleton = rearrange(x_skeleton, '(b n) t d -> b t n d', b=B, n=N)
        x_face += residual_face
        x_skeleton += residual_skeleton

        # 前馈网络
        residual_face = x_face
        x_face = self.norm2(x_face)
        x_face = self.mlp(x_face)
        x_face += residual_face

        residual_skeleton = x_skeleton
        x_skeleton = self.norm2(x_skeleton)
        x_skeleton = self.mlp(x_skeleton)
        x_skeleton += residual_skeleton

        return x_face, x_skeleton