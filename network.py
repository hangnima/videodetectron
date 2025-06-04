import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from einops import rearrange

class SpatioTemporalFusion(nn.Module):
    def __init__(self, t_dim=16):
        super().__init__()
        self.t_dim = t_dim  # 时间步长

        # 面部特征提取（3D ResNet）
        self.face_encoder = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            ResBlock3D(64, 256, stride=1),
            ResBlock3D(256, 512, stride=2)
        )

        # 骨骼特征提取（3D Conv）
        self.skeleton_encoder = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4)),
            ResBlock3D(64, 128),
            ResBlock3D(128, 256)
        )

        # 时空注意力融合
        self.st_attn = SpatioTemporalAttention(
            channels=512+256,
            t_dim=t_dim,
            reduction=8
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(512+256, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.ConvTranspose3d(256, 128, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 0, 0)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        )

    def forward(self, face, skeleton):
        # 输入维度均为 [b,t,3,224,224]
        # face = face.permute(0, 2, 1, 3, 4)  # -> [b,3,t,224,224]
        # skeleton = skeleton.permute(0, 2, 1, 3, 4)

        # 特征提取
        face_feat = self.face_encoder(face)  # [b,512,t,56,56]
        skel_feat = self.skeleton_encoder(skeleton)  # [b,256,t,56,56]

        # 特征拼接
        fused = torch.cat([face_feat, skel_feat], dim=1)  # [b,768,t,56,56]

        # 时空注意力
        attn_feat = self.st_attn(fused)  # [b,512,t,56,56]

        # 解码恢复尺寸
        output = self.decoder(attn_feat)  # [b,3,t,224,224]
        # return output.permute(0, 2, 1, 3, 4)  # -> [b,t,3,224,224]
        return output


class ResBlock3D(nn.Module):
    """3D残差块"""

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=(1, 3, 3),
                               stride=(1, stride, stride), padding=(0, 1, 1))
        self.bn1 = nn.BatchNorm3d(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.bn2 = nn.BatchNorm3d(out_c)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=1, stride=(1, stride, stride)),
                nn.BatchNorm3d(out_c)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class SpatioTemporalAttention(nn.Module):
    """时空注意力机制"""

    def __init__(self, channels, t_dim, reduction=8):
        super().__init__()
        self.t_att = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv3d(channels // reduction, t_dim, 1),
            nn.Softmax(dim=2)
        )
        self.s_att = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 时间注意力 [b,c,t,h,w] -> [b,t,h,w]
        t_weight = self.t_att(x.mean([3, 4], keepdim=True))  # 全局平均池化
        t_attn = torch.einsum('bcthw,bthw->bcthw', x, t_weight.mean(1))

        # 空间注意力 [b,c,h,w]
        s_weight = self.s_att(x.mean(2))  # 沿时间维度平均
        s_attn = torch.einsum('bchw,bchw->bchw', x.mean(2), s_weight)

        return t_attn + s_attn.unsqueeze(2)

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

        # 相对位置编码
        self.rel_pos_bias = nn.Parameter(torch.randn(num_heads, 1, 1, 2 * num_heads - 1))

        # 输出层
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """输入维度: [batch*group, time, feat]"""
        B_T, T, D = x.shape  # B_T = batch * group

        # 生成Q/K/V投影
        q = rearrange(self.query_proj(x), "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(self.key_proj(x), "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(self.value_proj(x), "b t (h d) -> b h t d", h=self.num_heads)

        # 计算注意力分数
        scores = torch.einsum("b h i d, b h j d -> b h i j", q, k) / (self.head_dim ** 0.5)

        # 添加相对位置偏置（参考T5的相对位置编码）
        # rel_pos = self._get_rel_pos(T)
        # scores += torch.einsum("b h i j, h k -> b h i j", rel_pos, self.rel_pos_bias)

        # 注意力权重计算
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 特征聚合
        output = torch.einsum("b h i j, b h j d -> b h i d", attn_weights, v)
        output = rearrange(output, "b h t d -> b t (h d)")

        return self.out_proj(output)

    def _get_rel_pos(self, seq_len):
        """生成相对位置索引矩阵"""
        device = self.rel_pos_bias.device
        rel_pos = torch.arange(seq_len, dtype=torch.long, device=device)
        rel_pos = rel_pos[:, None] - rel_pos[None, :]  # [T, T]
        rel_pos = rel_pos.clamp(-self.num_heads + 1, self.num_heads - 1) + self.num_heads - 1
        return F.one_hot(rel_pos, num_classes=2 * self.num_heads - 1).float()

class VideoTransformer(nn.Module):
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

        # 融合
        # self.fusion = SpatioTemporalFusion(t_dim=20)

    def forward(self, x_face, x_skeleton):
        # x: (B, C, T, H, W)

        # x = self.fusion(x_face, x_skeleton)
        # x = torch.cat((x_face, x_skeleton), dim=1)
        x = x_skeleton
        B, C, T, H, W = x.shape

        # 空间嵌入
        spatial_tokens = []
        for t in range(T):
            frame = x[:, :, t]  # (B,C,H,W)
            tokens = self.patch_embed(frame)  # (B, D, h, w)
            tokens = rearrange(tokens, 'b d h w -> b (h w) d')
            tokens += self.frame_pos_embed
            spatial_tokens.append(tokens)

        # 时间维度整合
        x = torch.stack(spatial_tokens, dim=1)  # (B, T, N, D)
        x += self.temporal_pos_embed.unsqueeze(2)

        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, T, -1, -1)
        x = torch.cat([cls_tokens, x], dim=2)  # (B, T, N+1, D)

        # 时空Transformer处理
        for layer in self.layers:
            x = layer(x)

        # 分类特征提取
        cls_feat = x[:, :, 0]  # (B, T, D)
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

    def forward(self, x):
        # x: (B, T, N, D)
        B, T, N, D = x.shape

        # 时间交叉注意力
        residual = x
        x = self.norm1(x)
        x = rearrange(x, 'b t n d -> (b n) t d')  # 空间展开
        x = self.temporal_attn(x)  # 时间维度处理
        x = rearrange(x, '(b n) t d -> b t n d', b=B, n=N)
        x += residual

        # 前馈网络
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x += residual

        return x

class TicDetector(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.window_size = args.window_size
        # 使用预训练的 ResNet 作为 backbone
        self.backbone = models.resnet101(pretrained=True)

        # 修改最后的全连接层以适应多任务输出
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # 去掉原有的全连接层
        self.label_len = len(args.labels)
        self.out_fc = nn.Linear(num_features, self.label_len)

        # 定义任务输出层
        # self.tic_output = nn.Linear(num_features, 1)  # 抽动 (0, 1)
        # self.severity_output = nn.Linear(num_features, 10)  # 严重程度 (0-9)
        # self.location_output = nn.Linear(num_features, 3)  # 部位 (0, 1, 2)

    def forward(self, x, skeleton):
        # x: [B, T, C, H, W], skeleton: [B, T, C, H, W]
        # 特征提取
        x = skeleton
        B, T, C, H, W = x.shape
        # x = self.cross_attention(x)
        x, skeleton = x.view(-1, C, H, W), skeleton.view(-1, C, H, W)
        features = self.backbone(x)
        features = self.out_fc(features)
        features = features.view(B, T, self.label_len)

        # 各任务的输出
        # tic = self.tic_output(features)  # 抽动预测
        # severity = self.severity_output(features)  # 严重程度预测
        # location = self.location_output(features)  # 部位预测

        # return tic, severity, location
        return features

# ---暂时不用,后期增加抽动严重程度再考虑---
class TicLoss(nn.Module):
    def __init__(self):
        super(TicLoss, self).__init__()
        self.bce_loss = nn.BCELoss()  # 二元交叉熵损失
        self.ce_loss = nn.CrossEntropyLoss()  # 交叉熵损失

    def forward(self, tic_pred, severity_pred, location_pred, tic_target, severity_target, location_target):
        # 计算各个任务的损失
        tic_loss = self.bce_loss(tic_pred, tic_target)  # 是否抽动的损失
        severity_loss = self.ce_loss(severity_pred, severity_target)  # 严重程度的损失
        location_loss = self.ce_loss(location_pred, location_target)  # 部位的损失

        # 合并损失
        total_loss = tic_loss + severity_loss + location_loss
        return total_loss