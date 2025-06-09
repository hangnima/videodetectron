import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class AudioEncoder(nn.Module):
    """音频特征提取器 - 处理MFCC特征"""
    def __init__(self, input_dim=39, hidden_dim=512):  # 39维：13 MFCC + 13 Delta + 13 Delta2
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, audio_features):
        # audio_features: [B, T, max_audio_length, 39]
        B, T, seq_len, feat_dim = audio_features.shape
        
        # 重塑为 [B*T, seq_len, feat_dim]
        x = audio_features.view(B * T, seq_len, feat_dim)
        
        # 对每个时间步的音频特征进行平均池化
        x = torch.mean(x, dim=1)  # [B*T, feat_dim]
        
        # 通过编码器
        x = self.encoder(x)  # [B*T, hidden_dim]
        
        # 重塑回 [B, T, hidden_dim]
        x = x.view(B, T, -1)
        return x

class MultiModalTicDetector(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.window_size = args.window_size
        self.use_audio = getattr(args, 'use_audio', True)
        
        # 视觉编码器
        self.backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        visual_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Identity()
        
        # 统一特征维度到512
        self.visual_proj = nn.Linear(visual_features, 512)
        
        # 音频编码器
        if self.use_audio:
            audio_input_dim = getattr(args, 'n_mfcc', 13) * 3  # 13 * 3 = 39
            self.audio_encoder = AudioEncoder(input_dim=audio_input_dim, hidden_dim=512)
            
            # 特征融合层
            self.fusion = nn.Sequential(
                nn.Linear(512 + 512, 512),  # visual + audio
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            fused_dim = 512
        else:
            # 即使不使用音频，也保持512维输出
            fused_dim = 512
        
        # 输出层
        self.label_len = len(args.labels)
        self.out_fc = nn.Linear(fused_dim, self.label_len)

    def forward(self, x, skeleton, audio_features=None):
        """
        Args:
            x: [B, C, T, H, W] 视觉输入 (经过permute后的格式)
            skeleton: [B, C, T, H, W] 骨架输入 (经过permute后的格式)
            audio_features: [B, T, max_audio_length, 39] 音频MFCC特征 (可选)
        """
        # 输入格式：[B, C, T, H, W] (经过permute后)
        B, C, T, H, W = skeleton.shape
        
        # 重新排列为 [B*T, C, H, W] 以便ResNet处理
        visual_input = skeleton.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]
        visual_input = visual_input.view(-1, C, H, W)  # [B*T, C, H, W]
        
        # 提取视觉特征
        visual_features = self.backbone(visual_input)  # [B*T, 2048]
        visual_features = visual_features.view(B, T, -1)  # [B, T, 2048]
        
        # 将视觉特征投影到512维
        visual_feat = self.visual_proj(visual_features)  # [B, T, 512]
        
        if self.use_audio and audio_features is not None:
            # 音频特征提取
            audio_feat = self.audio_encoder(audio_features)  # [B, T, 512]
            
            # 特征融合
            combined_features = torch.cat([visual_feat, audio_feat], dim=-1)  # [B, T, 1024]
            fused_features = self.fusion(combined_features)  # [B, T, 512]
        else:
            # 只使用视觉特征
            fused_features = visual_feat  # [B, T, 512]
        
        # 输出预测
        output = self.out_fc(fused_features)  # [B, T, label_len]
        return output


# 使用示例
class Args:
    def __init__(self):
        self.window_size = 5
        self.labels = ['normal', 'tic_mild', 'tic_moderate', 'tic_severe', 'other']  # 5个类别
        self.use_audio = True
        self.n_mfcc = 13

# 创建模型
args = Args()
model = MultiModalTicDetector(args)

# 使用方法：
# 1. 有音频的情况：
# output = model(input_frame.permute(0, 2, 1, 3, 4), 
#                input_frame_skeleton.permute(0, 2, 1, 3, 4), 
#                input_frame_audio)
#
# 2. 没有音频的情况：
# output = model(input_frame.permute(0, 2, 1, 3, 4), 
#                input_frame_skeleton.permute(0, 2, 1, 3, 4))

print("Model created successfully!")
print(f"Expected output shape: [batch_size, time_steps, {len(args.labels)}]")
