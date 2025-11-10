import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class LearnableLPMPLayer(nn.Module):
    """
    LPMP特征提取层。
    
    这个模块将传统LPMP算法的思想转化为一个端到端的神经网络层
    可以直接嵌入到模型中进行训练。
    """
    def __init__(self, in_channels: int = 3, mid_channels: int = 64, descriptor_dim: int = 128):
        """
        初始化可学习的LPMP层。

        Args:
            in_channels (int): 输入图像块的通道数 (彩色为3, 灰度为1)。
            mid_channels (int): 中间卷积层使用的通道数，控制特征的丰富度。
            descriptor_dim (int): 最终输出的特征描述符的维度。
        """
        super().__init__()

        # --- 步骤 1 & 2 的替代: 可学习的梯度与模长计算 ---
        # 使用一个Conv2d学习提取类Gx和Gy的特征。输出2个通道。
        self.gradient_conv = nn.Conv2d(in_channels, 2, kernel_size=3, padding=1, bias=False)
        
        # --- 步骤 3 的替代: 可学习的局部纹理模式分析 ---
        # 这个卷积层在梯度模长图上操作，学习捕捉重要的局部纹理。
        # 输入通道为1 (梯度模长)，输出通道为mid_channels。
        self.texture_conv = nn.Sequential(
            nn.Conv2d(1, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # --- 步骤 4 的替代: 可学习的特征聚合 (替代直方图) ---
        # 使用全局平均池化将空间特征图聚合成一个向量
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        
        # 使用一个小型MLP来生成最终的描述符
        self.mlp_head = nn.Sequential(
            nn.Linear(mid_channels, mid_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels // 4, descriptor_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播函数。

        Args:
            x (Tensor): 输入的图像块批次，形状为 [Batch, Channels, Height, Width]。

        Returns:
            Tensor: 批次的特征描述符，形状为 [Batch, descriptor_dim]，并经过L2归一化。
        """
        # --- 步骤 1: 学习提取梯度 ---
        # x: [B, C_in, H, W] -> grads: [B, 2, H, W]
        grads = self.gradient_conv(x)
        
        # --- 步骤 2: 计算梯度模长 ---
        # 将通道0和1分别视为gx和gy
        gx = grads[:, 0, :, :].unsqueeze(1) # [B, 1, H, W]
        gy = grads[:, 1, :, :].unsqueeze(1) # [B, 1, H, W]
        
        # 计算模长，并添加一个小的epsilon防止梯度爆炸
        magnitude = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6) # [B, 1, H, W]
        
        # --- 步骤 3: 学习分析局部纹理 ---
        # magnitude: [B, 1, H, W] -> texture_features: [B, mid_channels, H, W]
        texture_features = self.texture_conv(magnitude)
        
        # --- 步骤 4: 聚合特征生成描述符 ---
        # texture_features: [B, mid_channels, H, W] -> pooled_features: [B, mid_channels, 1, 1]
        pooled_features = self.pooling(texture_features)
        
        # 展平以便送入全连接层
        # pooled_features: [B, mid_channels, 1, 1] -> flattened_features: [B, mid_channels]
        flattened_features = torch.flatten(pooled_features, 1)
        
        # 生成最终描述符
        # flattened_features: [B, mid_channels] -> descriptor: [B, descriptor_dim]
        descriptor = self.mlp_head(flattened_features)
        
        # 对描述符进行L2归一化，这是度量学习中的标准做法
        descriptor = F.normalize(descriptor, p=2, dim=1)
        
        return descriptor