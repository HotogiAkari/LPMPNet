# comparison_models.py

"""
学习型对照组模型库
======================================
本文件提供了四个有代表性的学习型模型，作为LPMP-Net系列深度学习模型的性能基准。
每个模型都旨在从一个特定的角度，凸显LPMP-Net的设计优势。

1.  SimpleCNN: 标准的轻量级CNN，作为“天真”的基线，预期在置乱图像上会失效。
2.  DeepSetNet: 纯粹的排列不变模型，用于证明引入频域分析这一领域知识的价值。
3.  DCTNet: 强大的频域竞争者，用于凸显LPMP-Net V4/V5对系数置乱的鲁棒性。
4.  HistogramMLP: 简单的可学习直方图模型，用于证明特征来源的重要性。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def create_dct_matrix(size: int) -> torch.Tensor:
    """辅助函数：创建一个N x N的DCT-II变换矩阵。"""
    matrix = torch.zeros(size, size, dtype=torch.float32)
    for k in range(size):
        for n in range(size):
            if k == 0:
                matrix[k, n] = 1.0 / np.sqrt(size)
            else:
                matrix[k, n] = np.sqrt(2.0 / size) * np.cos(np.pi * k * (2 * n + 1) / (2.0 * size))
    return matrix

# =============================================================================
# --- 1. “天真”的基线：标准CNN (SimpleCNN) ---
# =============================================================================

class SimpleCNN(nn.Module):
    """
    一个标准的、轻量级的CNN。它依赖于像素的局部空间结构。
    其设计目的是为了在像素置乱的加密场景下，展示其性能的脆弱性。
    """
    def __init__(self, output_dim: int, channels: int = 32, block_size: int = 32):
        super().__init__()
        # 根据输入块大小计算卷积后的特征维度
        feature_size = (block_size // 4)**2 
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # block_size / 2
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # block_size / 4
            nn.AdaptiveAvgPool2d(1)
        )
        self.final_fc = nn.Linear(channels * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, 3, H, W]
        features = self.encoder(x).squeeze(-1).squeeze(-1)
        return self.final_fc(features)


# =============================================================================
# --- 2. 纯粹的排列不变模型：深度集合网络 (DeepSetNet) ---
# =============================================================================

class DeepSetNet(nn.Module):
    """
    一个基于Deep Sets理论的模型，它将图像块视为一个无序的像素集合。
    它完全忽略空间信息，只学习像素值的统计分布。
    其目的是为了证明，相比于纯统计，结合频域等领域知识的重要性。
    """
    def __init__(self, output_dim: int, pixel_embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        # phi_net: 将每个像素(3个通道)映射到高维空间
        self.phi = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, pixel_embed_dim)
        )
        # rho_net: 处理聚合后的集合特征
        self.rho = nn.Sequential(
            nn.Linear(pixel_embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, 3, H, W]
        B, C, H, W = x.shape
        
        # 将图像块视为一个像素集合: [B, 3, H*W] -> [B, H*W, 3]
        pixel_set = x.view(B, C, -1).permute(0, 2, 1)
        
        # 1. 元素变换 (phi)
        pixel_embeds = self.phi(pixel_set) # [B, H*W, pixel_embed_dim]
        
        # 2. 排列不变聚合 (sum pooling)
        set_representation = pixel_embeds.sum(dim=1) # [B, pixel_embed_dim]
        
        # 3. 输出变换 (rho)
        return self.rho(set_representation)


# =============================================================================
# --- 3. 强大的频域竞争者：DCTNet ---
# =============================================================================

class DCTNet(nn.Module):
    """
    一个代表性的频域模型，它在图像块的2D-DCT系数上应用一个小型CNN。
    这个模型虽然强大，但其内部的CNN对DCT系数的*空间位置*是敏感的。
    """
    def __init__(self, output_dim: int, dct_kernel_size: int = 8, channels: int = 32):
        super().__init__()
        self.dct_kernel_size = dct_kernel_size
        dct_matrix = create_dct_matrix(dct_kernel_size)
        self.register_buffer('dct_matrix', dct_matrix); self.register_buffer('dct_matrix_t', dct_matrix.t())
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.final_fc = nn.Linear(3 * channels, output_dim)

    def _process_channel(self, c: torch.Tensor) -> torch.Tensor:
        B, _, H, W = c.shape
        # 使用 floor division 确保 L 是整数
        L = (H // self.dct_kernel_size) * (W // self.dct_kernel_size)
        patches = F.unfold(c, kernel_size=self.dct_kernel_size, stride=self.dct_kernel_size).permute(0, 2, 1)
        patches = patches.reshape(-1, self.dct_kernel_size, self.dct_kernel_size)
        dct_coeffs = (self.dct_matrix @ patches @ self.dct_matrix_t).unsqueeze(1)
        encoded_vectors = self.encoder(dct_coeffs).squeeze(-1).squeeze(-1)
        return encoded_vectors.view(B, L, self.encoder[-2].out_channels).mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused_b = self._process_channel(x[:, 0:1, :, :])
        fused_g = self._process_channel(x[:, 1:2, :, :])
        fused_r = self._process_channel(x[:, 2:3, :, :])
        return self.final_fc(torch.cat([fused_b, fused_g, fused_r], dim=1))


# =============================================================================
# --- 4. 简单的可学习直方图模型：HistogramMLP ---
# =============================================================================

class HistogramMLP(nn.Module):
    """
    一个简单的基线模型，它计算三通道颜色直方图，并通过一个MLP进行学习。
    这代表了最基础的、排列不变的统计特征方法。
    """
    def __init__(self, output_dim: int, hist_bins: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.hist_bins = hist_bins
        self.mlp = nn.Sequential(
            nn.Linear(3 * hist_bins, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        hist_features = []
        img_batch_255 = x * 255.0
        for i in range(B):
            img_sample = img_batch_255[i]
            hists = [torch.histc(img_sample[ch], bins=self.hist_bins, min=0, max=255) for ch in range(C)]
            hist_features.append(torch.cat(hists))
        
        batch_hists = torch.stack(hist_features, dim=0)
        batch_hists = F.normalize(batch_hists, p=1, dim=-1)
        return self.mlp(batch_hists)