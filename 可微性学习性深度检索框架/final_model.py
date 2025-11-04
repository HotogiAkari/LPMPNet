import torch
from torch import nn
from typing import List
import numpy as np
import torch.nn.functional as F

class HDCT(nn.Module):
    """
    高效分层聚合网络 (HieraDCT)
    结合多尺度分析和统计聚合来捕捉图像的丰富信息

    通过在多个尺度上对输入图像的每个通道进行分块处理，并对这些块进行特征变换和统计聚合，从而实现高效的特征提取。
    首先使用一个独立的变换网络 (phi) 来处理每个尺度下的块，
    然后计算每个尺度的全局统计特征（均值和标准差），
    最后将所有通道和所有尺度的特征融合，通过一个全连接层输出最终的特征向量。

    dct_kernel_sizes (List[int]):
        一个整数列表，定义了用于分块和处理的多个内核大小（尺度）， 默认为 [8, 16]。

    phi_channels (int):
        phi 网络输出的特征维度。决定了后续统计聚合的特征维度。默认为 32。

    output_dim (int,):
        模型最终输出的特征向量维度。默认为 256。
    """
    def __init__(
        self,
        dct_kernel_sizes: List[int] = [8, 16],
        phi_channels: int = 32,
        output_dim: int = 256
    ):
        super().__init__()
        self.dct_kernel_sizes = dct_kernel_sizes
        
        # --- 为每个尺度创建独立的phi网络 ---
        # 由于输入的系数数量不同, 不能共享phi
        self.phis = nn.ModuleList()
        for size in dct_kernel_sizes:
            num_coeffs = size * size
            self.phis.append(
                nn.Sequential(
                    nn.Linear(num_coeffs, phi_channels * 4),
                    nn.LayerNorm(phi_channels * 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(phi_channels * 4, phi_channels)
                )
            )

        # 预计算所有需要的DCT矩阵
        for size in dct_kernel_sizes:
            dct_matrix = create_dct_matrix(size)
            self.register_buffer(f'dct_matrix_{size}', dct_matrix)
            self.register_buffer(f'dct_matrix_t_{size}', dct_matrix.t())
        
        # 最终投影层
        # 融合所有尺度的特征，每个尺度贡献 2 * phi_channels
        total_input_dim = len(dct_kernel_sizes) * (phi_channels * 2)
        self.final_fc = nn.Linear(3 * total_input_dim, output_dim)

    def _process_channel(self, c: torch.Tensor) -> torch.Tensor:
        """
        处理单个图像通道，提取多尺度统计特征。
        input:
        单个通道的输入张量，形状为 [B, 1, H, W]。
        return::
        该通道融合后的特征向量，形状为 [B, num_scales * phi_channels * 2]。
        """
        B, _, H, W = c.shape
        
        scale_features = []
        # --- 分别处理每个尺度，然后再融合 ---
        for i, size in enumerate(self.dct_kernel_sizes):
            # 1. 分块与DCT
            patches = F.unfold(c, kernel_size=size, stride=size).permute(0, 2, 1) # [B, L, k*k]
            
            # 2. 特征变换 (phi) - 使用对应尺度的phi网络
            # [B, L, k*k]
            phi_net = self.phis[i]
            transformed_features = phi_net(patches) # [B, L, phi_channels]

            # 3. 分层统计聚合
            # a. 局部统计 (已由phi学习)
            
            # b. 全局统计
            global_mean = transformed_features.mean(dim=1) # [B, phi_channels]
            global_std = transformed_features.std(dim=1, unbiased=False) # [B, phi_channels]
            
            # 4. 拼接当前尺度的全局统计特征
            scale_stat_vector = torch.cat([global_mean, global_std], dim=1) # [B, phi_channels * 2]
            scale_features.append(scale_stat_vector)
        
        # 5. 在最后融合所有尺度的特征 ---
        # 将 [B, 64] 和 [B, 64] 拼接为 [B, 128]
        final_channel_vector = torch.cat(scale_features, dim=1)
        
        return final_channel_vector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input:
        输入的图像张量，形状应为 [B, 3, H, W]，其中 B 是批量大小, 3 代表 RGB 通道。
        return:
        最终输出的特征向量，形状为 [B, output_dim]。
        """
        fused = torch.cat([self._process_channel(c) for c in x.chunk(3, dim=1)], dim=1)
        return self.final_fc(fused)
    

class HDCT_G(nn.Module):
    """
    """
    def __init__(self, dct_stream_channels=64, output_dim=256):
        super().__init__()
        # 三个独立的流，对应R, G, B三通道，不共享权重
        # 这允许模型学习特定于每个颜色通道的模式
        self.stream_b = GuardianDCTStream(channels=dct_stream_channels)
        self.stream_g = GuardianDCTStream(channels=dct_stream_channels)
        self.stream_r = GuardianDCTStream(channels=dct_stream_channels)
        
        total_input_dim = 3 * dct_stream_channels
        self.final_fc = nn.Linear(total_input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 并行处理三通道
        feat_b = self.stream_b(x[:, 0:1, :, :])
        feat_g = self.stream_g(x[:, 1:2, :, :])
        feat_r = self.stream_r(x[:, 2:3, :, :])
        
        # 融合
        fused = torch.cat([feat_b, feat_g, feat_r], dim=1)
        return self.final_fc(fused)
# --- 辅助函数 ---

def create_dct_matrix(size: int) -> torch.Tensor:
    """辅助函数：创建一个N x N的DCT-II变换矩阵。"""
    matrix = torch.zeros(size, size, dtype=torch.float32)
    for k in range(size):
        for n in range(size):
            if k == 0: matrix[k, n] = 1.0 / np.sqrt(size)
            else: matrix[k, n] = np.sqrt(2.0 / size) * np.cos(np.pi * k * (2 * n + 1) / (2.0 * size))
    return matrix

class GuardianDCTStream(nn.Module):
    """
    HDCT-G核心：通过排序重构实现排列不变性，并通过CNN学习深度特征。
    """
    def __init__(self, block_size=8, channels=64): # 增加通道数以增强表达能力
        super().__init__()
        self.block_size = block_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels // 2, 3, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
    
    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """
        :param c: 单通道图像块 [B, 1, H, W]
        :return: 特征向量 [B, channels]
        """
        B, _, H, W = c.shape
        # 1. 分块
        patches = F.unfold(c, kernel_size=self.block_size, stride=self.block_size).permute(0, 2, 1)
        patches = patches.reshape(-1, self.block_size**2) # [B*L, k*k]
        
        # 2. 重构
        sorted_patches, _ = torch.sort(patches, dim=1)
        reconstructed_blocks = sorted_patches.view(-1, 1, self.block_size, self.block_size) # [B*L, 1, k, k]
        
        # 3. CNN编码
        encoded_vectors = self.encoder(reconstructed_blocks).squeeze() # [B*L, C]
        
        # 4. 全局聚合
        return encoded_vectors.view(B, -1, encoded_vectors.shape[-1]).mean(dim=1)

class HieraDCT_v2(nn.Module):
    """
    高效分层聚合网络 v2 (HieraDCT_v2)

    该模型首先使用一个卷积主干 (Stem) 对输入图像进行初步的特征提取和通道融合。
    然后，在多个尺度上对生成的特征图进行分块，并对每个块应用离散余弦变换 (DCT)
    将其转换到频域。接着，一个独立的变换网络 (phi) 处理这些频域系数，
    并计算每个尺度的全局统计特征（均值和标准差）。最后，所有尺度的特征被融合，
    通过一个全连接层输出最终的特征向量。
    """
    def __init__(
        self,
        dct_kernel_sizes: List[int] = [8, 16],
        stem_channels: int = 64,
        phi_channels: int = 32,
        output_dim: int = 256
    ):
        super().__init__()
        # 按照从大到小排序，便于后续逻辑
        self.dct_kernel_sizes = sorted(dct_kernel_sizes, reverse=True)
        self.phi_channels = phi_channels
        
        self.stem = nn.Conv2d(3, stem_channels, kernel_size=4, stride=4)
        
        # phi 网络和 DCT 矩阵的创建方式不变
        self.phis = nn.ModuleList()
        for size in self.dct_kernel_sizes:
            num_coeffs = stem_channels * size * size
            self.phis.append(
                nn.Sequential(
                    nn.Linear(num_coeffs, phi_channels * 4),
                    nn.LayerNorm(phi_channels * 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(phi_channels * 4, phi_channels)
                )
            )
        for size in self.dct_kernel_sizes:
            dct_matrix = create_dct_matrix(size * size)
            self.register_buffer(f'dct_matrix_{size}', dct_matrix)
        
        # 最终投影层的输入维度是固定的，等于所有可能的尺度都参与时的最大维度
        self.max_input_dim = len(self.dct_kernel_sizes) * (phi_channels * 2)
        self.final_fc = nn.Linear(self.max_input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.stem(x)
        B, C, H_feat, W_feat = feature_map.shape
        
        scale_features = []
        # 遍历预设的每个尺度
        for i, size in enumerate(self.dct_kernel_sizes):
            # --- 核心改动：动态过滤 ---
            # 如果当前核尺寸大于特征图尺寸，则跳过
            if size > H_feat or size > W_feat:
                continue
            
            # --- 后续处理与 v2 版本相同 ---
            patches = F.unfold(feature_map, kernel_size=size, stride=size)
            patches = patches.permute(0, 2, 1).view(B, -1, C, size * size)
            
            dct_matrix = getattr(self, f'dct_matrix_{size}')
            freq_patches = patches @ dct_matrix.t()
            freq_patches_flat = freq_patches.view(B, -1, C * size * size)

            phi_net = self.phis[i]
            transformed_features = phi_net(freq_patches_flat)

            global_mean = transformed_features.mean(dim=1)
            global_std = transformed_features.std(dim=1, unbiased=False)
            
            scale_stat_vector = torch.cat([global_mean, global_std], dim=1)
            scale_features.append(scale_stat_vector)
        
        # --- 核心改动：处理拼接和填充 ---
        if not scale_features:
            # 如果列表为空（所有尺度都被跳过），创建一个零张量作为输入
            # .to(x.device) 确保张量在正确的设备上 (CPU/GPU)
            final_vector = torch.zeros(B, self.max_input_dim, device=x.device)
        else:
            # 正常拼接
            fused_vector = torch.cat(scale_features, dim=1)
            
            # 对拼接后的向量进行右侧零填充，使其维度达到固定的 max_input_dim
            # F.pad 的参数格式是 (左填充, 右填充, 上填充, 下填充, ...)
            pad_size = self.max_input_dim - fused_vector.shape[1]
            final_vector = F.pad(fused_vector, (0, pad_size))

        return self.final_fc(final_vector)