import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import numpy as np

# =============================================================================
# --- 1. 共享与辅助模块 ---
# =============================================================================

def create_dct_matrix(size: int) -> torch.Tensor:
    """辅助函数：创建一个N x N的DCT-II变换矩阵。"""
    matrix = torch.zeros(size, size, dtype=torch.float32)
    for k in range(size):
        for n in range(size):
            if k == 0: matrix[k, n] = 1.0 / np.sqrt(size)
            else: matrix[k, n] = np.sqrt(2.0 / size) * np.cos(np.pi * k * (2 * n + 1) / (2.0 * size))
    return matrix

class Sobel(nn.Module):
    """计算图像梯度模长的模块，作为固定的预处理层。"""
    def __init__(self):
        super().__init__()
        sobel_x_k = torch.tensor([[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]], dtype=torch.float32).unsqueeze(1)
        sobel_y_k = torch.tensor([[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]], dtype=torch.float32).unsqueeze(1)
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False); self.sobel_x.weight = nn.Parameter(sobel_x_k, requires_grad=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False); self.sobel_y.weight = nn.Parameter(sobel_y_k, requires_grad=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grad_x = self.sobel_x(x); grad_y = self.sobel_y(x)
        return torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

class ComplexityAnalyzer(nn.Module):
    """统一的图像块复杂度分析器，为门控/路由网络输出一个特征向量。"""
    def __init__(self, output_dim: int, hist_bins: int = 64, hidden_dim: int = 32):
        super().__init__()
        self.hist_bins = hist_bins
        self.mlp = nn.Sequential(nn.Linear(hist_bins, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, output_dim))
    def forward(self, grad_map: torch.Tensor) -> torch.Tensor:
        B = grad_map.shape[0]
        hists = [torch.histc(grad_map[i].flatten(), bins=self.hist_bins, min=0, max=255) for i in range(B)]
        hist_tensor = F.normalize(torch.stack(hists, dim=0), p=1, dim=-1)
        return self.mlp(hist_tensor)

# =============================================================================
# --- 2. 核心专家单元 (Expert Core Units) ---
# =============================================================================

class LPMP_DeepSet_Expert_Core(nn.Module):
    """V7模型 LPMP_DeepSetNet的核心：在滑动邻域上应用Deep Sets。"""
    def __init__(self, neighborhood_size: int = 3, phi_channels: int = 32, rho_channels: int = 64):
        super().__init__()
        self.neighborhood_size = neighborhood_size
        self.output_channels = rho_channels
        num_pixels = neighborhood_size**2
        self.phi = nn.Sequential(nn.Linear(1, phi_channels), nn.ReLU(inplace=True), nn.Linear(phi_channels, phi_channels))
        self.rho = nn.Sequential(nn.Linear(phi_channels, rho_channels), nn.ReLU(inplace=True))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        patches = F.unfold(x, kernel_size=self.neighborhood_size, padding=self.neighborhood_size//2)
        pixel_sets = patches.permute(0, 2, 1).reshape(-1, self.neighborhood_size**2, 1)
        quantized_sets = torch.floor(pixel_sets * (255.0 / 32.0))
        transformed_pixels = self.phi(quantized_sets)
        aggregated_vector = transformed_pixels.mean(dim=1)
        encoded_vectors = self.rho(aggregated_vector)
        feature_map = encoded_vectors.reshape(B, H*W, self.output_channels).permute(0, 2, 1)
        return F.fold(feature_map, output_size=(H, W), kernel_size=1)
    
class LpmpGRLayer(nn.Module):
    """
    V7 核心模块
    GPU 版 LPMP-GR：梯度模长 + 可变半径邻域方差
    输出 shape: [B, lpmp_dim]
    """
    def __init__(self, radius: int = 1, lpmp_dim: int = 64):
        super().__init__()
        self.radius = radius
        self.lpmp_dim = lpmp_dim
        # 一个小 MLP 将全局统计映射到 lpmp_dim
        self.fc = nn.Sequential(
            nn.Linear(6, 32),  # 每通道 mean/std/var 共 6
            nn.ReLU(inplace=True),
            nn.Linear(32, lpmp_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W], 已归一化到 [0,1] 或 [0,255]
        # Sobel kernel
        sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=x.dtype,device=x.device).view(1,1,3,3)
        sobel_y = sobel_x.transpose(2,3)

        mags = []
        for c in range(3):
            gx = F.conv2d(x[:,c:c+1], sobel_x, padding=1)
            gy = F.conv2d(x[:,c:c+1], sobel_y, padding=1)
            mag = torch.sqrt(gx**2 + gy**2 + 1e-6)     # 梯度模长
            if self.radius > 1:                        # 空洞邻域模仿
                mag = F.avg_pool2d(mag, kernel_size=self.radius*2-1,
                                    stride=1, padding=self.radius-1)
            mags.append(mag)

        # 全局统计: 每通道 mean/std
        feats = []
        for m in mags:
            feats.append(m.mean(dim=[2,3]))
            feats.append(m.std(dim=[2,3]))
        feats = torch.cat(feats, dim=1)   # [B,6]
        return self.fc(feats)             # [B, lpmp_dim]
    
class GuardianDCTStream(nn.Module):
    """
    通过排序重构实现排列不变性，并通过CNN学习深度特征。
    一个自包含的、端到端的特征提取器。
    """
    def __init__(self, block_size=8, channels=64): # 增加了通道数以增强表达能力
        super().__init__()
        self.block_size = block_size
        
        # 编码器现在更强大，以充分利用被规范化的输入
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
        
        # 2. 守护者重构 (排序)
        sorted_patches, _ = torch.sort(patches, dim=1)
        reconstructed_blocks = sorted_patches.view(-1, 1, self.block_size, self.block_size) # [B*L, 1, k, k]
        
        # 3. CNN编码
        encoded_vectors = self.encoder(reconstructed_blocks).squeeze() # [B*L, C]
        
        # 4. 全局聚合
        return encoded_vectors.view(B, -1, encoded_vectors.shape[-1]).mean(dim=1)
    

class TurboV6Stream(nn.Module):
    """
    直接学习子块内完整像素值的排列不变表示，精度高但对像素置乱敏感。
    """
    def __init__(self, block_sizes: List[int] = [8, 16], phi_channels: int = 32):
        super().__init__()
        self.block_sizes = block_sizes
        self.phis = nn.ModuleList()
        for size in block_sizes:
            num_pixels_in_block = size * size
            self.phis.append(
                nn.Sequential(
                    nn.Linear(num_pixels_in_block, phi_channels * 4),
                    nn.LayerNorm(phi_channels * 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(phi_channels * 4, phi_channels)
                )
            )
        # 输出维度 = 尺度数量 * (phi_channels * 2 for mean/std)
        self.output_dim = len(block_sizes) * phi_channels * 2

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        scale_features = []
        for i, size in enumerate(self.block_sizes):
            patches = F.unfold(c, kernel_size=size, stride=size).permute(0, 2, 1)
            phi_net = self.phis[i]
            transformed_features = phi_net(patches)
            global_mean = transformed_features.mean(dim=1)
            global_std = transformed_features.std(dim=1, unbiased=False)
            scale_stat_vector = torch.cat([global_mean, global_std], dim=1)
            scale_features.append(scale_stat_vector)
        
        return torch.cat(scale_features, dim=1)


class GuardianV8Stream(nn.Module):
    """
    通过排序重构，实现对块内像素置乱的鲁棒性。
    """
    def __init__(self, block_sizes: List[int] = [8, 16], channels: int = 32):
        super().__init__()
        self.block_sizes = block_sizes
        self.encoders = nn.ModuleList()
        for _ in block_sizes:
            self.encoders.append(
                nn.Sequential(
                    nn.Conv2d(1, channels // 2, 3, padding=1),
                    nn.BatchNorm2d(channels // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels // 2, channels, 3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1)
                )
            )
        self.output_dim = len(block_sizes) * channels

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        B, _, H, W = c.shape
        scale_features = []
        for i, size in enumerate(self.block_sizes):
            patches = F.unfold(c, kernel_size=size, stride=size).permute(0, 2, 1)
            patches = patches.reshape(-1, size**2)
            
            sorted_patches, _ = torch.sort(patches, dim=1)
            reconstructed_blocks = sorted_patches.view(-1, 1, size, size)
            
            encoder = self.encoders[i]
            encoded_vectors = encoder(reconstructed_blocks).squeeze()
            
            # 全局聚合
            aggregated_vector = encoded_vectors.view(B, -1, encoded_vectors.shape[-1]).mean(dim=1)
            scale_features.append(aggregated_vector)
            
        return torch.cat(scale_features, dim=1)
    

class TurboV6Expert(nn.Module):
    """V6 Turbo模块，处理原始像素。"""
    def __init__(self, block_sizes: List[int] = [8, 16], phi_channels: int = 32, expert_output_dim: int = 128):
        super().__init__()
        self.block_sizes = block_sizes
        self.phis = nn.ModuleList()
        for size in block_sizes:
            self.phis.append(nn.Sequential(
                nn.Linear(size * size, phi_channels * 4),
                nn.LayerNorm(phi_channels * 4),
                nn.ReLU(inplace=True),
                nn.Linear(phi_channels * 4, phi_channels)
            ))
        
        internal_dim = len(block_sizes) * phi_channels * 2
        self.projection = nn.Linear(internal_dim, expert_output_dim)
        self.output_dim = expert_output_dim # 明确定义输出维度

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        scale_features = []
        for i, size in enumerate(self.block_sizes):
            patches = F.unfold(c, kernel_size=size, stride=size).permute(0, 2, 1)
            transformed_features = self.phis[i](patches)
            global_mean = transformed_features.mean(dim=1)
            global_std = transformed_features.std(dim=1, unbiased=False)
            scale_features.append(torch.cat([global_mean, global_std], dim=1))
        
        # 在投影之前拼接
        concatenated_features = torch.cat(scale_features, dim=1)
        # 通过投影层，确保输出维度统一
        return self.projection(concatenated_features)
    
class GuardianExpert(nn.Module):
    """V8 Guardian模块，通过排序处理像素。"""
    def __init__(self, block_sizes: List[int] = [8, 16], channels: int = 32, expert_output_dim: int = 128):
        super().__init__()
        self.block_sizes = block_sizes
        self.encoders = nn.ModuleList()
        for _ in block_sizes:
            self.encoders.append(nn.Sequential(
                nn.Conv2d(1, channels // 2, 3, padding=1),
                nn.BatchNorm2d(channels // 2), nn.ReLU(inplace=True),
                nn.Conv2d(channels // 2, channels, 3, padding=1),
                nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1)
            ))
            
        internal_dim = len(block_sizes) * channels
        self.projection = nn.Linear(internal_dim, expert_output_dim)
        self.output_dim = expert_output_dim # 明确定义输出维度

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        B = c.shape[0]
        scale_features = []
        for i, size in enumerate(self.block_sizes):
            patches = F.unfold(c, kernel_size=size, stride=size).permute(0, 2, 1)
            patches = patches.reshape(-1, size**2)
            sorted_patches, _ = torch.sort(patches, dim=1)
            reconstructed = sorted_patches.view(-1, 1, size, size)
            encoded = self.encoders[i](reconstructed).squeeze()
            aggregated = encoded.view(B, -1, encoded.shape[-1]).mean(dim=1)
            scale_features.append(aggregated)
            
        concatenated_features = torch.cat(scale_features, dim=1)
        return self.projection(concatenated_features)

class DeepSetExpertCore(nn.Module):
    """
    将一个子块内的像素（或梯度）集合，编码为一个特征向量。
    """
    def __init__(self, phi_channels: int = 32, rho_channels: int = 64):
        super().__init__()
        # phi网络: 独立地变换集合中的每一个元素
        self.phi = nn.Sequential(
            nn.Linear(1, phi_channels),
            nn.ReLU(inplace=True),
            nn.Linear(phi_channels, phi_channels)
        )
        # rho网络: 处理聚合后的集合特征
        self.rho = nn.Sequential(
            nn.Linear(phi_channels, rho_channels),
            nn.ReLU(inplace=True),
            nn.Linear(rho_channels, rho_channels)
        )

    def forward(self, x_set: torch.Tensor) -> torch.Tensor:
        """
        :param x_set: 输入的子块集合, 形状 [B*L, k*k]
        :return: 编码后的特征向量, 形状 [B*L, rho_channels]
        """
        # [B*L, k*k] -> [B*L, k*k, 1]
        x_set = x_set.unsqueeze(-1)
        
        # 1. 元素变换 (phi)
        # [B*L, k*k, 1] -> [B*L, k*k, phi_channels]
        element_embeds = self.phi(x_set)
        
        # 2. 排列不变聚合 (mean pooling)
        # [B*L, k*k, phi_channels] -> [B*L, phi_channels]
        set_representation = element_embeds.mean(dim=1)
        
        # 3. 输出变换 (rho)
        # [B*L, phi_channels] -> [B*L, rho_channels]
        return self.rho(set_representation)

class DCT_CNN_Expert_Core(nn.Module):
    """使用微型CNN学习2D频域模式。"""
    def __init__(self, dct_kernel_size: int = 8, channels: int = 32):
        super().__init__(); self.dct_kernel_size, self.channels = dct_kernel_size, channels
        dct_matrix = create_dct_matrix(dct_kernel_size); self.register_buffer('dct_matrix', dct_matrix); self.register_buffer('dct_matrix_t', dct_matrix.t())
        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        patches = F.unfold(x, kernel_size=self.dct_kernel_size, stride=self.dct_kernel_size).permute(0, 2, 1).reshape(-1, self.dct_kernel_size, self.dct_kernel_size)
        dct_coeffs = (self.dct_matrix @ patches @ self.dct_matrix_t).unsqueeze(1)
        encoded_vectors = self.encoder(dct_coeffs).squeeze(-1).squeeze(-1)
        return encoded_vectors.view(B, -1, self.channels).mean(dim=1)

class DCT_FastDeepSet_Expert(nn.Module):
    """基于深度集合的频域专家。"""
    def __init__(self, dct_kernel_size: int = 8, phi_channels: int = 32, rho_channels: int = 64):
        super().__init__(); self.dct_kernel_size, self.output_channels = dct_kernel_size, rho_channels
        dct_matrix = create_dct_matrix(dct_kernel_size); self.register_buffer('dct_matrix', dct_matrix); self.register_buffer('dct_matrix_t', dct_matrix.t())
        self.phi = nn.Sequential(nn.Conv1d(1, phi_channels, kernel_size=1), nn.ReLU(inplace=True))
        self.rho = nn.Sequential(nn.Linear(phi_channels, rho_channels), nn.ReLU(inplace=True))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        patches = F.unfold(x, kernel_size=self.dct_kernel_size, stride=self.dct_kernel_size).permute(0, 2, 1).reshape(-1, self.dct_kernel_size, self.dct_kernel_size)
        dct_coeffs = self.dct_matrix @ patches @ self.dct_matrix_t
        coeff_set = dct_coeffs.reshape(-1, 1, self.dct_kernel_size**2)
        transformed_coeffs = self.phi(coeff_set)
        aggregated_vector = transformed_coeffs.mean(dim=2)
        encoded_vectors = self.rho(aggregated_vector)
        return encoded_vectors.view(B, -1, self.output_channels).mean(dim=1)
    
class DCT_Feature_Extractor(nn.Module):
    """
    特征提取器。
    负责将图像块转换为一系列子块的特征向量集合（点云）。
    """
    def __init__(self, dct_kernel_size: int = 8, phi_channels: int = 32, rho_channels: int = 64):
        super().__init__()
        self.dct_kernel_size = dct_kernel_size
        self.output_channels = rho_channels
        dct_matrix = create_dct_matrix(dct_kernel_size)
        self.register_buffer('dct_matrix', dct_matrix); self.register_buffer('dct_matrix_t', dct_matrix.t())
        self.phi = nn.Sequential(nn.Conv1d(1, phi_channels, 1), nn.ReLU(inplace=True))
        self.rho = nn.Sequential(nn.Linear(phi_channels, rho_channels), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        patches = F.unfold(x, kernel_size=self.dct_kernel_size, stride=self.dct_kernel_size).permute(0, 2, 1)
        patches = patches.reshape(-1, self.dct_kernel_size, self.dct_kernel_size)
        dct_coeffs = self.dct_matrix @ patches @ self.dct_matrix.t()
        coeff_set = dct_coeffs.reshape(-1, 1, self.dct_kernel_size**2)
        transformed_coeffs = self.phi(coeff_set)
        aggregated_vector = transformed_coeffs.mean(dim=2)
        encoded_vectors = self.rho(aggregated_vector)
        # 返回每个子块的特征集合: [B, L, D]
        return encoded_vectors.view(B, -1, self.output_channels)
    
# --- V6 模型 ---
class LPMPNet_V6_Turbo(nn.Module):
    """ 高效分层聚合，还没有把DCT纳入训练"""
    def __init__(
        self,
        dct_kernel_sizes: List[int] = [8, 16],
        phi_channels: int = 32,
        output_dim: int = 256
    ):
        super().__init__()
        self.dct_kernel_sizes = dct_kernel_sizes
        
        # --- 为每个尺度创建独立的phi网络 ---
        # 我们不能共享phi，因为输入的系数数量不同
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
        # 我们将融合所有尺度的特征，每个尺度贡献 2 * phi_channels
        total_input_dim = len(dct_kernel_sizes) * (phi_channels * 2)
        self.final_fc = nn.Linear(3 * total_input_dim, output_dim)

    def _process_channel(self, c: torch.Tensor) -> torch.Tensor:
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
        fused = torch.cat([self._process_channel(c) for c in x.chunk(3, dim=1)], dim=1)
        return self.final_fc(fused)
    
# --- V7模型 ---
    
class LPMPNet_DCT_Turbo_V7(nn.Module):
    """
    DCT + LPMP_GR 并行特征融合
    维度: 输出 output_dim
    块间置乱性能极好，但没有像素置乱鲁棒性，速度慢
    """
    def __init__(self,
                 dct_kernel_sizes: List[int] = [8,16],
                 phi_channels: int = 32,
                 lpmp_dim: int = 64,
                 output_dim: int = 256):
        super().__init__()
        self.dct_kernel_sizes = dct_kernel_sizes
        self.phis = nn.ModuleList()
        for size in dct_kernel_sizes:
            num_coeffs = size * size
            self.phis.append(
                nn.Sequential(
                    nn.Linear(num_coeffs, phi_channels*4),
                    nn.LayerNorm(phi_channels*4),
                    nn.ReLU(inplace=True),
                    nn.Linear(phi_channels*4, phi_channels)
                )
            )
        # 预计算 DCT
        for size in dct_kernel_sizes:
            dct_matrix = create_dct_matrix(size)          # 需保留你的实现
            self.register_buffer(f'dct_matrix_{size}', dct_matrix)
            self.register_buffer(f'dct_matrix_t_{size}', dct_matrix.t())

        self.lpmp_branch = LpmpGRLayer(radius=1, lpmp_dim=lpmp_dim)

        total_input_dim = len(dct_kernel_sizes)*(phi_channels*2)*3 + lpmp_dim
        self.final_fc = nn.Linear(total_input_dim, output_dim)

    def _process_channel(self, c: torch.Tensor, idx: int) -> torch.Tensor:
        B, _, H, W = c.shape
        scale_features = []
        for i,size in enumerate(self.dct_kernel_sizes):
            patches = F.unfold(c, kernel_size=size, stride=size).permute(0,2,1)
            feat = self.phis[i](patches)
            gmean = feat.mean(dim=1)
            gstd  = feat.std(dim=1, unbiased=False)
            scale_features.append(torch.cat([gmean,gstd],dim=1))
        return torch.cat(scale_features, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DCT 分支
        dct_feats = torch.cat([self._process_channel(c, i) for i,c in enumerate(x.chunk(3, dim=1))], dim=1)
        # LPMP-GR 分支
        lpmp_feat = self.lpmp_branch(x)
        # 融合
        fused = torch.cat([dct_feats, lpmp_feat], dim=1)
        return self.final_fc(fused)

class LPMP_DeepSetNet_V7(nn.Module):
    """
    让模型自己学习对于一个邻域像素集合，最优的排列不变统计特征是什么。
    速度一般，精度一般
    """
    def __init__(self, output_dim: int, neighborhood_size: int = 3, rho_channels: int = 64):
        super().__init__()
        self.lpmp_core = LPMP_DeepSet_Expert_Core(neighborhood_size=neighborhood_size, rho_channels=rho_channels)
        self.final_fc = nn.Linear(3 * rho_channels, output_dim)

    def _process_channel(self, c: torch.Tensor) -> torch.Tensor:
        feature_map = self.lpmp_core(c)
        return feature_map.mean(dim=[2, 3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([self._process_channel(c) for c in x.chunk(3, dim=1)], dim=1)
        return self.final_fc(fused)

 # --- V8模型 ---

class DCT_Guardian_V8(nn.Module):
    """
    精度中上，速度较快
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
    
# --- V9模型 ---

class SEBlock(nn.Module):
    """通道注意力模块——输入: [B, N_pixels, C]，输出同形状"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, max(1, in_channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, in_channels // reduction), in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, N_pixels, C]
        b, n, c = x.shape
        y = self.avg_pool(x.transpose(1, 2)).view(b, c)   # [B, C]
        y = self.fc(y).view(b, 1, c)                      # [B,1,C]
        return x * y                                      # 按通道广播相乘


# 精度优化：引入 GeM 池化层
class GeMPooling(nn.Module):
    """
    Generalized Mean Pooling (GeM)
    论文: Fine-tuning CNN Image Retrieval with No Human Annotation
    在图像检索中通常比 AvgPool 或 MaxPool 效果更好
    """
    def __init__(self, p=3.0, eps=1e-6):
        super(GeMPooling, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class EnhancedLPMP(nn.Module):
    # __init__ 方法与之前完全相同，这里省略
    def __init__(self,
                 block_size=32,
                 num_neighbors=9,
                 feature_dim=256,
                 radius_levels=None,
                 r_min=0.5,
                 r_max=5.0,
                 num_bins=8,
                 soft_temp=0.1,
                 max_offset=4.0,
                 sample_mode='bilinear',
                 use_soft_quant=True):
        super().__init__()
        self.block_size = block_size
        self.feature_dim = feature_dim
        self.r_min, self.r_max = r_min, r_max
        self.num_bins = num_bins
        self.use_soft_quant = use_soft_quant
        self.max_offset = max_offset
        self.sample_mode = sample_mode
        self.soft_temp_param = nn.Parameter(torch.tensor(float(soft_temp)))
        k = int(math.sqrt(num_neighbors))
        self.kernel_size = k if k * k == num_neighbors else 3
        self.effective_num_neighbors = self.kernel_size * self.kernel_size
        if radius_levels is None:
            self.radius_param = nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float32))
            self.num_radii = 2
        else:
            self.radius_param = nn.Parameter(torch.tensor(radius_levels, dtype=torch.float32))
            self.num_radii = len(radius_levels)
        self.bin_size_param = nn.Parameter(torch.tensor(math.log(math.exp(32.0) - 1.0)))
        self.conv_x = nn.Conv2d(3, 3, (1, 3), padding=(0, 1), bias=False, groups=3)
        self.conv_y = nn.Conv2d(3, 3, (3, 1), padding=(1, 0), bias=False, groups=3)
        with torch.no_grad():
            sobel_x = torch.tensor([[[[-1., 0., 1.]]]])
            sobel_y = torch.tensor([[[[-1.], [0.], [1.]]]])
            self.conv_x.weight.copy_(torch.cat([sobel_x] * 3, dim=0))
            self.conv_y.weight.copy_(torch.cat([sobel_y] * 3, dim=0))
        self.conv_x.weight.requires_grad = False
        self.conv_y.weight.requires_grad = False
        self.norm = nn.InstanceNorm2d(3, affine=True)
        offset_out_channels = 3 * self.effective_num_neighbors * self.num_radii * 2
        self.offset_conv = nn.Conv2d(3, offset_out_channels, 3, 1, 1, groups=3)
        total_neighbors = self.effective_num_neighbors * self.num_radii
        self.se_fusion = SEBlock(total_neighbors)
        self.bin_centers = nn.Parameter(torch.linspace(0.0, 1.0, num_bins))
        self.agg_mlp = nn.Sequential(nn.Linear(total_neighbors, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 1))
        self.global_pool = nn.Sequential(
            GeMPooling(),
            nn.Flatten(1),
            nn.Linear(3, feature_dim),
            nn.LayerNorm(feature_dim)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert C == 3 and H == self.block_size and W == self.block_size, \
            f"Input shape must be [B, 3, {self.block_size}, {self.block_size}]"
        
        current_device = x.device
        radii_param_dev = self.radius_param.to(current_device)
        bin_centers_dev = self.bin_centers.to(current_device)

        radii = self.r_min + (self.r_max - self.r_min) * torch.sigmoid(radii_param_dev)
        bin_size = F.softplus(self.bin_size_param) + 1e-6

        # 向量化梯度计算
        gx = self.conv_x(x)
        gy = self.conv_y(x)
        magnitude = torch.sqrt(gx * gx + gy * gy + 1e-8)
        grad_map = self.norm(magnitude)

        # 向量化偏移预测与采样
        offsets = self.offset_conv(grad_map)
        offsets = offsets.view(B, 3, self.num_radii, self.effective_num_neighbors, 2, H, W)
        offsets = offsets.permute(0, 1, 5, 6, 2, 3, 4).contiguous()
        off_ri = torch.tanh(offsets) * self.max_offset
        radii_exp = radii.view(1, 1, 1, 1, self.num_radii, 1, 1)
        off_ri = off_ri * radii_exp
        
        factor_x = 2.0 / (W - 1) if W > 1 else 0.0
        factor_y = 2.0 / (H - 1) if H > 1 else 0.0
        norm_factor = torch.tensor([factor_x, factor_y], device=current_device).view(1, 1, 1, 1, 1, 1, 2)
        off_norm = off_ri * norm_factor

        ys = torch.linspace(-1.0, 1.0, H, device=current_device)
        xs = torch.linspace(-1.0, 1.0, W, device=current_device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        base_grid = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, H, W, 1, 1, 2)
        
        grid = (base_grid + off_norm).view(B, 3, H, W, -1, 2)

        total_neighbors = self.num_radii * self.effective_num_neighbors
        grad_map_reshaped = grad_map.view(B * 3, 1, H, W)
        grid_reshaped = grid.view(B * 3, H * W, total_neighbors, 2)

        samples = F.grid_sample(
            grad_map_reshaped, grid_reshaped,
            mode=self.sample_mode, padding_mode='border', align_corners=True
        )
        samples = samples.squeeze(1)
        samples_fused = self.se_fusion(samples)

        # 软量化与聚合
        if self.use_soft_quant:
            temp = F.softplus(self.soft_temp_param) + 1e-6
            normed = samples_fused / bin_size
            
            dists = (normed.unsqueeze(-1) - bin_centers_dev)**2
            weights = F.softmax(-dists / temp, dim=-1)
            soft_q = (weights * bin_centers_dev).sum(dim=-1)
        else:
            soft_q = samples_fused

        pixel_vals = self.agg_mlp(soft_q).view(B * 3, 1, H, W)
        all_features = pixel_vals.view(B, 3, H, W)

        # 全局池化与投影
        out = self.global_pool(all_features)
        out = F.normalize(out, p=2, dim=1)
        return out