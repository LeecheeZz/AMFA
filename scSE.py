import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelSE(nn.Module):
    """通道注意力 (SE Block)"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelSE, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)  # 全局平均池化
        y = self.fc(y).view(b, c, 1, 1)  # 计算注意力权重
        return x * y  # 逐通道缩放


class SpatialSE(nn.Module):
    """空间注意力"""
    def __init__(self, in_channels):
        super(SpatialSE, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)  # 1x1 卷积生成空间注意力
        y = self.sigmoid(y)  # 归一化
        return x * y  # 逐像素缩放


class SCSE(nn.Module):
    """SCSE: 结合通道和空间注意力"""
    def __init__(self, in_channels, reduction=16):
        super(SCSE, self).__init__()
        self.channel_se = ChannelSE(in_channels, reduction)
        self.spatial_se = SpatialSE(in_channels)

    def forward(self, x):
        return self.channel_se(x) + self.spatial_se(x)  # 融合通道和空间注意力