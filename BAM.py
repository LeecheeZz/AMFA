import torch
import torch.nn as nn

class BAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):  # 默认 kernel_size=7
        super(BAM, self).__init__()

        # 1. Channel Attention Branch ----------------------------------------
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化 (global average pooling)
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),  # 减少通道数
            nn.BatchNorm2d(in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False),  # 恢复通道数
            nn.Sigmoid()  # 激活函数，生成通道权重
        )

        # 2. Spatial Attention Branch -----------------------------------------
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),  # 减少通道数
            nn.BatchNorm2d(in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),  # 空间注意力，kernel_size 决定感受野
            nn.Sigmoid()  # 激活函数，生成空间权重
        )


    def forward(self, x):
        # 1. Channel Attention
        channel_att_map = self.channel_att(x)  # 生成通道注意力图
        # 对特征图 x 进行通道注意力加权
        out = x * channel_att_map

        # 2. Spatial Attention
        spatial_att_map = self.spatial_att(x)  # 生成空间注意力图，对原始特征图加权
        out = out*spatial_att_map

        return out