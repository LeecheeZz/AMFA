import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """通道注意力 (Channel Attention, CA)"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 最大池化
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out  # 融合池化信息
        return x * self.sigmoid(out).view(b, c, 1, 1)  # 加权输入


class SpatialAttention(nn.Module):
    """空间注意力 (Spatial Attention, SA)"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 按通道求平均
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 按通道求最大值
        out = torch.cat([avg_out, max_out], dim=1)  # 组合两个特征图
        return x * self.sigmoid(self.conv(out))  # 计算空间注意力并加权输入


class CBAM(nn.Module):
    """CBAM: 结合通道注意力和空间注意力"""
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)  # 先计算通道注意力
        x = self.spatial_att(x)  # 再计算空间注意力
        return x
