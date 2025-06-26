import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from timm.models import create_model
from .backbones.model_convnext import convnext_tiny
from .backbones.resnet import Resnet
from torch.nn import init
from torch.nn.parameter import Parameter
import math
from sample4geo.Utils import init
from typing import Optional, Sequence
from BAM import BAM
from TripletAttention import TripletAttention
from scSE import SCSE
from CBAM import CBAM

class SA(nn.Module):
    """
    Synergistic Attention (SA)
    Args:
        in_channels (int): 
        reduction_ratio (int): 
        strip_kernel_size (int): 
    """
    def __init__(self, in_channels, reduction_ratio=16, strip_kernel_size=5):
        super().__init__()
        self.in_channels = in_channels
        mid_channels = max(in_channels // reduction_ratio, 4)  # 最小通道数限制
        # ---------------------- 空间注意力分支 ----------------------
        # 水平条带池化 + 1D卷积
        self.h_pool = nn.AdaptiveAvgPool2d((None, 1))  # (H,1)
        self.h_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=strip_kernel_size, 
                      padding=strip_kernel_size//2),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, in_channels, kernel_size=strip_kernel_size, 
                      padding=strip_kernel_size//2),
        )
        
        # 垂直条带池化 + 1D卷积
        self.v_pool = nn.AdaptiveAvgPool2d((1, None))  # (1,W)
        self.v_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=strip_kernel_size,
                      padding=strip_kernel_size//2),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, in_channels, kernel_size=strip_kernel_size, 
                      padding=strip_kernel_size//2),
        )
        
        # 空间注意力融合
        self.spatial_fusion = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1))
        
        # ---------------------- 通道注意力分支 ----------------------
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(in_channels, in_channels//4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # ---------------------- 协同门控机制 ----------------------
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//reduction_ratio, kernel_size=1),
            nn.BatchNorm2d(in_channels//reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//reduction_ratio, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        
        # ===================== 空间注意力计算 =====================
        # 水平条带分支
        x_h = self.h_pool(x).squeeze(-1)        # [B,C,H]
        x_h = self.h_conv(x_h).unsqueeze(-1)    # [B,C,H,1]
        
        # 垂直条带分支
        x_v = self.v_pool(x).squeeze(-2)        # [B,C,W]
        x_v = self.v_conv(x_v).unsqueeze(-2)    # [B,C,1,W]
        
        # 空间注意力融合
        spatial_attn = self.spatial_fusion(x_h + x_v)  # [B,C,H,W]
        spatial_attn = self.sigmoid(spatial_attn)
        
        # ===================== 通道注意力计算 =====================
        channel_attn = self.channel_attention(x)  # [B,C,1,1]
        
        # ===================== 协同注意力 =====================
        # 门控权重生成
        channel_attn = channel_attn.expand_as(spatial_attn)
        # gate_input = torch.cat([spatial_attn, channel_attn], dim=1)
        gate_input = spatial_attn * channel_attn
        gate_weight = self.gate(gate_input)  # [B,1,H,W]
        
        # 加权融合
        fused_attn = gate_weight * spatial_attn + (1 - gate_weight) * channel_attn
        return fused_attn
    
class AMFA(nn.Module):
    """Bottleneck with Inception module - Torch Version"""

    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            kernel_sizes: Sequence[int] = (1, 3, 5, 7, 9), # (3, 5, 7, 9, 11)
            dilations: Sequence[int] = (1, 1, 1, 1, 1),
            expansion: float = 1.0, # 1.0
            add_identity: bool = True,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='LeakyReLU'),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.expansion = expansion
        self.add_identity = add_identity
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.hidden_channels = make_divisible(int(self.out_channels * self.expansion), 8) # 24

        self.pre_conv = self._make_conv_module(in_channels, self.hidden_channels, kernel_size=1, padding=0,
                                                norm_cfg=norm_cfg, act_cfg=act_cfg)


        self.dw_conv = self._make_conv_module(self.hidden_channels, self.hidden_channels, kernel_size=kernel_sizes[0],
                                                padding=autopad(kernel_sizes[0], None, dilations[0]),
                                                groups=self.hidden_channels, norm_cfg=None, act_cfg=None, dilation=dilations[0])
        self.dw_conv1 = self._make_conv_module(self.hidden_channels, self.hidden_channels, kernel_size=kernel_sizes[1],
                                                padding=autopad(kernel_sizes[1], None, dilations[1]),
                                                groups=self.hidden_channels, norm_cfg=None, act_cfg=None, dilation=dilations[1])
        self.dw_conv2 = self._make_conv_module(self.hidden_channels, self.hidden_channels, kernel_size=kernel_sizes[2],
                                                padding=autopad(kernel_sizes[2], None, dilations[2]),
                                                groups=self.hidden_channels, norm_cfg=None, act_cfg=None, dilation=dilations[2])
        self.dw_conv3 = self._make_conv_module(self.hidden_channels, self.hidden_channels, kernel_size=kernel_sizes[3],
                                                padding=autopad(kernel_sizes[3], None, dilations[3]),
                                                groups=self.hidden_channels, norm_cfg=None, act_cfg=None, dilation=dilations[3])
        self.dw_conv4 = self._make_conv_module(self.hidden_channels, self.hidden_channels, kernel_size=kernel_sizes[4],
                                                padding=autopad(kernel_sizes[4], None, dilations[4]),
                                                groups=self.hidden_channels, norm_cfg=None, act_cfg=None, dilation=dilations[4])

        self.fconv = nn.Sequential(self._make_conv_module(self.hidden_channels * 5, self.hidden_channels, kernel_size=1,
                                                padding=autopad(1, None, 1),
                                                norm_cfg=norm_cfg, act_cfg=act_cfg, dilation=1),
                                    self._make_conv_module(self.hidden_channels, self.hidden_channels, kernel_size=3,
                                                padding=autopad(3, None, 1),groups=self.hidden_channels,
                                                norm_cfg=norm_cfg, act_cfg=act_cfg, dilation=1))

        self.pw_conv = self._make_conv_module(self.hidden_channels, self.hidden_channels, kernel_size=1, padding=0,
                                                norm_cfg=norm_cfg, act_cfg=act_cfg) # 1 * 1 卷积

        self.sa_factor = SA(in_channels=self.hidden_channels)
        
        self.add_identity = add_identity and in_channels == self.out_channels

        self.post_conv = self._make_conv_module(self.hidden_channels, self.out_channels, kernel_size=1, padding=0,
                                                 norm_cfg=norm_cfg, act_cfg=act_cfg)
    

    def _make_conv_module(self, in_channels, out_channels, kernel_size, padding, groups=1, norm_cfg=None, act_cfg=None, dilation=1):
        """Helper function to create a convolutional module with optional normalization and activation."""
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, groups=groups, bias= norm_cfg is None, dilation=dilation))
        if norm_cfg:
            if norm_cfg['type'] == 'BN':
                layers.append(nn.BatchNorm2d(out_channels, momentum=norm_cfg['momentum'], eps=norm_cfg['eps']))
            elif norm_cfg['type'] == 'LN':
                layers.append(nn.LayerNorm(out_channels))
        if act_cfg:
            if act_cfg['type'] == 'SiLU':
                layers.append(nn.SiLU())
            elif act_cfg['type'] == 'ReLU':
                 layers.append(nn.ReLU())
            elif act_cfg['type'] == 'LeakyReLU':
                 layers.append(nn.LeakyReLU())
            else:
                raise NotImplementedError

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre_conv(x) # 降维

        y = x  # if there is an inplace operation of x, use y = x.clone() instead of y = x

        # x = self.dw_conv(x)
        # x = self.dw_conv(x) + self.dw_conv1(x) + self.dw_conv2(x) + self.dw_conv3(x) + self.dw_conv4(x)
        x = torch.cat([self.dw_conv(x), self.dw_conv1(x), self.dw_conv2(x), self.dw_conv3(x), self.dw_conv4(x)], dim=1)
        x = self.fconv(x)

        x = self.pw_conv(x)

        y = self.sa_factor(y)

        if self.add_identity:
            y = x * y  # 或者用element-wise相乘
            x = x + y
        else:
            x = x * y

        x = self.post_conv(x) # 升维
        return x
    
# 实现 make_divisible 功能
def make_divisible(v, divisor=8, min_value=None):
    """
    确保一个数值可以被 divisor 整除.

    Args:
        v (int): 输入值.
        divisor (int): 除数 (默认为 8).
        min_value (int): 最小值 (默认为 None).

    Returns:
        int: 可以被 divisor 整除的数值.
    """
    if min_value is None:
        min_value = divisor # 8
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # 确保下限不低于 10%
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# 实现 autopad 功能
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """
    自动计算合适的 padding 大小, 保持卷积操作后的输出尺寸不变.
    Args:
        k (int): 卷积核大小.
        p (int, optional): 预先指定的 padding 大小 (默认为 None).
        d (int): 扩张率 (默认为 1).

    Returns:
        int: 计算得到的 padding 大小.
    """
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p
#-----------------------------------------------------------------------------------------------

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True,
                 return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.training:
            if self.return_f:
                f = x
                x = self.classifier(x)
                return x, f
            else:
                x = self.classifier(x)
                return x
        else:
            return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)


class MLP1D(nn.Module):
    """
    The non-linear neck in byol: fc-bn-relu-fc
    """
    def __init__(self, in_channels, hid_channels, out_channels,
                 norm_layer=None, bias=False, num_mlp=2):
        super(MLP1D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        mlps = []
        for _ in range(num_mlp-1):
            mlps.append(nn.Conv1d(in_channels, hid_channels, 1, bias=bias))
            mlps.append(norm_layer(hid_channels))
            mlps.append(nn.ReLU(inplace=True))
            in_channels = hid_channels
        mlps.append(nn.Conv1d(hid_channels, out_channels, 1, bias=bias))
        self.mlp = nn.Sequential(*mlps)

    def init_weights(self, init_linear='kaiming'): # origin is 'normal'
        init.init_weights(self, init_linear)

    def forward(self, x):
        x = self.mlp(x)
        return x


class build_convnext(nn.Module):
    def __init__(self, num_classes, block=4, return_f=False, resnet=False):
        super(build_convnext, self).__init__()
        self.return_f = return_f
        if resnet:
            convnext_name = "resnet101"
            print('using model_type: {} as a backbone'.format(convnext_name))
            self.in_planes = 2048
            self.convnext = Resnet(pretrained=True)
        else:
            convnext_name = "convnext_base"
            print('using model_type: {} as a backbone'.format(convnext_name))
            if 'base' in convnext_name:
                self.in_planes = 1024
            elif 'large' in convnext_name:
                self.in_planes = 1536
            elif 'xlarge' in convnext_name:
                self.in_planes = 2048
            else:
                self.in_planes = 768
            self.convnext = create_model(convnext_name, pretrained=True)
            
        self.num_classes = num_classes
        self.classifier = ClassBlock(self.in_planes, num_classes, 0.5, return_f=return_f)
        self.block = block

        self.amfa = AMFA(in_channels=self.in_planes)
        # self.bam = BAM(in_channels=self.in_planes)
        # self.trip = TripletAttention()
        # self.scse = SCSE(in_channels=self.in_planes)
        # self.cbam = CBAM(in_channels=self.in_planes)
        # self.cbam = TripletAttention(in_channels =self.in_planes)
        # self.rep = RepBlock(in_channels=self.in_planes, out_channels=self.in_planes)
        # for i in range(self.block):
        #     name = 'classifier_mcb' + str(i + 1)
        #     setattr(self, name, ClassBlock(self.in_planes, num_classes, 0.5, return_f=self.return_f))

    def forward(self, x):
        # -- backbone feature extractor
        gap_feature, part_features = self.convnext(x)
        main_feature = gap_feature.mean([-2, -1])
        # main_feature = gap_feature.mean([-2, -1])

        #--------------- AMFA ------------------
        gap_feature = self.amfa(gap_feature)
        #---------------------------------------------
        # gap_feature = self.cbam(gap_feature)
        # -- Training
        if self.training:
            convnext_feature = self.classifier(gap_feature.mean([-2, -1]))  # class: (bs, 701); feature: (bs, 512)  

            y = [convnext_feature]
            if self.return_f:  # return_f是triplet loss的设置，0.3
                cls, features = [], []
                for i in y:
                    cls.append(i[0])
                    features.append(i[1])
                # return cea_feature, cls, features, main_feature, part_features
                return convnext_feature, cls, features, main_feature, part_features

        # -- Eval
        else:
            # ffeature = convnext_feature.view(convnext_feature.size(0), -1, 1)
            # y = torch.cat([y, ffeature], dim=2)
            pass

        # return gap_feature, part_features
        return main_feature, part_features

    # def part_classifier(self, block, x, cls_name='classifier_mcb'):
    #     part = {}
    #     predict = {}
    #     for i in range(block):
    #         part[i] = x[:, :, i].view(x.size(0), -1)
    #         name = cls_name + str(i + 1)
    #         c = getattr(self, name)
    #         predict[i] = c(part[i])
    #     y = []
    #     for i in range(block):
    #         y.append(predict[i])
    #     if not self.training:
    #         return torch.stack(y, dim=2)
    #     return y

    def fine_grained_transform(self):

        pass


def make_convnext_model(num_class, block=4, return_f=False, resnet=False):
    print('===========building convnext===========')
    model = build_convnext(num_class, block=block, return_f=return_f, resnet=resnet)
    return model
