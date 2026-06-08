import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


def autopad(kernel_size: Union[int, Tuple[int, ...]], 
            padding: Optional[Union[int, Tuple[int, ...]]] = None, 
            dilation: int = 1) -> Union[int, Tuple[int, ...]]:
    """
    自动计算填充大小以保持输出形状不变。
    
    Args:
        kernel_size: 卷积核大小
        padding: 填充大小，如果为None则自动计算
        dilation: 膨胀率
        
    Returns:
        计算得到的填充大小
    """
    if dilation > 1:
        if isinstance(kernel_size, int):
            kernel_size = dilation * (kernel_size - 1) + 1
        else:
            kernel_size = tuple(dilation * (x - 1) + 1 for x in kernel_size)
    
    if padding is None:
        if isinstance(kernel_size, int):
            padding = kernel_size // 2
        else:
            padding = tuple(x // 2 for x in kernel_size)
    
    return padding


class ConvMSI(nn.Module):
    """
    多光谱图像3D卷积模块。
    
    该模块专门设计用于处理多光谱图像数据，通过3D卷积提取时空特征，
    然后通过深度可分离卷积融合时间维度信息，最终输出2D特征图。
    
    Attributes:
        default_act: 默认激活函数
    """
    
    default_act = nn.ReLU()

    def __init__(self, 
                 c1: int,
                 c2: int, 
                 c3: int = 8,
                 k: Tuple[int, int, int] = (3, 7, 7),
                 s: Tuple[int, int, int] = (1, 2, 2),
                 p: Tuple[int, int, int] = (1, 3, 3),
                 groups: Optional[int] = None,
                 use_bn_3d: bool = True,
                 use_gn_3d: bool = False,
                 final_bn: bool = True,
                 final_act: bool = True) -> None:
        """
        初始化ConvMSI模块。
        
        Args:
            c1: 输入通道数，必须为1
            c2: 输出通道数
            c3: 光谱通道数（多光谱图像的波段数）
            k: 3D卷积核尺寸 (temporal, height, width)
            s: 3D卷积步长 (temporal_stride, height_stride, width_stride)
            p: 3D卷积填充 (temporal_pad, height_pad, width_pad)
            groups: 深度可分离卷积的分组数，默认为c2
            use_bn_3d: 是否使用3D批归一化
            use_gn_3d: 是否使用3D组归一化
            final_bn: 是否使用最终的2D批归一化
            final_act: 是否使用最终的激活函数
            
        Raises:
            AssertionError: 当c1不为1或c3不大于1时
        """
        super().__init__()
        
        # 参数验证
        assert c1 == 1, f'c1 must be 1, got {c1}'
        assert c3 > 1, f'c3 must be > 1, got {c3}'
        
        # 计算3D卷积后的时间维度输出大小
        temporal_output_size = self._calculate_temporal_output_size(c3, p[0], k[0], s[0])
        
        # 第一个3D卷积层
        self.conv3d = nn.Conv3d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=p,
            bias=False
        )
        
        # 3D归一化层
        if use_bn_3d:
            self.bn3d = nn.BatchNorm3d(c2)
            self.gn3d = None
        elif use_gn_3d:
            self.bn3d = None
            self.gn3d = nn.GroupNorm(16, c2)
        else:
            self.bn3d = None
            self.gn3d = None
        
        # 深度方向融合卷积（深度可分离卷积）
        self.fuse = nn.Conv3d(
            in_channels=c2,
            out_channels=c2,
            kernel_size=(temporal_output_size, 1, 1),
            groups=groups or c2,
            bias=False
        )
        
        # 最终2D归一化层
        if final_bn:
            self.bn2d = nn.BatchNorm2d(c2)
        else:
            self.bn2d = None
        
        # 激活函数
        self.act = self.default_act
        
        # 保存配置参数
        self.final_act = final_act
        self.final_bn = final_bn
        self.use_bn_3d = use_bn_3d
        self.use_gn_3d = use_gn_3d

    def _calculate_temporal_output_size(self, 
                                      c3: int,
                                      temporal_padding: int,
                                      temporal_kernel: int,
                                      temporal_stride: int) -> int:
        """
        计算3D卷积后时间维度的输出大小。
        
        Args:
            c3: 光谱通道数
            temporal_padding: 时间维度填充
            temporal_kernel: 时间维度卷积核大小
            temporal_stride: 时间维度步长
            
        Returns:
            时间维度的输出大小
        """
        return math.floor((c3 + 2 * temporal_padding - 
                          (temporal_kernel - 1) - 1) / temporal_stride) + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: 输入张量，形状为 [B, c3, H, W] 或 [B, 1, c3, H, W]
            
        Returns:
            输出张量，形状为 [B, c2, H', W']
        """
        # 处理输入维度
        if x.ndim == 4:
            x = x.unsqueeze(1)  # [B, c3, H, W] -> [B, 1, c3, H, W]
        
        # 3D卷积 + 归一化 + 激活
        x = self.conv3d(x)
        
        if self.bn3d is not None:
            x = self.bn3d(x)
        elif self.gn3d is not None:
            x = self.gn3d(x)
        
        x = self.act(x)
        
        # 深度方向融合 -> [B, c2, 1, H', W']
        x = self.fuse(x)
        
        # 压缩时间维度 -> [B, c2, H', W']
        x = x.squeeze(2)
        # 最后 BN2d + SiLU
        if self.bn2d is not None:
            x = self.bn2d(x)
            if self.final_act:
                x = self.act(x)
        return x



class ConvMSI_SE(nn.Module):
    """
    多光谱图像3D卷积模块，fuse被pixel-wise SE加权+D维求和替代。
    """
    default_act = nn.ReLU()

    def __init__(self, 
                 c1: int,
                 c2: int, 
                 c3: int = 8,
                 k: Tuple[int, int, int] = (3, 7, 7),
                 s: Tuple[int, int, int] = (1, 2, 2),
                 p: Tuple[int, int, int] = (1, 3, 3),
                 use_bn_3d: bool = True,
                 use_gn_3d: bool = False,
                 final_bn: bool = True,
                 final_act: bool = True,
                 reduction: int = 4,
                 return_before_sigmoid: bool = False) -> None:
        super().__init__()
        assert c1 == 1, f'c1 must be 1, got {c1}'
        assert c3 > 1, f'c3 must be > 1, got {c3}'
        self.conv3d = nn.Conv3d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=p,
            bias=False
        )
        if use_bn_3d:
            self.bn3d = nn.BatchNorm3d(c2)
            self.gn3d = None
        elif use_gn_3d:
            self.bn3d = None
            self.gn3d = nn.GroupNorm(16, c2)
        else:
            self.bn3d = None
            self.gn3d = None

        temporal_output_size = self._calculate_temporal_output_size(c3, p[0], k[0], s[0])
        # pixel-wise SE模块
        self.se_conv1 = nn.Conv2d(temporal_output_size, temporal_output_size//reduction, kernel_size=3, padding=1)
        self.se_conv2 = nn.Conv2d(temporal_output_size//reduction, temporal_output_size, kernel_size=3, padding=1)

        if final_bn:
            self.bn2d = nn.BatchNorm2d(c2)
        else:
            self.bn2d = None
        self.act = self.default_act
        self.final_act = final_act
        self.final_bn = final_bn
        self.use_bn_3d = use_bn_3d
        self.use_gn_3d = use_gn_3d

        #是否在返回sigmoid之前的se_weight
        self.return_before_sigmoid = return_before_sigmoid


    def _calculate_temporal_output_size(self, 
                                      c3: int,
                                      temporal_padding: int,
                                      temporal_kernel: int,
                                      temporal_stride: int) -> int:
        """
        计算3D卷积后时间维度的输出大小。
        
        Args:
            c3: 光谱通道数
            temporal_padding: 时间维度填充
            temporal_kernel: 时间维度卷积核大小
            temporal_stride: 时间维度步长
            
        Returns:
            时间维度的输出大小
        """
        return math.floor((c3 + 2 * temporal_padding - 
                          (temporal_kernel - 1) - 1) / temporal_stride) + 1

    def forward(self, x: torch.Tensor):

        if x.ndim == 4:
            x = x.unsqueeze(1)

        x = self.conv3d(x)

        if self.bn3d is not None:
            x = self.bn3d(x)
        elif self.gn3d is not None:
            x = self.gn3d(x)

        x = self.act(x)

        # spectral signature branch
        x_se = x.mean(dim=1)

        sig_raw = self.act(self.se_conv1(x_se))
        sig_raw = self.se_conv2(sig_raw)

        gate = torch.sigmoid(sig_raw)

        # feature fusion
        x = x * gate.unsqueeze(1)
        x = x.sum(dim=2)

        if self.bn2d is not None:
            x = self.bn2d(x)
            if self.final_act:
                x = self.act(x)

        if self.return_before_sigmoid:
            return x, sig_raw     # S0
        else:
            return x, gate