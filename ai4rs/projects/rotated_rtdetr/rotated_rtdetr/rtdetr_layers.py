import math
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule
from torch import Tensor, nn

from mmdet.models.layers.transformer.detr_layers import DetrTransformerEncoder
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig


class RepVGGBlock(nn.Module):
    """RepVGGBlock is a basic rep-style block, including training and deploy
    status This code is based on
    https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple): Stride of the convolution. Default: 1
        padding (int, tuple): Padding added to all four sides of
            the input. Default: 1
        dilation (int or tuple): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        padding_mode (string, optional): Default: 'zeros'
        use_se (bool): Whether to use se. Default: False
        use_alpha (bool): Whether to use `alpha` parameter at 1x1 conv.
            In PPYOLOE+ model backbone, `use_alpha` will be set to True.
            Default: False.
        use_bn_first (bool): Whether to use bn layer before conv.
            In YOLOv6 and YOLOv7, this will be set to True.
            In PPYOLOE, this will be set to False.
            Default: True.
        deploy (bool): Whether in deploy mode. Default: False
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int]] = 3,
                 stride: Union[int, Tuple[int]] = 1,
                 padding: Union[int, Tuple[int]] = 1,
                 dilation: Union[int, Tuple[int]] = 1,
                 groups: Optional[int] = 1,
                 padding_mode: Optional[str] = 'zeros',
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 use_se: bool = False,
                 use_alpha: bool = False,
                 use_bn_first=True,
                 deploy: bool = False):
        super().__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = MODELS.build(act_cfg)

        if use_se:
            raise NotImplementedError('se block not supported yet')
        else:
            self.se = nn.Identity()

        if use_alpha:
            alpha = torch.ones([
                1,
            ], dtype=torch.float32, requires_grad=True)
            self.alpha = nn.Parameter(alpha, requires_grad=True)
        else:
            self.alpha = None

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode)

        else:
            if use_bn_first and (out_channels == in_channels) and stride == 1:
                self.rbr_identity = build_norm_layer(
                    norm_cfg, num_features=in_channels)[1]
            else:
                self.rbr_identity = None

            self.rbr_dense = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None)
            self.rbr_1x1 = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding_11,
                groups=groups,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward process.
        Args:
            inputs (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        if self.alpha:
            return self.nonlinearity(
                self.se(
                    self.rbr_dense(inputs) +
                    self.alpha * self.rbr_1x1(inputs) + id_out))
        else:
            return self.nonlinearity(
                self.se(
                    self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        """Derives the equivalent kernel and bias in a differentiable way.

        Returns:
            tuple: Equivalent kernel and bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        if self.alpha:
            return kernel3x3 + self.alpha * self._pad_1x1_to_3x3_tensor(
                kernel1x1) + kernelid, bias3x3 + self.alpha * bias1x1 + biasid
        else:
            return kernel3x3 + self._pad_1x1_to_3x3_tensor(
                kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pad 1x1 tensor to 3x3.
        Args:
            kernel1x1 (Tensor): The input 1x1 kernel need to be padded.

        Returns:
            Tensor: 3x3 kernel after padded.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: nn.Module) -> Tuple[np.ndarray, Tensor]:
        """Derives the equivalent kernel and bias of a specific branch layer.

        Args:
            branch (nn.Module): The layer that needs to be equivalently
                transformed, which can be nn.Sequential or nn.Batchnorm2d

        Returns:
            tuple: Equivalent kernel and bias
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, ConvModule):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, (nn.SyncBatchNorm, nn.BatchNorm2d))
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3),
                                        dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(
                    branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        """Switch to deploy mode."""
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class CSPLayer(BaseModule):
    """CSPLayer from RTDETR.

    Args:
        in_channels (int): The input channels of the CSP layer.
        out_channels (int): The output channels of the CSP layer.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Defaults to 1.0.
        num_blocks (int): Number of blocks. Defaults to 3.
        conv_cfg (:obj:`ConfigDict`, optional): Config dict for convolution
            layer. Defaults to None, which means using conv2d.
        norm_cfg (:obj:`ConfigDict`, optional): Config dict for normalization
            layer. Defaults to dict(type='BN', requires_grad=True)
        act_cfg (:obj:`ConfigDict`, optional): Config dict for activation
            layer. Defaults to dict(type='SiLU', inplace=True)
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: float = 1.0,
                 num_blocks: int = 3,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.short_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.blocks = nn.Sequential(*[
            RepVGGBlock(
                in_channels=mid_channels,
                out_channels=mid_channels,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                use_bn_first=False) for _ in range(num_blocks)
        ])
        if mid_channels != out_channels:
            self.final_conv = ConvModule(
                mid_channels,
                out_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.final_conv = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)
        return self.final_conv(x_main + x_short)


class RTDETRFPN(BaseModule):
    """FPN of RTDETR.

    Args:
        in_channels (List[int], optional): The input channels of the
            feature maps. Defaults to [256, 256, 256].
        out_channels (int, optional): The output dimension of the MLP.
            Defaults to 256.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer.
            Defaults to 3.
        expansion (float, optional): The expansion of the CSPLayer.
            Defaults to 1.0.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(scale_factor=2, mode='nearest')`
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (:obj:`ConfigDict` or dict, optional): The config dict for
            normalization layers. Defaults to dict(type='BN').
        act_cfg (:obj:`ConfigDict` or dict, optional): The config dict for
            activation layers. Defaults to dict(type='SiLU', inplace=True).
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`], optional): Initialization config dict.
    """

    csp_block = CSPLayer

    def __init__(
        self,
        in_channels: List[int] = [256, 256, 256],
        out_channels: int = 256,
        num_csp_blocks: int = 3,
        expansion: float = 1.0,
        upsample_cfg: ConfigType = dict(scale_factor=2, mode='nearest'),
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
        act_cfg: OptConfigType = dict(type='SiLU', inplace=True),
        init_cfg: OptMultiConfig = dict(
            type='Kaiming',
            layer='Conv2d',
            a=math.sqrt(5),
            distribution='uniform',
            mode='fan_in',
            nonlinearity='leaky_relu')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # top-down fpn
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(
                    in_channels[idx],
                    in_channels[idx - 1],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.top_down_blocks.append(
                self.csp_block(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    expand_ratio=expansion,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                ConvModule(
                    in_channels[idx],
                    in_channels[idx],
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.bottom_up_blocks.append(
                self.csp_block(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    expand_ratio=expansion,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.out_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.out_convs.append(
                ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=None))

    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        """
        Args:
            inputs (tuple[Tensor]): input features.

        Returns:
            tuple[Tensor]: FPN features.
        """
        assert len(inputs) == len(self.in_channels)

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_high = self.reduce_layers[len(self.in_channels) - 1 - idx](
                feat_high)
            inner_outs[0] = feat_high

            if feat_high.dtype == torch.bfloat16:
                # PyTorch 2.0/CUDA 11.8 has no BF16 nearest2d kernel.
                with torch.autocast(
                        device_type=feat_high.device.type, enabled=False):
                    upsample_feat = self.upsample(feat_high.float()).to(
                        feat_high.dtype)
            else:
                upsample_feat = self.upsample(feat_high)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_high], 1))
            outs.append(out)

        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])

        return tuple(outs)


class PairTemporalAdapter(BaseModule):
    """Bidirectional cross-frame adapter for paired high-level features.

    The batch order must be ``[prev_0..prev_N, curr_0..curr_N]``.  ``gamma`` is
    initialized to zero, so the module is an exact identity at initialization.
    """

    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 4,
                 dropout: float = 0.0,
                 gamma_init: float = 0.0,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.query_norm = nn.LayerNorm(embed_dims)
        self.key_value_norm = nn.LayerNorm(embed_dims)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True)
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def _cross_attend(self, query: Tensor, key_value: Tensor) -> Tensor:
        query = self.query_norm(query)
        key_value = self.key_value_norm(key_value)
        delta = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=False)[0]
        return self.out_proj(delta)

    def forward(self, feat: Tensor) -> Tensor:
        batch_size, channels, height, width = feat.shape
        if batch_size % 2 != 0:
            raise ValueError(
                'PairTemporalAdapter expects an even batch ordered as '
                '[prev batch, curr batch].')
        if channels != self.embed_dims:
            raise ValueError(
                f'PairTemporalAdapter embed_dims={self.embed_dims}, '
                f'but got feature channels={channels}.')

        pair_batch = batch_size // 2
        prev = feat[:pair_batch]
        curr = feat[pair_batch:]
        prev_seq = prev.flatten(2).transpose(1, 2).contiguous()
        curr_seq = curr.flatten(2).transpose(1, 2).contiguous()

        delta_prev = self._cross_attend(prev_seq, curr_seq)
        delta_curr = self._cross_attend(curr_seq, prev_seq)
        delta_prev = delta_prev.transpose(1, 2).reshape(
            pair_batch, channels, height, width)
        delta_curr = delta_curr.transpose(1, 2).reshape(
            pair_batch, channels, height, width)

        return torch.cat(
            [prev + self.gamma * delta_prev,
             curr + self.gamma * delta_curr],
            dim=0)


class PairTemporalPoolGateAdapter(BaseModule):
    """Lightweight bidirectional temporal adapter using pooled P5 context.

    The batch order must be ``[prev_0..prev_N, curr_0..curr_N]``.  The module
    keeps an exact identity at initialization through a zero residual scale.
    """

    def __init__(self,
                 embed_dims: int = 256,
                 reduction: int = 4,
                 gamma_init: float = 0.0,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        hidden_dims = max(embed_dims // reduction, 16)
        self.context_mlp = nn.Sequential(
            nn.Linear(embed_dims * 4, hidden_dims),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dims, embed_dims),
            nn.Sigmoid(),
        )
        self.delta_conv = nn.Sequential(
            nn.Conv2d(
                embed_dims,
                embed_dims,
                kernel_size=3,
                padding=1,
                groups=embed_dims,
                bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.SiLU(inplace=True),
            nn.Conv2d(embed_dims, embed_dims, kernel_size=1, bias=True),
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def _delta(self, query: Tensor, context: Tensor) -> Tensor:
        query_pool = query.mean(dim=(-2, -1))
        context_pool = context.mean(dim=(-2, -1))
        gate_input = torch.cat(
            [
                query_pool,
                context_pool,
                context_pool - query_pool,
                query_pool * context_pool,
            ],
            dim=1)
        gate = self.context_mlp(gate_input).view(
            query.size(0), query.size(1), 1, 1)
        return self.delta_conv(context * gate)

    def forward(self, feat: Tensor) -> Tensor:
        batch_size, channels, _, _ = feat.shape
        if batch_size % 2 != 0:
            raise ValueError(
                'PairTemporalPoolGateAdapter expects an even batch ordered as '
                '[prev batch, curr batch].')
        if channels != self.embed_dims:
            raise ValueError(
                f'PairTemporalPoolGateAdapter embed_dims={self.embed_dims}, '
                f'but got feature channels={channels}.')

        pair_batch = batch_size // 2
        prev = feat[:pair_batch]
        curr = feat[pair_batch:]
        delta_prev = self._delta(prev, curr)
        delta_curr = self._delta(curr, prev)
        return torch.cat(
            [prev + self.gamma * delta_prev,
             curr + self.gamma * delta_curr],
            dim=0)


class PairTemporalPyramidLocalAdapter(BaseModule):
    """Bidirectional local temporal adapter for multi-scale pair features.

    The adapter runs on FPN outputs and is intentionally local: it uses pooled
    two-frame context to gate a lightweight depthwise/grouped convolution over
    the cross-frame feature difference.  ``gamma`` is zero-initialized per
    feature level, so the initial forward path is an exact identity.
    """

    def __init__(self,
                 in_channels: List[int],
                 level_indices: Optional[List[int]] = None,
                 reduction: int = 4,
                 pointwise_groups: int = 8,
                 gamma_init: float = 0.0,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = list(in_channels)
        if level_indices is None:
            level_indices = list(range(len(self.in_channels)))
        self.level_indices = [
            idx if idx >= 0 else len(self.in_channels) + idx
            for idx in level_indices
        ]
        if any(idx < 0 or idx >= len(self.in_channels)
               for idx in self.level_indices):
            raise ValueError(
                f'Invalid level_indices={level_indices} for '
                f'{len(self.in_channels)} input levels.')

        self.gate_mlps = nn.ModuleList()
        self.local_blocks = nn.ModuleList()
        for idx in self.level_indices:
            channels = self.in_channels[idx]
            hidden_dims = max(channels // reduction, 16)
            groups = min(pointwise_groups, channels)
            while channels % groups != 0:
                groups -= 1
            self.gate_mlps.append(
                nn.Sequential(
                    nn.Linear(channels * 4, hidden_dims),
                    nn.SiLU(inplace=True),
                    nn.Linear(hidden_dims, channels),
                    nn.Sigmoid(),
                ))
            self.local_blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=3,
                        padding=1,
                        groups=channels,
                        bias=False),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=1,
                        groups=groups,
                        bias=True),
                ))
        self.gamma = nn.Parameter(
            torch.full((len(self.level_indices), ), float(gamma_init)))

    def _delta(self, query: Tensor, context: Tensor, module_idx: int) -> Tensor:
        query_pool = query.mean(dim=(-2, -1))
        context_pool = context.mean(dim=(-2, -1))
        diff_pool = context_pool - query_pool
        gate_input = torch.cat(
            [
                query_pool,
                context_pool,
                diff_pool,
                query_pool * context_pool,
            ],
            dim=1)
        gate = self.gate_mlps[module_idx](gate_input).view(
            query.size(0), query.size(1), 1, 1)
        return self.local_blocks[module_idx](context - query) * gate

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor]:
        outs = list(feats)
        if not outs:
            return feats
        batch_size = outs[0].shape[0]
        if batch_size % 2 != 0:
            raise ValueError(
                'PairTemporalPyramidLocalAdapter expects an even batch '
                'ordered as [prev batch, curr batch].')
        pair_batch = batch_size // 2

        for module_idx, level_idx in enumerate(self.level_indices):
            feat = outs[level_idx]
            channels = feat.shape[1]
            expected_channels = self.in_channels[level_idx]
            if channels != expected_channels:
                raise ValueError(
                    'PairTemporalPyramidLocalAdapter expected '
                    f'{expected_channels} channels at level {level_idx}, '
                    f'but got {channels}.')
            prev = feat[:pair_batch]
            curr = feat[pair_batch:]
            delta_prev = self._delta(prev, curr, module_idx)
            delta_curr = self._delta(curr, prev, module_idx)
            gamma = self.gamma[module_idx].view(1, 1, 1, 1)
            outs[level_idx] = torch.cat(
                [prev + gamma * delta_prev, curr + gamma * delta_curr],
                dim=0)
        return tuple(outs)


class PairTemporalPyramidCommonDetailAdapter(BaseModule):
    """Order-equivariant local temporal adapter with pair-mean preservation.

    Each pyramid level is decomposed into a pair-common feature and a signed
    temporal detail.  An order-invariant descriptor gates an odd local
    transform of the detail, then equal and opposite residuals are applied to
    the two frames.  Consequently, swapping the two input frames swaps the
    outputs and the per-location pair mean is preserved exactly.
    """

    def __init__(self,
                 in_channels: List[int],
                 level_indices: Optional[List[int]] = None,
                 reduction: int = 4,
                 pointwise_groups: int = 8,
                 conserve_detail_energy: bool = False,
                 use_spatial_reliability: bool = False,
                 use_shared_scalar_gain: bool = False,
                 gamma_init: float = 0.0,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = list(in_channels)
        self.conserve_detail_energy = bool(conserve_detail_energy)
        self.use_spatial_reliability = bool(use_spatial_reliability)
        self.use_shared_scalar_gain = bool(use_shared_scalar_gain)
        if level_indices is None:
            level_indices = list(range(len(self.in_channels)))
        self.level_indices = [
            idx if idx >= 0 else len(self.in_channels) + idx
            for idx in level_indices
        ]
        if any(idx < 0 or idx >= len(self.in_channels)
               for idx in self.level_indices):
            raise ValueError(
                f'Invalid level_indices={level_indices} for '
                f'{len(self.in_channels)} input levels.')

        self.gate_mlps = nn.ModuleList()
        self.local_blocks = nn.ModuleList()
        self.spatial_gates = nn.ModuleList()
        self.shared_gain_gates = nn.ModuleList()
        for idx in self.level_indices:
            channels = self.in_channels[idx]
            hidden_dims = max(channels // reduction, 16)
            groups = min(pointwise_groups, channels)
            while channels % groups != 0:
                groups -= 1
            self.gate_mlps.append(
                nn.Sequential(
                    nn.Linear(channels * 2, hidden_dims),
                    nn.SiLU(inplace=True),
                    nn.Linear(hidden_dims, channels),
                    nn.Sigmoid(),
                ))
            # Bias-free linear convolutions followed by tanh form an odd
            # transform, which makes the adapter equivariant to frame swaps.
            self.local_blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=3,
                        padding=1,
                        groups=channels,
                        bias=False),
                    nn.Tanh(),
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=1,
                        groups=groups,
                        bias=False),
                ))
            if self.use_spatial_reliability:
                spatial_gate = nn.Conv2d(
                    2, 1, kernel_size=3, padding=1, bias=True)
                nn.init.zeros_(spatial_gate.weight)
                nn.init.zeros_(spatial_gate.bias)
                self.spatial_gates.append(spatial_gate)
            if self.use_shared_scalar_gain:
                shared_gain_gate = nn.Conv2d(
                    2, 1, kernel_size=3, padding=1, bias=True)
                nn.init.zeros_(shared_gain_gate.weight)
                nn.init.zeros_(shared_gain_gate.bias)
                self.shared_gain_gates.append(shared_gain_gate)
        self.gamma = nn.Parameter(
            torch.full((len(self.level_indices), ), float(gamma_init)))

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor]:
        outs = list(feats)
        if not outs:
            return feats
        batch_size = outs[0].shape[0]
        if batch_size % 2 != 0:
            raise ValueError(
                'PairTemporalPyramidCommonDetailAdapter expects an even batch '
                'ordered as [prev batch, curr batch].')
        pair_batch = batch_size // 2

        for module_idx, level_idx in enumerate(self.level_indices):
            feat = outs[level_idx]
            channels = feat.shape[1]
            expected_channels = self.in_channels[level_idx]
            if channels != expected_channels:
                raise ValueError(
                    'PairTemporalPyramidCommonDetailAdapter expected '
                    f'{expected_channels} channels at level {level_idx}, '
                    f'but got {channels}.')

            prev = feat[:pair_batch]
            curr = feat[pair_batch:]
            common = (prev + curr) * 0.5
            detail = (curr - prev) * 0.5
            descriptor = torch.cat(
                [
                    common.mean(dim=(-2, -1)),
                    detail.abs().mean(dim=(-2, -1)),
                ],
                dim=1)
            gate = self.gate_mlps[module_idx](descriptor).view(
                pair_batch, channels, 1, 1)
            delta = self.local_blocks[module_idx](detail) * gate
            spatial_descriptor = None
            if (self.use_spatial_reliability
                    or self.use_shared_scalar_gain):
                common_energy = common.abs().mean(dim=1, keepdim=True)
                detail_energy = detail.abs().mean(dim=1, keepdim=True)
                common_scale = common_energy.detach().mean(
                    dim=(-2, -1), keepdim=True).clamp_min(1e-6)
                detail_scale = detail_energy.detach().mean(
                    dim=(-2, -1), keepdim=True).clamp_min(1e-6)
                spatial_descriptor = torch.cat(
                    [
                        common_energy / common_scale,
                        detail_energy / detail_scale,
                    ],
                    dim=1)
            if self.use_spatial_reliability:
                # Zero logits produce unit modulation. The bounded range can
                # suppress unreliable differences without unbounded updates.
                spatial_reliability = 2.0 * torch.sigmoid(
                    self.spatial_gates[module_idx](spatial_descriptor))
                delta = delta * spatial_reliability
            gamma = self.gamma[module_idx].view(1, 1, 1, 1)
            signed_update = gamma * delta
            if self.conserve_detail_energy:
                detail_rms = detail.detach().square().mean(
                    dim=(-2, -1), keepdim=True).clamp_min(1e-6).sqrt()
                update_rms = signed_update.detach().square().mean(
                    dim=(-2, -1), keepdim=True).clamp_min(1e-6).sqrt()
                energy_scale = (detail_rms / update_rms).clamp(max=1.0)
                signed_update = signed_update * energy_scale
            if self.use_shared_scalar_gain:
                # A shared scalar cannot rotate or mix the channel vector at
                # a location. Its bounded positive gain preserves pair-local
                # feature geometry while allowing common evidence to grow.
                shared_gain = 1.0 + torch.tanh(
                    self.shared_gain_gates[module_idx](spatial_descriptor))
                prev = prev * shared_gain
                curr = curr * shared_gain
            outs[level_idx] = torch.cat(
                [prev + signed_update, curr - signed_update], dim=0)
        return tuple(outs)


class PairTemporalPyramidDualEvidenceAdapter(BaseModule):
    """Pair-equivariant common/detail fusion for pyramid features.

    The common branch adds the same cross-frame evidence to both frames,
    while the odd detail branch adds equal and opposite residuals. This
    separates shared detection evidence from temporal association evidence.
    An optional two-channel spatial gate suppresses local updates where common
    evidence is weak or frame disagreement is high.
    """

    def __init__(self,
                 in_channels: List[int],
                 level_indices: Optional[List[int]] = None,
                 common_level_indices: Optional[List[int]] = None,
                 detail_level_indices: Optional[List[int]] = None,
                 reduction: int = 4,
                 pointwise_groups: int = 8,
                 use_spatial_evidence: bool = False,
                 spatial_common_evidence: bool = True,
                 spatial_detail_evidence: bool = True,
                 spatial_unit_init: bool = False,
                 spatial_detach_descriptor: bool = False,
                 spatial_preserve_mean: bool = False,
                 conserve_branch_energy: bool = False,
                 moment_competitive_gating: bool = False,
                 cross_scale_evidence_budget: bool = False,
                 cross_scale_hidden_dims: int = 32,
                 gamma_init: float = 0.0,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = list(in_channels)
        if level_indices is None:
            level_indices = list(range(len(self.in_channels)))
        self.level_indices = [
            idx if idx >= 0 else len(self.in_channels) + idx
            for idx in level_indices
        ]
        if any(idx < 0 or idx >= len(self.in_channels)
               for idx in self.level_indices):
            raise ValueError(
                f'Invalid level_indices={level_indices} for '
                f'{len(self.in_channels)} input levels.')
        selected_levels = set(self.level_indices)

        def _normalize_branch_levels(
                indices: Optional[List[int]], branch: str) -> set[int]:
            if indices is None:
                return selected_levels.copy()
            normalized = {
                idx if idx >= 0 else len(self.in_channels) + idx
                for idx in indices
            }
            if not normalized.issubset(selected_levels):
                raise ValueError(
                    f'{branch}_level_indices={indices} must be a subset of '
                    f'level_indices={self.level_indices}.')
            return normalized

        self.common_level_indices = _normalize_branch_levels(
            common_level_indices, 'common')
        self.detail_level_indices = _normalize_branch_levels(
            detail_level_indices, 'detail')
        self.use_spatial_evidence = bool(use_spatial_evidence)
        self.spatial_common_evidence = bool(spatial_common_evidence)
        self.spatial_detail_evidence = bool(spatial_detail_evidence)
        self.spatial_unit_init = bool(spatial_unit_init)
        self.spatial_detach_descriptor = bool(spatial_detach_descriptor)
        self.spatial_preserve_mean = bool(spatial_preserve_mean)
        if (self.use_spatial_evidence
                and not (self.spatial_common_evidence
                         or self.spatial_detail_evidence)):
            raise ValueError(
                'At least one spatial evidence branch must be enabled.')
        self.conserve_branch_energy = bool(conserve_branch_energy)
        self.moment_competitive_gating = bool(
            moment_competitive_gating)
        self.cross_scale_evidence_budget = bool(
            cross_scale_evidence_budget)
        self.latest_branch_energy_stats = {}
        self.latest_cross_scale_budget = None

        if self.cross_scale_evidence_budget:
            selected_channels = [
                self.in_channels[idx] for idx in self.level_indices
            ]
            if len(selected_channels) < 2:
                raise ValueError(
                    'cross_scale_evidence_budget requires at least two '
                    'selected feature levels.')
            if len(set(selected_channels)) != 1:
                raise ValueError(
                    'cross_scale_evidence_budget requires equal channel '
                    f'dimensions, but got {selected_channels}.')
            if cross_scale_hidden_dims <= 0:
                raise ValueError(
                    'cross_scale_hidden_dims must be positive, but got '
                    f'{cross_scale_hidden_dims}.')
            channels = selected_channels[0]
            descriptor_dims = channels * 2
            self.cross_scale_token_norm = nn.LayerNorm(descriptor_dims)
            self.cross_scale_token_proj = nn.Linear(
                descriptor_dims, cross_scale_hidden_dims)
            self.cross_scale_embeddings = nn.Parameter(
                torch.zeros(
                    len(self.level_indices), cross_scale_hidden_dims))
            self.cross_scale_mixer = nn.Sequential(
                nn.LayerNorm(cross_scale_hidden_dims * 2),
                nn.Linear(
                    cross_scale_hidden_dims * 2,
                    cross_scale_hidden_dims),
                nn.SiLU(inplace=True),
                nn.Linear(
                    cross_scale_hidden_dims,
                    cross_scale_hidden_dims),
                nn.SiLU(inplace=True),
            )
            self.cross_scale_out = nn.Linear(
                cross_scale_hidden_dims, descriptor_dims)
            nn.init.zeros_(self.cross_scale_out.weight)
            nn.init.zeros_(self.cross_scale_out.bias)

        self.gate_mlps = nn.ModuleList()
        self.moment_mix = nn.ParameterList()
        self.local_blocks = nn.ModuleList()
        self.detail_blocks = nn.ModuleList()
        self.spatial_gates = nn.ModuleList()
        for idx in self.level_indices:
            channels = self.in_channels[idx]
            hidden_dims = max(channels // reduction, 16)
            groups = min(pointwise_groups, channels)
            while channels % groups != 0:
                groups -= 1
            gate_layers = [
                nn.Linear(channels * 2, hidden_dims),
                nn.SiLU(inplace=True),
                nn.Linear(hidden_dims, channels * 2),
            ]
            if not self.moment_competitive_gating:
                gate_layers.append(nn.Sigmoid())
            self.gate_mlps.append(nn.Sequential(*gate_layers))
            if self.moment_competitive_gating:
                self.moment_mix.append(
                    nn.Parameter(torch.zeros(2, channels)))
            if idx in self.common_level_indices:
                self.local_blocks.append(
                    nn.Sequential(
                        nn.Conv2d(
                            channels,
                            channels,
                            kernel_size=3,
                            padding=1,
                            groups=channels,
                            bias=False),
                        nn.SiLU(inplace=True),
                        nn.Conv2d(
                            channels,
                            channels,
                            kernel_size=1,
                            groups=groups,
                            bias=False),
                    ))
            else:
                self.local_blocks.append(nn.Identity())
            # Bias-free convolutions and tanh make this transform odd, which
            # is required for exact equivariance when frame order is swapped.
            if idx in self.detail_level_indices:
                self.detail_blocks.append(
                    nn.Sequential(
                        nn.Conv2d(
                            channels,
                            channels,
                            kernel_size=3,
                            padding=1,
                            groups=channels,
                            bias=False),
                        nn.Tanh(),
                        nn.Conv2d(
                            channels,
                            channels,
                            kernel_size=1,
                            groups=groups,
                            bias=False),
                    ))
            else:
                self.detail_blocks.append(nn.Identity())
            if self.use_spatial_evidence:
                spatial_outputs = (
                    int(self.spatial_common_evidence)
                    + int(self.spatial_detail_evidence))
                spatial_gate = nn.Conv2d(
                    2,
                    spatial_outputs,
                    kernel_size=3,
                    padding=1,
                    bias=True)
                if self.spatial_unit_init:
                    nn.init.zeros_(spatial_gate.weight)
                else:
                    nn.init.normal_(spatial_gate.weight, std=0.01)
                nn.init.zeros_(spatial_gate.bias)
                self.spatial_gates.append(spatial_gate)
        # Column 0 controls common evidence and column 1 controls detail.
        self.gamma = nn.Parameter(
            torch.full(
                (len(self.level_indices), 2), float(gamma_init)))

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor]:
        outs = list(feats)
        if not outs:
            return feats
        self.latest_branch_energy_stats = {}
        self.latest_cross_scale_budget = None
        batch_size = outs[0].shape[0]
        if batch_size % 2 != 0:
            raise ValueError(
                'PairTemporalPyramidDualEvidenceAdapter expects an even '
                'batch ordered as [prev batch, curr batch].')
        pair_batch = batch_size // 2
        common_energy_scales = []
        detail_energy_scales = []
        prepared_levels = []

        for module_idx, level_idx in enumerate(self.level_indices):
            feat = outs[level_idx]
            channels = feat.shape[1]
            expected_channels = self.in_channels[level_idx]
            if channels != expected_channels:
                raise ValueError(
                    'PairTemporalPyramidDualEvidenceAdapter expected '
                    f'{expected_channels} channels at level {level_idx}, '
                    f'but got {channels}.')

            prev = feat[:pair_batch]
            curr = feat[pair_batch:]
            common = (prev + curr) * 0.5
            detail = (curr - prev) * 0.5
            common_mean = common.mean(dim=(-2, -1))
            detail_abs_mean = detail.abs().mean(dim=(-2, -1))
            if self.moment_competitive_gating:
                # RMS-minus-mean-absolute responds to sparse strong evidence
                # without adding a high-resolution gradient path.
                common_stats = common.detach()
                detail_stats = detail.detach()
                common_gap = (
                    common_stats.square().mean(dim=(-2, -1)).sqrt()
                    - common_stats.abs().mean(dim=(-2, -1))).clamp_min(0.0)
                detail_gap = (
                    detail_stats.square().mean(dim=(-2, -1)).sqrt()
                    - detail_stats.abs().mean(dim=(-2, -1))).clamp_min(0.0)
                moment_mix = self.moment_mix[module_idx].tanh()
                common_mean = (
                    common_mean + moment_mix[0] * common_gap)
                detail_abs_mean = (
                    detail_abs_mean + moment_mix[1] * detail_gap)
            descriptor = torch.cat(
                [common_mean, detail_abs_mean], dim=1)
            channel_gates = self.gate_mlps[module_idx](descriptor)
            if self.moment_competitive_gating:
                channel_gates = channel_gates.view(
                    pair_batch, 2, channels).softmax(dim=1)
                common_gate, detail_gate = channel_gates.unbind(dim=1)
            else:
                common_gate, detail_gate = channel_gates.chunk(2, dim=1)
            prepared_levels.append(
                (prev, curr, common, detail, descriptor, common_gate,
                 detail_gate))

        cross_scale_budget = None
        if self.cross_scale_evidence_budget:
            descriptors = torch.stack(
                [item[4].detach() for item in prepared_levels], dim=1)
            scale_tokens = self.cross_scale_token_proj(
                self.cross_scale_token_norm(descriptors))
            scale_tokens = (
                scale_tokens + self.cross_scale_embeddings.unsqueeze(0))
            global_token = scale_tokens.mean(dim=1, keepdim=True).expand_as(
                scale_tokens)
            mixed_tokens = self.cross_scale_mixer(
                torch.cat([scale_tokens, global_token], dim=-1))
            channels = prepared_levels[0][0].shape[1]
            budget_logits = self.cross_scale_out(mixed_tokens).view(
                pair_batch, len(self.level_indices), 2, channels)
            cross_scale_budget = budget_logits.softmax(dim=1)
            cross_scale_budget = (
                cross_scale_budget * len(self.level_indices))
            self.latest_cross_scale_budget = cross_scale_budget.detach()

        for module_idx, level_idx in enumerate(self.level_indices):
            prev, curr, common, detail, _, common_gate, detail_gate = (
                prepared_levels[module_idx])
            channels = common.shape[1]
            if cross_scale_budget is not None:
                common_gate = (
                    common_gate * cross_scale_budget[:, module_idx, 0])
                detail_gate = (
                    detail_gate * cross_scale_budget[:, module_idx, 1])
            common_gate = common_gate.view(pair_batch, channels, 1, 1)
            detail_gate = detail_gate.view(pair_batch, channels, 1, 1)

            if self.use_spatial_evidence:
                spatial_descriptor = torch.cat(
                    [
                        common.abs().mean(dim=1, keepdim=True),
                        detail.abs().mean(dim=1, keepdim=True),
                    ],
                    dim=1)
                if self.spatial_detach_descriptor:
                    spatial_descriptor = spatial_descriptor.detach()
                spatial_gates = torch.sigmoid(
                    self.spatial_gates[module_idx](spatial_descriptor))
                if self.spatial_unit_init:
                    spatial_gates = spatial_gates * 2.0
                if self.spatial_preserve_mean:
                    spatial_gates = spatial_gates / spatial_gates.mean(
                        dim=(-2, -1), keepdim=True).clamp_min(
                            torch.finfo(spatial_gates.dtype).tiny)
                spatial_idx = 0
                if self.spatial_common_evidence:
                    common_spatial = spatial_gates[:, spatial_idx:spatial_idx + 1]
                    spatial_idx += 1
                else:
                    common_spatial = 1.0
                if self.spatial_detail_evidence:
                    detail_spatial = spatial_gates[:, spatial_idx:spatial_idx + 1]
                else:
                    detail_spatial = 1.0
            else:
                common_spatial = detail_spatial = 1.0

            if level_idx in self.common_level_indices:
                common_delta = (
                    self.local_blocks[module_idx](common) * common_gate *
                    common_spatial)
            else:
                common_delta = torch.zeros_like(common)
            if level_idx in self.detail_level_indices:
                detail_delta = (
                    self.detail_blocks[module_idx](detail) * detail_gate *
                    detail_spatial)
            else:
                detail_delta = torch.zeros_like(detail)
            gamma_common = self.gamma[module_idx, 0].view(1, 1, 1, 1)
            gamma_detail = self.gamma[module_idx, 1].view(1, 1, 1, 1)
            shared_update = gamma_common * common_delta
            signed_update = gamma_detail * detail_delta
            if self.conserve_branch_energy:
                common_rms = common.detach().square().mean(
                    dim=(-2, -1), keepdim=True).sqrt()
                shared_rms = shared_update.detach().square().mean(
                    dim=(-2, -1), keepdim=True).sqrt()
                shared_denom = shared_rms.clamp_min(
                    torch.finfo(shared_rms.dtype).tiny)
                common_energy_scale = (
                    common_rms / shared_denom).clamp(max=1.0)
                shared_update = shared_update * common_energy_scale

                detail_rms = detail.detach().square().mean(
                    dim=(-2, -1), keepdim=True).sqrt()
                signed_rms = signed_update.detach().square().mean(
                    dim=(-2, -1), keepdim=True).sqrt()
                signed_denom = signed_rms.clamp_min(
                    torch.finfo(signed_rms.dtype).tiny)
                detail_energy_scale = (
                    detail_rms / signed_denom).clamp(max=1.0)
                signed_update = signed_update * detail_energy_scale
                common_energy_scales.append(common_energy_scale)
                detail_energy_scales.append(detail_energy_scale)
            outs[level_idx] = torch.cat(
                [
                    prev + shared_update - signed_update,
                    curr + shared_update + signed_update,
                ],
                dim=0)
        if common_energy_scales:
            self.latest_branch_energy_stats = {
                'common_scale': torch.stack([
                    scale.detach().mean()
                    for scale in common_energy_scales
                ]).mean(),
                'detail_scale': torch.stack([
                    scale.detach().mean()
                    for scale in detail_energy_scales
                ]).mean(),
                'common_clip': torch.stack([
                    scale.detach().lt(1.0).float().mean()
                    for scale in common_energy_scales
                ]).mean(),
                'detail_clip': torch.stack([
                    scale.detach().lt(1.0).float().mean()
                    for scale in detail_energy_scales
                ]).mean(),
            }
        return tuple(outs)


class RTDETRHybridEncoder(BaseModule):
    """HybridEncoder of RTDETR.

    Args:
        layer_cfg (:obj:`ConfigDict` or dict): The config dict for the encode
            layer. Defaults to None.
        in_channels (List[int], optional): The input channels of the
            feature maps. Defaults to [256, 256, 256].
        use_encoder_idx (List[int], optional): The indices of the encoder
            layers to use. Defaults to [2].
        num_encoder_layers (int, optional): The number of encoder layers.
            Defaults to 1.
        pe_temperature (float, optional): The temperature of the positional
            encoding. Defaults to 10000.
        encode_before_fpn (bool, optional): Encoding the features before FPN
            layer. Defaults to True.
        pair_temporal_adapter_cfg (:obj:`ConfigDict` or dict, optional):
            Optional bidirectional temporal adapter applied to one encoded
            feature level before FPN.  The intended pair batch order is
            ``[prev batch, curr batch]``.
        pair_temporal_adapter_idx (int, optional): Feature level index used by
            the temporal adapter. Defaults to the last encoded level.
        post_pair_temporal_adapter_cfg (:obj:`ConfigDict` or dict, optional):
            Optional bidirectional temporal adapter applied to FPN outputs.
        fpn_cfg (:obj:`ConfigDict` or dict): The config dict for the FPN layer.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 layer_cfg: OptConfigType = None,
                 in_channels: List[int] = [256, 256, 256],
                 use_encoder_idx: List[int] = [2],
                 num_encoder_layers: int = 1,
                 pe_temperature: float = 10000.0,
                 spatial_shapes: Optional[Tuple[Tuple[int, int]]] = None,
                 encode_before_fpn: bool = True,
                 with_cp: bool = False,
                 pair_temporal_adapter_cfg: OptConfigType = None,
                 pair_temporal_adapter_idx: Optional[int] = None,
                 post_pair_temporal_adapter_cfg: OptConfigType = None,
                 fpn_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.use_encoder_idx = use_encoder_idx
        self.pe_temperature = pe_temperature
        self.encode_before_fpn = encode_before_fpn
        self.pair_temporal_adapter_idx = pair_temporal_adapter_idx
        self.post_pair_temporal_adapter = None

        if isinstance(num_encoder_layers, int):
            num_encoder_layers = (num_encoder_layers, ) * len(
                self.use_encoder_idx)
        else:
            assert isinstance(num_encoder_layers, (tuple, list))
            assert len(num_encoder_layers) == len(self.use_encoder_idx)

        # fpn layer
        self.fpn = MODELS.build(fpn_cfg) \
            if fpn_cfg is not None else nn.Identity()

        self.pair_temporal_adapter = None
        if pair_temporal_adapter_cfg is not None:
            if not self.encode_before_fpn:
                raise ValueError(
                    'pair_temporal_adapter_cfg requires encode_before_fpn=True '
                    'so the adapter runs after AIFI and before FPN.')
            if self.pair_temporal_adapter_idx is None:
                self.pair_temporal_adapter_idx = self.use_encoder_idx[-1]
            adapter_cfg = dict(pair_temporal_adapter_cfg)
            adapter_type = adapter_cfg.pop('type', 'mha')
            adapter_cfg.setdefault(
                'embed_dims', self.in_channels[self.pair_temporal_adapter_idx])
            if adapter_type == 'mha':
                self.pair_temporal_adapter = PairTemporalAdapter(**adapter_cfg)
            elif adapter_type == 'pool_gate':
                self.pair_temporal_adapter = PairTemporalPoolGateAdapter(
                    **adapter_cfg)
            else:
                raise ValueError(
                    f'Unsupported pair temporal adapter type: {adapter_type}')

        if post_pair_temporal_adapter_cfg is not None:
            adapter_cfg = dict(post_pair_temporal_adapter_cfg)
            adapter_type = adapter_cfg.pop('type', 'pyramid_local')
            adapter_cfg.setdefault('in_channels', self.in_channels)
            if adapter_type == 'pyramid_local':
                self.post_pair_temporal_adapter = (
                    PairTemporalPyramidLocalAdapter(**adapter_cfg))
            elif adapter_type == 'pyramid_common_detail':
                self.post_pair_temporal_adapter = (
                    PairTemporalPyramidCommonDetailAdapter(**adapter_cfg))
            elif adapter_type == 'pyramid_dual_evidence':
                self.post_pair_temporal_adapter = (
                    PairTemporalPyramidDualEvidenceAdapter(**adapter_cfg))
            else:
                raise ValueError(
                    'Unsupported post pair temporal adapter type: '
                    f'{adapter_type}')

        # encoder transformer
        self.transformer_blocks = nn.ModuleList([
            DetrTransformerEncoder(num_layers, layer_cfg,
                                   num_layers if with_cp else -1)
            for num_layers in num_encoder_layers
        ])

        if spatial_shapes is not None:
            for idx in range(len(use_encoder_idx)):
                spatial_shapes = tuple(map(tuple, spatial_shapes))
                position_embedding = self.build_2d_sincos_position_embedding(
                    *spatial_shapes[idx], in_channels[idx], pe_temperature)
                self.register_buffer(
                    f'position_embedding_{idx}',
                    position_embedding,
                    persistent=False)

    @staticmethod
    @lru_cache
    def build_2d_sincos_position_embedding(
        w: int,
        h: int,
        embed_dim: int = 256,
        temperature: float = 10000.,
        device: Optional[str] = None,
    ) -> Tensor:
        grid_w = torch.arange(w, dtype=torch.float32, device=device)
        grid_h = torch.arange(h, dtype=torch.float32, device=device)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, ('Embed dimension must be divisible by 4 '
                                    'for 2D sin-cos position embedding')
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32, device=device)
        omega = temperature**(omega / -pos_dim)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        pos_embd = [
            torch.sin(out_w),
            torch.cos(out_w),
            torch.sin(out_h),
            torch.cos(out_h)
        ]
        return torch.cat(pos_embd, axis=1)[None, :, :]

    def encode_forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        """
        Args:
            inputs (tuple[Tensor]): input features.

        Returns:
            tuple[Tensor]: encoded features.
        """
        assert len(inputs) == len(self.in_channels)
        outs = list(inputs)

        # encoder
        for i, enc_ind in enumerate(self.use_encoder_idx):
            b, c, h, w = outs[enc_ind].shape
            # flatten [B, C, H, W] to [B, HxW, C]
            src_flatten = outs[enc_ind].flatten(2).permute(0, 2,
                                                           1).contiguous()
            pos_embed = getattr(self, f'position_embedding_{enc_ind}', None)
            if pos_embed is None:
                pos_embed = self.build_2d_sincos_position_embedding(
                    w,
                    h,
                    embed_dim=c,
                    temperature=self.pe_temperature,
                    device=src_flatten.device)
            memory = self.transformer_blocks[i](
                src_flatten, query_pos=pos_embed, key_padding_mask=None)
            outs[enc_ind] = memory.permute(0, 2,
                                           1).contiguous().reshape(b, c, h, w)

        return tuple(outs)

    def pair_temporal_forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        if self.pair_temporal_adapter is None:
            return inputs
        outs = list(inputs)
        idx = self.pair_temporal_adapter_idx
        outs[idx] = self.pair_temporal_adapter(outs[idx])
        return tuple(outs)

    def post_pair_temporal_forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        if self.post_pair_temporal_adapter is None:
            return inputs
        return self.post_pair_temporal_adapter(inputs)

    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        if self.encode_before_fpn:
            return self.post_pair_temporal_forward(
                self.fpn(
                    self.pair_temporal_forward(self.encode_forward(inputs))))
        else:
            return self.post_pair_temporal_forward(
                self.encode_forward(self.fpn(inputs)))
