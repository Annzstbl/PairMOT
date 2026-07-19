"""YOLO11 feature adapter with the MMOT official ConvMSI stem."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLO11ConvMSIStem(nn.Module):
    """Exact MMOT ConvMSI: Conv3D, BN3D and depthwise spectral fusion."""

    def __init__(self, out_channels=64, num_spectral=8):
        super().__init__()
        self.num_spectral = int(num_spectral)
        self.conv3d = nn.Conv3d(
            1, out_channels, kernel_size=(3, 3, 3), stride=(1, 2, 2),
            padding=(1, 1, 1), bias=False)
        self.bn3d = nn.BatchNorm3d(out_channels)
        self.fuse = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=(self.num_spectral, 1, 1),
            groups=out_channels, bias=False)
        self.bn2d = nn.BatchNorm2d(
            out_channels, eps=0.001, momentum=0.03)
        self.act = nn.SiLU(inplace=True)

    def init_from_mmot_stem(self, stem_module):
        """Load every trainable/buffer parameter of the official ConvMSI."""
        expected = {
            "conv3d": self.conv3d,
            "bn3d": self.bn3d,
            "fuse": self.fuse,
            "bn2d": self.bn2d,
        }
        for name, target in expected.items():
            source = getattr(stem_module, name)
            target.load_state_dict(source.float().state_dict(), strict=True)

    def init_from_yolo_conv(self, conv_module):
        """Fallback initialization when only an RGB YOLO checkpoint exists."""
        with torch.no_grad():
            weight = conv_module.conv.weight
            if tuple(weight.shape[1:]) != (3, 3, 3):
                raise ValueError(
                    "expected YOLO RGB stem weight [C,3,3,3], got {}".
                    format(tuple(weight.shape)))
            self.conv3d.weight.copy_(weight.unsqueeze(1))
            self.fuse.weight.fill_(1.0 / self.num_spectral)
            self.bn2d.load_state_dict(conv_module.bn.state_dict())

    def forward(self, x):
        num_spectral = getattr(
            self, "num_spectral", int(self.fuse.kernel_size[0]))
        if x.ndim == 4 and x.shape[1] == num_spectral:
            x = x.unsqueeze(1)
        elif x.ndim != 5 or x.shape[1:3] != (1, num_spectral):
            raise ValueError("YOLO11 ConvMSI expects [B,8,H,W], got {}".
                             format(tuple(x.shape)))
        x = self.act(self.bn3d(self.conv3d(x)))
        output_dtype = x.dtype
        # Torch 2.0/cu118 does not implement the depthwise Conv3D CUDA kernel
        # for BF16. This is the exact same convolution and weights; only its
        # arithmetic runs in FP32 before returning to the autocast dtype.
        if output_dtype == torch.bfloat16:
            with torch.cuda.amp.autocast(enabled=False):
                x = self.fuse(x.float()).squeeze(2)
            x = x.to(output_dtype)
        else:
            x = self.fuse(x).squeeze(2)
        return self.act(self.bn2d(x))


def _load_yolo_task_model(model_cfg, weights):
    from ultralytics import YOLO

    if not weights:
        return YOLO(str(model_cfg)).model

    # Register the missing class at the exact pickle import location.  This
    # does not modify the installed package on disk.
    import ultralytics.nn.modules.conv as ultralytics_conv
    if not hasattr(ultralytics_conv, "ConvMSI"):
        ultralytics_conv.ConvMSI = YOLO11ConvMSIStem

    payload = torch.load(str(weights), map_location="cpu", weights_only=False)
    if isinstance(payload, nn.Module):
        task_model = payload
    elif isinstance(payload, dict):
        task_model = payload.get("ema")
        if task_model is None:
            task_model = payload.get("model")
    else:
        task_model = None
    if isinstance(task_model, nn.Module):
        task_model = task_model.float()
        # Match Ultralytics' normal attempt_load_weights compatibility pass.
        for module in task_model.modules():
            if hasattr(module, "inplace"):
                module.inplace = True
            elif isinstance(module, nn.Upsample) and not hasattr(
                    module, "recompute_scale_factor"):
                module.recompute_scale_factor = None
        return task_model
    # Retain ordinary Ultralytics behavior for unusual state-dict-only files.
    return YOLO(str(weights)).model


class YOLO11BackboneAdapter(nn.Module):
    """Return YOLO11L OBB P3/P4/P5 inputs while skipping its OBB head."""

    def __init__(self, model_cfg="yolo11l-obb.yaml", weights="",
                 freeze=False, num_spectral=8):
        super().__init__()
        task_model = _load_yolo_task_model(model_cfg, weights)
        original_stem = task_model.model[0]
        stem_conv = (original_stem.conv3d if hasattr(original_stem, "conv3d")
                     else original_stem.conv)
        spectral_stem = YOLO11ConvMSIStem(
            out_channels=stem_conv.out_channels,
            num_spectral=num_spectral)
        if hasattr(original_stem, "conv3d") and hasattr(original_stem, "bn2d"):
            spectral_stem.init_from_mmot_stem(original_stem)
            self.pretrained_source = "mmot_convmsi"
        else:
            spectral_stem.init_from_yolo_conv(original_stem)
            self.pretrained_source = "rgb_yolo"
        # Ultralytics' graph executor stores routing metadata on every layer.
        for name in ("i", "f", "type", "np"):
            if hasattr(original_stem, name):
                setattr(spectral_stem, name, getattr(original_stem, name))
        task_model.model[0] = spectral_stem

        # Ultralytics checkpoints prefer the EMA module, whose parameters are
        # serialized with requires_grad=False.  Training policy must be set by
        # this adapter rather than inherited from that serialization detail.
        for parameter in task_model.parameters():
            parameter.requires_grad_(not freeze)

        self.task_model = task_model
        self.layers = task_model.model
        self.head = self.layers[-1]
        self.feature_indices = list(self.head.f)
        self.out_channels = [
            int(branch[0].conv.in_channels) for branch in self.head.cv2]
        self.in_channels = self.out_channels
        self.channel_scale = 1.0

        for parameter in self.head.parameters():
            parameter.requires_grad = False
        if freeze:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # HSMOT's native 900x1200 size is not divisible by YOLO's stride 32.
        # Pad only inside the backbone; all public box coordinates remain in
        # the requested 900x1200 image space.
        pad_h = (-x.shape[-2]) % 32
        pad_w = (-x.shape[-1]) % 32
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        saved = []
        current = x
        for module in self.layers[:-1]:
            if module.f != -1:
                if isinstance(module.f, int):
                    current = saved[module.f]
                else:
                    current = [current if index == -1 else saved[index]
                               for index in module.f]
            # Torch 2.0/cu118 likewise lacks BF16 nearest-neighbour upsample.
            # Keep only this parameter-free resize in FP32.
            if isinstance(module, nn.Upsample) and current.dtype == torch.bfloat16:
                with torch.cuda.amp.autocast(enabled=False):
                    current = module(current.float()).to(torch.bfloat16)
            else:
                current = module(current)
            saved.append(current if module.i in self.task_model.save else None)
        return tuple(saved[index] for index in self.feature_indices)
