# Copyright (c) OpenMMLab. All rights reserved.
# Image utilities vendored from mmcv, implemented with OpenCV + NumPy.
import functools
from typing import Callable, Optional, Sequence, Tuple, Type, Union

import cv2
import numpy as np
import torch

_IMREAD_FLAGS = {
    "color": cv2.IMREAD_COLOR,
    "grayscale": cv2.IMREAD_GRAYSCALE,
    "unchanged": cv2.IMREAD_UNCHANGED,
}

_INTERPOLATION_MAP = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


def assert_tensor_type(func: Callable) -> Callable:

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not isinstance(args[0].data, torch.Tensor):
            raise AttributeError(
                f"{args[0].__class__.__name__} has no attribute {func.__name__} for type {args[0].datatype}"
            )
        return func(*args, **kwargs)

    return wrapper


class DataContainer:
    """A container for any type of objects."""

    def __init__(
        self,
        data: Union[torch.Tensor, np.ndarray],
        stack: bool = False,
        padding_value: int = 0,
        cpu_only: bool = False,
        pad_dims: int = 2,
    ):
        self._data = data
        self._cpu_only = cpu_only
        self._stack = stack
        self._padding_value = padding_value
        assert pad_dims in [None, 1, 2, 3]
        self._pad_dims = pad_dims

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.data)})"

    def __len__(self) -> int:
        return len(self._data)

    @property
    def data(self) -> Union[torch.Tensor, np.ndarray]:
        return self._data

    @property
    def datatype(self) -> Union[Type, str]:
        if isinstance(self.data, torch.Tensor):
            return self.data.type()
        return type(self.data)

    @property
    def cpu_only(self) -> bool:
        return self._cpu_only

    @property
    def stack(self) -> bool:
        return self._stack

    @property
    def padding_value(self) -> int:
        return self._padding_value

    @property
    def pad_dims(self) -> int:
        return self._pad_dims

    @assert_tensor_type
    def size(self, *args, **kwargs) -> torch.Size:
        return self.data.size(*args, **kwargs)

    @assert_tensor_type
    def dim(self) -> int:
        return self.data.dim()


class FileClient:
    """Minimal local-disk file client replacing mmcv/mmengine FileClient."""

    def __init__(self, backend: str = "disk", **kwargs):
        self.backend = backend
        self.kwargs = kwargs

    def get(self, filepath: str) -> bytes:
        with open(filepath, "rb") as f:
            return f.read()


def is_str(x) -> bool:
    return isinstance(x, str)


def _flag_to_cv2(flag: str) -> int:
    if flag not in _IMREAD_FLAGS:
        raise ValueError(f"Unsupported flag: {flag!r}")
    return _IMREAD_FLAGS[flag]


def _apply_channel_order(img: np.ndarray, channel_order: str) -> np.ndarray:
    if channel_order == "rgb" and img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def imread(
    filepath: str,
    flag: str = "color",
    channel_order: str = "bgr",
) -> np.ndarray:
    img = cv2.imread(filepath, _flag_to_cv2(flag))
    if img is None:
        raise IOError(f"Failed to read image: {filepath}")
    return _apply_channel_order(img, channel_order)


def imfrombytes(
    content: bytes,
    flag: str = "color",
    channel_order: str = "bgr",
    backend: str = "cv2",
) -> np.ndarray:
    del backend
    buf = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(buf, _flag_to_cv2(flag))
    if img is None:
        raise ValueError("Failed to decode image bytes")
    return _apply_channel_order(img, channel_order)


def _interp(name: str) -> int:
    return _INTERPOLATION_MAP.get(name, cv2.INTER_LINEAR)


def imresize(
    img: np.ndarray,
    size: Tuple[int, int],
    return_scale: bool = False,
    interpolation: str = "bilinear",
    out: Optional[np.ndarray] = None,
    backend: str = "cv2",
) -> Union[np.ndarray, Tuple[np.ndarray, float, float]]:
    del backend
    h, w = img.shape[:2]
    target_w, target_h = size
    resized = cv2.resize(img, (int(target_w), int(target_h)), interpolation=_interp(interpolation))
    if out is not None:
        np.copyto(out, resized)
        resized = out
    if return_scale:
        w_scale = target_w / max(w, 1e-6)
        h_scale = target_h / max(h, 1e-6)
        return resized, w_scale, h_scale
    return resized


def imrescale(
    img: np.ndarray,
    scale: Union[float, Sequence[float]],
    return_scale: bool = False,
    interpolation: str = "bilinear",
    backend: str = "cv2",
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    del backend
    h, w = img.shape[:2]
    if isinstance(scale, (list, tuple)):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
    else:
        scale_factor = scale / max(h, w)
    new_w = int(round(w * scale_factor))
    new_h = int(round(h * scale_factor))
    resized = cv2.resize(img, (new_w, new_h), interpolation=_interp(interpolation))
    if return_scale:
        return resized, scale_factor
    return resized


def imflip(img: np.ndarray, direction: str = "horizontal") -> np.ndarray:
    if direction == "horizontal":
        return cv2.flip(img, 1)
    if direction == "vertical":
        return cv2.flip(img, 0)
    if direction == "diagonal":
        return cv2.flip(img, -1)
    raise ValueError(f"Invalid flip direction: {direction!r}")


def impad(
    img: np.ndarray,
    shape: Optional[Tuple[int, int]] = None,
    padding: Union[int, Tuple[int, int], Tuple[int, int, int, int], None] = None,
    pad_val: Union[float, Sequence[float]] = 0,
    padding_mode: str = "constant",
) -> np.ndarray:
    del padding_mode
    if shape is not None:
        target_h, target_w = shape
        pad_h = max(target_h - img.shape[0], 0)
        pad_w = max(target_w - img.shape[1], 0)
        padding = (0, 0, pad_w, pad_h)
    elif padding is None:
        raise ValueError("Either shape or padding must be specified")
    elif isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    elif len(padding) == 2:
        padding = (padding[1], padding[1], padding[0], padding[0])

    if isinstance(pad_val, (list, tuple, np.ndarray)):
        if img.ndim == 3:
            pad_val = tuple(pad_val)
        else:
            pad_val = float(pad_val[0])
    return cv2.copyMakeBorder(
        img,
        padding[1],
        padding[3],
        padding[0],
        padding[2],
        borderType=cv2.BORDER_CONSTANT,
        value=pad_val,
    )


def impad_to_multiple(
    img: np.ndarray,
    divisor: int,
    pad_val: Union[float, Sequence[float]] = 0,
) -> np.ndarray:
    h, w = img.shape[:2]
    target_h = int(np.ceil(h / divisor) * divisor)
    target_w = int(np.ceil(w / divisor) * divisor)
    return impad(img, shape=(target_h, target_w), pad_val=pad_val)


def imnormalize_(
    img: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    to_rgb: bool = False,
) -> np.ndarray:
    if to_rgb and img.ndim == 3 and img.shape[2] == 3:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    mean = np.asarray(mean, dtype=np.float32).reshape(1, -1)
    std = np.asarray(std, dtype=np.float32).reshape(1, -1)
    np.subtract(img, mean, out=img)
    np.divide(img, std, out=img)
    return img


def imdenormalize(img, mean, std, to_bgr=True):
    assert img.dtype != np.uint8
    mean = mean.reshape(1, -1).astype(np.float64)
    std = std.reshape(1, -1).astype(np.float64)
    img = cv2.multiply(img, std)
    cv2.add(img, mean, img)
    if to_bgr:
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
    return img
