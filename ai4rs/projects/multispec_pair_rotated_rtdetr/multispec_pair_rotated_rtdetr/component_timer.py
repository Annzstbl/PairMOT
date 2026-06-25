# Copyright (c) AI4RS. All rights reserved.
"""CUDA-aware per-component timing for Pair RT-DETR training."""

from __future__ import annotations

import time
from typing import Callable, Dict, Optional, TypeVar

import torch

T = TypeVar('T')


class CudaComponentTimer:
    """Accumulate wall-time per named component within one train iter."""

    def __init__(self, use_cuda: Optional[bool] = None) -> None:
        if use_cuda is None:
            use_cuda = torch.cuda.is_available()
        self.use_cuda = use_cuda
        self._durations: Dict[str, float] = {}

    def record(self, name: str, fn: Callable[[], T]) -> T:
        """Time ``fn`` and add elapsed seconds to ``name``."""
        if not self.use_cuda:
            start = time.perf_counter()
            out = fn()
            self._durations[name] = self._durations.get(name, 0.0) + (
                time.perf_counter() - start)
            return out

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        out = fn()
        end_evt.record()
        end_evt.synchronize()
        elapsed = start_evt.elapsed_time(end_evt) / 1000.0
        self._durations[name] = self._durations.get(name, 0.0) + elapsed
        return out

    def get_durations(self) -> Dict[str, float]:
        return dict(self._durations)

    def forward_total(self) -> float:
        return sum(self._durations.values())
