# Copyright (c) AI4RS. All rights reserved.
"""MMLogger with independent console / file log levels."""
from __future__ import annotations

import logging
import sys
from typing import List, Optional, Sequence, Union

from mmengine.dist import get_rank
from mmengine.logging.logger import MMLogger


def _coerce_log_level(level: Union[int, str]) -> int:
    if isinstance(level, str):
        return logging._nameToLevel[level]
    return level


class _ConsoleSuppressFilter(logging.Filter):
    """Drop log records whose message contains any blocked substring."""

    def __init__(self, patterns: Sequence[str]) -> None:
        super().__init__()
        self.patterns = tuple(patterns)

    def filter(self, record: logging.LogRecord) -> bool:
        if not self.patterns:
            return True
        msg = record.getMessage()
        return not any(p in msg for p in self.patterns)


class MultispecMMLogger(MMLogger):
    """MMLogger with separate console and file handler levels.

    Args:
        log_level: Default level for both handlers when ``console_log_level``
            / ``file_log_level`` are omitted. Same semantics as ``MMLogger``.
        console_log_level: Stream (stdout) handler level on rank 0.
        file_log_level: File handler level. Defaults to ``log_level``.
        console_suppress_patterns: Substrings; matching INFO+ messages are
            dropped on stdout only (file handler keeps them).
    """

    def __init__(self,
                 name: str,
                 logger_name: str = 'mmengine',
                 log_file: Optional[str] = None,
                 log_level: Union[int, str] = 'INFO',
                 file_mode: str = 'w',
                 distributed: bool = False,
                 file_handler_cfg: Optional[dict] = None,
                 console_log_level: Optional[Union[int, str]] = None,
                 file_log_level: Optional[Union[int, str]] = None,
                 console_suppress_patterns: Optional[List[str]] = None,
                 **kwargs) -> None:
        file_level = _coerce_log_level(file_log_level or log_level)
        console_level = _coerce_log_level(console_log_level or log_level)

        super().__init__(
            name=name,
            logger_name=logger_name,
            log_file=log_file,
            log_level=file_level,
            file_mode=file_mode,
            distributed=distributed,
            file_handler_cfg=file_handler_cfg,
            **kwargs)

        global_rank = get_rank()
        for handler in self.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                    handler, logging.FileHandler):
                if handler.stream is sys.stdout:
                    if global_rank == 0:
                        handler.setLevel(console_level)
                    else:
                        handler.setLevel(logging.ERROR)
                    if console_suppress_patterns:
                        handler.addFilter(
                            _ConsoleSuppressFilter(console_suppress_patterns))

        self._console_log_level = console_level
        self._file_log_level = file_level


def _patch_mmengine_current_logger() -> None:
    """Route ``print_log(..., logger='current')`` to ``MultispecMMLogger``."""
    if getattr(MMLogger, '_multispec_current_logger_patched', False):
        return

    _orig_get_current = MMLogger.get_current_instance.__func__

    @classmethod
    def _get_current_instance(cls):  # type: ignore[no-untyped-def]
        if MultispecMMLogger._instance_dict:
            name = next(iter(reversed(MultispecMMLogger._instance_dict)))
            return MultispecMMLogger._instance_dict[name]
        return _orig_get_current(cls)

    MMLogger.get_current_instance = _get_current_instance  # type: ignore
    MMLogger._multispec_current_logger_patched = True


_patch_mmengine_current_logger()
