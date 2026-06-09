"""Lightweight type-check helpers (replacements for mmengine utilities)."""

from typing import Any, Sequence, Type


def is_list_of(obj: Any, item_type: Type) -> bool:
    if not isinstance(obj, list):
        return False
    return all(isinstance(item, item_type) for item in obj)


def is_seq_of(obj: Any, item_type: Type) -> bool:
    if not isinstance(obj, Sequence) or isinstance(obj, (str, bytes)):
        return False
    return all(isinstance(item, item_type) for item in obj)
