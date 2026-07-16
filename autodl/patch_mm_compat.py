#!/usr/bin/env python3
"""Apply the mmcv 2.2 compatibility edits documented by PairMOT."""

from __future__ import annotations

import importlib.metadata
import re
from pathlib import Path


def package_root(distribution: str, import_name: str) -> Path:
    dist = importlib.metadata.distribution(distribution)
    for relative in dist.files or ():
        if relative.parts and relative.parts[0] == import_name:
            return Path(dist.locate_file(relative.parts[0])).resolve()
    raise RuntimeError(f"Cannot locate package directory for {distribution}")


def replace(path: Path, pattern: str, replacement: str) -> None:
    content = path.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, replacement, content)
    if count == 0 and replacement not in content:
        raise RuntimeError(f"Compatibility constant not found in {path}")
    if updated != content:
        path.write_text(updated, encoding="utf-8")
        print(f"Patched {path}")
    else:
        print(f"Already compatible: {path}")


def main() -> None:
    replace(package_root("mmdet", "mmdet") / "__init__.py",
            r"mmcv_maximum_version\s*=\s*['\"]2\.2\.0['\"]",
            "mmcv_maximum_version = '2.3.0'")
    replace(package_root("mmsegmentation", "mmseg") / "__init__.py",
            r"MMCV_MAX\s*=\s*['\"]2\.2\.0['\"]",
            "MMCV_MAX = '2.3.0'")


if __name__ == "__main__":
    main()
