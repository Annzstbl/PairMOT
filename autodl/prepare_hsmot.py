#!/usr/bin/env python3
"""Safely extract an HSMOT archive into one canonical data directory."""

from __future__ import annotations

import argparse
import os
import shutil
import tarfile
import tempfile
from pathlib import Path


def is_hsmot_root(path: Path) -> bool:
    return all((path / split / child).is_dir()
               for split in ("train", "test")
               for child in ("mot", "npy2jpg"))


def validate_members(archive: tarfile.TarFile) -> None:
    for member in archive.getmembers():
        path = Path(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"Unsafe archive member: {member.name}")
        if member.issym() or member.islnk():
            raise RuntimeError(f"Links are not allowed in the archive: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise RuntimeError(f"Special archive member is not allowed: {member.name}")


def find_root(extract_dir: Path) -> Path:
    candidates = [path for path in (extract_dir, *extract_dir.rglob("*"))
                  if path.is_dir() and is_hsmot_root(path)]
    outermost = [path for path in candidates
                 if not any(parent in candidates for parent in path.parents)]
    if len(outermost) != 1:
        names = ", ".join(str(path.relative_to(extract_dir))
                          for path in outermost) or "none"
        raise RuntimeError(f"Expected one HSMOT root, found: {names}")
    return outermost[0]


def summarize(root: Path) -> None:
    print(f"HSMOT_ROOT={root}")
    for split in ("train", "test"):
        ann = root / split / "mot"
        images = root / split / "npy2jpg"
        annotations = len(list(ann.glob("*.txt")))
        sequences = sum(path.is_dir() for path in images.iterdir())
        frames = sum(1 for _ in images.rglob("*_p1.jpg"))
        if annotations == 0 or sequences == 0 or frames == 0:
            raise RuntimeError(f"{split} is empty or has an unexpected layout")
        print(f"  {split}: annotations={annotations}, sequences={sequences}, "
              f"p1_frames={frames}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    archive = args.archive.expanduser().resolve()
    target = args.target.expanduser().resolve()

    if is_hsmot_root(target):
        print("Dataset is already initialized; extraction skipped.")
        summarize(target)
        return
    if target.exists():
        raise RuntimeError(f"Refusing to overwrite incomplete target: {target}")
    if not archive.is_file():
        raise FileNotFoundError(archive)

    target.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".hsmot-extract-",
                                dir=target.parent))
    try:
        print(f"Extracting {archive} into temporary directory {temp}")
        with tarfile.open(archive, "r:gz") as handle:
            validate_members(handle)
            # Members were validated above; avoid the Python-version-specific
            # tarfile filter argument used only by newer AutoDL images.
            handle.extractall(temp)
        source = find_root(temp)
        os.replace(source, target)
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    summarize(target)


if __name__ == "__main__":
    main()
