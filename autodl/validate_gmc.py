#!/usr/bin/env python3
"""Verify complete, non-placeholder gap-1 GMC for an HSMOT split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def frame_ids(sequence: Path) -> list[int]:
    result = []
    for path in sequence.glob("*_p1.jpg"):
        try:
            result.append(int(path.name.removesuffix("_p1.jpg")))
        except ValueError:
            pass
    return sorted(set(result))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    failures: list[str] = []
    expected = 0

    for ann in sorted((args.data_root / "mot").glob("*.txt")):
        seq = ann.stem
        ids = frame_ids(args.data_root / "npy2jpg" / seq)
        id_set = set(ids)
        for curr in ids:
            prev = curr - 1
            if prev not in id_set:
                continue
            expected += 1
            path = args.cache_root / seq / f"{prev:06d}_{curr:06d}.json"
            if not path.is_file():
                failures.append(f"missing {path}")
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                matrix = np.asarray(payload["matrix"], dtype=np.float64)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                failures.append(f"invalid {path}: {exc}")
                continue
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                failures.append(f"non-finite or malformed matrix {path}")
            elif not payload.get("ok", False):
                failures.append(f"failed GMC estimate {path}")
            elif np.array_equal(matrix, np.eye(3)):
                failures.append(f"identity placeholder {path}")

    if failures:
        preview = "\n".join(failures[:20])
        raise SystemExit(
            f"GMC validation failed ({len(failures)} errors):\n{preview}")
    if expected == 0:
        raise SystemExit("No adjacent HSMOT frame pairs found")
    print(f"GMC validation passed: expected_pairs={expected}, "
          f"cache_root={args.cache_root}")


if __name__ == "__main__":
    main()
