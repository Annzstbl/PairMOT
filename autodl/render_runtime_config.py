#!/usr/bin/env python3
"""Render filesystem paths into an MMEngine lazy-import config."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hsmot-root', required=True)
    parser.add_argument('--pretrain-root', required=True)
    parser.add_argument('--gmc-root', required=True)
    parser.add_argument('--work-dir', required=True)
    args = parser.parse_args()

    content = args.template.read_text(encoding='utf-8')
    replacements = {
        '__HSMOT_ROOT_REPR__': repr(args.hsmot_root),
        '__PRETRAIN_ROOT_REPR__': repr(args.pretrain_root),
        '__GMC_ROOT_REPR__': repr(args.gmc_root),
        '__WORK_DIR_REPR__': repr(args.work_dir),
    }
    for token, value in replacements.items():
        if content.count(token) != 1:
            raise RuntimeError(f'Expected exactly one template token: {token}')
        content = content.replace(token, value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding='utf-8')
    print(f'Rendered runtime config: {args.output}')


if __name__ == '__main__':
    main()
