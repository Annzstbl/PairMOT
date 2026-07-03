#!/usr/bin/env python3
"""Run PairMOT TrackEval asynchronously from a JSON payload."""

import json
import sys

from projects.multispec_pair_rotated_rtdetr.multispec_pair_rotated_rtdetr.pair_ap_metric import (  # noqa: E501
    _async_track_eval_worker,
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit('Usage: async_pair_track_eval.py PAYLOAD_JSON')
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        kwargs = json.load(f)
    _async_track_eval_worker(kwargs)


if __name__ == '__main__':
    main()
