#!/usr/bin/env python3
"""Profile HSMOT NPY reads and the real augmented training DataLoader."""

import argparse
import glob
import os
import random
import statistics
import sys
import time

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * q))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--raw-reads", type=int, default=40)
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.root, "npy", "*", "*.npy"))
    random.Random(8823).shuffle(files)
    files = files[:min(args.raw_reads, len(files))]
    start = time.perf_counter()
    total_bytes = 0
    for path in files:
        array = np.load(path)
        total_bytes += array.nbytes
        _ = int(array[0, 0, 0])
    elapsed = time.perf_counter() - start
    print("raw files={} time={:.3f}s throughput={:.1f}MiB/s".format(
        len(files), elapsed, total_bytes / elapsed / 2 ** 20))

    from exps.example.mot.yolo11l_diffusion_det_hsmot import Exp

    exp = Exp()
    exp.train_data_dir = args.root
    exp.data_num_workers = args.workers
    loader = exp.get_data_loader(args.batch_size, False)
    iterator = iter(loader)
    timings = []
    for _ in range(args.batches):
        start = time.perf_counter()
        next(iterator)
        timings.append(time.perf_counter() - start)
    steady = timings[min(2, len(timings) - 1):]
    print("loader workers={} batch={} first={:.3f}s mean={:.3f}s "
          "median={:.3f}s p95={:.3f}s".format(
              args.workers, args.batch_size, timings[0],
              statistics.mean(steady), statistics.median(steady),
              percentile(steady, .95)))


if __name__ == "__main__":
    main()
