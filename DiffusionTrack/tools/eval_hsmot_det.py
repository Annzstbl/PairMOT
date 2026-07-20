"""Distributed HSMOT rotated-detection validation from a checkpoint."""

import argparse
import os
import sys

import torch
from loguru import logger


PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from yolox.core import launch  # noqa: E402
from yolox.exp import get_exp  # noqa: E402
from yolox.utils import is_main_process, load_ckpt  # noqa: E402


def make_parser():
    parser = argparse.ArgumentParser(
        "Distributed HSMOT rotated detection evaluator")
    parser.add_argument("-f", "--exp-file", required=True)
    parser.add_argument("-c", "--checkpoint", required=True)
    parser.add_argument("-d", "--devices", type=int, default=2)
    parser.add_argument("-b", "--batch-size", type=int, default=6)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--validation-name", default="epoch_005")
    parser.add_argument("--dist-backend", default="nccl")
    parser.add_argument("--dist-url", default=None)
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-rank", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser


@logger.catch(reraise=True)
def main(exp, args):
    model = exp.get_model().to("cuda:{}".format(args.local_rank))
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = load_ckpt(model, checkpoint["model"])
    exp.val_batch_size = args.batch_size
    evaluator = exp.get_evaluator(args.batch_size, True)
    evaluator.validation_name = args.validation_name
    evaluator.cache_root = os.path.join(
        exp.output_dir, args.experiment_name, "val_det")
    metrics = exp.eval(model, evaluator, True)
    if is_main_process():
        logger.info("standalone validation complete: {}", metrics[:2])


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, None)
    exp.merge(args.opts)
    launch(
        main, args.devices, args.num_machines, args.machine_rank,
        backend=args.dist_backend, dist_url=args.dist_url, args=(exp, args))
