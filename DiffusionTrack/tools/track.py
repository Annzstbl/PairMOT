from loguru import logger
import numpy as np
np.float = float
np.int = int
np.object = object
np.bool = bool
import sys
import os

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)
    
import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger
from yolox.evaluators import DiffusionMOTEvaluatorKL

import argparse
import os
import random
import warnings
import glob
import subprocess


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=6,
                        help="pair-detection batch size")
    parser.add_argument(
        "-d", "--devices", default=1, type=int, help="device for training"
    )
    parser.add_argument(
        "--local_rank", default=0, type=int, help="local rank for dist training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default="exps/example/mot/yolox_x_diffusion_track_dancetrack.py",
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    
    parser.add_argument("-c", "--ckpt", default="diffusiontrack_dancetrack.pth.tar", type=str, help="ckpt for eval")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=8823, type=int, help="eval seed")

    # det args
    parser.add_argument("--det_thresh", default=0.7, type=float, help="detection conf")
    parser.add_argument("--nms2d", default=0.75, type=float, help="detection nms threshold")
    # tracking args

    parser.add_argument("--conf_thresh", type=float, default=0.25, help="tracking confidence threshold")
    parser.add_argument("--nms3d", default=0.7, type=float, help="association nms threshold")
    parser.add_argument("--interval", default=5, type=int, help="relink interval")
    parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument("--output-dir", default="", help="inference/evaluation output root")
    parser.add_argument("--tracker-name", default="diffusiontrack")
    parser.add_argument("--tracker-sub-folder", default="preds")
    parser.add_argument("--evaluate", action="store_true", help="run HSMOT TrackEval after inference")
    parser.add_argument("--trackeval-root", default="../TrackEval")
    parser.add_argument("--gt-dir", default="")
    parser.add_argument("--img-dir", default="")
    parser.add_argument(
        "--detection-cache", default="",
        help="skip the network and run KL tracking from a val_det cache")
    return parser


@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    cudnn.benchmark = True

    rank = args.local_rank
    # rank = get_local_rank()

    file_name = args.output_dir or os.path.join(
        exp.output_dir, args.experiment_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    results_folder = os.path.join(
        file_name, "trackers", args.tracker_name, args.tracker_sub_folder)
    os.makedirs(results_folder, exist_ok=True)

    setup_logger(file_name, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    if args.conf_thresh is not None:
        exp.conf_thresh = args.conf_thresh
    if args.nms2d is not None:
        exp.nms_thresh2d = args.nms2d
    if args.det_thresh is not None:
        exp.det_thresh = args.det_thresh
    if args.nms3d is not None:
        exp.nms_thresh3d = args.nms3d
    if args.interval is not None:
        exp.interval=args.interval
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    evaluator = DiffusionMOTEvaluatorKL(
        args=args,
        dataloader=None,
        img_size=exp.test_size,
        confthre=exp.conf_thresh,
        nmsthre3d=exp.nms_thresh3d,
        detthre=exp.det_thresh,
        nmsthre2d=exp.nms_thresh2d,
        interval=exp.interval,
        num_classes=exp.num_classes,
        )

    if args.detection_cache:
        if is_distributed:
            raise ValueError(
                "cached KL tracking is single-process; use --devices 1")
        *_, summary = evaluator.evaluate_cache(
            os.path.abspath(args.detection_cache), results_folder)
    else:
        if is_distributed:
            raise ValueError(
                "two-stage HSMOT inference uses one GPU with batched pair "
                "detection; use --devices 1")
        model = exp.get_model()
        val_loader = exp.get_eval_loader(
            min(args.batch_size, getattr(exp, 'val_batch_size',
                                         args.batch_size)),
            False, args.test)
        torch.cuda.set_device(rank)
        model.cuda(rank)
        model.eval()

        if not args.speed and not args.trt:
            if args.ckpt is None:
                ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
            else:
                ckpt_file = args.ckpt
            logger.info("loading checkpoint")
            loc = "cuda:{}".format(rank)
            ckpt = torch.load(ckpt_file, map_location=loc)
            model.load_state_dict(ckpt["model"])
            logger.info("loaded checkpoint done.")

        if is_distributed:
            model = DDP(model, device_ids=[rank])

        if args.fuse:
            logger.info("\tFusing model...")
            model = fuse_model(model)

        if args.trt:
            assert (
                not args.fuse and not is_distributed and args.batch_size == 1
            ), "TensorRT model is not support model fusing and distributed inferencing!"
            trt_file = os.path.join(file_name, "model_trt.pth")
            assert os.path.exists(
                trt_file
            ), "TensorRT model is not found!\n Run tools/trt.py first!"
            model.head.decode_in_inference = False
            decoder = model.head.decode_outputs
        else:
            trt_file = None
            decoder = None
        from yolox.evaluators import HSMOTRotatedDetectionEvaluator
        cache_root = os.path.join(file_name, 'val_det')
        cache_name = os.path.splitext(
            os.path.basename(args.ckpt or 'best_ckpt'))[0]
        detector_evaluator = HSMOTRotatedDetectionEvaluator(
            dataloader=val_loader, num_classes=exp.num_classes,
            confthre=0.001, detthre=0.001,
            nmsthre3d=exp.nms_thresh3d, nmsthre2d=exp.nms_thresh2d,
            amp=False, cache_root=cache_root)
        detector_evaluator.validation_name = cache_name
        _, _, detection_summary = detector_evaluator.evaluate(
            model, distributed=False, half=False)
        detection_cache = os.path.join(cache_root, cache_name)
        _, _, tracking_summary = evaluator.evaluate_cache(
            detection_cache, results_folder)
        summary = detection_summary + '\n' + tracking_summary
    logger.info("\n" + summary)

    if args.evaluate:
        gt_dir = args.gt_dir or exp.val_data_dir + "/mot"
        img_dir = args.img_dir or exp.val_data_dir + "/npy"
        command = [
            sys.executable,
            os.path.join(os.path.abspath(args.trackeval_root),
                         "scripts/run_hsmot_8ch.py"),
            "--USE_PARALLEL", "False", "--METRICS", "HOTA", "CLEAR",
            "Identity", "--TRACKERS_TO_EVAL", args.tracker_name,
            "--TRACKER_SUB_FOLDER", args.tracker_sub_folder,
            "--GT_FOLDER", os.path.abspath(gt_dir),
            "--IMG_FOLDER", os.path.abspath(img_dir),
            "--TRACKERS_FOLDER", os.path.abspath(
                os.path.join(file_name, "trackers")),
            "--OUTPUT_FOLDER", os.path.abspath(
                os.path.join(file_name, "trackers")),
        ]
        logger.info("Running TrackEval: {}", " ".join(command))
        environment = os.environ.copy()
        pairmot_root = os.path.abspath(os.path.join(prj_path, ".."))
        python_paths = [pairmot_root, os.path.join(pairmot_root, "ai4rs"),
                        os.path.join(pairmot_root, "hsmot")]
        if environment.get("PYTHONPATH"):
            python_paths.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(python_paths)
        subprocess.run(command, cwd=os.path.abspath(args.trackeval_root),
                       check=True, env=environment)
    logger.info('Completed')


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=args.dist_url,
        args=(exp, args, num_gpu),
    )
