#!/usr/bin/env python3
"""Inspect raw/EMA overfit checkpoints before detection post-processing."""

import argparse
import os

import torch

from yolox.exp import get_exp
from yolox.utils.rotated_boxes import qbox_to_rbox, rotated_iou


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--fixed-training-t", type=int, default=None)
    parser.add_argument("--fixed-noise-seed", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def proposal_stats(model, exp):
    loader = exp.get_eval_loader(1, False)
    images, targets, _, _ = next(iter(loader))
    images = images.cuda(non_blocking=True)
    model.head.device = images.device
    model.head.dtype = images.dtype
    target = targets[0]
    target = target[target[:, 1:9].abs().sum(dim=1) > 0]
    gt_boxes = qbox_to_rbox(target[:, 1:9]).cuda()
    gt_classes = target[:, 0].long().cuda()

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        fpn = model.backbone(images)
        features = [proj(feat) for proj, feat in zip(model.projs, fpn)]
        paired = []
        for feature in features:
            paired.append(torch.cat([feature, feature], dim=0))
        _, _, height, width = images.shape
        image_whwh = torch.tensor(
            [width, height, width, height], device=images.device,
            dtype=images.dtype)[None].expand(4, 4)
        outputs, association, _ = model.head.new_ddim_sample(
            (paired, paired), image_whwh, num_timesteps=1,
            num_proposals=500, dynamic_time=True, track_candidate=0)

    # The second duplicated pair is the detector's current/current branch.
    ref, cur = outputs.chunk(2, dim=0)
    pred_a, pred_b = ref[1], cur[1]
    association = association[1].flatten().float()
    cls_a = pred_a[:, 5:].float().sigmoid()
    cls_b = pred_b[:, 5:].float().sigmoid()
    pair_cls = torch.sqrt(cls_a * cls_b)
    class_conf, pred_class = pair_cls.max(dim=1)
    final_score = torch.sqrt(class_conf * association)
    # Both sides are emitted as current-frame detections by the established
    # duplicated-pair detector path, so report each side independently.
    lines = []
    for gt_index, (gt_box, gt_class) in enumerate(zip(gt_boxes, gt_classes)):
        side_rows = []
        for side_name, boxes in (("a", pred_a[:, :5]),
                                 ("b", pred_b[:, :5])):
            overlaps = rotated_iou(boxes.float(), gt_box[None]).squeeze(1)
            oracle_iou, oracle_index = overlaps.max(dim=0)
            class_mask = pred_class == gt_class
            if class_mask.any():
                masked = overlaps.masked_fill(~class_mask, -1)
                class_iou, class_index = masked.max(dim=0)
            else:
                class_iou = overlaps.new_tensor(-1)
                class_index = overlaps.new_tensor(0, dtype=torch.long)
            side_rows.append(
                "{} oracle_iou={:.4f} oracle_cls={} oracle_score={:.4f} "
                "class_iou={:.4f} class_score={:.4f}".format(
                    side_name, oracle_iou.item(),
                    pred_class[oracle_index].item(),
                    final_score[oracle_index].item(), class_iou.item(),
                    final_score[class_index].item()))
        lines.append("GT{} class={}: {} | {}".format(
            gt_index, gt_class.item(), *side_rows))
    return "\n".join(lines)


def main():
    args = parse_args()
    os.environ["HSMOT_OVERFIT_ROOT"] = args.data_root
    os.environ["YOLO11_WEIGHTS"] = args.weights
    exp = get_exp(args.exp, None)
    exp.train_data_dir = args.data_root
    exp.val_data_dir = args.data_root
    exp.data_num_workers = 0
    if hasattr(exp, "fixed_training_t"):
        exp.fixed_training_t = args.fixed_training_t
    if hasattr(exp, "fixed_noise_seed"):
        exp.fixed_noise_seed = args.fixed_noise_seed
    model = exp.get_model().cuda().eval()
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    for name, key in (("raw", "raw_model"), ("ema", "ema_model")):
        state = checkpoint.get(key)
        if state is None:
            continue
        model.load_state_dict(state, strict=True)
        print("=== {} proposals ===".format(name), flush=True)
        print(proposal_stats(model, exp), flush=True)
        evaluator = exp.get_evaluator(1, False)
        ap50_95, ap50, _ = evaluator.evaluate(model)
        print("{} AP50={:.6f} AP50:95={:.6f}".format(
            name, ap50, ap50_95), flush=True)


if __name__ == "__main__":
    main()
