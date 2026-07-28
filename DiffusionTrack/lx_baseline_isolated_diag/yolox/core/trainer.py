#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter
from utils.box_ops import box_xyxyxyxy_to_cxcywhtheta
from utils.box_ops import box_cxcywhtheta_to_xyxyxyxy
from yolox.data import DataPrefetcher
from yolox.utils import (
    MeterBuffer,
    ModelEMA,
    all_reduce_norm,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize,
    save_diffusion_train_diagnostic,
)

import datetime
import os
import time
import cv2
import numpy as np
# torch.autograd.set_detect_anomaly(True)


class Trainer:
    def __init__(self, exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args

        # training related attr
        self.task=exp.task
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = args.local_rank
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.random_flip=exp.random_flip
        self.best_ap = 0
        # 缓存最近一次迭代的匹配框对（用于后续可视化）
        self.last_match = None
        # 当前 batch 的 Dataset 元信息（list，长度=batch_size）；每步写入 img_meta_trace_rank*.txt
        self.last_batch_img_infos = None
        self.last_batch_img_ids = None
        self._img_meta_fp = None

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        log_filename = "train_log.txt" if self.rank == 0 else f"train_log_rank{self.rank}.txt"
        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename=log_filename,
            mode="a",
        )

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def train_one_iter(self):
        iter_start_time = time.time()
        pre_inps, pre_targets,cur_inps,cur_targets, batch_img_infos, batch_img_ids= self.prefetcher.next()
        # collate 后文件名在 batch_img_infos[-1]（list of str）
        
        pre_inps = pre_inps.to(self.data_type)

        # =========================
        # GT debug visualization
        # =========================
        # do_gt_vis = (self.rank == 0) and ((self.iter + 1) % 500 == 0)
        # if do_gt_vis:
        #     try:
        #         vis_dir = os.path.join(self.file_name, "debug_gt")
        #         os.makedirs(vis_dir, exist_ok=True)

        #         # ---------- image ----------
        #         # pre_inps: (1, 8, H, W)
        #         img_chw = pre_inps[0].detach().float().cpu().numpy()  # (C, H, W)
        #         img_hwc = img_chw.transpose(1, 2, 0)                  # HWC
        #         H, W, C = img_hwc.shape

        #         # 选择用于显示的 3 个通道：默认 2:5；若不满足则回退；
        #         # 若图像几乎“全灰/无对比”，再按方差挑 3 个信息量更大的通道
        #         ch = [2, 3, 4]
        #         vis = img_hwc[:, :, ch].astype(np.float32)

        #         # 尝试从 dataloader 的 preproc 拿到 means/std，进行反归一化
        #         means = std = None
        #         try:
        #             ds = getattr(self.train_loader, "dataset", None)
        #             cur = ds
        #             preproc = None
        #             for _ in range(4):
        #                 if cur is None:
        #                     break
        #                 if getattr(cur, "preproc", None) is not None:
        #                     preproc = cur.preproc
        #                     break
        #                 cur = getattr(cur, "_dataset", None)
        #             if preproc is not None:
        #                 means = getattr(preproc, "means", None)
        #                 std = getattr(preproc, "std", None)
        #                 means = np.asarray(means, dtype=np.float32) if means is not None else None
        #                 std = np.asarray(std, dtype=np.float32) if std is not None else None
        #         except Exception:
        #             means = std = None

        #         did_unnorm = False
        #         if (
        #             means is not None
        #             and std is not None
        #             and means.ndim == 1
        #             and std.ndim == 1
        #             and len(means) >= max(ch) + 1
        #             and len(std) >= max(ch) + 1
        #         ):
        #             vis = vis * std[ch] + means[ch]
        #             did_unnorm = True

        #         # 映射到 uint8；如果对比度仍然很低，则做一次简单的分位数拉伸
        #         vis_u8 = (vis * 255.0).clip(0, 255).astype(np.uint8)
        #         if vis_u8.std() < 5:  # 经验阈值：太灰/太平
        #             lo = np.percentile(vis, 1, axis=(0, 1), keepdims=True)
        #             hi = np.percentile(vis, 99, axis=(0, 1), keepdims=True)
        #             vis_norm = (vis - lo) / (hi - lo + 1e-6)
        #             vis_u8 = (vis_norm * 255.0).clip(0, 255).astype(np.uint8)

        #         img = np.ascontiguousarray(vis_u8)

        #         # 画线粗细：随分辨率自适应
        #         thickness = max(2, int(round(min(H, W) / 250)))

        #         # ---------- targets ----------
        #         # pre_targets: (1, 1000, 9)
        #         tgt = pre_targets[0].detach().cpu().numpy()       # (1000, 9)

        #         # padding 过滤：只要 xyxyxyxy 全为 0 就丢掉
        #         polys = tgt[:, 1:9]                                # (1000, 8)
        #         valid_mask = np.any(polys != 0, axis=1)
        #         polys = polys[valid_mask].astype(np.float32)

        #         # 坐标可能是像素，也可能是 [0,1] 归一化：用一个小启发式自动判断
        #         polys = polys.reshape(-1, 4, 2)
        #         maxv = float(np.nanmax(polys)) if polys.size else 0.0
        #         if maxv <= 2.0:
        #             polys[:, :, 0] *= W
        #             polys[:, :, 1] *= H

        #         # 裁剪到图像范围，避免越界导致“看不见/全在外面”
        #         polys[:, :, 0] = np.clip(polys[:, :, 0], 0, W - 1)
        #         polys[:, :, 1] = np.clip(polys[:, :, 1], 0, H - 1)

        #         # ---------- draw ----------
        #         for pts in polys:
        #             poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        #             # 先画黑色描边再画绿色，保证在浅色/复杂背景上也清晰
        #             cv2.polylines(
        #                 img, [poly], isClosed=True, color=(0, 0, 0),
        #                 thickness=thickness + 2, lineType=cv2.LINE_AA
        #             )
        #             cv2.polylines(
        #                 img, [poly], isClosed=True, color=(0, 255, 0),
        #                 thickness=thickness, lineType=cv2.LINE_AA
        #             )

        #         save_path = os.path.join(
        #             vis_dir, f"gt_e{self.epoch+1}_it{self.iter+1}.png"
        #         )
        #         cv2.imwrite(save_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])

        #         self._last_debug_gt_count = len(polys)  # 供 vis 图叠加调试信息
        #         logger.info(
        #             f"[GT vis] saved {save_path}, num_gt={len(polys)}, ch={ch}, unnorm={did_unnorm}, thickness={thickness}"
        #         )

        #     except Exception as e:
        #         logger.warning(f"[GT vis] failed: {e}")

        # pre_targets = pre_targets[:,:,:5].to(self.data_type)
        pre_targets = pre_targets[:,:,:9].to(self.data_type)
        
        pre_targets.requires_grad = False
        if self.task=="tracking":
            cur_inps = cur_inps.to(self.data_type)
            # cur_targets = cur_targets[:,:,:5].to(self.data_type)
            cur_targets = cur_targets[:,:,:9].to(self.data_type)
            cur_targets.requires_grad = False

        data_end_time = time.time()
        inps,targets=(pre_inps,cur_inps),(pre_targets,cur_targets)

        model_ref = self.model.module if self.is_distributed else self.model
        capture_debug = self.rank == 0 and self.iter == 0
        try:
            model_ref.head.capture_train_debug = capture_debug
            model_ref.head.criterion.capture_debug = capture_debug
        except Exception:
            capture_debug = False
        # 必须在 forward 之前计算：此前 do_vis 写在后面，这里会 NameError 被下面 try 吞掉，
        # enable_last_match 从未为 True，last_match 会一直为 None。
        do_vis = ((self.iter + 1) == self.max_iter)
        try:
            # DiffusionNet.head.criterion
            model_ref.head.criterion.enable_last_match = do_vis
        except Exception:
            pass

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            outputs = self.model(inps,targets,self.random_flip,self.input_size)
        if capture_debug:
            self._save_diffusion_diagnostic(
                pre_inps, getattr(model_ref.head, "last_train_debug", None))
            model_ref.head.capture_train_debug = False
            model_ref.head.criterion.capture_debug = False
        # 从模型侧读取匈牙利匹配后的 pred/gt 框对（仅 do_vis 时才会有内容）
        self.last_match = getattr(model_ref, "last_match", None)
        if do_vis and self.last_match is not None:
            try:
                os.makedirs(os.path.join(self.file_name, "vis"), exist_ok=True)

                # 只可视化 batch 的第 1 张 pre 图（速度更快，也更直观）
                img_t = pre_inps[0].detach().float().cpu()  # (C,H,W)
                img_chw = img_t.numpy()  # normalized CHW (after preproc)
                c, h, w = img_chw.shape

                # 从 dataloader 的 preproc 里取 means/std，用于反归一化（兼容 MosaicDetection / 直接 Dataset）
                ds = getattr(self.train_loader, "dataset", None)
                preproc = getattr(ds, "preproc", None)
                if preproc is None and hasattr(ds, "_dataset"):
                    preproc = getattr(ds._dataset, "preproc", None)
                means = getattr(preproc, "means", None)
                std = getattr(preproc, "std", None)
                means = np.array(means, dtype=np.float32) if means is not None else None
                std = np.array(std, dtype=np.float32) if std is not None else None

                # 参考你在 data_augment.py 的做法：
                # CHW -> HWC（如果是 8 通道就能直接转），再取 2:5 三个通道用于显示
                vis_img = img_chw.transpose(1, 2, 0)  # HWC
                if vis_img.shape[2] >= 5:
                    vis_img = vis_img[:, :, 2:5]
                    if means is not None and len(means) >= 5:
                        vis_img = vis_img * std[2:5] + means[2:5] if std is not None else vis_img + means[2:5]

                vis_img = (vis_img * 255.0).clip(0, 255).astype(np.uint8)
                img = np.ascontiguousarray(vis_img)  # HWC uint8

                pred = self.last_match["matched_pred_pre"][0].numpy() if len(self.last_match["matched_pred_pre"]) > 0 else np.zeros((0, 5), dtype=np.float32)
                matched_gt = self.last_match["matched_gt_pre"][0].numpy() if len(self.last_match["matched_gt_pre"]) > 0 else np.zeros((0, 5), dtype=np.float32)

                # 可视化全部 GT：从 dataloader 的 pre_targets 读取（xyxyxyxy -> cxcywhtheta）。
                # 注意这里是当前训练输入尺度下的 GT，便于和 matched/rpn 在同一尺度下直接对比。
                all_gt = np.zeros((0, 5), dtype=np.float32)
                try:
                    pre_t0 = pre_targets[0].detach().float().cpu()  # [max_labels, >=9]
                    valid_mask = (pre_t0[:, 1:9].abs().sum(dim=1) > 0)
                    if valid_mask.any():
                        gt_poly = pre_t0[valid_mask, 1:9]  # xyxyxyxy
                        all_gt = box_xyxyxyxy_to_cxcywhtheta(gt_poly).numpy()
                except Exception:
                    all_gt = np.zeros((0, 5), dtype=np.float32)

                def draw_rboxes(im, rboxes_cxcywhtheta, color, thickness=2):
                    if rboxes_cxcywhtheta is None or len(rboxes_cxcywhtheta) == 0:
                        return im
                    boxes_t = torch.from_numpy(rboxes_cxcywhtheta).float()
                    pts8 = box_cxcywhtheta_to_xyxyxyxy(boxes_t).numpy().reshape(-1, 4, 2)
                    for pts in pts8:
                        poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
                        cv2.polylines(im, [poly], isClosed=True, color=color, thickness=thickness)
                    return im

                def norm5_to_abs(norm_boxes):
                    if norm_boxes is None:
                        return np.zeros((0, 5), dtype=np.float32)
                    if torch.is_tensor(norm_boxes):
                        norm_boxes = norm_boxes.detach().float().cpu().numpy()
                    if norm_boxes.ndim != 2 or norm_boxes.shape[1] < 5:
                        return np.zeros((0, 5), dtype=np.float32)
                    valid = (norm_boxes[:, 2] > 1e-6) & (norm_boxes[:, 3] > 1e-6)
                    abs_boxes = norm_boxes[valid, :5].copy()
                    if len(abs_boxes) > 0:
                        abs_boxes[:, 0] *= float(w)
                        abs_boxes[:, 1] *= float(h)
                        abs_boxes[:, 2] *= float(w)
                        abs_boxes[:, 3] *= float(h)
                    return abs_boxes

                # 图1：全部 GT(绿) + 匹配 Pred(红)
                img_match = img.copy()
                draw_rboxes(img_match, all_gt, color=(0, 255, 0), thickness=2)
                draw_rboxes(img_match, pred, color=(0, 0, 255), thickness=2)

                # 在图上叠加调试信息，便于直接比对
                nlabel_str = str(getattr(model_ref.head, "last_nlabel", "N/A"))
                debug_gt_cnt = getattr(self, "_last_debug_gt_count", "N/A")
                info_lines = [
                    f"dataloader_gt={debug_gt_cnt}  prepare_targets_nlabel={nlabel_str}",
                    f"all_gt={len(all_gt)}  matched_gt={len(matched_gt)}  matched_pred={len(pred)}",
                ]
                for i, line in enumerate(info_lines):
                    y = 28 + i * 26
                    cv2.putText(img_match, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(img_match, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

                out_path = os.path.join(self.file_name, "vis", f"e{self.epoch+1}_it{self.iter+1}.jpg")
                cv2.imwrite(out_path, img_match)
                logger.info(f"[vis] saved: {out_path}  (all_gt={len(all_gt)}, matched_pred={len(pred)})")

                # 图2：全部 GT(绿) + RPN proposals(蓝)；仅 RPN 变体模型会有 last_rpn_proposals。
                rpn_props = getattr(model_ref, "last_rpn_proposals", None)
                if rpn_props is not None and rpn_props.numel() > 0:
                    rpn_abs = norm5_to_abs(rpn_props[0])

                    img_rpn = img.copy()
                    draw_rboxes(img_rpn, all_gt, color=(0, 255, 0), thickness=2)
                    draw_rboxes(img_rpn, rpn_abs, color=(255, 0, 0), thickness=2)
                    rpn_lines = [
                        f"dataloader_gt={debug_gt_cnt}  prepare_targets_nlabel={nlabel_str}",
                        f"all_gt={len(all_gt)}  rpn_boxes={len(rpn_abs)}",
                    ]
                    for i, line in enumerate(rpn_lines):
                        y = 28 + i * 26
                        cv2.putText(img_rpn, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(img_rpn, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

                    out_rpn_path = os.path.join(self.file_name, "vis", f"e{self.epoch+1}_it{self.iter+1}_rpn.jpg")
                    cv2.imwrite(out_rpn_path, img_rpn)
                    logger.info(f"[vis] saved: {out_rpn_path}  (all_gt={len(all_gt)}, rpn_boxes={len(rpn_abs)})")

                # 图3：全部 GT(绿) + 最终注入 diffusion head 的 RPN 子集(黄)
                head_rpn_list = getattr(model_ref.head, "last_head_rpn_proposals", None)
                if head_rpn_list is not None and len(head_rpn_list) > 0 and head_rpn_list[0] is not None:
                    head_rpn_abs = norm5_to_abs(head_rpn_list[0])
                    img_rpn_head = img.copy()
                    draw_rboxes(img_rpn_head, all_gt, color=(0, 255, 0), thickness=2)
                    draw_rboxes(img_rpn_head, head_rpn_abs, color=(0, 255, 255), thickness=2)
                    head_rpn_lines = [
                        f"dataloader_gt={debug_gt_cnt}  prepare_targets_nlabel={nlabel_str}",
                        f"all_gt={len(all_gt)}  head_rpn_boxes={len(head_rpn_abs)}",
                    ]
                    for i, line in enumerate(head_rpn_lines):
                        y = 28 + i * 26
                        cv2.putText(img_rpn_head, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(img_rpn_head, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

                    out_rpn_head_path = os.path.join(self.file_name, "vis", f"e{self.epoch+1}_it{self.iter+1}_rpn_head.jpg")
                    cv2.imwrite(out_rpn_head_path, img_rpn_head)
                    logger.info(f"[vis] saved: {out_rpn_head_path}  (all_gt={len(all_gt)}, head_rpn_boxes={len(head_rpn_abs)})")
            except Exception as e:
                logger.warning(f"[vis] failed: {e}")

        # 关闭缓存，避免后续 iter 发生 cpu 拷贝
        if do_vis:
            try:
                model_ref.head.criterion.enable_last_match = False
            except Exception:
                pass
        loss = outputs["total_loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

    def _save_diffusion_diagnostic(self, images, debug):
        if debug is None:
            logger.warning("LX diffusion diagnostic requested without snapshot")
            return
        output_dir = os.path.join(
            self.file_name, "train_diffusion_diagnostics")
        stem = "epoch_{:03d}_iter_{:05d}_global_{:07d}".format(
            self.epoch + 1, self.iter + 1, self.progress_in_iter + 1)
        try:
            paths = save_diffusion_train_diagnostic(
                output_dir, stem, images, debug,
                class_names=getattr(
                    getattr(self.train_loader, "dataset", None),
                    "classes", None),
                max_proposals=60, save_snapshot=False)
            logger.info(
                "saved LX diffusion diagnostics to {} ({} files)",
                output_dir, len(paths))
        except Exception as error:
            logger.warning(
                "failed to save LX diffusion diagnostic: {}", error)

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        # logger.info(
        #     "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        # )
        model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
        )
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader,self.task)
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(
                model,
                device_ids=[self.local_rank],
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model
        self.model.train()

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
        )
        # Tensorboard logger
        if self.rank == 0:
            self.tblogger = SummaryWriter(self.file_name)

        logger.info("Training start...")
        #logger.info("\n{}".format(model))

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(
                self.best_ap * 100
            )
        )

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            
            logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            if self.is_distributed:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True
            
            # self.exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")

    def after_epoch(self):
        if self.use_model_ema:
            self.ema_model.update_attr(self.model)

        self.save_ckpt(ckpt_name="latest")
        if (self.epoch + 1) % 10 == 0:
            self.save_ckpt(ckpt_name="epoch_{}".format(self.epoch+1))
        if self.evaluator is not None and (self.epoch + 1) % self.exp.eval_interval == 0: 
            all_reduce_norm(self.model)
            self.evaluate_and_save_model() 

    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        # (self.iter + 1) % self.exp.print_interval == 0 and
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.3f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )
            self.meter.clear_meters()

        # random resizing
        if self.exp.random_size is not None and (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth.tar")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        if self.evaluator is None:
            return
        evalmodel = self.ema_model.ema if self.use_model_ema else self.model
        if hasattr(self.evaluator, "cache_root"):
            self.evaluator.cache_root = os.path.join(
                self.file_name, "val_det")
            self.evaluator.validation_name = "epoch_{:03d}".format(
                self.epoch + 1)
        ap50_95, ap50, summary = self.exp.eval(
            evalmodel, self.evaluator, self.is_distributed
        )
        self.model.train()
        if self.rank == 0:
            self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
            self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            logger.info("\n" + summary)
        synchronize()

        self.best_ap = max(self.best_ap, ap50_95)
        self.save_ckpt("last_epoch", ap50 > self.best_ap)
        self.best_ap = max(self.best_ap, ap50)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if not getattr(self.exp, "save_checkpoints", True):
            return
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )
