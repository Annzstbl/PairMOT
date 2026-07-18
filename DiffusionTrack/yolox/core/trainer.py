#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter

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
)

import datetime
import contextlib
import math
import os
import time


class Trainer:
    def __init__(self, exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args

        # training related attr
        self.task=exp.task
        self.max_epoch = exp.max_epoch
        amp_dtype = getattr(args, "amp_dtype", "fp32")
        if args.fp16:
            if amp_dtype not in ("fp32", "fp16"):
                raise ValueError("--fp16 conflicts with --amp-dtype {}".format(
                    amp_dtype))
            amp_dtype = "fp16"
        self.amp_dtype_name = amp_dtype
        self.amp_training = amp_dtype != "fp32"
        self.amp_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[amp_dtype]
        if amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 was requested but this GPU does not support it")
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype == "fp16")
        self.accumulate = int(getattr(args, "accumulate", 1))
        if self.accumulate < 1:
            raise ValueError("--accumulate must be at least 1")
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = args.local_rank
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema

        # data/dataloader related attr
        self.data_type = self.amp_dtype
        self.target_type = torch.float32
        self.optimizer_step = 0
        self.skipped_optimizer_steps = 0
        self._ema_state = None
        self.input_size = exp.input_size
        self.random_flip=exp.random_flip
        self.best_ap = 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
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
        pre_inps, pre_targets,cur_inps,cur_targets= self.prefetcher.next()
        pre_inps = pre_inps.to(self.data_type)
        target_dim = getattr(self.exp, "target_dim", 5)
        pre_targets = pre_targets[:, :, :target_dim].to(self.target_type)
        pre_targets.requires_grad = False
        if self.task=="tracking":
            cur_inps = cur_inps.to(self.data_type)
            cur_targets = cur_targets[:, :, :target_dim].to(self.target_type)
            cur_targets.requires_grad = False

        data_end_time = time.time()
        inps,targets=(pre_inps,cur_inps),(pre_targets,cur_targets)
        group_start = (self.iter // self.accumulate) * self.accumulate
        group_size = min(self.accumulate, self.max_iter - group_start)
        should_step = self.iter + 1 == group_start + group_size
        if self.iter == group_start:
            self.optimizer.zero_grad()

        # Avoid an unnecessary DDP all-reduce for intermediate micro-batches.
        sync_context = (
            contextlib.nullcontext()
            if should_step or not self.is_distributed
            else self.model.no_sync()
        )
        with sync_context:
            with torch.cuda.amp.autocast(
                    enabled=self.amp_training, dtype=self.amp_dtype):
                outputs = self.model(inps,targets,self.random_flip,self.input_size)
            loss = outputs["total_loss"]
            self.scaler.scale(loss / group_size).backward()

        if should_step:
            optimizer_ran = True
            if self.scaler.is_enabled():
                old_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                optimizer_ran = self.scaler.get_scale() >= old_scale
            else:
                self.optimizer.step()

            if optimizer_ran:
                self.optimizer_step += 1
                if self.use_model_ema:
                    self.ema_model.update(self.model)
                self.current_lr = self.lr_scheduler.update_lr(
                    self.optimizer_step)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.current_lr
            else:
                self.skipped_optimizer_steps += 1

        lr = self.current_lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

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
        self.optimizer_iters_per_epoch = math.ceil(
            self.max_iter / self.accumulate)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size * self.accumulate,
            self.optimizer_iters_per_epoch,
        )
        if self.optimizer_step < 0:
            # Backward compatibility for checkpoints written before the
            # successful optimizer-step counter was persisted.
            self.optimizer_step = (
                self.start_epoch * self.optimizer_iters_per_epoch)
        self.current_lr = self.optimizer.param_groups[0]["lr"]
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False,find_unused_parameters=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            if self._ema_state is not None:
                self.ema_model.ema.load_state_dict(self._ema_state)
            self.ema_model.updates = self.optimizer_step

        self.model = model
        self.model.train()

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )
        # Tensorboard logger
        if self.rank == 0:
            self.tblogger = SummaryWriter(self.file_name)

        logger.info(
            "Training start... physical global batch={}, accumulate={}, "
            "effective global batch={}".format(
                self.args.batch_size,
                self.accumulate,
                self.args.batch_size * self.accumulate,
            )
        )
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
            
            self.exp.eval_interval = getattr(
                self.exp, "no_aug_eval_interval", 1)
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")

    def after_epoch(self):
        if self.use_model_ema:
            self.ema_model.update_attr(self.model)

        self.save_ckpt(ckpt_name="latest")
        save_interval = getattr(self.exp, "save_interval", 10)
        if (self.epoch + 1) % save_interval == 0:
            self.save_ckpt(ckpt_name="epoch_{}".format(self.epoch+1))
        if (self.epoch + 1) % self.exp.eval_interval == 0: 
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
            # New checkpoints retain the raw train model separately while the
            # legacy ``model`` key remains EMA for inference compatibility.
            model.load_state_dict(ckpt.get("raw_model", ckpt["model"]))
            self.optimizer.load_state_dict(ckpt["optimizer"])
            if self.scaler.is_enabled() and ckpt.get("scaler") is not None:
                self.scaler.load_state_dict(ckpt["scaler"])
            self._ema_state = ckpt.get("ema_model", ckpt["model"])
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            self.optimizer_step = ckpt.get("optimizer_step", -1)
            self.skipped_optimizer_steps = ckpt.get(
                "skipped_optimizer_steps", 0)
            self.best_ap = ckpt.get("best_ap", 0)
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
        evalmodel = self.ema_model.ema if self.use_model_ema else self.model
        ap50_95, ap50, summary = self.exp.eval(
            evalmodel, self.evaluator, self.is_distributed
        )
        self.model.train()
        if self.rank == 0:
            self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
            self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            logger.info("\n" + summary)
        synchronize()

        is_best = ap50_95 > self.best_ap
        self.best_ap = max(self.best_ap, ap50_95)
        self.save_ckpt("last_epoch", is_best)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank == 0:
            raw_model = (self.model.module if hasattr(self.model, "module")
                         else self.model)
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                # Keep the established inference contract: ``model`` is EMA.
                "model": (self.ema_model.ema.state_dict()
                          if self.use_model_ema else raw_model.state_dict()),
                "raw_model": raw_model.state_dict(),
                "ema_model": (self.ema_model.ema.state_dict()
                              if self.use_model_ema else None),
                "optimizer": self.optimizer.state_dict(),
                "optimizer_step": self.optimizer_step,
                "scaler": (self.scaler.state_dict()
                           if self.scaler.is_enabled() else None),
                "amp_dtype": self.amp_dtype_name,
                "skipped_optimizer_steps": self.skipped_optimizer_steps,
                "best_ap": self.best_ap,
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )
