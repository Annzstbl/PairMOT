import math
import os
import random
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import nms,box_iou

from .diffusion_losses import SetCriterionDynamicK, HungarianMatcherDynamicK
from .diffusion_models import DynamicHead

from yolox.utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from yolox.utils import synchronize
from detectron2.layers import batched_nms
from utils.box_ops import box_cxcywhtheta_to_xyxyxyxy, box_xyxyxyxy_to_cxcywhtheta
from loguru import logger
import time


ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# GT 加噪可视化：不需要时在 prepare_targets 里注释掉对 save_diffusion_noisy_gt_vis 的调用即可。
_DEBUG_DIFFUSION_NOISY_GT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "debug_diffusion_noisy_gt_vis"
)
_debug_diffusion_noisy_gt_seq = 0


def save_diffusion_noisy_gt_vis(
    head,
    image_size_xyxy,
    x_gt_boxes,
    d_boxes_full,
    timestep,
    d_mask,
    batch_idx,
):
    """
    只画「真实 GT」及其在训练里对应的加噪结果（x_start 里排在随机补全 placeholder 之前的那几行），
    不画补到 num_proposals 的随机框。d_boxes_full 为整包 (num_proposals,5)，此处只取前 n_gt_slot 行。
    """
    global _debug_diffusion_noisy_gt_seq
    if batch_idx != 0:
        return
    if d_mask is not None:
        return
    if x_gt_boxes is None or x_gt_boxes.numel() == 0:
        return
    tc = int(getattr(head, "track_candidate", 1))
    n_in = int(x_gt_boxes.shape[0])
    n_slot = n_in * tc
    if n_slot > d_boxes_full.shape[0]:
        return
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.warning("diffusion noisy-gt vis: need cv2/numpy, skip")
        return

    # 与 forward 一致：mate_shape=(B,C,H,W) -> images_whwh 为 [W,H,W,H,...]，此处 [0]=宽 [1]=高
    w_img = float(image_size_xyxy[0].detach().cpu())
    h_img = float(image_size_xyxy[1].detach().cpu())
    W, H = max(1, int(round(w_img))), max(1, int(round(h_img)))
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255

    def norm5_to_polygons_xyxyxyxy(b_norm: torch.Tensor, for_noisy: bool):
        b = b_norm.detach().float().cpu().clone()
        if for_noisy:
            # 高 t 时五维接近独立随机，w/h 常 ~0.5*图像边长 → 红框平均远大于小目标 GT，属预期。
            # 仅做角度折叠，减轻 theta 越界导致多边形「扯到画布外再 clip 成怪线」的情况。
            tn = b[:, 4]
            b[:, 4] = torch.remainder(torch.remainder(tn, 1.0) + 1.0, 1.0)
        cx = b[:, 0] * w_img
        cy = b[:, 1] * h_img
        bw = b[:, 2] * w_img
        bh = b[:, 3] * h_img
        theta_deg = b[:, 4] * 180.0 - 45.0
        cxcy = torch.stack([cx, cy, bw, bh, theta_deg], dim=1)
        return box_cxcywhtheta_to_xyxyxyxy(cxcy)

    idx = torch.tensor([tc] * n_in, device=x_gt_boxes.device, dtype=torch.long)
    gt_clean_vis = torch.repeat_interleave(x_gt_boxes, idx, dim=0)
    noisy_part = d_boxes_full[:n_slot].detach()
    polys_gt = norm5_to_polygons_xyxyxyxy(gt_clean_vis, for_noisy=False)
    polys_nz = norm5_to_polygons_xyxyxyxy(noisy_part, for_noisy=True)

    def draw_polys(polys_tensor, bgr, thickness=2):
        arr = polys_tensor.numpy().reshape(-1, 4, 2)
        for pts in arr:
            poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
            poly[:, :, 0] = np.clip(poly[:, :, 0], 0, W - 1)
            poly[:, :, 1] = np.clip(poly[:, :, 1], 0, H - 1)
            cv2.polylines(canvas, [poly], isClosed=True, color=bgr, thickness=thickness)

    draw_polys(polys_gt, (0, 200, 0), 2)
    draw_polys(polys_nz, (0, 0, 255), 2)

    t_int = int(timestep.detach().cpu()) if torch.is_tensor(timestep) else int(timestep)
    cap = f"t={t_int}  n_gt_slot={n_slot}  green=GT red=noisy(theta wrapped to draw)"
    cap2 = "scale OK: W,H from tensor[W,H]; large t -> red ~ random in [0,1]^5 not jitter"
    for text, y0, scale in ((cap, 22, 0.55), (cap2, 48, 0.45)):
        cv2.putText(canvas, text, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, text, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)

    os.makedirs(_DEBUG_DIFFUSION_NOISY_GT_DIR, exist_ok=True)
    _debug_diffusion_noisy_gt_seq += 1
    out_path = os.path.join(
        _DEBUG_DIFFUSION_NOISY_GT_DIR,
        f"noisy_gt_{_debug_diffusion_noisy_gt_seq:06d}_t{t_int}.png",
    )
    cv2.imwrite(out_path, canvas)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class DiffusionHead(nn.Module):
    """
    Implement DiffusionHead
    """

    def __init__(self,
                num_classes,
                width=1.0,
                strides=[8, 16, 32],
                num_proposals=500,
                num_heads=6,):
        super().__init__()
        self.device="cpu"
        self.dtype=torch.float32
        self.width=width
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.capture_train_debug = False
        self.last_train_debug = None
        # self.num_proposals = 512
        self.hidden_dim = int(256*width)
        self.num_heads = num_heads

        # build diffusion
        timesteps = 1000
        sampling_timesteps = 1
        self.objective = 'pred_x0'
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # tracking setting
        self.inference_time_range=1
        self.track_candidate=1
        self.candidate_num_strategy=max

        # 训练时扩散初值 x_start（prepare_diffusion_concat）：
        #   "gt_then_rand" — 与原逻辑一致：GT（×track_candidate）再 randn 补满 num_proposals；
        #   "randn_proposals" — 全部 num_proposals 行均为 randn 初值（分布与原 box_placeholder 一致），不拼 GT。
        # self.diffusion_x_start_mode = "randn_proposals"
        self.diffusion_x_start_mode = "gt_then_rand"

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = 2.0
        self.box_renewal = True
        self.use_ensemble = True

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Build Dynamic Head.
        class_weight = 2.0
        giou_weight = 2.0
        l1_weight = 5.0
        no_object_weight =0.1
        self.deep_supervision = True
        self.use_focal = True
        self.use_fed_loss = False
        self.use_nms = False
        self.pooler_resolution=7
        # self.noise_strategy="xywh"
        self.noise_strategy="xywhtheta"
   
        self.head = DynamicHead(num_classes,self.hidden_dim,self.pooler_resolution,strides,[self.hidden_dim]*len(strides),return_intermediate=self.deep_supervision,num_heads=self.num_heads,use_focal=self.use_focal,use_fed_loss=self.use_fed_loss)
        # Loss parameters:

        # Build Criterion.
        matcher = HungarianMatcherDynamicK(
            cost_class=class_weight, cost_bbox=l1_weight, cost_giou=giou_weight, use_focal=self.use_focal,use_fed_loss=self.use_fed_loss
        )
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_iou": giou_weight}

        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes"]

        self.criterion = SetCriterionDynamicK(
            num_classes=self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight,
            losses=losses, use_focal=self.use_focal,use_fed_loss=self.use_fed_loss)

        # 缓存最近一次训练 forward 的匹配框对（来自 criterion.last_match）
        # 方便在 Trainer 中取出做可视化，不污染 loss_dict
        self.last_match = None

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, backbone_feats, images_whwh, x, t, lost_features=None, fix_bboxes=False, x_self_cond=None, clip_x_start=False):

        def prepare(x, images_whwh):
            """
            将 diffusion 空间的 5D 框转换成模型输入格式
            输入: x (batch, num_proposals, 5) -> (cx, cy, w, h, theta_norm)，范围 [-scale, scale]
            输出: (batch, num_proposals, 5) -> (cx, cy, w, h, theta)，绝对坐标，theta 为度数
            """
            x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
            x_boxes = ((x_boxes / self.scale) + 1) / 2  # [-scale, scale] -> [0, 1]
            x_boxes = x_boxes.clone()
            x_boxes[..., :4] = x_boxes[..., :4] * images_whwh[:, None, :4]
            x_boxes[..., 4] = x_boxes[..., 4] * 180 - 45
            return x_boxes

        def post(x_start, images_whwh):
            """
            将模型输出转换回 diffusion 空间（prepare 的逆操作）
            输入: x_start (batch, num_proposals, 5) -> (cx, cy, w, h, theta)，绝对坐标，theta 为度数
            输出: (batch, num_proposals, 5) -> (cx, cy, w, h, theta_norm)，范围 [-scale, scale]
            """
            x_start = x_start.clone()
            x_start[..., :4] = x_start[..., :4] / images_whwh[:, None, :4]
            x_start[..., 4] = (x_start[..., 4] + 45) / 180
            x_start = (x_start * 2 - 1.) * self.scale
            x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
            return x_start
        
        bs=len(x)//2
        bboxes=prepare(x,images_whwh=images_whwh)
        start_time=time.time()
        outputs_class, outputs_coord,outputs_score = self.head(backbone_feats,torch.split(bboxes,bs,dim=0),t,lost_features,fix_bboxes)
        end_time=time.time()

        x_start = outputs_coord[-1]  # (batch, num_proposals, 4) predict boxes: absolute coordinates (x1, y1, x2, y2)
        x_start=post(x_start,images_whwh=images_whwh)
        pred_noise = self.predict_noise_from_start(x,t,x_start)
        return ModelPrediction(pred_noise, x_start), outputs_class,outputs_coord,outputs_score,end_time-start_time
    
    @torch.no_grad()
    def new_ddim_sample(self,backbone_feats,images_whwh,ref_targets=None,dynamic_time=True,num_timesteps=1,num_proposals=500,inference_time_range=1,track_candidate=1,diffusion_t=200,clip_denoised=True):
        batch = images_whwh.shape[0]//2
        self.sampling_timesteps,self.num_proposals,self.track_candidate,self.inference_time_range = num_timesteps,num_proposals,track_candidate,inference_time_range
        shape = (batch, self.num_proposals, 5)
        cur_bboxes = torch.randn(shape, device=self.device, dtype=self.dtype)
        ref_t_list=[]
        track_t_list=[]
        total_time=0
        if ref_targets is None or self.track_candidate==0:
            ref_bboxes = torch.randn(shape, device=self.device, dtype=self.dtype)
            cur_bboxes = torch.randn(shape, device=self.device, dtype=self.dtype)
            for i in range(batch):
                t = torch.randint(self.num_timesteps-self.inference_time_range, self.num_timesteps,(2,), device=self.device).long()
                if dynamic_time:
                    ref_t,track_t=t[0],t[1]
                else:
                    ref_t,track_t=t[0],t[0]
                ref_t_list.append(ref_t)
                track_t_list.append(track_t)
        else:
            labels = ref_targets[..., :5]
            nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
            shape = (batch, self.num_proposals, 5)
            diffused_boxes = []
            cur_diffused_boxes = []
            for batch_idx, num_gt in enumerate(nlabel):
                # ref_targets from tracker are already normalized (0-1): (cx, cy, w, h, theta_norm)
                gt_boxes = labels[batch_idx, :num_gt, :].clone()
                # t = torch.randint(self.num_timesteps-self.inference_time_range, self.num_timesteps,(2,), device=self.device).long()
                # if dynamic_time:
                #     ref_t,track_t=t[0],t[1]
                # else:
                #     ref_t,track_t=t[0],t[0]
                if batch_idx==0:
                    ref_t=diffusion_t
                    track_t=diffusion_t
                else:
                    ref_t=diffusion_t
                    track_t=diffusion_t
                    self.track_candidate=4
                d_boxes,d_noise,ref_label= self.prepare_diffusion_concat(gt_boxes,ref_t)
                diffused_boxes.append(d_boxes)
                ref_t_list.append(ref_t)
                d_boxes,d_noise,ref_label= self.prepare_diffusion_concat(gt_boxes,track_t,ref_label)
                cur_diffused_boxes.append(d_boxes)
                track_t_list.append(track_t)
            ref_bboxes=torch.stack(diffused_boxes)
            cur_bboxes=torch.stack(cur_diffused_boxes)


        sampling_timesteps, eta= self.sampling_timesteps, self.ddim_sampling_eta

        def get_time_pairs(t,sampling_timesteps):
            # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
            times = torch.linspace(-1, t - 1, steps=sampling_timesteps + 1)
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
            return time_pairs
        
        ref_t_time_pairs_list=torch.tensor([get_time_pairs(t,sampling_timesteps) for t in ref_t_list],device=self.device,dtype=torch.long)
        track_t_time_pairs_list=torch.tensor([get_time_pairs(t,sampling_timesteps) for t in track_t_list],device=self.device,dtype=torch.long)
        # (batch,sampling_timesteps,2)
        bboxes=torch.cat([ref_bboxes,cur_bboxes],dim=0)

        x_start = None
        # for (ref_time, ref_time_next),(cur_time, cur_time_next) in zip(ref_time_pairs,cur_time_pairs):
        for sampling_timestep in range(sampling_timesteps):
            is_last=sampling_timestep==(sampling_timesteps-1)

            ref_time_cond = ref_t_time_pairs_list[:,sampling_timestep,0]
            cur_time_cond = track_t_time_pairs_list[:,sampling_timestep,0]

            time_cond=torch.cat([ref_time_cond,cur_time_cond],dim=0)

            self_cond = x_start if self.self_condition else None

            preds, outputs_class, outputs_coord,outputs_score,association_time = self.model_predictions(backbone_feats,images_whwh,bboxes,time_cond,fix_bboxes=False,
                                                                         x_self_cond=self_cond, clip_x_start=clip_denoised)
            total_time+=association_time
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start
                

            if is_last:
                bboxes = x_start
                continue

            if self.box_renewal:  # filter
                remain_list=[]
                pre_remain_bboxes=[]
                pre_remain_x_start=[]
                pre_remain_pred_noise=[]
                cur_remain_bboxes=[]
                cur_remain_x_start=[]
                cur_remain_pred_noise=[]
                for i in range(batch):
                    # if i==0:
                    #     remain_list.append(len(pred_noise[i,:,:]))
                    #     pre_remain_pred_noise.append(pred_noise[i,:,:])
                    #     cur_remain_pred_noise.append(pred_noise[i+batch,:,:])
                    #     pre_remain_x_start.append(x_start[i,:,:])
                    #     cur_remain_x_start.append(x_start[i+batch,:,:])
                    #     pre_remain_bboxes.append(bboxes[i,:,:])
                    #     cur_remain_bboxes.append(bboxes[i+batch,:,:])
                    # else:
                    threshold = 0.2
                    score_per_image = outputs_score[-1][i]
                    # pre_score=torch.sqrt(score_per_image*torch.sigmoid(outputs_class[-1][i]))
                    # cur_score=torch.sqrt(score_per_image*torch.sigmoid(outputs_class[-1][i+batch]))
                    # value=((pre_score+cur_score)/2).flatten()
                    value, _ = torch.max(score_per_image, -1, keepdim=False)
                    keep_idx = value >=threshold
                    num_remain = torch.sum(keep_idx)
                    remain_list.append(num_remain)
                    pre_remain_pred_noise.append(pred_noise[i,keep_idx,:])
                    cur_remain_pred_noise.append(pred_noise[i+batch,keep_idx,:])
                    pre_remain_x_start.append(x_start[i,keep_idx,:])
                    cur_remain_x_start.append(x_start[i+batch,keep_idx,:])
                    pre_remain_bboxes.append(bboxes[i,keep_idx,:])
                    cur_remain_bboxes.append(bboxes[i+batch,keep_idx,:])
                x_start=pre_remain_x_start+cur_remain_x_start
                bboxes=pre_remain_bboxes+cur_remain_bboxes
                pred_noise=pre_remain_pred_noise+cur_remain_pred_noise

            def diffusion(sampling_times,bboxes,x_start,pred_noise):
                
                times,time_nexts=sampling_times[:,0],sampling_times[:,1]

                alpha = torch.tensor([self.alphas_cumprod[time] for time in times],dtype=self.dtype,device=self.device)
                alpha_next = torch.tensor([self.alphas_cumprod[time_next] for time_next in time_nexts],dtype=self.dtype,device=self.device)

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                if self.box_renewal:
                    for i in range(batch):
                        noise = torch.randn_like(bboxes[i])
                        bboxes[i] = x_start[i] * alpha_next[i].sqrt() + \
                            c[i] * pred_noise[i] + \
                            sigma[i] * noise
                        
                        # bboxes are 5D in diffusion space; pad to fixed num_proposals with matching last-dim.
                        bboxes[i] = torch.cat(
                            (
                                bboxes[i],
                                torch.randn(
                                    self.num_proposals - remain_list[i],
                                    5,
                                    device=self.device,
                                    dtype=self.dtype,
                                ),
                            ),
                            dim=0,
                        )
                else:
                    noise = torch.randn_like(bboxes)

                    bboxes = x_start * alpha_next.sqrt()[:,None,None] + \
                        c[:,None,None] * pred_noise + \
                        sigma[:,None,None] * noise
                
                return bboxes
            
            bboxes[:batch]=diffusion(ref_t_time_pairs_list[:,sampling_timestep],bboxes[:batch],x_start[:batch],pred_noise[:batch])
            bboxes[batch:]=diffusion(track_t_time_pairs_list[:,sampling_timestep],bboxes[batch:],x_start[batch:],pred_noise[batch:])

            if self.box_renewal:
                bboxes=torch.stack(bboxes)

        box_cls = outputs_class[-1]
        box_pred = outputs_coord[-1]
        conf_score=outputs_score[-1]

        # box_pred: (2*batch, num_proposals, 5) = (cx, cy, w, h, theta_deg)
        # box_cls:  (2*batch, num_proposals, C)  多类别 logits（C = num_classes）
        # 返回给 tracker 时保留完整多类 logits，后续在 postprocess 里取 max 得到 class_id/conf
        return torch.cat([box_pred, box_cls], dim=-1), conf_score.view(batch, -1, 1), total_time
    
    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self,features,mate_info,targets=None):
        mate_shape,mate_device,mate_dtype=mate_info
        self.device=mate_device
        self.dtype=mate_dtype
        b,_,h,w=mate_shape
        
        images_whwh = torch.tensor([w, h, w, h, w, h, w, h],  dtype=self.dtype,  device=self.device)[None,:].expand(2*b, 8)

        if not self.training:
            results = self.new_ddim_sample(features,images_whwh,targets,dynamic_time=False)
            return results

        if self.training:
            targets, x_boxes, noises, t = self.prepare_targets(targets,images_whwh)
            t=t.squeeze(-1)
            # t[b:]=t[:b]
            x_boxes[:,:,:4] = x_boxes[:,:,:4] * images_whwh[:,None,:4]
            x_boxes[:,:,4] = x_boxes[:,:,4] * 180 - 45
            initial_boxes_debug = (
                x_boxes.detach().float().cpu().clone()
                if self.capture_train_debug else None)
            pre_x_boxes,cur_x_boxes=torch.split(x_boxes,b,dim=0)

            outputs_class,outputs_coord,outputs_score = self.head(features,(pre_x_boxes,cur_x_boxes),t)
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],'pred_scores':outputs_score[-1]}

            if self.deep_supervision:
                output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b,'pred_scores': c}
                                         for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1],outputs_score[:-1])]
            loss_dict = self.criterion(output, targets)  
            if self.capture_train_debug:
                gt_boxes = []
                gt_classes = []
                for target in targets:
                    boxes = target["boxes"].detach().float().clone()
                    size = target["image_size_xyxyxyxy"].detach().float()
                    boxes[:, :4] *= size[:4]
                    boxes[:, 4] = boxes[:, 4] * 180.0 - 45.0
                    gt_boxes.append(boxes.cpu())
                    gt_classes.append(
                        target["labels"].detach().long().cpu().clone())
                self.last_train_debug = {
                    "timesteps": t.detach().long().cpu().clone(),
                    "image_whwh": images_whwh.detach().float().cpu().clone(),
                    "gt_boxes": gt_boxes,
                    "gt_classes": gt_classes,
                    "initial_boxes": initial_boxes_debug,
                    "stage_boxes": outputs_coord.detach().float().cpu().clone(),
                    "stage_logits": outputs_class.detach().float().cpu().clone(),
                    "stage_scores": outputs_score.detach().float().cpu().clone(),
                    "assignments": self.criterion.last_debug_assignments,
                }
            else:
                self.last_train_debug = None
            # 从 criterion 中取出匹配后的 pred/gt 框对（cpu tensor list）
            self.last_match = getattr(self.criterion, "last_match", None)
            weight_dict = self.criterion.weight_dict

            for k in loss_dict.keys():
                if k in weight_dict: 
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
 
    def prepare_diffusion_repeat(self,gt_boxes,t,ref_repeat_tensor=None):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.full((1,),t,device=self.device).long()

        noise = torch.randn(self.num_proposals,4,device=self.device,dtype=self.dtype)

        num_gt = gt_boxes.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=self.dtype, device=self.device)
            num_gt = 1

        num_repeat = self.num_proposals // num_gt  # number of repeat except the last gt box in one image
        repeat_tensor = [num_repeat] * (num_gt - self.num_proposals % num_gt) + [num_repeat + 1] * (
                self.num_proposals % num_gt)
        assert sum(repeat_tensor) == self.num_proposals
        random.shuffle(repeat_tensor)
        repeat_tensor = torch.tensor(repeat_tensor, device=self.device)
        if ref_repeat_tensor is not None:
            repeat_tensor=ref_repeat_tensor

        gt_boxes = (gt_boxes * 2. - 1.) * self.scale
        x_start = torch.repeat_interleave(gt_boxes, repeat_tensor, dim=0)

        if self.noise_strategy=="xy":
            noise[:,2:]=0
        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        if self.training:
            x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
            x = ((x / self.scale) + 1) / 2.

            diff_boxes = box_cxcywh_to_xyxy(x)
        else:
            diff_boxes=x

        return diff_boxes,noise,repeat_tensor

    def prepare_diffusion_concat(self,gt_boxes,t,ref_mask=None):
        """
        :param gt_boxes: (cx, cy, w, h, theta_norm), normalized 5D boxes
        :param num_proposals:
        """
        if self.training:
            self.track_candidate=1
        t = torch.full((1,),t,device=self.device).long()
        noise = torch.randn(self.num_proposals, 5, device=self.device,dtype=self.dtype)
        select_mask=None

        mode = getattr(self, "diffusion_x_start_mode", "gt_then_rand")
        if mode == "randn_proposals":
            # 与原先 box_placeholder 同分布：N(0,1)/6+0.5，w/h/theta 下限裁剪
            x_start = torch.randn(self.num_proposals, 5, device=self.device, dtype=self.dtype) / 6.0 + 0.5
            x_start[:, 2:5] = torch.clip(x_start[:, 2:5], min=1e-4)
        else:
            num_gt = gt_boxes.shape[0]*self.track_candidate
            if not num_gt:  # generate fake gt boxes if empty gt boxes
                # 5D 占位框: (cx, cy, w, h, theta_norm)，theta_norm 取中间值 0.5
                gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1., 0.5]], dtype=self.dtype, device=self.device)
                num_gt = 1
            else:
                gt_boxes=torch.repeat_interleave(gt_boxes,torch.tensor([self.track_candidate]*gt_boxes.shape[0],device=self.device),dim=0)
            if num_gt < self.num_proposals:
                box_placeholder = torch.randn(self.num_proposals - num_gt, 5,
                                              device=self.device,dtype=self.dtype) / 6. + 0.5  # 3sigma = 1/2 --> sigma: 1/6

                box_placeholder[:, 2:5] = torch.clip(box_placeholder[:, 2:5], min=1e-4)
                x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
            elif num_gt > self.num_proposals:
                select_mask = [True] * self.num_proposals + [False] * (num_gt - self.num_proposals)
                random.shuffle(select_mask)
                if ref_mask is not None:
                    select_mask=ref_mask
                x_start = gt_boxes[select_mask]
            else:
                x_start = gt_boxes

        x_start = (x_start * 2. - 1.) * self.scale
        if self.noise_strategy=="xy":
            noise[:,2:]=0
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        if self.training:
            x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
            x = ((x / self.scale) + 1) / 2.
            diff_boxes = x  

        else:
            diff_boxes = x

        return diff_boxes, noise, select_mask

    def prepare_targets(self,targets,images_whwh):
        # labels = targets[..., :5]        
        labels = targets[..., :9]
        # 重要：不要用 sum()>0 来判断 padding。
        # 经过随机透视/裁剪等增强后，坐标可能为负，sum 可能 <=0，导致把真实 GT 误当 padding 丢掉。
        # padding 的语义是“8 点坐标全为 0”。
        coord_nonzero = (labels[:, :, 1:9].abs().sum(dim=2) > 0)
        nlabel = coord_nonzero.sum(dim=1)  # number of objects
        self.last_nlabel = nlabel.tolist()  # 供 trainer 在 vis 图上叠加调试信息
        
        # print(nlabel)
        
        new_targets = []
        diffused_boxes = []
        noises = []
        ts = []
        select_mask={}
        # select_t={}
        # select_gt_boxes={}
        for batch_idx,num_gt in enumerate(nlabel):
            target = {}
            # gt_bboxes_per_image = box_cxcywh_to_xyxy(labels[batch_idx, :num_gt, 1:5])
            gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:9]
            gt_classes = labels[batch_idx, :num_gt, 0]
            image_size_xyxy = images_whwh[batch_idx]
            gt_boxes = gt_bboxes_per_image  / image_size_xyxy
            # cxcywh
            # gt_boxes = box_xyxyxyxy_to_cxcywhtheta(gt_boxes)
            # theta_norm = (gt_boxes[:, 4] + 45) / 180
            # gt_boxes[:, 4] = theta_norm
            # 重要：不要在 (x/w, y/h) 的非等比归一化坐标系里去计算旋转框的 w/h/theta，
            # 否则旋转时欧氏距离被扭曲，会导致监督框“细长/角度异常”。
            # 正确做法：先在像素坐标系下从 4 点得到 (cx,cy,w,h,theta_deg)，再按 (w_img,h_img) 归一化。
            gt_boxes_abs = box_xyxyxyxy_to_cxcywhtheta(gt_bboxes_per_image)  # absolute (cx,cy,w,h,theta_deg)
            w_img = image_size_xyxy[0].clamp(min=1.0)
            h_img = image_size_xyxy[1].clamp(min=1.0)
            gt_boxes = gt_boxes_abs.clone()
            gt_boxes[:, 0] = gt_boxes[:, 0] / w_img
            gt_boxes[:, 1] = gt_boxes[:, 1] / h_img
            gt_boxes[:, 2] = gt_boxes[:, 2] / w_img
            gt_boxes[:, 3] = gt_boxes[:, 3] / h_img
            gt_boxes[:, 4] = (gt_boxes_abs[:, 4] + 45.0) / 180.0  # theta_norm in [0,1]
            x_gt_boxes=gt_boxes
            d_t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()[0]
            ## baseline setting
            # if batch_idx<len(nlabel)//2:
            #     d_t = torch.randint(0, 40, (1,), device=self.device).long()[0]
            # else:
            #     d_t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()[0]
            # if select_t.get(batch_idx%(len(nlabel)//2),None) is not None:
            #     d_t=select_t.get(batch_idx%(len(nlabel)//2),None)
            # if select_gt_boxes.get(batch_idx%(len(nlabel)//2),None) is not None:
            #     x_gt_boxes=select_gt_boxes.get(batch_idx%(len(nlabel)//2),None)    
            d_boxes,d_noise,d_mask= self.prepare_diffusion_concat(x_gt_boxes,d_t,select_mask.get(batch_idx%(len(nlabel)//2),None))
            # 调试：GT vs 加噪框可视化（randn_proposals 初值与 GT 无对应关系，跳过）
            # if getattr(self, "diffusion_x_start_mode", "gt_then_rand") != "randn_proposals":
            #     save_diffusion_noisy_gt_vis(
            #         self, image_size_xyxy, x_gt_boxes, d_boxes, d_t, d_mask, batch_idx
            #     )
            if d_mask is not None:
                select_mask[batch_idx%(len(nlabel)//2)]=d_mask
            # if d_t is not None:
            #     select_t[batch_idx%(len(nlabel)//2)]=d_t
            # if select_gt_boxes.get(batch_idx%(len(nlabel)//2),None) is None:
            #     select_gt_boxes[batch_idx%(len(nlabel)//2)]=gt_boxes 
            diffused_boxes.append(d_boxes)
            noises.append(d_noise)
            ts.append(d_t)
            target["labels"] = gt_classes.long()
            target["boxes"] = gt_boxes
            target["boxes_xyxyxyxy"] = gt_bboxes_per_image
            target["image_size_xyxyxyxy"] = image_size_xyxy
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt
            new_targets.append(target)

        return new_targets, torch.stack(diffused_boxes), torch.stack(noises), torch.stack(ts)
