'''
    其它各种形式和 mmrotate 格式转换
'''
from pathlib import Path
import torch
import math
import cv2
from hsmot.mmlab.hs_mmrotate import poly2obb, obb2poly
import numpy as np
from sklearn.cluster import KMeans
import os
import pickle
from tqdm import tqdm
import time
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import json
try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False
    
class MotrToMmrotate:
    
    def __init__(self):
        pass

    def __call__(self, results):
        return results

class MotipToMmrotate:
    def __init__(self):
        pass

    def __call__(self, results):
        return results

def rotate_norm_angles_to_angles(angles, version='le135'):
    if version == 'oc':
        raise NotImplementedError
    elif version == 'le135':
        angle_range = 1
        angle_offset = -1/4
    elif version == 'le90':
        angle_range = 1
        angle_offset = -1/2
    angle_range *= math.pi
    angle_offset *= math.pi
    angles = angles * angle_range + angle_offset
    return angles

def rotate_boxes_to_norm_boxes(boxes, img_shape, version='le135'):
    '''
        计算从真实坐标到归一化坐标的转换
    '''
    h, w = img_shape
    if version == 'oc':
        raise NotImplementedError
    elif version == 'le135':
        angle_range = 1
        angle_offset = -1/4
    elif version == 'le90':
        angle_range = 1
        angle_offset = -1/2
    angle_range *= math.pi
    angle_offset *= math.pi
    norm_boxes = (boxes - torch.as_tensor([0,0,0,0,angle_offset], dtype=boxes.dtype, device=boxes.device)) /  torch.as_tensor([w, h, w, h, angle_range],dtype=boxes.dtype, device=boxes.device)
    return norm_boxes

def rotate_norm_boxes_to_boxes(norm_boxes, img_shape, version='le135'):
    '''
        计算从归一化坐标到真实坐标的转换
    '''
    h, w = img_shape
    if version == 'oc':
        raise NotImplementedError
    elif version == 'le135':
        angle_range = 1
        angle_offset = -1/4
    elif version == 'le90':
        angle_range = 1
        angle_offset = -1/2
    angle_range *= math.pi
    angle_offset *= math.pi
    boxes = norm_boxes * torch.as_tensor([w, h, w, h, angle_range],dtype=norm_boxes.dtype, device=norm_boxes.device) + torch.as_tensor([0,0,0,0,angle_offset],dtype=norm_boxes.dtype, device=norm_boxes.device)
    return boxes

def version_str_to_index(version: str) -> int:
    if version == 'oc':
        return 1
    elif version == 'le90':
        return 2
    elif version == 'le135':
        return 3
    else:
        raise ValueError
    
def version_index_to_str(version: int) -> str:
    if version == 1:
        return 'oc'
    elif version == 2:
        return 'le90'
    elif version == 3:
        return 'le135'
    else:
        raise ValueError


def _extract_effective_and_padded_shape(results):
    """Extract effective (pre-pad) and padded image shapes as (h, w)."""
    img_tensor = results['img'].data
    pad_h, pad_w = img_tensor.shape[1:]
    effective_h, effective_w = int(pad_h), int(pad_w)

    transform_metas = results.get('img_metas', None)
    if transform_metas is not None:
        meta_data = transform_metas.data if hasattr(transform_metas, 'data') else transform_metas
        if isinstance(meta_data, dict):
            img_shape = meta_data.get('img_shape', None)
            if img_shape is not None and len(img_shape) >= 2:
                effective_h = int(img_shape[0])
                effective_w = int(img_shape[1])

    return (effective_h, effective_w), (int(pad_h), int(pad_w))
    
class MmrotateToMotr:
    def __init__(self, version='le135'):
        self.version = version

    def __call__(self, results_list):
        images = []
        targets = []
        img_metas = []

        if self.version == 'oc':
            raise NotImplementedError
        elif self.version == 'le135':
            angle_range = 1
            angle_offset = -1/4
        elif self.version == 'le90':
            angle_range = 1
            angle_offset = -1/2
        angle_range *= math.pi
        angle_offset *= math.pi

        for results in results_list:
            images.append(results['img'].data)
            (h, w), (pad_h, pad_w) = _extract_effective_and_padded_shape(results)

            img_metas.append({
                'img_shape': torch.as_tensor((h, w), device=results['img'].data.device),
                'pad_shape': torch.as_tensor((pad_h, pad_w), device=results['img'].data.device),
                'version':torch.as_tensor(version_str_to_index(self.version), dtype=torch.int, device=results['img'].data.device) })
            # gt_bboxes-> norm_gt_bboxes
            gt_bboxes = results['gt_bboxes'].data
            # norm_gt_bboxes = rotate_boxes_to_norm_boxes(gt_bboxes, (h, w), self.version) 函数版，未测试
            norm_gt_bboxes = (gt_bboxes - torch.as_tensor([0,0,0,0,angle_offset], dtype=gt_bboxes.dtype, device=gt_bboxes.device)) /  torch.as_tensor([w, h, w, h, angle_range],dtype=gt_bboxes.dtype, device=gt_bboxes.device)
            targets.append({'boxes':results['gt_bboxes'].data, 'norm_boxes':norm_gt_bboxes, 'labels':results['gt_labels'].data, 'obj_ids':results['gt_trackids'].data,
                            })
        return images, targets, img_metas

class MmrotateToMotrv2:
    def __init__(self, version='le135'):
        self.version = version

    def __call__(self, results_list):
        images = []
        targets = []
        img_metas = []

        if self.version == 'oc':
            raise NotImplementedError
        elif self.version == 'le135':
            angle_range = 1
            angle_offset = -1/4
        elif self.version == 'le90':
            angle_range = 1
            angle_offset = -1/2
        angle_range *= math.pi
        angle_offset *= math.pi

        for results in results_list:
            images.append(results['img'].data)
            (h, w), (pad_h, pad_w) = _extract_effective_and_padded_shape(results)

            img_metas.append({
                'img_shape': torch.as_tensor((h, w), device=results['img'].data.device),
                'pad_shape': torch.as_tensor((pad_h, pad_w), device=results['img'].data.device),
                'version':torch.as_tensor(version_str_to_index(self.version), dtype=torch.int, device=results['img'].data.device) })
            # gt_bboxes-> norm_gt_bboxes
            gt_bboxes = results['gt_bboxes'].data
            norm_gt_bboxes = (gt_bboxes - torch.as_tensor([0,0,0,0,angle_offset], dtype=gt_bboxes.dtype, device=gt_bboxes.device)) /  torch.as_tensor([w, h, w, h, angle_range],dtype=gt_bboxes.dtype, device=gt_bboxes.device)

            # process proposals
            proposals = results['proposals'].data
            proposal_scores = results['proposal_scores'].data
            norm_proposals = (proposals - torch.as_tensor([0,0,0,0,angle_offset], dtype=proposals.dtype, device=proposals.device)) /  torch.as_tensor([w, h, w, h, angle_range],dtype=proposals.dtype, device=proposals.device)

            targets.append({
                'boxes':results['gt_bboxes'].data, 
                'norm_boxes':norm_gt_bboxes, 
                'labels':results['gt_labels'].data, 
                'obj_ids':results['gt_trackids'].data,
                'proposals':results['proposals'].data,
                'norm_proposals':norm_proposals,
                'proposal_scores':proposal_scores,})
            
        return images, targets, img_metas
    
class MmrotateToMotip:
    def __init__(self, version='le135'):
        self.version = version

    def __call__(self, results_list):
        images = []
        targets = []
        img_metas = []

        if self.version == 'oc':
            raise NotImplementedError
        elif self.version == 'le135':
            angle_range = 1
            angle_offset = -1/4
        elif self.version == 'le90':
            angle_range = 1
            angle_offset = -1/2
        angle_range *= math.pi
        angle_offset *= math.pi

        for results in results_list:
            images.append(results['img'].data)
            (h, w), (pad_h, pad_w) = _extract_effective_and_padded_shape(results)

            img_metas.append({
                'img_shape': torch.as_tensor((h, w), device=results['img'].data.device),
                'pad_shape': torch.as_tensor((pad_h, pad_w), device=results['img'].data.device),
                'version':torch.as_tensor(version_str_to_index(self.version), dtype=torch.int, device=results['img'].data.device),
                'transform_metas':results['img_metas']
                })
            # gt_bboxes-> norm_gt_bboxes
            gt_bboxes = results['gt_bboxes'].data
            # norm_gt_bboxes = rotate_boxes_to_norm_boxes(gt_bboxes, (h, w), self.version) 函数版，未测试
            norm_gt_bboxes = (gt_bboxes - torch.as_tensor([0,0,0,0,angle_offset], dtype=gt_bboxes.dtype, device=gt_bboxes.device)) /  torch.as_tensor([w, h, w, h, angle_range],dtype=gt_bboxes.dtype, device=gt_bboxes.device)
            # 构造heatmap
            heatmap = HeatmapFromRotateGt.heatmap_from_rotate_gt_xywha(gt_bboxes, (pad_h, pad_w), self.version)

            targets.append({'boxes':results['gt_bboxes'].data, 'norm_boxes':norm_gt_bboxes, 'labels':results['gt_labels'].data, 'obj_ids':results['gt_trackids'].data, 'heatmap':heatmap})

        return images, targets, img_metas


class HeatmapFromRotateGt:
    """
    统一的旋转框热力图生成器：
    - 输入 OBB 或 4 点多边形（自动互转）
    - 连续 σ(s) 由 √(wh) 尺度生成；s=8 时峰值 = 0.5 → σ* = sqrt(1/pi)
    - 支持 normalized（真密度）、fixed_peak（固定峰值）、capped_peak（真密度但截断到上限）
    - 聚合：mean/sum/max
    """

    # ---------- 公共入口 ----------
    @staticmethod
    def heatmap_from_rotate_gt_xyxyxyxy(gt_xyxyxyxy: torch.Tensor, img_shape, version='le135',
                                        **kwargs):
        """
        gt_xyxyxyxy: [N, 8]，四点有序多边形
        img_shape: (H, W)
        """
        gt_xywha = poly2obb(gt_xyxyxyxy, version)  # 这里假定返回 tensor，[N,5] (xc,yc,w,h,theta[rad])
        out = HeatmapFromRotateGt._heatmap_from_rotate_gt_xywha_fast(gt_xywha, img_shape, **kwargs)
        return out

    @staticmethod
    def heatmap_from_rotate_gt_xywha(gt_xywha: torch.Tensor, img_shape, version='le135', scores: torch.Tensor = None, **kwargs):
        """
        gt_xywha: [N,5] (xc,yc,w,h,theta[rad])
        scores: [N,] 可选的置信度分数，如果提供则对每个框的贡献进行加权
        """
        out = HeatmapFromRotateGt._heatmap_from_rotate_gt_xywha_fast(gt_xywha, img_shape, version=version, scores=scores, **kwargs)
        return out 


    @staticmethod
    def _heatmap_from_rotate_gt_xywha_fast(gt_bboxes: torch.Tensor, img_shape,
                                     version='le135', #暂时没有作用
                                     mode: str = 'fixed_peak',   # 'normalized'|'fixed_peak'|'capped_peak'
                                     peak: float = 1.0,          # for fixed/capped
                                     reduce: str = 'sum',       # 'mean'|'sum'|'max'
                                     chunk_boxes: int = 1,
                                     k = 5.0, #5sigma覆盖
                                     scores: torch.Tensor = None,  # [N,] 可选的置信度分数，用于加权
                                     **sigma_kwargs,):
        """
        gt_bboxes: [N,5] (xc,yc,w,h,theta[rad])
        scores: [N,] 可选的置信度分数，如果提供则对每个框的贡献进行加权
        mode:
          - 'normalized': 真密度  exp(-0.5*q) / (2π σw σh)
          - 'fixed_peak': 以 exp(-0.5*q) 为形状，中心值固定为 peak（非真密度）
          - 'capped_peak': 真密度，但用 min(phi, peak) 截断最高值到 peak
        """
        H, W = img_shape
        device = gt_bboxes.device
        assert chunk_boxes == 1
        
        if scores is not None:
            assert len(gt_bboxes) == len(scores), f"gt_bboxes和scores长度不匹配: {len(gt_bboxes)} vs {len(scores)}"
        
        ys = torch.arange(H, device=device, dtype=torch.float32)
        xs = torch.arange(W, device=device, dtype=torch.float32)
        Y, X = torch.meshgrid(ys, xs, indexing='ij')

        N = gt_bboxes.shape[0]
        if N == 0:
            return torch.zeros(H, W, device=device)

        accum = torch.zeros(1, H, W, device=device)

        for st in range(0, N, chunk_boxes):
            ed = min(st + chunk_boxes, N)
            b = gt_bboxes[st:ed] #[1, 5]
            if b.numel() == 0:
                continue
            
            # 如果提供了scores，获取对应的score
            if scores is not None:
                score_batch = scores[st:ed]  # [1,]
                # 跳过score <= 0的框
                if score_batch.item() <= 0:
                    continue

            xc, yc = b[:,0], b[:,1]
            w  = b[:,2].clamp_min(1e-6)
            h  = b[:,3].clamp_min(1e-6)
            th = b[:,4]
            c, s = torch.cos(th), torch.sin(th)

            # 连续 σ(s)：几何均值 + γ 幂律 + 各向异性比例
            sig_w, sig_h = HeatmapFromRotateGt.sigmas_from_box(w, h, **sigma_kwargs)

            peak_scale = HeatmapFromRotateGt.peak_scale_from_box(w, h, **sigma_kwargs)

            rx = k * torch.sqrt((sig_w * c)**2 + (sig_h * s)**2) # [1,]
            ry = k * torch.sqrt((sig_w * s)**2 + (sig_h * c)**2) # [1,]

            x0 = int(torch.clamp(xc - rx, 0, W))
            x1 = int(torch.clamp(xc + rx, 0, W))
            y0 = int(torch.clamp(yc - ry, 0, H))
            y1 = int(torch.clamp(yc + ry, 0, H))


            # 计算局部区域的偏差
            dx = X[y0:y1, x0:x1] - xc[:, None, None]
            dy = Y[y0:y1, x0:x1] - yc[:, None, None]
            dxp = c[:, None, None] * dx + s[:, None, None] * dy
            dyp = -s[:, None, None] * dx + c[:, None, None] * dy

            qf = (dxp / sig_w[:, None, None])**2 + (dyp / sig_h[:, None, None])**2  # (n, h', w')

            # 根据 mode 计算贡献
            if mode == 'normalized':
                denom = (2.0 * math.pi) * sig_w[:, None, None] * sig_h[:, None, None]
                contrib = torch.exp(-0.5 * qf) / denom
            elif mode == 'fixed_peak':
                contrib = peak * torch.exp(-0.5 * qf) * peak_scale
            elif mode == 'capped_peak':
                denom = (2.0 * math.pi) * sig_w[:, None, None] * sig_h[:, None, None]
                phi = torch.exp(-0.5 * qf) / denom
                contrib = torch.minimum(phi, torch.tensor(peak, device=device))
            else:
                raise ValueError("mode must be 'normalized'|'fixed_peak'|'capped_peak'")
            
            # 如果提供了scores，则乘以score
            if scores is not None:
                contrib = contrib * score_batch.item()

            if reduce in ('mean', 'sum'):
                # block = contrib.sum(dim=0, keepdim=True)      # (1, H, W)
                # accum += block
                accum[:, y0:y1, x0:x1] += contrib
            else:  # 'max'
                # accum = torch.maximum(accum, contrib.max(dim=0, keepdim=True).values)
                accum[y0:y1, x0:x1] = torch.maximum(accum[y0:y1, x0:x1], contrib)

        if reduce == 'mean':
            out = accum / N
        else:
            out = accum

        return torch.clamp(out.squeeze(0), 0, 1)

    @staticmethod
    def sigmas_from_box(
        w: torch.Tensor, h: torch.Tensor,
        boundary_size: float = 4.0,     # s=8 时峰值=0.5 → σ* = sqrt(1/pi)
        gamma: float = 1.0,             # σ(s) = σ* · (s/boundary_size)^gamma
        anisotropy: str = "proportional",
        eps: float = 1e-6,
        lambda_: float = 0.6,
        s_ref: float = 64.0
    ):
        """
        用 s = sqrt(w*h) 作为尺寸度量；s=8 时 σ*=sqrt(1/pi) 使中心峰值=0.5。
        s>8 → σ↑（更宽，峰值自然<0.5）；s<8 → σ↓（更尖，可配合 'capped_peak' 限顶）。
        anisotropy='proportional' 让 σ_w:σ_h ≈ w:h；= 'isotropic' 时 σ_w=σ_h。
        """
        sigma_star = math.sqrt(1.0 / math.pi)
        s = torch.sqrt(torch.clamp(w, min=eps) * torch.clamp(h, min=eps))

        s_eff = s + lambda_ * s_ref * torch.exp(-s / s_ref)

        sigma_scalar = sigma_star * torch.clamp(s_eff / boundary_size, min=eps).pow(gamma)
        if anisotropy == "proportional":
            sigma_w = sigma_scalar * (w / torch.clamp(s, min=eps))
            sigma_h = sigma_scalar * (h / torch.clamp(s, min=eps))
        elif anisotropy == "isotropic":
            sigma_w = sigma_scalar
            sigma_h = sigma_scalar
        else:
            raise ValueError("anisotropy must be 'proportional' or 'isotropic'")
        return sigma_w.clamp_min(eps), sigma_h.clamp_min(eps)

    @staticmethod
    def peak_scale_from_box(w: torch.Tensor, h: torch.Tensor, s_peak: float = 16.0, alpha: float = 0.5, eps: float = 1e-6, w_min: float = 0.5):
        '''
        根据框的尺寸计算峰值缩放因子
        
        Args:
            w: 框的宽度
            h: 框的高度
            s_peak: 参考尺寸
            alpha: 缩放指数
            eps: 小值防止除零
            w_min: 最小缩放值
            
        Returns:
            scale: 缩放因子
        '''
        s = torch.sqrt(torch.clamp(w, min=eps) * torch.clamp(h, min=eps))
        scale = (s_peak / (s + eps)).pow(alpha)
        scale = torch.clip(scale, w_min, 1.0)
        return scale

class MemotrSpectralWeights:
    """
    专门用于计算MeMOTR光谱权重的类
    """
    
    def __init__(self, version='le135', spectral_method='mean', spectral_n_clusters=3, use_cache=True, cache_path=None, mean=None, std=None):
        """
        初始化MeMOTR光谱权重计算器
        
        Args:
            version: 角度版本，支持 'le135', 'le90', 'oc'
            spectral_method: 光谱权重计算方法，'mean' 或 'kmeans'
            spectral_n_clusters: K-means聚类数量（仅在 method='kmeans' 时使用）
        """
        self.version = version
        self.spectral_method = spectral_method
        self.spectral_n_clusters = spectral_n_clusters
        
        # 验证参数
        if spectral_method not in ['mean', 'kmeans']:
            raise ValueError(f"Unsupported spectral_method: {spectral_method}. Use 'mean' or 'kmeans'")
        
        if spectral_method == 'kmeans' and spectral_n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {spectral_n_clusters}")
        if spectral_method == 'mean' and spectral_n_clusters != 1:
            raise ValueError(f"n_clusters must be 1 when spectral_method is 'mean', got {spectral_n_clusters}")
        
        # 如果使用cache，则需要提供mean和std
        if use_cache:
            if mean is None or std is None:
                raise ValueError("mean and std must be provided when use_cache is True")
            self.mean = torch.from_numpy(np.array(mean)).to(torch.float32)
            self.std = torch.from_numpy(np.array(std)).to(torch.float32)
            
        if use_cache:
            self.use_cache = True
            cache_name = f'memotr_spectral_weights_cache_{version}_{spectral_method}_{spectral_n_clusters}.npy'
            cache_index = cache_name.replace("npy", "json")
            self.cache_path = os.path.join(cache_path, cache_name)
            self.cache_index_path = os.path.join(cache_path, cache_index)
            if os.path.exists(self.cache_path):
                self.cache = np.load(self.cache_path, allow_pickle=True)
                self.cache_index = json.load(open(self.cache_index_path))
                self.cluster_num = self.cache_index["meta"]["cluster_num"]
                print(f"MemotrSpectralWeights 加载缓存成功: {self.cache_path}")
            else:
                raise FileNotFoundError(f"MemotrSpectralWeights 缓存文件不存在: {self.cache_path}")
        else:
            self.use_cache = False
    
    @staticmethod
    def create_cache_hsmot(version, spectral_method, spectral_n_clusters, cache_path, dataset_path, num_workers=4):
        """
        为HSMOT数据集建立光谱权重缓存（并行版本）
        
        Args:
            version: 角度版本，支持 'le135', 'le90', 'oc'
            spectral_method: 光谱权重计算方法，'mean' 或 'kmeans'
            spectral_n_clusters: K-means聚类数量（仅在 method='kmeans' 时使用）
            cache_path: 缓存保存路径
            dataset_path: 数据集路径，包含mot和npy子文件夹
            num_workers: 并行工作线程数
        
        Returns:
            cache: 建立的缓存字典
        """

        
        cache_name = f'memotr_spectral_weights_cache_{version}_{spectral_method}_{spectral_n_clusters}.pkl'
        mot_path = Path(dataset_path) / 'mot'
        npy_path = Path(dataset_path) / 'npy'
        cache_path = Path(cache_path)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # 获取所有视频序列
        mot_files = list(mot_path.glob('*.txt'))
        print(f"找到 {len(mot_files)} 个视频序列标注文件，使用 {num_workers} 个并行线程")
        
        # 线程安全的缓存字典
        cache_lock = threading.Lock()
        cache = {}
        
        # 线程安全的统计信息
        stats_lock = threading.Lock()
        total_frames = 0
        total_targets = 0
        processed_videos = 0
        
        def process_single_video(mot_file, thread_position):
            """处理单个视频序列的函数"""
            vid_name = mot_file.stem
            local_cache = {}
            local_frames = 0
            local_targets = 0
            
            
            # 检查对应的npy文件夹是否存在
            npy_vid_path = npy_path / vid_name
            if not npy_vid_path.exists():
                print(f"警告: 视频 {vid_name} 的npy文件夹不存在，跳过")
                return local_cache, local_frames, local_targets
            
            # 读取标注文件
            annotations = {}
            try:
                with open(mot_file, 'r') as f:
                    for line in f:
                        line = line.strip().split(',')
                        if len(line) >= 12:
                            frame_id = int(line[0])
                            track_id = int(line[1])
                            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, line[2:10])
                            label = int(line[11])
                            
                            if frame_id not in annotations:
                                annotations[frame_id] = []
                         
                            bbox = poly2obb(torch.tensor([[x1, y1, x2, y2, x3, y3, x4, y4]]), version)
                            
                            annotations[frame_id].append({
                                'track_id': track_id,
                                'bbox': bbox,
                                'label': label
                            })
            except Exception as e:
                print(f"警告: 读取视频 {vid_name} 的标注文件失败: {e}")
                return local_cache, local_frames, local_targets
            
            # 处理每一帧，使用进度条显示
            frame_ids = sorted(annotations.keys())
            with tqdm(frame_ids, 
                     desc=f"线程{thread_position}-{vid_name}", 
                     leave=False, 
                     position=thread_position,
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}') as pbar:
                for frame_id in pbar:
                    # 构建npy文件名
                    npy_filename = f"{frame_id:06d}.npy"
                    npy_file_path = npy_vid_path / npy_filename
                    
                    if not npy_file_path.exists():
                        print(f"警告: 帧文件 {npy_file_path} 不存在，跳过")
                        continue
   
                    # # 加载图像数据
                    try:
                        img_data = np.load(npy_file_path)
                    except Exception as e:
                        print(f"警告: 无法加载帧文件 {npy_file_path}: {e}")
                        continue
                    
                    # 处理当前帧的所有目标
                    for target_info in annotations[frame_id]:
                        track_id = target_info['track_id']
                        bbox = target_info['bbox']
                        label = target_info['label']
                        
                        # 构建缓存键
                        cache_key = f"{vid_name}_{frame_id:06d}_{track_id:06d}"
                        
                        try:
                            # 计算光谱权重
                            spectral_weights = MemotrSpectralWeights.spectral_weights_from_gt_xywha(
                                bbox, img_data, 
                                version=version,
                                method=spectral_method,
                                n_clusters=spectral_n_clusters
                            )
                            
                            # 存储到本地缓存
                            local_cache[cache_key] = {
                                'spectral_weights': spectral_weights,
                                'bbox': bbox,
                                'label': label,
                                'vid_name': vid_name,
                                'frame_id': frame_id,
                                'track_id': track_id
                            }
                            
                            local_targets += 1
                            
                        except Exception as e:
                            print(f"警告: 计算目标 {cache_key} 的光谱权重失败: {e}")
                            continue
                    
                    local_frames += 1
                    
                    # 更新进度条描述，显示当前处理的帧数和目标数
                    pbar.set_postfix({
                        'frames': local_frames,
                        'targets': local_targets,
                        'rate': f"{local_frames}/{len(frame_ids)}"
                    })
            
            tqdm.write(f"线程{thread_position}完成视频 {vid_name}: 处理了 {local_frames} 帧，{local_targets} 个目标")
            return local_cache, local_frames, local_targets
        


        start_time = time.time()

        task_q = Queue()
        for mf in mot_files:
            task_q.put(mf)

        main_pbar = tqdm(total=len(mot_files), desc="总体进度", position=0, leave=True)

        def worker_loop(slot_idx):
            nonlocal total_frames, total_targets, processed_videos
            while True:
                try:
                    mot_file = task_q.get_nowait()
                except Exception:
                    break

                local_cache, local_frames, local_targets = process_single_video(mot_file, thread_position=slot_idx)

                with cache_lock:
                    cache.update(local_cache)
                with stats_lock:
                    total_frames += local_frames
                    total_targets += local_targets
                    processed_videos += 1

                # 更新总进度（主进度条安全可并发更新；此处简单直接）
                main_pbar.update(1)
                main_pbar.set_postfix({
                    'videos': f"{processed_videos}/{len(mot_files)}",
                    'frames': total_frames,
                    'targets': total_targets
                })

                task_q.task_done()

        # 启动固定数量的 workers：每个 worker 绑定 position = 1..num_workers
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_loop, slot) for slot in range(1, num_workers + 1)]
            for f in futures:
                f.result()

        main_pbar.close()

        # # 记录开始时间
        # start_time = time.time()
        
        # # 使用线程池并行处理视频
        # with ThreadPoolExecutor(max_workers=num_workers) as executor:
        #     # 提交所有任务
        #     future_to_file = {executor.submit(process_single_video, mot_file): mot_file for mot_file in mot_files}
            
        #     # 使用tqdm显示总体进度
        #     with tqdm(total=len(mot_files), desc="总体进度", position=0, leave=True) as main_pbar:
        #         for future in as_completed(future_to_file):
        #             mot_file = future_to_file[future]
        #             try:
        #                 local_cache, local_frames, local_targets = future.result()
                        
        #                 # 线程安全地更新全局缓存和统计信息
        #                 with cache_lock:
        #                     cache.update(local_cache)
                        
        #                 with stats_lock:
        #                     total_frames += local_frames
        #                     total_targets += local_targets
        #                     processed_videos += 1
                        
        #                 # 更新主进度条，显示详细信息
        #                 main_pbar.set_postfix({
        #                     'videos': f"{processed_videos}/{len(mot_files)}",
        #                     'frames': total_frames,
        #                     'targets': total_targets
        #                 })
        #                 main_pbar.update(1)
                        
        #             except Exception as e:
        #                 print(f"警告: 处理视频 {mot_file.stem} 时发生异常: {e}")
        #                 main_pbar.update(1)
        
        # 计算总耗时
        total_time = time.time() - start_time
        
        # 添加元数据
        cache['meta'] = {
            'creation_time': datetime.now().isoformat(),
            'version': version,
            'spectral_method': spectral_method,
            'spectral_n_clusters': spectral_n_clusters,
            'dataset_path': str(dataset_path),
            'mot_path': str(mot_path),
            'npy_path': str(npy_path),
            'num_workers': num_workers,
            'processed_videos': processed_videos,
            'total_frames': total_frames,
            'total_targets': total_targets,
            'total_time_seconds': total_time,
            'cache_size': len(cache) - 1,  # 减去meta键
            'cache_file': str(cache_path / cache_name)
        }
        
        # 保存缓存
        cache_file_path = cache_path / cache_name
        with open(cache_file_path, 'wb') as f:
            pickle.dump(cache, f)
        
        print(f"\n缓存建立完成!")
        print(f"缓存文件: {cache_file_path}")
        print(f"并行线程数: {num_workers}")
        print(f"处理视频数: {processed_videos}")
        print(f"总帧数: {total_frames}")
        print(f"总目标数: {total_targets}")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"缓存大小: {len(cache) - 1} 个条目")
        
        return cache

    def get_spectral_weights(self, bboxes, img, obj_ids, meta): 
        if self.use_cache:
            results = []
            for ids in obj_ids:
                real_id = ids % 100000
                cache_head = meta.data['ori_filename'].split('.')[0].split('/')[-2:]
                cache_key = f'{cache_head[0]}_{cache_head[1]}_{real_id:06d}'
                cache_index = self.cache_index[cache_key]
                results.append(self.cache[cache_index])
            
            # 如果results为空，则返回全0
            if len(results) == 0:
                #TODO
                return torch.zeros((0, self.cluster_num, 8))
            
            # numpy 2 tensor
            results = torch.from_numpy(np.concatenate(results, axis=0))# (N, k, 8)

            #使用mean和std归一化(1, 8)
            _results = results.view(-1, results.shape[-1])# (N*k, 8)
            _results = (_results - self.mean) / self.std
            results = _results.view(results.shape[0], -1, results.shape[-1])# (N, k, 8)

            return results
        else:
            return self.compute_spectral_weights(bboxes, img)

    def compute_spectral_weights(self, bboxes, img):
        """
        计算单个图像的光谱分布权重
        
        Args:
            bboxes: 边界框坐标，形状为 [N, 5] (x, y, w, h, a)
            img: 输入图像，形状为 [C, H, W] 或 [H, W, C]
        
        Returns:
            spectral_weights: 光谱权重，形状为 [N, 1, C] 或 [N, n_clusters, C]
        """
        return MemotrSpectralWeights.spectral_weights_from_gt_xywha(
            bboxes, img, 
            version=self.version,
            method=self.spectral_method,
            n_clusters=self.spectral_n_clusters
        )
    
    def compute_spectral_weights_batch(self, results_list):
        """
        计算批量结果的光谱分布权重
        
        Args:
            results_list: 结果列表，每个元素包含 'img', 'gt_bboxes' 等字段
        
        Returns:
            spectral_weights_list: 每个结果对应的光谱权重列表
        """
        spectral_weights_list = []
        
        for results in results_list:
            # 计算光谱分布权重
            spectral_weights = self.compute_spectral_weights(
                results['gt_bboxes'].data.cpu().detach(), 
                results['img'].data.cpu().numpy()
            )
            spectral_weights_list.append(spectral_weights)
        
        return spectral_weights_list
    
    def get_method_info(self):
        """
        获取当前方法信息
        
        Returns:
            dict: 包含方法信息的字典
        """
        return {
            'version': self.version,
            'spectral_method': self.spectral_method,
            'n_clusters': self.spectral_n_clusters if self.spectral_method == 'kmeans' else None
        }


    @staticmethod
    def spectral_weights_from_gt_xywha(gt_bboxes, img, version='le135', method='mean', n_clusters=3):
        '''
        gt_bboxes: [N, 5] (x, y, w, h, a)
        img: [C, H, W] or [H, W, C]
        method: 'mean' 或 'kmeans'，指定计算光谱权重的方法
        n_clusters: 当method='kmeans'时，指定聚类数量
        return: [N, C] 每个框的光谱分布权重
        '''
        gt_xyxyxyxy = obb2poly(gt_bboxes, version)
        spectral_weights_np = MemotrSpectralWeights.spectral_weights_from_gt(gt_bboxes, gt_xyxyxyxy, img, method=method, n_clusters=n_clusters)
        return torch.from_numpy(spectral_weights_np).float()

    @staticmethod
    def spectral_weights_from_gt(gt_xywha, gt_xyxyxyxy, img, version='le135', method='mean', n_clusters=3):
        '''
        gt_xywha: [N, 5]
        gt_xyxyxyxy: [N, 8]
        img: [C, H, W] or [H, W, C]
        method: 'mean' 或 'kmeans'，指定计算光谱权重的方法
        n_clusters: 当method='kmeans'时，指定聚类数量
        '''
        weights = []
        img_np = img.cpu().numpy() if hasattr(img, 'cpu') else img
        if img_np.shape[0] <= 8:  # [C, H, W]
            img_np = np.transpose(img_np, (1, 2, 0))
        
        #debug
        cluster_labels = np.ones((img_np.shape[0], img_np.shape[1])) * -1

        for xyxyxyxy in gt_xyxyxyxy:
            pts = np.array(xyxyxyxy).reshape(4, 2).astype(np.int32)
            mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 1)
            # 提取框内像素
            pixels = img_np[mask == 1]
            
            if len(pixels) == 0:
                weights.append(np.zeros(img_np.shape[2]))
            else:
                if method == 'mean':
                    # 计算平均光谱分布
                    weights.append(pixels.mean(axis=0)[None, :])
                elif method == 'kmeans':
                    # 使用K-means聚类计算光谱分布
                    cluster_weights, sub_cluster_labels = MemotrSpectralWeights._compute_kmeans_spectral_weights(pixels, n_clusters)
                    cluster_labels[mask==1] = sub_cluster_labels
                    weights.append(cluster_weights)
                else:
                    raise ValueError(f"Unsupported method: {method}. Use 'mean' or 'kmeans'.")
        
        if hasattr(MemotrSpectralWeights, '_debug_mode') and MemotrSpectralWeights._debug_mode:
            MemotrSpectralWeights._draw_cluster_labels(cluster_labels, img_np, MemotrSpectralWeights._debug_save_path)

        if len(weights) == 0:
            return np.zeros((0, 1, img_np.shape[2]))#TODO cluster
        else:
            return np.stack(weights)# (N, 1 or n_clusters, C)

    @staticmethod
    def _compute_kmeans_spectral_weights(pixels, n_clusters):
        '''
        使用K-means聚类计算光谱权重
        pixels: [N, C] 像素光谱值
        n_clusters: 聚类数量
        return: [C] 聚类后的光谱权重
        '''

        assert len(pixels) >= n_clusters, f'pixels.shape:{pixels.shape}, n_clusters:{n_clusters}'
        
        # 执行K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(pixels)
        
        # 计算每个聚类的中心点
        cluster_centers = kmeans.cluster_centers_  # [n_clusters, C]
        
        # 计算每个聚类的像素数量
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        
        # 按像素数量排序，取前n_clusters个聚类
        sorted_indices = np.argsort(counts)[::-1]  # 降序排列
        top_clusters = sorted_indices[:n_clusters]
        
        # 计算加权平均（权重为聚类大小）
        total_pixels = len(pixels)
        
        # 按顺序返回聚类中心点
        return cluster_centers[top_clusters], cluster_labels

    @staticmethod
    def _draw_cluster_labels(cluster_labels, img_np, save_path=None):
        '''
        绘制聚类标签叠加图
        cluster_labels: [H, W]
        img_np: [H, W, C]
        save_path: 保存路径，如果为None则不保存
        '''
        if img_np.shape[0] <= 8:  # [C, H, W]
            img = np.transpose(img_np, (1, 2, 0))
        else:
            img = img_np

        if img.shape[2] == 8:  # 8通道图像，取RGB通道
            img = img[:, :, [4,2,1]]

        color_map = np.array([
            [255, 0, 0],   # 红色 - 类别0
            [0, 0, 255],   # 蓝色 - 类别1  
            [0, 255, 0],   # 绿色 - 类别2
        ])

        color_mask = np.zeros_like(img)
        for k in range(3):
            color_mask[cluster_labels == k] = color_map[k]

        # 叠加显示
        alpha = 0.5  # 透明度
        overlay = img.copy()
        overlay = (alpha * color_mask + (1-alpha) * img).astype(np.uint8)
        
        if save_path is not None:
            import os
            os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(os.path.join(save_path, 'ori_img.png'), img)
            cv2.imwrite(os.path.join(save_path, 'cluster_labels.png'), overlay)
            cv2.imwrite(os.path.join(save_path, 'cluster_mask.png'), color_mask)
        
        return overlay, color_mask

class MemotrBboxProcessor:
    """
    专门用于处理MeMOTR边界框的类
    """
    
    def __init__(self, version='le135'):
        """
        初始化边界框处理器
        
        Args:
            version: 角度版本，支持 'le135', 'le90', 'oc'
        """
        self.version = version
    
    def compute_normalized_bboxes(self, bboxes, img_shape):
        """
        计算归一化的边界框坐标
        
        Args:
            bboxes: 边界框坐标，形状为 [N, 5] (x, y, w, h, a)
            img_shape: 图像形状 (h, w)
        
        Returns:
            norm_bboxes: 归一化的边界框坐标
        """
        h, w = img_shape
        
        if self.version == 'oc':
            raise NotImplementedError(f"Version 'oc' is not supported yet")
        elif self.version == 'le135':
            angle_range = 1
            angle_offset = -1/4
        elif self.version == 'le90':
            angle_range = 1
            angle_offset = -1/2
        else:
            raise ValueError(f"Unsupported version: {self.version}")
        
        angle_range *= math.pi
        angle_offset *= math.pi
        
        norm_bboxes = (bboxes - torch.as_tensor([0,0,0,0,angle_offset], dtype=bboxes.dtype, device=bboxes.device)) / torch.as_tensor([w, h, w, h, angle_range], dtype=bboxes.dtype, device=bboxes.device)
        
        return norm_bboxes
    
    def get_version_info(self):
        """
        获取版本信息
        
        Returns:
            dict: 包含版本信息的字典
        """
        return {
            'version': self.version,
            'version_index': version_str_to_index(self.version)
        }

class MmrotateToMemotr:
    def __init__(self, version='le135', spectral_method='mean', spectral_n_clusters=3, use_cache=False, cache_path=None, mean=None, std=None, get_spectral_weights=True):
        self.version = version
        self.spectral_method = spectral_method
        self.spectral_n_clusters = spectral_n_clusters
        self.get_spectral_weights = get_spectral_weights
        
        if self.get_spectral_weights:
            # 初始化专门的处理器
            self.spectral_processor = MemotrSpectralWeights(
                version=version,
                spectral_method=spectral_method,
                spectral_n_clusters=spectral_n_clusters,
                cache_path=cache_path,
                use_cache=use_cache,
                mean=mean,
                std=std
            )
        self.bbox_processor = MemotrBboxProcessor(version=version)

    def __call__(self, results_list):
        images = []
        targets = []
        img_metas = []

        for results in results_list:
            images.append(results['img'].data)
            (h, w), (pad_h, pad_w) = _extract_effective_and_padded_shape(results)
            img_metas.append({
                'img_shape': torch.as_tensor((h, w), device=results['img'].data.device),
                'pad_shape': torch.as_tensor((pad_h, pad_w), device=results['img'].data.device),
                'version': torch.as_tensor(version_str_to_index(self.version), dtype=torch.int, device=results['img'].data.device),
                'transform_metas': results['img_metas']
            })
            
            gt_bboxes = results['gt_bboxes'].data
            
            # 使用专门的边界框处理器计算归一化边界框
            norm_gt_bboxes = self.bbox_processor.compute_normalized_bboxes(gt_bboxes, (h, w))
            
            # 使用专门的光谱权重处理器计算光谱分布权重
            if self.get_spectral_weights:
                spectral_weights = self.spectral_processor.get_spectral_weights(
                    gt_bboxes.cpu().detach(), 
                    results['img'].data.cpu().numpy(),
                    obj_ids=results['gt_trackids'].data.cpu().detach(),
                    meta=results['img_metas']
                )
            else:
                spectral_weights = None
            
            # 构造heatmap
            heatmap = HeatmapFromRotateGt.heatmap_from_rotate_gt_xywha(gt_bboxes, (pad_h, pad_w), self.version)

            targets.append({
                'boxes': results['gt_bboxes'].data,
                'norm_boxes': norm_gt_bboxes,
                'labels': results['gt_labels'].data,
                'obj_ids': results['gt_trackids'].data,
                'spectral_weights': spectral_weights,
                'heatmap': heatmap,
            })

        return images, targets, img_metas
    
    def get_processor_info(self):
        """
        获取处理器信息
        
        Returns:
            dict: 包含处理器信息的字典
        """
        return {
            'spectral_processor': self.spectral_processor.get_method_info(),
            'bbox_processor': self.bbox_processor.get_version_info()
        }


class RectMotToMemotr:
    """水平正框 pipeline 输出 -> MeMOTR 训练格式。"""

    def __init__(self, get_spectral_weights: bool = False):
        self.get_spectral_weights = get_spectral_weights

    def __call__(self, results_list):
        from hsmot.mmlab.hs_rectmot import normalize_cxcywh, xyxy_to_cxcywh

        images = []
        targets = []
        img_metas = []

        for results in results_list:
            img_tensor = results['img'].data
            images.append(img_tensor)

            meta = results.get('img_metas')
            if meta is not None and hasattr(meta, 'data'):
                meta = meta.data

            (h, w), (pad_h, pad_w) = _extract_effective_and_padded_shape(results)

            gt_xyxy = results['gt_bboxes'].data.float()
            if gt_xyxy.numel() > 0:
                gt_cxcywh = xyxy_to_cxcywh(gt_xyxy)
                norm_boxes = normalize_cxcywh(gt_cxcywh, (h, w))
                gt_xywha = torch.cat(
                    [gt_cxcywh, gt_xyxy.new_zeros((gt_cxcywh.size(0), 1))], dim=-1
                )
                heatmap = HeatmapFromRotateGt.heatmap_from_rotate_gt_xywha(
                    gt_xywha, (pad_h, pad_w), 'le135'
                )
            else:
                norm_boxes = gt_xyxy.reshape(0, 4)
                heatmap = HeatmapFromRotateGt.heatmap_from_rotate_gt_xywha(
                    gt_xyxy.reshape(0, 5), (pad_h, pad_w), 'le135'
                )

            img_metas.append({
                'img_shape': torch.as_tensor((h, w), device=img_tensor.device),
                'pad_shape': torch.as_tensor((pad_h, pad_w), device=img_tensor.device),
                'transform_metas': meta if isinstance(meta, dict) else results.get('img_metas'),
            })
            targets.append({
                'boxes': gt_xyxy,
                'norm_boxes': norm_boxes,
                'labels': results['gt_labels'].data,
                'obj_ids': results['gt_trackids'].data,
                'spectral_weights': None,
                'heatmap': heatmap,
            })

        return images, targets, img_metas


#####  test   ####
def load_test_data(npy=False):
    """加载测试数据"""
    from pathlib import Path
    import cv2
    
    cwd = Path(__file__).absolute().parent
    img_path = cwd / '../../../..' / 'data/hsmot/rgb/data52-7/000003.png' if not npy else cwd / '../../../..' / 'data/hsmot/npy/data52-7/000003.npy'
    mot_path = cwd / '../../../..' / 'data/hsmot/mot/data52-7.txt'
    # img_path = cwd / '../../../..' / 'data/hsmot/rgb/data36-7/000009.png' if not npy else cwd / '../../../..' / 'data/hsmot/npy/data36-7/000009.npy'
    # mot_path = cwd / '../../../..' / 'data/hsmot/mot/data36-7.txt'

    img = cv2.imread(str(img_path)) if not npy else np.load(str(img_path))
    xyxyxyxy = []
    
    frame = int(img_path.stem.split('/')[-1].split('.')[0])

    with open(mot_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split(',')
        frame_id, obj_id, x1, y1, x2, y2, x3, y3, x4, y4, _, cls, _ = line
        if int(frame_id) == frame:
            xyxyxyxy.append([int(x1), int(y1), int(x2), int(y2), int(x3), int(y3), int(x4), int(y4)])
    
    xyxyxyxy = np.array(xyxyxyxy)
    xyxyxyxy = torch.from_numpy(xyxyxyxy).float()
    xywha = poly2obb(xyxyxyxy, 'le135')
    
    return img, xywha, xyxyxyxy

def test_mean_spectral_weights():
    """测试平均光谱权重计算"""
    print("=== 测试平均光谱权重计算 ===")
    
    img, xywha, _ = load_test_data()
    
    mean_weights = MemotrSpectralWeights.spectral_weights_from_gt_xywha(
        xywha, img, version='le135', method='mean'
    )
    print(f"平均光谱权重形状: {mean_weights.shape}")
    # print(f"平均光谱权重: {mean_weights}")
    
    return mean_weights

def test_kmeans_spectral_weights():
    """测试K-means聚类光谱权重计算"""
    print("=== 测试K-means聚类光谱权重计算 ===")
    
    img, xywha, _ = load_test_data(npy=True)
    
    try:
        kmeans_weights = MemotrSpectralWeights.spectral_weights_from_gt_xywha(
            xywha, img, version='le135', method='kmeans', n_clusters=3
        )
        print(f"K-means光谱权重形状: {kmeans_weights.shape}")
        # print(f"K-means光谱权重: {kmeans_weights}")
        return kmeans_weights
    except ImportError as e:
        print(f"K-means测试跳过: {e}")
        return None

def test_cluster_visualization():
    """测试聚类可视化功能"""
    print("=== 测试聚类可视化功能 ===")
    
    img, xywha, _ = load_test_data(npy=True)
    
    # 启用调试模式
    MemotrSpectralWeights._debug_mode = True
    MemotrSpectralWeights._debug_save_path = '/data/users/litianhao/hsmot_code/workdir_2/memotr/debug'
    
    try:
        kmeans_weights = MemotrSpectralWeights.spectral_weights_from_gt_xywha(
            xywha, img, version='le135', method='kmeans', n_clusters=3
        )
        print("聚类可视化图像已保存到: /data/users/litianhao/hsmot_code/workdir_2/memotr/debug/")
        return kmeans_weights
    except Exception as e:
        print(f"聚类可视化测试失败: {e}")
        return None
    finally:
        # 关闭调试模式
        MemotrSpectralWeights._debug_mode = False

def test_performance():
    """测试性能"""
    print("=== 测试性能 ===")
    
    img, xywha, _ = load_test_data()
    
    import time
    
    # 测试平均方法性能
    time_start = time.time()
    for i in range(10):
        mean_weights = MemotrSpectralWeights.spectral_weights_from_gt_xywha(
            xywha, img, version='le135', method='mean'
        )
    time_end = time.time()
    print(f"平均方法耗时: {time_end - time_start:.4f}秒")
    
    # 测试K-means方法性能
    try:
        time_start = time.time()
        for i in range(10):
            kmeans_weights = MemotrSpectralWeights.spectral_weights_from_gt_xywha(
                xywha, img, version='le135', method='kmeans', n_clusters=3
            )
        time_end = time.time()
        print(f"K-means方法耗时: {time_end - time_start:.4f}秒")
    except ImportError as e:
        print(f"K-means性能测试跳过: {e}")

def test_heatmap_generation():
    """测试热力图生成功能"""
    print("=== 测试热力图生成功能 ===")
    
    img, xywha, xyxyxyxy = load_test_data()

    heatmap = HeatmapFromRotateGt._heatmap_from_rotate_gt_xywha_fast(xywha, img.shape[:2], 'le135', mode='fixed_peak', peak=1, reduce='sum')

    # 保存热力图
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, cmap='plasma', vmin=0, vmax=1)
    plt.colorbar()
    plt.title('热力图')
    plt.savefig('/data/users/litianhao/hsmot_code/workdir_2/memotr/debug/heatmap_fast.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("热力图已保存到: /data/users/litianhao/hsmot_code/workdir_2/memotr/debug/heatmap_fast.png")
    
    # 将框用红色画在热力图上，保存下来
    # 将热力图转换为numpy数组
    if isinstance(heatmap, torch.Tensor):
        heatmap_np = heatmap.cpu().numpy()
    else:
        heatmap_np = heatmap
    
    # 归一化到0-255范围
    if heatmap_np.max() > 0:
        heatmap_np = (heatmap_np / heatmap_np.max() * 255).astype(np.uint8)
    else:
        heatmap_np = heatmap_np.astype(np.uint8)
    
    # 转换为3通道图像（用于绘制彩色框）
    if len(heatmap_np.shape) == 2:
        heatmap_img = cv2.cvtColor(heatmap_np, cv2.COLOR_GRAY2BGR)
    else:
        heatmap_img = heatmap_np.copy()
    
    # 将xyxyxyxy转换为numpy数组
    if isinstance(xyxyxyxy, torch.Tensor):
        boxes = xyxyxyxy.cpu().numpy()
    else:
        boxes = xyxyxyxy
    
    # 绘制每个旋转框（红色）
    red_color = (0, 0, 255)  # BGR格式的红色
    thickness = 2
    
    for box in boxes:
        x1, y1, x2, y2, x3, y3, x4, y4 = box.astype(np.int32)
        # 绘制旋转框的4条边
        pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)
        cv2.line(heatmap_img, tuple(pts[0]), tuple(pts[1]), red_color, thickness)
        cv2.line(heatmap_img, tuple(pts[1]), tuple(pts[2]), red_color, thickness)
        cv2.line(heatmap_img, tuple(pts[2]), tuple(pts[3]), red_color, thickness)
        cv2.line(heatmap_img, tuple(pts[3]), tuple(pts[0]), red_color, thickness)
    
    # 保存带框的热力图
    cv2.imwrite('/data/users/litianhao/hsmot_code/workdir_2/memotr/debug/heatmap_fast_with_boxes.png', heatmap_img)
    print("带框的热力图已保存到: /data/users/litianhao/hsmot_code/workdir_2/memotr/debug/heatmap_fast_with_boxes.png")

    # 保存一下原图
    cv2.imwrite('/data/users/litianhao/hsmot_code/workdir_2/memotr/debug/img.png', img)
    print("原图已保存到: /data/users/litianhao/hsmot_code/workdir_2/memotr/debug/img.png")

    # 打印xywha信息
    print(f"xywha信息: {xywha}")
    # 数量
    print(f"xywha数量: {xywha.shape[0]}")

    # 打印heatmap的总和
    print(f"heatmap的总和: {heatmap.sum().item()}")
    # 打印heatmap的平均值
    print(f"heatmap的平均值: {heatmap.mean().item()}")
    # 打印heatmap的最大值
    print(f"heatmap的最大值: {heatmap.max().item()}")
    # 打印heatmap的最小值
    print(f"heatmap的最小值: {heatmap.min().item()}")
    # 打印heatmap的标准差
    print(f"heatmap的标准差: {heatmap.std().item()}")
    # 统计大于0的数量
    print(f"heatmap大于0的数量: {torch.sum(heatmap > 0).item()}")

    return heatmap

def test_heatmap_performance():
    """测试热力图生成性能"""
    print("=== 测试热力图生成性能 ===")
    
    img, xywha, xyxyxyxy = load_test_data()
    
    import time
    
    time_start = time.time()
    for i in range(10):
        heatmap = HeatmapFromRotateGt.heatmap_from_rotate_gt_xyxyxyxy(xyxyxyxy, img.shape[:2], 'le135')
    time_end = time.time()
    print(f"热力图生成100次耗时: {time_end - time_start:.4f}秒")

def test_heatmap_contrib_sum_by_scale():
    """测试不同尺度下contrib的和"""
    print("=== 测试不同尺度下contrib的和 ===")
    
    import csv
    from pathlib import Path
    
    # 固定图像大小和中心位置
    img_shape = (512, 512)
    center_x, center_y = 256.0, 256.0
    theta = 0.0  # 角度设为0，便于分析
    
    # 尺度范围
    w_range = range(4, 101)  # 4到100
    h_range = range(4, 101)  # 4到100
    
    # 存储结果
    results = []
    
    print(f"开始测试，共 {len(w_range) * len(h_range)} 个组合...")
    
    for w in w_range:
        for h in h_range:
            # 创建单个bbox (xc, yc, w, h, theta)
            bbox = torch.tensor([[center_x, center_y, float(w), float(h), theta]], dtype=torch.float32)
            
            # 计算热力图（使用fixed_peak模式，reduce='sum'）
            heatmap = HeatmapFromRotateGt._heatmap_from_rotate_gt_xywha_fast(
                bbox, 
                img_shape, 
                'le135', 
                mode='fixed_peak', 
                peak=1.0, 
                reduce='sum'
            )
            
            # contrib的和等于热力图的总和（因为reduce='sum'且只有一个框）
            contrib_sum = heatmap.sum().item()
            
            results.append({
                'w': w,
                'h': h,
                'contrib_sum': contrib_sum
            })
            
            # 每100个组合打印一次进度
            if len(results) % 100 == 0:
                print(f"已完成 {len(results)} / {len(w_range) * len(h_range)} 个组合...")
    
    # 将结果组织成二维表格：w为行，h为列
    # 构建二维字典：{w: {h: contrib_sum}}
    table = {}
    for r in results:
        w = r['w']
        h = r['h']
        contrib_sum = r['contrib_sum']
        if w not in table:
            table[w] = {}
        table[w][h] = contrib_sum
    
    # 保存为CSV（二维表格格式）
    output_path = Path('/data/users/litianhao/hsmot_code/workdir_2/memotr/debug/heatmap_contrib_sum_by_scale.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        # 第一行：'w' + 所有h值
        h_list = sorted(h_range)
        writer = csv.writer(f)
        writer.writerow(['w'] + [str(h) for h in h_list])
        
        # 后续行：w值 + 对应的contrib_sum值
        for w in sorted(w_range):
            row = [str(w)]
            for h in h_list:
                row.append(f"{table[w][h]:.6f}")
            writer.writerow(row)
    
    print(f"结果已保存到: {output_path}")
    print(f"表格大小: {len(w_range)} 行 x {len(h_range)} 列")
    
    # 打印一些统计信息
    contrib_sums = [r['contrib_sum'] for r in results]
    print(f"contrib_sum 最小值: {min(contrib_sums):.6f}")
    print(f"contrib_sum 最大值: {max(contrib_sums):.6f}")
    print(f"contrib_sum 平均值: {sum(contrib_sums) / len(contrib_sums):.6f}")
    
    return results

def test_mmrotate_converter():
    """测试MmrotateToMemotr转换器"""
    print("=== 测试MmrotateToMemotr转换器 ===")
    
    img, xywha, _ = load_test_data()
    
    # 创建转换器
    converter = MmrotateToMemotr(
        version='le135',
        spectral_method='kmeans',
        spectral_n_clusters=3
    )
    
    # 模拟results_list格式
    results_list = [{
        'img': type('obj', (object,), {'data': torch.from_numpy(img.transpose(2, 0, 1))})(),
        'gt_bboxes': type('obj', (object,), {'data': xywha.float()})(),
        'gt_labels': type('obj', (object,), {'data': torch.zeros(len(xywha), dtype=torch.long)})(),
        'gt_trackids': type('obj', (object,), {'data': torch.arange(len(xywha), dtype=torch.long)})(),
        'img_metas': {}
    }]
    
    try:
        images, targets, img_metas = converter(results_list)
        print(f"转换后的目标数量: {len(targets)}")
        if len(targets) > 0:
            print(f"光谱权重形状: {targets[0]['spectral_weights'].shape}")
        
        # 测试处理器信息
        processor_info = converter.get_processor_info()
        print(f"处理器信息: {processor_info}")
        
        return images, targets, img_metas
    except Exception as e:
        print(f"转换器测试失败: {e}")
        return None, None, None

def test_memotr_spectral_weights():
    """测试MemotrSpectralWeights类"""
    print("=== 测试MemotrSpectralWeights类 ===")
    
    img, xywha, _ = load_test_data(npy=True)
    
    # 测试平均方法
    print("测试平均方法...")
    mean_processor = MemotrSpectralWeights(
        version='le135',
        spectral_method='mean'
    )
    
    mean_weights = mean_processor.compute_spectral_weights(xywha, img)
    print(f"平均方法权重形状: {mean_weights.shape}")
    print(f"平均方法信息: {mean_processor.get_method_info()}")
    
    # 测试K-means方法
    print("测试K-means方法...")
    try:
        kmeans_processor = MemotrSpectralWeights(
            version='le135',
            spectral_method='kmeans',
            spectral_n_clusters=3
        )
        
        kmeans_weights = kmeans_processor.compute_spectral_weights(xywha, img)
        print(f"K-means方法权重形状: {kmeans_weights.shape}")
        print(f"K-means方法信息: {kmeans_processor.get_method_info()}")
        
        return mean_weights, kmeans_weights
    except Exception as e:
        print(f"K-means方法测试失败: {e}")
        return mean_weights, None

def test_memotr_bbox_processor():
    """测试MemotrBboxProcessor类"""
    print("=== 测试MemotrBboxProcessor类 ===")
    
    img, xywha, _ = load_test_data()
    
    # 测试边界框处理器
    bbox_processor = MemotrBboxProcessor(version='le135')
    
    # 测试归一化边界框计算
    norm_bboxes = bbox_processor.compute_normalized_bboxes(xywha, img.shape[:2])
    print(f"原始边界框形状: {xywha.shape}")
    print(f"归一化边界框形状: {norm_bboxes.shape}")
    print(f"版本信息: {bbox_processor.get_version_info()}")
    
    return norm_bboxes

def test_cache_creation():
    """测试缓存建立功能"""
    print("=== 测试缓存建立功能 ===")
    
    # 设置测试路径（需要根据实际情况调整）
    test_dataset_path = "/data/users/litianhao/data/HSMOT"
    test_cache_path = "/data/users/litianhao/data/HSMOT"
    
    try:
        # # 测试建立缓存（串行版本）
        # print("测试串行缓存建立...")
        # cache_serial = MemotrSpectralWeights.create_cache_hsmot(
        #     version='le135',
        #     spectral_method='mean',
        #     spectral_n_clusters=3,
        #     cache_path=test_cache_path,
        #     dataset_path=test_dataset_path,
        #     num_workers=1  # 串行处理
        # )
        
        # print("串行缓存建立成功!")
        # print(f"串行缓存元数据: {cache_serial['meta']}")
        
        # 测试建立缓存（并行版本）
        print("\n测试并行缓存建立...")
        cache_parallel = MemotrSpectralWeights.create_cache_hsmot(
            version='le135',
            spectral_method='kmeans',
            spectral_n_clusters=3,
            cache_path=test_cache_path,
            dataset_path=test_dataset_path,
            num_workers=16  # 4个并行线程
        )
        
        print("并行缓存建立成功!")
        print(f"并行缓存元数据: {cache_parallel['meta']}")
        
        # # 比较性能
        # if 'total_time_seconds' in cache_serial['meta'] and 'total_time_seconds' in cache_parallel['meta']:
        #     speedup = cache_serial['meta']['total_time_seconds'] / cache_parallel['meta']['total_time_seconds']
        #     print(f"\n性能比较:")
        #     print(f"串行耗时: {cache_serial['meta']['total_time_seconds']:.2f} 秒")
        #     print(f"并行耗时: {cache_parallel['meta']['total_time_seconds']:.2f} 秒")
        #     print(f"加速比: {speedup:.2f}x")
        
        # 测试加载缓存
        cache_file_path = Path(test_cache_path) / f'memotr_spectral_weights_cache_le135_kmeans_3.pkl'
        if cache_file_path.exists():
            with open(cache_file_path, 'rb') as f:
                loaded_cache = pickle.load(f)
            print(f"\n缓存加载成功! 包含 {len(loaded_cache) - 1} 个条目")
            print(f"加载的元数据: {loaded_cache['meta']}")
        
        return cache_parallel
        
    except Exception as e:
        print(f"缓存建立测试失败: {e}")
        return None

def test_parallel_performance():
    """测试不同线程数的性能"""
    print("=== 测试并行性能 ===")
    
    # 设置测试路径
    test_dataset_path = "/data/users/litianhao/data/HSMOT"
    test_cache_path = "/data/users/litianhao/data/HSMOT"
    
    # 测试不同的线程数
    thread_counts = [1, 2, 4, 8]
    results = {}
    
    for num_workers in thread_counts:
        try:
            print(f"\n测试 {num_workers} 个线程...")
            cache = MemotrSpectralWeights.create_cache_hsmot(
                version='le135',
                spectral_method='mean',
                spectral_n_clusters=3,
                cache_path=test_cache_path,
                dataset_path=test_dataset_path,
                num_workers=num_workers
            )
            
            results[num_workers] = {
                'time': cache['meta']['total_time_seconds'],
                'processed_videos': cache['meta']['processed_videos'],
                'total_targets': cache['meta']['total_targets']
            }
            
            print(f"{num_workers} 线程完成，耗时: {results[num_workers]['time']:.2f} 秒")
            
        except Exception as e:
            print(f"{num_workers} 线程测试失败: {e}")
            results[num_workers] = None
    
    # 分析结果
    print(f"\n性能分析结果:")
    print(f"{'线程数':<8} {'耗时(秒)':<12} {'加速比':<10}")
    print("-" * 30)
    
    baseline_time = results.get(1, {}).get('time', 0)
    for num_workers in thread_counts:
        if results.get(num_workers) and results[num_workers] is not None:
            time_taken = results[num_workers]['time']
            speedup = baseline_time / time_taken if baseline_time > 0 else 0
            print(f"{num_workers:<8} {time_taken:<12.2f} {speedup:<10.2f}x")
    
    return results

def transfer_cache_to_npy(
    pkl_path,
    npy_path,
    drop_leading_one: bool = False,   # (1,k,C)->(k,C)
    force_dtype = np.float32,        # 统一精度，建议 float32
    save_index_path = None  # 可选：保存 key->row 索引（.json 或 .npy），类型：str | None
):
    """
    从 pkl 缓存提取 `spectral_weights`，直接堆叠保存为一个 .npy 文件（不使用 mmap）。
    可选同时保存 key->row 的索引文件，便于通过字符串 key 找到行号。

    生成：
      - <npy_path> : 形状 (N, *elem_shape) 的纯数值 ndarray
      - 如果 save_index_path:
          * .json  -> {"key": row, ...}  （小索引建议用 json）
          或
          * .npy   -> 两个文件：<base>.keys.npy（定长字节串）, <base>.rows.npy（int32）
    """
    pkl_path = Path(pkl_path)
    npy_path = Path(npy_path)

    # 1) 读 pkl
    with open(pkl_path, "rb") as f:
        cache = pickle.load(f)

    # 2) 过滤条目（忽略 meta）
    items = [(k, v) for k, v in cache.items()
             if k != "meta" and isinstance(v, dict) and "spectral_weights" in v]
    if not items:
        raise ValueError("缓存中没有可用条目（未找到 'spectral_weights'）。")

    # 固定顺序（可复现）
    items.sort(key=lambda kv: kv[0])
    keys = [k for k, _ in items]

    # 3) 规范化单条为纯数值数组
    def _to_array(sw):
        arr = np.asarray(sw)
        if drop_leading_one and arr.ndim >= 3 and arr.shape[0] == 1:
            arr = arr.reshape(arr.shape[1:])
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"spectral_weights 含非数值类型，得到 dtype={arr.dtype}")
        return np.asarray(arr, dtype=force_dtype, order="C")

    first = _to_array(items[0][1]["spectral_weights"])
    elem_shape = first.shape

    # 4) 堆叠为 (N, *elem_shape) 并保存为 .npy
    out = np.empty((len(items), *elem_shape), dtype=first.dtype)
    out[0] = first
    for i, (_, v) in enumerate(items[1:], start=1):
        arr = _to_array(v["spectral_weights"])
        if arr.shape != elem_shape:
            raise ValueError(f"第 {i} 条 spectral_weights 形状不一致：{arr.shape} vs {elem_shape}")
        out[i] = arr

    np.save(npy_path, out)  # 直接存 .npy（一次性写入）
    print(f"[OK] 保存: {npy_path} shape={out.shape} dtype={out.dtype}")

    # 5) 可选：保存 key->row 索引
    if save_index_path is None:
        save_index_path = npy_path.with_suffix('.json')
    save_index_path = Path(save_index_path)
    if save_index_path.suffix.lower() == ".json":
        # 直接保存为字典（小索引最简单最快）
        idx = {k: i for i, k in enumerate(keys)}
        idx["meta"] = {"cluster_num": 3}
        save_index_path.write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] 索引(JSON): {save_index_path}  (共 {len(idx)} 条)")
    else:
        # 纯 numpy 版本（两个文件，可内存映射或一次性加载）
        base = save_index_path.with_suffix('')
        keys_arr = np.array([k.encode('utf-8') for k in keys],
                            dtype=f"S{max(len(k.encode('utf-8')) for k in keys)}")
        rows_arr = np.arange(len(keys), dtype=np.int32)
        np.save(base.with_suffix(".keys.npy"), keys_arr)
        np.save(base.with_suffix(".rows.npy"), rows_arr)
        print(f"[OK] 索引(NPY): {base.with_suffix('.keys.npy')}, {base.with_suffix('.rows.npy')}")

    return npy_path

def test_npy_cache(npy_path):
    npy_path = Path(npy_path)
    base = npy_path.with_suffix('')
    npy_path = base.with_suffix('.npy')
    json_path = base.with_suffix('.json')
    
    def what_is_this_file(path):
        path = Path(path)
        with open(path, "rb") as f:
            head = f.read(8)
        if head.startswith(b"\x93NUMPY"):   # 正确的 .npy
            return "npy"
        if head[:2] == b"PK":               # zip 容器：npz / joblib 压缩包
            return "zip"
        if head[:2] == b"\x80\x04":         # pickle 魔术头
            return "pickle"
        return "unknown"

    print(what_is_this_file(npy_path))

    mm = np.load(npy_path)
    key_value = json.load(open(json_path))

    print(mm.shape)
    print(mm.dtype)
    print(mm.nbytes)
    print(mm.size)
    print(mm.ndim)
    print(mm.strides)
    row0 = mm[100000]
    print(row0)
    print(key_value[0])

def run_all_tests():
    """运行所有测试"""
    print("开始运行所有测试...")
    print("=" * 50)
    
    # # 测试平均光谱权重
    # test_mean_spectral_weights()
    # print()
    
    # # 测试K-means聚类光谱权重
    # test_kmeans_spectral_weights()
    # print()
    
    # # 测试聚类可视化
    # test_cluster_visualization()
    # print()
    
    # # 测试新的独立类
    # test_memotr_spectral_weights()
    # print()
    
    # test_memotr_bbox_processor()
    # print()
    
    # # 测试转换器
    # test_mmrotate_converter()
    # print()
    
    # 测试缓存建立功能
    # test_cache_creation()
    # print()
    
    # 测试并行性能
    # test_parallel_performance()
    # print()
    
    # # 测试性能
    # test_performance()
    # print()
    
    # # 测试热力图生成
    test_heatmap_generation()
    print()

    test_heatmap_contrib_sum_by_scale()
    print()
    # test_heatmap_performance()
    # print()
    
    # 测试转换mmap
    # transfer_cache_to_npy(pkl_path="/data/users/litianhao/data/HSMOT/memotr_spectral_weights_cache_le135_kmeans_3.pkl", npy_path="/data/users/litianhao/data/HSMOT/memotr_spectral_weights_cache_le135_kmeans_3.npy")
    # test_npy_cache("/data/users/litianhao/data/HSMOT/memotr_spectral_weights_cache_le135_kmeans_3.npy")

    print("=" * 50)
    print("所有测试完成!")

if __name__ == '__main__':
    # 运行所有测试
    run_all_tests()



