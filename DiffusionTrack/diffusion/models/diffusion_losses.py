import math

import torch
import torch.nn.functional as F
from torch import nn
from mmcv.ops import box_iou_rotated, diff_iou_rotated_2d
from yolox.utils.dist import get_world_size, is_dist_avail_and_initialized
from yolox.utils.rotated_boxes import (aligned_rotated_iou,
                                       mean_pair_rotated_iou, rbox_to_qbox,
                                       regularize_rboxes, valid_rbox_mask)


def encode_le135_l1(boxes, image_whwh, angle_weight=0.05):
    """Encode absolute LE135 boxes for image-balanced direct L1.

    Canonicalization happens in physical coordinates first, so the two
    equivalent ``(w, h, theta)`` / ``(h, w, theta + pi/2)`` descriptions
    receive identical supervision.  Rotated-box width and height are local
    edge lengths rather than image x/y components; therefore all four spatial
    terms use one geometric-mean image scale instead of separately dividing
    them by image width/height.  The angle remains a direct, non-periodic L1
    term, with the same fixed small weight as the PairMOT mainline.
    """
    if boxes.size(-1) != 5:
        raise ValueError(
            f"Expected (..., 5) rotated boxes, got {tuple(boxes.shape)}")
    if image_whwh.size(-1) != 4:
        raise ValueError(
            f"Expected (..., 4) image scale, got {tuple(image_whwh.shape)}")
    if angle_weight < 0:
        raise ValueError("angle_weight must be non-negative")

    physical = regularize_rboxes(boxes)
    image_scale = torch.sqrt(
        image_whwh[..., 0] * image_whwh[..., 1]).clamp_min(1.0)
    while image_scale.ndim < physical.ndim:
        image_scale = image_scale.unsqueeze(-1)
    spatial = physical[..., :4] / image_scale
    angle = ((physical[..., 4:5] + math.pi / 4) / math.pi
             * float(angle_weight))
    return torch.cat([spatial, angle], dim=-1)


def _rbox_l1_cost(boxes1, boxes2):
    """Pairwise direct L1 cost over normalized le135 boxes."""
    return torch.cdist(boxes1, boxes2, p=1)


def encode_qbox8_l1(boxes, image_whwh):
    """Encode absolute rotated boxes as LX-style normalized qbox8.

    LX converts predictions to ordered corners and divides alternating x/y
    coordinates by image width/height before applying pairwise L1.  Keep this
    separate from the default LE135 encoder so controlled experiments can
    change matcher geometry without changing the regression loss.
    """
    if boxes.size(-1) != 5:
        raise ValueError(
            f"Expected (..., 5) rotated boxes, got {tuple(boxes.shape)}")
    if image_whwh.size(-1) != 4:
        raise ValueError(
            f"Expected (..., 4) image scale, got {tuple(image_whwh.shape)}")
    qboxes = rbox_to_qbox(boxes)
    qbox_scale = image_whwh.repeat(2)
    return qboxes / qbox_scale


def normalize_qbox8_l1(qboxes, image_whwh):
    """Normalize raw GT qbox8 exactly as used by the LX matcher."""
    if qboxes.size(-1) != 8:
        raise ValueError(
            f"Expected (..., 8) quadrilaterals, got {tuple(qboxes.shape)}")
    if image_whwh.size(-1) != 4:
        raise ValueError(
            f"Expected (..., 4) image scale, got {tuple(image_whwh.shape)}")
    qbox_scale = image_whwh.repeat(2)
    return qboxes / qbox_scale


def encode_lx_norm5_l1(boxes, image_whwh, angle_weight=1.0):
    """Encode absolute radian boxes as LX's raw normalized five-vector.

    LX divides ``(cx, cy, w, h)`` by ``(W, H, W, H)`` and maps the LE135
    angle to ``(theta + pi/4) / pi`` before applying ordinary, non-periodic
    elementwise L1. It does not canonicalize predictions in this loss.
    """
    if boxes.size(-1) != 5:
        raise ValueError(
            f"Expected (..., 5) rotated boxes, got {tuple(boxes.shape)}")
    spatial = boxes[..., :4] / image_whwh
    angle = ((boxes[..., 4:5] + math.pi / 4) / math.pi
             * float(angle_weight))
    return torch.cat([spatial, angle], dim=-1)


def sigmoid_focal_loss_jit(inputs, targets, alpha=-1, gamma=2,
                           reduction="none"):
    """Torch-native focal loss, avoiding DiffusionTrack's Detectron2/FVCore dependency."""
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class SetCriterionDynamicK(nn.Module):
    """ This class computes the loss for DiffusionDet.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self,num_classes, matcher, weight_dict, eos_coef, losses,
                 use_focal, use_fed_loss, bbox_angle_l1_weight=0.05):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.use_focal = use_focal
        self.use_fed_loss = use_fed_loss
        self.bbox_angle_l1_weight = float(bbox_angle_l1_weight)
        if self.bbox_angle_l1_weight < 0:
            raise ValueError("bbox_angle_l1_weight must be non-negative")
        # Current HSMOT experiments historically sum the two frame IoU losses
        # per pair.  LX averages over all frame matches instead.  Keep the
        # historical default and expose LX normalization as a controlled
        # criterion-only switch.
        self.average_pair_iou_loss = False
        self.bbox_l1_representation = "le135_geomean"
        self.iou_numerical_mode = "guarded"
        self.class_fusion_mode = "stable"
        # Diagnostic snapshots are opt-in and are only enabled by the
        # single-image overfit experiment.  The formal training hot path does
        # not retain matcher tensors.
        self.capture_debug = False
        self.last_debug_assignments = None
        if self.use_fed_loss:
            self.fed_loss_num_classes = 50
            from detectron2.data.detection_utils import get_fed_loss_cls_weights
            cls_weight_fun = lambda: get_fed_loss_cls_weights(dataset_names=cfg.DATASETS.TRAIN, freq_weight_power=cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT_POWER)  # noqa
            fed_loss_cls_weights = cls_weight_fun()
            assert (
                    len(fed_loss_cls_weights) == self.num_classes
            ), "Please check the provided fed_loss_cls_weights. Their size should match num_classes"
            self.register_buffer("fed_loss_cls_weights", fed_loss_cls_weights)

        if self.use_focal:
            self.focal_loss_alpha = 0.25
            self.focal_loss_gamma = 2.0
        else:
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = self.eos_coef
            self.register_buffer('empty_weight', empty_weight)

    # copy-paste from https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/fast_rcnn.py#L356
    def get_fed_loss_classes(self, gt_classes, num_fed_loss_classes, num_classes, weight):
        """
        Args:
            gt_classes: a long tensor of shape R that contains the gt class label of each proposal.
            num_fed_loss_classes: minimum number of classes to keep when calculating federated loss.
            Will sample negative classes if number of unique gt_classes is smaller than this value.
            num_classes: number of foreground classes
            weight: probabilities used to sample negative classes
        Returns:
            Tensor:
                classes to keep when calculating the federated loss, including both unique gt
                classes and sampled negative classes.
        """
        unique_gt_classes = torch.unique(gt_classes)
        prob = unique_gt_classes.new_ones(num_classes + 1).float()
        prob[-1] = 0
        if len(unique_gt_classes) < num_fed_loss_classes:
            prob[:num_classes] = weight.float().clone()
            prob[unique_gt_classes] = 0
            sampled_negative_classes = torch.multinomial(
                prob, num_fed_loss_classes - len(unique_gt_classes), replacement=False
            )
            fed_loss_classes = torch.cat([unique_gt_classes, sampled_negative_classes])
        else:
            fed_loss_classes = unique_gt_classes
        return fed_loss_classes

    def loss_labels(self, outputs, targets, indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        # The original inverse-logit fusion is unsafe in both FP16 and BF16:
        # ``1 - 1e-6`` rounds to exactly one, making log(p / (1-p)) infinite.
        # Preserve the original geometric-mean fusion but evaluate it, focal
        # loss included, in FP32 with a representable logit clamp.
        if self.class_fusion_mode == "lx_raw":
            src_logits = outputs['pred_logits']
            conf_score = torch.cat(
                [outputs['pred_scores'], outputs['pred_scores']], dim=0)
            p = torch.sqrt(torch.sigmoid(src_logits) * conf_score)
            src_logits = torch.log(p / (1 - p))
        elif self.class_fusion_mode == "stable":
            with torch.cuda.amp.autocast(enabled=False):
                src_logits = outputs['pred_logits'].float()
                conf_score = torch.cat(
                    [outputs['pred_scores'],
                     outputs['pred_scores']], dim=0).float()
                p = torch.sqrt(torch.sigmoid(src_logits) * conf_score)
                src_logits = torch.logit(p, eps=1e-6)
        else:
            raise ValueError(
                "class_fusion_mode must be 'stable' or 'lx_raw', got "
                f"{self.class_fusion_mode!r}")
        batch_size = len(targets)

        # src_logits_re=torch.cat((src_logits[:batch_size//2],src_logits[batch_size//2:]),dim=0)
        # src_logits=(src_logits+src_logits_re)/2

        # idx = self._get_src_permutation_idx(indices)
        # target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # src_logits_list = []
        target_classes_o_list = []
        # target_classes[idx] = target_classes_o
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx%(batch_size//2)][0]
            gt_multi_idx = indices[batch_idx%(batch_size//2)][1]
            if len(gt_multi_idx) == 0:
                continue
            # bz_src_logits = src_logits[batch_idx]
            target_classes_o = targets[batch_idx]["labels"]
            target_classes[batch_idx, valid_query] = target_classes_o[gt_multi_idx]

            # src_logits_list.append(bz_src_logits[valid_query])
            target_classes_o_list.append(target_classes_o[gt_multi_idx])

        if self.use_focal or self.use_fed_loss:
            num_boxes = torch.cat(target_classes_o_list).shape[0] if len(target_classes_o_list) != 0 else 1

            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], self.num_classes + 1],
                                                dtype=torch.float32, layout=src_logits.layout,
                                                device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            loss_ce=0
            gt_classes = torch.argmax(target_classes_onehot, dim=-1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]
            target_classes_onehot = target_classes_onehot.flatten(0, 1)
            src_logits = src_logits.flatten(0, 1)
            if self.use_focal:
                cls_loss = sigmoid_focal_loss_jit(src_logits, target_classes_onehot, alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma, reduction="none")
            else:
                cls_loss = F.binary_cross_entropy_with_logits(src_logits, target_classes_onehot, reduction="none")
            if self.use_fed_loss:
                K = self.num_classes
                N = src_logits.shape[0]
                fed_loss_classes = self.get_fed_loss_classes(
                    gt_classes,
                    num_fed_loss_classes=self.fed_loss_num_classes,
                    num_classes=K,
                    weight=self.fed_loss_cls_weights,
                )
                fed_loss_classes_mask = fed_loss_classes.new_zeros(K + 1)
                fed_loss_classes_mask[fed_loss_classes] = 1
                fed_loss_classes_mask = fed_loss_classes_mask[:K]
                weight = fed_loss_classes_mask.view(1, K).expand(N, K).float()

                loss_ce += torch.sum(cls_loss * weight) / num_boxes
            else:
                loss_ce += torch.sum(cls_loss) / num_boxes

            losses = {'loss_ce': loss_ce}
        else:
            raise NotImplementedError

        return losses

    def normalize_pair_iou_sum(self, loss_riou, matched):
        """Normalize a two-frame IoU-loss sum under the selected protocol."""
        normalizer = max(matched, 1)
        if self.average_pair_iou_loss:
            normalizer *= 2
        return loss_riou / normalizer

    def encode_bbox_l1(self, boxes, image_whwh):
        if self.bbox_l1_representation == "le135_geomean":
            return encode_le135_l1(
                boxes, image_whwh, self.bbox_angle_l1_weight)
        if self.bbox_l1_representation == "lx_norm5":
            return encode_lx_norm5_l1(
                boxes, image_whwh, self.bbox_angle_l1_weight)
        raise ValueError(
            "bbox_l1_representation must be 'le135_geomean' or "
            f"'lx_norm5', got {self.bbox_l1_representation!r}")

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute direct LE135 L1 and independent frame-wise rotated IoU.

        The ``loss_giou`` key is intentionally retained for checkpoint/config
        compatibility.  It contains the sum of reference/current ordinary
        rotated-IoU losses; it is not a pair-IoU or GIoU loss.
        """
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes']
        batch_size = len(targets)
        pair_batch = batch_size // 2
        loss_bbox = src_boxes.sum() * 0
        loss_riou = src_boxes.sum() * 0
        matched = 0
        for batch_idx in range(pair_batch):
            valid_query, gt_multi_idx = indices[batch_idx]
            if len(gt_multi_idx) == 0:
                continue
            cur_idx = batch_idx + pair_batch
            pred_ref = src_boxes[batch_idx, valid_query]
            pred_cur = src_boxes[cur_idx, valid_query]
            tgt_ref = targets[batch_idx]["boxes_abs"][gt_multi_idx]
            tgt_cur = targets[cur_idx]["boxes_abs"][gt_multi_idx]

            pred_ref_norm = self.encode_bbox_l1(
                pred_ref, targets[batch_idx]['image_size_xyxy'])
            pred_cur_norm = self.encode_bbox_l1(
                pred_cur, targets[cur_idx]['image_size_xyxy'])
            tgt_ref_norm = self.encode_bbox_l1(
                tgt_ref, targets[batch_idx]['image_size_xyxy'])
            tgt_cur_norm = self.encode_bbox_l1(
                tgt_cur, targets[cur_idx]['image_size_xyxy'])

            for pred_norm, tgt_norm in ((pred_ref_norm, tgt_ref_norm),
                                        (pred_cur_norm, tgt_cur_norm)):
                loss_bbox = loss_bbox + F.l1_loss(
                    pred_norm, tgt_norm, reduction='none').sum()

            if self.iou_numerical_mode == "lx_raw":
                iou_ref = diff_iou_rotated_2d(
                    pred_ref[None], tgt_ref[None])[0]
                iou_cur = diff_iou_rotated_2d(
                    pred_cur[None], tgt_cur[None])[0]
            elif self.iou_numerical_mode == "guarded":
                iou_ref = aligned_rotated_iou(pred_ref, tgt_ref)
                iou_cur = aligned_rotated_iou(pred_cur, tgt_cur)
            else:
                raise ValueError(
                    "iou_numerical_mode must be 'guarded' or 'lx_raw', got "
                    f"{self.iou_numerical_mode!r}")
            loss_riou = loss_riou + (
                (1 - iou_ref).sum() + (1 - iou_cur).sum())
            matched += len(gt_multi_idx)

        normalizer = max(matched, 1)
        return {
            'loss_bbox': loss_bbox / (2 * normalizer),
            'loss_giou': self.normalize_pair_iou_sum(loss_riou, matched),
        }

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        self.matcher.capture_debug = self.capture_debug
        indices, matched_ids = self.matcher(outputs_without_aux, targets)
        final_debug = (
            self._pack_debug_assignment(
                indices, matched_ids, outputs_without_aux, targets,
                self.matcher.last_debug_costs)
            if self.capture_debug else None)
        auxiliary_debug = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)//2
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices, matched_ids = self.matcher(aux_outputs, targets)
                if self.capture_debug:
                    auxiliary_debug.append(self._pack_debug_assignment(
                        indices, matched_ids, aux_outputs, targets,
                        self.matcher.last_debug_costs))
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        self.last_debug_assignments = (
            auxiliary_debug + [final_debug] if self.capture_debug else None)

        return losses

    @torch.no_grad()
    def _pack_debug_assignment(self, indices, matched_ids, outputs, targets,
                               matcher_costs):
        """Detach assignments plus exact selected costs/loss contributions."""
        packed = []
        pair_batch = len(targets) // 2
        pred_boxes = outputs['pred_boxes'].float()
        pred_logits = outputs['pred_logits'].float()
        pair_scores = outputs['pred_scores'].float()
        fused_scores = torch.cat([pair_scores, pair_scores], dim=0)
        fused_probability = torch.sqrt(
            pred_logits.sigmoid() * fused_scores)
        fused_logits = torch.logit(fused_probability, eps=1e-6)

        for pair_index, ((selected_query, gt_indices), best_query) in enumerate(
                zip(indices, matched_ids)):
            query_indices = torch.nonzero(
                selected_query, as_tuple=False).squeeze(1)
            item = {
                'query_indices': torch.nonzero(
                    selected_query, as_tuple=False).squeeze(1).detach().cpu(),
                'gt_indices': gt_indices.detach().cpu(),
                'best_query_per_gt': best_query.detach().cpu(),
            }
            if matcher_costs is not None and pair_index < len(matcher_costs):
                item.update(matcher_costs[pair_index])

            if len(query_indices):
                cur_index = pair_index + pair_batch
                gt_ref = targets[pair_index]['boxes_abs'][gt_indices]
                gt_cur = targets[cur_index]['boxes_abs'][gt_indices]
                pred_ref = pred_boxes[pair_index, query_indices]
                pred_cur = pred_boxes[cur_index, query_indices]
                size_ref = targets[pair_index]['image_size_xyxy']
                size_cur = targets[cur_index]['image_size_xyxy']
                l1_ref = (
                    self.encode_bbox_l1(pred_ref, size_ref)
                    - self.encode_bbox_l1(gt_ref, size_ref)
                ).abs().sum(dim=1)
                l1_cur = (
                    self.encode_bbox_l1(pred_cur, size_cur)
                    - self.encode_bbox_l1(gt_cur, size_cur)
                ).abs().sum(dim=1)
                iou_ref = aligned_rotated_iou(pred_ref, gt_ref)
                iou_cur = aligned_rotated_iou(pred_cur, gt_cur)

                labels = targets[pair_index]['labels'][gt_indices]
                class_losses = []
                for side_index in (pair_index, cur_index):
                    logits = fused_logits[side_index, query_indices]
                    onehot = torch.zeros_like(logits)
                    onehot.scatter_(1, labels[:, None], 1)
                    class_losses.append(sigmoid_focal_loss_jit(
                        logits, onehot, alpha=self.focal_loss_alpha,
                        gamma=self.focal_loss_gamma,
                        reduction='none').sum(dim=1))

                pair_l1 = (l1_ref + l1_cur) / 2
                pair_riou = (1 - iou_ref) + (1 - iou_cur)
                pair_class = (class_losses[0] + class_losses[1]) / 2
                item.update({
                    'loss_l1_ref': l1_ref.detach().cpu(),
                    'loss_l1_cur': l1_cur.detach().cpu(),
                    'loss_l1_pair': pair_l1.detach().cpu(),
                    'loss_l1_weighted': (
                        pair_l1 * self.weight_dict['loss_bbox']
                    ).detach().cpu(),
                    'loss_riou_ref': (1 - iou_ref).detach().cpu(),
                    'loss_riou_cur': (1 - iou_cur).detach().cpu(),
                    'loss_riou_pair': pair_riou.detach().cpu(),
                    'loss_riou_weighted': (
                        pair_riou * self.weight_dict['loss_giou']
                    ).detach().cpu(),
                    # This is the exact focal loss of each matched query.
                    # The reported global CE additionally contains all
                    # unmatched-query background terms.
                    'loss_cls_matched_query': pair_class.detach().cpu(),
                    'loss_cls_matched_query_weighted': (
                        pair_class * self.weight_dict['loss_ce']
                    ).detach().cpu(),
                })
            packed.append(item)
        return packed


class HungarianMatcherDynamicK(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-k (dynamic) matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1,
                 cost_giou: float = 1, cost_mask: float = 1,
                 use_focal: bool = False, use_fed_loss=False,
                 bbox_angle_l1_weight: float = 0.05):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_focal = use_focal
        self.use_fed_loss = use_fed_loss
        self.bbox_angle_l1_weight = float(bbox_angle_l1_weight)
        if self.bbox_angle_l1_weight < 0:
            raise ValueError("bbox_angle_l1_weight must be non-negative")
        self.ota_k = 5
        # Diagnostic switches.  Formal training keeps the repaired defaults:
        # no hard box-and-center prior and guaranteed GT coverage.  Individual
        # ablations may restore the legacy center prior and permit GTs to lose
        # their positive after query-conflict resolution.
        self.center_prior_penalty_weight = 0.0
        # Keep the historical no-center behavior as the default so existing
        # experiments remain reproducible. Controlled bridge experiments can
        # explicitly enable the penalty without changing any other matcher
        # variable.
        self.apply_center_prior_penalty = False
        self.force_gt_coverage = True
        # ``corrected`` uses the current conflict mask and guarantees unique
        # all-GT coverage. ``lx_stale`` exactly reproduces LX's stale-mask
        # repair. The legacy force_gt_coverage=False switch remains available
        # for the prior no-repair ablation.
        self.coverage_mode = "corrected"
        # The corrected HSMOT baseline uses representation-consistent LE135
        # matching.  ``qbox8`` reproduces LX's raw-corner L1 as an explicit
        # controlled variable without changing the criterion box loss.
        self.matching_l1_representation = "le135"
        self.numerical_mode = "guarded"
        self.capture_debug = False
        self.last_debug_costs = None
        if self.use_focal:
            self.focal_loss_alpha = 0.25
            self.focal_loss_gamma = 2.0
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0,  "all costs cant be 0"

    def compose_matching_cost(self, weighted_bbox, weighted_class,
                              weighted_riou, center_prior_penalty):
        """Compose learned matching terms with the optional center prior."""
        cost = weighted_bbox + weighted_class + weighted_riou
        if self.apply_center_prior_penalty:
            cost = cost + center_prior_penalty
        return cost

    def forward(self, outputs, targets):
        """ simOTA for detr"""
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            bs, num_queries = outputs["pred_logits"].shape[:2]
            if self.numerical_mode == "guarded":
                conf_score = outputs["pred_scores"].float()
                pred_logits = outputs["pred_logits"].float()
                pred_boxes = outputs["pred_boxes"].float()
            elif self.numerical_mode == "lx_raw":
                conf_score = outputs["pred_scores"]
                pred_logits = outputs["pred_logits"]
                pred_boxes = outputs["pred_boxes"]
            else:
                raise ValueError(
                    "numerical_mode must be 'guarded' or 'lx_raw', got "
                    f"{self.numerical_mode!r}")
            # We flatten to compute the cost matrices in a batch
            pred_logits_pre,pred_logits_curr=torch.split(pred_logits,bs//2,dim=0)
            out_bbox_pre,out_bbox_curr = torch.split(pred_boxes,bs//2,dim=0)
            if self.use_focal or self.use_fed_loss:
                out_prob_pre = torch.sqrt(pred_logits_pre.sigmoid()*conf_score)  # [batch_size, num_queries, num_classes]
                out_prob_curr = torch.sqrt(pred_logits_curr.sigmoid()*conf_score)
            else:
                out_prob_pre = torch.sqrt(pred_logits_pre.softmax(-1)*conf_score) # [batch_size, num_queries, num_classes]
                out_prob_curr=torch.sqrt(pred_logits_curr.softmax(-1)*conf_score)
            indices = []
            matched_ids = []
            debug_costs = []
            assert bs == len(targets)
            for batch_idx in range(bs//2):
                bz_boxes_pre = out_bbox_pre[batch_idx]  # [num_proposals, 5]
                bz_out_prob_pre = out_prob_pre[batch_idx]
                bz_boxes_curr = out_bbox_curr[batch_idx]  # [num_proposals, 5]
                bz_out_prob_curr = out_prob_curr[batch_idx]
                bz_tgt_ids_pre = targets[batch_idx]["labels"]
                bz_tgt_ids_curr = targets[batch_idx+bs//2]["labels"]
                num_insts = len(bz_tgt_ids_pre)
                assert len(bz_tgt_ids_curr)==num_insts,"YaHoo {},{}!".format(len(bz_tgt_ids_curr),num_insts)
                if num_insts == 0:  # empty object in key frame
                    non_valid = torch.zeros(bz_out_prob_pre.shape[0]).to(bz_out_prob_pre) > 0
                    indices_batchi = (non_valid, torch.arange(0, 0).to(bz_out_prob_pre))
                    matched_qidx = torch.arange(0, 0).to(bz_out_prob_pre)
                    indices.append(indices_batchi)
                    matched_ids.append(matched_qidx)
                    if self.capture_debug:
                        debug_costs.append({})
                    continue

                bz_gtboxs_abs_pre = targets[batch_idx]['boxes_abs']
                bz_gtboxs_abs_curr = targets[batch_idx+bs//2]['boxes_abs']
                fg_mask_pre, is_in_boxes_and_center_pre = self.get_in_boxes_info(
                    bz_boxes_pre, bz_gtboxs_abs_pre,
                    expanded_strides=32
                )
                fg_mask_curr, is_in_boxes_and_center_curr = self.get_in_boxes_info(
                    bz_boxes_curr, bz_gtboxs_abs_curr,
                    expanded_strides=32
                )
                fg_mask=fg_mask_pre&fg_mask_curr 
                is_in_boxes_and_center=is_in_boxes_and_center_pre&is_in_boxes_and_center_curr

                # SimOTA/Dynamic-K needs a score in [0, 1].  Average the two
                # independent frame IoUs here; using their sum would inflate
                # Dynamic-K and create too many positives.
                if self.numerical_mode == "lx_raw":
                    pair_wise_ious = (
                        box_iou_rotated(
                            bz_boxes_pre, bz_gtboxs_abs_pre,
                            mode="iou", aligned=False, clockwise=True)
                        + box_iou_rotated(
                            bz_boxes_curr, bz_gtboxs_abs_curr,
                            mode="iou", aligned=False, clockwise=True)
                    ) * 0.5
                else:
                    pair_wise_ious = mean_pair_rotated_iou(
                        bz_boxes_pre, bz_gtboxs_abs_pre,
                        bz_boxes_curr, bz_gtboxs_abs_curr)
                cost_class=0
                bz_out_prob_set=[bz_out_prob_pre,bz_out_prob_curr]
                bz_tgt_ids_set=[bz_tgt_ids_pre,bz_tgt_ids_curr]
                # Compute the classification cost.
                if self.use_focal:
                    alpha = self.focal_loss_alpha
                    gamma = self.focal_loss_gamma
                    for bz_out_prob,bz_tgt_ids in zip(bz_out_prob_set,bz_tgt_ids_set):
                        neg_cost_class = (1 - alpha) * (bz_out_prob ** gamma) * (-(1 - bz_out_prob + 1e-8).log())
                        pos_cost_class = alpha * ((1 - bz_out_prob) ** gamma) * (-(bz_out_prob + 1e-8).log())
                        cost_class += pos_cost_class[:, bz_tgt_ids] - neg_cost_class[:, bz_tgt_ids]
                elif self.use_fed_loss:
                    # focal loss degenerates to naive one
                    for bz_out_prob,bz_tgt_ids in zip(bz_out_prob_set,bz_tgt_ids_set):
                        neg_cost_class = (-(1 - bz_out_prob + 1e-8).log())
                        pos_cost_class = (-(bz_out_prob + 1e-8).log())
                        cost_class += pos_cost_class[:, bz_tgt_ids] - neg_cost_class[:, bz_tgt_ids]
                else:
                    for bz_out_prob,bz_tgt_ids in zip(bz_out_prob_set,bz_tgt_ids_set):
                        cost_class += -bz_out_prob[:, bz_tgt_ids]

                # Compute the L1 cost between boxes
                # image_size_out = torch.cat([v["image_size_xyxy"].unsqueeze(0) for v in targets])
                # image_size_out = image_size_out.unsqueeze(1).repeat(1, num_queries, 1).flatten(0, 1)
                # image_size_tgt = torch.cat([v["image_size_xyxy_tgt"] for v in targets])

                bz_image_size_out_pre = targets[batch_idx]['image_size_xyxy']
                bz_image_size_out_curr = targets[batch_idx+bs//2]['image_size_xyxy']

                if self.matching_l1_representation == "le135":
                    bz_out_bbox_pre = encode_le135_l1(
                        bz_boxes_pre, bz_image_size_out_pre,
                        self.bbox_angle_l1_weight)
                    bz_out_bbox_curr = encode_le135_l1(
                        bz_boxes_curr, bz_image_size_out_curr,
                        self.bbox_angle_l1_weight)
                    bz_gtboxs_pre = encode_le135_l1(
                        bz_gtboxs_abs_pre, bz_image_size_out_pre,
                        self.bbox_angle_l1_weight)
                    bz_gtboxs_curr = encode_le135_l1(
                        bz_gtboxs_abs_curr, bz_image_size_out_curr,
                        self.bbox_angle_l1_weight)
                elif self.matching_l1_representation == "qbox8":
                    bz_out_bbox_pre = encode_qbox8_l1(
                        bz_boxes_pre, bz_image_size_out_pre)
                    bz_out_bbox_curr = encode_qbox8_l1(
                        bz_boxes_curr, bz_image_size_out_curr)
                    bz_gtboxs_pre = normalize_qbox8_l1(
                        targets[batch_idx]["boxes_qbox"],
                        bz_image_size_out_pre)
                    bz_gtboxs_curr = normalize_qbox8_l1(
                        targets[batch_idx + bs // 2]["boxes_qbox"],
                        bz_image_size_out_curr)
                else:
                    raise ValueError(
                        "matching_l1_representation must be 'le135' or "
                        f"'qbox8', got {self.matching_l1_representation!r}")
                cost_bbox_pre = _rbox_l1_cost(
                    bz_out_bbox_pre, bz_gtboxs_pre)
                cost_bbox_curr = _rbox_l1_cost(
                    bz_out_bbox_curr, bz_gtboxs_curr)

                cost_giou = -pair_wise_ious

                # Final cost matrix
                cost_bbox_pair = (cost_bbox_pre + cost_bbox_curr) / 2
                cost_class_pair = cost_class / 2
                weighted_bbox = self.cost_bbox * cost_bbox_pair
                weighted_class = self.cost_class * cost_class_pair
                weighted_riou = self.cost_giou * cost_giou
                # Diffusion queries are continuous random boxes rather than
                # fixed grid anchors.  Do not impose YOLOX's strict
                # box-and-center prior on them; it otherwise dominates all
                # learned costs by two orders of magnitude.  Keep the wider
                # foreground exclusion below to reject clearly unrelated
                # proposals.
                center_prior_penalty = (
                    float(self.center_prior_penalty_weight)
                    * (~is_in_boxes_and_center).to(cost_bbox_pair.dtype)
                )
                cost = self.compose_matching_cost(
                    weighted_bbox, weighted_class, weighted_riou,
                    center_prior_penalty)
                # A single invalid diffusion proposal must not terminate all
                # DDP ranks. Exclude non-finite proposal rows from matching;
                # valid rows and their original costs are left untouched.
                if self.numerical_mode == "guarded":
                    valid_rows = (
                        valid_rbox_mask(bz_boxes_pre)
                        & valid_rbox_mask(bz_boxes_curr)
                        & torch.isfinite(bz_out_prob_pre).all(dim=1)
                        & torch.isfinite(bz_out_prob_curr).all(dim=1)
                    )
                    if not valid_rows.any():
                        raise FloatingPointError(
                            "all diffusion proposals are non-finite during "
                            "matching")
                    cost = torch.where(
                        torch.isfinite(cost), cost,
                        torch.full_like(cost, 1e8))
                    cost[~valid_rows] = 1e8
                else:
                    assert not torch.isnan(cost).any()
                # cost = (cost_class + 3.0 * cost_giou + 100.0 * (~is_in_boxes_and_center))  # [num_query,num_gt]
                fg_penalty = (
                    10000.0 * (~fg_mask).to(cost.dtype).unsqueeze(1)
                ).expand_as(cost)
                cost = cost + fg_penalty

                # ``dynamic_k_matching`` intentionally mutates its cost
                # argument in the LX stale-coverage branch: already matched
                # rows receive +100000 before each uncovered-GT repair.
                # That is assignment control flow, not the cost that selected
                # the original positive.  Preserve the pre-repair matrix for
                # diagnostics so reported totals stay comparable to LX and
                # equal the displayed components.  The clone is made only
                # when visual diagnostics are requested and never enters the
                # matching or gradient path.
                debug_cost = cost.clone() if self.capture_debug else None

                # if bz_gtboxs.shape[0]>0:
                indices_batchi, matched_qidx = self.dynamic_k_matching(cost, pair_wise_ious, bz_gtboxs_pre.shape[0])

                indices.append(indices_batchi)
                matched_ids.append(matched_qidx)
                if self.capture_debug:
                    query_indices = torch.nonzero(
                        indices_batchi[0], as_tuple=False).squeeze(1)
                    gt_indices = indices_batchi[1]

                    def selected(values):
                        return values[query_indices, gt_indices].detach().cpu()

                    debug_costs.append({
                        'match_cost_class': selected(cost_class_pair),
                        'match_cost_l1_ref': selected(cost_bbox_pre),
                        'match_cost_l1_cur': selected(cost_bbox_curr),
                        'match_cost_l1_pair': selected(cost_bbox_pair),
                        'match_cost_pair_iou': selected(pair_wise_ious),
                        'match_cost_riou': selected(cost_giou),
                        'match_cost_weighted_class': selected(weighted_class),
                        'match_cost_weighted_l1': selected(weighted_bbox),
                        'match_cost_weighted_riou': selected(weighted_riou),
                        'match_cost_center_penalty': selected(
                            center_prior_penalty),
                        'match_cost_fg_penalty': selected(fg_penalty),
                        'match_cost_total': selected(debug_cost),
                    })

        self.last_debug_costs = debug_costs if self.capture_debug else None
        return indices, matched_ids

    def get_in_boxes_info(self, boxes, target_gts, expanded_strides):
        # SimOTA's center prior operates in global image x/y coordinates.
        # Rotated-box w/h are local-axis extents and cannot be used directly as
        # global x/y extents.  Use the enclosing AABB for the original prior;
        # the actual assignment cost still uses exact paired rotated IoU.
        corners = rbox_to_qbox(target_gts).reshape(-1, 4, 2)
        xy_target_gts = torch.cat(
            [corners.amin(dim=1), corners.amax(dim=1)], dim=1)

        anchor_center_x = boxes[:, 0].unsqueeze(1)
        anchor_center_y = boxes[:, 1].unsqueeze(1)

        # whether the center of each anchor is inside a gt box
        b_l = anchor_center_x > xy_target_gts[:, 0].unsqueeze(0)
        b_r = anchor_center_x < xy_target_gts[:, 2].unsqueeze(0)
        b_t = anchor_center_y > xy_target_gts[:, 1].unsqueeze(0)
        b_b = anchor_center_y < xy_target_gts[:, 3].unsqueeze(0)
        # (b_l.long()+b_r.long()+b_t.long()+b_b.long())==4 [300,num_gt] ,
        is_in_boxes = ((b_l.long() + b_r.long() + b_t.long() + b_b.long()) == 4)
        is_in_boxes_all = is_in_boxes.sum(1) > 0  # [num_query]
        # in fixed center
        center_radius = 2.5
        # Modified to self-adapted sampling --- the center size depends on the size of the gt boxes
        # https://github.com/dulucas/UVO_Challenge/blob/main/Track1/detection/mmdet/core/bbox/assigners/rpn_sim_ota_assigner.py#L212
        b_l = anchor_center_x > (target_gts[:, 0] - (center_radius * (xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        b_r = anchor_center_x < (target_gts[:, 0] + (center_radius * (xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        b_t = anchor_center_y > (target_gts[:, 1] - (center_radius * (xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)
        b_b = anchor_center_y < (target_gts[:, 1] + (center_radius * (xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)

        is_in_centers = ((b_l.long() + b_r.long() + b_t.long() + b_b.long()) == 4)
        is_in_centers_all = is_in_centers.sum(1) > 0

        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = (is_in_boxes & is_in_centers)

        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, num_gt):
        matching_matrix = torch.zeros_like(cost)  # [300,num_gt]
        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = self.ota_k

        if num_gt > cost.shape[0]:
            raise RuntimeError(
                f"Dynamic-K requires at least one query per GT, but got "
                f"{cost.shape[0]} queries for {num_gt} GTs")

        # Take the sum of the predicted value and the top 10 iou of gt with the largest iou as dynamic_k
        topk_ious, _ = torch.topk(
            ious_in_boxes_matrix, min(n_candidate_k, cost.shape[0]), dim=0)
        dynamic_ks = torch.clamp(
            topk_ious.sum(0).int(), min=1, max=cost.shape[0])

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        # Preserve LX's original pre-resolution mask for the explicit stale
        # repair bridge. Corrected matching always recomputes conflicts.
        initial_conflicting_queries = matching_matrix.sum(1) > 1

        # A query may be selected independently by several GTs. Keep only the
        # lowest-cost GT for every such query. The conflict mask must always be
        # computed from the current matrix (the old implementation reused a
        # stale pre-repair mask below).
        conflicting_queries = initial_conflicting_queries
        if conflicting_queries.any():
            conflict_rows = torch.nonzero(
                conflicting_queries, as_tuple=False).squeeze(1)
            best_gt = cost[conflict_rows].argmin(dim=1)
            matching_matrix[conflict_rows] = 0
            matching_matrix[conflict_rows, best_gt] = 1

        if self.coverage_mode == "lx_stale":
            # Exact LX behavior: all currently matched rows are penalized,
            # every missing GT independently takes its cheapest row, then any
            # newly-created conflict is "resolved" using the stale mask from
            # before the first conflict pass. This can leave several GT
            # columns on one newly conflicting query; returned assignments
            # retain only one of them.
            while (matching_matrix.sum(0) == 0).any():
                matched_query_id = matching_matrix.sum(1) > 0
                cost[matched_query_id] += 100000.0
                unmatched_gt = torch.nonzero(
                    matching_matrix.sum(0) == 0,
                    as_tuple=False).squeeze(1)
                for gt_idx in unmatched_gt:
                    pos_idx = torch.argmin(cost[:, gt_idx])
                    matching_matrix[pos_idx, gt_idx] = 1.0
                if (matching_matrix.sum(1) > 1).any():
                    _, cost_argmin = torch.min(
                        cost[initial_conflicting_queries], dim=1)
                    matching_matrix[initial_conflicting_queries] = 0
                    matching_matrix[
                        initial_conflicting_queries, cost_argmin] = 1
            assert not (matching_matrix.sum(0) == 0).any()
        elif self.coverage_mode == "corrected" and self.force_gt_coverage:
            # Conflict resolution can leave GTs without positives. Repair them
            # one at a time so two missing GTs cannot choose the same free
            # query. If no free query remains, move a query only from a GT
            # that currently has more than one positive.
            missing_gts = torch.nonzero(
                matching_matrix.sum(0) == 0, as_tuple=False).squeeze(1)
            for gt_idx in missing_gts:
                free_queries = matching_matrix.sum(1) == 0
                if free_queries.any():
                    candidates = torch.nonzero(
                        free_queries, as_tuple=False).squeeze(1)
                else:
                    assigned_gt = matching_matrix.argmax(dim=1)
                    gt_positive_counts = matching_matrix.sum(0)
                    donor_queries = gt_positive_counts[assigned_gt] > 1
                    if not donor_queries.any():
                        raise RuntimeError(
                            "cannot give every GT a unique positive query")
                    candidates = torch.nonzero(
                        donor_queries, as_tuple=False).squeeze(1)

                pos_idx = candidates[cost[candidates, gt_idx].argmin()]
                matching_matrix[pos_idx] = 0
                matching_matrix[pos_idx, gt_idx] = 1

        elif self.coverage_mode != "corrected":
            raise ValueError(
                "coverage_mode must be 'corrected' or 'lx_stale', got "
                f"{self.coverage_mode!r}")

        if self.coverage_mode != "lx_stale":
            assert (matching_matrix.sum(1) <= 1).all()
        if (self.coverage_mode == "corrected"
                and self.force_gt_coverage):
            assert (matching_matrix.sum(0) >= 1).all()
        selected_query = matching_matrix.sum(1) > 0
        gt_indices = matching_matrix[selected_query].max(1)[1]
        assert selected_query.sum() == len(gt_indices)

        matched_cost = cost.masked_fill(matching_matrix == 0, float('inf'))
        matched_values, matched_query_id = torch.min(matched_cost, dim=0)
        matched_query_id = matched_query_id.masked_fill(
            ~torch.isfinite(matched_values), -1)

        return (selected_query, gt_indices), matched_query_id
