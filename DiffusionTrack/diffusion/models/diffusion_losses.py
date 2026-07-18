import math

import torch
import torch.nn.functional as F
from torch import nn
from yolox.utils.dist import get_world_size, is_dist_avail_and_initialized
from yolox.utils.rotated_boxes import pair_rotated_iou, rotated_iou


def _normalize_rboxes(boxes, image_whwh):
    """Normalize absolute rboxes to cxcywh in [0, 1] and le90 angle in [0, 1]."""
    normalized = boxes.clone()
    normalized[..., :4] /= image_whwh
    normalized[..., 4] = (normalized[..., 4] + math.pi / 2) / math.pi
    return normalized


def _periodic_l1_cost(boxes1, boxes2):
    """Pairwise L1 cost with the pi-periodic rotated-box angle distance."""
    cost = torch.cdist(boxes1[..., :4], boxes2[..., :4], p=1)
    angle = (boxes1[:, None, 4] - boxes2[None, :, 4]).abs()
    return cost + torch.minimum(angle, 1.0 - angle)


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
    def __init__(self,num_classes, matcher, weight_dict, eos_coef, losses, use_focal,use_fed_loss):
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
        src_logits = outputs['pred_logits']
        conf_score=torch.cat([outputs['pred_scores'],outputs['pred_scores']],dim=0)
        p=torch.sqrt(torch.sigmoid(src_logits)*conf_score).clamp(1e-6, 1 - 1e-6)
        src_logits=torch.log(p/(1-p))
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
                                                dtype=src_logits.dtype, layout=src_logits.layout,
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

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute periodic rotated L1 and paired rotated IoU losses.

        The ``loss_giou`` key is intentionally retained for checkpoint/config
        compatibility although it now contains the paired rotated IoU loss.
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

            pred_ref_norm = _normalize_rboxes(
                pred_ref, targets[batch_idx]['image_size_xyxy'])
            pred_cur_norm = _normalize_rboxes(
                pred_cur, targets[cur_idx]['image_size_xyxy'])
            tgt_ref_norm = targets[batch_idx]["boxes"][gt_multi_idx]
            tgt_cur_norm = targets[cur_idx]["boxes"][gt_multi_idx]

            for pred_norm, tgt_norm in ((pred_ref_norm, tgt_ref_norm),
                                        (pred_cur_norm, tgt_cur_norm)):
                coord = F.l1_loss(pred_norm[:, :4], tgt_norm[:, :4],
                                  reduction='none').sum()
                angle = (pred_norm[:, 4] - tgt_norm[:, 4]).abs()
                loss_bbox = loss_bbox + coord + torch.minimum(angle, 1 - angle).sum()

            pair_iou = pair_rotated_iou(
                pred_ref, tgt_ref, pred_cur, tgt_cur).diag()
            loss_riou = loss_riou + (1 - pair_iou).sum()
            matched += len(gt_multi_idx)

        normalizer = max(matched, 1)
        return {
            'loss_bbox': loss_bbox / (2 * normalizer),
            'loss_giou': loss_riou / normalizer,
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
        indices, _ = self.matcher(outputs_without_aux, targets)

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
                indices, _ = self.matcher(aux_outputs, targets)
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

        return losses


class HungarianMatcherDynamicK(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-k (dynamic) matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self,  cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, cost_mask: float = 1, use_focal: bool = False,use_fed_loss=False):
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
        self.ota_k = 5
        if self.use_focal:
            self.focal_loss_alpha = 0.25
            self.focal_loss_gamma = 2.0
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0,  "all costs cant be 0"

    def forward(self, outputs, targets):
        """ simOTA for detr"""
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            conf_score=outputs["pred_scores"]
            # We flatten to compute the cost matrices in a batch
            pred_logits_pre,pred_logits_curr=torch.split(outputs["pred_logits"],bs//2,dim=0)
            out_bbox_pre,out_bbox_curr = torch.split(outputs["pred_boxes"],bs//2,dim=0)
            if self.use_focal or self.use_fed_loss:
                out_prob_pre = torch.sqrt(pred_logits_pre.sigmoid()*conf_score)  # [batch_size, num_queries, num_classes]
                out_prob_curr = torch.sqrt(pred_logits_curr.sigmoid()*conf_score)
            else:
                out_prob_pre = torch.sqrt(pred_logits_pre.softmax(-1)*conf_score) # [batch_size, num_queries, num_classes]
                out_prob_curr=torch.sqrt(pred_logits_curr.softmax(-1)*conf_score)
            indices = []
            matched_ids = []
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
                    continue

                bz_gtboxs_pre = targets[batch_idx]['boxes']
                bz_gtboxs_abs_pre = targets[batch_idx]['boxes_abs']
                bz_gtboxs_curr = targets[batch_idx+bs//2]['boxes']
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

                pair_wise_ious = pair_rotated_iou(
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

                bz_out_bbox_pre = _normalize_rboxes(bz_boxes_pre, bz_image_size_out_pre)
                bz_out_bbox_curr = _normalize_rboxes(bz_boxes_curr, bz_image_size_out_curr)
                cost_bbox_pre = _periodic_l1_cost(bz_out_bbox_pre, bz_gtboxs_pre)
                cost_bbox_curr = _periodic_l1_cost(bz_out_bbox_curr, bz_gtboxs_curr)

                cost_giou = -pair_wise_ious

                # Final cost matrix
                cost = self.cost_bbox * (cost_bbox_pre+cost_bbox_curr)/2 + self.cost_class * cost_class/2 + self.cost_giou * cost_giou + 100.0 * (~is_in_boxes_and_center)
                # A single invalid diffusion proposal must not terminate all
                # DDP ranks. Exclude non-finite proposal rows from matching;
                # valid rows and their original costs are left untouched.
                valid_rows = (
                    torch.isfinite(bz_boxes_pre).all(dim=1)
                    & torch.isfinite(bz_boxes_curr).all(dim=1)
                    & torch.isfinite(bz_out_prob_pre).all(dim=1)
                    & torch.isfinite(bz_out_prob_curr).all(dim=1)
                )
                if not valid_rows.any():
                    raise FloatingPointError(
                        "all diffusion proposals are non-finite during matching")
                cost = torch.where(
                    torch.isfinite(cost), cost,
                    torch.full_like(cost, 1e8))
                cost[~valid_rows] = 1e8
                # cost = (cost_class + 3.0 * cost_giou + 100.0 * (~is_in_boxes_and_center))  # [num_query,num_gt]
                cost[~fg_mask] = cost[~fg_mask] + 10000.0

                # if bz_gtboxs.shape[0]>0:
                indices_batchi, matched_qidx = self.dynamic_k_matching(cost, pair_wise_ious, bz_gtboxs_pre.shape[0])

                indices.append(indices_batchi)
                matched_ids.append(matched_qidx)

        return indices, matched_ids

    def get_in_boxes_info(self, boxes, target_gts, expanded_strides):
        # Keep the original SimOTA center prior.  For rotated boxes its extent
        # is represented by the enclosing local cxcywh rectangle; exact rotated
        # overlap is still used by the actual matching cost.
        half_wh = target_gts[:, 2:4] / 2
        xy_target_gts = torch.cat(
            [target_gts[:, :2] - half_wh, target_gts[:, :2] + half_wh], dim=1)

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

        # Take the sum of the predicted value and the top 10 iou of gt with the largest iou as dynamic_k
        topk_ious, _ = torch.topk(
            ious_in_boxes_matrix, min(n_candidate_k, cost.shape[0]), dim=0)
        dynamic_ks = torch.clamp(
            topk_ious.sum(0).int(), min=1, max=cost.shape[0])

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(1)

        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1)
            matching_matrix[anchor_matching_gt > 1] *= 0
            matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1

        while (matching_matrix.sum(0) == 0).any():
            num_zero_gt = (matching_matrix.sum(0) == 0).sum()
            matched_query_id = matching_matrix.sum(1) > 0
            cost[matched_query_id] += 100000.0
            unmatch_id = torch.nonzero(matching_matrix.sum(0) == 0, as_tuple=False).squeeze(1)
            for gt_idx in unmatch_id:
                pos_idx = torch.argmin(cost[:, gt_idx])
                matching_matrix[:, gt_idx][pos_idx] = 1.0
            if (matching_matrix.sum(1) > 1).sum() > 0:  # If a query matches more than one gt
                _, cost_argmin = torch.min(cost[anchor_matching_gt > 1],
                                           dim=1)  # find gt for these queries with minimal cost
                matching_matrix[anchor_matching_gt > 1] *= 0  # reset mapping relationship
                matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1  # keep gt with minimal cost

        assert not (matching_matrix.sum(0) == 0).any()
        selected_query = matching_matrix.sum(1) > 0
        gt_indices = matching_matrix[selected_query].max(1)[1]
        assert selected_query.sum() == len(gt_indices)

        cost[matching_matrix == 0] = cost[matching_matrix == 0] + float('inf')
        matched_query_id = torch.min(cost, dim=0)[1]

        return (selected_query, gt_indices), matched_query_id
