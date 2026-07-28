#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import csv
import os

import cv2
import numpy as np

__all__ = ["vis", "save_hsmot_pair_visualization",
           "save_diffusion_train_diagnostic",
           "save_train_feature_visualization"]


def _multispectral_to_bgr(image):
    if hasattr(image, "detach"):
        image = image.detach().float().cpu().numpy()
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError("expected a 3-D image, got {}".format(image.shape))
    if image.shape[0] <= 16 and image.shape[0] < image.shape[-1]:
        image = image.transpose(1, 2, 0)
    if image.shape[2] < 3:
        raise ValueError("expected at least three channels")
    rgb = image[:, :, :3].astype(np.float32)
    if rgb.size and float(np.nanmax(rgb)) <= 1.5:
        rgb *= 255.0
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
    return np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8)[:, :, ::-1])


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def _feature_heatmap(value, width=360, height=240):
    value = np.nan_to_num(
        _to_numpy(value).astype(np.float32),
        nan=0.0, posinf=0.0, neginf=0.0)
    finite = value[np.isfinite(value)]
    if finite.size:
        low, high = np.percentile(finite, (1.0, 99.0))
    else:
        low, high = 0.0, 0.0
    if high <= low + 1e-12:
        normalized = np.zeros_like(value, dtype=np.uint8)
    else:
        normalized = np.clip(
            (value - low) * (255.0 / (high - low)), 0, 255
        ).astype(np.uint8)
    normalized = cv2.resize(
        normalized, (width, height), interpolation=cv2.INTER_LINEAR)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def save_train_feature_visualization(path, snapshots, title=""):
    """Save REF/CUR P3-P5 mean/max activation maps from one train batch."""
    if not snapshots:
        raise ValueError("feature snapshots are empty")
    by_key = {
        (item["side"], item["level"]): item for item in snapshots
    }
    rows = []
    for side in ("REF", "CUR"):
        for statistic, label in (
                ("mean_abs", "channel mean |x|"),
                ("max_abs", "channel max |x|")):
            cells = []
            for level in ("P3", "P4", "P5"):
                item = by_key[(side, level)]
                cell = _feature_heatmap(item[statistic])
                header = np.zeros((58, cell.shape[1], 3), dtype=np.uint8)
                shape = "x".join(str(value) for value in item["shape"])
                cv2.putText(
                    header, "{} {} {}".format(side, level, label),
                    (7, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(
                    header,
                    "{} mean={:.3g} std={:.3g} max={:.3g}".format(
                        shape, item["mean"], item["std"], item["absmax"]),
                    (7, 44), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, (210, 210, 210), 1, cv2.LINE_AA)
                cells.append(np.concatenate((header, cell), axis=0))
            rows.append(np.concatenate(cells, axis=1))
    canvas = np.concatenate(rows, axis=0)
    if title:
        header = np.zeros((34, canvas.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            header, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX,
            0.58, (255, 255, 255), 1, cv2.LINE_AA)
        canvas = np.concatenate((header, canvas), axis=0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not cv2.imwrite(path, canvas):
        raise IOError("failed to write {}".format(path))
    return path


def _draw_qboxes(image, qboxes, classes, scores=None, prefix="GT",
                 color=(0, 255, 0), class_names=None):
    qboxes = _to_numpy(qboxes).reshape(-1, 8)
    classes = _to_numpy(classes)
    scores = None if scores is None else _to_numpy(scores)
    for index, qbox in enumerate(qboxes):
        class_id = int(classes[index])
        points = np.rint(qbox.reshape(4, 2)).astype(np.int32)
        cv2.polylines(image, [points], True, color, 2, cv2.LINE_AA)
        anchor = points[np.argmin(points[:, 1])].copy()
        anchor[0] = np.clip(anchor[0], 0, max(image.shape[1] - 1, 0))
        anchor[1] = np.clip(anchor[1] - 3, 15, max(image.shape[0] - 1, 15))
        name = (class_names[class_id]
                if class_names is not None and 0 <= class_id < len(class_names)
                else str(class_id))
        text = "{} {}".format(prefix, name)
        if scores is not None:
            text += " {:.2f}".format(float(scores[index]))
        cv2.putText(image, text, tuple(anchor), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA)
    return image


def _render_hsmot_sample(image, targets, pred_qboxes=None,
                         pred_classes=None, pred_scores=None,
                         class_names=None, title=""):
    canvas = _multispectral_to_bgr(image)
    targets = _to_numpy(targets)
    if targets.ndim == 2 and targets.shape[1] >= 9:
        valid = np.abs(targets[:, 1:9]).sum(axis=1) > 0
        targets = targets[valid]
        if len(targets):
            _draw_qboxes(canvas, targets[:, 1:9], targets[:, 0],
                         prefix="GT", color=(0, 255, 0),
                         class_names=class_names)
    if pred_qboxes is not None and len(pred_qboxes):
        _draw_qboxes(canvas, pred_qboxes, pred_classes, pred_scores,
                     prefix="PRED", color=(0, 80, 255),
                     class_names=class_names)
    if title:
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(canvas, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def save_hsmot_pair_visualization(path, ref_image, ref_targets,
                                  cur_image=None, cur_targets=None,
                                  pred_qboxes=None, pred_classes=None,
                                  pred_scores=None, class_names=None,
                                  title=""):
    """Save post-preprocessing HSMOT GT, optionally with CUR predictions."""
    ref_title = "{} REF".format(title).strip() if cur_image is not None else title
    ref = _render_hsmot_sample(
        ref_image, ref_targets, class_names=class_names, title=ref_title)
    if cur_image is not None:
        cur_title = "{} CUR".format(title).strip()
        cur = _render_hsmot_sample(
            cur_image, cur_targets, pred_qboxes, pred_classes, pred_scores,
            class_names, cur_title)
        ref = np.concatenate((ref, cur), axis=1)
    elif pred_qboxes is not None:
        ref = _render_hsmot_sample(
            ref_image, ref_targets, pred_qboxes, pred_classes, pred_scores,
            class_names, title)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not cv2.imwrite(path, ref):
        raise IOError("failed to write visualization {}".format(path))
    return path


def _draw_plain_rboxes(image, rboxes, color, thickness=1,
                       labels=None):
    """Draw LE135 boxes without the dense class labels used by AP views."""
    if rboxes is None or len(rboxes) == 0:
        return image
    import torch
    from utils.box_ops import box_cxcywhtheta_to_xyxyxyxy
    qboxes = box_cxcywhtheta_to_xyxyxyxy(
        torch.as_tensor(_to_numpy(rboxes), dtype=torch.float32)
    ).cpu().numpy().reshape(-1, 4, 2)
    for index, qbox in enumerate(qboxes):
        points = np.rint(qbox).astype(np.int32)
        cv2.polylines(image, [points], True, color, thickness, cv2.LINE_AA)
        if labels is not None:
            anchor = points[np.argmin(points[:, 1])].copy()
            anchor[0] = np.clip(anchor[0], 0, max(image.shape[1] - 1, 0))
            anchor[1] = np.clip(anchor[1] - 2, 12,
                                max(image.shape[0] - 1, 12))
            cv2.putText(image, str(labels[index]), tuple(anchor),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1,
                        cv2.LINE_AA)
    return image


def _diagnostic_panel(image, gt_boxes, gt_classes, title,
                      proposal_boxes=None, matched_boxes=None,
                      matched_labels=None, class_names=None):
    canvas = _multispectral_to_bgr(image)
    if proposal_boxes is not None and len(proposal_boxes):
        overlay = canvas.copy()
        _draw_plain_rboxes(overlay, proposal_boxes, (0, 90, 255), 1)
        canvas = cv2.addWeighted(canvas, 0.55, overlay, 0.45, 0)
    if len(gt_boxes):
        gt_labels = []
        for gt_index, class_id in enumerate(_to_numpy(gt_classes)):
            class_id = int(class_id)
            class_name = (
                class_names[class_id]
                if class_names is not None
                and 0 <= class_id < len(class_names)
                else str(class_id))
            gt_labels.append('g{}:{}'.format(gt_index, class_name))
        _draw_plain_rboxes(
            canvas, gt_boxes, (0, 255, 0), 2, gt_labels)
    if matched_boxes is not None and len(matched_boxes):
        _draw_plain_rboxes(canvas, matched_boxes, (255, 255, 0), 2,
                           matched_labels)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(canvas, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _assignment_movement_panel(image, gt_boxes, gt_classes, before_boxes,
                               after_boxes, query_indices, gt_indices, title,
                               class_names=None):
    """Overlay matched-query geometry before and after one refinement layer."""
    canvas = _multispectral_to_bgr(image)
    query_indices = np.asarray(query_indices, dtype=np.int64)
    gt_indices = np.asarray(gt_indices, dtype=np.int64)
    before = np.asarray(before_boxes)[query_indices]
    after = np.asarray(after_boxes)[query_indices]
    if len(gt_boxes):
        _draw_plain_rboxes(canvas, gt_boxes, (0, 255, 0), 2)
    if len(before):
        _draw_plain_rboxes(canvas, before, (255, 80, 0), 2)
        labels = [
            'q{}g{}'.format(int(query), int(gt))
            for query, gt in zip(query_indices, gt_indices)
        ]
        _draw_plain_rboxes(canvas, after, (0, 255, 255), 2, labels)
        for old_box, new_box in zip(before, after):
            old_center = tuple(np.rint(old_box[:2]).astype(np.int32))
            new_center = tuple(np.rint(new_box[:2]).astype(np.int32))
            cv2.arrowedLine(
                canvas, old_center, new_center, (255, 255, 255), 2,
                cv2.LINE_AA, tipLength=0.25)
        center_move = np.linalg.norm(after[:, :2] - before[:, :2], axis=1)
        scale_move = np.abs(
            np.log(np.maximum(after[:, 2:4], 1e-6)
                   / np.maximum(before[:, 2:4], 1e-6))).mean(axis=1)
        angle_move = np.abs(
            (after[:, 4] - before[:, 4] + np.pi / 2) % np.pi
            - np.pi / 2) * 180.0 / np.pi
        summary = (
            'n={} center px mean/max={:.1f}/{:.1f} '
            '|log wh|={:.3f} angle deg={:.1f}'.format(
                len(before), float(center_move.mean()),
                float(center_move.max()), float(scale_move.mean()),
                float(angle_move.mean())))
    else:
        summary = 'n=0'
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 52), (0, 0, 0), -1)
    cv2.putText(canvas, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        canvas, 'blue=before yellow=after green=GT; ' + summary,
        (8, 43), cv2.FONT_HERSHEY_SIMPLEX,
        0.40, (230, 230, 230), 1, cv2.LINE_AA)
    return canvas


def _tile_diagnostic_panels(panels, columns=4, panel_width=480):
    resized = []
    for panel in panels:
        scale = panel_width / panel.shape[1]
        resized.append(cv2.resize(
            panel, (panel_width, int(round(panel.shape[0] * scale))),
            interpolation=cv2.INTER_AREA))
    panel_height = max(panel.shape[0] for panel in resized)
    rows = (len(resized) + columns - 1) // columns
    blank = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    while len(resized) < rows * columns:
        resized.append(blank.copy())
    row_images = []
    for row in range(rows):
        cells = []
        for panel in resized[row * columns:(row + 1) * columns]:
            if panel.shape[0] < panel_height:
                panel = cv2.copyMakeBorder(
                    panel, 0, panel_height - panel.shape[0], 0, 0,
                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
            cells.append(panel)
        row_images.append(np.concatenate(cells, axis=1))
    return np.concatenate(row_images, axis=0)


def _match_value(assignment, key, index):
    values = _to_numpy(
        assignment.get(key, np.zeros(
            len(assignment.get('query_indices', []))))).reshape(-1)
    return float(values[index])


def _match_detail_table(assignment, layer, height, width=700):
    """Render exact selected matcher costs and local weighted losses."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    queries = _to_numpy(assignment['query_indices']).astype(np.int64)
    gt_indices = _to_numpy(assignment['gt_indices']).astype(np.int64)
    lines = [
        'layer {} selected matches: {}'.format(layer, len(queries)),
        'cost = total [weighted cls / L1 / (-IoU) + priors]',
        'loss = weighted matched-query cls / pair L1 / pair RIoU',
    ]
    for index, (query, gt_index) in enumerate(zip(queries, gt_indices)):
        lines.append(
            'q{:03d}->g{:02d} cost {:8.3f} [{:6.3f}/{:6.3f}/{:6.3f}'
            ' +{:5.1f}+{:7.1f}] loss [{:6.3f}/{:6.3f}/{:6.3f}]'.
            format(
                int(query), int(gt_index),
                _match_value(assignment, 'match_cost_total', index),
                _match_value(
                    assignment, 'match_cost_weighted_class', index),
                _match_value(assignment, 'match_cost_weighted_l1', index),
                _match_value(
                    assignment, 'match_cost_weighted_riou', index),
                _match_value(
                    assignment, 'match_cost_center_penalty', index),
                _match_value(assignment, 'match_cost_fg_penalty', index),
                _match_value(
                    assignment,
                    'loss_cls_matched_query_weighted', index),
                _match_value(assignment, 'loss_l1_weighted', index),
                _match_value(assignment, 'loss_riou_weighted', index)))
    line_height = max(15, min(25, (height - 20) // max(len(lines), 1)))
    font_scale = 0.42 if line_height >= 18 else 0.34
    for line_index, line in enumerate(lines):
        y = 18 + line_index * line_height
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (230, 230, 230), 1, cv2.LINE_AA)
    return canvas


def save_diffusion_train_diagnostic(output_dir, stem, image, debug,
                                    class_names=None, max_proposals=60,
                                    save_snapshot=True):
    """Save GT -> diffusion -> six refinements -> assignment diagnostics.

    The function consumes detached CPU tensors captured by ``DiffusionHead``;
    it is never called by the formal experiment unless explicitly enabled.
    """
    os.makedirs(output_dir, exist_ok=True)
    timesteps = _to_numpy(debug['timesteps']).astype(np.int64)
    initial_boxes = _to_numpy(debug['initial_boxes'])
    stage_boxes = _to_numpy(debug['stage_boxes'])
    stage_logits = _to_numpy(debug['stage_logits'])
    stage_scores = _to_numpy(debug['stage_scores'])
    gt_boxes_all = [_to_numpy(value) for value in debug['gt_boxes']]
    gt_classes_all = [_to_numpy(value).astype(np.int64)
                      for value in debug['gt_classes']]
    assignments = debug.get('assignments') or []
    pair_batch = initial_boxes.shape[0] // 2
    images = _to_numpy(image)
    if images.ndim == 3:
        images = images[None]
    if images.ndim != 4:
        raise ValueError(
            'expected one image or a batch of images, got {}'.format(
                images.shape))
    first_image = images[0]
    saved = []

    archive = {
        'timesteps': timesteps,
        'initial_boxes': initial_boxes,
        'stage_boxes': stage_boxes,
        'stage_logits': stage_logits,
        'stage_scores': stage_scores,
    }
    for side_index in range(initial_boxes.shape[0]):
        archive['gt_boxes_{}'.format(side_index)] = gt_boxes_all[side_index]
        archive['gt_classes_{}'.format(side_index)] = gt_classes_all[side_index]
    for stage_index, stage_assignment in enumerate(assignments):
        for pair_index, assignment in enumerate(stage_assignment):
            prefix = 'assignment_s{}_p{}'.format(stage_index, pair_index)
            for key, value in assignment.items():
                archive[prefix + '_' + key] = _to_numpy(value)
    if save_snapshot:
        archive_path = os.path.join(output_dir, stem + '_snapshot.npz')
        np.savez_compressed(archive_path, **archive)
        saved.append(archive_path)

    csv_fields = [
        'layer', 'pair', 'query', 'gt',
        'match_cost_class', 'match_cost_l1_ref', 'match_cost_l1_cur',
        'match_cost_l1_pair', 'match_cost_pair_iou', 'match_cost_riou',
        'match_cost_weighted_class', 'match_cost_weighted_l1',
        'match_cost_weighted_riou', 'match_cost_center_penalty',
        'match_cost_fg_penalty', 'match_cost_total',
        'loss_cls_matched_query', 'loss_cls_matched_query_weighted',
        'loss_l1_ref', 'loss_l1_cur', 'loss_l1_pair',
        'loss_l1_weighted', 'loss_riou_ref', 'loss_riou_cur',
        'loss_riou_pair', 'loss_riou_weighted',
    ]
    csv_path = os.path.join(output_dir, stem + '_matches.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for stage_index, stage_assignment in enumerate(assignments):
            for pair_index, assignment in enumerate(stage_assignment):
                queries = _to_numpy(
                    assignment['query_indices']).astype(np.int64)
                gt_indices = _to_numpy(
                    assignment['gt_indices']).astype(np.int64)
                for match_index, (query, gt_index) in enumerate(
                        zip(queries, gt_indices)):
                    row = {
                        'layer': stage_index + 1,
                        'pair': pair_index,
                        'query': int(query),
                        'gt': int(gt_index),
                    }
                    for field in csv_fields[4:]:
                        if field not in assignment:
                            continue
                        values = _to_numpy(assignment[field]).reshape(-1)
                        row[field] = float(values[match_index])
                    writer.writerow(row)
    saved.append(csv_path)

    for side_index, side_name in ((0, 'ref'), (pair_batch, 'cur')):
        pair_index = side_index % pair_batch
        gt_boxes = gt_boxes_all[side_index]
        gt_classes = gt_classes_all[side_index]
        panels = [_diagnostic_panel(
            first_image, gt_boxes, gt_classes,
            '{} GT n={}'.format(side_name.upper(), len(gt_boxes)),
            class_names=class_names)]
        initial = initial_boxes[side_index, :max_proposals]
        panels.append(_diagnostic_panel(
            first_image, gt_boxes, gt_classes,
            '{} diffusion t={} shown={}/{}'.format(
                side_name.upper(), int(timesteps[side_index]), len(initial),
                initial_boxes.shape[1]),
            proposal_boxes=initial, class_names=class_names))

        assignment_panels = []
        for stage_index in range(stage_boxes.shape[0]):
            logits = stage_logits[stage_index, side_index]
            association = stage_scores[stage_index, pair_index].reshape(-1)
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(
                logits, -30.0, 30.0)))
            classes = probabilities.argmax(axis=1)
            scores = np.sqrt(
                probabilities.max(axis=1) * np.clip(association, 0, 1))
            top = np.argsort(-scores)[:max_proposals]
            assignment = (assignments[stage_index][pair_index]
                          if stage_index < len(assignments) else None)
            query_indices = (np.empty(0, dtype=np.int64)
                             if assignment is None else _to_numpy(
                                 assignment['query_indices']).astype(np.int64))
            gt_indices = (np.empty(0, dtype=np.int64)
                          if assignment is None else _to_numpy(
                              assignment['gt_indices']).astype(np.int64))
            matched_labels = [
                'q{}g{} c{:.1f} l{:.2f}'.format(
                    int(query), int(gt),
                    float(_to_numpy(
                        assignment.get(
                            'match_cost_total',
                            np.zeros(len(query_indices))))[match_index]),
                    float(_to_numpy(
                        assignment.get(
                            'loss_l1_weighted',
                            np.zeros(len(query_indices))))[match_index]))
                for match_index, (query, gt) in enumerate(
                    zip(query_indices, gt_indices))]
            counts = (np.bincount(gt_indices, minlength=len(gt_boxes))
                      if len(gt_boxes) else np.empty(0, dtype=np.int64))
            title = '{} layer{} top={} pos={} perGT={}'.format(
                side_name.upper(), stage_index + 1, len(top),
                len(query_indices),
                '/'.join(map(str, counts.tolist())))
            panels.append(_diagnostic_panel(
                first_image, gt_boxes, gt_classes, title,
                proposal_boxes=stage_boxes[stage_index, side_index, top],
                matched_boxes=stage_boxes[
                    stage_index, side_index, query_indices],
                class_names=class_names))
            assignment_panels.append(_diagnostic_panel(
                first_image, gt_boxes, gt_classes,
                '{} match layer{} pos={} perGT={}'.format(
                    side_name.upper(), stage_index + 1, len(query_indices),
                    '/'.join(map(str, counts.tolist()))),
                matched_boxes=stage_boxes[
                    stage_index, side_index, query_indices],
                matched_labels=matched_labels,
                class_names=class_names))

        pipeline_path = os.path.join(
            output_dir, '{}_{}_pipeline.jpg'.format(stem, side_name))
        assignment_path = os.path.join(
            output_dir, '{}_{}_assignments.jpg'.format(stem, side_name))
        if not cv2.imwrite(pipeline_path, _tile_diagnostic_panels(panels, 4)):
            raise IOError('failed to write {}'.format(pipeline_path))
        if not cv2.imwrite(
                assignment_path,
                _tile_diagnostic_panels(assignment_panels, 3)):
            raise IOError('failed to write {}'.format(assignment_path))
        saved.extend([pipeline_path, assignment_path])

    # One readable full-resolution page per refinement layer.  The image
    # shows the first batch sample's REF/CUR matched boxes, while the table
    # prints every selected query's exact matcher cost decomposition and its
    # local weighted classification/L1/rotated-IoU loss.  The CSV above keeps
    # the same values for every sample in the physical batch.
    if pair_batch and assignments:
        for pair_index in range(pair_batch):
            ref_index = pair_index
            cur_index = pair_index + pair_batch
            pair_image = images[min(pair_index, len(images) - 1)]
            for stage_index in range(
                    min(stage_boxes.shape[0], len(assignments))):
                assignment = assignments[stage_index][pair_index]
                query_indices = _to_numpy(
                    assignment['query_indices']).astype(np.int64)
                gt_indices = _to_numpy(
                    assignment['gt_indices']).astype(np.int64)
                labels = [
                    'q{}g{}'.format(int(query), int(gt))
                    for query, gt in zip(query_indices, gt_indices)]
                ref_panel = _diagnostic_panel(
                    pair_image, gt_boxes_all[ref_index],
                    gt_classes_all[ref_index],
                    'pair{} REF match layer{}'.format(
                        pair_index, stage_index + 1),
                    matched_boxes=stage_boxes[
                        stage_index, ref_index, query_indices],
                    matched_labels=labels, class_names=class_names)
                cur_panel = _diagnostic_panel(
                    pair_image, gt_boxes_all[cur_index],
                    gt_classes_all[cur_index],
                    'pair{} CUR match layer{}'.format(
                        pair_index, stage_index + 1),
                    matched_boxes=stage_boxes[
                        stage_index, cur_index, query_indices],
                    matched_labels=labels, class_names=class_names)
                table = _match_detail_table(
                    assignment, stage_index + 1, ref_panel.shape[0])
                detail = np.concatenate(
                    (ref_panel, cur_panel, table), axis=1)
                detail_path = os.path.join(
                    output_dir,
                    '{}_pair{}_layer{}_match_details.jpg'.format(
                        stem, pair_index, stage_index + 1))
                if not cv2.imwrite(detail_path, detail):
                    raise IOError('failed to write {}'.format(detail_path))
                saved.append(detail_path)

                before_boxes = (
                    initial_boxes
                    if stage_index == 0 else stage_boxes[stage_index - 1])
                ref_movement = _assignment_movement_panel(
                    pair_image, gt_boxes_all[ref_index],
                    gt_classes_all[ref_index],
                    before_boxes[ref_index],
                    stage_boxes[stage_index, ref_index],
                    query_indices, gt_indices,
                    'pair{} REF before layer{} -> after layer{}'.format(
                        pair_index, stage_index + 1, stage_index + 1),
                    class_names=class_names)
                cur_movement = _assignment_movement_panel(
                    pair_image, gt_boxes_all[cur_index],
                    gt_classes_all[cur_index],
                    before_boxes[cur_index],
                    stage_boxes[stage_index, cur_index],
                    query_indices, gt_indices,
                    'pair{} CUR before layer{} -> after layer{}'.format(
                        pair_index, stage_index + 1, stage_index + 1),
                    class_names=class_names)
                movement = np.concatenate(
                    (ref_movement, cur_movement), axis=1)
                movement_path = os.path.join(
                    output_dir,
                    '{}_pair{}_layer{}_assign_movement.jpg'.format(
                        stem, pair_index, stage_index + 1))
                if not cv2.imwrite(movement_path, movement):
                    raise IOError('failed to write {}'.format(movement_path))
                saved.append(movement_path)
    return saved


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    #text_scale = max(1, image.shape[1] / 1600.)
    #text_thickness = 2
    #line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
