#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os

import cv2
import numpy as np

__all__ = ["vis", "save_hsmot_pair_visualization",
           "save_diffusion_train_diagnostic"]


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
    from .rotated_boxes import rbox_to_qbox
    qboxes = rbox_to_qbox(_to_numpy(rboxes)).reshape(-1, 4, 2)
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
    from .rotated_boxes import rbox_to_qbox
    canvas = _multispectral_to_bgr(image)
    if proposal_boxes is not None and len(proposal_boxes):
        overlay = canvas.copy()
        _draw_plain_rboxes(overlay, proposal_boxes, (0, 90, 255), 1)
        canvas = cv2.addWeighted(canvas, 0.55, overlay, 0.45, 0)
    if len(gt_boxes):
        _draw_qboxes(canvas, rbox_to_qbox(_to_numpy(gt_boxes)), gt_classes,
                     prefix="GT", color=(0, 255, 0),
                     class_names=class_names)
    if matched_boxes is not None and len(matched_boxes):
        _draw_plain_rboxes(canvas, matched_boxes, (255, 255, 0), 2,
                           matched_labels)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(canvas, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (255, 255, 255), 1, cv2.LINE_AA)
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


def save_diffusion_train_diagnostic(output_dir, stem, image, debug,
                                    class_names=None, max_proposals=60):
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
            archive[prefix + '_queries'] = _to_numpy(
                assignment['query_indices']).astype(np.int64)
            archive[prefix + '_gt'] = _to_numpy(
                assignment['gt_indices']).astype(np.int64)
            archive[prefix + '_best'] = _to_numpy(
                assignment['best_query_per_gt']).astype(np.int64)
    archive_path = os.path.join(output_dir, stem + '_snapshot.npz')
    np.savez_compressed(archive_path, **archive)
    saved.append(archive_path)

    for side_index, side_name in ((0, 'ref'), (pair_batch, 'cur')):
        pair_index = side_index % pair_batch
        gt_boxes = gt_boxes_all[side_index]
        gt_classes = gt_classes_all[side_index]
        panels = [_diagnostic_panel(
            image, gt_boxes, gt_classes,
            '{} GT n={}'.format(side_name.upper(), len(gt_boxes)),
            class_names=class_names)]
        initial = initial_boxes[side_index, :max_proposals]
        panels.append(_diagnostic_panel(
            image, gt_boxes, gt_classes,
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
                'q{}->g{}'.format(int(query), int(gt))
                for query, gt in zip(query_indices, gt_indices)]
            counts = (np.bincount(gt_indices, minlength=len(gt_boxes))
                      if len(gt_boxes) else np.empty(0, dtype=np.int64))
            title = '{} refine{} top={} pos={} perGT={}'.format(
                side_name.upper(), stage_index, len(top), len(query_indices),
                '/'.join(map(str, counts.tolist())))
            panels.append(_diagnostic_panel(
                image, gt_boxes, gt_classes, title,
                proposal_boxes=stage_boxes[stage_index, side_index, top],
                matched_boxes=stage_boxes[
                    stage_index, side_index, query_indices],
                class_names=class_names))
            assignment_panels.append(_diagnostic_panel(
                image, gt_boxes, gt_classes,
                '{} assignment{} pos={} perGT={}'.format(
                    side_name.upper(), stage_index, len(query_indices),
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
