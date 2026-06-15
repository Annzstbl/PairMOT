# Copyright (c) AI4RS. All rights reserved.
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import List, Optional, Union

import numpy as np
from mmengine.logging import MMLogger, print_log
from terminaltables import AsciiTable

from mmrotate.evaluation import eval_rbbox_map_multi_iou
from mmrotate.registry import METRICS

from .dota_metric import DOTAMetric

HSMOT_IOU_THRS = [
    0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
]


def _round_metric(value: float) -> float:
    return round(float(value), 3)


def _extract_class_ap(cls_result: dict) -> float:
    ap = cls_result['ap']
    if isinstance(ap, np.ndarray):
        ap = ap[0]
    return float(ap)


def _extract_class_recall(cls_result: dict) -> float:
    recall = cls_result['recall']
    if recall.size == 0:
        return 0.0
    return float(np.array(recall, ndmin=2)[0, -1])


def _append_classwise_metrics(
        eval_results: OrderedDict,
        dataset_name: Union[List[str], tuple],
        cls_results: list) -> None:
    recalls = []
    for cls_name, cls_result in zip(dataset_name, cls_results):
        ap = _extract_class_ap(cls_result)
        recall = _extract_class_recall(cls_result)
        recalls.append(recall)
        eval_results[f'{cls_name}_AP'] = _round_metric(ap)
        eval_results[f'{cls_name}_recall'] = _round_metric(recall)
    if recalls:
        eval_results['mean_recall'] = _round_metric(float(np.mean(recalls)))


def _iou_thr_index(iou_thrs: list, target_thr: float) -> Optional[int]:
    for idx, iou_thr in enumerate(iou_thrs):
        if abs(iou_thr - target_thr) < 1e-6:
            return idx
    return None


def _scalar_ap(ap) -> float:
    if isinstance(ap, np.ndarray):
        ap = ap[0]
    return float(ap)


def _format_num_gts(num_gts) -> int:
    if isinstance(num_gts, np.ndarray):
        return int(num_gts[0])
    return int(num_gts)


def _print_hsmot_summary_table(mean_aps: list,
                               cls_results_per_thr: list,
                               iou_thrs: list,
                               dataset_name: Union[List[str], tuple],
                               logger: MMLogger) -> None:
    """Print consolidated AP50 / AP75 / mAP50:95 summary table."""
    idx50 = _iou_thr_index(iou_thrs, 0.5)
    idx75 = _iou_thr_index(iou_thrs, 0.75)
    if idx50 is None or idx75 is None:
        return

    cls_results_50 = cls_results_per_thr[idx50]
    num_classes = len(cls_results_50)
    per_class_map = []
    for cls_idx in range(num_classes):
        cls_aps = [
            _scalar_ap(cls_results_per_thr[thr_idx][cls_idx]['ap'])
            for thr_idx in range(len(iou_thrs))
        ]
        per_class_map.append(float(np.mean(cls_aps)))

    ap50 = _scalar_ap(mean_aps[idx50])
    ap75 = _scalar_ap(mean_aps[idx75])
    map50_95 = float(np.mean([_scalar_ap(mean_ap) for mean_ap in mean_aps]))

    header = ['class', 'gts', 'dets', 'recall', 'AP50', 'AP75', 'mAP50:95']
    table_data = [header]
    for cls_idx, cls_name in enumerate(dataset_name):
        cls50 = cls_results_50[cls_idx]
        cls75 = cls_results_per_thr[idx75][cls_idx]
        table_data.append([
            cls_name,
            _format_num_gts(cls50['num_gts']),
            cls50['num_dets'],
            f'{_extract_class_recall(cls50):.5f}',
            f'{_scalar_ap(cls50["ap"]):.5f}',
            f'{_scalar_ap(cls75["ap"]):.5f}',
            f'{per_class_map[cls_idx]:.5f}',
        ])

    table_data.append([
        'mAP', '', '', '', f'{ap50:.5f}', f'{ap75:.5f}', f'{map50_95:.5f}'
    ])
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    print_log('\n' + '-' * 15 + 'HSMOT summary' + '-' * 15, logger=logger)
    print_log(table.table, logger=logger)


@METRICS.register_module()
class HSMOTDetMetric(DOTAMetric):
    """HSMOT detection metric with rich validation outputs.

    Compared with :class:`DOTAMetric`, this metric additionally reports:

    - mAP averaged over multiple IoU thresholds (default 0.50:0.95)
    - AP at each IoU threshold (AP50, AP55, ..., AP95)
    - Per-class AP / recall at ``classwise_iou_thr`` (default 0.5)
    - Mean recall across classes at ``classwise_iou_thr``

    Args:
        iou_thrs (float or List[float]): IoU thresholds for mAP. Defaults to
            ``HSMOT_IOU_THRS``.
        classwise (bool): Whether to return per-class AP/recall.
            Defaults to True.
        classwise_iou_thr (float): IoU threshold for per-class metrics.
            Defaults to 0.5.
    """

    default_prefix: Optional[str] = 'hsmot'

    def __init__(self,
                 iou_thrs: Union[float, List[float], None] = None,
                 classwise: bool = True,
                 classwise_iou_thr: float = 0.5,
                 **kwargs) -> None:
        if iou_thrs is None:
            iou_thrs = list(HSMOT_IOU_THRS)
        super().__init__(iou_thrs=iou_thrs, **kwargs)
        self.classwise = classwise
        self.classwise_iou_thr = classwise_iou_thr

    def compute_metrics(self, results: list) -> dict:
        logger: MMLogger = MMLogger.get_current_instance()
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        eval_results = OrderedDict()
        if self.merge_patches:
            zip_path = self.merge_results(preds, outfile_prefix)
            logger.info(f'The submission file save at {zip_path}')
            return eval_results

        self.results2json(preds, outfile_prefix)
        if self.format_only:
            logger.info(
                f'results are saved in {osp.dirname(outfile_prefix)}')
            return eval_results

        if self.metric != 'mAP':
            raise NotImplementedError

        dataset_name = self.dataset_meta['classes']
        dets = [pred['pred_bbox_scores'] for pred in preds]

        mean_aps, cls_results_per_thr = eval_rbbox_map_multi_iou(
            dets,
            gts,
            iou_thrs=self.iou_thrs,
            scale_ranges=self.scale_ranges,
            use_07_metric=self.use_07_metric,
            box_type=self.predict_box_type,
            dataset=dataset_name,
            logger=logger,
            print_per_thr_summary=False)
        classwise_done = False
        for iou_thr, mean_ap, cls_results in zip(self.iou_thrs, mean_aps,
                                                 cls_results_per_thr):
            eval_results[f'AP{int(iou_thr * 100):02d}'] = _round_metric(
                mean_ap)

            if (self.classwise and not classwise_done
                    and abs(iou_thr - self.classwise_iou_thr) < 1e-6):
                _append_classwise_metrics(
                    eval_results, dataset_name, cls_results)
                classwise_done = True

        eval_results['mAP'] = _round_metric(sum(mean_aps) / len(mean_aps))
        eval_results.move_to_end('mAP', last=False)
        _print_hsmot_summary_table(
            mean_aps,
            cls_results_per_thr,
            self.iou_thrs,
            dataset_name,
            logger=logger)
        return eval_results
