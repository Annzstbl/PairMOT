import os.path as osp
from typing import Optional, Sequence

import mmcv
from mmengine.fileio import get
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist
from mmdet.engine.hooks import DetVisualizationHook
from mmdet.structures import DetDataSample

from mmrotate.registry import HOOKS


def get_hsmot_seq_name(data_sample: DetDataSample) -> str:
    """Resolve HSMOT sequence name from metainfo or image path."""
    seq_name = data_sample.metainfo.get('seq_name', None)
    if seq_name:
        return str(seq_name)
    img_path = data_sample.img_path
    if not img_path:
        return 'unknown'
    return osp.basename(osp.dirname(img_path))


@HOOKS.register_module()
class HSMOTVisualizationHook(DetVisualizationHook):
    """Detection visualization hook that groups outputs by HSMOT sequence."""

    def _init_test_out_dir(self, runner: Runner) -> None:
        if self.test_out_dir is None:
            return
        if getattr(self, '_test_out_dir_inited', False):
            return
        self.test_out_dir = osp.join(runner.work_dir, runner.timestamp,
                                     self.test_out_dir)
        mkdir_or_exist(self.test_out_dir)
        self._test_out_dir_inited = True

    def _get_out_file(self, data_sample: DetDataSample) -> Optional[str]:
        if self.test_out_dir is None:
            return None
        seq_dir = osp.join(self.test_out_dir, get_hsmot_seq_name(data_sample))
        mkdir_or_exist(seq_dir)
        return osp.join(seq_dir, osp.basename(data_sample.img_path))

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[DetDataSample]) -> None:
        if self.draw is False:
            return

        self._init_test_out_dir(runner)

        for data_sample in outputs:
            self._test_index += 1

            img_path = data_sample.img_path
            img_bytes = get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')

            self._visualizer.add_datasample(
                osp.basename(img_path) if self.show else 'test_img',
                img,
                data_sample=data_sample,
                show=self.show,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
                out_file=self._get_out_file(data_sample),
                step=self._test_index)
