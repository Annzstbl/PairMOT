
"""Run TrackEval for VT-Tiny-MOT (axis-aligned boxes).

Example:
python scripts/run_vt_tiny_mot.py \
  --USE_PARALLEL False \
  --METRICS HOTA CLEAR Identity \
  --GT_COCO_ANN /path/to/VT-Tiny-MOT/annotations/instances_test2017.json \
  --IMG_FOLDER /path/to/VT-Tiny-MOT/test2017 \
  --TRACKERS_FOLDER /path/to/epoch_17 \
  --TRACKERS_TO_EVAL test \
  --TRACKER_SUB_FOLDER tracker \
  --IOU_THRESHOLD 0.3 \
  --EVAL_CLASS_AGNOSTIC True
"""

import argparse
import os
import sys
from multiprocessing import freeze_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval  # noqa: E402

if __name__ == '__main__':
    freeze_support()

    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['PRINT_ONLY_COMBINED'] = True
    default_dataset_config = trackeval.datasets.VT_Tiny_MOT.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity']}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}

    parser = argparse.ArgumentParser()
    for setting in config.keys():
        if setting == 'OUTPUT_FOLDER':
            parser.add_argument("--" + setting)
        elif type(config[setting]) == list or type(config[setting]) == type(None):
            parser.add_argument("--" + setting, nargs='+')
        else:
            parser.add_argument("--" + setting)
    args = parser.parse_args().__dict__
    for setting in args.keys():
        if args[setting] is not None:
            if type(config[setting]) == type(True):
                if args[setting] == 'True':
                    x = True
                elif args[setting] == 'False':
                    x = False
                else:
                    raise Exception('Command line parameter ' + setting + ' must be True or False')
            elif type(config[setting]) == type(1):
                x = int(args[setting])
            elif type(args[setting]) == type(float):
                x = float(args[setting])
            elif type(args[setting]) == type(None):
                x = None
            else:
                x = args[setting]
            config[setting] = x

    _of = config.get('OUTPUT_FOLDER')
    if isinstance(_of, list):
        if len(_of) == 1:
            config['OUTPUT_FOLDER'] = _of[0]
        elif len(_of) == 0:
            config['OUTPUT_FOLDER'] = ''
        else:
            raise SystemExit(
                'OUTPUT_FOLDER 只能指定一个目录；若来自旧脚本 nargs 传参，请改为单个路径。得到: %r' % (_of,)
            )

    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}
    metrics_config['METRICS'] = [m.lower() for m in metrics_config['METRICS']]

    iou_threshold = float(dataset_config.get('IOU_THRESHOLD', 0.5))
    match_threshold_config = {'THRESHOLD': iou_threshold}

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.VT_Tiny_MOT(dataset_config)]
    metrics_list = []
    metric_factories = {
        'hota': lambda: trackeval.metrics.HOTA(),
        'clear': lambda: trackeval.metrics.CLEAR(match_threshold_config),
        'identity': lambda: trackeval.metrics.Identity(match_threshold_config),
    }
    for name, factory in metric_factories.items():
        if name in metrics_config['METRICS']:
            metrics_list.append(factory())
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    evaluator.evaluate(dataset_list, metrics_list)
