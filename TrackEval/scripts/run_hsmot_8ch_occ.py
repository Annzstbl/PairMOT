
""" run_bdd.py

Run example:
run_bdd.py --USE_PARALLEL False --METRICS Hota --TRACKERS_TO_EVAL qdtrack

Command Line Arguments: Defaults, # Comments
    Eval arguments:
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 8,
        'BREAK_ON_ERROR': True,
        'PRINT_RESULTS': True,
        'PRINT_ONLY_COMBINED': False,
        'PRINT_CONFIG': True,
        'TIME_PROGRESS': True,
        'OUTPUT_SUMMARY': True,
        'OUTPUT_DETAILED': True,
        'PLOT_CURVES': True,
    Dataset arguments:
            # GT路径
            'GT_FOLDER': os.path.join(code_path, 'data/gt/bdd100k/bdd100k_val'),  # Location of GT data
            # 待评估结果路径
            'TRACKERS_FOLDER': os.path.join(code_path, 'data/trackers/bdd100k/bdd100k_val'),  # Trackers location
            # 输出路径
            'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            # 跟踪器名字？
            'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
            # 待评估的类别
            'CLASSES_TO_EVAL': ['pedestrian', 'rider', 'car', 'bus', 'truck', 'train', 'motorcycle', 'bicycle'],
            # Valid: ['pedestrian', 'rider', 'car', 'bus', 'truck', 'train', 'motorcycle', 'bicycle']
            'SPLIT_TO_EVAL': 'val',  # Valid: 'training', 'val',
            'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
            'PRINT_CONFIG': True,  # Whether to print current config
            'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
    Metric arguments:
        'METRICS': ['Hota','Clear', 'ID', 'Count']
"""

import sys
import os
import argparse
from multiprocessing import freeze_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval  # noqa: E402

if __name__ == '__main__':
    freeze_support()

    # Command line interface:
    default_eval_config = trackeval.Evaluator_OCC.get_default_eval_config()
    default_eval_config['PRINT_ONLY_COMBINED'] = True
    default_dataset_config = trackeval.datasets.HSMOT_8ch_OCC.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['CLEAR_OCC']}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
    parser = argparse.ArgumentParser()
    for setting in config.keys():
        if setting == 'OUTPUT_FOLDER':
            parser.add_argument("--" + setting, default=None)
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
                    raise Exception('Command line parameter ' + setting + 'must be True or False')
            elif type(config[setting]) == type(1):
                x = int(args[setting])
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

    # Run code
    evaluator = trackeval.Evaluator_OCC(eval_config)
    dataset_list = [trackeval.datasets.HSMOT_8ch_OCC(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.CLEAR_OCC]:
        if metric.get_name().lower() in metrics_config['METRICS']:
            metrics_list.append(metric())
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')


    # target_ids_dict = {
    #     "data30-3": [117, 106, 14, 90, 91, 92, 93],
    #     "data30-4": [34, 33, 20, 7, 21, 26, 32, 24],
    #     "data36-4": [110, 81, 82, 98, 100],
    #     "data36-5": [154, 159, 50, 60, 155, 161, 167],
    #     "data39-1": [4, 16],
    #     "data40-3": [7, 19, 21, 24, 25],
    #     "data42-1": [35, 39, 51, 26, 58, 62, 39, 49, 91, 97, 96, 100, 114, 115, 125, 134, 139, 140, 145],
    #     "data42-2": [10, 12, 9,  3],
    #     "data42-3": [51, 18, 19, 23, 26, 30, 31, 32, 33, 34, 35, 38, 40],
    #     "data46-11": [26, 29, 73, 22, 23, 24, 35, 46, 87, 88, 89, 90, 91, 92, 93, 94, 96, 95, 110],
    #     "data46-12": [22, 37, 39, 13, 17, 10, 12, 29, 41, 45, 38, 36, 58, 28, 34, 19, 32, 44],
    #     "data49-2": [],
    # }
    #set1
    target_ids_dict = {
        "data30-3": [117, 106, 14, 90, 91, 92, 93],
        "data30-4": [34, 33, 20, 7, 21, 26, 32, 24],
        "data36-4": [110, 81, 82, 98, 100],
        "data36-5": [154, 159, 50, 60, 155, 161, 167],
        "data39-1": [4, 16],
        "data40-3": [7, 19, 21, 24, 25],
        "data42-1": [21, 39, 51, 35, 57, 26, 58, 62, 39, 50, 49, 91, 97, 96, 100, 114, 115, 125, 128, 134, 139, 140, 145],
        "data42-2": [10, 12, 5, 9, 1, 3],
        "data42-3": [40, 38, 33, 32, 35, 34, 30, 31, 19, 23, 26, 18, 51, 50],
        "data46-11": [78, 26, 12, 29, 73, 22, 23, 24, 35, 46, 87, 88, 89, 90, 91, 92, 93, 94, 96, 95, 110, 101],
        "data46-12": [44, 40, 22, 37, 39, 13, 32, 17, 10, 12, 29, 41, 45, 38, 36, 58, 28, 34, 19],
        "data49-2": [5, 9, 13, 7, 14],
    }
    #set2
    target_ids_dict = {
        "data30-3": [117, 106, 14, 90, 91, 92, 93],
        "data30-4": [34, 33, 20, 7, 21, 26, 32, 24],
        "data36-4": [110, 81, 82, 98, 100],
        "data36-5": [154, 159, 50, 60, 155, 161, 167],
        "data39-1": [4, 16],
        "data40-3": [7, 19, 21, 24, 25],
        "data42-1": [21, 39, 51, 35, 57, 26, 58, 62, 39, 50, 49, 91, 97, 96, 100, 114, 115, 125, 128, 134, 139, 140, 145],
        "data42-2": [10, 12, 5, 9, 1, 3],
        "data42-3": [40, 38, 33, 32, 35, 34, 30, 31, 19, 23, 26, 18, 51, 50],
        "data46-11": [],
        "data46-12": [44, 40, 22, 37, 39, 13, 32, 17, 10, 12, 29, 41, 45, 38, 36, 58, 28, 34, 19],
        "data49-2": [5, 9, 13, 7, 14],
    }
    #set3
    target_ids_dict = {
        "data30-3": [117, 106, 14, 90, 91, 92, 93],
        "data30-4": [34, 33, 20, 7, 21, 26, 32, 24],
        "data36-4": [110, 81, 82, 98, 100],
        "data36-5": [154, 159, 50, 60, 155, 161, 167],
        "data39-1": [4, 16],
        "data40-3": [7, 19, 21, 24, 25],
        "data42-1": [21, 39, 51, 35, 57, 26, 58, 62, 39, 50, 49, 91, 97, 96, 100, 114, 115, 125, 128, 134, 139, 140, 145],
        "data42-2": [10, 12, 5, 9, 1, 3],
        "data42-3": [40, 38, 33, 32, 35, 34, 30, 31, 19, 23, 26, 18, 51, 50],
        "data46-11": [78, 26, 12, 29, 73, 22, 23, 24, 35, 46, 87, 88, 89, 90, 91, 92, 93, 94, 96, 95, 110, 101],
        "data46-12": [0],
        "data49-2": [5, 9, 13, 7, 14],
    }
    #set 4
    target_ids_dict = {
        "data30-3": [117, 106, 14, 90, 91, 92, 93],
        "data30-4": [34, 33, 20, 7, 21, 26, 32, 24],
        "data36-4": [110, 81, 82, 98, 100],
        "data36-5": [154, 159, 50, 60, 155, 161, 167],
        "data39-1": [4, 16],
        "data40-3": [7, 19, 21, 24, 25],
        "data42-1": [21, 39, 51, 35, 57, 26, 58, 62, 39, 50, 49, 91, 97, 96, 100, 114, 115, 125, 128, 134, 139, 140, 145],
        "data42-2": [10, 12, 5, 9, 1, 3],
        "data42-3": [40, 38, 33, 32, 35, 34, 30, 31, 19, 23, 26, 18, 51, 50],
        "data46-11": [],
        "data46-12": [0],
        "data49-2": [5, 9, 13, 7, 14],
    }

    #set 5
    target_ids_dict = {
        "data30-3": [117, 106, 14, 90, 91, 92, 93],
        "data30-4": [34, 33, 20, 7, 21, 26, 32, 24],
        "data36-4": [110, 81, 82, 98, 100],
        "data36-5": [154, 159, 50, 60, 155, 161, 167],
        "data39-1": [4, 16],
        "data40-3": [7, 19, 21, 24, 25],
        "data42-1": [21, 39, 51, 35, 57, 26, 58, 62, 39, 50, 91, 97, 96, 100, 114, 115, 125, 128, 134, 139, 140, 145],
        "data42-2": [10, 12, 5, 9, 1, 3],
        "data42-3": [40, 38, 33, 32, 35, 30, 31, 19, 23, 26, 18, 51, 50],
        "data46-11": [78, 26, 12, 29, 73, 22, 23, 24, 35, 46, 87, 88, 89, 90, 91, 92, 93, 94, 96, 95, 110, 101],
        "data46-12": [44, 40, 22, 37, 39, 13, 32, 17, 10, 12, 29, 41, 45, 38, 36, 28, 34, 19],
        "data49-2": [5, 9, 13, 7, 14],
    }

    #set 6
    target_ids_dict = {
        "data30-3": [117, 106, 14, 90, 91, 92, 93],
        "data30-4": [34, 33, 20, 7, 21, 26, 32, 24],
        "data36-4": [110, 81, 82, 98, 100],
        "data36-5": [154, 159, 50, 60, 161, 167],
        "data39-1": [4, 16],
        "data40-3": [7, 19, 21, 24, 25],
        "data42-1": [21, 39, 51, 35, 57, 26, 58, 62, 39, 50, 91, 97, 96, 100, 114, 115, 125, 128, 134, 139, 140, 145],
        "data42-2": [10, 12, 5, 9, 1, 3],
        "data42-3": [38, 33, 32, 35, 30, 31, 19, 23, 26, 18, 51, 50],
        "data46-11": [78, 26, 29, 73, 22, 24, 35, 87, 88, 89, 90, 92, 93, 94, 96, 95, 110, 101],
        "data46-12": [44, 40, 22, 37, 39, 13, 32, 17, 10, 12, 29, 41, 45, 38, 36, 28, 34, 19],
        "data49-2": [5, 9, 13, 7, 14],
    }
    evaluator.evaluate(dataset_list, metrics_list, target_ids_dict)