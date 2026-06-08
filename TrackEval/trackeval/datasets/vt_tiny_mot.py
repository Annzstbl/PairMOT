
import json
import os

import numpy as np

from ..utils import TrackEvalException
from ._base_dataset import _BaseDataset
from .. import utils
from .. import _timing


class VT_Tiny_MOT(_BaseDataset):
    """Dataset class for VT-Tiny-MOT tracking (axis-aligned boxes)."""

    CLASS_AGNOSTIC_CLS = 'class_agnostic'
    VALID_CLASSES = ['ship', 'car', 'cyclist', 'pedestrian', 'bus', 'drone', 'plane']
    CLASS_NAME_TO_ID = {
        'ship': 0, 'car': 1, 'cyclist': 2, 'pedestrian': 3, 'bus': 4, 'drone': 5, 'plane': 6,
    }
    SUPER_CATEGORIES = {
        'HUMAN': ['pedestrian', 'cyclist'],
        'VEHICLE': ['car', 'bus'],
        'AIR': ['ship', 'drone', 'plane'],
    }

    @staticmethod
    def get_default_dataset_config():
        default_config = {
            'GT_COCO_ANN': '/data1/users/litianhao01/hsmot/data/VT-Tiny-MOT/annotations/instances_test2017.json',
            'IMG_FOLDER': '/data1/users/litianhao01/hsmot/data/VT-Tiny-MOT/test2017',
            'TRACKERS_FOLDER': '/data1/users/litianhao01/experiment/memotr/vt_tiny/debug',
            'OUTPUT_FOLDER': '',
            'TRACKERS_TO_EVAL': None,
            'CLASSES_TO_EVAL': ['ship', 'car', 'cyclist', 'pedestrian', 'bus', 'drone', 'plane'],
            'IOU_THRESHOLD': 0.3, #根据https://arxiv.org/abs/2412.10861设置为0.3
            'SPLIT_TO_EVAL': 'test',
            'INPUT_AS_ZIP': False,
            'PRINT_CONFIG': True,
            'TRACKER_SUB_FOLDER': 'tracker',
            'OUTPUT_SUB_FOLDER': 'eval',
            'TRACKER_DISPLAY_NAMES': None,
            'EVAL_CLASS_AGNOSTIC': True,
        }
        assert default_config['INPUT_AS_ZIP'] is False
        return default_config

    def __init__(self, config=None):
        super().__init__()
        self.config = utils.init_config(config, self.get_default_dataset_config(), self.get_name())
        self.img_fol = self.config['IMG_FOLDER']
        self.tracker_fol = self.config['TRACKERS_FOLDER']
        self.should_classes_combine = True
        self.use_super_categories = True
        self.iou_threshold = float(self.config.get('IOU_THRESHOLD', 0.5))

        self.output_fol = self.config['OUTPUT_FOLDER']
        if self.output_fol is None or self.output_fol == '':
            self.output_fol = self.tracker_fol

        self.tracker_sub_fol = self.config['TRACKER_SUB_FOLDER']
        self.output_sub_fol = self.config['OUTPUT_SUB_FOLDER']

        self.valid_classes = self.VALID_CLASSES
        self.class_name_to_class_id = self.CLASS_NAME_TO_ID
        self.class_list = [
            cls.lower() if cls.lower() in self.valid_classes else None
            for cls in self.config['CLASSES_TO_EVAL']
        ]
        if not all(self.class_list):
            raise TrackEvalException(
                'Attempted to evaluate an invalid class. Valid classes: ' + str(self.valid_classes)
            )
        self.eval_class_agnostic = bool(self.config.get('EVAL_CLASS_AGNOSTIC', True))
        self.extra_combined_classes = []
        if self.eval_class_agnostic:
            self.class_list.append(self.CLASS_AGNOSTIC_CLS)
            self.extra_combined_classes.append(self.CLASS_AGNOSTIC_CLS)
        self.super_categories = {
            name: [cls for cls in classes if cls in self.class_list]
            for name, classes in self.SUPER_CATEGORIES.items()
        }

        gt_coco_ann = self.config.get('GT_COCO_ANN')
        if not gt_coco_ann or not os.path.isfile(gt_coco_ann):
            raise TrackEvalException('GT COCO annotation file not found: ' + str(gt_coco_ann))
        if not os.path.isdir(self.img_fol):
            raise TrackEvalException('Image folder not found: ' + self.img_fol)

        self.seq_list = []
        self.seq_lengths = {}
        self._coco_gt_cache = None
        self.seq_list = self._discover_sequences()

        if self.config['TRACKERS_TO_EVAL'] is None:
            self.tracker_list = os.listdir(self.tracker_fol)
        else:
            self.tracker_list = self.config['TRACKERS_TO_EVAL']

        if self.config['TRACKER_DISPLAY_NAMES'] is None:
            self.tracker_to_disp = dict(zip(self.tracker_list, self.tracker_list))
        elif (self.config['TRACKERS_TO_EVAL'] is not None) and (
                len(self.config['TRACKER_DISPLAY_NAMES']) == len(self.tracker_list)):
            self.tracker_to_disp = dict(zip(self.tracker_list, self.config['TRACKER_DISPLAY_NAMES']))
        else:
            raise TrackEvalException('List of tracker files and tracker display names do not match.')

        for tracker in self.tracker_list:
            for seq in self.seq_list:
                curr_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq + '.txt')
                if not os.path.isfile(curr_file):
                    print('Tracker file not found: ' + curr_file)
                    raise TrackEvalException(
                        'Tracker file not found: ' + tracker + '/' + self.tracker_sub_fol + '/' + os.path.basename(
                            curr_file))

    def get_display_name(self, tracker):
        return self.tracker_to_disp[tracker]

    def _discover_sequences(self):
        coco_gt = self._load_coco_gt_cache()
        annotated_seqs = set(coco_gt.keys())

        img_seqs = self._list_image_sequences()
        candidates = img_seqs if img_seqs else sorted(annotated_seqs)

        seq_list = []
        skipped = []
        for seq in candidates:
            if seq in annotated_seqs:
                seq_list.append(seq)
            else:
                skipped.append(seq)

        if skipped:
            print(
                'Skipping sequence(s) without GT annotations in GT_COCO_ANN: '
                + ', '.join(skipped)
            )
        if not seq_list:
            raise TrackEvalException(
                'No sequences with GT annotations found for evaluation.'
            )
        return seq_list

    def _list_image_sequences(self):
        seqs = []
        for name in sorted(os.listdir(self.img_fol)):
            seq_path = os.path.join(self.img_fol, name)
            rgb_dir = os.path.join(seq_path, '00')
            if os.path.isdir(rgb_dir):
                seqs.append(name)
        return seqs

    def _count_seq_frames(self, seq):
        rgb_dir = os.path.join(self.img_fol, seq, '00')
        if not os.path.isdir(rgb_dir):
            raise TrackEvalException(f'RGB folder not found for sequence {seq}: {rgb_dir}')
        return len([f for f in os.listdir(rgb_dir) if f.endswith('.jpg')])

    def _load_coco_gt_cache(self):
        if self._coco_gt_cache is not None:
            return self._coco_gt_cache

        with open(self.config['GT_COCO_ANN'], 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        images_by_id = {img['id']: img for img in coco_data['images']}
        dedup_rows = {}
        dropped_duplicates = 0
        for ann in coco_data['annotations']:
            img = images_by_id.get(ann['image_id'])
            if img is None or ('/00/' not in img['file_name'] and '/01/' not in img['file_name']):
                continue
            scene = img['file_name'].split('/')[0]
            if 'mot_frame_id' in img:
                frame_id = int(img['mot_frame_id']) + 1
            elif 'frame_id' in img:
                frame_id = int(img['frame_id']) + 1
            else:
                frame_id = int(os.path.splitext(img['file_name'].split('/')[-1])[0]) + 1
            x, y, w, h = ann['bbox']
            row = [frame_id, ann['track_id'], x, y, w, h, 1.0, ann['category_id'], -1, -1]
            dedup_key = (scene, frame_id, ann['track_id'])
            prev = dedup_rows.get(dedup_key)
            if prev is not None and ann['id'] >= prev[0]:
                dropped_duplicates += 1
                continue
            if prev is not None:
                dropped_duplicates += 1
            dedup_rows[dedup_key] = (ann['id'], row)

        if dropped_duplicates > 0:
            print(
                'Warning: dropped %d duplicate GT annotations with the same scene/frame/track_id '
                'in %s' % (dropped_duplicates, self.config['GT_COCO_ANN'])
            )

        cache = {}
        for (scene, frame_id, _), (_, row) in dedup_rows.items():
            cache.setdefault(scene, {}).setdefault(str(frame_id), []).append(row)

        self._coco_gt_cache = cache
        return self._coco_gt_cache

    @staticmethod
    def _parse_rect_rows(time_data, is_gt):
        parsed = {
            'dets': np.atleast_2d(time_data[:, 2:6]),
            'ids': np.atleast_1d(time_data[:, 1]).astype(int),
            'classes': np.atleast_1d(time_data[:, 7]).astype(int),
        }
        if is_gt:
            parsed['gt_extras'] = {'zero_marked': np.zeros(len(parsed['ids']), dtype=int)}
            parsed['gt_truncation'] = np.zeros(len(parsed['ids']), dtype=int)
        else:
            parsed['tracker_confidences'] = np.atleast_1d(time_data[:, 6])
        return parsed

    @staticmethod
    def _empty_timestep(is_gt):
        empty = {
            'dets': np.empty((0, 4)),
            'ids': np.empty(0, dtype=int),
            'classes': np.empty(0, dtype=int),
        }
        if is_gt:
            empty['gt_extras'] = {'zero_marked': np.empty(0, dtype=int)}
            empty['gt_truncation'] = np.empty(0, dtype=int)
        else:
            empty['tracker_confidences'] = np.empty(0)
        return empty

    def _load_raw_file(self, tracker, seq, is_gt):
        """Load GT or tracker file.

        Rect format:
            frame,id,x,y,w,h,score,cls,-1,-1
        """
        if is_gt:
            return self._load_gt_from_coco(seq)

        file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq + '.txt')
        data, _ = self._load_simple_text_file(file, is_zipped=False, zip_file=None)
        num_timesteps = self.seq_lengths[seq]

        current_time_keys = [str(t + 1) for t in range(num_timesteps)]
        extra_time_keys = [x for x in data.keys() if x not in current_time_keys]
        lack_time_keys = [x for x in current_time_keys if x not in data.keys()]

        if len(extra_time_keys) > 0:
            raise TrackEvalException(
                'Tracking data contains invalid timesteps in seq %s: %s' % (seq, ', '.join(extra_time_keys)))
        if len(lack_time_keys) > 0:
            print('Warning! Tracking data missing timesteps in seq %s: %s' % (seq, ', '.join(lack_time_keys)))

        raw_data = {
            'tracker_ids': [None] * num_timesteps,
            'tracker_classes': [None] * num_timesteps,
            'tracker_dets': [None] * num_timesteps,
            'tracker_confidences': [None] * num_timesteps,
        }

        for t in range(num_timesteps):
            time_key = str(t + 1)
            if time_key in data:
                time_data = np.asarray(data[time_key], dtype=float)
                parsed = self._parse_rect_rows(time_data, is_gt=False)
                raw_data['tracker_dets'][t] = parsed['dets']
                raw_data['tracker_ids'][t] = parsed['ids']
                raw_data['tracker_classes'][t] = parsed['classes']
                raw_data['tracker_confidences'][t] = parsed['tracker_confidences']
            else:
                empty = self._empty_timestep(is_gt=False)
                raw_data['tracker_dets'][t] = empty['dets']
                raw_data['tracker_ids'][t] = empty['ids']
                raw_data['tracker_classes'][t] = empty['classes']
                raw_data['tracker_confidences'][t] = empty['tracker_confidences']

        raw_data['num_timesteps'] = num_timesteps
        raw_data['seq'] = seq
        return raw_data

    def _load_gt_from_coco(self, seq):
        coco_gt = self._load_coco_gt_cache()
        if seq not in coco_gt:
            raise TrackEvalException(
                f'Sequence {seq} has no GT annotations in GT_COCO_ANN (should have been skipped).'
            )

        self.seq_lengths[seq] = self._count_seq_frames(seq)
        num_timesteps = self.seq_lengths[seq]

        raw_data = {
            'gt_ids': [None] * num_timesteps,
            'gt_classes': [None] * num_timesteps,
            'gt_dets': [None] * num_timesteps,
            'gt_extras': [None] * num_timesteps,
            'gt_truncation': [None] * num_timesteps,
        }

        for t in range(num_timesteps):
            time_key = str(t + 1)
            rows = coco_gt[seq].get(time_key, [])
            if rows:
                parsed = self._parse_rect_rows(np.asarray(rows, dtype=float), is_gt=True)
                raw_data['gt_dets'][t] = parsed['dets']
                raw_data['gt_ids'][t] = parsed['ids']
                raw_data['gt_classes'][t] = parsed['classes']
                raw_data['gt_extras'][t] = parsed['gt_extras']
                raw_data['gt_truncation'][t] = parsed['gt_truncation']
            else:
                empty = self._empty_timestep(is_gt=True)
                raw_data['gt_dets'][t] = empty['dets']
                raw_data['gt_ids'][t] = empty['ids']
                raw_data['gt_classes'][t] = empty['classes']
                raw_data['gt_extras'][t] = empty['gt_extras']
                raw_data['gt_truncation'][t] = empty['gt_truncation']

        raw_data['num_timesteps'] = num_timesteps
        raw_data['seq'] = seq
        return raw_data

    def _eval_class_ids(self):
        return np.array([
            self.class_name_to_class_id[cls]
            for cls in self.class_list
            if cls != self.CLASS_AGNOSTIC_CLS
        ], dtype=int)

    def _finalize_preprocessed_seq_data(self, raw_data, data, unique_gt_ids, unique_tracker_ids,
                                        num_gt_dets, num_tracker_dets):
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['gt_ids'][t]) > 0:
                    data['gt_ids'][t] = gt_id_map[data['gt_ids'][t]].astype(int)
        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['tracker_ids'][t]) > 0:
                    data['tracker_ids'][t] = tracker_id_map[data['tracker_ids'][t]].astype(int)

        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets'] = num_gt_dets
        data['num_tracker_ids'] = len(unique_tracker_ids)
        data['num_gt_ids'] = len(unique_gt_ids)
        data['num_timesteps'] = raw_data['num_timesteps']
        data['seq'] = raw_data['seq']
        self._check_unique_ids(data)
        return data

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        if cls == self.CLASS_AGNOSTIC_CLS:
            return self._get_preprocessed_seq_data_class_agnostic(raw_data)

        cls_id = self.class_name_to_class_id[cls]

        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'similarity_scores']
        data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0

        for t in range(raw_data['num_timesteps']):
            gt_class_mask = np.atleast_1d(raw_data['gt_classes'][t] == cls_id).astype(bool)
            gt_ids = raw_data['gt_ids'][t][gt_class_mask]
            gt_dets = raw_data['gt_dets'][t][gt_class_mask]

            tracker_class_mask = np.atleast_1d(raw_data['tracker_classes'][t] == cls_id).astype(bool)
            tracker_ids = raw_data['tracker_ids'][t][tracker_class_mask]
            tracker_dets = raw_data['tracker_dets'][t][tracker_class_mask]
            similarity_scores = raw_data['similarity_scores'][t][gt_class_mask, :][:, tracker_class_mask]

            data['tracker_ids'][t] = tracker_ids
            data['tracker_dets'][t] = tracker_dets
            data['gt_ids'][t] = gt_ids
            data['gt_dets'][t] = gt_dets
            data['similarity_scores'][t] = similarity_scores

            unique_gt_ids += list(np.unique(data['gt_ids'][t]))
            unique_tracker_ids += list(np.unique(data['tracker_ids'][t]))
            num_tracker_dets += len(data['tracker_ids'][t])
            num_gt_dets += len(data['gt_ids'][t])

        return self._finalize_preprocessed_seq_data(
            raw_data, data, unique_gt_ids, unique_tracker_ids, num_gt_dets, num_tracker_dets
        )

    def _get_preprocessed_seq_data_class_agnostic(self, raw_data):
        """Evaluate all classes jointly; matching is purely spatial (IoU), ignoring predicted/GT class."""
        eval_cls_ids = self._eval_class_ids()

        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'similarity_scores']
        data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0

        for t in range(raw_data['num_timesteps']):
            gt_class_mask = np.isin(raw_data['gt_classes'][t], eval_cls_ids)
            tracker_class_mask = np.isin(raw_data['tracker_classes'][t], eval_cls_ids)

            gt_ids = raw_data['gt_ids'][t][gt_class_mask]
            gt_dets = raw_data['gt_dets'][t][gt_class_mask]
            tracker_ids = raw_data['tracker_ids'][t][tracker_class_mask]
            tracker_dets = raw_data['tracker_dets'][t][tracker_class_mask]
            similarity_scores = raw_data['similarity_scores'][t][gt_class_mask, :][:, tracker_class_mask]

            data['tracker_ids'][t] = tracker_ids
            data['tracker_dets'][t] = tracker_dets
            data['gt_ids'][t] = gt_ids
            data['gt_dets'][t] = gt_dets
            data['similarity_scores'][t] = similarity_scores

            unique_gt_ids += list(np.unique(data['gt_ids'][t]))
            unique_tracker_ids += list(np.unique(data['tracker_ids'][t]))
            num_tracker_dets += len(data['tracker_ids'][t])
            num_gt_dets += len(data['gt_ids'][t])

        return self._finalize_preprocessed_seq_data(
            raw_data, data, unique_gt_ids, unique_tracker_ids, num_gt_dets, num_tracker_dets
        )

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        if gt_dets_t.shape[0] == 0 or tracker_dets_t.shape[0] == 0:
            return np.zeros((gt_dets_t.shape[0], tracker_dets_t.shape[0]), dtype=float)
        return self._calculate_box_ious(gt_dets_t, tracker_dets_t, box_format='xywh')
