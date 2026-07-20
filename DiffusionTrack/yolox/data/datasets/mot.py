import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from collections import defaultdict
import os
import glob

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset


class MOTDataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="train_half.json",
        name="train",
        img_size=(608, 1088),
        preproc=None,
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "mot")
        self.data_dir = data_dir
        self.json_file = json_file

        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.video_info=defaultdict(list) 
        self.annotations = self._load_coco_annotations()
        # "DanceTrack FRCNN" in self.coco.loadImgs(min(v))[0]["file_name"] or "MOT20" in self.coco.loadImgs(min(v))[0]["file_name"]
        self.video_info={k:(min(v),max(v),True) for k,v in self.video_info.items()}
        self.name = name
        self.img_size = img_size
        self.preproc = preproc

    def __len__(self):
        return len(self.ids)

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(index,_ids) for index,_ids in enumerate(self.ids)]

    def load_anno_from_ids(self,index,id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        frame_id = im_ann["frame_id"]
        video_id = im_ann["video_id"]
        self.video_info[video_id].append(index)
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = obj["bbox"][0]
            y1 = obj["bbox"][1]
            x2 = x1 + obj["bbox"][2]
            y2 = y1 + obj["bbox"][3]
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 6))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls
            res[ix, 5] = obj["track_id"]

        file_name = im_ann["file_name"] if "file_name" in im_ann else "{:012}".format(id_) + ".jpg"
        img_info = (height, width, frame_id, video_id, file_name)

        del im_ann, annotations

        return (res, img_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def pull_item(self, index):
        id_ = self.ids[index]

        res, img_info, file_name = self.annotations[index]
        # load image and preprocess
        img_file = os.path.join(
            self.data_dir, self.name, file_name
        )
        # img_file=file_name
        img = cv2.imread(img_file)
        assert img is not None

        return img, res.copy(), img_info, np.array([id_])

    @Dataset.resize_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img,target,img_info,img_id


class HSMOTDataset(Dataset):
    """Native HSMOT dataset with 8-band NPY images and quadrilateral boxes.

    Targets are ``[x1,y1,x2,y2,x3,y3,x4,y4,class_id,track_id]``.  All image
    frames are enumerated, including frames without annotations, which is
    required for faithful online tracking evaluation.
    """

    classes = ("car", "bike", "pedestrian", "van", "truck", "bus",
               "tricycle", "awning-bike")

    def __init__(self, data_dir, img_size=(900, 1200), preproc=None,
                 ann_subdir="mot", img_subdir="npy", sequence_file=""):
        super().__init__(img_size)
        self.data_dir = os.path.abspath(data_dir)
        self.img_root = os.path.join(self.data_dir, img_subdir)
        self.ann_root = os.path.join(self.data_dir, ann_subdir)
        self.img_size = img_size
        self.preproc = preproc
        self._classes = self.classes
        self.class_ids = list(range(len(self.classes)))

        if sequence_file:
            with open(sequence_file, "r", encoding="utf-8") as stream:
                sequences = [line.strip() for line in stream if line.strip()]
        else:
            sequences = sorted(os.path.splitext(os.path.basename(path))[0]
                               for path in glob.glob(
                                   os.path.join(self.ann_root, "*.txt")))
        if not sequences:
            raise FileNotFoundError(
                "no HSMOT sequence annotations found in {}".format(
                    self.ann_root))

        self.annotations = []
        self.ids = []
        self.video_info = {}
        for video_id, sequence in enumerate(sequences, start=1):
            frame_annotations = self._load_sequence_annotations(sequence)
            frame_paths = sorted(glob.glob(
                os.path.join(self.img_root, sequence, "*.npy")))
            if not frame_paths:
                raise FileNotFoundError(
                    "no NPY frames found for {}".format(sequence))
            start = len(self.annotations)
            for frame_path in frame_paths:
                frame_id = int(os.path.splitext(
                    os.path.basename(frame_path))[0].split("_")[0])
                rows = frame_annotations.get(frame_id, [])
                target = np.asarray(rows, dtype=np.float32).reshape(-1, 10)
                image_id = len(self.ids) + 1
                rel_name = os.path.join(
                    sequence, os.path.basename(frame_path))
                info = (900, 1200, frame_id, video_id, rel_name)
                self.annotations.append((target, info, frame_path))
                self.ids.append(image_id)
            self.video_info[video_id] = (
                start, len(self.annotations) - 1, True)

    def _load_sequence_annotations(self, sequence):
        grouped = defaultdict(list)
        path = os.path.join(self.ann_root, sequence + ".txt")
        with open(path, "r", encoding="utf-8") as stream:
            for line in stream:
                values = line.strip().split(",")
                if len(values) < 12:
                    continue
                frame_id = int(float(values[0]))
                track_id = int(float(values[1]))
                polygon = [float(value) for value in values[2:10]]
                class_id = int(float(values[11]))
                truncation = int(float(values[12])) if len(values) > 12 else 0
                if not 0 <= class_id < len(self.classes) or truncation > 0:
                    continue
                grouped[frame_id].append(
                    polygon + [class_id, track_id])
        return grouped

    def __len__(self):
        return len(self.annotations)

    def load_anno(self, index):
        return self.annotations[index][0]

    def pull_item(self, index):
        target, img_info, frame_path = self.annotations[index]
        image = np.load(frame_path)
        if image.ndim != 3 or image.shape[2] != 8:
            raise ValueError("expected HSMOT HxWx8 image at {}, got {}".
                             format(frame_path, image.shape))
        return image, target.copy(), img_info, np.array([self.ids[index]])

    @Dataset.resize_getitem
    def __getitem__(self, index):
        image, target, info, image_id = self.pull_item(index)
        if self.preproc is not None:
            image, target = self.preproc(image, target, self.input_dim)
        return image, target, info, image_id


class HSMOTPairEvalDataset(torch.utils.data.Dataset):
    """Adjacent-frame view of :class:`HSMOTDataset` for batched inference.

    The first frame of every sequence is paired with itself.  Every later
    frame is paired with the immediately preceding enumerated frame.  Unlike
    the online tracker this dataset is stateless, so it is safe to use with a
    multi-sample or distributed validation loader.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self._classes = dataset._classes
        self.classes = dataset.classes
        self.ids = dataset.ids
        self.img_size = dataset.img_size
        self.prev_indices = []
        previous_by_video = {}
        for index, (_, info, _) in enumerate(dataset.annotations):
            video_id = int(info[3])
            self.prev_indices.append(previous_by_video.get(video_id, index))
            previous_by_video[video_id] = index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        prev_index = self.prev_indices[index]
        prev_image, _, prev_info, _ = self.dataset.pull_item(prev_index)
        curr_image, target, curr_info, image_id = self.dataset.pull_item(index)
        if self.dataset.preproc is not None:
            prev_image, _ = self.dataset.preproc(
                prev_image, np.empty((0, 10), dtype=np.float32),
                self.dataset.input_dim)
            curr_image, target = self.dataset.preproc(
                curr_image, target, self.dataset.input_dim)
        meta = dict(
            image_id=int(image_id[0]),
            sequence=str(curr_info[4]).split(os.sep)[0],
            prev_frame_id=int(prev_info[2]),
            frame_id=int(curr_info[2]),
            original_height=float(curr_info[0]),
            original_width=float(curr_info[1]),
            image_name=str(curr_info[4]),
        )
        return prev_image, curr_image, target, meta
