"""Small compatibility test that does not require the MOT17 dataset."""

import numpy as np
import torch

import model
from dataloader import Normalizer, collater
from lib.nms import cython_soft_nms_wrapper


def test_soft_nms():
    detections = np.array([
        [0, 0, 10, 10, 1, 1, 11, 11, 0.90, 0.8],
        [0, 0, 10, 10, 2, 2, 12, 12, 0.80, 0.7],
        [30, 30, 40, 40, 31, 31, 41, 41, 0.70, 0.6],
    ], dtype=np.float32)
    output = cython_soft_nms_wrapper(0.7, method='gaussian')(detections)
    assert output.shape == (3, 10)
    assert np.isfinite(output).all()


def test_data_pipeline():
    annotations = np.array([[5.0, 6.0, 20.0, 30.0, 0.0, 1.0]])
    sample = {
        'img': np.zeros((64, 64, 3), dtype=np.uint8),
        'annot': annotations.copy(),
        'img_next': np.zeros((64, 64, 3), dtype=np.uint8),
        'annot_next': annotations.copy(),
    }
    batch = collater([Normalizer()(sample)])
    assert batch['img'].shape == (1, 3, 64, 64)
    assert batch['annot'].shape == (1, 1, 6)


def test_training_step(device):
    network = model.resnet18(num_classes=1, pretrained=False).to(device).train()
    network.freeze_bn()
    image_1 = torch.randn(1, 3, 64, 64, device=device)
    image_2 = torch.randn(1, 3, 64, 64, device=device)
    annotations_1 = torch.tensor(
        [[[10.0, 5.0, 30.0, 45.0, 0.0, 1.0]]], device=device
    )
    annotations_2 = torch.tensor(
        [[[12.0, 6.0, 32.0, 46.0, 0.0, 1.0]]], device=device
    )
    (classification, regression), reid = network(
        [image_1, annotations_1, image_2, annotations_2]
    )
    loss = classification.mean() + regression.mean() + reid.mean()
    assert torch.isfinite(loss)
    loss.backward()


def test_inference(device):
    network = model.resnet18(num_classes=1, pretrained=False).to(device).eval()
    network.classificationModel.output.bias.data.fill_(-2.0)
    frame = torch.randn(1, 3, 64, 64, device=device)
    with torch.no_grad():
        _, _, features = network(frame)
        scores, boxes, _ = network(frame, last_feat=features)
    assert isinstance(scores, np.ndarray)
    assert boxes.ndim == 2 and boxes.shape[1] == 10
    assert boxes.shape[0] > 0 and np.isfinite(boxes).all()


def main():
    test_soft_nms()
    test_data_pipeline()
    test_training_step(torch.device('cpu'))
    if torch.cuda.is_available():
        test_inference(torch.device('cuda:0'))
    print('CTracker Python 3.10 compatibility smoke test passed.')


if __name__ == '__main__':
    main()
