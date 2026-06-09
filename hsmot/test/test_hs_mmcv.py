import numpy as np

from hsmot.mmlab import hs_mmcv


def test_imread_imflip_imresize_roundtrip(tmp_path):
    img = np.random.randint(0, 255, (100, 120, 3), dtype=np.uint8)
    path = tmp_path / "sample.png"
    import cv2

    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    loaded = hs_mmcv.imread(str(path), channel_order="rgb")
    assert loaded.shape == img.shape

    flipped = hs_mmcv.imflip(loaded, direction="horizontal")
    assert flipped.shape == loaded.shape

    resized, w_scale, h_scale = hs_mmcv.imresize(loaded, (60, 50), return_scale=True)
    assert resized.shape == (50, 60, 3)
    assert w_scale > 0 and h_scale > 0


def test_imnormalize_inplace():
    img = np.ones((4, 4, 3), dtype=np.float32) * 128.0
    mean = np.array([128.0, 128.0, 128.0], dtype=np.float32)
    std = np.array([64.0, 64.0, 64.0], dtype=np.float32)
    hs_mmcv.imnormalize_(img, mean, std, to_rgb=False)
    assert np.allclose(img, 0.0)
