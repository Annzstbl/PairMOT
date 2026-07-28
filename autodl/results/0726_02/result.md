# 0726_02 AutoDL Result

## Experiment

- Name: `Paper Base + Liquid + 0705_01 Encoder`
- Protocol: full HSMOT, ordered `t-1 -> t` pairs, 1200x900, R18 COCO-adapted initialization, BF16, two GPUs.
- Completed asynchronous TrackEval points: `18/18`.
- Checkpoint selection: unique maximum of `cls_HOTA + det_HOTA`.

## Tracking Results

| epoch | cls HOTA | cls MOTA | cls IDF1 | det HOTA | det MOTA | det IDF1 | HOTA sum |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 36.928 | 29.226 | 42.648 | 40.657 | 35.832 | 45.519 | 77.585 |
| 8 | 46.591 | 40.155 | 55.411 | 50.746 | 51.229 | 59.140 | 97.337 |
| 12 | 49.180 | 42.210 | 58.659 | 56.171 | 54.634 | 65.595 | 105.351 |
| 16 | 50.375 | 43.072 | 60.376 | 57.745 | 55.643 | 67.466 | 108.120 |
| 20 | 51.235 | 42.995 | 61.213 | 58.193 | 55.983 | 67.971 | 109.428 |
| 24 | 51.498 | 42.992 | 61.231 | 58.545 | 56.287 | 68.362 | 110.043 |
| 28 | 51.990 | 43.644 | 61.871 | 59.152 | 56.814 | 69.331 | 111.142 |
| 32 | 52.462 | 43.964 | 62.214 | 59.696 | 57.416 | 69.924 | 112.158 |
| 36 | 52.815 | 44.613 | 62.697 | 59.901 | 57.966 | 70.245 | 112.716 |
| 40 | 53.939 | 46.078 | 64.204 | 60.401 | 59.030 | 70.867 | 114.340 |
| 44 | 53.947 | 46.090 | 64.011 | 60.623 | 59.178 | 71.121 | 114.570 |
| 48 | 54.204 | 46.877 | 64.357 | 60.690 | 59.499 | 71.141 | 114.894 |
| 52 | 54.443 | 46.237 | 64.676 | 61.127 | 59.954 | 71.791 | 115.570 |
| 56 | 54.267 | 45.582 | 64.316 | 61.244 | 60.115 | 72.018 | 115.511 |
| 60 | 54.202 | 45.759 | 64.353 | 61.287 | 60.018 | 72.137 | 115.489 |
| 64 | 54.679 | 46.018 | 64.801 | 61.595 | 60.210 | 72.421 | 116.274 |
| 68 | 54.663 | 46.041 | 64.881 | 61.606 | 59.902 | 72.418 | 116.269 |
| 72 | 54.742 | 45.957 | 64.798 | 61.631 | 59.930 | 72.408 | 116.373 |

## Selected Checkpoint

- Best epoch: `72`.
- `cls_HOTA=54.742`.
- `det_HOTA=61.631`.
- `cls_HOTA + det_HOTA=116.373`.
- Same-epoch `pair_mAP=0.3223`.
- Same-epoch `pair_AP50=0.5429`.
- Final epoch sum: `116.373`; best-to-final delta: `+0.000`.

## Per-Class HOTA At Selected Epoch

| class | HOTA | baseline | delta |
| --- | ---: | ---: | ---: |
| awning-bike | 48.211 | 49.109 | -0.898 |
| bike | 42.268 | 43.807 | -1.539 |
| bus | 71.416 | 71.483 | -0.067 |
| car | 81.258 | 81.516 | -0.258 |
| pedestrian | 44.022 | 44.640 | -0.618 |
| tricycle | 44.833 | 39.283 | +5.550 |
| truck | 42.896 | 33.852 | +9.044 |
| van | 63.030 | 62.825 | +0.205 |

## Conclusion

Against `0716_02_paper_base_epoch68`, cls HOTA changes by `+1.428`, det HOTA by `-0.351`, and their sum by `+1.077`.
Both primary HOTA axes do not improve simultaneously; this model should not replace the current mainline model.
