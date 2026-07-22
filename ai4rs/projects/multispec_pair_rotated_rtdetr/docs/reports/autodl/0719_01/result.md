# 0719_01 AutoDL Result

## Experiment

- Name: `paper_liquid_pairconsensus_relaxedset_pacde_fused`
- Protocol: full HSMOT, ordered `t-1 -> t` pairs, 1200x900, R18 COCO-adapted initialization, BF16, two GPUs.
- Completed asynchronous TrackEval points: `18/18`.
- Checkpoint selection: unique maximum of `cls_HOTA + det_HOTA`.

## Tracking Results

| epoch | cls HOTA | cls MOTA | cls IDF1 | det HOTA | det MOTA | det IDF1 | HOTA sum |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 36.948 | 28.783 | 44.734 | 44.909 | 38.502 | 51.401 | 81.857 |
| 8 | 44.932 | 36.682 | 53.884 | 50.407 | 44.839 | 57.629 | 95.339 |
| 12 | 47.626 | 39.800 | 57.424 | 54.759 | 50.156 | 63.534 | 102.385 |
| 16 | 49.158 | 41.044 | 58.980 | 56.523 | 52.596 | 65.982 | 105.681 |
| 20 | 50.320 | 43.023 | 60.646 | 57.506 | 54.039 | 67.397 | 107.826 |
| 24 | 50.936 | 43.109 | 60.868 | 58.279 | 54.945 | 68.328 | 109.215 |
| 28 | 51.504 | 44.273 | 61.413 | 58.613 | 55.644 | 68.718 | 110.117 |
| 32 | 52.084 | 44.944 | 62.080 | 59.216 | 56.651 | 69.570 | 111.300 |
| 36 | 52.375 | 45.249 | 62.372 | 59.276 | 57.159 | 69.501 | 111.651 |
| 40 | 52.957 | 45.756 | 62.941 | 59.968 | 58.103 | 70.724 | 112.925 |
| 44 | 53.184 | 46.115 | 63.434 | 60.281 | 58.653 | 71.118 | 113.465 |
| 48 | 53.085 | 46.350 | 63.124 | 60.309 | 58.856 | 71.003 | 113.394 |
| 52 | 53.392 | 46.274 | 63.748 | 60.651 | 58.892 | 71.590 | 114.043 |
| 56 | 53.202 | 45.730 | 63.243 | 60.719 | 58.932 | 71.459 | 113.921 |
| 60 | 53.105 | 45.124 | 63.002 | 60.892 | 59.057 | 71.700 | 113.997 |
| 64 | 53.626 | 46.351 | 63.834 | 61.054 | 59.436 | 72.063 | 114.680 |
| 68 | 53.720 | 46.659 | 63.791 | 61.166 | 59.748 | 72.142 | 114.886 |
| 72 | 53.883 | 46.831 | 63.920 | 61.411 | 59.931 | 72.366 | 115.294 |

## Selected Checkpoint

- Best epoch: `72`.
- `cls_HOTA=53.883`.
- `det_HOTA=61.411`.
- `cls_HOTA + det_HOTA=115.294`.
- Same-epoch `pair_mAP=0.3161`.
- Same-epoch `pair_AP50=0.5339`.
- Final epoch sum: `115.294`; best-to-final delta: `+0.000`.

## Per-Class HOTA At Selected Epoch

| class | HOTA | baseline | delta |
| --- | ---: | ---: | ---: |
| awning-bike | 48.478 | 49.109 | -0.631 |
| bike | 41.199 | 43.807 | -2.608 |
| bus | 71.125 | 71.483 | -0.358 |
| car | 80.824 | 81.516 | -0.692 |
| pedestrian | 43.905 | 44.640 | -0.735 |
| tricycle | 42.037 | 39.283 | +2.754 |
| truck | 43.650 | 33.852 | +9.798 |
| van | 59.843 | 62.825 | -2.982 |

## Conclusion

Against `0716_02_paper_base_epoch68`, cls HOTA changes by `+0.569`, det HOTA by `-0.571`, and their sum by `-0.002`.
Both primary HOTA axes do not improve simultaneously; this model should not replace the current mainline model.
