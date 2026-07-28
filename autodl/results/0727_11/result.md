# 0727_11 AutoDL Result

## Experiment

- Name: `Paper Base + Liquid + MCDE Encoder`
- Protocol: full HSMOT, ordered `t-1 -> t` pairs, 1200x900, R18 COCO-adapted initialization, BF16, two GPUs.
- Completed asynchronous TrackEval points: `18/18`.
- Checkpoint selection: unique maximum of `cls_HOTA + det_HOTA`.

## Tracking Results

| epoch | cls HOTA | cls MOTA | cls IDF1 | det HOTA | det MOTA | det IDF1 | HOTA sum |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 37.547 | 27.607 | 43.406 | 44.047 | 36.226 | 49.642 | 81.594 |
| 8 | 45.184 | 37.098 | 53.895 | 52.740 | 50.187 | 61.297 | 97.924 |
| 12 | 48.410 | 41.185 | 58.094 | 56.313 | 54.877 | 66.318 | 104.723 |
| 16 | 50.192 | 43.457 | 60.779 | 57.605 | 56.863 | 68.071 | 107.797 |
| 20 | 51.021 | 44.381 | 61.604 | 58.172 | 57.896 | 68.729 | 109.193 |
| 24 | 51.839 | 45.121 | 62.552 | 58.908 | 58.534 | 69.646 | 110.747 |
| 28 | 52.562 | 45.609 | 63.498 | 59.623 | 59.000 | 70.511 | 112.185 |
| 32 | 53.064 | 46.005 | 63.863 | 60.119 | 59.362 | 71.113 | 113.183 |
| 36 | 53.406 | 46.212 | 64.134 | 60.496 | 59.663 | 71.590 | 113.902 |
| 40 | 53.735 | 46.119 | 64.390 | 60.727 | 59.610 | 71.887 | 114.462 |
| 44 | 53.728 | 45.762 | 64.172 | 61.073 | 59.810 | 72.258 | 114.801 |
| 48 | 53.813 | 45.952 | 64.229 | 61.232 | 59.995 | 72.254 | 115.045 |
| 52 | 53.763 | 45.438 | 63.815 | 61.413 | 60.074 | 72.336 | 115.176 |
| 56 | 53.897 | 44.775 | 64.001 | 61.636 | 60.298 | 72.647 | 115.533 |
| 60 | 54.462 | 46.392 | 64.554 | 61.881 | 60.837 | 72.929 | 116.343 |
| 64 | 54.415 | 46.073 | 64.272 | 61.966 | 60.644 | 72.986 | 116.381 |
| 68 | 54.853 | 46.434 | 65.016 | 62.050 | 60.885 | 72.961 | 116.903 |
| 72 | 54.605 | 45.976 | 64.527 | 62.090 | 60.984 | 73.055 | 116.695 |

## Selected Checkpoint

- Best epoch: `68`.
- `cls_HOTA=54.853`.
- `det_HOTA=62.050`.
- `cls_HOTA + det_HOTA=116.903`.
- Same-epoch `pair_mAP=0.3190`.
- Same-epoch `pair_AP50=0.5384`.
- Final epoch sum: `116.695`; best-to-final delta: `+0.208`.

## Per-Class HOTA At Selected Epoch

| class | HOTA | baseline | delta |
| --- | ---: | ---: | ---: |
| awning-bike | 47.233 | 49.109 | -1.876 |
| bike | 43.784 | 43.807 | -0.023 |
| bus | 71.819 | 71.483 | +0.336 |
| car | 81.163 | 81.516 | -0.353 |
| pedestrian | 44.989 | 44.640 | +0.349 |
| tricycle | 43.055 | 39.283 | +3.772 |
| truck | 44.091 | 33.852 | +10.239 |
| van | 62.684 | 62.825 | -0.141 |

## Conclusion

Against `0716_02_paper_base_epoch68`, cls HOTA changes by `+1.539`, det HOTA by `+0.068`, and their sum by `+1.607`.
Both primary HOTA axes improve, so this model is a positive candidate under the same protocol.
