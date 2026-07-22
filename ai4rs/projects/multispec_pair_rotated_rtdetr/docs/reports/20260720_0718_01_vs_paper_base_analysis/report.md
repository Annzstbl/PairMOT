# 0718_01 vs Paper Base: Detection and Tracking Diagnosis

Comparison protocol: Paper Base 0716_02 epoch 68 versus 0718_01 epoch 64. Both checkpoints are their unique maxima of cls_HOTA + det_HOTA; tracker parameters are identical.

## Overall decomposition

| metric | BASE | 0718_01 | delta |
| --- | ---: | ---: | ---: |
| HOTA | 61.982 | 61.812 | -0.170 |
| DetA | 53.890 | 53.701 | -0.189 |
| AssA | 73.643 | 73.614 | -0.029 |
| DetRe | 61.913 | 61.416 | -0.497 |
| DetPr | 71.741 | 71.861 | +0.120 |
| MOTA | 60.599 | 60.786 | +0.187 |
| IDF1 | 72.303 | 72.423 | +0.120 |
| IDSW | 2136.000 | 2112.000 | -24.000 |
| Frag | 6428.000 | 6653.000 | +225.000 |
| CLR_FP | 24146.000 | 23153.000 | -993.000 |
| CLR_FN | 51021.000 | 51670.000 | +649.000 |

In the exact identity `HOTA = sqrt(DetA * AssA)`, the negative log-HOTA change is `0.00195`. DetA contributes `89.9%` and AssA contributes `10.1%` of the loss.

**Primary conclusion:** `0718_01` does not reproduce the systematic association collapse seen in
`0719_02`. Its det HOTA is only `0.170` below Base; DetA explains `89.9%` of that small loss and
AssA only `10.1%`. IDSW decreases by `24`, det MOTA increases by `0.187`, det IDF1 increases by
`0.120`, and false positives decrease by `993`. The tradeoff is slightly lower recall (`-0.497`)
and more fragmentation (`+225`), rather than widespread cross-ID pairing.

## Raw pair-detection audit

Fixed diagnostic thresholds: side score >= `0.2` and rotated IoU >= `0.5`. This audit is not the paper AP protocol; it isolates detector outputs before tracking.

| metric | BASE | 0718_01 | delta |
| --- | ---: | ---: | ---: |
| raw_precision | 80.234 | 80.128 | -0.107 |
| raw_recall | 79.916 | 79.756 | -0.160 |
| raw_f1 | 80.075 | 79.942 | -0.133 |
| pair_precision | 99.638 | 99.721 | +0.084 |
| pair_recall | 76.481 | 76.298 | -0.183 |

Paper pair mAP improves from `0.3149` to `0.3200`, and pair AP50 from `0.5225` to `0.5417`.
Independent AP50 improves from `0.5445` to `0.5654`, while the association gap increases only
from `0.0220` to `0.0237`. This confirms that `0718_01` mainly improves detection/class ranking;
its global pair-correspondence penalty is small, although a few sequences still regress.

## Class comparison from all_cls_summary

| class | BASE HOTA | 0718_01 HOTA | delta | DetA delta | AssA delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| car | 81.516 | 81.642 | +0.126 | -0.167 | +0.465 |
| bike | 43.807 | 42.695 | -1.112 | -2.120 | +0.504 |
| pedestrian | 44.640 | 43.386 | -1.254 | -1.010 | -1.646 |
| van | 62.825 | 61.444 | -1.381 | -0.082 | -3.235 |
| truck | 33.852 | 46.005 | +12.153 | +14.592 | -0.005 |
| bus | 71.483 | 73.759 | +2.276 | +3.510 | +0.697 |
| tricycle | 39.283 | 43.150 | +3.867 | +4.075 | +2.598 |
| awning-bike | 49.109 | 50.127 | +1.018 | +3.739 | -3.024 |

## Sequence comparison from all_seq_summary

Rows are sorted by det HOTA delta. Across 50 sequences, `0718_01` wins det HOTA on `26`, Base wins
on `23`, and one ties; `0718_01` wins cls HOTA on `29` and Base on `21`. Diagnostic videos are
generated only for the five key sequences selected below.

| sequence | det winner | det HOTA delta | cls winner | cls HOTA delta | DetA delta | AssA delta | raw F1 delta | pair-link recall delta | diagnosis |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| data39-1 | BASE | -6.027 | BASE | -1.488 | -7.489 | -2.603 | -7.426 | -4.942 | detection-dominant |
| data28-4 | BASE | -5.236 | BASE | -2.802 | -10.597 | +0.628 | -7.181 | -9.804 | detection-only |
| data41-1 | BASE | -4.078 | BASE | -7.502 | -5.818 | -1.595 | -5.017 | -0.585 | detection-dominant |
| data47-3 | BASE | -3.695 | BASE | -3.657 | -6.136 | +0.134 | -6.553 | -5.588 | detection-only |
| data37-1 | BASE | -3.023 | BASE | -2.840 | -4.204 | -1.465 | -1.971 | -5.646 | detection-dominant |
| data47-2 | BASE | -2.826 | BASE | -2.147 | -4.424 | -0.761 | -4.072 | -5.662 | detection-dominant |
| data30-4 | BASE | -2.721 | BASE | -2.989 | -2.240 | -3.087 | -0.964 | -2.142 | association-dominant |
| data37-2 | BASE | -2.544 | 0718_01 | +0.959 | -2.809 | -1.484 | -0.760 | -1.640 | detection-dominant |
| data28-2 | BASE | -2.501 | BASE | -0.714 | +1.967 | -7.290 | -5.267 | -2.787 | association-only |
| data47-4 | BASE | -2.430 | 0718_01 | +0.317 | -6.472 | +3.121 | -9.018 | -11.273 | detection-only |
| data30-9 | BASE | -2.254 | BASE | -1.384 | -3.080 | -0.919 | -1.416 | -1.098 | detection-dominant |
| data36-11 | BASE | -1.672 | BASE | -0.214 | -1.947 | -1.329 | +1.191 | +1.862 | detection-dominant |
| data39-6 | BASE | -1.274 | BASE | -3.754 | +0.420 | -3.412 | +1.863 | -0.455 | association-only |
| data39-3 | BASE | -1.089 | BASE | -0.124 | -4.679 | +3.456 | -6.881 | -6.336 | detection-only |
| data36-4 | BASE | -1.014 | 0718_01 | +1.072 | -1.961 | +0.166 | -1.756 | -2.171 | detection-only |
| data36-10 | BASE | -1.003 | 0718_01 | +1.960 | -0.527 | -1.678 | +0.340 | -0.129 | association-dominant |
| data37-10 | BASE | -0.661 | 0718_01 | +0.327 | -0.971 | -0.136 | +0.188 | -0.317 | detection-dominant |
| data30-2 | BASE | -0.643 | 0718_01 | +5.162 | -0.274 | -0.913 | -0.612 | -1.430 | association-dominant |
| data42-1 | BASE | -0.635 | BASE | -0.572 | -0.866 | -0.426 | -0.779 | -1.943 | detection-dominant |
| data34-3 | BASE | -0.551 | BASE | -9.297 | -0.025 | -0.046 | -0.362 | +0.121 | association-dominant |
| data46-12 | BASE | -0.348 | 0718_01 | +3.962 | -0.752 | +0.042 | -0.818 | +2.112 | detection-only |
| data40-4 | BASE | -0.193 | BASE | -0.404 | -3.768 | +5.723 | -3.315 | -0.839 | detection-only |
| data30-5 | BASE | -0.189 | 0718_01 | +6.003 | +0.123 | -1.052 | +3.401 | +1.951 | association-only |
| data37-12 | TIE | -0.026 | 0718_01 | +2.058 | -0.195 | +0.263 | +0.460 | -0.254 | detection-only |
| data28-1 | 0718_01 | +0.334 | BASE | -3.073 | -1.324 | +2.189 | -2.406 | +1.532 | detection-only |
| data30-10 | 0718_01 | +0.409 | 0718_01 | +5.276 | +1.379 | -0.868 | +0.927 | +1.410 | association-only |
| data40-3 | 0718_01 | +0.537 | 0718_01 | +6.524 | +2.634 | -2.492 | -8.573 | -4.530 | association-only |
| data42-2 | 0718_01 | +0.775 | BASE | -0.235 | +2.226 | -0.819 | +1.595 | +2.623 | association-only |
| data37-11 | 0718_01 | +0.778 | 0718_01 | +5.469 | -0.389 | +2.141 | -0.013 | -0.067 | detection-only |
| data36-13 | 0718_01 | +0.914 | BASE | -0.061 | -0.225 | +2.508 | +0.202 | -1.510 | detection-only |
| data36-12 | 0718_01 | +1.043 | 0718_01 | +5.389 | +0.610 | +1.523 | +0.796 | -0.346 | improved |
| data24-1 | 0718_01 | +1.113 | 0718_01 | +4.068 | +1.680 | +0.311 | +0.916 | +3.145 | improved |
| data34-2 | 0718_01 | +1.156 | 0718_01 | +5.748 | +3.004 | -1.329 | +1.751 | +1.781 | association-only |
| data42-3 | 0718_01 | +1.442 | 0718_01 | +0.132 | +0.542 | +2.422 | -3.289 | -2.862 | improved |
| data46-11 | 0718_01 | +1.519 | 0718_01 | +4.596 | +3.670 | -2.664 | +1.875 | +2.813 | association-only |
| data39-2 | 0718_01 | +1.584 | 0718_01 | +0.240 | +3.954 | -2.518 | +8.985 | +1.101 | association-only |
| data36-5 | 0718_01 | +1.591 | BASE | -2.527 | +1.199 | +1.908 | -0.849 | -0.275 | improved |
| data28-3 | 0718_01 | +1.823 | 0718_01 | +1.547 | +10.444 | -14.684 | +4.185 | -3.663 | association-only |
| data48-1 | 0718_01 | +1.885 | 0718_01 | +1.007 | +2.235 | +0.117 | +4.629 | +0.000 | improved |
| data34-1 | 0718_01 | +2.364 | BASE | -6.521 | +2.619 | +2.014 | +2.424 | +3.313 | improved |
| data28-5 | 0718_01 | +2.426 | 0718_01 | +1.875 | +5.263 | -1.808 | +7.257 | +1.212 | association-only |
| data49-2 | 0718_01 | +2.858 | 0718_01 | +5.145 | +3.636 | +1.498 | +2.771 | +0.361 | improved |
| data40-5 | 0718_01 | +2.867 | 0718_01 | +1.756 | +2.076 | +3.864 | -0.679 | -0.227 | improved |
| data30-3 | 0718_01 | +3.309 | 0718_01 | +8.975 | +2.836 | +3.971 | +1.627 | +1.992 | improved |
| data27-1 | 0718_01 | +3.335 | 0718_01 | +2.053 | +4.046 | +1.674 | -0.252 | +8.511 | improved |
| data23-1 | 0718_01 | +3.588 | BASE | -0.825 | +8.061 | -1.771 | +3.328 | +7.884 | association-only |
| data33-1 | 0718_01 | +4.430 | 0718_01 | +0.554 | +7.983 | +1.380 | +17.112 | -2.373 | improved |
| data31-1 | 0718_01 | +4.619 | 0718_01 | +0.578 | +0.852 | +9.327 | +3.501 | +7.143 | improved |
| data47-1 | 0718_01 | +4.717 | 0718_01 | +0.939 | +7.610 | +1.155 | +3.867 | +10.000 | improved |
| data28-6 | 0718_01 | +32.601 | 0718_01 | +10.930 | +26.770 | +29.373 | +5.587 | +8.297 | improved |

## Worst-sequence class breakdown

### data39-1

- det HOTA `-6.027`; DetA `-7.489`; AssA `-2.603`; IDSW `+6`; Frag `-1`.
- Raw detection F1 `-7.426` and pair-link recall `-4.942`.

| class | GT dets | HOTA delta | DetA delta | AssA delta |
| --- | ---: | ---: | ---: | ---: |
| pedestrian | 416 | -10.382 | -9.557 | -11.763 |
| car | 231 | -1.519 | -1.871 | -0.606 |
| bike | 0 | +0.000 | +0.000 | +0.000 |
| van | 0 | +0.000 | +0.000 | +0.000 |
| truck | 0 | +0.000 | +0.000 | +0.000 |
| bus | 0 | +0.000 | +0.000 | +0.000 |
| tricycle | 0 | +0.000 | +0.000 | +0.000 |
| awning-bike | 0 | +0.000 | +0.000 | +0.000 |

### data28-4

- det HOTA `-5.236`; DetA `-10.597`; AssA `+0.628`; IDSW `-11`; Frag `-2`.
- Raw detection F1 `-7.181` and pair-link recall `-9.804`.

| class | GT dets | HOTA delta | DetA delta | AssA delta |
| --- | ---: | ---: | ---: | ---: |
| pedestrian | 50 | -18.904 | -30.901 | +8.061 |
| bike | 33 | -6.458 | -7.577 | -5.194 |
| van | 22 | -4.560 | -4.560 | -4.560 |
| car | 111 | -4.357 | -9.879 | +0.823 |
| bus | 0 | +0.000 | +0.000 | +0.000 |
| tricycle | 0 | +0.000 | +0.000 | +0.000 |
| awning-bike | 0 | +0.000 | +0.000 | +0.000 |
| truck | 15 | +11.861 | +8.207 | +17.147 |

### data41-1

- det HOTA `-4.078`; DetA `-5.818`; AssA `-1.595`; IDSW `+0`; Frag `+0`.
- Raw detection F1 `-5.017` and pair-link recall `-0.585`.

| class | GT dets | HOTA delta | DetA delta | AssA delta |
| --- | ---: | ---: | ---: | ---: |
| truck | 2 | -44.737 | -44.737 | -44.737 |
| bike | 34 | -4.532 | -4.168 | -5.118 |
| bus | 50 | -4.523 | -4.281 | -4.798 |
| pedestrian | 34 | -4.256 | -3.203 | -5.357 |
| car | 243 | -1.964 | -3.253 | -0.635 |
| van | 0 | +0.000 | +0.000 | +0.000 |
| tricycle | 0 | +0.000 | +0.000 | +0.000 |
| awning-bike | 0 | +0.000 | +0.000 | +0.000 |

### data47-3

- det HOTA `-3.695`; DetA `-6.136`; AssA `+0.134`; IDSW `+0`; Frag `-1`.
- Raw detection F1 `-6.553` and pair-link recall `-5.588`.

| class | GT dets | HOTA delta | DetA delta | AssA delta |
| --- | ---: | ---: | ---: | ---: |
| van | 159 | -13.182 | -9.288 | -18.675 |
| pedestrian | 353 | -5.729 | -5.925 | -5.439 |
| bike | 495 | -4.713 | -11.888 | +12.495 |
| car | 1422 | -3.714 | -6.644 | -0.169 |
| awning-bike | 202 | -1.915 | +4.268 | -16.132 |
| truck | 22 | +0.000 | +0.000 | +0.000 |
| bus | 0 | +0.000 | +0.000 | +0.000 |
| tricycle | 0 | +0.000 | +0.000 | +0.000 |

### data37-1

- det HOTA `-3.023`; DetA `-4.204`; AssA `-1.465`; IDSW `+32`; Frag `+104`.
- Raw detection F1 `-1.971` and pair-link recall `-5.646`.

| class | GT dets | HOTA delta | DetA delta | AssA delta |
| --- | ---: | ---: | ---: | ---: |
| tricycle | 113 | -29.257 | -19.168 | -30.481 |
| pedestrian | 5609 | -11.927 | -10.938 | -12.984 |
| van | 374 | -6.828 | -3.457 | -10.697 |
| awning-bike | 91 | -2.909 | -3.875 | -0.710 |
| bike | 1083 | -1.997 | -0.179 | -3.852 |
| bus | 0 | +0.000 | +0.000 | +0.000 |
| car | 6981 | +1.134 | +0.004 | +2.296 |
| truck | 69 | +29.067 | +26.262 | +31.501 |

## Key-sequence diagnosis

Identity continuity is audited by matching tracker outputs to GT at rotated IoU `0.5`. The
diagnostic switch count is independent from TrackEval's threshold-averaged IDSW definition.

| sequence | side | matched | switches | split GT tracks | identity purity | predicted IDs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| data39-1 | Base | 456 | 3 | 3 | 98.46% | 33 |
| data39-1 | 0718_01 | 402 | 9 | 7 | 95.52% | 35 |
| data37-1 | Base | 12323 | 100 | 56 | 91.46% | 257 |
| data37-1 | 0718_01 | 11336 | 134 | 59 | 90.97% | 274 |
| data28-2 | Base | 882 | 9 | 4 | 98.64% | 37 |
| data28-2 | 0718_01 | 859 | 21 | 12 | 91.74% | 39 |
| data34-1 | Base | 6699 | 476 | 90 | 59.70% | 357 |
| data34-1 | 0718_01 | 7302 | 449 | 82 | 62.53% | 333 |
| data30-3 | Base | 2897 | 50 | 36 | 90.78% | 167 |
| data30-3 | 0718_01 | 2956 | 39 | 27 | 93.57% | 156 |
| data28-6 | Base | 152 | 7 | 3 | 90.79% | 16 |
| data28-6 | 0718_01 | 195 | 1 | 1 | 99.49% | 15 |

### data39-1: largest absolute det regression

This sequence has the largest per-sequence det HOTA drop (`-6.027`). The loss is primarily
detection-side: DetA drops `7.489`, versus AssA `2.603`; raw precision falls
`81.942% -> 72.549%`, raw recall falls `74.343% -> 68.624%`, and matched track observations fall
`456 -> 402`. Pair-link recall also falls `70.181% -> 65.239%`, but wrong pair links remain exactly
`1` for both models. Switches rise `3 -> 9` only after the detector has already lost coverage.
Therefore this is a confidence/coverage regression, dominated by pedestrian (`-10.382` HOTA over
416 GT detections), rather than duplicate cross-ID pair assignment.

### data37-1: principal aggregate regression

This long sequence is the largest GT-weighted negative contributor. det HOTA drops `3.023`, with
DetA `-4.204` and AssA `-1.465`. Raw recall falls `88.142% -> 82.996%`, pair-link recall falls
`87.254% -> 81.607%`, and matched track observations fall `12323 -> 11336`. The dominant loss is
therefore missed detections/pair links; the increase in switches `100 -> 134` is secondary.
Pedestrian is the main affected class (`-11.927` HOTA over 5609 GT detections), while car improves
`1.134` and truck improves `29.067` on only 69 GT detections.

### data28-2: localized association regression

This is the clearest association-side failure: DetA improves `1.967`, but AssA drops `7.290`.
Wrong pair links increase `5 -> 8`, one-side failures increase `60 -> 156`, switches increase
`9 -> 21`, and identity purity falls `98.64% -> 91.74%`. It is qualitatively the same failure
family as the D6/D16 case, but much weaker than `0719_02`, whose AssA loss on this sequence was
`30.338` and switches rose to `53`.

### data34-1: principal aggregate improvement

det HOTA improves `2.364`, with DetA/AssA both improving `2.619/2.014`. Raw F1 rises `2.424`,
pair-link recall rises `3.313`, wrong links fall `121 -> 62`, matched observations increase
`6699 -> 7302`, and switches decrease `476 -> 449`. The apparent cls HOTA loss `-6.521` is a
class-average effect: pedestrian, which supplies 10773 GT detections, improves `3.933`, while
the much rarer van class (204 GT detections) drops `45.105` and receives equal class-average weight.

### data30-3 and data28-6: clean improvements

`data30-3` improves det/cls HOTA by `3.309/8.975`; wrong pair links fall `10 -> 3`, switches fall
`50 -> 39`, and purity rises `90.78% -> 93.57%`. `data28-6` is a short-sequence outlier with det
HOTA `+32.601`: raw recall rises `5.738`, pair-link recall rises `8.297`, switches fall `7 -> 1`,
and purity reaches `99.49%`. It is a valid improvement but should not be treated as representative
of the full validation set because it contains only 244 GT detections.

## Diagnostic videos

Full independent GT/Base/0718_01 videos are provided only for `data39-1`, `data37-1`, `data28-2`,
`data34-1`, `data30-3`, and `data28-6` under `videos/<sequence>/`. Detailed prev/curr pair-detection
plus previous-frame active-track videos are provided for `data39-1`, `data37-1`, `data28-2`, and
`data34-1` under `videos/pair_diagnostic/`.
Each pair video uses the tracker operating rule `prev_score >= 0.2 or curr_score >= 0.6`, labels
both side scores, and connects corresponding pair boxes across frames.

## Final attribution

`0718_01` is substantially healthier than `0719_02`: it delivers the largest cls HOTA gain over
Base (`+1.962`) while keeping det HOTA nearly unchanged (`-0.170`). Its remaining det deficit is
primarily a recall/fragmentation tradeoff concentrated in several dense sequences, especially
`data37-1`; it is not driven by a validation-wide increase in wrong pair association. The model
still has localized cross-ID failures (`data28-2`), so the anchor/cls-association concerns remain
real, but they are not the dominant global limitation of this checkpoint.

## Attribution rule

- `raw F1` measures class-agnostic side-box quality before tracking.
- `pair-link recall` measures whether both boxes of a pair detection recover the same persistent GT identity.
- `DetA` measures detection quality after tracker filtering; `AssA`, `IDSW`, and `Frag` measure final trajectory association.
- A stable/improved raw detector with lower AssA is tracking-stage association degradation. Lower raw F1 or pair-link recall indicates the pair detector supplies weaker boxes or correspondence.

Overview: `sequence_delta_overview.png`. Key-sequence videos: `videos/`. Machine-readable values:
`sequence_metrics.csv`.
