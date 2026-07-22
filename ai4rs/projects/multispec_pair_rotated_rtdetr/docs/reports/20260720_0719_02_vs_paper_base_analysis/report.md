# 0719_02 vs Paper Base: Detection and Tracking Diagnosis

Comparison protocol: Paper Base `0716_02` epoch 68 versus `0719_02` epoch 64. Both checkpoints are their unique maxima of `cls_HOTA + det_HOTA`; tracker parameters are identical.

## Overall decomposition

| metric | Base | 0719_02 | delta |
| --- | ---: | ---: | ---: |
| HOTA | 61.982 | 60.689 | -1.293 |
| DetA | 53.890 | 53.242 | -0.648 |
| AssA | 73.643 | 71.445 | -2.198 |
| DetRe | 61.913 | 61.304 | -0.609 |
| DetPr | 71.741 | 71.332 | -0.409 |
| MOTA | 60.599 | 59.359 | -1.240 |
| IDF1 | 72.303 | 70.487 | -1.816 |
| IDSW | 2136.000 | 2486.000 | +350.000 |
| Frag | 6428.000 | 6563.000 | +135.000 |
| CLR_FP | 24146.000 | 24834.000 | +688.000 |
| CLR_FN | 51021.000 | 52415.000 | +1394.000 |

In the exact identity `HOTA = sqrt(DetA * AssA)`, the negative log-HOTA change is `0.02120`. DetA contributes `28.5%` and AssA contributes `71.5%` of the loss.

**Primary conclusion:** the `-1.293` det HOTA regression is mainly an association/tracking-side
regression, not a general loss of detector capability. AssA explains `71.5%` of the HOTA loss,
while DetA explains `28.5%`; IDSW increases by `350` and Frag by `135`. Because both runs use
the same tracker configuration, this is not caused by a tracker-parameter change. It means that
`0719_02` supplies less temporally consistent scores, boxes, or pair correspondence to the fixed
tracker, causing more identity fragmentation.

## Raw pair-detection audit

Fixed diagnostic thresholds: side score >= `0.2` and rotated IoU >= `0.5`. This audit is not the paper AP protocol; it isolates detector outputs before tracking.

| metric | Base | 0719_02 | delta |
| --- | ---: | ---: | ---: |
| raw_precision | 80.234 | 78.148 | -2.086 |
| raw_recall | 79.916 | 79.653 | -0.263 |
| raw_f1 | 80.075 | 78.894 | -1.181 |
| pair_precision | 99.638 | 99.502 | -0.136 |
| pair_recall | 76.481 | 75.626 | -0.855 |

The paper pair-detection metrics move in the opposite direction: pair mAP improves from `0.3149`
to `0.3277`, and pair AP50 from `0.5225` to `0.5492`. Therefore, `0719_02` has better global
ranking quality, but at the tracker's fixed operating threshold its raw precision drops by
`2.086` points and persistent pair-link recall drops by `0.855` points. Pair detection is a
secondary contributor through calibration and temporal continuity, rather than the main source
of the final det HOTA drop.

## Class comparison from all_cls_summary

| class | Base HOTA | 0719_02 HOTA | delta | DetA delta | AssA delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| car | 81.516 | 80.103 | -1.413 | -0.002 | -2.929 |
| bike | 43.807 | 42.856 | -0.951 | -1.683 | +0.222 |
| pedestrian | 44.640 | 42.376 | -2.264 | -2.099 | -2.473 |
| van | 62.825 | 60.712 | -2.113 | -0.218 | -4.824 |
| truck | 33.852 | 45.681 | +11.829 | +13.429 | +1.575 |
| bus | 71.483 | 74.161 | +2.678 | +4.008 | +1.093 |
| tricycle | 39.283 | 45.920 | +6.637 | +9.167 | -0.045 |
| awning-bike | 49.109 | 49.709 | +0.600 | +0.573 | +0.677 |

## Sequence comparison from all_seq_summary

Rows are sorted by det HOTA delta. The winner fields are also embedded in every sequence visualization title.

Across all 50 sequences, `0719_02` wins det HOTA on `20` and Base wins on `30`; for cls HOTA,
`0719_02` wins on `33`, Base wins on `16`, and one sequence ties. The component diagnosis contains
`25` association-degraded sequences, `15` detection-degraded sequences, and `10` sequences where
both DetA and AssA improve.

| sequence | det winner | det HOTA delta | cls winner | cls HOTA delta | DetA delta | AssA delta | raw F1 delta | pair-link recall delta | diagnosis |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| data28-2 | BASE | -14.487 | BASE | -10.300 | +2.047 | -30.338 | -8.333 | -8.039 | association-only |
| data47-3 | BASE | -6.217 | BASE | -4.859 | -5.905 | -6.792 | -5.751 | -5.434 | association-dominant |
| data28-5 | BASE | -5.753 | BASE | -1.373 | -9.387 | -0.109 | -10.441 | -19.394 | detection-dominant |
| data39-1 | BASE | -5.003 | BASE | -1.199 | -5.442 | -5.616 | -6.359 | -6.590 | association-dominant |
| data23-1 | BASE | -4.773 | 0719_02 | +2.661 | +1.628 | -12.814 | -3.012 | -4.023 | association-only |
| data37-11 | BASE | -4.749 | 0719_02 | +3.970 | -0.892 | -8.668 | -1.522 | -1.466 | association-dominant |
| data47-2 | BASE | -4.445 | BASE | -3.223 | -8.868 | +2.623 | -4.583 | -5.221 | detection-only |
| data37-12 | BASE | -3.831 | TIE | -0.002 | -1.378 | -6.495 | -0.157 | -0.508 | association-dominant |
| data34-1 | BASE | -3.357 | BASE | -7.931 | -3.090 | -3.370 | -4.606 | -4.409 | association-dominant |
| data37-10 | BASE | -2.995 | BASE | -1.450 | -2.301 | -3.660 | -1.544 | -1.043 | association-dominant |
| data41-1 | BASE | -2.763 | BASE | -6.439 | -3.489 | -1.812 | -6.421 | -1.462 | detection-dominant |
| data37-2 | BASE | -2.697 | 0719_02 | +1.426 | -2.286 | -3.435 | -2.904 | -5.285 | association-dominant |
| data37-1 | BASE | -2.692 | 0719_02 | +7.434 | -2.646 | -2.533 | -1.591 | -4.525 | detection-dominant |
| data36-4 | BASE | -2.567 | 0719_02 | +1.955 | -2.068 | -2.664 | -2.439 | -3.249 | association-dominant |
| data39-6 | BASE | -2.542 | BASE | -2.384 | -1.709 | -3.798 | -4.771 | -4.798 | association-dominant |
| data47-4 | BASE | -2.169 | BASE | -1.700 | -2.490 | -1.653 | -4.064 | -2.653 | detection-dominant |
| data36-5 | BASE | -1.789 | 0719_02 | +3.072 | -1.067 | -2.918 | -2.582 | -2.119 | association-dominant |
| data30-5 | BASE | -1.570 | 0719_02 | +5.641 | -2.230 | -0.270 | -0.938 | -0.897 | detection-dominant |
| data36-11 | BASE | -1.495 | 0719_02 | +1.246 | -2.335 | -0.352 | +0.110 | +0.489 | detection-dominant |
| data39-2 | BASE | -1.427 | BASE | -0.206 | -2.736 | +1.246 | -1.224 | +3.304 | detection-only |
| data34-3 | BASE | -1.269 | BASE | -7.829 | -1.058 | -0.588 | +0.452 | -1.693 | detection-dominant |
| data40-3 | BASE | -1.130 | 0719_02 | +3.479 | +1.869 | -5.163 | -4.094 | -1.007 | association-only |
| data28-1 | BASE | -1.095 | BASE | -0.464 | +1.867 | -4.039 | +0.924 | +1.896 | association-only |
| data30-9 | BASE | -0.944 | 0719_02 | +2.113 | -2.043 | +0.694 | -1.854 | -1.358 | detection-only |
| data30-4 | BASE | -0.706 | 0719_02 | +6.262 | -0.485 | -0.842 | +0.067 | +0.204 | association-dominant |
| data42-1 | BASE | -0.590 | 0719_02 | +0.166 | -1.727 | +0.672 | -0.716 | -0.653 | detection-only |
| data46-12 | BASE | -0.567 | 0719_02 | +6.903 | -0.141 | -1.376 | +0.211 | +3.393 | association-dominant |
| data48-1 | BASE | -0.510 | BASE | -0.146 | +0.047 | -2.408 | +2.824 | -2.941 | association-only |
| data40-4 | BASE | -0.436 | 0719_02 | +0.391 | -2.439 | +3.790 | -3.752 | -3.356 | detection-only |
| data36-10 | BASE | -0.377 | 0719_02 | +3.382 | -0.400 | -0.472 | -0.573 | +0.900 | association-dominant |
| data34-2 | 0719_02 | +0.177 | BASE | -1.657 | +0.748 | -1.406 | +0.504 | +0.137 | association-only |
| data42-2 | 0719_02 | +0.302 | 0719_02 | +0.643 | +1.356 | -1.087 | +0.967 | +0.874 | association-only |
| data46-11 | 0719_02 | +0.334 | BASE | -1.138 | +2.784 | -3.769 | +2.430 | +4.959 | association-only |
| data49-2 | 0719_02 | +0.479 | 0719_02 | +4.295 | -0.876 | +2.640 | -3.128 | +0.211 | detection-only |
| data30-10 | 0719_02 | +0.559 | 0719_02 | +5.139 | +1.993 | -1.384 | +1.063 | +2.273 | association-only |
| data28-4 | 0719_02 | +0.813 | 0719_02 | +3.482 | +2.873 | -1.753 | +1.750 | +3.922 | association-only |
| data24-1 | 0719_02 | +0.915 | 0719_02 | +1.723 | +1.721 | -0.387 | -0.184 | +2.146 | association-only |
| data36-12 | 0719_02 | +0.988 | 0719_02 | +2.858 | +0.471 | +1.609 | -0.330 | -1.679 | improved |
| data30-2 | 0719_02 | +1.055 | 0719_02 | +8.816 | +1.183 | +0.799 | -0.100 | -0.703 | improved |
| data39-3 | 0719_02 | +1.083 | 0719_02 | +0.119 | +0.687 | +1.743 | -5.422 | +0.162 | improved |
| data36-13 | 0719_02 | +1.109 | 0719_02 | +0.265 | -0.044 | +2.758 | -0.454 | -1.916 | detection-only |
| data27-1 | 0719_02 | +1.223 | 0719_02 | +2.385 | +0.603 | +2.042 | +1.255 | +11.170 | improved |
| data42-3 | 0719_02 | +1.243 | 0719_02 | +0.451 | +0.998 | +1.359 | -1.854 | -0.079 | improved |
| data40-5 | 0719_02 | +1.546 | 0719_02 | +4.222 | +3.575 | -0.713 | -0.023 | +1.930 | association-only |
| data31-1 | 0719_02 | +2.139 | 0719_02 | +0.267 | -0.916 | +6.150 | -1.858 | -0.974 | detection-only |
| data30-3 | 0719_02 | +3.502 | 0719_02 | +10.807 | +3.866 | +3.182 | +2.294 | +1.909 | improved |
| data28-3 | 0719_02 | +7.669 | 0719_02 | +5.927 | +10.618 | +0.266 | +8.902 | +4.396 | improved |
| data47-1 | 0719_02 | +10.254 | 0719_02 | +2.067 | +10.952 | +8.052 | +1.172 | +17.083 | improved |
| data33-1 | 0719_02 | +10.381 | 0719_02 | +1.298 | +14.864 | +6.496 | +21.548 | +1.424 | improved |
| data28-6 | 0719_02 | +29.170 | 0719_02 | +16.581 | +26.090 | +20.687 | +18.655 | +21.397 | improved |

## Worst-sequence class breakdown

### data28-2

- det HOTA `-14.487`; DetA `+2.047`; AssA `-30.338`; IDSW `+40`; Frag `+0`.
- Raw detection F1 `-8.333` and pair-link recall `-8.039`.

| class | GT dets | HOTA delta | DetA delta | AssA delta |
| --- | ---: | ---: | ---: | ---: |
| awning-bike | 11 | -50.198 | -50.198 | -50.198 |
| bike | 24 | -26.711 | -10.548 | -41.091 |
| car | 807 | -17.559 | +0.531 | -33.250 |
| pedestrian | 23 | -10.104 | -10.104 | -10.104 |
| tricycle | 49 | -0.898 | +0.206 | -1.716 |
| van | 0 | +0.000 | +0.000 | +0.000 |
| bus | 0 | +0.000 | +0.000 | +0.000 |
| truck | 54 | +23.070 | +33.813 | +9.259 |

Visualization: `sequences/data28-2.jpg`.

### data47-3

- det HOTA `-6.217`; DetA `-5.905`; AssA `-6.792`; IDSW `+12`; Frag `+10`.
- Raw detection F1 `-5.751` and pair-link recall `-5.434`.

| class | GT dets | HOTA delta | DetA delta | AssA delta |
| --- | ---: | ---: | ---: | ---: |
| van | 159 | -24.747 | -16.880 | -36.289 |
| car | 1422 | -8.267 | -8.196 | -8.460 |
| bike | 495 | -4.824 | -10.712 | +8.585 |
| awning-bike | 202 | -0.853 | -0.822 | -0.498 |
| pedestrian | 353 | -0.183 | +3.552 | -4.394 |
| truck | 22 | +0.000 | +0.000 | +0.000 |
| bus | 0 | +0.000 | +0.000 | +0.000 |
| tricycle | 0 | +0.000 | +0.000 | +0.000 |

Visualization: `sequences/data47-3.jpg`.

### data28-5

- det HOTA `-5.753`; DetA `-9.387`; AssA `-0.109`; IDSW `+1`; Frag `-2`.
- Raw detection F1 `-10.441` and pair-link recall `-19.394`.

| class | GT dets | HOTA delta | DetA delta | AssA delta |
| --- | ---: | ---: | ---: | ---: |
| pedestrian | 97 | -11.126 | -15.925 | -3.965 |
| truck | 24 | -4.633 | +4.834 | -20.406 |
| car | 58 | -2.596 | -6.282 | +1.337 |
| van | 0 | +0.000 | +0.000 | +0.000 |
| bus | 0 | +0.000 | +0.000 | +0.000 |
| tricycle | 0 | +0.000 | +0.000 | +0.000 |
| awning-bike | 0 | +0.000 | +0.000 | +0.000 |
| bike | 4 | +7.369 | +7.369 | +7.369 |

Visualization: `sequences/data28-5.jpg`.

### data39-1

- det HOTA `-5.003`; DetA `-5.442`; AssA `-5.616`; IDSW `+6`; Frag `-9`.
- Raw detection F1 `-6.359` and pair-link recall `-6.590`.

| class | GT dets | HOTA delta | DetA delta | AssA delta |
| --- | ---: | ---: | ---: | ---: |
| pedestrian | 416 | -5.295 | -5.830 | -4.876 |
| car | 231 | -4.297 | -5.819 | -2.891 |
| bike | 0 | +0.000 | +0.000 | +0.000 |
| van | 0 | +0.000 | +0.000 | +0.000 |
| truck | 0 | +0.000 | +0.000 | +0.000 |
| bus | 0 | +0.000 | +0.000 | +0.000 |
| tricycle | 0 | +0.000 | +0.000 | +0.000 |
| awning-bike | 0 | +0.000 | +0.000 | +0.000 |

Visualization: `sequences/data39-1.jpg`.

### data23-1

- det HOTA `-4.773`; DetA `+1.628`; AssA `-12.814`; IDSW `+13`; Frag `-3`.
- Raw detection F1 `-3.012` and pair-link recall `-4.023`.

| class | GT dets | HOTA delta | DetA delta | AssA delta |
| --- | ---: | ---: | ---: | ---: |
| car | 966 | -6.426 | +2.125 | -15.390 |
| bike | 0 | +0.000 | +0.000 | +0.000 |
| van | 0 | +0.000 | +0.000 | +0.000 |
| truck | 0 | +0.000 | +0.000 | +0.000 |
| bus | 0 | +0.000 | +0.000 | +0.000 |
| tricycle | 0 | +0.000 | +0.000 | +0.000 |
| pedestrian | 294 | +0.300 | -4.765 | +8.567 |
| awning-bike | 13 | +27.412 | +27.412 | +27.412 |

Visualization: `sequences/data23-1.jpg`.

## Deep identity-continuity audit

The following diagnostic independently matches tracker outputs to GT at rotated IoU `0.5`, then
counts changes in the matched prediction ID for each GT trajectory. `matched` measures box coverage;
`switches` and `purity` measure identity continuity. These values are explanatory diagnostics and
are separate from TrackEval's threshold-averaged IDSW definition.

| sequence | side | matched | switches | split GT tracks | identity purity | predicted IDs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| data28-2 | Base | 882 | 9 | 4 | 98.64% | 37 |
| data28-2 | 0719_02 | 856 | 53 | 13 | 69.63% | 46 |
| data47-3 | Base | 2083 | 10 | 5 | 97.55% | 59 |
| data47-3 | 0719_02 | 1932 | 26 | 10 | 94.15% | 64 |
| data28-5 | Base | 137 | 1 |  | 98.54% |  |
| data28-5 | 0719_02 | 105 | 3 |  | 93.33% |  |
| data39-1 | Base | 456 | 3 |  | 98.46% |  |
| data39-1 | 0719_02 | 407 | 11 |  | 93.12% |  |
| data23-1 | Base | 894 | 6 |  | 96.76% |  |
| data23-1 | 0719_02 | 960 | 23 |  | 86.25% |  |
| data37-11 | Base | 10115 | 44 | 27 | 94.79% | 164 |
| data37-11 | 0719_02 | 10154 | 148 | 42 | 86.53% | 192 |
| data34-1 | Base | 6699 | 476 | 90 | 59.70% | 357 |
| data34-1 | 0719_02 | 6300 | 519 | 85 | 58.00% | 370 |
| data37-1 | Base | 12323 | 100 | 56 | 91.46% | 257 |
| data37-1 | 0719_02 | 11549 | 119 | 62 | 89.79% | 250 |
| data36-4 | Base | 10665 | 138 | 72 | 90.74% | 360 |
| data36-4 | 0719_02 | 10272 | 172 | 93 | 88.96% | 378 |

### Per-sequence attribution

- `data28-2` is the clearest tracking-side failure. DetA improves by `2.047`, but AssA drops by
  `30.338`; matched boxes are close, while switches rise `9 -> 53` and purity collapses
  `98.64% -> 69.63%`. The dominant car class alone loses `33.250` AssA points. This is identity
  fragmentation, not missing detections.
- `data28-5` is the clearest pair-detection-side failure. DetA drops `9.387`, raw F1 drops `10.441`,
  pair-link recall drops `19.394`, and matched boxes fall `137 -> 105`, while AssA is nearly flat.
  The main source is pedestrian recall, whose DetA drops `15.925`.
- `data47-3` and `data39-1` are mixed failures. Both raw detection and DetA decline, but switches
  and identity splits also rise; neither can be assigned solely to detection or association.
- `data23-1` is association-side despite better final detection coverage: DetA rises `1.628` and
  matched boxes rise `894 -> 960`, but AssA drops `12.814`, switches rise `6 -> 23`, and identity
  purity falls by `10.51` points.
- `data37-11` is the largest dataset-weighted negative contributor. Matched boxes remain effectively
  unchanged (`10115 -> 10154`), yet switches more than triple (`44 -> 148`) and predicted IDs rise
  `164 -> 192`; its `-4.749` det HOTA is association-driven.
- `data34-1`, `data37-1`, and `data36-4` are long, dense sequences where both coverage and identity
  continuity worsen. Together with `data37-11`, they dominate the aggregate regression more than
  the visually largest but shorter sequence `data28-2`.

## Final attribution

The det-side drop is **primarily a tracking association problem induced by less temporally stable
model outputs**, with a smaller pair-detection operating-point problem. In practical terms:

1. The detector is not globally weaker: pair mAP and AP50 both improve.
2. At score `0.2`, precision and pair-link recall are lower, so some sequences lose usable pair
   detections before tracking (`data28-5`, part of `data47-3` and `data39-1`).
3. The larger aggregate loss occurs after boxes enter the tracker: AssA, IDSW, Frag, and the
   identity-purity audit all show fragmentation (`data28-2`, `data23-1`, `data37-11`).
4. Since tracker parameters are identical, the next fix should target pair-output temporal
   consistency and score calibration, rather than changing the tracker to hide the regression.

## Full-sequence videos

Each of the 50 sequences has three independent full-length videos: `GT`, `BASE`, and `0719_02`.
The source image remains at its native `1200x900` resolution; a separate 92-pixel header records
the sequence, frame, result name, det/cls winner, and DetA/AssA deltas without covering the image.
Rotated boxes and trajectory IDs are rendered directly on every frame. The 150 videos contain
16398 encoded frames in total and are stored under `videos/<sequence>/`.

Examples:

- `videos/data28-2/data28-2_GT.mp4`
- `videos/data28-2/data28-2_BASE.mp4`
- `videos/data28-2/data28-2_0719_02.mp4`
- `videos/data37-11/data37-11_BASE.mp4`
- `videos/data37-11/data37-11_0719_02.mp4`

### data28-2 pair-detection/track diagnostic

Two additional side-by-side videos expose the tracker input for every transition in `data28-2`:
the left panel is the previous frame with pair detections and active tracks, while the right panel
is the current frame with pair detections. Corresponding previous/current pair boxes share a
`D<index>` and color and are joined by a cross-panel line. Labels `p=<score>` and `c=<score>` are
the two side scores; magenta `T<id> s=<score>` boxes are active previous-frame tracks. To match the
actual tracker operating point, the visualization includes detections that can participate in
matching (`prev_score >= 0.2`) or birth (`curr_score >= 0.6`), instead of drawing all 300 low-score
decoder queries.

- `videos/pair_diagnostic/data28-2_BASE_pair_det_track.mp4`
- `videos/pair_diagnostic/data28-2_0719_02_pair_det_track.mp4`

## Attribution rule

- `raw F1` measures class-agnostic side-box quality before tracking.
- `pair-link recall` measures whether both boxes of a pair detection recover the same persistent GT identity.
- `DetA` measures detection quality after tracker filtering; `AssA`, `IDSW`, and `Frag` measure final trajectory association.
- A stable/improved raw detector with lower AssA is tracking-stage association degradation. Lower raw F1 or pair-link recall indicates the pair detector supplies weaker boxes or correspondence.

Overview: `sequence_delta_overview.png`. Full videos: `videos/<sequence>/*.mp4`. Static contact
sheets: `sequences/*.jpg`. Machine-readable values: `sequence_metrics.csv`.
