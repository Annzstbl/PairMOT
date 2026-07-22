# 0717_01 Set-Transport vs Paper Base Visualization

## Protocol

- Paper Base `0716_02`: epoch 68 (`val_track_0017`).
- Set-Transport `0717_01`: epoch 68 (`val_track_0017`), the unique maximum of
  `cls_HOTA + det_HOTA`.
- Both use full HSMOT, 1200x900 ordered pairs and the same tracking protocol.
- The formal `0717_01` run was completed on AutoDL. Its archived epoch-68 TrackEval predictions
  were restored from `/autodl-fs/data/PairMOT_results/0717_01`.

## Overall result

| experiment | cls HOTA | det HOTA | sum |
| --- | ---: | ---: | ---: |
| Paper Base | 53.314 | 61.982 | 115.296 |
| `0717_01` Set-Transport | **54.941** | 61.836 | **116.777** |
| delta | **+1.627** | -0.146 | **+1.481** |

Across 50 sequences, `0717_01` wins cls HOTA on 30, loses on 18 and ties on 2. It wins det HOTA
on 23, loses on 26 and ties on 1. This is consistent with the aggregate result: Set-Transport
improves class-aware tracking broadly, while class-agnostic detection/tracking remains close to Base.

## Visualized key sequences

Each sequence has three independent videos under `videos/<sequence>/`: GT, Base and `0717_01`.
Every title states the det/cls winner and the sequence-level DetA/AssA deltas.

| sequence | role | det HOTA delta | cls HOTA delta | DetA delta | AssA delta | GT detections |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `data34-1` | largest dataset-weighted improvement | +2.968 | +2.969 | +1.358 | +5.789 | 11860 |
| `data37-1` | largest dataset-weighted regression | -1.552 | +1.766 | -2.072 | -0.849 | 14320 |
| `data36-4` | second-largest weighted regression | -1.438 | -1.850 | -1.625 | -0.996 | 13050 |
| `data47-3` | strongest meaningful absolute regression | -4.373 | -3.379 | -4.591 | -3.812 | 2653 |
| `data30-4` | cls/det disagreement; association-side loss | -1.764 | +11.725 | -0.244 | -3.533 | 3038 |
| `data28-2` | cls/det disagreement; det improves but cls drops | +1.105 | -9.464 | +2.148 | -0.159 | 968 |
| `data47-1` | largest absolute improvement; short sequence | +13.712 | +2.440 | +11.629 | +13.796 | 246 |

`data34-1` is the most representative positive case because it is long and improves both DetA and
AssA. `data37-1` and `data36-4` dominate the negative side by dataset size. `data30-4` is especially
useful for diagnosing the difference between class-aware gains and class-agnostic association:
cls HOTA rises sharply while AssA falls. `data28-2` provides the opposite disagreement: its DetA
and det HOTA improve, but cls HOTA drops sharply, so its videos should be inspected for class-specific
trajectory errors rather than a general box-coverage failure.

For `data28-2`, the original epoch-68 validation detection cache was recovered from the formal AutoDL
workdir. Two additional videos are provided under `videos/pair_diagnostic/`, one for Base and one for
`0717_01`. They use the same diagnostic protocol as the earlier comparisons: the left panel contains
previous-frame detections and active tracks, the right panel contains current-frame detections, pair
detections are connected across panels, and both side scores are labeled. The display rule is
`prev_score >= 0.2 or curr_score >= 0.6`.

Overview: `sequence_delta_overview.png`. Machine-readable sequence metrics: `sequence_metrics.csv`.
