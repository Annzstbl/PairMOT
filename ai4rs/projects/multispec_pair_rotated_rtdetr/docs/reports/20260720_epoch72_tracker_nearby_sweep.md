# Epoch-72 Tracker Nearby Sweep: Paper Base vs 0718_01

## 1. Protocol

This sweep reuses the cached epoch-72 pair detections and only reruns PairMOT
tracking and TrackEval. No detector inference or checkpoint modification is
involved.

- Paper Base cache: `0716_02 ... /val_det/epoch_71`
- `0718_01` cache: `0718_01 ... /val_det/epoch_71`
- Fixed parameters: `match_iou_th=0.25`, `new_birth_iou_th=0.50`,
  `init_same_iou_th=0.30`, `class_aware=False`
- Search grid: `new_born_th in {0.58, 0.60}`,
  `track_th in {0.18, 0.20}`, `max_age in {8, 15}`
- The original `0.60/0.20/max_age=30` result is included as a reference.
- Base and `0718_01` always use the same tracker parameters in each row.

Search artifacts:

- Base: `/data4/litianhao/PairMmot/workdir_99/tracker_sweeps/20260720_base_epoch72_nearby`
- `0718_01`: `/data4/litianhao/PairMmot/workdir_99/tracker_sweeps/20260720_0718_01_epoch72_nearby`

## 2. HOTA Results

Rows are ordered by the tracker grid, not by a separately tuned parameter set
for each model. `delta` is `0718_01 - Base` under exactly the same parameters.

| new born | track | max age | Base cls HOTA | Base det HOTA | 0718_01 cls HOTA | 0718_01 det HOTA | cls delta | det delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.60 | 0.20 | 30 (original) | 53.035 | 61.955 | 55.002 | 61.993 | +1.967 | +0.038 |
| 0.58 | 0.18 | 8 | 53.056 | 61.891 | 54.866 | 61.932 | +1.810 | +0.041 |
| 0.58 | 0.18 | 15 | 53.046 | 61.881 | 54.869 | 61.913 | +1.823 | +0.032 |
| 0.58 | 0.20 | 8 | 53.065 | 61.967 | 55.034 | 61.951 | +1.969 | -0.016 |
| 0.58 | 0.20 | 15 | 53.049 | 61.955 | 55.039 | 61.947 | +1.990 | -0.008 |
| 0.60 | 0.18 | 8 | 53.060 | 61.931 | 54.839 | 61.986 | +1.779 | +0.055 |
| 0.60 | 0.18 | 15 | 53.051 | 61.921 | 54.830 | 61.961 | +1.779 | +0.040 |
| 0.60 | 0.20 | 8 | **53.069** | **61.997** | **55.004** | **61.995** | +1.935 | -0.002 |
| 0.60 | 0.20 | 15 | 53.053 | 61.984 | 55.001 | 61.990 | +1.948 | +0.006 |

The unique per-model maximum of `cls HOTA + det HOTA` in this table is the
`0.60/0.20/max_age=8` row for both models. Relative to the original epoch-72
tracker, its HOTA changes are:

| model | cls HOTA change | det HOTA change | HOTA-sum change |
| --- | ---: | ---: | ---: |
| Paper Base | +0.034 | +0.042 | +0.076 |
| `0718_01` | +0.002 | +0.002 | +0.004 |

The `0718_01` change is too small to treat as a meaningful metric gain. The
shorter lifetime benefits Base more, and at `max_age=8` the same-parameter det
HOTA difference becomes `-0.002`. If a shorter lifetime is required for model
logic, `max_age=15` is the conservative choice: it retains a same-parameter
`+1.948/+0.006` cls/det HOTA advantage while remaining close to the original
absolute metrics.

## 3. Lifecycle Decomposition

Keeping the two score thresholds unchanged and shortening `max_age` from 30
to 8 gives:

| model | max age | cls MOTA | cls IDF1 | det MOTA | det IDF1 | FN | FP | IDSW | Frag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Paper Base | 30 | 44.490 | 61.940 | 60.598 | 72.258 | 50,906 | 24,227 | 2,171 | 6,436 |
| Paper Base | 8 | 44.633 | 61.997 | 60.660 | 72.334 | 50,978 | 24,064 | 2,140 | 6,412 |
| `0718_01` | 30 | 48.801 | 65.381 | 60.933 | 72.754 | 51,085 | 23,469 | 2,093 | 6,534 |
| `0718_01` | 8 | 48.843 | 65.386 | 60.970 | 72.731 | 51,132 | 23,366 | 2,077 | 6,520 |

Shortening the lifetime removes stale trajectories: FP, IDSW and Frag all
decrease. The cost is a small FN increase because tracks are removed before a
later pair can reactivate them. Base gains more from stale-track removal than
`0718_01`, which explains why the relative det advantage does not increase.

Lowering `track_th` to `0.18` is not recommended. It can increase the
same-parameter det advantage of `0718_01`, but lowers its cls HOTA by about
`0.13--0.17` and lowers the total HOTA selection objective. Lowering
`new_born_th` to `0.58` also increases FP and does not produce a stronger
common operating point.

## 4. Conclusion of the Shortened-Age Sweep

At epoch 72 and the original common tracker parameters, `0718_01` already
exceeds the same-epoch Base by `+1.967` cls HOTA and `+0.038` det HOTA. This is
different from comparing `0718_01` epoch 72 against the Base's independently
selected epoch 68.

The nearby tracker search does not reveal a parameter set that materially
improves `0718_01` beyond the original setting. Use one of the following fixed
choices:

1. Keep `max_age=30` when the objective is the strongest and clearest
   same-parameter Liquid advantage.
2. Use `max_age=15` when a shorter lost-track lifetime is required; it changes
   absolute metrics negligibly and preserves a small positive det delta.

Do not tune Base and Liquid with different tracker parameters. The chosen
common tracker setting must be frozen before final test-set evaluation.

## 5. Fixed-Age-30 Sweep with 0.05 Steps

The first sweep changed score thresholds by `0.02` and did not test those
changes while keeping `max_age=30`. A second sweep therefore fixes
`max_age=30` and uses the original parameters
`0.60/0.20/0.25/0.50` as the center. Every changed scalar differs by exactly
`0.05`:

- `new_born_th`: `0.55`, `0.65`
- `track_th`: `0.15`, `0.25`
- `match_iou_th`: `0.20`, `0.30`
- `new_birth_iou_th`: `0.45`, `0.55`
- one final combination: `match_iou_th=0.20` and
  `new_birth_iou_th=0.45`

This gives nine new parameter groups plus the original reference. All rows use
the same parameters for Base and `0718_01`.

| new born | track | match IoU | birth IoU | Base cls HOTA | Base det HOTA | 0718_01 cls HOTA | 0718_01 det HOTA | cls delta | det delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.60 | 0.20 | 0.25 | 0.50 | 53.035 | 61.955 | 55.002 | 61.993 | +1.967 | +0.038 |
| 0.55 | 0.20 | 0.25 | 0.50 | 53.024 | 61.852 | 55.058 | 61.842 | +2.034 | -0.010 |
| 0.65 | 0.20 | 0.25 | 0.50 | 52.995 | 61.988 | 54.845 | 62.007 | +1.850 | +0.019 |
| 0.60 | 0.15 | 0.25 | 0.50 | 53.074 | 61.861 | 55.047 | 61.871 | +1.973 | +0.010 |
| 0.60 | 0.25 | 0.25 | 0.50 | 53.051 | 61.927 | 54.892 | 62.000 | +1.841 | +0.073 |
| 0.60 | 0.20 | 0.20 | 0.50 | 53.063 | 61.964 | 55.095 | 62.033 | +2.032 | +0.069 |
| 0.60 | 0.20 | 0.30 | 0.50 | 52.954 | 61.880 | 54.931 | 61.974 | +1.977 | +0.094 |
| 0.60 | 0.20 | 0.25 | 0.45 | 53.047 | 61.972 | 55.019 | 62.010 | +1.972 | +0.038 |
| 0.60 | 0.20 | 0.25 | 0.55 | 53.031 | 61.950 | 54.996 | 61.981 | +1.965 | +0.031 |
| **0.60** | **0.20** | **0.20** | **0.45** | **53.075** | **61.980** | **55.114** | **62.052** | **+2.039** | **+0.072** |

The last row is the unique maximum of `cls HOTA + det HOTA` for `0718_01` in
this fixed-age sweep. It also gives the largest same-parameter HOTA-sum margin
over Base. Relative to the original tracker parameters:

| model | cls HOTA change | det HOTA change | HOTA-sum change |
| --- | ---: | ---: | ---: |
| Paper Base epoch 72 | +0.040 | +0.025 | +0.065 |
| `0718_01` epoch 72 | +0.112 | +0.059 | +0.171 |

Detailed metrics for the selected setting are:

| model | cls HOTA | cls MOTA | cls IDF1 | det HOTA | DetA | AssA | det MOTA | det IDF1 | FN | FP | IDSW | Frag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Paper Base | 53.075 | 44.453 | 62.039 | 61.980 | 53.860 | 73.691 | 60.584 | 72.335 | 50,789 | 24,414 | 2,129 | 6,480 |
| `0718_01` | 55.114 | 48.882 | 65.617 | 62.052 | 53.839 | 73.979 | 60.965 | 72.936 | 50,949 | 23,587 | 2,048 | 6,583 |

Lowering `match_iou_th` from `0.25` to `0.20` is the main effective change. It
allows valid pair detections with moderate previous-side overlap to continue
tracks and improves both HOTA axes for `0718_01`. Lowering
`new_birth_iou_th` from `0.50` to `0.45` then suppresses more duplicate births
and adds a smaller complementary gain. Changes to `new_born_th` or
`track_th` trade cls HOTA for det HOTA and reduce the selection objective.

The selected epoch-72 common tracker parameters are therefore:

```text
new_born_th       = 0.60
track_th          = 0.20
match_iou_th      = 0.20
new_birth_iou_th  = 0.45
max_age           = 30
```

This section only establishes the best common operating point for the two
epoch-72 caches requested here. The formal paper table still selects each
experiment's epoch by the unique maximum of `cls HOTA + det HOTA`. Before
replacing formal numbers, this common tracker setting must be rerun on all
candidate epochs used by that selection protocol.
