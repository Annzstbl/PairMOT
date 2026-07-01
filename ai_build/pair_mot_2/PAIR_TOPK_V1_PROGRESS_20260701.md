# PairTopK-v1 Progress

Date: 2026-07-01

## Goal

Start the pair-detection innovation line after the current PairMOT baseline.
The first stage replaces the previous-frame-only `dual_topk` proposal
initialization with independent dual-frame proposals, proposal matching, and
query/reference fusion.

## Implemented

Code:

```text
projects/multispec_pair_rotated_rtdetr/multispec_pair_rotated_rtdetr/multispec_pair_rotated_rtdetr.py
```

New query init mode:

```python
query_init='pair_topk_v1'
```

Main behavior:

- Generate encoder proposals independently for prev and curr frames.
- Select per-frame top-k proposals.
- Score prev/curr proposal pairs using:
  - cosine similarity of proposal query features;
  - normalized center/scale geometry consistency;
  - fused single-frame class prior.
- Use a lightweight greedy matching:
  - each prev proposal takes its best curr proposal;
  - candidates are sorted by best pair score;
  - already-used prev/curr proposals are skipped.
- Add dustbin-style candidates:
  - unmatched high-score curr proposals become birth candidates;
  - unmatched high-score prev proposals become death/lost candidates.
- Fuse pair query content with an average-initialized linear layer:

```python
pair_query_fusion([query_prev, query_curr]) -> query_pair
```

This first version does not yet add proposal loss. It only changes decoder
query/reference initialization.

## Config

New config:

```text
projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_pairtopk_v1.py
```

Important settings:

```python
pair_proposal_cfg = dict(
    pre_topk=600,
    sim_weight=1.0,
    geom_weight=1.0,
    score_weight=1.0,
    geom_sigma=0.08,
    max_center_dist=0.35,
    max_log_scale=1.2,
    birth_score_thr=0.35,
    death_score_thr=0.35,
    enable_birth=True,
    enable_death=True,
)
```

Relaxed/shorter innovation training:

```python
max_epochs = 48
val_interval = 4
EarlyStoppingHook:
  monitor = pair/independent_AP50
  min_delta = 0.001
  patience = 3
```

## Validation Before Training

Static check:

```text
python -m py_compile multispec_pair_rotated_rtdetr.py
```

Model construction:

```text
MultispecPairRotatedRTDETR pair_topk_v1
pair_query_fusion True
```

Tensor smoke:

```text
query    (1, 300, 256), finite=True
ref_prev (1, 300, 5), in sigmoid range
ref_curr (1, 300, 5), in sigmoid range
```

Real-data single-batch loss smoke:

```text
elapsed_sec = 0.957
loss_sum    = 58.6697
```

An earlier full-matrix greedy matching implementation was too slow and was
interrupted. The bottleneck was Python iteration over flattened `Kp*Kc` match
candidates. It was replaced with per-prev best-curr matching.

## Current Training Run

Launch script:

```text
/data/users/litianhao01/PairMmot/workdir/_aux/scripts/launch_pairtopk_v1_gap1train_20260701.sh
```

Launch log:

```text
/data/users/litianhao01/PairMmot/workdir/_ops/launch_logs/pairtopk_v1_gap1train_20260701.nohup.log
```

Work dir:

```text
/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_pairtopk_v1
```

Latest run log:

```text
/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_pairtopk_v1/20260701_005031/20260701_005031.log
```

Process:

```text
GPU 1/2, 2-card distributed training
PID parent: 716617
rank0: 716699
rank1: 716700
```

First visible training log:

```text
Epoch(train) [1][50/484]
time: 2.0606
loss: 18.0802
loss_cls: 0.8583
loss_bbox_prev: 0.6224
loss_bbox_curr: 0.6120
loss_iou_prev: 0.5841
loss_iou_curr: 0.5706
```

## PairTopK-v1 Result

The first free-matching implementation finished with early stopping at epoch
20.

| epoch | pair_AP50 | independent_AP50 | independent_mAP50_95 | assoc_gap_AP50 | match_ratio | mean_iou_prev | mean_iou_curr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.3908 | 0.4508 | 0.2502 | 0.0600 | 0.7693 | 0.6536 | 0.6518 |
| 8 | 0.4001 | 0.4576 | 0.2690 | 0.0575 | 0.7577 | 0.6658 | 0.6726 |
| 12 | 0.3947 | 0.4519 | 0.2648 | 0.0572 | 0.7657 | 0.6661 | 0.6732 |
| 16 | 0.3841 | 0.4430 | 0.2556 | 0.0588 | 0.7603 | 0.6612 | 0.6700 |
| 20 | 0.4025 | 0.4545 | 0.2682 | 0.0520 | 0.7601 | 0.6682 | 0.6754 |

Baseline `dual_topk` gap1 reached roughly `pair_AP50=0.4353` and
`independent_AP50=0.4846` at epoch 72. PairTopK-v1 is therefore worse in
detection quality. `assoc_gap_AP50 = independent_AP50 - pair_AP50`, so smaller
is better when detection AP is comparable; a larger gap means pair association
lost more AP relative to independent detection. Current interpretation: free
proposal matching and birth/death candidates make proposal initialization less
stable than the original aligned top-k path.

## PairTopK Same-Index v1

Implemented a conservative follow-up mode:

```python
query_init='pair_topk_sameidx_v1'
pair_proposal_cfg=dict(
    sameidx_score_mode='sqrt',
    sameidx_ref_source='frame',
)
```

Main behavior:

- Generate encoder proposal memory independently on prev and curr frames.
- Compute prev/curr class scores at each aligned flattened spatial index.
- Select top-k by a joint score, default `sqrt(score_prev * score_curr)`.
- Fuse query content as `pair_query_fusion([query_prev, query_curr])`.
- Keep reference regression frame-specific:
  - prev reference uses prev query and prev proposal;
  - curr reference uses curr query and curr proposal.
- No free cross-index matching and no birth/death dustbin in this version.

This isolates whether dual-frame proposal scoring/query fusion helps while
preserving the original RT-DETR proposal ordering.

New files:

```text
projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_pairtopk_sameidx_v1.py
/data/users/litianhao01/PairMmot/workdir/_aux/scripts/launch_pairtopk_sameidx_v1_gap1train_20260701.sh
```

Validation before training:

```text
py_compile: passed
model build: MultispecPairRotatedRTDETR pair_topk_sameidx_v1
real batch backward: elapsed_sec=1.028, loss_sum=69.0454, 42 loss keys
```

Current run:

```text
work_dir: /data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_pairtopk_sameidx_v1
run_log:  /data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_pairtopk_sameidx_v1/20260701_103014/20260701_103014.log
GPU:      0,1
parent:   807141
rank0:    807230
rank1:    807231
```

First visible training logs:

```text
Epoch 1 iter 50:  time=0.8185, loss=18.3477, grad_norm=33.8543
Epoch 1 iter 100: time=0.7846, loss=17.5096, grad_norm=28.8816
```

First validation is scheduled at epoch 4.

Epoch-4 validation result:

| mode | epoch | pair_AP50 | independent_AP50 | independent_mAP50_95 | association_gap_AP50 | match_ratio | mean_iou_prev | mean_iou_curr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sameidx sqrt | 4 | 0.3472 | 0.4109 | 0.2237 | 0.0637 | 0.7790 | 0.6451 | 0.6313 |
| sameidx sqrt | 8 | 0.3662 | 0.4311 | 0.2382 | 0.0649 | 0.8091 | 0.6599 | 0.6483 |

This is worse than both PairTopK-v1 epoch 4
(`pair_AP50=0.3908`, `independent_AP50=0.4508`) and the later baseline. The
larger `association_gap_AP50` is not a positive signal; it means pair AP falls
farther behind independent detection AP. Interpretation: using
`sqrt(score_prev * score_curr)` for top-k selection is not conservative enough;
the current-frame score can suppress useful previous-frame proposals and hurt
detector quality.

Epoch 8 improves over epoch 4 but remains well below PairTopK-v1 epoch 8
(`pair_AP50=0.4001`, `independent_AP50=0.4576`) and the original dual-topk
baseline (`pair_AP50=0.4353`, `independent_AP50=0.4846`). Current conclusion:
same-index query/reference fusion may be usable, but `sqrt(prev*curr)` top-k
selection is too aggressive for detector quality.

Decision: stopped sameidx-sqrt after epoch-8 validation. It was still training
through epoch 10, but the two completed validation points were enough to reject
the idea as a mainline direction. Continuing to epoch 12 would likely only
confirm a clearly worse branch while occupying GPU 0,1.

Follow-up started:

```text
projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_pairtopk_sameidx_prevscore_v1.py
/data/users/litianhao01/PairMmot/workdir/_aux/scripts/launch_pairtopk_sameidx_prevscore_v1_gap1train_20260701.sh
```

This keeps same-index query/reference fusion but selects top-k only by the prev
frame score:

```python
pair_proposal_cfg=dict(
    sameidx_score_mode='prev',
    sameidx_ref_source='frame',
)
```

Rationale: this isolates whether the AP drop comes from dual-frame score
selection or from query/reference fusion itself. It should be closer to the
original `dual_topk` proposal order.

Launch:

```text
work_dir: /data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_pairtopk_sameidx_prevscore_v1
GPU:      2,3
parent:   834283
```

Runtime status:

```text
2026-07-01 11:21-11:35: training normal through epoch 3 start.
No NaN, Traceback, or distributed process exit observed.
Epoch 4 validation completed at 2026-07-01 12:07:43.
```

Epoch-4 validation result:

| mode | epoch | pair_AP50 | independent_AP50 | independent_mAP50_95 | association_gap_AP50 | match_ratio | mean_iou_prev | mean_iou_curr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sameidx prevscore | 4 | 0.3469 | 0.4307 | 0.2359 | 0.0838 | 0.7983 | 0.6691 | 0.6330 |

Decision: stopped early after epoch-4 validation because it is clearly worse
than PairTopK-v1 epoch 4 (`pair_AP50=0.3908`, `independent_AP50=0.4508`) and
does not improve over sameidx-sqrt in pair AP. The large association gap means
the independent detection gain over sameidx-sqrt does not transfer into pair AP.
Stopped processes on GPU 2,3 and released both GPUs.

After stopping sameidx-sqrt as well, all four GPUs were released.

## Validation Log Readability

Added formatted validation metric table output in:

```text
projects/multispec_pair_rotated_rtdetr/multispec_pair_rotated_rtdetr/pair_overfit_metric.py
```

The metric still returns the original flat dict, so `LoggerHook`,
`EarlyStoppingHook`, TensorBoard/scalars, and training-curve generation keep the
same keys. The added table is for human-readable logs only, with sections for
Detection AP, Pair AP, Matching Diagnostics, and optional per-gap AP summaries.

Verification:

```text
python -m py_compile projects/multispec_pair_rotated_rtdetr/multispec_pair_rotated_rtdetr/pair_overfit_metric.py
formatting smoke test in py310: passed
```

Note: already-running training processes loaded the old module before this
change, so they will not show the table until restarted/resumed or until a new
experiment is launched.

## Proposal Loss Finding

Original single-frame RT-DETR has encoder proposal supervision: it passes
selected `topk_score` and `topk_coords` as `enc_outputs_class` and
`enc_outputs_coord` during training. Current `PairRotatedRTDETRHead` accepts
these arguments but deletes them in `loss_by_feat`, so pair detection currently
does not supervise encoder proposals directly. This is a likely v2 improvement
after same-index v1 is evaluated.
