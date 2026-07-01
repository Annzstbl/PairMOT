# Both-Visible Dual-Cls No-Presence Experiment Review

Date: 2026-07-01

## 1. Goal

This change implements a new pair-detection training variant for review before launching training.

Requested design:

1. Training only uses GT targets visible in both frames of the pair.
2. Remove the `presence_prev` / `presence_curr` branches from the new experiment.
3. Replace shared pair cls with separated per-frame cls predictions:
   - `cls_prev`
   - `cls_curr`
4. Update Hungarian matching, AP metric, and tracking compatibility.
5. Use the stable baseline proposal mode, not PairTopK-v1 or sameidx variants.
6. Validation logs should keep table formatting and output per-class AP values.
7. Early stopping should monitor `pair_AP50`.
8. After training, select the best `pair_AP50` checkpoint and run tracking with:
   `pairmot_gap1_hota_search1_nb0.6_tr0.2_iou0.25_birthiou0.5_age30`

Training has not been started yet.

## 2. New Experiment Config

New config:

`projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_bothvis_dualcls_nopres.py`

Main settings:

```python
model.update(
    query_init='dual_topk',
    pair_dn_cfg=None,
)

model.bbox_head.update(
    type='PairRotatedRTDETRHead',
    use_presence=False,
    dual_cls=True,
    train_both_visible_only=True,
    dn_loss_weight=0.0,
)

val_evaluator['metrics'].update(
    both_visible_gt_only=True,
)
```

Work dir:

`/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_bothvis_dualcls_nopres`

Proposal mode:

- Uses `dual_topk`, same as the stable baseline.
- Does not use `pair_topk_v1`.
- Does not use `pair_topk_sameidx_v1`.

PairDN:

- Disabled in this config.
- Reason: current PairDN target generation contains presence supervision. Adapting PairDN to dual-cls/no-presence is possible, but would add another variable to the first experiment.

Early stopping:

```python
monitor='pair/pair_AP50'
rule='greater'
min_delta=0.001
patience=4
```

## 3. Head Changes

Modified file:

`projects/multispec_pair_rotated_rtdetr/multispec_pair_rotated_rtdetr/pair_rotated_rtdetr_head.py`

The existing `PairRotatedRTDETRHead` was extended with config flags:

```python
use_presence: bool = True
dual_cls: bool = False
train_both_visible_only: bool = False
```

Default behavior remains compatible with previous experiments.

For the new experiment:

- `use_presence=False`
- `dual_cls=True`
- `train_both_visible_only=True`

### 3.1 Output Structure

Old head:

```text
shared cls
presence_prev
presence_curr
box_prev
box_curr
```

New head:

```text
cls_prev
cls_curr
box_prev
box_curr
```

At inference, the predicted pair score is:

```python
pair_score = sqrt(score_prev * score_curr)
```

The prediction result stores:

```text
scores
labels
bboxes_prev
bboxes_curr
scores_prev
scores_curr
labels_prev
labels_curr
```

No `presence_prev` or `presence_curr` is emitted by the new head.

### 3.2 GT Filtering

The new head filters training GT internally:

```python
valid = pair_gt.valid_prev.bool() & pair_gt.valid_curr.bool()
```

Only both-visible GT instances are used for Hungarian assignment and loss.

This keeps the pair-detection objective focused on real pair association rather than rare birth/death cases.

## 4. Loss Changes

Old loss:

```text
loss_cls
loss_pres_prev
loss_pres_curr
loss_bbox_prev
loss_bbox_curr
loss_iou_prev
loss_iou_curr
```

New loss:

```text
loss_cls_prev
loss_cls_curr
loss_cls = loss_cls_prev + loss_cls_curr
loss_bbox_prev
loss_bbox_curr
loss_iou_prev
loss_iou_curr
```

Decoder auxiliary layers also output:

```text
d0.loss_cls_prev / d0.loss_cls_curr / d0.loss_cls
d1.loss_cls_prev / d1.loss_cls_curr / d1.loss_cls
...
```

No presence loss is computed in this variant.

## 5. Hungarian Matching

The assigner class is unchanged:

`PairHungarianAssigner`

But the new config removes presence costs.

Old cost set:

```text
FocalLossCost
PairChamferCost(prev)
PairChamferCost(curr)
PairGDCost(prev)
PairGDCost(curr)
PairPresenceBCECost(prev)
PairPresenceBCECost(curr)
```

New cost set:

```text
FocalLossCost
PairChamferCost(prev)
PairChamferCost(curr)
PairGDCost(prev)
PairGDCost(curr)
```

For matching, the new dual-cls head uses:

```python
pair_cls = 0.5 * (cls_prev + cls_curr)
```

Then this averaged pair cls logit is passed into the existing Hungarian classification cost.

## 6. AP Metric Changes

Modified files:

`projects/multispec_pair_rotated_rtdetr/multispec_pair_rotated_rtdetr/overfit_ap.py`

`projects/multispec_pair_rotated_rtdetr/multispec_pair_rotated_rtdetr/pair_overfit_metric.py`

### 6.1 Both-Visible Validation

New metric option:

```python
both_visible_gt_only=True
```

When enabled, validation GT is filtered using the same rule as training:

```python
valid_prev & valid_curr
```

This makes training and validation AP use the same target definition.

### 6.2 Presence-Free AP

For predictions without `presence_prev/presence_curr`:

- `pred_valid_prev=True`
- `pred_valid_curr=True`
- `pair_score=sqrt(score_prev * score_curr)`
- independent prev AP uses `score_prev`
- independent curr AP uses `score_curr`

For old predictions with presence fields, old behavior remains supported.

### 6.3 Per-Class AP

The AP helper now emits per-class AP50 keys, for example:

```text
pair_class0_AP50
pair_class1_AP50
independent_prev_class0_AP50
independent_curr_class0_AP50
gap1_pair_class0_AP50
...
```

The validation log table was widened to keep long metric names readable.

## 7. Tracking Changes

Modified files:

`projects/multispec_pair_rotated_rtdetr/multispec_pair_rotated_rtdetr/pair_mot_tracker.py`

`projects/multispec_pair_rotated_rtdetr/tools/run_pair_mot.py`

### 7.1 Cache Fields

`PairDetection` now optionally supports:

```text
score_prev
score_curr
label_prev
label_curr
```

Old cache format remains compatible because these fields are optional.

### 7.2 Threshold Logic Without Presence

This section was updated after review.

The earlier implementation used:

```python
track_score = min(score_prev, score_curr)
```

That is not sufficient for birth/death semantics. If a target is newly born,
the expected pattern can be:

```text
score_prev low
score_curr high
```

Using the minimum score would incorrectly suppress the birth candidate.

The tracker now separates the three decisions:

```python
can_match_prev = prev_side_score >= track_th
can_update_curr = curr_side_score >= track_th
can_birth = curr_side_score >= new_born_th
```

Concrete behavior:

1. `prev_side_score` controls whether a pair detection participates in IoU
   matching with previous-frame tracks.
2. If a detection matches a previous track, `curr_side_score` controls whether
   the matched track is updated to the current frame.
3. Birth is evaluated from all pair detections using only `curr_side_score`,
   excluding detections that already successfully updated an existing track.

This means:

- `prev low, curr high` can still create a new track.
- `prev high, curr low` can match the previous track but will not update it;
  the track naturally becomes lost/predicted by Kalman.
- `prev high, curr high` can update an existing track.

For old presence-based models, side scores are derived as:

```python
prev_side_score = cls_score * presence_prev
curr_side_score = cls_score * presence_curr
```

For the new dual-cls/no-presence model, side scores are:

```python
prev_side_score = score_prev
curr_side_score = score_curr
```

## 8. Validation Already Run

No training was started.

Completed checks:

### 8.1 Syntax Check

Passed:

```text
pair_rotated_rtdetr_head.py
pair_overfit_metric.py
overfit_ap.py
pair_mot_tracker.py
run_pair_mot.py
pair_val_visualization_hook.py
```

### 8.2 Config Build Check

The new config builds successfully.

Observed:

```text
query_init dual_topk
pair_dn_query_generator None
head flags dual_cls=True use_presence=False train_both_visible_only=True
has cls curr True
has presence False
assigner costs ['FocalLossCost', 'PairChamferCost', 'PairChamferCost', 'PairGDCost', 'PairGDCost']
val metric both_visible True
```

### 8.3 Forward Predict / Loss Check

One real test pair was run through predict and loss.

Prediction fields:

```text
scores
scores_prev
bboxes_prev
scores_curr
labels_prev
labels
labels_curr
bboxes_curr
```

No presence fields were emitted.

Loss keys:

```text
loss_cls_prev
loss_cls_curr
loss_cls
loss_bbox_prev
loss_bbox_curr
loss_iou_prev
loss_iou_curr
d0.*
d1.*
```

Example untrained/random-head loss values from the smoke check:

```text
loss_cls_prev: 17.4476
loss_cls_curr: 17.4476
loss_cls: 34.8953
loss_bbox_prev: 2.2706
loss_bbox_curr: 2.2451
loss_iou_prev: 1.0466
loss_iou_curr: 1.0479
```

### 8.4 Metric Check

The metric ran on two test pairs and printed the table successfully.

The table contains:

- independent AP
- pair AP
- gap1 AP
- per-class AP50

Because the model is untrained/random-loaded in this smoke check, AP values are expected to be zero.

### 8.5 Tracking Logic Check

After the review comment, a synthetic tracking case was run:

1. Existing track at frame 1.
2. Detection A:
   - `prev_score=0.9`
   - `curr_score=0.1`
   - overlaps the existing previous-frame track
3. Detection B:
   - `prev_score=0.05`
   - `curr_score=0.95`
   - represents a current-frame new birth

Observed events:

```text
matched_prev_curr_low track_id=1 det_index=0 prev_score=0.9 curr_score=0.1
match_diag reason=curr_score_below_track_threshold
birth track_id=2 det_index=1 prev_score=0.05 curr_score=0.95
```

Final state:

```text
track 1: Lost at frame 2
track 2: Tracked at frame 2, score=0.95
```

This confirms the intended behavior:

- high prev / low curr does not update a track;
- low prev / high curr can still create a new track.

## 9. Known Risks / Review Points

### 9.1 PairDN Disabled

This is intentional for the first run.

Risk:

- baseline with PairDN and new version without PairDN are not purely one-variable comparable.

Reason:

- current PairDN target code includes presence targets.
- adapting it would be a second design change.

Recommendation:

- Run this version first to validate the both-visible/dual-cls/no-presence idea.
- If it works, add a second version with PairDN adapted to both-visible dual-cls.

### 9.2 Dual cls uses same hidden state

Both `cls_prev` and `cls_curr` are predicted from the same pair decoder hidden state.

This is acceptable for a first version because the references and box branches are still frame-specific, but it may not fully decouple per-frame classification.

Future improvement:

- expose frame-specific decoder features, or
- add side-specific cls inputs from prev/curr query features.

### 9.3 Matching cls cost uses averaged logits

Hungarian matching uses:

```python
0.5 * (cls_prev + cls_curr)
```

This is simple and stable, but alternatives may be better:

- `sqrt(sigmoid(cls_prev) * sigmoid(cls_curr))`
- `min(cls_prev, cls_curr)`
- separate cls costs for prev and curr

Current choice is conservative.

### 9.4 AP target definition changes

This experiment evaluates only both-visible GT when `both_visible_gt_only=True`.

This is correct for the new objective, but its AP is not directly identical in meaning to old AP that included birth/death style GT.

For fair reporting, label this as:

```text
both-visible pair AP
```

### 9.5 Tracking birth/death depends on curr/dual score calibration

Without presence, tracking now relies on separated side scores:

- `score_prev` for previous-track matching eligibility
- `score_curr` for current-frame update eligibility
- `score_curr` for new-birth eligibility

This is logically consistent, but threshold calibration may differ from old presence-based models.

The requested fixed tracking params are still supported:

```text
nb=0.6
tr=0.2
iou=0.25
birthiou=0.5
age=30
```

## 10. Planned Next Step After Approval

If approved, start training with 2 GPUs on the new config:

```bash
source /data/users/litianhao01/anaconda3/etc/profile.d/conda.sh
conda activate py310
cd /data/users/litianhao01/PairMmot/ai4rs
CUDA_VISIBLE_DEVICES=1,2 PORT=<free_port> bash tools/dist_train.sh \
  projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_bothvis_dualcls_nopres.py \
  2 \
  --work-dir /data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_bothvis_dualcls_nopres
```

After training:

1. Select checkpoint with best `pair/pair_AP50`.
2. Run pair MOT tracking once with:

```text
pairmot_gap1_hota_search1_nb0.6_tr0.2_iou0.25_birthiou0.5_age30
```

3. Compare HOTA / DetA / AssA / MOTA / IDF1 against existing baseline.
