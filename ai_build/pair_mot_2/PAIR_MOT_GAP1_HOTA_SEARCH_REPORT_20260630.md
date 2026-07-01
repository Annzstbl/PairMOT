# PairMOT Gap1 HOTA Search Report 2026-06-30

## Scope

This report covers MOT tracking on the gap=1 pair detector checkpoint:

- Config: `projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train.py`
- Checkpoint: `/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1_fixed_20260628/epoch_72.pth`
- Test data: `/data/users/litianhao01/PairMmot/data/hsmot/test`
- Detector cache reused: no model inference was rerun during the parameter search.

## Important Fixes Before Search

1. Coordinate rescale bug fixed.
   The model config has `model.test_cfg.rescale=False`, so `model.predict(..., rescale=True)` was not enough. The MOT cache writer now explicitly converts resized rboxes back to original image coordinates with `scale_factor`.

2. PairMOT tracking time order fixed.
   For pair `t -> t+1`, matching now uses `track(t)` against `prev_det(t)`, then updates the track with `curr_det(t+1)`. The previous implementation predicted track state to `t+1` before matching to `prev_det(t)`, which was incorrect.

3. Lost-track recovery retained.
   Lost tracks are advanced by Kalman state to the current prev frame and can be recovered if they match `prev_det`. Lost predictions are not written as TrackEval detections unless recovered by a pair detection.

4. Duplicate birth suppression added.
   `new_birth_iou_th` suppresses new tracks whose `curr_det` overlaps an already tracked object. This is tracker-side birth suppression, not detector NMS.

5. Visualization updated.
   `track_match_diag` now shows only diagnosis: left side draws track and prev_det with IoU/reason; right side draws corresponding curr_det, linked from prev_det.

## Directory Map

| Path | Purpose |
|---|---|
| `/data/users/litianhao01/PairMmot/workdir/pair_mot_full_gap1_epoch72_20260629/pair_cache/` | Cached pair detections for all 50 test videos. Each sequence is one `.jsonl`; boxes are original-image rboxes after the rescale fix. |
| `/data/users/litianhao01/PairMmot/workdir/pair_mot_full_gap1_epoch72_20260629/trackers/` | All tracker outputs for gap1 experiments. |
| `trackers/<tracker_name>/preds/` | TrackEval-format prediction txt, one file per video. |
| `trackers/<tracker_name>/debug_matches/` | JSONL debug events: `birth`, `match`, `match_diag`, `birth_suppressed`. |
| `trackers/<tracker_name>/eval/` | TrackEval result summaries, detailed files, and plots. |
| `trackers/pairmot_gap1_ep72_nb0.5_tr0.2_iou0.3_age30_data24_allframes/vis/data24-1/` | Full-frame visualization for `data24-1`, 471 images per view. |
| `.../vis/data24-1/final_tracks/` | Pair-wise final track view, e.g. `000003_to_000004.jpg`. |
| `.../vis/data24-1/matched_pair/` | Accepted pair association view. |
| `.../vis/data24-1/raw_pair/` | Raw pair detections above visualization score threshold. |
| `.../vis/data24-1/track_match_diag/` | Diagnosis view for failed/unmatched track association. |
| `/data/users/litianhao01/PairMmot/workdir/pair_mot_full_gap1_epoch72_20260629/sweep_summary.csv` | Final parameter-search summary table. |

## Current Main Results

Metric source: `cls_comb_det_av_summary.csv` from TrackEval, merged into `sweep_summary.csv`.

Best HOTA in the 16-run search:

| tracker | new_born_th | track_th | match_iou_th | new_birth_iou_th | max_age | HOTA | DetA | AssA | IDF1 | MOTA | Dets | IDs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pairmot_gap1_hota_search1_nb0.6_tr0.2_iou0.25_birthiou0.5_age30 | 0.6 | 0.2 | 0.25 | 0.5 | 30 | 57.195 | 48.418 | 70.116 | 66.396 | 51.626 | 162465 | 7033 |

Reference corrected baseline:

| tracker | new_born_th | track_th | match_iou_th | new_birth_iou_th | max_age | HOTA | DetA | AssA | IDF1 | MOTA | Dets | IDs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pairmot_gap1_search_smoke_nb0.5_tr0.2_iou0.3_birthiou0.6_age30 | 0.5 | 0.2 | 0.3 | 0.6 | 30 | 56.819 | 48.518 | 69.239 | 65.444 | 50.415 | 168690 | 8692 |

The best searched setting improves HOTA by `+0.376` over the corrected baseline and reduces ID count from `8692` to `7033`.

## Parameter Search Table

| tracker | new_born_th | track_th | match_iou_th | new_birth_iou_th | max_age | HOTA | DetA | AssA | IDF1 | MOTA | Dets | IDs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pairmot_gap1_hota_search1_nb0.6_tr0.2_iou0.25_birthiou0.5_age30 | 0.6 | 0.2 | 0.25 | 0.5 | 30 | 57.195 | 48.418 | 70.116 | 66.396 | 51.626 | 162465 | 7033 |
| pairmot_gap1_hota_search1_nb0.6_tr0.2_iou0.25_birthiou0.6_age30 | 0.6 | 0.2 | 0.25 | 0.6 | 30 | 57.154 | 48.395 | 70.056 | 66.298 | 51.543 | 162662 | 7114 |
| pairmot_gap1_hota_search1_nb0.6_tr0.2_iou0.3_birthiou0.5_age30 | 0.6 | 0.2 | 0.3 | 0.5 | 30 | 57.130 | 48.412 | 69.969 | 66.216 | 51.642 | 162011 | 7169 |
| pairmot_gap1_hota_search1_nb0.6_tr0.2_iou0.3_birthiou0.6_age30 | 0.6 | 0.2 | 0.3 | 0.6 | 30 | 57.091 | 48.387 | 69.917 | 66.126 | 51.556 | 162207 | 7256 |
| pairmot_gap1_hota_search1_nb0.6_tr0.15_iou0.25_birthiou0.5_age30 | 0.6 | 0.15 | 0.25 | 0.5 | 30 | 57.068 | 48.288 | 70.011 | 66.239 | 51.083 | 164410 | 7036 |
| pairmot_gap1_hota_search1_nb0.6_tr0.15_iou0.3_birthiou0.5_age30 | 0.6 | 0.15 | 0.3 | 0.5 | 30 | 57.034 | 48.304 | 69.906 | 66.110 | 51.151 | 163862 | 7161 |
| pairmot_gap1_hota_search1_nb0.6_tr0.15_iou0.25_birthiou0.6_age30 | 0.6 | 0.15 | 0.25 | 0.6 | 30 | 56.996 | 48.253 | 69.898 | 66.086 | 50.957 | 164697 | 7142 |
| pairmot_gap1_hota_search1_nb0.6_tr0.15_iou0.3_birthiou0.6_age30 | 0.6 | 0.15 | 0.3 | 0.6 | 30 | 56.964 | 48.266 | 69.802 | 65.967 | 51.032 | 164139 | 7271 |
| pairmot_gap1_hota_search1_nb0.5_tr0.2_iou0.25_birthiou0.5_age30 | 0.5 | 0.2 | 0.25 | 0.5 | 30 | 56.927 | 48.540 | 69.453 | 65.694 | 50.527 | 168754 | 8337 |
| pairmot_gap1_hota_search1_nb0.5_tr0.2_iou0.25_birthiou0.6_age30 | 0.5 | 0.2 | 0.25 | 0.6 | 30 | 56.871 | 48.492 | 69.393 | 65.592 | 50.373 | 169090 | 8480 |
| pairmot_gap1_hota_search1_nb0.5_tr0.2_iou0.3_birthiou0.5_age30 | 0.5 | 0.2 | 0.3 | 0.5 | 30 | 56.875 | 48.563 | 69.301 | 65.554 | 50.569 | 168363 | 8541 |
| pairmot_gap1_hota_search1_nb0.5_tr0.2_iou0.3_birthiou0.6_age30 | 0.5 | 0.2 | 0.3 | 0.6 | 30 | 56.819 | 48.518 | 69.239 | 65.444 | 50.415 | 168690 | 8692 |
| pairmot_gap1_hota_search1_nb0.5_tr0.15_iou0.25_birthiou0.5_age30 | 0.5 | 0.15 | 0.25 | 0.5 | 30 | 56.762 | 48.352 | 69.336 | 65.486 | 49.824 | 171082 | 8315 |
| pairmot_gap1_hota_search1_nb0.5_tr0.15_iou0.3_birthiou0.5_age30 | 0.5 | 0.15 | 0.3 | 0.5 | 30 | 56.748 | 48.396 | 69.244 | 65.386 | 49.949 | 170518 | 8495 |
| pairmot_gap1_hota_search1_nb0.5_tr0.15_iou0.25_birthiou0.6_age30 | 0.5 | 0.15 | 0.25 | 0.6 | 30 | 56.666 | 48.290 | 69.207 | 65.289 | 49.618 | 171549 | 8487 |
| pairmot_gap1_hota_search1_nb0.5_tr0.15_iou0.3_birthiou0.6_age30 | 0.5 | 0.15 | 0.3 | 0.6 | 30 | 56.637 | 48.326 | 69.093 | 65.154 | 49.737 | 170987 | 8683 |

## Observations

- Raising `new_born_th` from `0.5` to `0.6` consistently improves HOTA and strongly reduces duplicate IDs.
- `track_th=0.2` is better than `0.15` in this grid.
- `new_birth_iou_th=0.5` is slightly better than `0.6`, likely because it suppresses duplicate births more aggressively.
- `match_iou_th=0.25` is marginally better than `0.3` for the best setting.
- TrackEval reports a warning for `data39-3`: invalid timesteps `90,91,92,93`. This should be investigated separately because it may indicate prediction frames beyond GT frame range for that sequence.

## Recommended Current Setting

Use:

```bash
--new-born-th 0.6 \
--track-th 0.2 \
--match-iou-th 0.25 \
--new-birth-iou-th 0.5 \
--max-age 30
```

Tracker directory:

`/data/users/litianhao01/PairMmot/workdir/pair_mot_full_gap1_epoch72_20260629/trackers/pairmot_gap1_hota_search1_nb0.6_tr0.2_iou0.25_birthiou0.5_age30`

## Debug Match Output Policy

`debug_matches/` stores per-pair diagnostic events used by `track_match_diag`
visualization. It is not a detector cache and is not required for normal
tracking or TrackEval evaluation. To keep formal sweeps lightweight,
`run_pair_mot.py` now writes `debug_matches/` only when either `--save-vis` or
`--save-debug-matches` is enabled.

The reusable detector cache remains `pair_cache/*.jsonl`; changing tracking
thresholds can still reuse `pair_cache` without rerunning network inference.
