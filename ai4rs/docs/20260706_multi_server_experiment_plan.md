# 2026-07-06 Multi-Server Experiment Plan

This file is the living multi-server state record for PairMOT experiments.
Update the status tables here whenever code is synced, a job is launched, or a
server path/credential convention changes.

Last updated: 2026-07-17 05:42 CST.

## Server Status

| Server | Role | SSH from 99 | Code root | Shared root | Work dir | Conda |
| --- | --- | --- | --- | --- | --- | --- |
| local `10.106.14.99` | current control/source workspace and execution resource | local shell as `wangying01` | `/data/users/wangying01/lth/PairMOT/ai4rs` | `/data4/litianhao/PairMmot` | `/data4/litianhao/PairMmot/workdir_99` | `/data/users/wangying01/anaconda3/envs/py310` |
| `10.106.14.197` | execution resource | `ssh -i ~/.ssh/litianhao01@10.106.14.197/id_rsa litianhao@10.106.14.197` | `/data/users/litianhao/PairMOT/ai4rs` | `/data4/litianhao/PairMmot` | `/data4/litianhao/PairMmot/workdir_197` | `/data/users/litianhao/anaconda3/envs/py310` |
| `10.106.15.178` | available resource, not verified this turn | key folder exists under `~/.ssh/litianhao01@10.106.15.178` | unknown | `/data4/litianhao/PairMmot` expected | unknown | unknown |
| `10.106.15.252` | available resource, not verified this turn | key folder exists under `~/.ssh/litianhao01@10.106.15.252` | unknown | `/data4/litianhao/PairMmot` expected | unknown | unknown |
| AutoDL `autodl-container-b77mjk6jn5-c7ceaf44` | temporary two-GPU execution resource | transient password SSH; local command is ignored under `autodl/ssh.md` | `/root/PairMOT/ai4rs` | `/root/autodl-fs/PairMOT_assets` | `/root/autodl-tmp/work_dirs` | image base Python, PyTorch `2.8.0+cu128` retained |

SSH directory convention: subdirectory names under `~/.ssh` are
`username@ip+port`, with port omitted for default `22`.  The 197 key directory
is named `litianhao01@10.106.14.197`, but the verified login account on
2026-07-11 is `litianhao`; `litianhao01@10.106.14.197` currently returns
`Permission denied (publickey)`.

252 verified login on 2026-07-09:

```bash
ssh -i ~/.ssh/litianhao01@10.106.15.252/id_ed25519 litianhao01@10.106.15.252
```

## Global Experiment ID Rule

Formal experiment IDs are global across 99, 197, and 252.  They are never
allocated independently per server.

- Format: `MMDD_NN`, for example `0715_02`.
- Before creating a config, queue, screen, or workdir, scan all three shared
  roots and this document for the largest ID on that date, then reserve the
  next number here.
- The reservation happens before code sync or launch, so simultaneous jobs on
  different servers cannot claim the same ID.
- A retry of exactly the same scientific experiment keeps its ID and uses a
  suffix such as `_rerun` or `_restart`; a changed model, data, loss, precision
  boundary, or ablation receives a new global ID.
- `tmp_profile_*`, diagnostics, detection/evaluation-only jobs, and canceled
  queues do not consume formal experiment IDs.
- Config filename, workdir, screen name, launch log, report, and status entry
  must use the same global ID.

Legacy 0714 paths are not renamed because active checkpoints, evaluator paths,
and reports already reference them.  The known historical collision is
`0714_01`: 99 used it for pair-aware liquid, while 252 used it for the
full-data COCO+Objects365 baseline.  The repeated 197 `0714_02` paths are
diagnostic/restart variants of its AMP investigation.  These are historical
exceptions only, not numbering precedent.

Current allocation state:

| Date | Last global ID | Experiment | Server | Next ID |
| --- | --- | --- | --- | --- |
| 2026-07-16 | `0716_02` | paper Base R18, COCO-only, full data, 1200x900, BF16 | 99 | `0716_03` |
| 2026-07-16 | `0716_03` | paper Base + final Liquid R18, COCO-only, full data, 1200x900, BF16 | 197 | `0716_04` |
| 2026-07-16 | `0716_04` | paper Base + final Liquid + hard group-set uniqueness, full data, 1200x900, BF16 | 197 | `0716_05` |
| 2026-07-16 | `0716_05` | paper Base + final Liquid group-set uniqueness + temporal/pyramid Encoder, full data, 1200x900, BF16 | 252 | `0716_06` |
| 2026-07-17 | `0717_01` | paper Liquid Set-Transport candidate, full data, 1200x900, BF16; same-ID fresh rerun migrated after the 99 cancellation | AutoDL | `0717_02` |

## Current Paper Runs

| Date | Server | Experiment | GPUs | Status | Log |
| --- | --- | --- | --- | --- | --- |
| 2026-07-16 | local `10.106.14.99` | `0716_02_paper_base_r18_coco_full_1200x900_bf16_orderedpairs_reboot_fresh` | `0,1` | completed 72 epochs and 18/18 async TrackEval points; unique best epoch 68 has cls HOTA 53.314, det HOTA 61.982, same-epoch pair mAP 0.3149 and AP50 0.5225 | `/data4/litianhao/PairMmot/workdir_99/0716_02_paper_base_r18_coco_full_1200x900_bf16_orderedpairs_reboot_fresh/launch.log` |
| 2026-07-16 | local `10.106.14.99` | `0716_03_paper_base_plus_liquid_r18_coco_full_1200x900_bf16_orderedpairs` | `2,3` | canceled and fully cleaned at 16:41 CST after GPU 2 hardware drop (`0000:B1:00.0: Unknown Error`); stopped in epoch 1 after iter 1000, no formal checkpoint, must fresh train on healthy GPUs; GPU 0/1 Base unaffected | `/data4/litianhao/PairMmot/workdir_99/0716_03_paper_base_plus_liquid_r18_coco_full_1200x900_bf16_orderedpairs/launch.log` |
| 2026-07-16 | `10.106.14.197` | `0716_03_paper_base_plus_liquid_r18_coco_full_1200x900_bf16_orderedpairs_fresh` | `0,3` | stopped intentionally at epoch 21 iter 50 without resume after confirming cross-group set collapse in the soft argmax preview; retained as diagnostic history | `/data4/litianhao/PairMmot/workdir_197/0716_03_paper_base_plus_liquid_r18_coco_full_1200x900_bf16_orderedpairs_fresh/launch.log` |
| 2026-07-16 | `10.106.14.197` | `0716_04_paper_base_plus_liquid_groupsetunique_r18_coco_full_1200x900_bf16_orderedpairs_fresh` | `0,3` | running; fresh start at 23:22 CST after 20/20 remote sampler tests; epoch 1 iter 50 is 0.9771 s/iter with finite losses/gradients and hard preview `unique_sets=8.00`, `max_set_repeat=1.00` | `/data4/litianhao/PairMmot/workdir_197/0716_04_paper_base_plus_liquid_groupsetunique_r18_coco_full_1200x900_bf16_orderedpairs_fresh/launch.log` |
| 2026-07-16 | `10.106.15.252` | `0716_05_paper_base_plus_liquid_groupsetunique_encoder_r18_coco_full_1200x900_bf16_orderedpairs_fresh` | `0,1` | running; 30 unit tests and a separate 100-iter DDP validation passed before fresh launch at 23:36 CST; formal epoch 1 iter 50 has finite losses/gradients, both temporal branches learning, memory 11387 MiB/rank, and Liquid hard preview `unique_sets=8.00` | `/data4/litianhao/PairMmot/workdir_252/0716_05_paper_base_plus_liquid_groupsetunique_encoder_r18_coco_full_1200x900_bf16_orderedpairs_fresh/launch.log` |
| 2026-07-17 | local `10.106.14.99` | `0717_01_paper_base_plus_liquid_settransport_r18_coco_full_1200x900_bf16_orderedpairs_fresh` | `2,3` | canceled intentionally at epoch 2 iter 250 because local GPUs 2/3 have prior drop-card risk; full process group and screen removed, both GPUs released to 10 MiB; model code, 23 tests and separate 100-iter DDP validation are retained, but this incomplete run is not a result | `/data4/litianhao/PairMmot/workdir_99/0717_01_paper_base_plus_liquid_settransport_r18_coco_full_1200x900_bf16_orderedpairs_fresh/launch.log` |
| 2026-07-17 | AutoDL `autodl-container-b77mjk6jn5-c7ceaf44` | `0717_01_paper_base_plus_liquid_settransport_r18_coco_full_1200x900_bf16_orderedpairs_autodl_fresh` | `0,1` | running fresh from COCO-adapted pretrain; epoch 1 iter 50 at 05:41 CST is 0.9282 s/iter with finite losses/gradients, 10703 MiB framework memory/rank, Set-Transport strength 0.004, `unique_sets=8.00`, and `max_set_repeat=1.00`; ETA about 19 h | `/root/autodl-tmp/work_dirs/0717_01_paper_base_plus_liquid_settransport_r18_coco_full_1200x900_bf16_orderedpairs_autodl_fresh/launch.log` |

## Current 0708 Runs

| Date | Server | Experiment | GPUs | Status | Log |
| --- | --- | --- | --- | --- | --- |
| 2026-07-09 | local `10.106.14.99` | `0708_01_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_tristate_decoder` | `1,3` | completed to `epoch_72.pth`; results pending report refresh if needed | `/data4/litianhao/PairMmot/workdir_99/0708_01_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_tristate_decoder/launch.log` |
| 2026-07-09 | local `10.106.14.99` | `0709_01_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8` | `1,3` | running, verified through `Epoch(train) [13][400/484]`; liquid pattern remains `701 / 012 / 123 / 234 / 345 / 456 / 567 / 670`; ETA about 7h from 23:47 CST | `/data4/litianhao/PairMmot/workdir_99/0709_01_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8/launch.log` |
| 2026-07-10 | `10.106.15.252` | `0709_02_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_liquidawarefusion` | `0,1` | running, verified through `Epoch(train) [1][100/484]`; liquid pattern starts as `701 / 012 / 123 / 234 / 345 / 456 / 567 / 670` | `/data4/litianhao/PairMmot/workdir_252/0709_02_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_liquidawarefusion/launch.log` |
| 2026-07-10 | `10.106.14.197` | `0709_03_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_overlap` | `4,5` | running, verified through `Epoch(train) [1][150/484]`; adds liquid-aware overlap context over source-band coverage | `/data4/litianhao/PairMmot/workdir_197/0709_03_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_overlap/launch.log` |
| 2026-07-10 | `10.106.15.252` | `0709_04_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_wide_overlap` | queued for `0,1` | queue active; waits for GPU memory below 1024 MB; wide liquid-aware overlap context with `embed_dims=64` | `/data4/litianhao/PairMmot/workdir_252/0709_04_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_wide_overlap/queue.log` |
| 2026-07-10 | local `10.106.14.99` | `0709_05_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_patternbias` | queued for `1,3` | queue active; waits for GPU memory below 1024 MB; pattern-only liquid-aware gate with overlap context and no spatial mixer | `/data4/litianhao/PairMmot/workdir_99/0709_05_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_patternbias/queue.log` |
| 2026-07-09 | `10.106.14.197` | `0708_02_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_tristate_decoder_sepffn` | `3,4` | completed to `epoch_72.pth`; decoder report selected epoch 71 with `cls_HOTA + det_HOTA = 104.511` | `/data4/litianhao/PairMmot/workdir_197/0708_02_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_tristate_decoder_sepffn/launch.log` |
| 2026-07-09 | `10.106.15.252` | `0704_01_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_resume_from_epoch40_to72` | `0,1` | completed to `epoch_72.pth`; results added to `20260709_0708_01_99_report.md` | `/data4/litianhao/PairMmot/workdir_252/0704_01_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_resume_from_epoch40_to72/launch.log` |

Shared assets verified on 197:

```text
pretrain: /data4/litianhao/PairMmot/pretrained_weights/o2_r18_hsmot_3dse_r2_e72_pair_dualcls_pairdn_adapted/pair_dualcls_pairdn_adapted_pretrain.pth
gmc train: /data/users/litianhao/PairMOT/workdir/aux/gmc_cache/hsmot_train_gap1
gmc test:  /data/users/litianhao/PairMOT/workdir/aux/gmc_cache/hsmot_test_gap1
```

## Status Update Checklist

When updating this file, keep these fields current:

- Reserve the next global `MMDD_NN` ID across 99, 197, and 252 before launch.
- SSH login command and key path if access changes.
- Code root, shared root, work dir, conda env, and GMC cache root per server.
- Current job name, GPUs, launch time, log path, and first observed
  `Epoch(train)` line.
- Finished jobs should be moved from "Current 0708 Runs" into a result/report
  document with the selected checkpoint rule.

## Shared Rules

- Train with exactly two GPUs per experiment.
- Keep experiments fair: start from the adapted pretrain, not from a high-epoch
  checkpoint.
- Use `/data4/litianhao/PairMmot` for shared artifacts when running on 99 or
  197.
- Track/eval may use one GPU or CPU-side async evaluation as configured.
- Do not run the same config on two servers unless explicitly requested.

## Code Sync

Primary sync flow from local 99 to 197:

```bash
rsync -az \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='work_dirs/' \
  --exclude='workdir/' \
  --exclude='data/' \
  --exclude='pretrained_weights/' \
  --exclude='val_det/' \
  --exclude='val_track_eval/' \
  --exclude='val_vis/' \
  -e "ssh -i ~/.ssh/litianhao01@10.106.14.197/id_rsa -o BatchMode=yes" \
  /data/users/wangying01/lth/PairMOT/ai4rs/ \
  litianhao@10.106.14.197:/data/users/litianhao/PairMOT/ai4rs/
```

Do not pass `--delete` unless the remote code tree is known disposable.  This
keeps remote reports, logs, and local-only notes from being removed by mistake.

On a resource server with a clean clone and correct remote access, a git-based
sync is also acceptable:

```bash
cd /path/to/PairMOT/ai4rs
git fetch
git pull --ff-only
git checkout main
```

The required pretrain should exist at:

```text
/data4/litianhao/PairMmot/pretrained_weights/o2_r18_hsmot_3dse_r2_e72_pair_dualcls_pairdn_adapted/pair_dualcls_pairdn_adapted_pretrain.pth
```

## Current Assignment

### 10.106.12.252

Current job:

```text
0705_01_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_p5temporal_pyramidlocal
```

After it finishes, run:

```text
0705_03_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_pyramidlocal_p4p5
```

Config:

```text
projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_pyramidlocal_p4p5.py
```

Purpose: ablate whether temporal local interaction should avoid the lowest FPN
level.  It applies the post-FPN pyramid-local adapter only on P4/P5.

### 10.106.14.99

Run immediately when the server has two free GPUs:

```text
0705_02_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_pyramidlocal
```

Config:

```text
projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_pyramidlocal.py
```

Purpose: test post-FPN pyramid-local temporal interaction on all FPN levels
without the P5 MHA branch.

Launch command:

```bash
mkdir -p /data4/litianhao/PairMmot/workdir_99/0705_02_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_pyramidlocal
cd /data4/litianhao/PairMmot/ai4rs
CUDA_VISIBLE_DEVICES=0,1 PORT=29762 bash tools/dist_train.sh \
  projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_pyramidlocal.py \
  2 \
  --work-dir /data4/litianhao/PairMmot/workdir_99/0705_02_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_pyramidlocal
```

### 10.106.14.197

Keep available for the next branch after 0705_02/0705_03 gives a signal, or
for running track/eval from selected checkpoints.  Do not duplicate 0705_02 or
0705_03 unless requested.

## Result Selection

Prioritize tracking metrics over AP:

- primary: `track/cls_hota`
- secondary: `track/cls_idf1`, `track/det_hota`, `track/det_idf1`
- AP remains diagnostic only.

For 0705_01, the best observed checkpoint before this plan was epoch 55 with:

```text
cls_hota=47.073, cls_mota=36.619, cls_idf1=55.106,
det_hota=58.351, det_mota=52.499, det_idf1=67.292
```

## 2026-07-11 Liquid Current Status

Baseline for liquid comparison is `0704_01` resume high metric:

```text
pair_mAP=0.2383, pair_AP50=0.4157,
cls_HOTA=45.523, det_HOTA=58.120, cls+det=103.643
```

Use the unique best tracking point selected by `cls_HOTA + det_HOTA`.

| server | experiment | status | latest/best result |
|---|---|---|---|
| 99 | `0709_01_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8` | finished epoch 72 | AP epoch 72: pair mAP 0.2457, AP50 0.4333. Track async 18: cls 46.803, det 57.899, sum 104.702. |
| 252 | `0709_02_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_liquidawarefusion` | finished epoch 72 | AP epoch 72: pair mAP 0.2432, AP50 0.4254. Track async 18: cls 46.328, det 57.994, sum 104.322. |
| 197 | `0709_03_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_overlap` | finished epoch 72 | AP epoch 72: pair mAP 0.2419, AP50 0.4293. Track async 17: cls 46.573, det 58.025, sum 104.598. |
| 252 | `0709_04_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_wide_overlap` | finished epoch 72 | AP epoch 72: pair mAP 0.2495, AP50 0.4367. Track async 18: cls 47.314, det 58.250, sum 105.564. Current best liquid result. |
| 99 | `0709_05_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_patternbias` | finished epoch 72 | AP epoch 72: pair mAP 0.2414, AP50 0.4263. Track async 18: cls 46.346, det 58.077, sum 104.423. |
| 99 | `0710_01_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_groupmod` | finished epoch 72 | AP epoch 72: pair mAP 0.2423, AP50 0.4283. Track async 18: cls 46.672, det 58.214, sum 104.886. |
| 197 | `0710_02_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_outputres` | finished epoch 72 | AP epoch 72: pair mAP 0.2434, AP50 0.4248. Track async 18: cls 46.190, det 58.275, sum 104.465. |
| 252 | `0710_03_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_sampler_bandattn` | running | Verified through epoch 48. Interim AP epoch 48: pair mAP 0.2424, AP50 0.4281. Interim track async 12: cls 46.099, det 57.445, sum 103.544. |

Current liquid conclusion:

- Best completed HOTA is 252 `laf_wide_overlap`: `105.564`, `+1.921` over `0704_01`.
- `0710_01_groupmod` gives the best non-wide-LAF follow-up sum among completed 0710 experiments: `104.886`, with a useful det-side signal.
- `0710_02_laf_outputres` raises det HOTA but hurts cls HOTA, so it is not the next main direction.
- `0710_03_sampler_bandattn` is still running and weak at the current interim point.

## 2026-07-10 Follow-up Liquid Experiments

These experiments are model changes, not hyperparameter-only changes.

| server | experiment | model change | launch status |
|---|---|---|---|
| 99 | `0710_01_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_groupmod` | Adds `LiquidGroupModulator` before SE fusion. The branch reads per-group sampling coverage, entropy, peak coverage and conv3d response, then reweights each liquid group feature. This tests whether the strongest plain `liquid8` can gain from coverage-aware group balancing without the heavier LAF branch. | Launched on GPUs `1,3`, port `29810`, at 2026-07-10 22:08 CST. Verified training reached epoch 1 and logs `LiquidSampler` pattern `701 / 012 / 123 / 234 / 345 / 456 / 567 / 670`. |
| 197 | `0710_02_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_outputres` | Extends the best `laf_overlap` variant with a pattern-aware output residual. LAF still changes SE logits, and its spatial delta also gates a small residual added to the final stem output, so liquid pattern information can affect the feature map directly. | Code synced to `/data/users/litianhao/PairMOT/ai4rs`; launched on GPUs `2,3`, port `29811`, at 2026-07-10 22:11 CST. Verified model build and checkpoint loading. |
| 252 | `0710_03_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_sampler_bandattn` | Adds inter-band self-attention inside `LiquidSpectralSampler` before the recurrent sampler head. The sampler now lets each raw spectral band descriptor attend to the other bands before selecting the 8 cyclic 3-band groups, testing whether learned inter-band contrast improves spectral group choice. | Code synced to `/data/users/litianhao01/PairMmot/ai4rs`; launched on GPUs `0,1`, port `29812`, at 2026-07-11 03:05 CST. Verified dry-run forward and training reached epoch 1 iter 50 with normal `LiquidSampler` logging. |

Workdirs:

```text
/data4/litianhao/PairMmot/workdir_99/0710_01_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_groupmod
/data4/litianhao/PairMmot/workdir_197/0710_02_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_outputres
/data4/litianhao/PairMmot/workdir_252/0710_03_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_sampler_bandattn
```

## 2026-07-11 Follow-up Liquid Experiments

These experiments build on the current best `0709_04_laf_wide_overlap`.

| server | experiment | model change | launch status |
|---|---|---|---|
| 99 | `0711_01_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_wide_groupmod` | Combines wide liquid-aware fusion with `LiquidGroupModulator`, testing whether coverage-aware group balancing can add the det-side gain seen in `0710_01` to the current best wide LAF. | Relaunched in detached `screen` session `pairmot_0711_01` on GPUs `1,2`, port `29813`, at 2026-07-11 12:13 CST. Verified training through epoch 1 iter 50 and stable `LiquidSampler` pattern logging. |
| 197 | `0711_02_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_wide_bandattn` | Combines wide liquid-aware fusion with inter-band sampler attention, testing whether band descriptor context helps group selection when the downstream LAF branch has enough capacity. | Code synced to `/data/users/litianhao/PairMOT/ai4rs`; launched on GPUs `2,3`, port `29814`, at 2026-07-11 12:05 CST. Verified training through epoch 1 iter 200. |
| 252 | `0711_03_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_wide_groupmod_bandattn` | Combines the current best wide LAF with both follow-up mechanisms: `LiquidGroupModulator` for coverage-aware group balancing and sampler inter-band attention for context-aware spectral group selection. This tests whether the 99 det-side signal and 197 sampler-context signal can stack on top of 252 `0709_04`. | Code synced to `/data/users/litianhao01/PairMmot/ai4rs`; launched in detached `screen` session `pairmot_0711_03` on GPUs `0,1`, port `29815`, at 2026-07-11 18:40 CST. Verified training through epoch 1 iter 50 and stable `LiquidSampler` pattern logging. |
| 99 | `0712_01_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_wide_groupmod_outputres` | Starts from the strongest 99 structure, wide LAF + `LiquidGroupModulator`, and enables a small liquid-aware output residual. This tests whether the det-side residual signal can be retained without the cls-side drop seen in the earlier output-residual-only branch. | Launched in detached `screen` session `pairmot_0712_01` on GPUs `0,1`, port `29816`, at 2026-07-12 01:16 CST. Verified checkpoint load and training through epoch 1 iter 50. |

Workdirs:

```text
/data4/litianhao/PairMmot/workdir_99/0711_01_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_wide_groupmod
/data4/litianhao/PairMmot/workdir_197/0711_02_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_wide_bandattn
/data4/litianhao/PairMmot/workdir_252/0711_03_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_wide_groupmod_bandattn
/data4/litianhao/PairMmot/workdir_99/0712_01_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_wide_groupmod_outputres
```

## 2026-07-13 Pair-aware Liquid Experiment

This experiment keeps frame-adaptive liquid sampling independent for prev/curr
frames and adds pair awareness only in the liquid fusion stage.  No band
attention is used.

| server | experiment | model change | launch status |
|---|---|---|---|
| 197 | `0713_05_fresh_novis_gpus1_4_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_pairaware_laf_wide` | Builds on wide liquid-aware fusion and adds `PairAwareLiquidFusion` in `MultispecStemConv3dSE`. The original per-frame `LiquidSpectralSampler` still samples prev/curr independently; a compact pair descriptor from coverage, entropy, peak coverage, group response, frame difference, and frame agreement generates an SE-logit residual. | The first `0713_05` run on GPUs `0,1` saved `epoch_4.pth` but crashed in epoch-4 DDP validation because rank0-only `HSMOTPairValVisualizationHook` caused long shared-storage I/O and NCCL collect timeout. A fresh no-resume run was relaunched with `default_hooks.visualization.draw=False`, first on `0,1` to verify training, then canceled per request and restarted in detached screen `pairmot_0713_05_gpus1_4` on GPUs `1,4`, port `29824`, at 2026-07-13 23:50 CST. It later reached epoch-12 validation and failed again with NCCL broadcast timeout after AP/val_det/async track-eval output, so it is no longer running. |
| 99 | `0714_01_fresh_novis_trackeval_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_pairaware_laf_wide` | Same pair-aware liquid model as the 197 run. | Relaunched on local 99 in detached screen `pairmot_0714_01_99_trackeval` on GPUs `0,1`, port `29826`, at 2026-07-14 10:10 CST. Only training-time `HSMOTPairValVisualizationHook` drawing is disabled (`draw=False`); AP, val_det export, and TrackEval remain enabled (`track_eval=True`, no `save_val_det=False`). Verified fresh start from the adapted pretrain and training through epoch 1 iter 100 with LiquidSampler pattern `701 / 012 / 123 / 234 / 345 / 456 / 567 / 670`. The briefly started `0714_01_fresh_novis_no_trackeval...` run was canceled because it disabled evaluation-related outputs. |

Workdir:

```text
/data4/litianhao/PairMmot/workdir_197/0713_05_fresh_novis_gpus1_4_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_pairaware_laf_wide
/data4/litianhao/PairMmot/workdir_99/0714_01_fresh_novis_trackeval_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_pairaware_laf_wide
```

## 2026-07-15 Pair-Consistent Spectral Transport

| server | experiment | model change | launch status |
|---|---|---|---|
| 99 | `0715_02_liquid8_laf_wide_groupmod_pairtransport` | Strictly uses `0711_01 wide LAF + groupmod` as the baseline. `PairCoupledSamplerRouter` adds bidirectional pair-conditioned sampler-logit residuals without forcing identical frame patterns; `PairTransportTokenCoupling` aligns wide-LAF group tokens by overlap between the resulting prev/curr spectral coverage distributions. Both branches are zero-initialized. | Finished epoch 72 after recovering from the physical GPU2 PCIe failure at epoch 70. The run remains FP32 `OptimWrapper` with `find_unused_parameters=True` for trajectory continuity. Unique best is final epoch 72 / payload `step=71`: `cls_HOTA=47.520`, `det_HOTA=58.600`, sum `106.120`, which is `+0.215` over the strict `0711_01` baseline and is the current liquid HOTA best. Final AP is pair mAP `0.2540`, pair AP50 `0.4448`. Resume reset the async counter, so this final TrackEval overwrote `val_track_0001`. |
| 99 | `0715_03_liquid8_laf_wide_groupmod_pairbandcontext` | Strictly uses `0711_01 wide LAF + groupmod`, without Pair Transport. A band-aligned prev/curr context jointly drives sampler descriptor/logit residuals and coverage-pooled wide-LAF group context. All injections are zero-initialized; the model adds only `24384` stem parameters. | Not launched. The original detached queue was terminated by the server reboot. Any future launch must be recreated explicitly using the canonical BF16-through-encoder and `find_unused_parameters=False` configuration. |
| 197 | `0715_04_liquid8_laf_wide_groupmod_pairchangegate` | Uses `0711_01 wide LAF + groupmod` as the structural baseline and the 99 `0715_01` BF16 setup as the training baseline. `PairChangeGatedTokenCoupling` uses per-group spectral-coverage intersection/distance and pooled response change to gate shared versus frame-specific liquid tokens. The residual is zero-initialized and adds no attention or spatial pair operation. | Running normally on GPUs `2,3` with BF16-through-encoder and `find_unused_parameters=False`; reached epoch 61 at 2026-07-15 21:50 CST, about `1.04 s/iter`, with roughly 1h45m training ETA. The unique best completed point is epoch 52 / payload `step=51`: `cls_HOTA=46.298`, `det_HOTA=57.768`, sum `104.066`. Epoch 56 is `103.690`; epoch 60 TrackEval is still running asynchronously. Current best is above `0704_01 resume` but `1.839` below the final historical FP32 `0711_01` sum `105.905`. Isolated stem overhead is about 1.3%. |
| 99 | `0715_05_liquid8_final_pairtransport_paironly_coco365_full_bf16` | Final Liquid candidate: eight groups, independent pair-conditioned samplers, wide overlap-aware LAF, group modulation, and coverage-based pair transport. Both relation MLPs consume ordered `[x,y]` only. Uses all 75 train sequences and direct COCO+Objects365 adapted initialization. | Completed 72 epochs and all 18 TrackEval points. Unique best is final `val_track_0018 / step 71`: `cls_HOTA=53.472`, `det_HOTA=60.907`; relative to full baseline `0714_01`, deltas are `+1.098` and `+0.589`. AP best is epoch 72: pair mAP `0.2988`, pair AP50 `0.5115`. All eight class HOTA values improve; tricycle is largest at `+5.072`. This is a positive system comparison, not a strict Liquid-only ablation because baseline is FP32/find-unused while this run is BF16/find-false with stability fixes. |
| 252 | `0715_06_liquid8_pairbandcontext_paironly_coco365_full_bf16` | Wide LAF + groupmod with a shared physical-band pair context. The context conditions both sampler descriptors/logits and coverage-pooled LAF tokens. Its directional relation consumes ordered `[x,y]` only; no pair router, pair transport, change gate, hand-crafted difference, or product is active. Uses all 75 train sequences and direct COCO+Objects365 adapted initialization. | Final fresh run started at 2026-07-15 22:05 CST on GPUs `0,1`, port `29878`, with BF16 through encoder, nearest sampler gradient expansion, and `find_unused_parameters=False`. Verified through epoch 1 iter 200: `0.9636 s/iter`, log memory `8444 MiB`, finite loss/grad and expected initial pattern; ETA is about 20h29m. The first 22:01 attempt stopped before model construction because 252 lacked the committed BF16 detector boundary; current detector/head/RT-DETR/GDLoss code was synchronized from the local stable implementation before the final launch. |

Workdir:

```text
/data4/litianhao/PairMmot/workdir_99/0715_02_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_wide_groupmod_pairtransport
/data4/litianhao/PairMmot/workdir_99/0715_03_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_wide_groupmod_pairbandcontext
/data4/litianhao/PairMmot/workdir_197/0715_04_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid8_laf_wide_groupmod_pairchangegate
/data4/litianhao/PairMmot/workdir_99/0715_05_liquid8_final_pairtransport_paironly_coco365_full_bf16
/data4/litianhao/PairMmot/workdir_252/0715_06_liquid8_pairbandcontext_paironly_coco365_full_bf16
```

## 2026-07-13 Long-tail cls-HOTA Repair Experiments

These experiments address the MOTRv2 vs PairMOT split: PairMOT `0704_01 resume`
has stronger `det_HOTA` but lower `cls_HOTA`, mainly from long-tail and
fine-grained classes such as `truck`, `bus`, `tricycle`, `van`, and bike-like
classes.  Per-class thresholding is treated only as diagnosis, not as a model
solution.

| server | experiment | model change | launch status |
|---|---|---|---|
| 252 | `0713_01_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_longtail_reweight` | Adds `cls_pos_loss_weights` in `PairRotatedRTDETRHead`, increasing positive classification loss for long-tail/fine-grained classes while keeping box, proposal, association and tracker unchanged. | Code synced to `/data/users/litianhao01/PairMmot/ai4rs`; launched in detached `screen` session `pairmot_0713_01` on GPUs `0,1`, port `29817`, at 2026-07-13 00:39 CST. Verified training through epoch 1 iter 150. |
| 252 | `0713_02_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_finecls_margin` | Adds `cls_pos_logit_margins` plus mild positive reweighting for fine-grained vehicle/bike-like classes, forcing a larger true-class logit gap without changing inference thresholds. | Code synced to `/data/users/litianhao01/PairMmot/ai4rs`; launched in detached `screen` session `pairmot_0713_02` on GPUs `2,3`, port `29818`, at 2026-07-13 00:39 CST. Verified training through epoch 1 iter 150. |
| 252 | `0713_03_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_longtail_proto_gate` | Builds on `0713_01` and adds a lightweight class-prototype gated classification bias in `PairRotatedRTDETRHead`. Each decoder query gets a learned class-prototype similarity bias, with stronger gates for long-tail/fine-grained classes, testing whether structural class-aware logit modulation can improve cls HOTA beyond static loss reweighting. | Code synced to `/data/users/litianhao01/PairMmot/ai4rs`; launched in detached `screen` session `pairmot_0713_03` on GPUs `0,1`, port `29819`, at 2026-07-13 15:27 CST. Verified remote model build, checkpoint load, and training through epoch 1 iter 50. |
| 99 | `0713_04_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_longtail_residual_adapter` | Builds on `0713_01` and adds a zero-initialized long-tail residual classifier branch in `PairRotatedRTDETRHead`. The original classification logits are preserved at initialization, while a small `256->128->8` MLP learns extra class-specific nonlinear residual logits, weighted toward `truck`, `bus`, `tricycle`, `awning-bike`, `bike`, and `van`. | Launched locally in detached `screen` session `pairmot_0713_04` on GPU `0` at 2026-07-13 18:07 CST. Verified local path overrides, checkpoint load, and training through epoch 1 iter 100. |

Workdirs:

```text
/data4/litianhao/PairMmot/workdir_252/0713_01_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_longtail_reweight
/data4/litianhao/PairMmot/workdir_252/0713_02_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_finecls_margin
/data4/litianhao/PairMmot/workdir_252/0713_03_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_longtail_proto_gate
/data4/litianhao/PairMmot/workdir_99/0713_04_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_longtail_residual_adapter
```

## 2026-07-14 AMP Acceleration

### Canonical configuration for experiments started after 2026-07-15

All newly launched experiments must inherit the validated 99-server `0715_01`
training/runtime configuration unless an experiment explicitly studies numerical
precision itself:

- `AmpOptimWrapper(dtype='bfloat16', loss_scale=1.0)`;
- BF16 for backbone, neck, and shared RT-DETR encoder, followed by one FP32 cast;
- FP32 for query initialization, decoder, prediction heads, matching, and losses;
- `find_unused_parameters=False` with every intended trainable parameter connected to loss;
- validation and TrackEval enabled; visualization/drawing may be disabled;
- fresh training by default (`resume=False`).

This rule standardizes precision and distributed training, but does not override the
dataset, initialization checkpoint, or structural parent required by an individual
ablation.  For new half-data experiments with the original `0704_01` structure, use
the completed 99 `0715_01` result (`cls_HOTA=46.531`, `det_HOTA=58.484`, sum
`105.015`) as the performance baseline.  Existing reports retain their historical
`0704_01 resume` comparisons.

This change keeps the `0704_01` model and loss definition and switches training
to `AmpOptimWrapper`.  Backbone and neck use AMP; transformer/deformable
attention remain FP32 because FP16 produced non-finite gradients and the CUDA
operator does not support BF16.  GDLoss filters only zero-weight missing pair
sides, restores the original `1e-3` width/height clamp, and runs directly in
FP32.  It does not silently discard non-finite visible samples or perform
per-loss covariance checks and GPU-to-CPU synchronization.

| server | experiment | change | status |
|---|---|---|---|
| 252 | `0714_01_0704_resume_coco365_full_unique_allgt` | `0704_01` structure with direct COCO+Objects365-adapted initialization and all 75 training sequences. This historical run uses FP32 and `find_unused_parameters=True`. | Finished 72 epochs and all 18 TrackEval points. Unique best is async 18 / val_det epoch 71: `cls_HOTA=52.374`, `det_HOTA=60.318`, sum `112.692`. Independent AP best is epoch 72: pair mAP `0.2928`, pair AP50 `0.5062`. |
| 99 | `tmp_profile_0714_pair_amp_fastgdloss_v1` | Single-GPU AMP smoke test for the current pair-valid-fill baseline, `find_unused_parameters=False`, dynamic loss scale initialized at `128`, with fast GDLoss fallback. | Passed 40 iters on GPU2. Loss and all logged components stayed finite; `grad_norm` stayed finite from iter 5 to 40. Mean `iter_wall` was `0.654s` vs `0.671s` for FP32 and `0.733s` for the earlier always-filtered AMP path. Memory was about `6.84GB`, compared with about `11.02GB` for FP32. |
| 252 | `0714_01_0704_resume_coco365_full_unique_allgt_amp` | Formal full-data COCO+Objects365-adapted `0704_resume` baseline with AMP. | Fast GDLoss AMP code synced to `/data/users/litianhao01/PairMmot/ai4rs`, but the queued screen `pairmot_0714_amp_queue` was canceled before launch on 2026-07-14 16:57 CST per request. No 252 AMP training process was started. |
| 197 | `0714_02_0704_01_half_unique_allgt_amp_fp32transformer` | Initial half-data stability run. It used `find_unused_parameters=True` and a defensive GDLoss implementation that changed the width/height clamp to `1.0`, silently dropped invalid rows, and introduced repeated synchronization. | Stopped on 2026-07-14 after the implementation review. The workdir and completed evaluation outputs are retained as diagnostic history and must not be used for the final AMP comparison. |
| 197 | `tmp_validate_0714_amp_fixed_gpu5` | Corrected hybrid AMP CUDA/DDP validation with `find_unused_parameters=False`, original GDLoss semantics, and no silent non-finite fallback. | Passed 100 consecutive iterations on GPU `5`. All loss components and `grad_norm` remained finite; stable time reached `0.69s/iter` over iterations 60-100. |
| 197 | `0714_03_0704_01_half_unique_allgt_hybrid_amp_fixed` | Formal half-data `0704_01` AMP performance-parity run using the corrected implementation. Backbone/neck use AMP; transformer/deformable attention and GDLoss use FP32; DDP uses `find_unused_parameters=False`. | Finished epoch 72 with all 18 TrackEval points. Unique best HOTA point is async 16 / val_det epoch 63: `cls_HOTA=46.271`, `det_HOTA=58.381`, sum `104.652`, which is `+1.009` over `0704_01 resume`. Independent best AP point is epoch 72: pair mAP `0.2424`, pair AP50 `0.4215`. No performance degradation was observed, but FP16 is not retained due its numerical stability risk. |

### 99 precision-boundary audit

On 2026-07-14, additional 99 profiles tested narrower FP32 boundaries against
the corrected full-transformer FP32 fallback.  Forcing only decoder deformable
attention to FP32 initially passed 120 random iterations and used about
`7.47GB`, but a fixed-order stress run produced a visible-sample NaN in
`d1.dn_loss_iou_curr`.  A follow-up audit found that the corresponding
`d1.dn_loss_bbox_curr` was finite, so this does not prove that the decoder's
refined box was non-finite.  Missing single-side DN targets are zero boxes, but
their box weights are also zero and `_loss_iou_valid` removes them before
GDLoss.  A scan of all 584,534 real boxes found no zero-area or non-finite
valid GT, and direct zero-missing-side DN regression plus 60-iteration
single-process and DDP diagnostics stayed finite.  The isolated NaN is
therefore more consistent with a borderline finite-box GD/KLD covariance
calculation than with an unsupervised zero box entering the loss; it was not
reproduced deterministically.  Running the whole transformer under AMP also
passed one seed but did not
improve sustained speed on RTX 3090 because the FP16 deformable-attention path
was slower.  Keeping only the pair decoder in FP32 was stable but slower than
the full-transformer FP32 fallback.  Enabling TF32 did not accelerate the
measured encoder/decoder path.

Decision: retain the `0714_03` boundary for formal training: backbone and neck
under AMP, the complete transformer in FP32, and only valid visible-box GDLoss
rows in explicit FP32.  The selective-decoder/deformable-attention and TF32
experiments remain only in the 99 workdirs; their temporary code/config flags
were removed.  The useful speed optimization remains AMP on the convolutional
feature extractor plus the synchronization-free, semantics-preserving GDLoss
path.

### 2026-07-14 BF16 implementation and stability audit

The source configs were subsequently changed from FP16 to BF16 for future
launches.  This does not change the already-running 197 `0714_03` process,
which loaded the earlier FP16/full-transformer-FP32 config when it started.

Current BF16 boundaries:

- `AmpOptimWrapper(dtype='bfloat16', loss_scale=1.0)`; BF16 does not need FP16
  gradient scaling.
- The retained boundary uses BF16 only for backbone, neck, and the shared
  encoder.  Encoder outputs are converted once to FP32; query initialization,
  decoder, head, matching, and all losses remain FP32.
- BF16 decoder support and its deformable-attention casting path have been
  removed.  Decoder BF16 is no longer a supported or planned configuration.
- RT-DETR nearest-neighbor FPN upsampling remains FP32 because PyTorch
  2.0/CUDA 11.8 has no BF16 `upsample_nearest2d` kernel.
- GDLoss remains FP32.  The active `xy_wh_r` KLD path uses a direct analytical
  formula in the predicted box's principal-axis frame, avoiding covariance
  construction, determinant and matrix inverse entirely.  It clamps width and
  height before reciprocal/log and enforces the analytical non-negative KL
  bound before `log1p`.  The generic covariance path retains determinant
  protection for non-`xy_wh_r` representations and other GD loss variants.

The fixed-order BF16 stress run initially reproduced the prior NaN at about
iteration 90 only in `d1.dn_loss_iou_prev/curr`.  The corresponding L1 losses
were finite, confirming that BF16 itself cannot fix an FP32 GDLoss covariance
failure.  After the KLD stability correction, the same seed and sample order
completed 200 DDP iterations with no NaN, Inf, exception, or unused-parameter
failure.  At iteration 200, total loss was `26.9892`, grad norm was `139.1643`,
and memory was about `7.45GB` on RTX 3090.

The first stable BF16 run measured `0.7586s/iter` versus `0.5677s` for the
controlled FP16/broad-FP32 profile.  A same-code repeat showed that this was
not a reliable precision-only comparison: its fixed first 60 batches ran at
`0.5885s/iter`, while an immediately preceding analytical-KLD run measured
`0.7691s/iter` on the same GPU.  The 31% same-code swing indicates transient
GPU load, clock, thermal, or resource contention in the slower profiles.  The
repeatable clean result currently puts BF16 about 3.7% behind FP16, not 33.6%.
BF16 is therefore numerically validated and close to FP16 throughput, but a
longer isolated A/B is still required before claiming either format is faster.

A direct hybrid-boundary comparison then limited AMP to backbone/neck and ran
the complete transformer and head in FP32.  On the same fixed first 60 batches,
BF16 hybrid measured `0.5768s/iter` versus `0.5677s/iter` for FP16 hybrid, so
BF16 was about 1.6% slower.  Memory was effectively identical at about
`8.45GB`, because the FP32 transformer dominates activation storage.  This is
kept only as a historical throughput comparison: FP16 is no longer used due
to the numerical-stability requirement.

The retained BF16 boundary includes backbone, neck, and the shared RT-DETR encoder,
then converts encoder outputs to FP32 once and disables autocast for query
initialization, decoder, head, matching, and GDLoss.  A new
`fp32_after_encoder_loss` model flag implements this boundary without repeated
per-layer casts.  The conversion is reported separately as
`encoder_to_fp32`, and loss, raw-forward, and prediction head calls are also
protected from an enclosing autocast context.  An automated CUDA test checks
the BF16 encoder output, FP32 post-encoder boundary, and finite gradients
through the cast.  On adjacent fixed-order 60-iteration runs, full FP32 measured
`0.7358s/iter` and `11.02GB`, while BF16-through-encoder measured
`0.6871s/iter` and `7.18GB`: about 6.6% faster and 34.9% less memory.  Component
timings showed backbone/neck improving from `0.0537s` to `0.0385s` and encoder
from `0.0251s` to `0.0175s`.  All logged losses and gradients remained finite.
One additional BF16 run was affected by the same whole-GPU timing variability
seen in earlier profiles, so the component-level gain is more reliable than a
single total-iteration percentage.

After removing the unsupported BF16-decoder path, the repaired boundary was
validated on local GPU0 for 20 fixed-order iterations with
`find_unused_parameters=False`.  The run completed without NaN, Inf, unused
parameters, or runtime errors; iteration 20 reported loss `33.2643`, finite
grad norm `1862.9223`, and `7.18GB` memory.  The explicit encoder-output cast
cost about `0.0003s/iter`, while encoder and decoder measured about `0.016s`
and `0.011s`.  The smoke-test workdir is
`/data4/litianhao/PairMmot/workdir_99/tmp_profile_0715_bf16_boundary_fixed`.

The formal half-data BF16 validation was launched on local 99 GPUs `0,1` at
2026-07-15 00:21 CST in screen `pairmot_0715_01_bf16_99`.  It uses
`AmpOptimWrapper(dtype='bfloat16', loss_scale=1.0)`, BF16 through the encoder,
FP32 thereafter, and `find_unused_parameters=False`; validation and TrackEval
remain enabled while image drawing is disabled.  Training was verified through
epoch 1 iteration 100 with finite loss `23.2721` and grad norm `61.5421`, about
`1.327s/iter`, and no NaN or unused-parameter error.  Workdir and log:
`/data4/litianhao/PairMmot/workdir_99/0715_01_0704_01_half_unique_allgt_bf16_encoder_findfalse`.

The run subsequently finished epoch 72 with all 18 TrackEval points.  Its
unique best HOTA point is async 18 / val_det epoch 71: `cls_HOTA=46.531`, `det_HOTA=58.484`, sum
`105.015`, or `+1.372` over `0704_01 resume`.  Its independent best AP point is
also epoch 72: pair mAP `0.2445`, pair AP50 `0.4257`.  Thus the retained BF16
boundary shows no observed accuracy degradation.

The unexpectedly slow initial run was traced to
`TORCH_DISTRIBUTED_DEBUG=DETAIL` in the local launch script.  On PyTorch 2.0.1
this wraps and validates DDP collectives; data time stayed near `0.03s` and the
GPUs were correctly bound, but formal training took about `1.32s/iter`.  A
concurrent two-GPU no-DETAIL control on GPUs `2,3` measured component
`iter_wall=0.78-0.82s`, so BF16 itself was not the slowdown.  DETAIL has been
removed from the launcher; the already-running process retains its launch-time
environment until restarted.  The slow workdir was then cleared and the
experiment was restarted from the adapted pretrain at 2026-07-15 00:48 CST,
without resume.  The fresh run reached epoch 1 iteration 50 at `0.7776s/iter`,
with finite loss `30.1673`, grad norm `84.7378`, and no unused-parameter error;
the ETA fell from about 13 hours to about 7.5 hours.

A same-host two-GPU FP32 control was then run on idle GPUs `2,3`, with the same
batch size, model, fixed sample order, `find_unused_parameters=False`, and no
DDP DETAIL.  Over the four component-timer samples at iterations 5/10/15/20,
BF16-through-encoder averaged `0.801s/iter` versus `0.904s/iter` for FP32, an
approximately 11.4% speedup; excluding the earliest warm-up sample gives a
roughly 9-10% gain.  Peak logged model memory was about `7.18GB` versus
`11.02GB` per GPU, a reduction of about 35%.  The modest speed gain is expected
because query initialization, decoder, head, matching, losses, backward
communication, and optimizer work remain FP32 or CPU-bound.

The first covariance-stabilized KLD implementation was added at about 22:43
on 2026-07-14 after the fixed-order run reproduced the DN IoU NaN.  A follow-up
analytical `xy_wh_r` implementation removed its matrix overhead.  In isolated
forward/backward benchmarks it was 30-39% faster than the stabilized
covariance implementation.  In the controlled first-60-iteration BF16 run it
reduced mean `head_loss` from `0.3156s` to `0.3039s` (3.7%); the corresponding
`iter_wall` change from `0.7788s` to `0.7691s` is only a noisy 1.25%.  A fresh
120-iteration fixed-order stability run crossed the former iteration-90 NaN
sample with all losses and gradients finite.  PairGDCost timing did not change because
Hungarian matching uses a separate KLD implementation in
`projects/rotated_dino/rotated_dino/match_cost.py`; therefore the earlier
33.6% BF16 slowdown cannot be attributed to GDLoss covariance protection.

Workdirs and logs:

```text
/data4/litianhao/PairMmot/workdir_99/tmp_profile_0714_pair_amp_findunused_false_v4
/data4/litianhao/PairMmot/workdir_99/tmp_profile_0714_pair_amp_fastgdloss_v1
/data4/litianhao/PairMmot/workdir_99/tmp_profile_0714_pair_selective_amp
/data4/litianhao/PairMmot/workdir_99/tmp_profile_0714_control_selective
/data4/litianhao/PairMmot/workdir_99/tmp_profile_0714_control_fp32_decoder
/data4/litianhao/PairMmot/workdir_99/tmp_profile_0714_control_broad
/data4/litianhao/PairMmot/workdir_99/tmp_profile_0714_control_broad_tf32
/data4/litianhao/PairMmot/workdir_99/tmp_profile_0714_pair_bf16_ddp_stable
/data4/litianhao/PairMmot/workdir_99/tmp_profile_0714_pair_bf16_analytic_gdloss
/data4/litianhao/PairMmot/workdir_99/tmp_profile_0714_pair_bf16_analytic_gdloss_stability
/data4/litianhao/PairMmot/workdir_99/tmp_profile_0714_pair_bf16_backbone_neck_fp32_rest
/data4/litianhao/PairMmot/workdir_99/tmp_profile_0714_pair_bf16_through_encoder
/data4/litianhao/PairMmot/workdir_99/tmp_profile_0714_pair_bf16_through_encoder_repeat
/data4/litianhao/PairMmot/workdir_99/tmp_profile_0714_pair_fp32_fixed60
/data4/litianhao/PairMmot/workdir_252/0714_01_0704_resume_coco365_full_unique_allgt_amp
/data4/litianhao/PairMmot/workdir_252/0714_01_0704_resume_coco365_full_unique_allgt_amp/queue_amp.log
/data4/litianhao/PairMmot/workdir_252/0714_01_0704_resume_coco365_full_unique_allgt_amp/launch_amp.log
/data4/litianhao/PairMmot/workdir_197/0714_02_0704_01_half_unique_allgt_amp_fp32transformer
/data4/litianhao/PairMmot/workdir_197/0714_02_0704_01_half_unique_allgt_amp_fp32transformer/launch.log
/data4/litianhao/PairMmot/workdir_197/tmp_validate_0714_amp_fixed_gpu5
/data4/litianhao/PairMmot/workdir_197/0714_03_0704_01_half_unique_allgt_hybrid_amp_fixed
/data4/litianhao/PairMmot/workdir_197/0714_03_0704_01_half_unique_allgt_hybrid_amp_fixed/launch.log
```

## 2026-07-15 Proposal zero-shot 状态

本机 99 的 `0715_07_full_baseline_elliptical_spectral_zeroshot` 已完成。实验使用空闲
GPU2，直接评测 252 full-data baseline 的 `epoch_72.pth`，未训练；现有 top-k、单侧
可见候选、unique selection 和真实 GMC 均保持不变，只在 pair affinity 中加入低开销
elliptical motion 与 5 点 box spectral descriptor。

独立 tracking 指标为 `cls_HOTA=52.780`、`det_HOTA=60.244`。相对 full baseline 分别
变化 `+0.406`、`-0.074`；用于选择最佳点的两项 HOTA 之和提高 `0.332`。独立 AP 指标
为 pair mAP `0.2952`、pair AP50 `0.5105`。详细设计、类别变化和耗时见
`projects/multispec_pair_rotated_rtdetr/docs/reports/20260714_module_ablation_report.md`
第 7 节。

```text
/data4/litianhao/PairMmot/workdir_99/0715_07_full_baseline_elliptical_spectral_zeroshot
```

`0715_08_full_classaware_elliptical_spectral_rank30_zeroshot` 已在本机 99 完成，使用
full baseline `epoch_72.pth` 做纯 zero-shot 评测。最终结果为
`cls_HOTA=52.921`、`det_HOTA=60.876`，相对 baseline 分别提高 `0.547`、`0.558`；
pair mAP 为 `0.2953`，pair AP50 为 `0.5108`。该版本使用类别门控，只保留为方法诊断
上界，不再作为最终通用方案。

```text
/data4/litianhao/PairMmot/workdir_99/0715_08_full_classaware_elliptical_spectral_rank30_zeroshot
```

`0716_01_full_sizeaware_elliptical_spectral_rank30_zeroshot` 去掉所有按类别选择 motion 或
spectrum 的分支，改为归一化面积 `3.5e-4` 门控：小目标回退 isotropic motion 并启用
relative spectral，大目标使用 elliptical motion。结果为 `cls_HOTA=52.886`、
`det_HOTA=60.942`，相对 baseline 分别提高 `0.512`、`0.624`；pair mAP 为 `0.2947`，
pair AP50 为 `0.5094`。该版本为正式通用方案，下一全局编号为 `0716_02`。

```text
/data4/litianhao/PairMmot/workdir_99/0716_01_full_sizeaware_elliptical_spectral_rank30_zeroshot
```
