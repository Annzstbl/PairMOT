# 2026-07-06 Multi-Server Experiment Plan

This file is the living multi-server state record for PairMOT experiments.
Update the status tables here whenever code is synced, a job is launched, or a
server path/credential convention changes.

Last updated: 2026-07-10 00:15 CST.

## Server Status

| Server | Role | SSH from 99 | Code root | Shared root | Work dir | Conda |
| --- | --- | --- | --- | --- | --- | --- |
| local `10.106.14.99` | current control/source workspace and execution resource | local shell as `wangying01` | `/data/users/wangying01/lth/PairMOT/ai4rs` | `/data4/litianhao/PairMmot` | `/data4/litianhao/PairMmot/workdir_99` | `/data/users/wangying01/anaconda3/envs/py310` |
| `10.106.14.197` | execution resource | `ssh -i ~/.ssh/litianhao01@10.106.14.197/id_rsa litianhao@10.106.14.197` | `/data/users/litianhao/PairMOT/ai4rs` | `/data4/litianhao/PairMmot` | `/data4/litianhao/PairMmot/workdir_197` | `/data/users/litianhao/anaconda3/envs/py310` |
| `10.106.15.178` | available resource, not verified this turn | key folder exists under `~/.ssh/litianhao01@10.106.15.178` | unknown | `/data4/litianhao/PairMmot` expected | unknown | unknown |
| `10.106.15.252` | available resource, not verified this turn | key folder exists under `~/.ssh/litianhao01@10.106.15.252` | unknown | `/data4/litianhao/PairMmot` expected | unknown | unknown |

SSH directory convention: subdirectory names under `~/.ssh` are
`username@ip+port`, with port omitted for default `22`.  The 197 key directory
is named `litianhao01@10.106.14.197`, but the verified login account on
2026-07-09 is `litianhao`, not `litianhao01`.

252 verified login on 2026-07-09:

```bash
ssh -i ~/.ssh/litianhao01@10.106.15.252/id_ed25519 litianhao01@10.106.15.252
```

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
