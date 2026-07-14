# 2026-07-06 Multi-Server Experiment Plan

This file is the living multi-server state record for PairMOT experiments.
Update the status tables here whenever code is synced, a job is launched, or a
server path/credential convention changes.

Last updated: 2026-07-12 01:20 CST.

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
2026-07-11 is `litianhao`; `litianhao01@10.106.14.197` currently returns
`Permission denied (publickey)`.

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
