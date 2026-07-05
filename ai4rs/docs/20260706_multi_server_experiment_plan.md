# 2026-07-06 Multi-Server Experiment Plan

This repo uses `10.106.12.252` as the source-of-truth code server.  Use
`10.106.14.99` and `10.106.14.197` only as execution resources unless a later
note explicitly changes ownership.

## Shared Rules

- Train with exactly two GPUs per experiment.
- Keep experiments fair: start from the adapted pretrain, not from a high-epoch
  checkpoint.
- Use `/data4/litianhao/PairMmot` for shared artifacts when running on 99 or
  197.
- Track/eval may use one GPU or CPU-side async evaluation as configured.
- Do not run the same config on two servers unless explicitly requested.

## Code Sync

On a resource server:

```bash
cd /data4/litianhao/PairMmot
git clone https://github.com/Annzstbl/PairMOT.git ai4rs
cd ai4rs
git checkout main
```

If `ai4rs` already exists:

```bash
cd /data4/litianhao/PairMmot/ai4rs
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
