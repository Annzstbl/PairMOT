# Current Window Memory - 2026-07-14

## Active User Goal

The current task stream is focused on PairMOT baseline speed optimization and AMP validation. The immediate running experiment is a half-data 0704_01 baseline AMP check on server 197, launched after fixing AMP NaN issues by keeping the transformer/head-loss path in FP32 while allowing earlier modules to use AMP.

## Workspace

- Local repo: `/data/users/wangying01/lth/PairMOT`
- Main project: `/data/users/wangying01/lth/PairMOT/ai4rs/projects/multispec_pair_rotated_rtdetr`
- Shared experiment root: `/data4/litianhao/PairMmot`
- Local server workdir: `/data4/litianhao/PairMmot/workdir_99`
- Server 197 repo: `/data/users/litianhao/PairMOT/ai4rs`
- Server 197 workdir: `/data4/litianhao/PairMmot/workdir_197`
- Server 252 workdir: `/data4/litianhao/PairMmot/workdir_252`

## Current 197 AMP Experiment

- SSH identity: `~/.ssh/litianhao01@10.106.14.197/id_rsa`
- SSH command pattern: `ssh -i ~/.ssh/litianhao01@10.106.14.197/id_rsa -o StrictHostKeyChecking=no litianhao@10.106.14.197`
- Screen name: `pairmot_0714_02_amp_half`
- GPUs: `2,3`
- Port: `29827`
- Experiment workdir: `/data4/litianhao/PairMmot/workdir_197/0714_02_0704_01_half_unique_allgt_amp_fp32transformer`
- Config: `projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_amp_fastgdloss_197.py`
- Launch script: `ai4rs/projects/multispec_pair_rotated_rtdetr/tools/launch_0714_02_amp_half_197.sh`
- Launch log: `/data4/litianhao/PairMmot/workdir_197/0714_02_0704_01_half_unique_allgt_amp_fp32transformer/launch.log`
- MMEngine log: `/data4/litianhao/PairMmot/workdir_197/0714_02_0704_01_half_unique_allgt_amp_fp32transformer/20260714_193749/20260714_193749.log`
- Start time: `2026-07-14 19:37:40 CST`
- Verified stable through at least epoch 4 iter 50 in the latest observed log.

## AMP Speed Observation

Current conclusion: AMP compute is faster, but 197 end-to-end speed is partly limited by data/shared-storage jitter.

- 197 AMP stable compute windows: about `0.77-0.82 s/iter`, often around `0.80 s/iter`.
- 252 FP32 0704_resume reference windows: about `1.02-1.13 s/iter`, average around `1.06-1.08 s/iter`.
- AMP compute-side gain when `data_time` is normal: roughly `25%`.
- Recent 197 AMP window average was about `1.06 s/iter` because `data_time` sometimes jumped to `0.3-1.9 s`.
- Practical estimate: if `data_time` returns to about `0.03 s`, AMP is faster than FP32; if 197 keeps I/O spikes, wall-clock speed can look close to FP32.

## ETA For Current 197 Run

The schedule is 72 epochs, 484 iterations per epoch.

- Pure stable AMP compute estimate: around `6.5 min/epoch`.
- Observed with 197 I/O jitter: closer to `8-9+ min/epoch`.
- From epoch 3-4 on `2026-07-14 20:00-20:10 CST`, expected completion was roughly `2026-07-15 06:00-08:00 CST`.
- Heavy validation or TrackEval can push this later.

## Relevant Code Changes

### `multispec_pair_rotated_rtdetr.py`

Added a model flag:

- `fp32_transformer_loss`

When enabled, the model casts feature tensors to float and runs `forward_transformer` plus `bbox_head.loss` under `torch.cuda.amp.autocast(enabled=False)`. This keeps the transformer/head-loss path numerically stable while still using AMP for earlier computation.

Reason: full FP16 AMP repeatedly produced `grad_norm: nan`; lowering loss scale did not solve it. BF16 failed because `ms_deform_attn_forward_cuda` does not support BFloat16.

### `pair_rotated_rtdetr_head.py`

Improved rotated-box GD loss validity handling:

- sanitize rboxes to float
- wrap angle
- clamp width/height
- filter non-finite and invalid boxes
- add determinant precheck before GD loss
- compute GD loss with autocast disabled

Reason: AMP exposed numerical instability in rotated GD loss and proposal/head loss. The goal is correctness first, then speed.

## Current Config Notes

`o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_amp_fastgdloss_197.py`:

- imports the 0704_01 half baseline
- uses `AmpOptimWrapper`
- sets `loss_scale=dict(init_scale=128., growth_interval=2000)`
- sets `model.update(fp32_transformer_loss=True)`
- loads from `/data4/litianhao/PairMmot/pretrained_weights/o2_r18_hsmot_3dse_r2_e72_pair_dualcls_pairdn_adapted/pair_dualcls_pairdn_adapted_pretrain.pth`
- maps data and GMC paths to the 197 local repo/data layout
- writes to `/data4/litianhao/PairMmot/workdir_197/0714_02_0704_01_half_unique_allgt_amp_fp32transformer`

## Troubleshooting History

- Full FP16 AMP generated repeated NaN gradients.
- Loss scale experiments with `1`, `0.01`, and `1e-5` did not fully solve nonfinite gradients.
- BF16 was attempted but failed because the CUDA op `ms_deform_attn_forward_cuda` does not implement BFloat16.
- The stable compromise is `fp32_transformer_loss=True`.
- Single-GPU debug with this path ran finite for dozens of iterations.
- Official 2-GPU 197 run has finite `grad_norm` in observed logs.

## Baseline Context

The current canonical baseline remains `0704_resume`:

- Workdir: `/data4/litianhao/PairMmot/workdir_252/0704_01_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_resume_from_epoch40_to72`
- User-provided baseline metrics earlier:
  - `pair mAP=2.05`
  - `pairAP50=40.38`
  - `HOTA_cls=44.7`
  - `MOTA_cls=33.3`
  - `IDF_cls=51.9`
  - `HOTA_det=57.3`
  - `MOTAdet=50.5`
  - `IDF1_det=65.8`

For future reports, metrics should not be merged into composite displays. Best epoch should be selected by unique best based on `cls_HOTA + det_HOTA`.

## Dirty Working Tree At Time Of Memory Save

Modified files observed:

- `ai4rs/docs/20260706_multi_server_experiment_plan.md`
- `ai4rs/projects/multispec_pair_rotated_rtdetr/configs/tmp_profile_0714_coco365_full_single_gpu_amp.py`
- `ai4rs/projects/multispec_pair_rotated_rtdetr/configs/tmp_profile_0714_coco365_full_single_gpu_amp_findunused_false.py`
- `ai4rs/projects/multispec_pair_rotated_rtdetr/multispec_pair_rotated_rtdetr/multispec_pair_rotated_rtdetr.py`
- `ai4rs/projects/multispec_pair_rotated_rtdetr/multispec_pair_rotated_rtdetr/pair_rotated_rtdetr_head.py`

Untracked files observed:

- `ai4rs/projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_amp_fastgdloss_197.py`
- `ai4rs/projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_coco365_full_amp_252.py`
- `ai4rs/projects/multispec_pair_rotated_rtdetr/configs/tmp_profile_0714_pair_valid_fill_amp_99.py`
- `ai4rs/projects/multispec_pair_rotated_rtdetr/tools/launch_0714_02_amp_half_197.sh`
- `ai4rs/projects/multispec_pair_rotated_rtdetr/tools/queue_0714_amp_252.sh`

## Important Cautions

- Do not disable evaluation logic when running official experiments. Only visualization/plotting can be disabled if needed.
- For GMC, do not generate identity GMC matrices for official tracking/eval; use real GMC cache/matrices.
- Do not overwrite active or historical experiment directories; create new workdirs for resume or variants.
- Server 197 previously showed possible GPU instability concerns. Current AMP run uses GPUs `2,3`.
- When comparing AMP speed, separate compute time from `data_time` because shared-storage jitter can hide the AMP gain.

## Useful Commands

Check current 197 run:

```bash
ssh -i ~/.ssh/litianhao01@10.106.14.197/id_rsa -o StrictHostKeyChecking=no litianhao@10.106.14.197 "screen -ls && tail -n 80 /data4/litianhao/PairMmot/workdir_197/0714_02_0704_01_half_unique_allgt_amp_fp32transformer/launch.log"
```

Compute recent AMP speed:

```bash
rg "Epoch\(train\).*time:" /data4/litianhao/PairMmot/workdir_197/0714_02_0704_01_half_unique_allgt_amp_fp32transformer/launch.log | tail -40
```

Compare with FP32 0704_resume:

```bash
rg "Epoch\(train\).*time:" /data4/litianhao/PairMmot/workdir_252/0704_01_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_resume_from_epoch40_to72/launch.log | head -80
```
