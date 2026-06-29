# Same-frame PairDN/NoDN Monitor - 2026-06-29

## Current Status

- Server reboot happened at `2026-06-29 07:55:53 CST`.
- Both same-frame experiments were interrupted by the reboot, not by early stopping or a logged training exception.
- Current user-level evidence cannot read kernel logs (`dmesg` requires permission), so the exact kernel-level reboot reason is still unverified.
- A new current-run monitor is running:
  `/data/users/litianhao01/PairMmot/workdir/system_monitor_20260629_sameframe/server_health_monitor.py`
- PairDN same-frame training has been resumed on GPU0/1 from epoch 60:
  `/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_sameframe_trainval_pretrain`
- NoDN same-frame training is currently paused after reboot at epoch 36 to avoid another immediate four-GPU full-load reboot.

## Resume Evidence

- PairDN resume log:
  `/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_sameframe_trainval_pretrain.resume_20260629_1028.log`
- It reports:
  - `Auto resumed ... epoch_60.pth`
  - `resumed epoch: 60, iter: 29040`

## Latest Metrics

PairDN same-frame validation:

| epoch | pair_AP50 | pair_mAP50_95 | independent_AP50 | independent_mAP50_95 | association_gap_AP50 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 0.4200 | 0.2277 | 0.4645 | 0.2498 | 0.0445 |
| 12 | 0.4303 | 0.2495 | 0.4651 | 0.2681 | 0.0348 |
| 18 | 0.4404 | 0.2523 | 0.4723 | 0.2695 | 0.0319 |
| 24 | 0.4480 | 0.2595 | 0.4825 | 0.2781 | 0.0345 |
| 30 | 0.4477 | 0.2582 | 0.4808 | 0.2763 | 0.0330 |
| 36 | 0.4570 | 0.2645 | 0.4883 | 0.2809 | 0.0313 |
| 42 | 0.4577 | 0.2651 | 0.4879 | 0.2805 | 0.0302 |
| 48 | 0.4595 | 0.2674 | 0.4904 | 0.2832 | 0.0309 |
| 54 | 0.4610 | 0.2692 | 0.4920 | 0.2850 | 0.0310 |
| 60 | 0.4603 | 0.2685 | 0.4911 | 0.2843 | 0.0308 |

NoDN same-frame validation:

| epoch | pair_AP50 | pair_mAP50_95 | independent_AP50 | independent_mAP50_95 | association_gap_AP50 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 0.4170 | 0.2368 | 0.4635 | 0.2616 | 0.0465 |
| 12 | 0.4252 | 0.2379 | 0.4612 | 0.2581 | 0.0360 |
| 18 | 0.4378 | 0.2490 | 0.4685 | 0.2656 | 0.0307 |
| 24 | 0.4487 | 0.2566 | 0.4778 | 0.2726 | 0.0290 |
| 30 | 0.4540 | 0.2657 | 0.4807 | 0.2807 | 0.0268 |
| 36 | 0.4601 | 0.2683 | 0.4900 | 0.2848 | 0.0299 |

## Interim Conclusions

- PairDN and NoDN are currently very close under same-frame train/val.
- NoDN reaches `pair_AP50=0.4601` and `independent_AP50=0.4900` by epoch 36, almost matching PairDN epoch 60.
- PairDN best so far is epoch 54: `pair_AP50=0.4610`, `independent_AP50=0.4920`.
- This suggests PairDN is not the sole cause of the independent AP gap versus the single-frame baseline.
- The same-frame pair architecture still appears to cap independent AP near 49 AP50, below the single-frame reference around 56 AP50.
- Because four-GPU concurrent training coincided with another reboot, the safer current action is sequential/two-GPU recovery unless power/thermal limits can be controlled by an admin.

## Next Actions

- Let resumed PairDN run to the next validation checkpoint and likely completion/early-stop.
- If GPU temperature remains stable and PairDN finishes, resume NoDN from epoch 36 on GPU0/1 or GPU2/3.
- Do not immediately resume both experiments in parallel unless the server stability issue is accepted or mitigated.

## 2026-06-29 10:33 CST Update

- PairDN resume remains healthy.
- Current active processes:
  - monitor: `server_health_monitor.py`, PID `12927`
  - PairDN parent: PID `13039`
  - PairDN workers: PIDs `13181`, `13182`
- PairDN resumed run is logging under:
  `/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_sameframe_trainval_pretrain/20260629_102947/20260629_102947.log`
- Latest observed training progress after resume:
  - `Epoch(train) [61][200/484]`
  - loss around `5.18`
  - no traceback/OOM/NCCL error observed
- Current GPU state while running only PairDN on GPU0/1:
  - GPU0 about `69-70C`, `~215-225W`, `16440MiB`
  - GPU1 about `73-75C`, `~210-222W`, `16442MiB`
  - GPU2/3 idle
- Decision: keep NoDN paused for now. Reason: GPU1 is already near mid-70C with only two GPUs active, power limit cannot be changed without admin permission, and the previous four-GPU run coincided with a reboot.

## 2026-06-29 10:38 CST Update

- PairDN same-frame resume is still running normally on GPU0/1.
- Latest observed progress:
  - log: `/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_sameframe_trainval_pretrain/20260629_102947/20260629_102947.log`
  - reached `Epoch(train) [62][50/484]`
  - recent loss around `4.93-5.22`
  - no traceback/OOM/NCCL error observed
- GPU state around `10:38 CST`:
  - GPU0 about `70-73C`, `~228-232W`, `16440MiB`
  - GPU1 about `75-80C`, `~217-233W`, `16442MiB`, fan at `100%`
  - GPU2/3 idle
- New parallel-experiment action:
  - Added guarded launcher: `/data/users/litianhao01/PairMmot/workdir/launch_nodn_single_gpu_when_safe_20260629.sh`
  - Guard log: `/data/users/litianhao01/PairMmot/workdir/nodn_single_gpu_guard_20260629.log`
  - It will resume NoDN on GPU2 as a single-GPU global-bs-compatible run with `train_dataloader.batch_size=8`.
  - Launch condition: GPU0 `<=74C`, GPU1 `<=78C`, GPU2 memory `<=100MiB`.
  - Current NoDN run log, once launched: `/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_sameframe_trainval_nodn_pretrain.resume_single_gpu_20260629.log`
- First guard attempt failed before Python due `conda.sh` + `set -u` (`PS1: unbound variable`); script was fixed by disabling `set -u` only around conda initialization.
- Current decision: NoDN is queued but not force-started while GPU1 remains at `~79-80C` with fan `100%`. This keeps the new experiment ready without immediately recreating the thermal/power risk seen before the reboot.

## 2026-06-29 10:40 CST Update

- PairDN same-frame resume remains healthy.
- Latest observed progress:
  - reached `Epoch(train) [62][250/484]`
  - recent loss range roughly `5.18-5.79`
  - next scheduled validation remains epoch 66; no new AP metric after epoch 60 yet
  - no traceback/OOM/NCCL error observed
- Current GPU state:
  - GPU0 about `73-74C`, `~235-241W`, `16440MiB`
  - GPU1 about `79-81C`, `~233-234W`, `16442MiB`, fan `100%`
  - GPU2/3 idle
- No active `gap1` or `gap1to2` training process was found in the current process table.
- Parallel experiment handling:
  - NoDN single-GPU bs=8 experiment remains queued by `/data/users/litianhao01/PairMmot/workdir/launch_nodn_single_gpu_when_safe_20260629.sh`.
  - Added `flock` single-instance protection to the guard script to prevent duplicate NoDN launches.
  - Old duplicate guard processes were stopped; one current guard is running.
  - Current guard launch condition is still GPU0 `<=74C`, GPU1 `<=78C`, GPU2 memory `<=100MiB`.
- Decision: do not force-start NoDN while GPU1 is at `~81C` and fan `100%`. The new experiment is prepared and will start automatically only if the thermal state drops into the configured safe window.

## 2026-06-29 10:43 CST Update

- Current process table still has no active `gap1` or `gap1to2` training process.
- Formal fixed gap experiments checked:

| experiment | status | latest checkpoint | latest/last metric |
| --- | --- | --- | --- |
| `gap1_fixed_20260628` | completed to epoch 72 | `epoch_72.pth` | epoch72 `pair_AP50=0.4353`, `independent_AP50=0.4846`, `pair_mAP50_95=0.2451`, `independent_mAP50_95=0.2831`, `association_gap_AP50=0.0493` |
| `gap1to2_fixed_20260628` | completed to epoch 72 | `epoch_72.pth` | epoch72 `pair_AP50=0.4277`, `independent_AP50=0.4788`, `pair_mAP50_95=0.2365`, `independent_mAP50_95=0.2746`, `association_gap_AP50=0.0512` |
| older `gap1train` | stopped at epoch 60 | `epoch_60.pth` | no in-run validation metric found in its latest log |

- Formal gap conclusion from the completed fixed runs:
  - `gap1_fixed` is slightly better than `gap1to2_fixed` at epoch72.
  - Adding random interval 1-2 did not improve gap=1 validation in this run; it slightly reduced both pair AP50 and independent AP50.
- Same-frame PairDN resume remains active:
  - reached at least `Epoch(train) [62][400/484]`
  - no new validation metric yet after epoch60; next expected validation is epoch66.
- NoDN single-GPU guard remains queued. At this check GPU1 briefly returned to `~80C`, so NoDN was not force-started.

## 2026-06-29 10:48 CST Update

- NoDN single-GPU bs=8 guard triggered at `2026-06-29 10:45:17 CST` when GPU0/GPU1/GPU2 were approximately `72C/78C/38C`.
- NoDN resume was successful:
  - command log: `/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_sameframe_trainval_nodn_pretrain.resume_single_gpu_20260629.log`
  - resumed from `/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_sameframe_trainval_nodn_pretrain/epoch_36.pth`
  - log reports `resumed epoch: 36, iter: 17424`
  - reached `Epoch(train) [37][100/484]`
  - recent NoDN loss around `4.88-4.99`
- Thermal result of three-GPU concurrent load:
  - GPU0 about `74C`, GPU1 about `81C`, GPU2 about `84C`
  - GPU2 power about `263W`, memory about `21396MiB`
  - GPU2 reached `84C` within a few minutes despite using only one extra GPU.
- Action taken:
  - stopped NoDN single-GPU run with `kill 25663`.
  - confirmed NoDN processes exited and GPU2 memory released.
  - GPU2 dropped to about `60C` immediately after stopping.
- Decision:
  - Do not keep NoDN running concurrently with PairDN under current thermal conditions.
  - The NoDN resume path is verified, but further NoDN continuation should wait until PairDN finishes, or require lower power/thermal settings/admin power limit.
  - PairDN same-frame remains the active training process on GPU0/1.

## 2026-06-29 10:55 CST Update

- PairDN same-frame remains active and healthy on GPU0/1.
- Latest observed progress:
  - active log: `/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_sameframe_trainval_pretrain/20260629_102947/20260629_102947.log`
  - reached `Epoch(train) [64][250/484]`
  - recent loss range roughly `5.14-5.53`
  - no traceback/OOM/NCCL error observed
  - next scheduled validation is still epoch 66; no new AP after epoch 60 yet
- Current thermal/resource state:
  - GPU0 about `72-73C`, `~232W`, `16440MiB`
  - GPU1 about `77-78C`, `~226W`, `16442MiB`, fan `100%`
  - GPU2 about `40C`, idle
  - GPU3 about `47C`, idle
- Parallel experiment decision:
  - Do not immediately start a second training while GPU1 is still near `78C` and fan `100%`.
  - Prepared a guarded low-LR continuation experiment instead:
    - script: `/data/users/litianhao01/PairMmot/workdir/launch_lr2e5_cont_when_safe_20260629.sh`
    - guard log: `/data/users/litianhao01/PairMmot/workdir/lr2e5_cont_guard_20260629.log`
    - run log, once launched: `/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_loadfrom_epoch72_to84_lr2e5_nosched.resume_20260629.log`
    - target config: `projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_loadfrom_epoch72_to84_lr2e5.py`
    - target workdir: `/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_loadfrom_epoch72_to84_lr2e5_nosched`
  - Guard launch condition: GPU0 `<=70C`, GPU1 `<=74C`, GPU2/GPU3 `<=50C`, GPU2/GPU3 memory `<=100MiB`.
  - Guard is running via `setsid`, PID `34025`; it currently waits instead of launching because GPU0/GPU1 are above the conservative thresholds.

## 2026-06-29 10:57 CST Update

- PairDN same-frame training remains active and healthy.
- Process state:
  - monitor PID `12927` active
  - PairDN launcher PID `13039`, torch launcher PID `13064`
  - rank processes PID `13181` and `13182`
  - lr2e-5 continuation guard PID `34025`
- Latest observed training progress:
  - reached `Epoch(train) [64][450/484]`
  - latest loss `4.9318`
  - LR remains `1e-4`
  - no traceback/OOM/NCCL error found
- Validation/checkpoint state:
  - current resumed run has no new `Epoch(val)` yet
  - checkpoint list still ends at `epoch_60.pth`
  - next expected validation/checkpoint is epoch 66
- Current GPU state:
  - GPU0 `~73C`, fan `93%`, `~233W`, `16440MiB`
  - GPU1 `~78C`, fan `100%`, `~227W`, `16442MiB`
  - GPU2 `~40C`, idle
  - GPU3 `~47C`, idle
- Decision:
  - Do not force-start a new parallel training while GPU1 remains near `78C` at `100%` fan.
  - Keep the lr2e-5 continuation guard waiting; it will only launch on GPU2/3 if GPU0/GPU1 cool below the conservative thresholds.
  - Next action is to wait for epoch66 validation AP, then compare against epoch60 (`pair_AP50=0.4603`, `independent_AP50=0.4911`) before deciding whether to stop, continue, or prioritize NoDN/lr2e-5 continuation.

## 2026-06-29 10:59 CST Update

- PairDN same-frame training has advanced into epoch 65.
- Latest observed progress:
  - `Epoch(train) [65][50/484]`
  - latest loss `5.4810`
  - LR still `1e-4`
  - no new validation metric yet
- Checkpoint state:
  - latest checkpoint is still `epoch_60.pth`
  - no `epoch_66.pth` yet
- Current GPU state:
  - GPU0 `~74C`, fan `93%`, `~233W`, `16440MiB`
  - GPU1 `~81C`, fan `100%`, `~229W`, `16442MiB`
  - GPU2/GPU3 idle
- Guard state:
  - lr2e-5 continuation guard PID `34025` remains active.
  - It has not launched because GPU0/GPU1 are still above the conservative temperature thresholds.
- Decision:
  - Continue current PairDN run and wait for epoch66 validation.
  - Do not start NoDN or lr2e-5 training manually under the current thermal state.
  - Primary next analysis is epoch66 AP versus epoch60 and prior best epoch54/60 plateau.

## 2026-06-29 11:00 CST Update

- PairDN same-frame training remains active.
- Latest observed progress:
  - `Epoch(train) [65][150/484]`
  - latest loss `5.2162`
  - LR still `1e-4`
  - no error found in the current log tail
- Validation/checkpoint state:
  - no `Epoch(val)` in the resumed `20260629_102947.log`
  - no `epoch_66.pth`; latest checkpoint remains `epoch_60.pth`
  - epoch66 validation has not started yet
- Current GPU state:
  - GPU0 `~73C`, fan `94%`, `~234W`, `16440MiB`
  - GPU1 `~78C`, fan `100%`, `~230W`, `16442MiB`
  - GPU2/GPU3 idle
- Guard state:
  - lr2e-5 continuation guard PID `34025` still active.
  - It has not launched; last logged threshold check still failed because GPU0/GPU1 were too warm.
- Decision:
  - No early-stop or parameter adjustment yet because there is no epoch66 AP evidence.
  - Keep waiting for epoch66 validation.
  - Do not manually start additional training while GPU1 remains at `100%` fan.

## 2026-06-29 11:01 CST Update

- PairDN same-frame training remains active and has advanced within epoch 65.
- Latest observed progress:
  - `Epoch(train) [65][250/484]`
  - latest loss `5.2708`
  - LR still `1e-4`
  - no error found in the latest log window
- Validation/checkpoint state:
  - no `Epoch(train) [66]` yet
  - no `Epoch(val)` yet
  - no `epoch_66.pth`; latest checkpoint remains `epoch_60.pth`
- Current GPU state:
  - GPU0 `~73C`, fan `94%`, `~238W`, `16440MiB`
  - GPU1 `~79C`, fan `100%`, `~231W`, `16442MiB`
  - GPU2/GPU3 idle
- Guard state:
  - lr2e-5 continuation guard PID `34025` remains active.
  - Latest guard check at `11:00:16 CST`: GPU0 `74C`, GPU1 `80C`, GPU2/GPU3 idle; conditions not met.
- Decision:
  - No manual parallel launch under current thermal state.
  - Continue waiting for epoch66 validation AP before making early-stop or hyperparameter decisions.

## 2026-06-29 11:08 CST Update

- PairDN same-frame main training remains active.
- Latest observed main-run progress:
  - resumed run has entered `Epoch(train) [66]`
  - latest checked point: `Epoch(train) [66][250/484]`
  - LR still `1e-4`
  - no epoch66 validation metric yet
- New parallel experiment attempted:
  - experiment: lr2e-5 continuation on GPU2/3
  - config: `projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_loadfrom_epoch72_to84_lr2e5.py`
  - workdir: `/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_loadfrom_epoch72_to84_lr2e5_nosched`
  - launched from `epoch_6.pth` at `11:05:17 CST`
  - it successfully resumed and reached at least `Epoch(train) [7][100/484]`
- Thermal outcome:
  - after enabling 4-GPU load, GPU0/1/2/3 reached about `75C/82C/77C/77C`
  - GPU1 was already at `100%` fan and `82C`
  - this reproduces the suspected whole-server thermal-risk pattern when 4 GPUs are active
  - I stopped the new lr2e-5 parallel run with SIGTERM at about `11:08 CST`
  - GPU2/3 memory was released and temperatures began falling
- Guard state:
  - the older lr2e-5 conservative guard was also stopped so it does not relaunch the same experiment automatically after cooldown.
- Decision:
  - keep only the main same-frame PairDN run active
  - do not use 4-GPU concurrent training under current cooling conditions
  - wait for epoch66 validation AP before early-stop or continuation decisions
  - if another parallel experiment is needed, use a stricter all-GPU thermal policy rather than checking only the target GPU pair
