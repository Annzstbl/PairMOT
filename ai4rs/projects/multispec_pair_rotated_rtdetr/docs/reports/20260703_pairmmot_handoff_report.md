# 20260703 PairMOT Handoff Report

## Current Code State

The repository code is synced to remote:

```text
branch: main
remote: origin/main
latest commit: fa66c0e 20260703_2
git status: clean, except ignored/untracked .dist_test temporary files
```

This means another server can get the current code with:

```bash
git clone https://github.com/Annzstbl/PairMOT.git
```

No extra code patch is required for the latest liquid experiment.

## Required Runtime Convention

Use the `py310` environment and launch from the repo root:

```bash
cd /data4/litianhao/PairMmot/ai4rs
source /data/users/litianhao01/anaconda3/etc/profile.d/conda.sh
conda activate py310
```

On the new server, adjust the `conda.sh` path if the conda installation is not
under `/data/users/litianhao01/anaconda3`.

## Required Files On New Server

### Pretrained Weight

The liquid config expects the pair-DN adapted pretrain checkpoint:

```text
pretrained_weights/o2_r18_hsmot_3dse_r2_e72_pair_dualcls_pairdn_adapted/pair_dualcls_pairdn_adapted_pretrain.pth
```

It has already been copied to:

```text
/data4/litianhao/PairMmot/pretrained_weights/o2_r18_hsmot_3dse_r2_e72_pair_dualcls_pairdn_adapted/pair_dualcls_pairdn_adapted_pretrain.pth
```

Checksum:

```text
sha256 = 2e93d16b07e1cde93dbc2e975070b728b12e9429dabe11d39f0cc8281996f82f
size   = 330M
```

### Dataset

The configs assume HSMOT data under:

```text
/data/users/litianhao01/PairMmot/data/hsmot/train
/data/users/litianhao01/PairMmot/data/hsmot/test
```

If the new server uses `/data4/litianhao/PairMmot/data`, either create a symlink:

```bash
mkdir -p /data/users/litianhao01/PairMmot
ln -s /data4/litianhao/PairMmot/data /data/users/litianhao01/PairMmot/data
```

or update the dataset root paths in the configs before launch.

### Optional TrackEval

Validation tracking uses TrackEval if `track_eval=True`:

```text
/data/users/litianhao01/PairMmot/TrackEval
```

If TrackEval is not ready on the new server, temporarily set `track_eval=False`
in the liquid config to run detection/AP validation only.

## Latest Liquid Experiment To Run Cross-server

Config:

```text
projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique_pairdn_liquid.py
```

Work directory:

```text
/data4/litianhao/PairMmot/workdir/0703_liquid_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique_pairdn_v2
```

Use a new `_v2` workdir to avoid mixing with the earlier liquid run.

Recommended two-GPU launch:

```bash
mkdir -p /data4/litianhao/PairMmot/workdir/0703_liquid_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique_pairdn_v2

: > /data4/litianhao/PairMmot/workdir/0703_liquid_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique_pairdn_v2/launch.log

setsid /bin/bash -lc "source /data/users/litianhao01/anaconda3/etc/profile.d/conda.sh; conda activate py310; cd /data4/litianhao/PairMmot/ai4rs; CUDA_VISIBLE_DEVICES=0,1 PORT=29713 bash tools/dist_train.sh projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique_pairdn_liquid.py 2 --work-dir /data4/litianhao/PairMmot/workdir/0703_liquid_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique_pairdn_v2" >> /data4/litianhao/PairMmot/workdir/0703_liquid_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique_pairdn_v2/launch.log 2>&1 < /dev/null &
```

Immediately verify:

```bash
sleep 15
tail -100 /data4/litianhao/PairMmot/workdir/0703_liquid_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique_pairdn_v2/launch.log
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
pgrep -af 'unique_pairdn_liquid|torch.distributed.launch|tools/train.py|dist_train.sh'
```

Healthy startup should show:

```text
Loads checkpoint ... pair_dualcls_pairdn_adapted_pretrain.pth
Checkpoints will be saved to ...
Epoch(train) [1][.../...] ...
```

## Latest Liquid Code Changes

The liquid sampler was modified because the previous run almost never changed
the spectral pattern. The earlier epoch-48 checkpoint showed:

```text
max_prob mean      = 0.9869
entropy mean       = 0.0647
changed_ratio mean = 0.0023
dominant pattern   = 012 / 123 / 234 / 345 / 456 / 567
```

Root cause:

```text
init_logit=8.0 created an approximately 8-logit fixed-window margin.
head.weight was initialized to zero.
P gradients used the low-resolution correction path only.
No explicit diversity or entropy objective exists.
```

The new liquid config now uses:

```python
liquid_sampler=dict(
    embed_dims=32,
    tau=2.0,
    hard=False,
    init_logit=2.0,
    head_weight_std=1e-3,
    eval_hard=True,
    lowres_grad_downsample=4,
    use_lowres_grad_correction=True,
)
```

And adds:

```python
dict(
    type='LiquidSamplerAnnealHook',
    tau_start=2.0,
    tau_end=0.5,
    anneal_epochs=36,
    hard_start_epoch=36,
    log_interval=200)
```

Expected behavior:

```text
early training: soft Gumbel sampling, higher tau, more exploration
later training: tau anneals to 0.5
after epoch 36: training hard=True
validation/inference: eval_hard=True, hard argmax one-hot sampling
```

Monitor the log lines:

```text
[LiquidSamplerAnneal] epoch=... tau=... hard=...
[LiquidSampler] max_prob=... entropy=... changed_ratio=... pattern=...
```

The new run should be considered healthier if `changed_ratio` rises materially
above the old mean `0.0023` during early training.

## Current Local Main-server Training

A separate 0703 typed baseline is currently running on the main server with two
GPUs. This is not the liquid experiment.

Config:

```text
projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_typed_pairtopk_v1.py
```

Workdir:

```text
/data/users/litianhao01/PairMmot/workdir/0703_baseline_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_typed_pairtopk_v1_pairdn
```

Status at report time:

```text
launched: 20260703_194112
max_epochs: 72
early stop monitor: pair/pair_mAP50_95
latest observed training: epoch 9
```

Validation at epoch 7:

```text
pair_AP50       = 0.3727
pair_mAP50_95   = 0.2103
independent_AP50= 0.3955
association_gap = 0.0228
val_det saved to val_det/epoch_07
async TrackEval launched to val_track_eval/val_track_0002
```

## AP / Tracking Validation Changes From Today

Pair AP now reports four GT views using shared IoU caches:

```text
all / both / new / disappear
```

This avoids recomputing full IoU per view. Synthetic benchmark showed the
shared implementation costs about `1.66x-1.96x` all-only AP, and about
`64%-73%` of the old repeated-breakdown implementation.

Validation also saves pair detections as txt:

```text
val_det/epoch_XX/*.txt
```

The txt format starts from adjacent pairs such as `01-02`; no `01-01` bootstrap
row is used for val_det.

Validation can launch TrackEval asynchronously so GPU training can continue
while CPU tracking/eval runs in the background.

## Known Interpretation From 0703 Analysis

The old `dualcls_nopres` both-visible-only view inflated pair AP. On the
100 hard new/disappear pair subset:

```text
0702 baseline all pair_mAP50_95  = 0.2157
0702 baseline both pair_mAP50_95 = 0.2292
new/disappear pair AP was essentially zero
```

Therefore future checkpoint selection should not rely on both-visible AP only.
Use all-GT `pair_mAP50_95`, and inspect tracking metrics when possible.

