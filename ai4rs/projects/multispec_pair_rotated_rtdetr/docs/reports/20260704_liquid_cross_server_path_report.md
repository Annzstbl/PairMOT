# 20260704 Liquid Cross-server Path Report

## Purpose

This update makes the latest liquid experiment easier to run on both servers
without hard-coding user-specific paths for shared assets.

Only the latest liquid config was changed:

```text
projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique_pairdn_liquid.py
```

Historical experiment configs were left unchanged.

## Path Convention

The config now resolves the PairMOT root from the config file location:

```text
PairMOT/
  ai4rs/
  data/
  pretrained_weights/
  workdir/aux/gmc_cache/
  TrackEval/
  hsmot/
```

This lets each server keep large assets wherever they fit, then expose them
through symlinks under `PairMOT`.

Required asset paths after resolution:

```text
PairMOT/data/hsmot/train
PairMOT/data/hsmot/test
PairMOT/data/hsmot/train_half.txt
PairMOT/pretrained_weights/o2_r18_hsmot_3dse_r2_e72_pair_dualcls_pairdn_adapted/pair_dualcls_pairdn_adapted_pretrain.pth
PairMOT/workdir/aux/gmc_cache/hsmot_train_gap1
PairMOT/workdir/aux/gmc_cache/hsmot_test_gap1
PairMOT/TrackEval
PairMOT/hsmot
```

On the current server, `PairMOT/data` is a symlink to:

```text
/data/users/litianhao/data
```

The pretrained checkpoint checksum was verified:

```text
sha256 = 2e93d16b07e1cde93dbc2e975070b728b12e9429dabe11d39f0cc8281996f82f
```

GMC cache was present:

```text
hsmot_train_gap1 files = 3839
hsmot_test_gap1 files  = 5416
```

## Work Directory

Because local disk is limited on this server, the liquid experiment output is
kept on `/data4`:

```text
/data4/litianhao/PairMmot/workdir_197/0703_liquid_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique_pairdn
```

The parent directory was checked as writable before launch.

## Launch Command Used

```bash
CUDA_VISIBLE_DEVICES=2,3 PORT=29713 bash tools/dist_train.sh \
  projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique_pairdn_liquid.py \
  2 \
  --work-dir /data4/litianhao/PairMmot/workdir_197/0703_liquid_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique_pairdn
```

The actual background launch also exported:

```bash
PYTHONPATH=/data/users/litianhao/PairMOT/TrackEval:/data/users/litianhao/PairMOT/hsmot:$PYTHONPATH
```

TrackEval subprocesses already extend `PYTHONPATH` internally with `ai4rs`,
`PairMOT`, and `PairMOT/hsmot`.

## Startup Verification

The first launch exposed a config formatting issue: MMEngine `cfg.pretty_text`
cannot format a config variable containing a `Path` object, because it renders
as an unquoted filesystem path. The config now converts the resolved root to a
string immediately.

The second launch started normally on GPUs 2 and 3:

```text
Loads checkpoint ... pair_dualcls_pairdn_adapted_pretrain.pth
Checkpoints will be saved to /data4/litianhao/PairMmot/workdir_197/...
Epoch(train) [1][ 50/484] ...
Epoch(train) [1][100/484] ...
Epoch(train) [1][250/484] ...
```

Liquid sampler diagnostics were healthy during early training:

```text
[LiquidSampler] max_prob=0.3220 entropy=1.8374 changed_ratio=0.4514
[LiquidSampler] max_prob=0.3441 entropy=1.8104 changed_ratio=0.4514
[LiquidSampler] max_prob=0.3401 entropy=1.7991 changed_ratio=0.4861
```

This is materially above the previous unhealthy `changed_ratio` mean of
`0.0023`.

## Notes

The pretrain load reports expected non-exact matching for the liquid stem and
new pair modules. Training continued after those warnings.

For another server, keep the same `PairMOT` relative asset layout and only
adjust the `/data4/.../workdir_197` output path if that server uses a different
large-disk mount.
