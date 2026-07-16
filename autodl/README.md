# PairMOT AutoDL initialization

This directory provides a repeatable first-instance setup. It retains the
PyTorch shipped by the AutoDL image, installs the MM stack and `ai4rs`, safely
normalizes HSMOT, prepares the official R18 COCO initialization and real GMC
cache, then runs a two-iteration baseline training plus one-batch inference
smoke test.

## Run

Upload this directory to the AutoDL instance, select an image with PyTorch
`>=2.2.0` and a GPU, then run:

```bash
cd /path/to/autodl
bash bootstrap.sh
```

Defaults:

- system disk: code and environment under `/root/PairMOT`
- shared file storage: archive and pretrain under
  `/root/autodl-fs` (auto-detected); the validated GMC cache is also archived
  there after its first build
- high-I/O data disk: extracted HSMOT, GMC, work directories and logs under
  `/root/autodl-tmp`

The FS auto-detection order is `/root/autodl-fs`, `/root/autoldl-fs`, then
`/autodl-fs`. Set `FS_ROOT` explicitly if the instance uses another mount.

The extractor accepts either `hsmot/train,test` or direct `train,test` archive
layouts. It validates `mot` and `npy2jpg` under both splits and never overwrites
an incomplete existing target.

To use a pushed experiment branch or commit:

```bash
PAIRMOT_REF=my-branch bash bootstrap.sh
```

All paths can be overridden with `PAIRMOT_ROOT`, `HSMOT_ARCHIVE`, `HSMOT_ROOT`,
`FS_ROOT`, `ASSET_ROOT`, `PRETRAIN_ROOT`, and `GMC_ROOT`. The process is
idempotent: valid
data, adapted weights, and existing GMC JSON files are reused. A fresh data
disk restores GMC from `PairMOT_assets/gmc_cache_hsmot_gap1.tar.gz` instead of
recomputing sparse-LK/RANSAC. Do not set
`BUILD_GMC=0` for a final initialization; the formal baseline deliberately
requires real GMC and rejects missing, failed, non-finite, or identity entries.
Package installation uses the image's `PIP_INDEX_URL` when present, otherwise
the Tsinghua mirror from the upstream tutorial. Set `PACKAGE_INDEX_URL` to
override it.

After migrating a saved system image to an instance with a fresh data disk,
run the staging and smoke process again without reinstalling packages:

```bash
SKIP_INSTALL=1 bash bootstrap.sh
```

Checkpoints in `/root/autodl-tmp` are optimized for training I/O but are not
shared between instances. Copy selected checkpoints and final reports to the
FS before releasing an instance.

## After initialization

New shells automatically source `/root/PairMOT/autodl_runtime.env`. Start the
formal full-data, 1200x900, BF16 R18 baseline with:

```bash
cd "$AI4RS_ROOT"
python tools/train.py "$AUTODL_PAIRMOT_CONFIG"
```

For two GPUs:

```bash
cd "$AI4RS_ROOT"
bash tools/dist_train.sh "$AUTODL_PAIRMOT_CONFIG" 2
```

The smoke test intentionally uses a smaller 640x480 input and two iterations
to control billed time. It still loads the formal model/pretrain, real data and
GMC, executes BF16 forward/backward/optimizer work, saves a checkpoint, and
runs model inference. It does not run full HSMOT evaluation or TrackEval.

## Automatic result publication and shutdown

`finalize_and_shutdown.sh` can run inside the instance while training is in
progress. It waits for the launcher to exit and for all 18 asynchronous
TrackEval jobs to produce successful `metrics.json` files. It then selects the
unique best epoch by `cls_HOTA + det_HOTA`, generates Markdown and JSON reports,
backs up reports, logs, TrackEval outputs, and selected checkpoints to
`/root/autodl-fs/PairMOT_results`, pushes the reports to a dedicated GitHub
result branch, and invokes `/usr/bin/shutdown`.

Use a write-enabled GitHub deploy key scoped only to the PairMOT repository.
The default key path is `/root/.ssh/pairmot_results_ed25519`; do not reuse a
personal SSH key. A same-protocol baseline can be supplied at
`/root/autodl-fs/PairMOT_results/baselines/0716_04.json`. Without it, the
generated report selects the checkpoint but deliberately makes no improvement
claim.
