# 2026-07-06 Multi-Server Experiment Plan

This file is the living multi-server state record for PairMOT experiments.
Update the status tables here whenever code is synced, a job is launched, or a
server path/credential convention changes.

Last updated: 2026-07-19 CST.

Current per-server status dashboard:
[`20260719_multi_server_experiment_status.md`](20260719_multi_server_experiment_status.md).

## Server Status

| Server | Role | SSH from 99 | Code root | Shared root | Work dir | Conda |
| --- | --- | --- | --- | --- | --- | --- |
| local `10.106.14.99` | current control/source workspace and execution resource | local shell as `wangying01` | `/data/users/wangying01/lth/PairMOT/ai4rs` | `/data4/litianhao/PairMmot` | `/data4/litianhao/PairMmot/workdir_99` | `/data/users/wangying01/anaconda3/envs/py310` |
| `10.106.14.197` | execution resource | `ssh -i ~/.ssh/litianhao@10.106.14.197/id_rsa litianhao@10.106.14.197` | `/data/users/litianhao/PairMOT/ai4rs` | `/data4/litianhao/PairMmot` | `/data4/litianhao/PairMmot/workdir_197` | `/data/users/litianhao/anaconda3/envs/py310` |
| `10.106.15.178` | two RTX 5090 execution resource; one GPU available for PairMOT | `ssh -i ~/.ssh/litianhao01@10.106.15.178/id_rsa litianhao01@10.106.15.178` | `/data1/users/litianhao01/PairMOT/ai4rs` | `/data4/litianhao/PairMmot` | `/data4/litianhao/PairMmot/workdir_178` | `/data1/users/litianhao01/anaconda3/envs/py310`, PyTorch `2.7.0+cu128` |
| `10.106.15.252` | execution resource | `ssh -i ~/.ssh/litianhao01@10.106.15.252/id_ed25519 litianhao01@10.106.15.252` | `/data/users/litianhao01/PairMmot/ai4rs` | `/data4/litianhao/PairMmot` | `/data4/litianhao/PairMmot/workdir_252` | `/data/users/litianhao01/anaconda3/envs/py310` |
| AutoDL `autodl-container-b77mjk6jn5-c7ceaf44` | temporary two-GPU execution resource | transient password SSH; local command is ignored under `autodl/ssh.md` | `/root/PairMOT/ai4rs` | `/root/autodl-fs/PairMOT_assets` | `/root/autodl-tmp/work_dirs` | image base Python, PyTorch `2.8.0+cu128` retained |
| AutoDL `autodl-container-10m07jujmn-fe6c2d42` | historical temporary resource for completed `0719_01` | transient password SSH from ignored `autodl/ssh.md` | `/root/PairMOT/ai4rs` | `/root/autodl-fs/PairMOT_assets` | `/root/autodl-tmp/work_dirs` | image PyTorch `2.8.0+cu128`; NumPy/SciPy/OpenCV `1.26.4/1.12.0/4.10.0.84` |

SSH directory convention: subdirectory names under `~/.ssh` are
`username@ip+port`, with port omitted for default `22`.  The 197 key directory
for 197 is `litianhao@10.106.14.197`, matching the verified login account
`litianhao`.

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
| 2026-07-17 | `0717_02` | paper original-hard Liquid strict control, full data, 1200x900, BF16 | 99 | `0717_03` |
| 2026-07-17 | `0717_03` | hard-sampled soft-context Liquid candidate, full data, 1200x900, BF16 | 197 | `0718_01` |
| 2026-07-18 | `0718_01` | AutoDL Liquid path with independent groups and explicit difference/product pair coupling | 197 | `0718_02` |
| 2026-07-18 | `0718_02` | anchor-residual competitive Liquid candidate; independent groups with bounded common routing and pair-conditioned content evidence | 99 validation | `0718_03` |
| 2026-07-18 | `0718_03` | evidence-consistent adaptive-anchor ARCR Liquid; relax stable anchors only when learned and image-conditioned evidence agree | 99 queued GPU 0,1 | `0718_04` |
| 2026-07-18 | `0718_04` | SASE-Liquid; shared scale-adaptive sparse spectral evidence for pre-sampling routing and local LAF enhancement | 252 GPU 0,1 | `0718_05` |
| 2026-07-19 | `0719_01` | pair-consensus relaxed-set Liquid with pair-aligned compact-detail enhancement | AutoDL GPU 0,1 | `0719_02` |
| 2026-07-19 | `0719_02` | reliability-weighted pair-consensus control | 178 GPU 0 | `0719_03` |
| 2026-07-19 | `0719_03` | strict PACDE ablation: pair-consensus relaxed-set Liquid without compact-detail enhancement | 252 GPU 0,1 | `0719_04` |
| 2026-07-19 | `0719_04` | paper-protocol replay of the historical `0711_01`: independent 8-group sampler + Wide LAF + GroupMod, without pair routing/transport or set constraints | 197 GPU 4,5, queued after `0718_06` | `0719_05` |
| 2026-07-19 | `0719_05` | strict Paper Base rerun with unchanged seed/protocol and equivalent single-GPU global batch | 178 GPU 0, queued after `0719_02` | `0719_06` |
| 2026-07-19 | `0719_06` | Paper Base + validated long-tail positive-class reweighting | 99 GPU 0,1, queued after `0718_05` | `0719_07` |
| 2026-07-20 | `0720_01` | `0718_01` + gate-mass tangent fusion quality conservation | 252 GPU 0,1 | `0720_02` |
| 2026-07-20 | `0720_02` | `0718_01` + response-weighted fusion quality conservation | 197 GPU 4,5 | `0720_03` |
| 2026-07-20 | `0720_03` | `0718_01` + dual-moment fusion quality conservation | 99 GPU 0,1, queued after `0719_06` | `0720_04` |
| 2026-07-21 | `0721_01` | `0718_01` + response-weighted fusion quality conservation; single-GPU global batch 8 with validated tmpfs JPEG cache and NVMe fallback | 178 GPU 0, queued after `0719_05` final evaluation | `0721_02` |
| 2026-07-21 | `0721_02` | Accuracy-fixed strict rerun of `0718_01`; independent groups and difference/product pair coupling, without quality conservation | completed on 197 with 72 epochs and 18/18 TrackEval; best epoch 72 is `54.327/61.659` cls/det HOTA | `0721_03` |
| 2026-07-21 | `0721_03` | BSR-Liquid; replace global route statistics with 12x16 blockwise eight-band recurrent descriptors and block hidden mean/std/max aggregation; use corrected Negative-DN outer-band sampling and contrastive-group attention mask | running on 178 GPU 0; epoch 64 and 15/18 TrackEval at 2026-07-23 03:44; stage-best epoch 60 has cls/det HOTA `52.943/60.739`, below the same-DN epoch-60 control by `1.549/0.625` | `0721_04` |
| 2026-07-21 | `0721_04` | BSAC-Liquid; accuracy-fixed `0718_01` plus 24-parameter physical-band/kernel-slot calibration before shared Conv3D | completed 72 epochs and 18/18 TrackEval on 252; unique best epoch 72 has cls/det HOTA `54.530/61.528` and same-epoch pair mAP/AP50 `0.3211/0.5353` | `0721_05` |
| 2026-07-21 | `0721_05` | DSE-Liquid; accuracy-fixed `0718_01` plus identity-initialized channel mean/RMS evidence mixing before SE/LAF | completed 72 epochs and 18/18 TrackEval on 99; unique best epoch 72 has cls/det HOTA `54.635/61.895` and same-epoch pair mAP/AP50 `0.3254/0.5438` | `0721_06` |
| 2026-07-21 | `0721_06` | CSPR-Liquid candidate; replace global route statistics with detached 24x32 cyclic shared-Conv3D spectral preview statistics | implemented and tested; not queued | `0721_07` |
| 2026-07-22 | `0722_01` | `0721_02` PairDN-generator correction: replace the near-zero-capable negative product with signed `[1,2)` sampling and let positive/negative blocks attend within each contrastive group; this is not a box-noise-only ablation | running on 197 GPU 4,5 since 2026-07-22 10:34; epoch 7 with 1/18 TrackEval at 12:20 | `0722_02` |
| 2026-07-23 | `0723_01` | Accuracy-fixed `0718_01` with pair-coherent relative DN noise, 2:1 positive/negative slots, harder positives, rotated-IoU-filtered negatives, padding-query isolation, and representation-consistent LE180-start0 L1 with fixed angle weight | completed 72 epochs and 18/18 TrackEval on local 99; unique best epoch 64 is `53.955/62.032`, a dual Paper-Base improvement of `+0.641/+0.050` | `0723_02` |
| 2026-07-23 | `0723_02` | strict `0723_01` ablation changing only pair-side DN relative noise from shared to independently sampled | completed 72 epochs and 18/18 TrackEval on 252; unique best epoch 72 is `53.637/61.679`, so independent noise raises cls but lowers det relative to Paper Base | `0723_03` |
| 2026-07-23 | `0723_03` | `0723_01` + 16-parameter identity-initialized DSE mean/RMS fusion evidence | completed 72 epochs and 18/18 TrackEval on 197; unique best epoch 68 is `55.036/61.745`, improving cls but leaving det below Paper Base | `0723_04` |
| 2026-07-23 | `0723_04` | `0723_01` + detached 24x32 shared-Conv3D CSPR route preview; single-GPU batch 8 preserves global batch | completed 72 epochs and 18/18 TrackEval on 178; unique best epoch 72 is `54.523/61.738`, or `+1.209/-0.244` versus Paper Base; CSPR is rejected for future extension | `0723_05` |
| 2026-07-24 | `0723_05` | `0723_01` + local consistency-preserving DSE; retain the mean-evidence path and add an eight-parameter zero-initialized bounded SE-logit residual from normalized channel dispersion | completed 72 epochs and 18/18 TrackEval; unique best epoch 68 is `53.536/61.619`, or `+0.222/-0.363` versus Paper Base; local CP-DSE is rejected and local 99 is intentionally left idle | `0723_06` |
| 2026-07-24 | `0723_06` | `0723_01` + pair-global consistency-preserving DSE; pool normalized dispersion into one shared group correction for both frames to prioritize association consistency | completed 72 epochs and 18/18 TrackEval; unique best epoch 72 is `54.124/61.914`, with det AssA `+0.645` and det DetA `-0.432` versus Paper Base | `0723_07` |
| 2026-07-24 | `0723_07` | `0723_01` + Pair Evidence Consensus Gate; contract paired SE gates only where spectral coverage and Conv3D evidence agree while preserving their pair mean exactly | completed 72 epochs and 18/18 TrackEval on 252; unique best epoch 52 is `53.693/61.151`, or `+0.379/-0.831` versus Paper Base; det loss is dominated by DetA `-1.227`, so PECG is rejected | `0723_08` |
| 2026-07-24 | `0723_08` | `0723_01` + Spectral-Coordinate Pair Dispersion; project read-only group dispersion to physical bands with soft coverage, form pair consensus in spectral coordinates, project into each frame route, and use a zero-mean bounded eight-parameter residual | completed 72 epochs and 18/18 TrackEval on 178; unique best epoch 72 is `54.465/61.213`, or `+1.151/-0.769` versus Paper Base | `0725_01` |
| 2026-07-25 | `0725_01` | `0723_01` + DSE + pair-global CP-DSE; combine DSE's local mean/RMS detection evidence with CP-DSE's pair-shared association residual, based on their complementary det DetA/AssA outcomes | completed 72 epochs and 18/18 TrackEval on 197; unique best epoch 72 is `55.126/61.998`, a dual Paper-Base improvement of `+1.812/+0.016`; versus DSE at epoch 72 it gains `+0.328/+0.245` | `0725_02` |
| 2026-07-25 | `0725_02` | `0725_01` with the pair-global CP-DSE residual centered across eight groups; physical batch 4 plus accumulation 2 uses 4000-micro-iter warmup and EMA `interval=2,gamma=4000` to preserve the original optimizer-update time scale | completed 72 epochs and 18/18 TrackEval on 178; unique best epoch 68 is cls HOTA `53.730` and det HOTA `61.864`, so centering does not exceed Base+Liquid | `0725_03` |
| 2026-07-25 | `0725_03` | `0725_01` + Detection-Tangent CP-DSE; project the pair-shared CP residual away from the first-order detection-importance direction formed by existing Conv3D second moments and pooled DSE-gate sensitivity | completed 72 epochs and 18/18 TrackEval on 252; unique best epoch 72 is `54.229/61.808`, or `+0.915/-0.174` versus Paper Base and `-0.897/-0.190` versus its direct parent | `0726_01` |
| 2026-07-26 | `0726_01` | `0725_01` + Sparse-Reserve CP-DSE; use the pair-shared spatial RMS-minus-mean of the existing normalized dispersion map to attenuate only negative CP residuals where sparse target evidence exists; positive residuals remain unchanged | 59 local/remote tests, config deepcopy, hashes and exact 4-iter DDP smoke passed; formal 197 GPU 4,5 run passed iter 100 at 01:59 with finite total, DN, encoder and gradient values | `0726_02` |
| 2026-07-26 | `0726_02` | `0723_01` Base + Liquid followed by the `0705_01` encoder design: global bidirectional pair-temporal MHA on P5 before FPN and zero-gated pyramid-local adapters on P3/P4/P5 after FPN | completed 72 epochs and 18/18 TrackEval on AutoDL; unique best epoch 72 is `54.742/61.631` cls/det HOTA with pair mAP/AP50 `0.3223/0.5429`. Versus direct Base+Liquid best it is `+0.787/-0.401`: classification and AP improve, while det DetA/AssA fall `0.386/0.299`, motivating the common/detail and dual-evidence successors | `0726_03` |
| 2026-07-26 | `0726_03` | `0726_02` encoder successor: retain P5 global MHA, replace directional post-FPN pyramid-local with an order-equivariant common/detail adapter whose invariant reliability gate controls an odd local detail transform and whose opposite residuals preserve the pair mean exactly | epoch 32 remains a strict `+0.159/+0.078` cls/det HOTA gain versus Base+Liquid. cls DetA/AssA are `+1.189/-1.937`, while det DetA/AssA are `+0.340/-0.195`; coverage improves but classification association remains the limiting factor, preserving the target for queued `0727_04` | `0726_04` |
| 2026-07-27 | `0727_01` | fixed `0723_01` Liquid + `0705_01` P5 MHA; replace post-FPN local fusion with a frame-swap-equivariant dual-evidence adapter that adds shared common residuals for detection and opposite signed-detail residuals for association | epoch 28 remains a strict dual gain of `+0.031/+0.445` cls/det HOTA versus same-epoch Base+Liquid, extending the run to six consecutive dual-gain points. However det DetA/AssA are now `+1.523/-1.147` and the cls margin is nearly zero, so late branch growth is consuming association margin and keeps `0727_08` branch trust well targeted | `0727_02` |
| 2026-07-27 | `0727_02` | spatially selective successor of `0727_01`; add a two-channel local common/detail magnitude gate so P3/P4/P5 temporal updates focus on reliable object evidence without changing Liquid or adding a loss | stopped after epoch 12 at `-0.676/-1.042` cls/det HOTA versus Base+Liquid; det DetA/AssA are `-1.052/-0.829` and mAP is `-0.0108`, identifying spatial suppression of the common detection branch as the failure | `0727_03` |
| 2026-07-27 | `0727_03` | scale-split successor of `0727_01`: retain shared common evidence on P3/P4/P5 but restrict signed temporal detail to P4/P5, based on historical evidence that P3 local interaction is detection-oriented while higher levels better support classification/association | queue stopped at 02:40 before any smoke/formal directory was created. Epoch-8 decomposition showed common evidence, not P3 detail, produced the actual DetA/AssA tradeoff, so this candidate did not target the observed failure and was replaced by `0727_06` | `0727_04` |
| 2026-07-27 | `0727_04` | association-conservative successor of `0726_03`: retain P5 MHA and pair-mean-preserving common/detail post-FPN adapter, but cap each signed-detail update by the original pair-detail channel RMS using detached, parameter-free statistics | epoch 36 parent improves cls/det HOTA by `+0.688/+0.416` and det DetA/AssA simultaneously by `+0.272/+0.744`, weakening the earlier over-energy diagnosis. 18 tests, full build and 252 audit passed; queue remains alive, but launch now requires a final parent-trajectory review rather than automatic handoff | `0727_05` |
| 2026-07-27 | `0727_05` | conservative alternative to dual common residuals: retain the pair-mean-preserving detail-only adapter and use scale-normalized local common/detail energy to produce a zero-logit, unit-initialized spatial reliability modulation for signed detail only | 57 additional parameters and no loss; 20 local/178 tests, exact hashes and full build passed, including elementwise equality to the parent at initialization with nonzero gamma. Prepared but not launched. `0727_01` epoch 4 is `-0.492/-6.257` versus same-epoch Base+Liquid, so epoch 8 will decide whether this replaces queued `0727_03` | `0727_06` |
| 2026-07-27 | `0727_06` | association-conservative common-evidence successor: replace the channel-mixing additive common residual with a positive bounded spatial scalar shared by both frames; retain P5 MHA and signed detail | queue canceled at 03:47 before smoke/formal launch. `0727_01` epoch 12 recovers det AssA to `+0.006` while retaining DetA `+1.449` versus Base+Liquid, so removing channel-mixing common capacity no longer targets an observed failure; replaced by `0727_08` | `0727_07` |
| 2026-07-27 | `0727_07` | branch-energy trust-region successor of `0727_02`: retain its spatial common/detail gates, but separately cap shared-common and signed-detail update RMS by the corresponding input-evidence RMS on every sample and channel | queue canceled at 05:24 before smoke/formal launch. The epoch-12 `0727_02` failure is caused by suppressing the common detection branch, which an upper energy cap cannot repair; replaced by `0727_09` | `0727_08` |
| 2026-07-27 | `0727_08` | strict `0727_01` successor that retains the validated channel-mixing common/detail branches and adds only the parameter-free per-branch RMS trust region, without the `0727_02` spatial gate | epoch 32 `0727_01` is still a strict dual gain (`+0.248/+0.553` cls/det HOTA), but det DetA/AssA are `+1.524/-0.906` and EMA branch gamma reaches `2.926`. The cap therefore targets late-stage stability and must preserve the validated free branch rather than being assumed beneficial. 24 local/178 tests, full build, exact hashes and config/shell audit passed; strict queue remains healthy | `0727_09` |
| 2026-07-27 | `0727_09` | strict `0727_01` successor that preserves its common detection branch exactly and applies a zero-initialized unit-output spatial reliability gate only to signed temporal detail | epoch 12 is `+1.123/-0.202` cls/det HOTA versus same-epoch Base+Liquid: cls DetA/AssA and det DetA improve, but det AssA falls `1.057`. It is also `-0.470/-1.108` behind `0727_01`, confirming that the unconstrained spatial gate improves local evidence but degrades association and the parent optimization path. Continue the run for a complete trajectory; structural correction is delegated to `0727_10` | `0727_10` |
| 2026-07-27 | `0727_10` | strict `0727_09` successor: detach the local common/detail energy descriptor and normalize each detail spatial modulation to unit spatial mean, so it can only redistribute signed-detail evidence without globally scaling it or injecting descriptor gradients into shared features | no added parameters, loss or threshold; exact parent function remains at initialization. 26 local/197 tests, config deepcopy, full `22,758,832`-parameter build, shell audit and six exact hashes passed. Strict 197 queue started at 08:10 and will run its own real-data 4-iter DDP smoke only after `0727_09` reaches epoch 72, 18/18 evaluations and free GPUs | `0727_11` |

## Current Paper Runs

| Date | Server | Experiment | GPUs | Status | Log |
| --- | --- | --- | --- | --- | --- |
| 2026-07-16 | local `10.106.14.99` | `0716_02_paper_base_r18_coco_full_1200x900_bf16_orderedpairs_reboot_fresh` | `0,1` | completed 72 epochs and 18/18 async TrackEval points; unique best epoch 68 has cls HOTA 53.314, det HOTA 61.982, same-epoch pair mAP 0.3149 and AP50 0.5225 | `/data4/litianhao/PairMmot/workdir_99/0716_02_paper_base_r18_coco_full_1200x900_bf16_orderedpairs_reboot_fresh/launch.log` |
| 2026-07-16 | local `10.106.14.99` | `0716_03_paper_base_plus_liquid_r18_coco_full_1200x900_bf16_orderedpairs` | `2,3` | canceled and fully cleaned at 16:41 CST after GPU 2 hardware drop (`0000:B1:00.0: Unknown Error`); stopped in epoch 1 after iter 1000, no formal checkpoint, must fresh train on healthy GPUs; GPU 0/1 Base unaffected | `/data4/litianhao/PairMmot/workdir_99/0716_03_paper_base_plus_liquid_r18_coco_full_1200x900_bf16_orderedpairs/launch.log` |
| 2026-07-16 | `10.106.14.197` | `0716_03_paper_base_plus_liquid_r18_coco_full_1200x900_bf16_orderedpairs_fresh` | `0,3` | stopped intentionally at epoch 21 iter 50 without resume after confirming cross-group set collapse in the soft argmax preview; retained as diagnostic history | `/data4/litianhao/PairMmot/workdir_197/0716_03_paper_base_plus_liquid_r18_coco_full_1200x900_bf16_orderedpairs_fresh/launch.log` |
| 2026-07-16 | `10.106.14.197` | `0716_04_paper_base_plus_liquid_groupsetunique_r18_coco_full_1200x900_bf16_orderedpairs_fresh` | `0,3` | running; fresh start at 23:22 CST after 20/20 remote sampler tests; epoch 1 iter 50 is 0.9771 s/iter with finite losses/gradients and hard preview `unique_sets=8.00`, `max_set_repeat=1.00` | `/data4/litianhao/PairMmot/workdir_197/0716_04_paper_base_plus_liquid_groupsetunique_r18_coco_full_1200x900_bf16_orderedpairs_fresh/launch.log` |
| 2026-07-16 | `10.106.15.252` | `0716_05_paper_base_plus_liquid_groupsetunique_encoder_r18_coco_full_1200x900_bf16_orderedpairs_fresh` | `0,1` | completed 72 epochs and 18/18 TrackEval; unique best epoch 72 has cls/det HOTA `54.635/61.488` and same-epoch pair mAP/AP50 `0.3215/0.5467` | `/data4/litianhao/PairMmot/workdir_252/0716_05_paper_base_plus_liquid_groupsetunique_encoder_r18_coco_full_1200x900_bf16_orderedpairs_fresh/launch.log` |
| 2026-07-17 | local `10.106.14.99` | `0717_01_paper_base_plus_liquid_settransport_r18_coco_full_1200x900_bf16_orderedpairs_fresh` | `2,3` | canceled intentionally at epoch 2 iter 250 because local GPUs 2/3 have prior drop-card risk; full process group and screen removed, both GPUs released to 10 MiB; model code, 23 tests and separate 100-iter DDP validation are retained, but this incomplete run is not a result | `/data4/litianhao/PairMmot/workdir_99/0717_01_paper_base_plus_liquid_settransport_r18_coco_full_1200x900_bf16_orderedpairs_fresh/launch.log` |
| 2026-07-17 | AutoDL `autodl-container-b77mjk6jn5-c7ceaf44` | `0717_01_paper_base_plus_liquid_settransport_r18_coco_full_1200x900_bf16_orderedpairs_autodl_fresh` | `0,1` | completed 72 epochs and 18/18 TrackEval; unique best epoch 68 has cls HOTA 54.941, det HOTA 61.836, pair mAP 0.3196 and AP50 0.5359; finalizer analysis was recovered manually after its no-card shell lacked `python`, and 940 MB of selected checkpoints/results are preserved under shared FS | `/root/autodl-fs/PairMOT_results/0717_01` |
| 2026-07-17 | local `10.106.14.99` | `0717_02_paper_base_plus_liquid_originalhard_r18_coco_full_1200x900_bf16_orderedpairs_fresh` | `0,1` | completed 72 epochs and 18/18 TrackEval; unique best epoch 72 has cls HOTA 54.335, det HOTA 61.445, same-epoch pair mAP 0.3215 and AP50 0.5429. Relative to Paper Base best, cls HOTA is +1.021 while det HOTA is -0.537 | `/data4/litianhao/PairMmot/workdir_99/0717_02_paper_base_plus_liquid_originalhard_r18_coco_full_1200x900_bf16_orderedpairs_fresh/launch.log` |
| 2026-07-17 | `10.106.14.197` | `0717_03_paper_base_plus_liquid_hardsoftcontext_r18_coco_full_1200x900_bf16_orderedpairs_fresh` | `4,5` | intentionally stopped at epoch 15 iter 150 on 2026-07-18; latest periodic checkpoint is epoch 12 and is retained, but this experiment will not be resumed | `/data4/litianhao/PairMmot/workdir_197/0717_03_paper_base_plus_liquid_hardsoftcontext_r18_coco_full_1200x900_bf16_orderedpairs_fresh/launch.log` |
| 2026-07-18 | `10.106.14.197` | `0718_01_paper_base_plus_liquid_independent_diffproduct_r18_coco_full_1200x900_bf16_orderedpairs_fresh` | `4,5` | completed 72 epochs and 18/18 TrackEval; unique best epoch 64 has cls/det HOTA `55.276/61.812` | `/data4/litianhao/PairMmot/workdir_197/0718_01_paper_base_plus_liquid_independent_diffproduct_r18_coco_full_1200x900_bf16_orderedpairs_fresh/launch.log` |
| 2026-07-18 | AutoDL `autodl-container-b77mjk6jn5-c7ceaf44` | `0718_02_paper_base_plus_liquid_anchorcompetitive_r18_coco_full_1200x900_bf16_orderedpairs_autodl_fresh` | `0,1` RTX 4080 SUPER | completed 72 epochs and 18/18 TrackEval; unique best epoch 64 has cls/det HOTA `53.357/61.054`, endpoint epoch 72 has `53.118/61.216`. Finalizer completed at 21:59 CST and archived report, logs, selected/final checkpoints and TrackEval to shared FS | `/autodl-fs/data/PairMOT_results/0718_02` |
| 2026-07-18 | local `10.106.14.99` | `0718_03_paper_base_plus_liquid_anchorcompetitive_adaptiveanchor_r18_coco_full_1200x900_bf16_orderedpairs_fresh` | `0,1` | completed 72 epochs and 18/18 TrackEval; unique best epoch 60 has cls/det HOTA `54.689/60.969`; endpoint epoch 72 is `54.645/60.969` | `/data4/litianhao/PairMmot/workdir_99/0718_03_paper_base_plus_liquid_anchorcompetitive_adaptiveanchor_r18_coco_full_1200x900_bf16_orderedpairs_fresh/launch.log` |
| 2026-07-18 | `10.106.15.252` | `0718_04_paper_liquid_adaptiveanchor_sase_r18_coco_full_1200x900_bf16_orderedpairs_fresh` | `0,1` | completed 72 epochs and 18/18 TrackEval; unique best epoch 68 has cls/det HOTA `53.398/60.928`; endpoint epoch 72 is `52.986/60.811` | `/data4/litianhao/PairMmot/workdir_252/0718_04_paper_liquid_adaptiveanchor_sase_r18_coco_full_1200x900_bf16_orderedpairs_fresh/launch.log` |
| 2026-07-18 | `10.106.14.197` | `0718_06_paper_liquid_cpas_settransport_r18_coco_full_1200x900_bf16_orderedpairs_fresh` | `4,5` | fresh-launched after code/config sync; epoch 1 iter 50 is 1.0627 s/iter, 10694 MiB/rank, finite loss 34.9247 and grad norm 158.0419; CPAS/Set-Transport/difference-product monitors active, GPU temperatures 59/50C | `/data4/litianhao/PairMmot/workdir_197/0718_06_paper_liquid_cpas_settransport_r18_coco_full_1200x900_bf16_orderedpairs_fresh/launch.log` |
| 2026-07-19 | `10.106.15.252` | `0719_03_paper_liquid_pairconsensus_relaxedset_nopacde_r18_coco_full_1200x900_bf16_orderedpairs_fresh` | `0,1` | strict single-variable PACDE ablation; 41 local unit tests and a real-data four-iteration two-GPU BF16 DDP smoke passed with `find_unused_parameters=False`, finite loss/grad and 10666 MiB/rank; fresh formal run started at 16:45 CST | `/data4/litianhao/PairMmot/workdir_252/0719_03_paper_liquid_pairconsensus_relaxedset_nopacde_r18_coco_full_1200x900_bf16_orderedpairs_fresh/launch.log` |
| 2026-07-19 | `10.106.14.197` | `0719_04_paper_liquid_widelaf_groupmod_r18_coco_full_1200x900_bf16_orderedpairs_fresh` | `4,5` | predecessor `0718_06` completed and the guarded queue launched `0719_04` at 20:41 CST; running normally at epoch 9 around 23:03. The strict replay keeps independent per-frame/per-group sampling, Wide LAF (`64`, 4 heads, overlap/spatial context) and GroupMod (`16`), while disabling pair router, PairTransport, SetTransport and cross-group uniqueness | `/data4/litianhao/PairMmot/workdir_197/0719_04_paper_liquid_widelaf_groupmod_r18_coco_full_1200x900_bf16_orderedpairs_fresh/launch.log` |
| 2026-07-19 | `10.106.15.178` | `0719_05_paper_base_rerun_r18_coco_full_1200x900_bf16_1xb8` | `0` | queued behind `0719_02`. This is the unmodified Paper Base model with the same seed 3407, COCO-adapted initialization, LR `1e-4`, 72 epochs, BF16 and full evaluation protocol; only the batch topology changes from `2x4` to the profile-validated `1x8` with 8 workers, preserving global batch 8. Queue requires the predecessor to exit and GPU 0 memory to remain below 2048 MiB for three consecutive checks | `/data4/litianhao/PairMmot/workdir_178/queue_0719_05_after_0719_02.log` |
| 2026-07-19 | local `10.106.14.99` | `0719_06_paper_base_longtail_reweight_r18_coco_full_1200x900_bf16_orderedpairs_fresh` | `0,1` | completed 72 epochs and final validation at 00:16 CST on 2026-07-21. This is a strict Paper Base single-variable experiment using positive-class weights `[1.0,1.3,1.0,1.25,1.8,1.6,1.7,1.25]`; its exit released the guarded `0720_03` queue | `/data4/litianhao/PairMmot/workdir_99/0719_06_paper_base_longtail_reweight_r18_coco_full_1200x900_bf16_orderedpairs_fresh/launch.log` |
| 2026-07-21 | local `10.106.14.99` | `0720_03_paper_liquid_diffproduct_qc_dualmoment_r18_coco_full_1200x900_bf16_orderedpairs_accuracyfix_20260721_fresh` | `0,1` | the first formal run started at 00:19 and was intentionally stopped at 03:02 after about 2 h 43 min because it predated the 20260721 training-accuracy fixes. The replacement is a fresh, non-resumed run from the same adapted COCO checkpoint; epoch 1 iter 550 is about 0.96 s/iter with finite loss and gradient | `/data4/litianhao/PairMmot/workdir_99/0720_03_paper_liquid_diffproduct_qc_dualmoment_r18_coco_full_1200x900_bf16_orderedpairs_accuracyfix_20260721_fresh/launch.log` |
| 2026-07-21 | `10.106.15.178` | `0721_01_paper_liquid_diffproduct_qc_responsemass_r18_coco_full_1200x900_bf16_orderedpairs_1xb8_shm_accuracyfix_20260721_fresh` | `0` | the original run started at 01:23 and was intentionally stopped at 03:02 after about 1 h 39 min because it predated the 20260721 training-accuracy fixes. The replacement is fresh and non-resumed, retains the validated tmpfs JPEG cache and NVMe fallback, and reached epoch 1 iter 200 at about 0.83 s/iter in the stable compute interval. A transient `/data4` NFS stall recovered without intervention | `/data4/litianhao/PairMmot/workdir_178/0721_01_paper_liquid_diffproduct_qc_responsemass_r18_coco_full_1200x900_bf16_orderedpairs_1xb8_shm_accuracyfix_20260721_fresh/launch.log` |

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

## 2026-07-18 Liquid并行实验与99温控

- 99 GPU 0/1：`0718_03` adaptive-anchor ARCR Liquid，运行中。
- 252 GPU 0/1：`0718_04` SASE-Liquid，运行中；该机RTX 3090当前功率上限为200W，低于默认
  350W，是其吞吐较慢的重要原因。
- 99 GPU 2/3：`0718_05` PCDP-Liquid于15:06 CST fresh启动，15:14:29因GPU 3达到
  90摄氏度被温控守护按设计暂停，随后按决定主动终止并释放GPU 2/3；该不完整运行不进入
  结果表，workdir为
  `/data4/litianhao/PairMmot/workdir_99/0718_05_paper_liquid_adaptiveanchor_pcdp_r18_coco_full_1200x900_bf16_orderedpairs_fresh_interrupted_20260718_epoch1`。

`0718_05`训练进程位于独立process group，PGID记录在
`thermal_guard/train.pgid`。`thermal_pause_guard.sh`每10秒检查本机GPU 0--3温度；任一卡
达到90摄氏度，立即仅对0718_05进程组发送`SIGSTOP`，并写入
`thermal_guard/THERMAL_PAUSED`。检查温度和原因后可用下式继续：

```bash
kill -CONT -- -$(cat /data4/litianhao/PairMmot/workdir_99/0718_05_paper_liquid_adaptiveanchor_pcdp_r18_coco_full_1200x900_bf16_orderedpairs_fresh/thermal_guard/train.pgid)
```

并发日志分析不支持训练数据I/O竞争。`0718_03`在0718_05启动前、并发期间、0718_05暂停后
的平均`time/data_time`分别为`0.9125/0.0335`、`0.8609/0.0329`、
`0.9179/0.0353 s`；`0718_05`并发期间为`0.9465/0.0312 s`。HSMOT训练数据位于本地
`/data` NVMe，只有workdir位于`/data4` NFS；因此checkpoint/异步评测可能在每4 epoch产生
阶段性NFS竞争，但常规训练迭代没有可测的I/O瓶颈。本次实际限制是四卡并行散热。

23:39状态：197的`0718_01`已完成72 epochs和18/18 TrackEval，唯一最佳epoch 64为
`cls_HOTA=55.276, det_HOTA=61.812`；AutoDL `0718_02`也已完成18/18 TrackEval，唯一最佳
epoch 64为`cls_HOTA=53.357, det_HOTA=61.054`，finalizer已归档；99 `0718_03`运行至
epoch 44；252 `0718_04`运行至epoch 28；99 `0718_05`已主动终止；197 `0718_06`已启动至
epoch 1 iter 50，GPU 4/5温度正常。

## 2026-07-19 0718_05顺序重跑队列

00:49 CST，`0718_05 PCDP-Liquid`已重新排队到99当前`0718_03 adaptive-anchor ARCR`
之后，改用GPU 0/1顺序运行，不再与另一双卡任务并发。队列同时要求0718_03精确配置进程
退出且GPU 0/1显存均低于1024 MiB，满足后才从COCO适配权重fresh启动，禁止resume。

- screen：`pairmot_queue_0718_05_99`
- queue driver：`tools/queue_0718_05_after_0718_03_99.sh`
- queue log：`/data4/litianhao/PairMmot/workdir_99/0718_05_queue_driver.log`
- 正式workdir：`/data4/litianhao/PairMmot/workdir_99/0718_05_paper_liquid_adaptiveanchor_pcdp_r18_coco_full_1200x900_bf16_orderedpairs_fresh`
- 旧的不完整epoch 1运行：正式workdir加后缀`_interrupted_20260718_epoch1`

启动后的训练仍使用2卡BF16、`find_unused_parameters=False`、1200x900全量数据和ordered
pairs。00:49检查时队列正确报告`predecessor=1`，GPU 0/1显存为20899/20933 MiB，未发现
任何提前启动的0718_05训练进程。

## 2026-07-19 0719_01 AutoDL候选

`0719_01 pair-consensus relaxed-set Liquid`已在本机实现并同步至当前AutoDL系统盘
`/root/PairMOT`。模型让pair共享同一group，
采用不强制hard唯一的margin门控Set-Transport、同group Pair-Aligned Fusion及
Pair-Aligned Compact Detail Enhancement；不使用ARCR、CPAS、SASE、coverage-based
PairTransport或额外diversity loss。

2026-07-19在新AutoDL双RTX 4080 SUPER实例完成最终校验：PACDE相关测试累计`39/39`
通过；正式配置完成单卡2步BF16训练、反向、checkpoint和推理；正式1200x900、全局batch 8、
`find_unused_parameters=False`配置完成4步双卡DDP训练，loss/grad有限且无未使用参数或NCCL
错误。真实GMC重新严格核验为train `8297` pairs、test `5416` pairs。镜像保留PyTorch
`2.8.0+cu128`，仅清理混装的科学计算包并固定NumPy/SciPy/OpenCV为
`1.26.4/1.12.0/4.10.0.84`。

02:08 CST已在GPU 0,1 fresh启动正式72-epoch训练，workdir为
`/root/autodl-tmp/work_dirs/0719_01_paper_liquid_pairconsensus_relaxedset_r18_coco_full_1200x900_bf16_orderedpairs_autodl_fresh`。
启动验收时双卡显存约`18861/21071 MiB`、温度`42/43 C`。finalizer使用
`0716_02` Paper Base作为同协议基线，等待18个异步TrackEval后按唯一最大
`cls_HOTA + det_HOTA`选epoch、归档共享盘并自动关机；当前实例无GitHub deploy key，因此
共享盘结果为权威副本，不执行自动push。

同日确认初版PACDE正式训练稳定约`0.938 s/iter`，较AutoDL既有Liquid约`0.845 s/iter`
慢11%，主要是PACDE重复计算`groups.mean(dim=1)`并单独执行第二次大张量group乘加归约。
随后实施数学等价优化：复用LAF已有`x_se`，将SE gate与detail gate相加后只执行一次
`[B,C,G,H,W]`乘法和group归约；仅`return_sampling=True`调试路径保留原gated-groups输出。
40项测试（含非零PACDE输出与梯度等价）通过，100-iter正式尺寸双卡短测的稳定点由
`0.9340`降至`0.8931 s/iter`，显存由约`11212`降至`11153 MiB/rank`。

旧实现于epoch 1主动停止并保存在后缀`preopt_interrupted_epoch1`目录，不产生正式结果、不
resume。优化版继续使用同一实验ID，于02:22 CST从COCO适配权重重新fresh启动，finalizer也
重新绑定新launcher；正式结论只使用本次优化版run。

## 2026-07-19 178单卡容量测试

178登录账户确认是`litianhao01`，代码、环境和工作目录分别为
`/data1/users/litianhao01/PairMOT/ai4rs`、`~/anaconda3/envs/py310`和
`/data4/litianhao/PairMmot/workdir_178`。两张GPU均为32 GB RTX 5090；本次仅使用GPU 0，
测试结束后两卡均恢复空闲。当前projects、mmrotate和真实GMC已同步，GMC覆盖train/test
`8297/5416` pairs。

以当前最重的`0719_01 Pair-Consensus + PACDE`、全量协议1200x900 BF16做80样本短测：

| single-GPU mode | micro iters / optimizer updates | peak memory | result |
| --- | ---: | ---: | --- |
| `batch_size=8` | 10 / 10 | 21715 MiB | 完整通过，无OOM和数值错误；推荐正式方案 |
| `batch_size=4, accumulative_counts=2` | 20 / 10 | 11206 MiB | 完整通过，但出现scheduler先于optimizer step警告，不应直接用于既定论文调度 |

两种方式有效batch均为8，但累计方案对同样80个样本执行两倍micro前反向，且现有iter-based
warmup/scheduler每个micro step推进，不能与原两卡global-batch-8轨迹等价。由于`bs=8`
仍有约10 GB物理显存余量，178正式单卡实验应优先直接使用`batch_size=8`，保持原学习率、
epoch、scheduler和`accumulative_counts=1`。短测还观察到HSMOT所在`/data1`偶发数秒读取
抖动；正式启动前应以4--8个worker做较长吞吐测试，显存容量本身不是阻塞。

## 2026-07-19 178 worker profile与0719_02

在相同`0719_02`模型、GPU 0、单卡batch 8和320个训练样本上比较worker数量；统计去掉前
10 iter后的30个训练点：

| workers | time / iter | data time | compute time |
| ---: | ---: | ---: | ---: |
| 2 | 2.9916 s | 2.1171 s | 0.8746 s |
| 4 | 2.4447 s | 1.5367 s | 0.9080 s |
| 8 | **1.7405 s** | **0.8530 s** | 0.8875 s |
| 16 | 2.1036 s | 1.2235 s | 0.8801 s |

8 workers最优；16 workers已出现I/O和调度竞争。有效profile均完成40 iter，峰值MMEngine
显存约21.7 GB，无NaN或模型OOM。一次4-worker误启动与尚未退出的2-worker尾部进程重叠
而OOM，该进程未完成任何iter，已清理，不属于bs=8容量问题。

正式`0719_02`使用GPU 0、单卡batch 8、8 workers、BF16、full HSMOT 1200x900、ordered
gap-1 pairs和COCO适配初始化，于03:02 CST fresh启动；GPU 1保持空闲。iter 50为
`time=0.9082 s`、`data_time=0.0758 s`、`memory=21833 MiB`，无OOM、NaN或traceback。
workdir为：

```text
/data4/litianhao/PairMmot/workdir_178/0719_02_paper_liquid_pairconsensus_reliability_r18_coco_full_1200x900_bf16_1xb8
```

## 2026-07-23 0723_02 PairDN两帧独立噪声消融

在252服务器GPU 0、1启动`0723_02`。该实验以`0723_01`为严格父配置，模型、全量1200x900
数据、COCO适配初始化、BF16、全局batch 8、学习率、PairDN正负比与难度、DN attention mask
及loss均保持一致，唯一变量是将同一pair两帧共享的相对box噪声改为两帧独立采样，用于检验
pair-coherent DN对时序一致性的贡献。

- 配置：`projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_independent_diffproduct_pairdn_independent_le180_coco_full_1200x900_bf16_252.py`
- workdir：`/data4/litianhao/PairMmot/workdir_252/0723_02_paper_liquid_independent_diffproduct_pairdn_independent_le180_r18_coco_full_1200x900_bf16_orderedpairs_fresh`
- 启动时间：2026-07-23 03:30 CST
- 启动验证：精确配置4/4双卡DDP smoke完成；正式epoch 1 iter 50约`1.1441 s/iter`，总loss、
  DN loss、encoder proposal loss及梯度均有限，未发现OOM、NaN、NCCL、DDP reduction或
  unused-parameter错误。

## 2026-07-25 99让出与Liquid后续探索

- 99的`0723_05 local CP-DSE`已完成72 epochs和18/18 TrackEval；按用户要求，99此后暂时
  让出，不再安排或启动PairMOT任务。
- 197的`0725_01 DSE + pair-global CP-DSE`已完成。唯一最佳epoch 72为
  `55.126/61.998`，相对Paper Base双提升`+1.812/+0.016`；det DetA/AssA为
  `-0.138/+0.545`。后续`0726_01 Sparse-Reserve CP-DSE`仅保护稀疏证据位置对应group的
  负向残差，于2026-07-26 01:57在GPU 4/5 fresh启动并通过iter 100五项门槛。
- 252的`0723_07 PECG`已完成且否决；当前运行`0725_03 Detection-Tangent CP-DSE`。epoch 4
  相对未投影组合为`+0.861/+0.345`，关联分量改善但det DetA仍低`0.241`。
- 178的`0723_08 SCPD`已完成，唯一最佳epoch 72为`54.465/61.213`，失败主要来自det AssA
  下降。后续`0725_02`将DSE与去group均值的pair-global CP-DSE组合，限制残差只做相对group
  重分配。单卡physical batch 8 smoke反向OOM后改为physical batch 4 + accumulation 2。
  后续源码审计发现scheduler和EMA仍按micro-iteration推进，初始run在epoch 4停止、不resume；
  protocol-fixed配置将warmup改为4000 micro-iter、EMA改为`interval=2,gamma=4000`，保持
  原bs8按optimizer update计算的时间尺度。修复版epoch 12相对纯DSE为
  `-1.759/-0.275`，仅det AssA提高`0.965`，继续训练但不作为当前首选方向；GPU 1不占用。

## 2026-07-27 0727_11 MCDE Encoder

根据当前Encoder中期结果，主线改为保留`0727_01 Dual-Evidence`的自由双残差容量，不再
派生高分辨率空间门或严格能量守恒。`0727_11 Moment-Competitive Dual Evidence`在现有
common/detail逐通道描述中加入停止输入梯度的`RMS-mean(abs(x))`稀疏性矩，并将两路独立
sigmoid改为分支维softmax共享预算，以增强小目标稀疏证据感知并减少common/detail同时
过激更新。模型只增加1,536参数（完整模型`+0.00675%`），无额外loss、阈值或空间attention。

197上的`0727_10`因其空间门父配置`0727_09`在epoch 16已相对Base+Liquid双降而于12:14
取消，未运行smoke或正式训练。`0727_11`代码、29项测试、配置、完整模型及远端哈希全部
通过；原197队列于12:15启动，但在未运行smoke或正式训练前按用户要求迁移至AutoDL，并于
12:59关闭。

AutoDL使用单卡RTX 5090 physical batch 8，严格保持global batch 8、LR、EMA、BF16、
72 epochs及每4 epoch评测协议。HSMOT与非identity GMC完整验证为train/test
`8297/5416` pairs；正式尺寸4-iter smoke的总/DN/encoder loss及grad norm全部有限，
MMEngine峰值显存约21.8 GB。正式训练于12:57 fresh启动，iter 50为`0.819 s/iter`，
MCDE参数已更新且无OOM、NaN、未使用参数错误。唯一finalizer已挂接，等待训练和18/18
TrackEval完成后选择唯一最大`cls_HOTA+det_HOTA`的epoch、归档共享盘并自动关机。

## 2026-07-27 99双卡0727_12 CSEB Encoder

178上的`0727_01 Dual-Evidence`在17/18个评测点中的阶段唯一最佳epoch 64达到
cls/det HOTA `54.673/62.140`，相对固定Base+Liquid基准双提升`+0.718/+0.108`。
其P3/P4/P5的common/detail缩放学习方向明显不同，说明下一步应协调尺度间证据分配，同时
保留已验证有效的双残差容量。

`0727_12 Cross-Scale Evidence Budget`在`0727_01`上加入轻量跨尺度协调器：复用每层
common/detail全局描述形成32维尺度token，结合三尺度均值上下文预测逐通道分支预算；预算
在P3/P4/P5维softmax并乘3，只做尺度间重分配。描述停止梯度、输出零初始化，保证初始函数
与父配置完全一致；无额外loss、空间attention或高分辨率卷积。新增37,696参数，占完整模型
`0.166%`。

32项测试、配置深拷贝、完整模型构建及真实数据2卡4-iter DDP smoke均通过。99使用GPU 0、1、
global batch 8、BF16、`find_unused_parameters=False`和完整论文协议，于20:18 CST fresh
启动；epoch 1 iter 50为`0.9595 s/iter`，峰值约11.25 GB/rank，总/DN/encoder loss和梯度
有限。GPU 2、3保持空闲，不设置温度watchdog。

## 2026-07-28 0728_01 Decoder阶段条件启动

197的`0727_09 detail-only spatial reliability`结束后先完成18/18 TrackEval，并按唯一最大
`cls_HOTA + det_HOTA`选点。为了把“明显超过`0727_01`”变成可执行条件，只有其最佳点同时
满足cls HOTA `>54.437`、det HOTA `>62.393`且HOTA和`>=117.130`（至少高`0.300`）时，
才停止后续派生；否则进入Decoder探索。

首个Decoder实验编号为`0728_01`。它严格继承`0727_01`的Paper Base、`0723_01` Liquid、
P5双向temporal MHA、Dual-Evidence post-FPN encoder、proposal、PairDN、loss、COCO适配
初始化、full HSMOT 1200x900、BF16边界和global batch 8，唯一模型变量是加入历史
`0708_03` decoder：

- 使用`pointer/query_prev/query_curr`三状态逐层解码；
- 启用`pointer_to_prev`、`pointer_to_curr`及`pointer_update`的零初始化循环耦合；
- 不启用separate FFN，避免混入`0708_04`变量；
- 不修改matching、proposal top-k、PairDN或head loss。

本地完整模型构建成功，17项decoder单测通过，三层decoder的18个循环耦合权重/偏置在
初始化后均严格为0且可训练。代码与配置已同步197并通过远端配置解析。条件队列于
2026-07-28 02:15 CST启动，PID为`671579`；首个心跳确认`0727_09`仍在运行、epoch 72尚未
生成、TrackEval为16/18、GPU 4/5均被前序实验占用。前序满足完成条件后，队列会先运行
精确正式配置的双卡4-iter真实数据DDP smoke，验证checkpoint、关键loss、梯度和DDP状态，
通过后才fresh启动正式训练。

实际切换记录：`0727_09`于04:02完成18/18 TrackEval，唯一最佳epoch 72为
`54.106/62.321`，未超过`0727_01`，条件队列正确进入`0728_01` smoke。首次smoke发现
历史`0708_03`依赖`find_unused_parameters=True`：tri-state下遗留的`cross_fusion`不可达，
末层post-frame pointer更新没有后续消费者。保持预测逻辑不变，将这些结构性无梯度参数
排除训练后，18项单测及双卡4-iter真实数据smoke通过。正式训练于09:30 CST在197 GPU 4、5
fresh启动；09:31确认epoch 1 iter 50为`0.9245 s/iter`，关键loss和梯度有限，无DDP、NaN、
OOM或NCCL错误。
