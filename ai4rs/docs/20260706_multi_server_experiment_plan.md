# 2026-07-06 Multi-Server Experiment Plan

This file is the living multi-server state record for PairMOT experiments.
Update the status tables here whenever code is synced, a job is launched, or a
server path/credential convention changes.

Last updated: 2026-08-04 07:44 CST.

Current per-server status dashboard:
[`20260719_multi_server_experiment_status.md`](20260719_multi_server_experiment_status.md).

## Server Status

Only server 252 has fixed GPU indices: GPU0/1. Servers 99, 178, and 197 have count-only caps of 2, 1, and 2 GPUs respectively; their indices may be selected from currently free cards without preempting external work. Server 252 is the slowest lane and is reserved for one mature or confirmation trajectory at a time.

At 07:09 CST, mature experiment `0803_13` completed epoch 28 on 252 fixed GPU0/1 at
`53.114/59.729`. It remains dual-positive against the original decoder at the same epoch
(`+0.937/+0.449`) but is not yet dual-positive against Encoder, so PGID `419164` continues to
epoch 32. The other servers retain count-only GPU caps; no GPU index is fixed outside 252.

At 07:20 CST, corrected `0803_23` completed epoch 4 at `36.342/44.739`, exceeding the
original decoder at the same point by `+2.036/+6.149` and Encoder by `+0.133/+5.986`.
The complete 50-sequence TrackEval closed successfully. Because decoder convergence is judged
on a mature trajectory, the same one-GPU job on 178 continues through epoch 8 and epoch 12.

At 07:29 CST, the orthogonal `0803_24` transported-shape-only candidate was prepared for
197 at clean HEAD `44395ea`. Its two-GPU launchers require the currently free indices to be
passed explicitly, preserving the count-only resource rule. It remains PREPARED/NO_GPU behind
the running `0803_18` e12 decision and the `0803_23` e8 result.

At 07:32 CST, `0803_18` completed epoch 8 at `42.014/47.865`, only
`+0.042/-0.313` against the original decoder and below the geometry-only branch. The full
artifacts are valid, and the same PGID `387859` continues to epoch 12 because epoch 8 is not
used as a direct rejection point.

At 07:44 CST, `0803_25` was prepared as the center-only orthogonal ablation of the strong
full-tangent candidate. Three targeted tests and a zero-parameter full-model build passed at
clean 178 HEAD `09a0d2f`. It is PREPARED/NO_GPU and its launcher requires one currently free
GPU index explicitly; it does not change the running `0803_23` allocation.

At 2026-08-03 23:15 CST, external work occupies 99 GPU0 while GPU1/2 are free. Because 99 has a count-only two-GPU cap, `0803_14` may use GPU1/2 after migration and smoke. Experiment 0803_13 reached e4 at `32.849/37.319` and continues to e8/e12 because early decoder convergence is not a rejection criterion.

At 23:22 CST, `0803_14` completed the 99 GPU1/2 smoke and reached formal iter50 with all five launch gates satisfied. It now runs to e4/e8/e12 on the faster 99 lane while 252 retains only the mature `0803_12` trajectory on fixed GPU0/1.

Experiment 0803_16 is the next orthogonal terminal projection: share only the final reference-local center displacement, leaving size and angle independent. It is parameter-free and queued after 0803_15 on the one-GPU 178 lane; preparation and CPU/build checks do not consume the active GPU allocation.

The isolated 0803_16 checkout at commit `c05cd21` passed its terminal-only call-count test and full model build: 22,771,111 parameters, zero parameter delta, and 711 state tensors. It remains PREPARED and consumes no GPU until the earlier 178 candidates finish.

Safe single-GPU smoke and formal launchers are now prepared for both 0803_15 and 0803_16. They create no workdir or GPU process during preparation; after the active 178 run releases its one-card allocation, each successor must pass a fresh real-data smoke before formal launch.

Both prepared 178 checkouts are clean at launch-provenance commit `e9f56dc`; the active 0803_13 checkout was not updated.

Experiment 0803_17 is a third zero-parameter 178 successor. At only the final
iterative classification layer it preserves each frame's residual class mean
and averages only the centered class-margin direction. It is class-permutation
equivariant, keeps DN and all box paths independent, and adds no reweighting,
attention, or layer. The isolated `e245127` checkout passed three targeted
tests and a full model comparison at 22,771,111 parameters, zero delta, and 711
state tensors. It remains PREPARED after 0803_15/16 and consumes no GPU.

Experiment 0803_13 reached a complete epoch-8 checkpoint at cls/det HOTA
`45.002/49.083`, a `+3.030/+0.905` gain over the original 0801_09 decoder at
the same epoch. All four AP diagnostics also improved, and the checkpoint,
50 sequences, 28 CSV files, 108 non-empty evaluation files, and async TrackEval
completion were verified. The 178 lane therefore remains on 0803_13 through
epoch 12 and later confirmation; prepared successors do not preempt it.

At epoch 12, 0803_13 reached cls/det HOTA `48.289/54.539`, still a dual
`+0.894/+0.103` improvement over the original decoder at the same checkpoint.
The combined parent-line gain has, however, narrowed from `+3.935` at epoch 8
to `+0.997` at epoch 12. Because this is still dual-positive and decoder
convergence is late, the 178 one-GPU lane remains assigned through epoch 16 to
test whether the terminal-only constraint prevents the later reversal seen in
the all-layer geometry branch. The complete checkpoint, AP diagnostics, 50
sequences, 28 CSV files, 108 evaluation files, and async completion are
verified.

The complete epoch-8 results for 0803_11 and 0803_12 are respectively
`40.377/45.730` and `40.430/46.542` cls/det HOTA. Both are below the original
decoder and all four AP diagnostics are also lower, while terminal-only
0803_13 dominates both. They nevertheless continue to epoch 12 because epoch
8 is not a direct decoder rejection point. No additional late-two-layer or
progressive-geometry descendants will be prepared from these weak branches.

Experiment 0803_12 completed its mature epoch-12 evaluation at
`45.677/52.131`, below the original decoder at the same checkpoint by
`1.718/2.305`. Together with its complete epoch-4 and epoch-8 results, this is
a three-node negative trajectory rather than an early rejection. After the
381,003,318-byte checkpoint, AP diagnostics, 50 sequences, 28 CSV files, 108
evaluation files, and async completion were verified, exact PGID `123974` was
terminated (`23 -> 0`). All four 252 GPUs returned to 1 MiB; the slowest lane
is now idle and reserved for later mature confirmation.

Experiment 0803_14 reached epoch 4 at `30.813/36.985` cls/det HOTA, below both
Encoder and the terminal full-size 0803_13 at the same point. Its checkpoint,
AP diagnostics, 50 sequences, 28 CSV files, and 108 evaluation files are
complete. This is recorded as early evidence only; 99 GPU1/2 continues to
epoch 8 and 12 without touching the external process on GPU0.

At epoch 8, 0803_14 reached `41.384/47.315`, below the original decoder by
`0.588/0.863` and below terminal full-size 0803_13 by `3.618/1.768`. Pair
mAP/AP50 `0.206968/0.367917` and both-independent `0.251401/0.427578` are also
lower. The 99 lane nevertheless continues this decoder to epoch 12 before the
mature decision, then gives priority to the prepared 0803_17 semantic branch.

To avoid blocking the semantic branch behind the strong long-running 178
trajectory, 0803_17 now has a zero-state-delta 99 two-GPU port at isolated
commit `ac02fc2`. It will use whichever two indices are free under the 99
count cap after 0803_14 reaches the mature epoch-12 decision; no GPU index is
fixed outside server 252.

Experiment 0803_18 combines the positive terminal log-size/periodic-angle
projection from 0803_13 with the terminal centered semantic-margin consensus.
It remains parameter-free, class-permutation equivariant, and keeps center,
per-frame objectness mean, recurrent references, and DN independent. Its 197
two-GPU config and full model comparison passed at 22,771,111 parameters and
711 state tensors. It is prepared after 0803_11 epoch 12 and currently uses no
GPU. Server 252 remains reserved for the fixed GPU0/1 mature confirmation only.

Experiment 0803_19 is a parameter-free terminal full-tangent geometry
successor. At only the final normal-query output it averages center correction
in each frame's reference-local coordinates, size correction in log-ratio
coordinates, and orientation in the pi-periodic tangent space. Earlier
recurrent references, classification, and DN remain independent. The isolated
178 checkout at `dc0e958` passed the targeted final-layer call-count test, both
launcher syntax checks, and a full zero-state-delta model comparison at
22,771,111 parameters and 711 tensors. It is PREPARED and consumes no GPU.

Experiment 0803_20 combines the 0803_19 full-tangent terminal geometry with
the 0803_17 centered semantic-margin projection. It is the zero-parameter
center-aware successor to 0803_18: each frame still preserves its class mean,
all recurrent references, and DN semantics. The isolated 197 checkout at
`f179249` passed both launcher syntax checks and the full parent/new model
comparison at 22,771,111 parameters, zero state delta, and 711 tensors. It is
PREPARED behind 0803_18 and consumes no GPU.

Experiment 0803_11 completed epoch 12 at `45.409/50.665` cls/det HOTA,
below the original decoder at the same checkpoint by `1.986/3.771` and below
Encoder by `4.271/5.876`. Its epoch-4/8/12 trajectory, AP diagnostics,
381,030,375-byte checkpoint, 50 sequences, 28 CSV files, 108 evaluation files,
and async completion were verified before exact PGID `53708` was terminated
(`23 -> 0`). This mature three-node stop released 197 GPU4/5; it was not an
epoch-4/8 rejection.

Experiment 0803_18 then passed a real four-iteration two-GPU smoke on 197.
The first free-GPU wrapper had exited before workdir creation because of a CSV
parser bug; after switching to per-GPU queries, two consecutive checks proved
GPU4/5 free and the corrected smoke produced finite loss, gradient, DN,
iterative-classification, and checkpoint values. The clean isolated checkout
at `ac02fc2` started fresh formal PGID `387859`; epoch-1 iter50 was
`1.7440 s/iter`, loss `21.3900`, and gradient norm `107.1625`, with seven
processes, about 19.2 GiB on each selected GPU, no fatal errors, and all five
formal gates satisfied. It is RUNNING to e4/e8/e12 and later evidence.

Experiment 0803_13 completed epoch 16 at `50.415/57.456`, still a dual
`+0.379/+0.523` gain over the original decoder at the same checkpoint. Pair
mAP/AP50 are `0.275158/0.486134` and both-independent mAP/AP50 are
`0.320648/0.537430`; the checkpoint, 50 sequences, 28 CSV files, 108 evaluation
files, and async completion are verified. The gain is smaller than at epochs
8 and 12 but remains dual-positive while absolute HOTA continues to rise, so
the one-GPU 178 lane remains assigned through epoch 20 rather than being
preempted by a prepared successor.

Experiment 0803_14 completed epoch 12 at `46.987/52.992`, below the original
decoder at the same checkpoint by `0.408/1.444` and below Encoder by
`2.693/3.549`. Its complete epoch-4/8/12 trajectory, AP diagnostics,
381,037,430-byte checkpoint, 50 sequences, 28 CSV files, 108 evaluation files,
and async completion were verified before exact PGID `1327092` was terminated
(`23 -> 0`). GPU1/2 returned to 10 MiB while the external GPU0 process was
unchanged. This was a mature three-node stop, not an epoch-4/8 rejection.

Experiment 0803_17 then passed two consecutive free checks and a real four-
iteration two-GPU smoke on 99 GPU1/2. All four total losses and gradient norms,
DN semantics, iterative-classification semantics, and 642 floating checkpoint
tensors were finite. The clean isolated checkout at `ac02fc2` started fresh
formal PGID `1357909`; epoch-1 iter50 was `0.9994 s/iter`, loss `21.3978`, and
gradient norm `110.7768`, with seven processes, about 19.2 GiB per selected
GPU, no fatal errors, and all five formal gates satisfied. It is RUNNING to
e4/e8/e12 and later evidence without binding GPU indices as a general 99 rule.

Global experiment ID 0803_21 is reserved for terminal transported semantic
margins. Instead of deleting all frame-specific terminal class-margin detail,
it projects that detail onto the detached centered difference of the running
pair logits accumulated by earlier decoder layers. This preserves each frame's
residual class mean and the pair residual mean, permits only an already
established class-ranking trajectory, and suppresses a new transverse terminal
class switch. It is parameter-free, class-permutation and frame-swap
equivariant, leaves DN absolute logits unchanged, and adds only reductions and
dot products. The 99 formal/smoke configs, zero-state build audit, and safe
launchers passed local syntax checks. The isolated 99 checkout at `a7b37ef`
then passed three targeted unittest cases, config load/deepcopy, both remote
launcher syntax checks, and the full parent/new comparison at 22,771,111
parameters, zero state delta, and 711 tensors. The first test command found
that the existing py310 environment has no pytest; no package was installed,
and the same test methods were run through the standard unittest loader. It is
PREPARED with no smoke/formal workdir or GPU use until 0803_17 reaches a
mature resource decision.

Global experiment ID 0803_22 is reserved for terminal geometry plus
transported semantic margins. It keeps the 0803_18 terminal log-size and
pi-periodic angle projection but replaces complete centered-margin averaging
with the 0803_21 transport projection. This is a strict semantic-mechanism
counterpart to 0803_18, preserving center, per-frame residual class means,
recurrent references, and DN while adding no parameters, class-aware routing,
reweighting, attention, layer, or loss. The 197 formal/smoke configs,
zero-state build audit, and safe launchers pass local Python and Bash syntax
checks. The isolated 197 checkout at `41c08d8` also passed the three targeted
transport-margin tests, config load/deepcopy, remote launcher syntax, and the
full parent/new model comparison at 22,771,111 parameters, zero state delta,
and 711 tensors. It is PREPARED with no smoke/formal workdir or GPU use and
will not preempt the active 0803_18 trajectory.

Global experiment ID 0803_23 is reserved for transported full-tangent
geometry on the one-GPU 178 lane. It represents each terminal box update as
reference-local center displacement, log-size change, and the shortest
pi-periodic angle change. Pair-common refinement is preserved, while
frame-specific detail is projected only onto the detached relative transform
already accumulated by earlier decoder references. This keeps established
translation, scale, and rotation motion but suppresses a new transverse final
layer discrepancy. The operation is parameter-free, class-agnostic and
frame-swap equivariant, leaves DN untouched, and adds no reweighting,
attention, layer, or loss. The isolated checkout at clean `d6af6d32` passed
two targeted tests, config/build validation, launcher syntax, and the full
parent/new comparison at 22,771,111 parameters, zero state delta, and 711
tensors. It remains PREPARED/NO_GPU until 0803_13 reaches the epoch-20 mature
decision; server 252 remains reserved for candidates already supported by
fast-lane evidence.

Experiment 0803_13 reached complete epoch-20 cls/det HOTA
`51.791/58.526`, a dual `+0.948/+0.493` improvement over the original decoder
at the same checkpoint and a combined parent-line gain of `+1.441`. It is
`+0.277/-0.396` versus Encoder at the same epoch. Pair mAP/AP50
`0.288615/0.506941` and both-independent `0.333302/0.555375` all improved over
epoch 16, with the 392,138,804-byte checkpoint, 5,416 detections, 50
sequences, 28 CSV files, 108 non-empty evaluation files, and async completion
verified. Because all complete epoch-8/12/16/20 nodes remain dual-positive
against the strong original decoder and the advantage expanded again at
epoch 20, the 178 lane continues this trajectory through epoch 24. Prepared
successors remain GPU-free and server 252 stays reserved for mature evidence.

Experiment 0803_17 completed epoch 4 at cls/det HOTA `32.203/37.822`, with
cls DetA/AssA `26.308/42.066` and det `32.233/45.135`. This is
`-2.103/-0.768` versus the original decoder and `-4.006/-0.931` versus Encoder
at the same epoch. Pair mAP/AP50 is `0.140801/0.262022` and both-independent
is `0.183663/0.330727`; the 369,970,486-byte checkpoint, 5,416 detections, 50
sequences, 28 CSV files, 108 non-empty evaluation files, and async completion
are verified. The first monitor expected an unpadded `epoch_3` directory while
the authoritative two-GPU output is `epoch_03`; this was a monitor-path issue,
not a training failure, and later monitors use the actual padded path. The run
continues through epoch 8 and 12 because epoch 4 is not a decoder rejection
point.

A strict 252 continuation port is now prepared for the mature 0803_13 branch.
It changes only physical placement from 178 1x8 to fixed 252 GPU0/1 at 2x4;
the effective model, optimizer, schedulers, train loop, hooks, and global batch
8 are equal. An initial protocol check caught disabled switches represented as
extra explicit `False` keys in the 252 parent; these were removed before any
GPU launch, after which the model dicts matched exactly. The clean isolated
checkout at `bec1a1c` builds at 22,771,111 parameters, zero state delta, and
711 tensors. A real four-iteration DDP smoke on fixed GPU0/1 produced finite
losses `12.9372/19.5473/19.6326/21.1872`, gradients
`106.2705/100.9766/89.2521/88.4672`, finite DN and encoder terms, and a
364,501,750-byte checkpoint with all 642 floating tensors finite. Both GPUs
returned to 1 MiB after smoke. Formal migration remains conditional on a
dual-positive complete epoch-24 result: stop the 178 PGID first, then resume
the single shared workdir from `epoch_24.pth` on 252, never concurrently.

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

## 2026-07-30 0730_11 Shared-Routing Decoder

`0730_10` 的全对称 decoder 在 epoch 4 提高 det HOTA/DetA 和 AP50，但 pair mAP 相对
`0727_01` 下降 `0.00765`，说明完全共享两帧 cross-attention 和显式序平均会损失方向相关
表征。`0730_11` 因此只共享每层 deformable cross-attention 的几何路由模块
`sampling_offsets` 与 `attention_weights`，同时保留两帧独立的 `value_proj` 和
`output_proj`。该改动约束两帧采用一致的采样策略，但不强迫其特征值与输出表示相同；不改
encoder、proposal、PairDN、head、loss、初始化、数据或训练协议。

代码提交 `9049422` 已精确同步至四台服务器。56 项 decoder 单测、配置深拷贝、launcher
审计以及 252 GPU 0/1 的真实数据 4-iter DDP smoke 通过；smoke 中总、DN、encoder loss
和 grad norm 均有限，checkpoint 的 shared routing 误差为 `0`，独立 projection 已分化
`7.7571e-4`。正式 fresh 训练于 20:49 CST 在 252 GPU 0/1 启动，iter 50 与 iter 100
均通过五项启动门槛，20:52 已到 epoch 1 iter 150。首个性能判断点为 epoch 4，使用与
`0730_09` 相同的 HOTA、DetA、pair mAP 和 both-independent AP 保护门槛。

当前并行分配：

- 197 GPU 4/5：继续 `0730_09 motion-trust decoder` 到 epoch 8；
- 252 GPU 0/1：运行 `0730_11 shared-routing decoder`，首判 epoch 4；
- 178 GPU 0：保留给下一项独立 decoder 结构，不做 loss scale 或类别 reweight 扫描；
- 99：用户已恢复双卡额度，但实时只有 GPU 1 空闲，GPU 0/2 被其他用户任务占用；在形成
  两张实际空闲卡以前只承担代码、测试和轻量 smoke，不启动双卡正式训练。

## 2026-07-30 0730_12 Motion-Trust + Shared-Evidence Decoder

`0730_06 shared-evidence` 在 epoch 8 的 cls/det HOTA 相对父配置仅为
`-0.051/+0.032`，但 pair mAP/AP50 提高 `+0.0036/+0.0196`、
both-independent AP50 提高 `+0.0179`。其失败表现为 DetA/AssA 搬运，而不是 AP
证据不足。`0730_09 motion-trust` 则在 epoch 4 同时改善 HOTA、DetA、AssA 和 AP，
直接约束框几何。`0730_12` 因此组合两种互补结构：shared-evidence 只修正共享 decoder
query，motion-trust 只对两帧框分支施加检测置信度门控、位移包络有界的反对称修正。
这不是参数或 loss 扫描，且不改变 encoder、proposal、PairDN、head、loss、初始化、数据
或训练协议。

代码提交 `c43635c`，检查器兼容修复提交 `f1ffbd7`。57 项 decoder 单测、配置深拷贝、
路径/GMC/预训练权重及 launcher 审计通过。178 GPU 0 的单卡 physical batch 8 真数据
4-iter smoke 四次总、DN、encoder loss 和 grad norm 均有限，并产出 `iter_4.pth`；
三层 motion-trust 与三层 shared-evidence adapter 均产生有限、非零更新。正式 fresh
训练于 21:10 CST 启动，iter 50 和 iter 100 已通过五项启动门槛；首个性能门控点为
epoch 4，沿用 `0730_09/0730_11` 的同 epoch HOTA、DetA、pair mAP 和
both-independent AP50 保护线。

当前三路并行分配：

- 197 GPU 4/5：`0730_09 motion-trust decoder`，继续到 epoch 8；
- 252 GPU 0/1：`0730_11 shared-routing decoder`，首判 epoch 4；
- 178 GPU 0：`0730_12 motion-trust + shared-evidence decoder`，首判 epoch 4；
- 99：实时仅 GPU 1 空闲，GPU 0/2 属于其他用户的运行任务；在两张授权卡实际空闲且存在
  新的结构假设前不启动正式实验，不用参数扫描填卡。

## 2026-07-30 0730_13 Frame-Localized Shared-Attention Decoder

`0730_11 shared-routing` 在 epoch 4 提高 det HOTA 和 DetA，但 pair mAP 相对
`0727_01` 同点下降 `0.007499`。这说明同时共享 `sampling_offsets` 和
`attention_weights` 仍过度约束两帧的定位路径。`0730_13` 进一步拆分两种作用：只共享
`attention_weights`，令两帧学习一致的关注权重；`sampling_offsets`、
`value_proj` 和 `output_proj` 保持独立，使每帧仍可针对自身目标位置采样并保留表征容量。
该实验不改 encoder、proposal、PairDN、head、loss、初始化、数据或训练协议，不属于参数
扫描或重权。

代码提交 `091af97`。61 项 decoder 单测、正式/短测配置深拷贝、launcher shell 审计、
预训练权重结构检查和 252 GPU 0/1 的真实数据 4-iter DDP smoke 均通过。smoke 四次总
loss、DN loss、encoder loss 与 grad norm 均有限；checkpoint 的 6 组共享 attention
误差为零，18 组应独立参数的最大差异为 `7.8475e-4`。正式 fresh 训练于 22:32 CST 启动，
22:35 到 epoch 1 iter 100，双卡约 `19.2 GiB`、利用率 100%，无 Traceback、OOM、
NaN、NCCL、DDP reduction 或 unused-parameter 错误。首个性能判断点为 epoch 4。

当前资源分配：

- 197 GPU 4/5：`0730_09 motion-trust decoder`，epoch 8 完整门控待汇总；
- 178 GPU 0：`0730_12 motion-trust + shared-evidence decoder`，epoch 4 完整门控待汇总；
- 252 GPU 0/1：`0730_13 shared-attention decoder`，首判 epoch 4；
- 99：22:31 实时核验三卡均空闲，预留两卡给不重复、具有独立诊断价值并通过真数据 smoke
  的下一 decoder 结构；不复制已有实验，也不用参数扫描填卡。

## 2026-07-30 0730_14 Motion-Trust + Shared-Attention Decoder

为形成可解释的主效应与交互项，99 双卡运行 `0730_14`：组合 197 的 motion-trust 与
252 的 shared-attention。motion-trust 只产生检测置信度门控、位移包络有界的反对称框
修正；shared-attention 只绑定两帧的 attention-weight predictor，保持
`sampling_offsets/value_proj/output_proj` 独立。两者分别约束几何更新与特征聚合，
不改 encoder、proposal、PairDN、head、loss、初始化、数据或训练协议。

代码提交 `b829704`。62 项 decoder 单测、配置深拷贝、launcher shell 审计和 99 GPU 0/1
真实数据 4-iter DDP smoke 通过。smoke 总、DN、encoder loss 和 grad norm 均有限；
三层 motion-trust adapter 均非零，6 组共享 attention 参数误差为零，18 组独立参数已
分化。正式 fresh 训练于 22:45 CST 启动，22:46 到 epoch 1 iter 50，约
`0.9575 s/iter`、loss `21.4104`、grad norm `100.1361`，双卡各约 `19.2 GiB`，
无异常。首判 epoch 4，并与 `0730_09`、`0730_13` 的同点主效应共同解释。

## 2026-07-30 0730_09 Epoch-8 Gate 与 0730_15

`0730_09 motion-trust` 在 epoch 8 的 cls/det HOTA 为 `45.498/51.160`，相对父配置
仅提高 `0.229/0.967`；cls/det DetA 却下降 `0.875/2.043`，pair mAP 从
`0.237734` 降至 `0.230058`。其 AssA 提高但检测覆盖和 AP 保护失败，因此完成 checkpoint、
检测、2/2 TrackEval 与结构审计后停止，不继续长跑。

`0730_12 motion-trust + shared-evidence` epoch 4 则全面通过：cls HOTA/DetA/AssA
`40.559/29.273/59.330`，det `46.017/34.057/64.397`，pair mAP/AP50
`0.187248/0.342315`，both-independent mAP/AP50 `0.213887/0.366343`。
这说明 shared-evidence 与 motion-trust 的交互在早期同时改善检测与关联，继续到 epoch 8。

为分离这一交互是否必须依赖 motion-trust，197 接替运行 `0730_15`：
shared-evidence 修正共享 query，shared-attention 仅共享 attention 权重并保持逐帧
sampling offsets/value/output projection 独立。代码提交 `0782826`；63 项单测、配置/
launcher 审计和双卡 4-iter 真数据 smoke 通过。正式 fresh 训练于 22:58 CST 启动，
22:59 到 epoch 1 iter 50，约 `1.0668 s/iter`，关键 loss 与梯度有限，首判 epoch 4。

## 2026-07-30 0730_12 Epoch-8 Gate 与 0730_16

`0730_12 motion-trust + shared-evidence` 的 epoch 8 cls/det HOTA 为
`44.387/49.824`，相对父配置同点 `-0.882/-0.369`；cls/det DetA 分别下降
`4.401/6.353`，pair mAP 下降 `0.02333`，both-independent AP50 下降
`0.05615`。其 AssA 虽提高 `4.584/8.340`，但属于明显的检测覆盖向关联搬运，
epoch 4 优势没有保持。完整保存 checkpoint、检测、2/2 TrackEval 与结构审计结果后
停止，不再扩大该组合。

接替实验 `0730_16 antisymmetric frame-detail decoder` 转向 detection-preserving
结构：保留 `0727_01` 的 shared recurrent query 和层间 decoder 状态，使用本层两帧
cross-attention 输出的归一化、detach 有符号证据，经共享零初始化线性层与 `tanh`
得到有界修正，仅在 cls/reg head 前对两帧特征施加 `-detail/+detail`。该设计使两帧
head 特征中点严格等于父模型共享输出，帧交换时修正严格变号，避免 shared-evidence
直接扰动 recurrent query，也避免 motion-trust 直接扰动框。69 项 decoder 单测、
配置深拷贝、launcher 审计和代码差异检查已通过。178 GPU 0 的单卡 physical batch 8
真实数据 4-iter smoke 中总、DN、encoder loss 和 grad norm 均有限，checkpoint 三层
adapter 均产生约 `4.0e-4` 非零更新并通过结构检查。正式 fresh 训练于 23:48 启动，
23:49 到 epoch 1 iter 50，约 `0.9458 s/iter`、loss `21.1606`、grad norm
`107.4478`，GPU 0 约 `31.4 GiB`，无异常；首判 epoch 4。至此 252 `0730_13`、
99 `0730_14`、197 `0730_15`、178 `0730_16` 恢复四路结构实验并行。

## 2026-07-31 0730_13 Epoch-4 Gate

`0730_13 shared-attention decoder` 的 epoch 4 cls HOTA/DetA/AssA 为
`37.559/27.119/55.846`，det 为 `43.257/33.530/56.895`；相对父配置同点，
cls/det HOTA 提高 `1.350/4.504`，DetA 提高 `0.051/1.076`。pair mAP/AP50
为 `0.1547/0.3182`，both-independent mAP/AP50 为 `0.1839/0.3467`；
pair mAP 仅下降约 `0.00255`，未超过 `0.003` 保护线，both AP50 提高约
`0.0235`。checkpoint 中 6 组共享 attention 权重误差为零，18 组应独立参数最大
差异 `0.03846`。完整 TrackEval 与结构审计通过，因此继续到 epoch 8，检验
detection-preserving 的早期共同优势能否保持。

## 2026-07-31 0730_14 Epoch-4 Gate

`0730_14 motion-trust + shared-attention` 的 epoch 4 cls HOTA/DetA/AssA 为
`37.075/27.355/53.989`，det 为 `42.159/32.966/55.263`；相对父配置同点，
cls/det HOTA 提高 `0.866/3.406`，DetA 提高 `0.287/0.512`。pair mAP/AP50
为 `0.1625/0.3105`，both-independent mAP/AP50 为 `0.1887/0.3369`，
全部固定门槛通过并继续到 epoch 8。联合结构审计确认 motion adapter 非零、共享
attention 权重严格相等、独立参数已分化。但其 cls/det HOTA 同点低于 `0730_13`
`0.484/1.098`，说明目前 shared-attention 主效应强于与 motion-trust 的组合，
尚无正交互补证据。

## 2026-07-31 0730_15 Epoch-4 Gate

`0730_15 shared-evidence + shared-attention` 的 epoch 4 cls HOTA/DetA/AssA 为
`36.732/27.680/51.849`，det 为 `41.818/33.239/53.766`。相对父配置同点，
cls/det HOTA 提高 `0.523/3.065`，DetA 提高 `0.612/0.785`。pair mAP/AP50
为 `0.157715/0.320595`，both-independent mAP/AP50 为
`0.186071/0.345886`，全部固定门槛通过。

checkpoint 联合结构审计确认三层 shared-evidence adapter 均产生有限非零更新，
6 组 attention 权重严格共享，18 组应独立参数已经分化。实验继续到 epoch 8。
但它的 cls/det HOTA 同点低于 `0730_13 shared-attention` `0.827/1.439`，
也略低于 `0730_14`；因此当前结构结论是 shared-attention 提供主要增益，
shared-evidence 暂未形成正交互补，不据此扩展更多组合。

## 2026-07-31 0730_16 Epoch-4 Gate 与 0731_01

`0730_16 antisymmetric frame-detail decoder` 的 epoch 4 cls HOTA/DetA/AssA 为
`36.684/27.590/52.398`，det 为 `39.221/31.788/49.436`；pair mAP/AP50
`0.1700/0.3110`，both-independent mAP/AP50 `0.1992/0.3428`。相对父配置，
cls/det HOTA 与 AP 均提高，但 det DetA 下降 `0.666`，超过固定 `0.5` 上限
`0.166`，所以完成全量评估和结构审计后停止。

下一项 `0731_01` 组合当前最强早期主效应 `0730_13 shared-attention` 与
`0730_16` 的 head-only antisymmetric detail。前者共享双帧 deformable cross-attention
的定位权重但保留帧特异 value，后者保持 head 特征中点并在帧交换时严格变号；两者作用
位置正交，不涉及 loss、类别重权或 residual-scale 调参。配置提交 `5c556e4` 已精确同步
至 99、197、252、178。

178 GPU0 的 physical batch 8 真数据 4-iter smoke 生成 `iter_4.pth`，总、DN、
encoder loss 与 grad norm 全部有限；checkpoint 同时通过 shared-attention 和
antisymmetric-detail 结构检查。正式 fresh 训练于 01:11 启动，01:12 到 epoch 1
iter 50，约 `0.9344 s/iter`、loss `20.9701`、grad norm `96.9294`，GPU0 约
`31.4 GiB`，无异常。当前 252 `0730_13`、99 `0730_14`、197 `0730_15` 已进入
epoch 8，178 `0731_01` 首判 epoch 4，四路结构实验并行恢复。
## 2026-07-31 01:54 CST decoder 四机并行恢复

- `0730_13/14/15` 的 epoch 8 完整评估均证明 epoch 4 的早期关联增益不可持续，
  并在中期转化为 DetA 与 AP 损失；三项均在 checkpoint、检测、TrackEval 和结构
  审计完整后停止，不再用共享 attention、motion-trust 或 shared-evidence 组合长跑。
- 新一轮只验证模型结构，不做类别重权、loss scale 或 residual-scale 扫描：
  99 `0731_02` 是受真实帧差包络约束的 head-only 反对称细节；
  252 `0731_03` 是不改变 recurrent query 的公共证据旁路；
  197 `0731_04` 将两者正交分解并分别零起点受限注入；
  178 继续 `0731_01 shared-attention + antisymmetric detail`。
- 四台服务器代码统一至 `3d65dc4`。新结构通过 78 项 decoder 单元测试、配置深拷贝、
  完整模型构建、launcher 审计和真实数据 smoke；四项正式训练均已有真实训练进程与
  有限的总/DN/encoder loss，99、252、197 均达到 iter 50，178 已到 epoch 3。
- 首个统一决策点为 epoch 4；固定比较 cls/det HOTA、DetA/AssA、pair mAP/AP50 与
  both-independent AP50。仅在检测覆盖和 AP 不被搬运的前提下保留 HOTA 增益。

## 2026-07-31 02:24 CST 0731_01 Epoch-4 Gate

`0731_01 shared-attention + antisymmetric detail` 已完成 epoch 4 checkpoint、
检测、TrackEval 与两项结构审计。cls HOTA/DetA/AssA 为
`37.590/28.607/52.920`，det 为 `40.313/33.923/49.759`；相对 `0727_01`
同点，cls 三项为 `+1.381/+1.539/+0.826`，det 三项为
`+1.560/+1.469/+2.293`。pair mAP `0.173430`、both-independent AP50
`0.356102`，分别提高 `0.016177/0.032953`，因此所有固定保护门槛均通过。

checkpoint 中 6 组共享 attention 权重误差为零，18 组应独立参数最大差异
`0.042917`；三层 antisymmetric-detail 权重范数为
`0.024836/0.021828/0.023927`，排除结构未生效。该实验继续到 epoch 8，重点验证
早期增益能否避免 `0730_13` 的中期退化。99 `0731_02` 已到 epoch 3，252
`0731_03` 和 197 `0731_04` 均接近 epoch 3，四路结构实验继续并行。

## 2026-07-31 03:08 CST 0731_02 Epoch-4 Gate

`0731_02 enveloped-detail decoder` 已完成 epoch 4 checkpoint、检测、TrackEval 与
checkpoint 结构审计。cls HOTA/DetA/AssA 为 `37.859/28.112/54.688`，det 为
`42.873/34.173/55.071`；相对 `0727_01` 同点，cls/det HOTA 提高
`1.650/4.120`，DetA 提高 `1.044/1.719`。pair mAP/AP50 为
`0.167896/0.317776`，both-independent mAP/AP50 为 `0.196745/0.349436`；
pair mAP 与 both-independent AP50 分别提高 `0.010643/0.026287`。
三层 enveloped-detail 权重为 `0.066020/0.042205/0.063736`，均有限非零。
结构、HOTA、DetA、AssA 与 AP 门槛同时通过，实验继续到 epoch 8；252 `0731_03`
和 197 `0731_04` 仍按同一固定门槛等待 epoch 4 完整评估。

## 2026-07-31 03:12 CST 0731_04 Epoch-4 Gate

`0731_04 orthogonal-evidence decoder` 的 epoch 4 cls HOTA/DetA/AssA 为
`36.831/27.861/52.246`，det 为 `43.581/34.573/56.147`；相对父配置同点，
cls/det HOTA 提高 `0.622/4.828`，DetA 提高 `0.793/2.119`。
pair mAP/AP50 为 `0.161207/0.315535`，both-independent mAP/AP50 为
`0.189718/0.346363`，pair mAP 与 both-independent AP50 分别提高
`0.003954/0.023214`。两类三层结构权重均有限非零，完整门槛通过并继续到
epoch 8。现阶段该组合 det 侧增益最高，但 cls 增益较弱；需由 epoch 8 判断
公共证据分量是否会重现中期检测覆盖退化。

## 2026-07-31 03:16 CST 0731_03 Gate 与 0731_05 设计

`0731_03 common-evidence-bypass` 的 epoch 4 cls HOTA/DetA/AssA 为
`36.564/27.324/52.415`，det 为 `43.279/34.694/55.415`。虽然 HOTA 与
DetA 均提高，pair mAP `0.153565` 相对父配置下降 `0.003688`，超过固定
`0.003` 保护线 `0.000688`；三层门控均有限非零，因此判定为结构引起的轻微
pair AP 搬运并停止，不继续到 epoch 8。

252 的接替方向为 `0731_05 shared-attention + enveloped-detail`：共享两帧
deformable cross-attention 的 attention-weight predictor，保留各帧独立的
sampling offsets、value/output projection；再由真实两帧 cross-attention 差异
逐元素包络 swap-odd head correction。该组合不引入 common-evidence bypass，
不改变 recurrent query、proposal、PairDN、loss 或训练协议，目标是同时继承
`0730_13` 的关联增益与 `0731_02` 更强的检测保持能力。

## 2026-07-31 03:25 CST 0731_05 Formal Start

组合兼容性代码与零起点组合测试已完成，79 项 decoder 单测通过；正式/烟测配置
均完成深拷贝与完整模型构建。252 双卡真实数据 4-iter smoke 的总、DN、encoder
loss 和 grad norm 全部有限，checkpoint 同时证明 6 组 attention 权重严格共享、
18 组 sampling/value/output 参数已分化，三层 enveloped-detail 门控获得非零更新。

正式 fresh 训练于 03:23 启动，03:25 到 epoch 1 iter 50，约
`1.1589 s/iter`，loss `21.4509`、grad norm `104.0930`；GPU0/1 各约
`19.2 GiB` 且满载，无训练异常。首个决策点仍为 epoch 4，并使用与其余候选
相同的 HOTA、DetA/AssA、pair mAP 与 both-independent AP50 保护门槛。

## 2026-07-31 03:40 CST 0731_01 Epoch-8 Gate 与 0731_06

`0731_01` 的 epoch 8 完整结果为：cls HOTA/DetA/AssA
`45.152/37.611/57.666`，det `50.817/46.745/57.206`，pair mAP/AP50
`0.246292/0.441537`，both-independent mAP/AP50 `0.284940/0.479887`。
相对 `0727_01` 同点，det HOTA、pair mAP 和 both AP50 分别提高
`0.624/0.008558/0.013936`，但 cls HOTA 下降 `0.117`。两项结构审计确认
共享 attention 与逐帧 detail 都已学习，故按“双 HOTA 不低于父配置”的固定门槛
停止，而不是把 AP/AssA 增益误判为目标完成。

178 的接替实验预留为 `0731_06 shared-attention + regression-only
enveloped-detail`。分类 head 继续读取共享 decoder hidden states；零起点、
swap-odd 且受真实 cross-attention 帧差逐元素约束的 correction 仅作用于两帧
regression heads 和 reference 更新。该结构从计算路径上解耦分类与几何细节，
直接针对 `0731_01` 的唯一失败项，不使用参数 scale、loss 重权或类别重权。

## 2026-07-31 03:47 CST 0731_06 Formal Start

`0731_06` 已通过 80 项 decoder 单测、配置深拷贝、完整模型构建和 178 单卡
真数据 4-iter smoke。smoke 的总/DN/encoder loss 与 grad norm 均有限；
checkpoint 证明 6 组 attention 权重严格共享、18 组 sampling/value/output
参数保持逐帧独立，三层 regression-only detail 门控均产生非零更新。

正式 fresh 训练于 03:45 启动，03:46 到 epoch 1 iter 50：
`0.9610 s/iter`、loss `20.9556`、grad norm `111.7257`，GPU 0 约
`31.4 GiB`，无异常。首判 epoch 4；固定门槛不变，仍要求 cls/det HOTA
不低于父配置、任一 DetA 下降不超过 `0.5`、pair mAP 与 both-independent AP50
下降不超过 `0.003`。

## 2026-07-31 04:35 CST 分类/回归作用路径正交拆分

- `0731_02` epoch 8 因 cls HOTA `-0.141`、cls DetA `-0.546`、det DetA
  `-2.398` 淘汰；`0731_04` 因 cls/det HOTA `-0.509/-0.461`、cls/det DetA
  `-1.335/-2.878`、pair mAP `-0.011275` 淘汰。两者都显示全路径 frame detail
  容易把检测覆盖搬运到 AssA，不能因 AssA 上升而继续。
- 下一轮不做超参数扫描，而做严格结构拆分：
  - 99 `0731_07`：原 attention 结构 + classification-only bounded detail；
  - 197 `0731_08`：shared-attention + classification-only bounded detail；
  - 252 `0731_05`：shared-attention + full-path bounded detail；
  - 178 `0731_06`：shared-attention + regression-only bounded detail。
- 新分类专用路径保证 regression heads 与 iterative references 不直接接收 frame
  correction，只向两个分类头暴露零起点、交换变号且不超过原始帧差的细节。82 项单测、
  两项完整模型构建和两组真数据 DDP smoke 已通过。
- 99/197 正式实验基于提交 `7dee533`，分别在 GPU 0/1 与 GPU 4/5 fresh 启动并越过
  iter 50；252/178 原实验不重启继续。四路统一在 epoch 4/8 使用既定 HOTA、DetA、
  pair mAP 和 both-independent AP50 门槛，不因单独 AssA 上升保留。

## 2026-07-31 06:02 CST 门槛修订与资源调度

- 主要目标明确为 decoder 最终 cls HOTA 和 det HOTA 同时超过 encoder
  `0727_01` 的 `54.437/62.393`；任一项未超过都不改写论文主线。
- 阶段决策优先级调整为：同 epoch cls/det HOTA > DetA/AssA 归因 >
  pair mAP 与 both-independent AP50 诊断。AP 的轻微单点波动不再作为硬停止条件；
  明显、持续且与 HOTA/DetA 同向的 AP 下降仍视为检测崩塌信号。
- 因此 197 `0731_08` 从 epoch 4 原位恢复到 epoch 8；99 的 `0731_07`
  因 cls HOTA/DetA 与 pair mAP 同时明显下降保持淘汰。
- 当前四路并行：252 `0731_05 full-path`、178 `0731_06 regression-only`、
  197 `0731_08 shared-attention + classification-only`、99 `0731_09`
  regression-only 双卡复现。优先在 epoch 8 判断哪种作用路径能同时保护 cls 与 det。
- 下一结构候选为 midpoint-preserving regression-only detail：新增帧细节在 5D
  box-logit residual 空间严格反对称且 pair midpoint 为零，分类路径保持共享。
  已通过零起点等价、梯度与中点守恒测试，等待下一空闲资源。

## 2026-07-31 06:08 CST 0731_06 决策

- `0731_06` epoch 8 的 cls/det HOTA 为 `44.398/48.552`，相对父配置同点
  `45.269/50.193` 双双下降；即使放松 AP 门槛也必须淘汰。
- 178 GPU 0 改跑 `0731_11 midpoint-preserving regression-only detail`。
  该实验不是参数扫描：它把帧细节的反对称约束从特征域推进到最终 5D 框残差域，
  精确消除两个独立回归头带来的 pair midpoint 漂移。
- `0731_11` 已通过构建、测试、真实数据 smoke、结构 checkpoint 检查与正式
  iter 50 门槛，06:11 fresh 启动；首个决策点为 epoch 4 的双 HOTA。

## 2026-07-31 06:17 CST 0731_05 epoch-8 决策

- `0731_05` epoch 8 的 cls/det HOTA 为 `45.341/51.589`，相对父配置同点
  `45.269/50.193` 提高 `0.072/1.396`，按 HOTA 主门槛继续。
- 该候选目前是阶段性最强 decoder，但 cls 裕量极小；det DetA 下降 `1.885`、
  AssA 提高 `5.672`。因此后续仍以双 HOTA 决策，同时单独记录“关联增益与检测覆盖”
  的分解，避免把 AssA 搬运解释成全面检测增强。
- 最终成功条件不变：训练末期 cls/det HOTA 同时超过 `54.437/62.393`。

## 2026-07-31 06:35 CST terminal-only detail 预案

- `0731_02/04/06` 的共同失败模式是早期 HOTA 提升、到 epoch 8 后 DetA 下降并把
  增益搬运到 AssA；`0731_05` 虽在 epoch 8 双 HOTA 通过，det DetA 仍低父配置
  `1.885`。证据指向 frame detail 经逐层 reference 更新递归进入后续 decoder，
  而不是 detail 本身完全无效。
- 新的 `terminal_enveloped_detail_decoder` 仅在最后一层输出前加入受观测帧差包络的
  `-detail/+detail`。此前所有 decoder 层、auxiliary outputs 和 iterative references
  与 shared-attention 父路径逐元素一致，因而同时保留父模型的分类/定位迭代和最终帧区分。
- 提交 `764ff7d` 已通过 86 项 decoder 单测、完整配置构建、模型初始化零门控检查并同步
  四台服务器。它当前只是经过代码验证的后备结构，不占实验编号和 GPU；197 释放时仍优先
  做已准备的 `0731_10` midpoint 双卡复现，terminal-only 只在后续证据支持时进入 smoke。

## 2026-07-31 07:25 CST classification-only 决策与 midpoint 并行验证

- `0731_08` epoch 8 的 cls/det HOTA `43.801/49.318` 双双低于父配置；cls/det
  DetA 分别下降 `2.227/3.269`，pair mAP 与 both-independent AP50 也明显下降。
  完整结构检查证明分类门控已学习，因此停止 classification-only 路径。
- 197 GPU 4/5 接替为 `0731_10 midpoint-regression` 2xb4 复现。配置、完整模型、
  双卡真数据 smoke、checkpoint 结构审计和正式 iter 100 五项门槛均通过。
- 99 `0731_09 regression-only` epoch 4 的 cls/det HOTA 为 `37.813/44.030`；
  178 `0731_11 midpoint-regression` 为 `38.668/43.586`。两者相对父配置双 HOTA、
  双 DetA 与 AP 均提高，继续到 epoch 8。
- 下一轮决策顺序：252 `0731_05` epoch 12 → 99/178 epoch 8 → 197 epoch 4。
  若 full-path 或 midpoint 在中期出现 DetA/双 HOTA 系统性退化，再启用仅最终层
  注入细节的 terminal-only 结构；不回到 scale 或权重扫描。

## 2026-07-31 07:45 CST 0731_05 epoch-12 决策

- epoch 12 的 cls/det HOTA 为 `50.171/56.430`；相对 `0727_01` 同点
  `49.680/56.541` 为 `+0.491/-0.111`，不记为双提升。
- det DetA/AssA 相对父配置为 `-0.805/+0.866`，但 DetA 差距较 epoch 8
  的 `-1.885` 明显收窄，检测 AP 也没有崩塌。因此允许该最成熟候选继续到
  epoch 16，不因小幅单点落后过早截断轨迹。
- epoch 16 仍使用同点双 HOTA 作主门槛；失败则在完整产物落盘后停止 0731_05，
  释放 252 部署 terminal-only。其他三台不受影响，继续完成 99/178 epoch 8
  和 197 epoch 4 的结构判定。

## 2026-07-31 07:58 CST 0731_12 PREPARED

- terminal-only 后备实验正式编号预留为 `0731_12`，目标资源为 252 GPU0/1。
  正式/烟测配置和 launcher 已通过深拷贝、完整构建、`bash -n` 与 detector
  初始化零门控检查；目标 workdir 均保持不存在。
- 当前不运行 smoke、不排队、不占 GPU。触发条件保持为：现有 full-path/midpoint
  候选在完整同点 HOTA 下失败且对应资源释放。触发后仍须先通过真数据 4-iter DDP
  smoke、有限 loss/grad 与 checkpoint 结构验收，再允许正式 fresh 训练。

## 2026-07-31 23:17 CST 轻量 decoder 约束与 0731_29

- 后续 decoder 候选必须保持结构简洁和计算可控：不堆叠 decoder 层、额外 attention、
  高分辨率分支或辅助 loss；论文候选还需做同卡同温速度验证，原则上吞吐下降不超过 5%。
- `0731_26 confidence-common+detail` 的 epoch 12 cls/det HOTA 为
  `48.766/55.694`，相对 Encoder 同点 `49.680/56.541` 下降
  `0.914/0.847`；其 DetA 与检测 AP 同样偏低。结合 `0731_24/25`，三种
  confidence 放置均被连续证据否定，完整产物验证后停止，不再扫描 confidence 或 scale。
- 252 接替运行 `0731_29 terminal diagonal center-motion factorization`。它只增加
  512 个逐通道门控标量，并将末层反对称框修正限制到中心 `x/y`；`w/h/angle`、
  recurrent reference、auxiliary outputs、loss 和训练协议均保持父模型不变。
- `0731_29` 已通过配置深拷贝、完整构建、双卡真实 4-iter smoke、有限总/DN/encoder
  loss、checkpoint 门控更新和正式 iter 50 五项启动验收。首个结构判定点为 epoch 4；
  最终成功条件仍为 cls/det HOTA 同时超过 `54.437/62.393`。

## 2026-07-31 23:22 CST 0731_21 epoch-28 机制判定

- `0731_21` epoch 28 cls/det HOTA 为 `52.135/59.522`，相对 Encoder 同点
  `51.740/59.830` 为 `+0.395/-0.308`，仍不满足双提升。
- cls DetA/AssA 相对父轨迹为 `+0.497/+0.091`；det DetA/AssA 为
  `-0.924/+0.573`。pair 与 both-independent 的 mAP/AP50 分别提高
  `0.006896/0.010218/0.007304/0.011789`。因此结构确实改善分类和关联，
  但完整 5D frame-detail 仍将一部分 det 检测覆盖搬运到 AssA。
- 该偏离幅度较小且 AP 全部提升，不按单点评估过严停止；`0731_21` 继续到 epoch 32。
  `0731_28/29` 对 frame detail 的中心运动限制正是针对该 DetA 瓶颈，保持为优先轻量候选。

## 2026-07-31 23:33 CST 0731_27 epoch-4 Gate

- 逐通道 terminal factorization 的 epoch 4 cls/det HOTA 为 `36.753/42.551`，
  相对 Encoder 同点提高 `0.544/3.798`。cls DetA/AssA 分别变化
  `+0.033/+2.107`，det 为 `+1.199/+7.305`，不是只靠 AssA 搬运的单侧结果。
- pair/both-independent AP50 分别提高 `0.013886/0.014332`；mAP 轻微下降
  `0.004003/0.001532`，按 HOTA 主门槛只记为诊断项，不构成早停。
- 正式 epoch-4 checkpoint 中两个逐通道门控最大绝对值为 `0.050928/0.180767`，
  独立 attention 最大分化 `0.030386`，排除结构未学习。该 512 参数轻量候选继续到 e8；
  e4 的大幅 det 增益只作强早期信号，不外推为最终结论。

## 2026-07-31 23:42 CST 历史 RUNNING 状态审计

- 四机全量 screen/训练主进程核对确认，当前每台仅有资源总览中的一个正式实验；
  状态表中 `0727_12/0728_01/0727_04/0731_16` 的旧 RUNNING 标记属于记录滞后，
  不代表重复训练或资源冲突。
- `0727_12 cross-scale budget` 已完成72 epochs与18/18评估，最佳epoch 60为
  `54.217/61.875`，没有超过Encoder `54.437/62.393`。
- `0728_01 tri-state decoder` 与 `0727_04 detail-energy encoder` 分别完整评估到
  epoch 48/56，最佳为 `52.587/60.682` 与 `53.796/61.711`；二者随后收到外部
  SIGTERM，与统一停止调度一致，不是模型崩溃，均不resume。
- `0731_16 terminal common-only` 的e8为 `43.972/49.378`，相对父同点双降，
  已在 `0731_21` 启动前停止。上述四项状态已纠正为 COMPLETED/STOPPED，
  不改变当前四路轻量decoder实验或论文主线。

## 2026-08-01 00:03 CST 0731_28 epoch-4 Gate

- 中心运动限制版本 `0731_28` 的 epoch 4 cls/det HOTA 为 `35.714/42.426`，
  相对 Encoder 同点 `36.209/38.753` 为 `-0.495/+3.673`。它明显增强 det，
  但没有在该早期点保护 cls。
- cls DetA/AssA 为 `25.915/53.044`，相对父轨迹 `-1.153/+0.950`；det 为
  `32.743/56.325`，相对父轨迹 `+0.289/+8.859`。pair mAP/AP50 为
  `0.150254/0.299865`，both-independent 为 `0.177156/0.325512`；两项 mAP
  分别下降 `0.006999/0.007309`，而 AP50 分别小升 `0.003731/0.002363`。
- 这说明只保留中心 `x/y` detail 能产生强运动关联增益，但稠密 common/detail 门控仍可能
  牺牲分类检测覆盖。checkpoint 中独立 attention 与两类门控均已非零学习，排除结构未生效。
- 由于这是单个 e4 点且 det 提升显著，按放宽后的早期判定继续到 e8；若 cls HOTA、
  cls DetA 与 mAP 的系统性下降持续，则停止该稠密版本。同期 `0731_27` e4 为
  `36.753/42.551`，在参数仅 512 的情况下双 HOTA 均高于 `0731_28`，因此轻量对角版本
  仍是当前优先候选；`0731_29` 将直接判断“对角门控 + 中心运动”能否兼得两者。

## 2026-08-01 00:30 CST 0731_21 epoch-32 决策

- 稠密 terminal orthogonal factorization 的 e32 cls/det HOTA 为 `52.235/59.813`，
  相对 Encoder 同点 `52.354/60.330` 为 `-0.119/-0.517`。这与 e28 的
  `+0.395/-0.308` 合起来，构成连续两个中期节点 det 未提升的证据。
- cls DetA/AssA 相对父轨迹仅为 `-0.026/-0.113`，基本持平；det 为
  `-0.958/+0.122`，再次定位到检测覆盖损失。与此同时，pair mAP/AP50 提高
  `0.005249/0.009248`，both-independent 提高 `0.005985/0.011227`，因此不是
  整体预测崩塌，而是该稠密 5D detail 结构稳定存在的 HOTA/DetA 取舍。
- 在 e32 checkpoint、检测 metrics、TrackEval metrics 和 54 个原始评估文件全部验证后，
  已精确停止 `0731_21` 并释放 178 GPU0。它不 resume，也不进入论文 decoder 主线。
- 178 暂不立即堆叠新模块；先等待 252 `0731_29` e4 判断 512 参数对角门控与中心运动限制
  是否兼容，再选择有明确依据的下一项轻量结构或效率验证，避免为保持 GPU 满载而运行低信息实验。

## 2026-08-01 00:47 CST 轻量候选分流与 0731_21 复核

- `0731_27` e8 cls/det HOTA 为 `43.344/49.456`，相对 Encoder 同点
  `-1.925/-0.737`；cls/det DetA 分别下降 `2.641/3.760`，四项 AP 下降
  `0.024681/0.022977/0.027357/0.024470`。e4 的强增益未保持，已在完整产物
  验证后停止并释放 197 GPU4/5。
- `0731_29` e4 为 `36.687/42.411`，相对 Encoder `+0.478/+3.658`；
  cls/det DetA 提高 `0.043/1.619`，且只增加 512 参数，继续到 e8。鉴于纯对角版本
  的轨迹，e8 是决定中心运动限制能否防止中期退化的关键节点。
- 对 `0731_21` 的停止规则做了一次一致性复核：e32 只有 det HOTA/DetA 小幅低于父轨迹，
  Cls 基本持平且四项 AP 全升；它还在 e8 明确双提升。因此它不满足 HOTA/DetA/AP
  同向恶化的强停止条件，00:29 的停止偏严。
- `epoch_32.pth` 已确认包含完整 state_dict、optimizer、scheduler、EMA 与 message hub，
  `last_checkpoint` 精确指向该文件。准备在相同 178、相同 workdir 原位 exact resume 到 e40，
  不新建重复实验；e40 再按双 HOTA 与 DetA/AP 持续性决定。该末层结构虽比 512 参数版本大，
  但新增参数不足总模型 1%，没有新增 decoder 层或 attention，仍可进入后续同卡效率审计。

## 2026-08-01 01:10 CST 0731_21 恢复与 0801_01 共用门控

- `0731_21` 的 epoch-32 停止按放宽后的持续性规则复核为过严：Cls 基本持平、四项 AP
  全升，不满足 HOTA/DetA/AP 同向恶化的强停止条件。完整 checkpoint 含 optimizer、
  scheduler、EMA 与 message hub，已在相同 178、相同 workdir 原位 exact resume。
- 首次 resume 仅在 checkpoint 加载阶段触发 PyTorch 2.6 的 `weights_only=True` 兼容错误，
  未执行训练更新；launcher 对本项目自产可信 checkpoint 显式恢复旧加载语义后，于 00:52
  正确恢复 epoch 32/iter 33216。epoch 33 iter 100 后五项运行证据持续正常，继续到 e36/e40。
- `0731_27` checkpoint 机制审计显示其 detail 逐通道 gate 的平均/RMS 幅值约为 common gate
  的三倍，且两者相关性很低；两类 gate 从 e4 到 e8 均约翻倍，但双 HOTA、双 DetA 与
  四项 AP 在 e8 同向退化。下一候选因此不扫描 scale，而从结构上取消两条路线独立放大。
- `0801_01` 使用一个 common/detail 共用的 256 维逐通道 terminal gate；相对 Encoder
  只新增 256 参数，不增加 decoder 层、attention、分支、loss 或矩阵乘法。零起点等价、
  common 与 detail 同时生效、box midpoint 守恒和共享 gate 非零梯度均由单测覆盖。
- 105 项 decoder 单测、配置深拷贝、完整模型构建、197 双卡真实 4-iter smoke 与 checkpoint
  结构检查全部通过；正式 fresh 训练于 01:07 使用 GPU4/5 启动，iter 100 的总/DN/encoder
  loss 与 grad norm 均有限，五项启动验收通过。先看 e4 结构信号，e8/e12 判断持续性；若有
  潜力至少训练到 e16/e20。论文候选仍须补做同卡同温速度审计，吞吐下降原则上不超过 5%。

## 2026-08-01 01:15 CST 0731_28 epoch-8 决策

- 中心运动限制的稠密 terminal factorization 在 e8 得到 cls/det HOTA
  `45.675/51.723`，相对 Encoder 同点 `45.269/50.193` 提高 `+0.406/+1.530`。
  因此 e4 的 `-0.495/+3.673` 单侧现象已转为双 HOTA 提升，按主门槛继续。
- 归因仍有清晰取舍：cls DetA/AssA 相对父轨迹为 `-0.827/+1.814`，det 为
  `-2.140/+6.310`；pair mAP/AP50 变化 `-0.007990/+0.008348`，
  both-independent 为 `-0.008993/+0.006133`。即 HOTA 增益主要来自关联改善，
  检测覆盖尚未得到同等保护，但没有出现 HOTA、DetA 与 AP 全部同向恶化。
- e8 checkpoint、检测 metrics、TrackEval metrics、50 个序列 txt 与汇总 CSV 均完整；
  6 组独立 attention 最大分化 `0.084948`，common/detail gate 最大绝对值
  `0.039202/0.091669`，排除未学习。继续到 e12 判断 DetA/mAP 取舍是否收窄；
  若双 HOTA 仍高于父轨迹，则至少看到 e16/e20，不以单点 AP 下降过早终止。

## 2026-08-01 02:16 CST 0731_29 e8 收口与 0731_21 e36 复核

- `0731_29` e8 cls/det HOTA 相对 Encoder 同点为 `-1.121/-0.817`，两路 DetA 与
  pair/both 的 mAP、AP50 也全部下降。该组合从 e4 正增益转为 e8 系统性退化，完整产物
  验证后停止，不继续浪费到 e12；这否定了“对角门控 + 中心运动限制”能够自然兼得检测覆盖
  与关联增益的假设。
- `0731_21` e36 为 `52.699/60.048`，相对同点 `-0.213/-0.659`；DetA 小降但四项 AP
  全升，因此继续到 e40。若 e40 仍双 HOTA 下降，则结合 e28/e32/e36 连续轨迹收口，不能
  仅凭 AP 改善把它选为论文 decoder。
- resume 评估目录冲突已通过完整归档保护，并从代码根因上改为扫描已有 TrackEval 最大编号。
  当前进程不热替换代码，e40 仍按旧进程行为完成后立即归档；今后 resume 将自动使用新编号。
- 结构约束升级为硬门槛：不堆 decoder 层、attention、高分辨率分支或辅助 loss；候选必须
  简单可解释、参数增量极小，并在形成性能候选后通过同卡同温速度测试，吞吐下降原则上不超过
  5%。252 的下一项实验必须基于现有 HOTA/DetA 失效机理提出独立结构假设，不为占满 GPU
  进行低信息参数扫描。

## 2026-08-01 02:28 CST 0801_01 epoch-4 决策

- 仅 256 参数的 common/detail 共用逐通道 gate 在 e4 获得 cls/det HOTA
  `36.757/42.605`，相对 Encoder 同点 `+0.548/+3.852`；det DetA/AssA 同时提高
  `0.934/7.816`，初步支持“约束两路末层修正共同幅度”这一结构假设。
- e4 仍只视为结构信号，继续到 e8/e12 判断持续性。后续候选优先采用同等级别的单点、低参数、
  不改变分类主干或 decoder 深度的机制改动；形成候选后再做同卡同温测速，吞吐下降上限保持 5%。

## 2026-08-01 02:38 CST 0801_02 中心运动 detail-only

- `0731_28` e12 相对 Encoder 双 HOTA 转为 `-0.494/-0.293`，说明 e8 的 common+detail
  联合增益不稳定；但 AP50 和 det AssA 尚有提升，因此原实验继续到 e16 复核。
- 252 下一项只做一项机制剥离：分类完全回到 Encoder，仅保留末层中心 x/y 的反对称 box detail；
  宽高角、辅助输出和 recurrent references 不变。新增 65,536 参数（约 0.29%），无新层、attention
  或 loss。单测和完整构建通过后执行真实 4-iter smoke，通过五项门槛才启动 formal fresh。

## 2026-08-01 02:47 CST 0801_02 启动验收

- 配置父链的数据根路径问题在首轮 smoke 数据加载前暴露并修正，未产生训练更新；失败日志保留。
  修正后 4-iter 双卡真实 smoke、detail-only checkpoint 审计与 6 组独立 attention 检查通过。
- 正式 fresh 已在 252 GPU0/1 运行到 iter 50，总/DN/encoder loss 和 grad norm 有限，无训练异常。
  先看 e4 结构信号，不在早期小波动上过严停止；e8/e12 决定是否继续到 e16/e20。

## 2026-08-01 03:19 CST 0731_21 e40 决策

- e40 cls/det HOTA `53.655/60.379`，相对 Encoder 同点 `-0.142/-0.684`；
  e32、e36、e40 连续未形成双提升。虽然 pair 与 both-independent 的 mAP/AP50 均高于父轨迹，
  但 det DetA/AssA 仍分别低 `0.572/0.902`，说明完整 5D 稠密正交 detail 的 HOTA 取舍没有消失。
- 完整产物已归档并停止该分支，不再因 AP 改善延长训练。178 暂时空闲，等待 `0801_01` e8
  和 `0801_02` e4 后再决定下一项；下一候选仍须是单点轻量机制，不叠加层、attention、分支或 loss。

## 2026-08-01 03:47 CST 0731_28 e16 决策

- e16 cls/det HOTA 为 `48.845/57.176`，相对 Encoder 同点下降 `2.246/1.144`；
  cls/det DetA 下降 `2.853/2.780`，pair 与 both-independent 的 mAP/AP50 也全部下降
  `0.0145–0.0206`。e12/e16 已形成连续系统性退化，因此完成产物核验后停止 99。
- 该结果说明仅把 5D detail 限制到中心 `x/y` 不能修复稠密 common/detail 联合结构的检测覆盖损失。
  后续不再延伸该绑定结构，也不增加层、attention、分支、loss 或参数扫描；先用 `0801_01` e8
  判断 256 参数共用 gate 是否保持早期增益，用 `0801_02` e4 判断完全保留 Encoder 分类路径是否有效。

## 2026-08-01 04:14 CST 0801_02 e4 决策

- `0801_02` e4 cls/det HOTA `36.757/42.974`，相对 Encoder 同点提高 `0.548/4.221`；
  cls DetA 仅下降 `0.153`，det DetA 提高 `0.937`，pair/both AP50 也提高约
  `0.0122/0.0127`。完全保留 Encoder 分类路径并仅修正最终中心运动的最小剥离通过结构 gate。
- 继续到 e8/e12 检查双 HOTA、DetA 与 AP 的持续性，不因 e4 强信号提前宣称成功，也不在
  99/178 填充新的低信息实验。若持续成为候选，再进行同卡同温效率测试，吞吐下降不得超过 `5%`。

## 2026-08-01 04:36 CST 0801_01 e8 决策

- `0801_01` e8 cls/det HOTA 相对 Encoder 为 `-1.723/+0.295`，双 DetA 与四项 AP 均明显下降，
  det 的微弱增益由 AssA 提升维持。共享 256 参数门没有保持 e4 的双提升，当前倾向否定。
- 为避免恢复到过严的单点早停，继续到 e12 作最后持续性检查；若 cls HOTA、双 DetA 与 AP
  未出现实质恢复，则完整归档并停止 197。该机制不再派生新组合，也不在成为性能候选前测速。

## 2026-08-01 05:45 CST 0801_02 e8 收口与 0801_03 轻量化

- `0801_02` e8 cls/det HOTA `44.632/49.687`，相对 Encoder 同点下降
  `0.637/0.506`；双 DetA 分别下降 `2.081/3.688`，pair 与 both-independent 的 mAP/AP50
  也全部下降。e4 强信号没有持续，完整 e8 产物核验后已停止并释放 252 GPU0/1。
- 下一候选 `0801_03` 不增加模块，只把 `0801_02` 的 `256×256` 稠密 detail gate 改成
  256 维逐通道 gate。分类、recurrent references、辅助输出、宽高角与训练协议不变；新增参数
  从 65,536 降为 256，并移除矩阵乘法。该对照直接检验稠密通道混合是否导致中期覆盖退化。
- 107 项 decoder 单测和配置深拷贝已通过。提交同步后先做双卡真实 4-iter smoke；只有 checkpoint
  中唯一 256 维 gate 非零、6 组 attention 独立、各类 loss/grad 有限且无分布式异常时才启动正式训练。
- 后续继续执行复杂度硬约束：新增参数原则上不超过 1%，同卡同温吞吐下降不超过 5%；不堆叠
  decoder 深度、额外 attention、高分辨率分支或辅助 loss，不用参数扫描代替模型机制验证。

## 2026-08-01 05:58 CST 0801_03 启动验收

- 提交 `9a18a0c` 已同步到四台服务器。252 双卡真实 4-iter smoke 产生 `iter_4.pth`；总损失、
  DN、encoder proposal 与 grad norm 全部有限，唯一 detail gate 为非零 256 维向量，6 组
  prev/curr attention 保持独立，无 Traceback/OOM/NaN/NCCL。
- 正式 fresh 于 05:56 启动，iter 50 为 `1.1619 s/iter`、loss `21.5299`、grad norm
  `106.3036`，GPU0/1 各约 19.2 GiB，五项门槛通过。先按既定规则看 e4 结构信号，再以
  e8/e12 判断持续性；性能未形成候选前不追加复杂结构或低价值速度测试。

## 2026-08-01 06:46 CST 0801_01 e12 收口

- e12 cls HOTA/DetA/AssA `47.158/38.483/59.975`，det
  `55.516/47.987/66.488`；相对 Encoder 同点 HOTA `-2.522/-1.025`、DetA
  `-2.877/-2.358`，仅 det AssA 提高 `0.909`。e4 增益在 e8 消失，e12 没有恢复。
- pair mAP/AP50 `0.2468/0.4538`，both-independent `0.2834/0.4872`；可直接核对的
  父配置 pair mAP 与 both AP50 分别高 `0.02637/0.02455`。结果继续表现为检测覆盖下降与
  AssA 搬运，而不是可持续双 HOTA 改善。
- epoch 12 checkpoint、检测 metrics、TrackEval `async_done=1`、50 序列结果与 108 个评估
  文件完整后，于 06:46 精确终止 PGID `1703376`；screen 与训练 worker 已退出，GPU4/5
  均为 `0%/1 MiB`。该共享门分支不再 resume 或派生，197 暂时空闲，等待 `0801_03` e4/e8。

## 2026-08-01 07:26 CST 0801_03 e4 结构 Gate

- e4 cls HOTA/DetA/AssA `36.944/27.076/54.129`，det `42.493/33.391/55.143`；
  相对 Encoder 同点 HOTA `+0.735/+3.740`、DetA `+0.008/+0.937`、AssA
  `+2.035/+7.677`。逐通道 detail-only 没有牺牲早期检测覆盖。
- pair mAP/AP50 `0.1493/0.3078`，both-independent `0.1789/0.3365`；相对 Encoder
  mAP 略降 `0.0079/0.0056`，AP50 提高 `0.0117/0.0134`，不是 HOTA/DetA/AP 全面恶化。
- 相比稠密 `0801_02` e4，cls HOTA 高 `0.187`，det HOTA 低 `0.481`；det DetA 均为
  `33.391`，说明移除矩阵乘法没有损伤检测覆盖，但关联增益略弱。完整 checkpoint、检测、
  TrackEval、50 序列与 108 个评估文件齐全。
- e4 只通过结构 gate，不宣称最终胜出；继续 e8/e12 检查是否避免 `0801_02` 的中期 DetA/AP
  退化。当前不使用 99/178/197 派生新模型，保持单变量、低复杂度证据链。

## 2026-08-01 08:54 CST 0801_03 e8 决策

- e8 cls/det HOTA `44.183/50.011`，相对 Encoder 同点下降 `1.086/0.182`；双 DetA
  下降 `2.432/2.772`，pair 与 both-independent 的 mAP/AP50 也全部下降约
  `0.0247–0.0310`。e4 强信号没有持续，结果仍是检测覆盖向 AssA 搬运。
- 唯一 256 维 gate 已学到最大绝对值 `0.337337`，6 组独立 attention 最大差异
  `0.059180`；完整 e8 checkpoint、检测和 TrackEval 产物均已核验，因此不是模块未生效。
- 该结果达到双 HOTA、双 DetA 与四项 AP 系统性退化的停止条件，已精确停止并释放 252。
  不继续 e12，不 resume，也不从该路径派生稠密门、额外 attention 或参数扫描。下一步先基于
  已有全部 decoder 结果做机制收口，只保留结构简单、可解释且预计效率下降不超过 `5%` 的候选。
## 2026-08-01：低复杂度 decoder 约束与 0731_01 持久性验证

- 新候选优先满足：参数增量约 `<=1%`，同卡实测吞吐下降 `<=5%`；不通过增加 decoder 深度、额外注意力层、大分辨率支路或附加 loss 换取性能。
- 在发明新结构前，先把旧严格早停遗漏的 `0731_01 shared-attention + antisymmetric detail` 从 epoch 8 续到 epoch 12。它在 epoch 8 相对 Encoder 同点 cls HOTA `-0.117`、det HOTA `+0.624`，且 pair mAP 与 both-independent AP50 同步提高，具备继续验证的最高信息价值。
- epoch 12 仍以 cls/det HOTA 为主判据；DetA/AssA 用于解释机制，AP 仅作诊断。若不能形成双 HOTA 持续提升，则停止该方向，不做参数扫描或复杂化衍生。

## 2026-08-01：0731_05 epoch-20 补证

- 利用空闲 252 GPU0/1，从 `0731_05` 原 epoch 16 断点只续一个评估周期到 epoch 20。该轨迹参数增量仅 `0.539%`，e8 曾双超、e12/e16 仅窄幅未过且 AP 稳定，比重新训练或添加新模块具有更高信息/计算比。
- epoch 20 若未同时超过 Encoder 同点 cls/det HOTA，则停止并否定继续长跑；不派生 scale、loss 权重、类别重权或更深 decoder。

## 2026-08-01 10:23 CST：0731_01 epoch-12 持久性结论

- `0731_01 shared-attention + antisymmetric detail` 从原 epoch 8 断点续到 epoch 12 后，
  cls HOTA/DetA/AssA 为 `48.465/40.393/60.978`，det 为
  `55.436/49.657/63.797`。相对 Encoder 同点分别为：cls
  `-1.215/-0.967/-0.832`，det `-1.105/-0.688/-1.782`。
- 同点 pair mAP/AP50 为 `0.271857/0.477335`，both-independent mAP/AP50 为
  `0.310303/0.513735`。epoch 12 checkpoint、检测 metrics、50 序列、TrackEval
  `async_done=1` 与 108 个评估文件完整。
- e8 的 det HOTA 与 AP 优势没有形成中期双 HOTA 持续提升；10:22 精确终止
  PGID `2522015`，178 GPU0 已释放。该结构不继续到 e16，也不派生 scale、额外
  attention、附加 loss 或更复杂 decoder。继续等待 `0731_05` epoch 20 的独立补证。

## 2026-08-01 10:52 CST：0731_05 epoch-20 持久性结论

- epoch 20 cls HOTA/DetA/AssA 为 `51.640/43.389/63.028`，det 为
  `58.491/51.360/68.934`；相对 Encoder 同点，cls `+0.126/+0.398/-0.721`，
  det `-0.431/-0.839/+0.205`，未形成双 HOTA 提升。
- pair mAP/AP50 `0.289663/0.516488`、both-independent `0.329802/0.551333`；
  AP50 增益说明结构没有崩塌，但 det 检测覆盖损失仍阻止 HOTA 通过。
- 完整 epoch 20 产物核验后于 10:51 精确停止，不继续 e24。结合 `0731_01`，
  shared-attention 与 head detail 的组合在中期反复表现为 DetA/AssA 搬运，下一项只能采用
  单点、低参数、直接保护原 Encoder 检测路径的结构假设；不做 scale 扫描、额外层或附加 loss。

## 2026-08-01 11:20 CST：预留 0801_04 symmetric-position

- 全局实验号 `0801_04` 分配给 197 GPU4/5；下一可用编号为 `0801_05`。目标 formal workdir 为
  `/data4/litianhao/PairMmot/workdir_197/0801_04_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_symmetricposition_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh`。
- 该候选仅将共享 decoder self-attention 的 pair-position 改为交换对称表示；两帧独立
  deformable cross-attention、原有有序 frame-feature fusion、Encoder、proposal、PairDN、
  head、loss 与训练协议保持不变。它不新增参数或矩阵乘法，预计效率变化远低于 `5%`。
- 99 上 110 项 decoder 单测、formal/smoke 配置深拷贝、完整父/新模型构建、同参数量
  `22,758,775` 与初始函数等价检查已通过。当前状态仅为 `PREPARED`；必须先在 197 GPU4/5
  完成真实双卡 4-iter smoke，并验证有限总/DN/encoder loss、grad norm、checkpoint 中唯一
  position-only 标志及独立 cross-attention，五项正式启动门槛通过后才记为 `RUNNING`。

## 2026-08-01 11:28 CST：0801_04 formal 启动

- `d6e8c9a` 已同步到 197。GPU4/5 连续空闲、数据/GMC/预训练权重、空 workdir 与配置深拷贝
  核验后，真实双卡 4-iter smoke 通过；checkpoint 中 24 组 prev/curr attention 张量保持独立，
  最大训练差异 `0.00078709`，总/DN/encoder loss 与 grad norm 全部有限。
- formal fresh 于 11:25 启动，11:27 epoch 1 iter 50 为 `1.7400 s/iter`、loss `21.5687`、
  grad norm `115.9326`，两卡各约 19.2 GiB，五项门槛通过。首个 e4 只判断结构信号，e8/e12
  比较同点 Cls/Det HOTA；AP 仅诊断，非系统性退化时不按单点 AP 过早停止。

## 2026-08-01 11:59 CST：0801_05 symmetric-feature formal 启动

- 全局实验号 `0801_05` 分配给 178 GPU0；下一可用编号为 `0801_06`。它只将每层独立
  prev/curr cross-attention 输出的 feature fusion 改为交换对称：两帧输出取均值后重复，
  经原 `cross_fusion` 一次。cross-attention 仍独立，pair-position 仍有序，训练协议不变。
- 该单因素与 197 `0801_04 position-only` 正交，二者共同拆分 `0730_10` 的完整对称化；
  不新增参数、矩阵乘法、decoder 层、attention、分支或 loss。父/新模型均为
  `22,758,775` 参数且 state_dict 结构一致。
- 提交 `9bda2ed` 已同步四机和 GitHub。113 项 decoder 单测、配置深拷贝、完整构建和
  launcher 检查通过。178 真实 4-iter smoke 的最终 loss/grad norm 为
  `20.1570/185.1551`，24 组独立 attention 最大差异 `0.00076836`；formal 于 11:57
  fresh 启动，11:58 iter 50 为 `0.9385 s/iter`、loss `21.1807`、grad norm
  `98.8892`，GPU0 约 31.4 GiB，五项门槛通过。按 e4/e8/e12 同点双 HOTA 规则推进。

## 2026-08-01 12:32 CST：0730_10 e8 淘汰与下一步归因

- 完整 symmetric-pair e8 cls/det HOTA `42.890/48.228`，相对 Encoder 同点下降
  `2.379/1.965`；cls/det DetA 下降 `4.335/5.630`。pair mAP/AP50 为
  `0.2034/0.3969`，both-independent 为 `0.2390/0.4301`，同样明显下降。
- checkpoint、检测 metrics、TrackEval `async_done=1`、50 序列和 108 个评估文件完整。
  HOTA、DetA、AP 构成系统性退化，12:31 精确停止 PGID `1466786`，252 GPU0/1 释放；
  不再继续到 e12，也不对该结构做 scale 或 loss 调参。
- 定性归因优先指向“共享 cross-attention + 完整对称化”抹平帧特异检测证据。197 的
  `0801_04 position-only` 和 178 的 `0801_05 feature-only` 均保留独立 cross-attention，
  因而继续到既定 e4/e8 检查点。在这两个单因素对照给出证据前，252 暂不启动额外复杂模型；
  下一全局实验号仍为 `0801_06`。

## 2026-08-01 13:12 CST：0801_05 e4 与复杂度硬约束

- 178 `0801_05 feature-only symmetry` e4 cls/det HOTA 为 `34.947/38.300`，相对
  Encoder 同点为 `-1.262/-0.453`；双 DetA 为 `-1.108/-1.618`，det AssA 则提高
  `1.404`。完整 checkpoint、检测、50 序列及 108 个 TrackEval 文件已核验。
- 该负向信号不足以在 e4 淘汰，继续到 e8；但暂不支持继续扩大 feature symmetry。
  197 `0801_04` 仍在 epoch 4 正常训练，等待其独立 position-only 结果。
- 后续 decoder 必须保持局部、可解释和高效：优先零增参或参数增量不超过 `1%`，同卡同温
  吞吐下降不超过 `5%`；不增加 decoder 深度、额外 attention 栈、高分辨率分支、辅助 loss、
  类别重加权或 scale sweep。若两项 symmetry 在 e8 均失败，下一候选只考虑保留 shared query
  并融合双帧 attention innovation 的 residual-preserving 形式，参数量和主要矩阵乘法均不增加。
## 2026-08-01：低复杂度 decoder 推进约束与 0801_06

- 模型改进必须局部、可解释且有效：新增参数原则上不超过 `1%`，同卡同温吞吐下降
  不超过 `5%`；禁止用更深 decoder、额外 attention 栈、高分辨率分支、辅助 loss、
  类别重加权或 scale sweep 换取小幅指标。
- `0801_06` 保留 symmetric-position 的早期 det 优势，同时显式保留 shared query，
  只融合两帧 cross-attention innovation。该改动零新增参数、零新增主矩阵乘法。
- 252 同卡 2xb4 的 iter-50 实测为 `1.2082 s/iter`，旧结构对照为
  `1.2052 s/iter`，差异约 `0.25%`。因此允许进入 e4/e8/e12 验证；最终仍须同时超过
  Encoder 的 cls HOTA `54.437` 与 det HOTA `62.393`。
## 2026-08-01 14:16 CST：feature symmetry 终止结论

- `0801_05` e8 相对 Encoder 的 cls/det HOTA 分别下降 `2.091/1.018`，DetA 分别下降
  `1.584/1.374`，AssA 分别下降 `1.925/0.559`，两项 mAP 同步下降约
  `0.014–0.016`。因此不再把 frame-feature 做交换对称化，也不继续 e12。
- 后续只保留 position symmetry 与 shared-query residual-preserving fusion 这条零参数、
  近零开销路线。是否启动下一候选必须等待 `0801_04` e8 或 `0801_06` e4 的机制证据；
  不因 178/99 空闲而填充低信息量实验。

## 2026-08-03：decoder 合并增益 >1.5 推进计划

- 最终只接受同一 checkpoint 的 cls/det HOTA 分别严格大于 `54.437/62.393`，且两项绝对
  增量之和严格大于 `1.5`；任意单项失败或总 HOTA 不严格大于 `118.330` 均继续实验。
- 第一并行主线是在 252 训练 `0803_01`：逐层对两帧 classification residual 做零参数、
  类置换等变的 objectness 均值投影，保留各帧全部类别 margin 与 DN 绝对分类语义。
- 第二并行主线是在 197 从 `0801_09` e56 恢复同一优化轨迹到 e60/e64 及以后，用于判断
  当前最佳结构是否仍有晚期上升空间；物理设备改为 2x4，但全局 batch 保持 8。
- 两条主线均不以 e4/e8 直接否决。评测继续使用 cls/det HOTA 主门槛、DetA/AssA 归因和 AP
  退化诊断；不引入 class-aware 模块、reweight、scale sweep、更深 decoder 或额外 attention/loss。
- 若 `0803_01` 成熟节点未达到门槛，下一机制只从其 margin 保真/objectness 耦合证据派生一个
  单因素、低开销变体；不做无解释的参数扫描，并在新 formal 前重复配置深拷贝、完整构建、
  真实双卡 smoke、checkpoint 更新和 iter-50 五项门槛。

## 2026-08-03：第三条正交几何主线

- 178 GPU0 增加 `0803_02`：保持 `0801_09` 的 DN-isolated end-to-end 分类机制，只在每层
  普通 query 的回归 residual 上共享两帧 `w/h/angle`，中心位移保持独立。它与 `0803_01`
  的 objectness 耦合严格正交，且参数/state 增量为零。
- 三条运行线先收集完整节点：`0803_01` e4/e8/e12、`0801_09` e60/e64、`0803_02`
  e4/e8/e12。早期节点只用于 DetA/AssA/AP 轨迹诊断；除运行故障或明显系统性退化外，不以
  e4/e8 直接否决 decoder。

## 2026-08-03 02:36 CST：objectness 早期证据后的推进

- `0803_01` e4 的 HOTA、DetA、AssA 和 AP 均系统性低于 Encoder 与 `0801_09` 父线，说明
  强制两帧共享逐 query objectness 均值当前过度约束了置信度动态。该方向不再派生 scale、gate
  或额外分类耦合，但原实验继续到 e8/e12，保留 decoder 晚收敛验证窗口。
- 新结构优先等待正交的 `0803_02` 几何证据；若共享全部 `w/h/angle` 也表现为系统性覆盖下降，
  下一候选只做更局部的几何不变量（优先角度或面积中的单一分量），仍保持零参数、非 class-aware、
  无 reweight/额外 loss/attention/layer，并先做同协议真实 smoke。

## 2026-08-03 02:49 CST：局部角度不变量后备方案

- `0803_03` 已按上述归因实现为只共享 angle residual；`x/y/w/h`、分类 residual 和 DN prefix
  均保持父模型语义。它不引入参数、state、主矩阵计算、loss、attention、class-aware 信号或
  reweight，且保持帧交换等变。
- 252 隔离环境的 3 项语义/梯度测试、配置深拷贝和完整模型构建均通过；父/新模型同为
  `22,771,111` 参数和 `711` 个 state tensor。候选保持 `PREPARED`，不热更新活动仓库。
- 决策顺序不变：先读 `0803_02` e4/e8/e12 与成熟节点的 DetA/AssA/AP；若完整 shape 共享
  出现覆盖损失而 angle 局部先验仍有机制依据，则在释放的授权资源上先跑真实 4-iter smoke，
  通过 checkpoint 更新与 iter-50 五项门槛后再正式训练。

## 2026-08-03 03:20 CST：晚期续训平台证据

- `0801_09` e60 的绝对 cls/det HOTA 为 `54.489/62.422`，合并增益仅 `0.081`，较 e56 的
  `0.279` 回落；检测 AP 也轻微下降。cls 路径表现为 DetA 下降、AssA 上升，说明继续训练已开始
  做覆盖/关联搬运，而不是向 `>1.5` 目标形成净增益。
- 仍按既定承诺保留 e64 完整节点，不用 e60 单点直接终止；若 e64 继续平台或回落，则不再把纯
  epoch 延长作为主要探索手段，释放 GPU2/3 给经结构验证且正交的 decoder 候选。

## 2026-08-03 03:28 CST：完整 shape 共享的首个归因点

- `0803_02` e4 相对 `0801_09` e4 的 cls/det HOTA 为 `-0.984/-1.105`；det DetA 略升
  `0.247`，但 det AssA 下降 `2.992`，cls AssA 也下降 `1.724`。检测侧 both-independent
  mAP 高 `0.000007`，说明不是简单的检测器整体失效，而是跨帧 shape 约束当前干扰了关联。
- 该证据支持把下一几何候选收缩到单一角度 residual，不支持继续给 `w/h/angle` 共享增加 gate、
  scale 或 loss。原实验仍保留 e8/e12 延迟恢复窗；只有成熟轨迹继续显示相同失配，才在释放资源上
  部署已经完成零参数与完整构建验证的 `0803_03`。

## 2026-08-03 04:08 CST：objectness 共享的 epoch-8 归因

- `0803_01` e8 的 cls/det HOTA 为 `39.208/46.359`，相对 `0801_09` 父线 e8 仍低
  `2.764/1.819`，相对 Encoder e8 低 `6.061/3.834`。cls/det DetA 相对父线低
  `2.481/3.195`，AssA 也低 `3.763/0.293`，因此不是单纯 DetA/AssA 搬运。
- pair 与 both-independent 的 mAP/AP50 相对 e4 均恢复约 `0.075–0.117`，直接证明不能用
  e4 外推；但相对父线 e8 四项仍低 `0.0127–0.0357`。完整产物已核验，训练继续到 e12，
  不按 e8 直接否决。
- 分类侧不再派生 objectness 共享的 gate、scale、class-aware 或 reweight 版本；后续资源优先
  用于 `0803_02` 成熟几何证据、`0801_09` e64 平台确认，以及已静态验证的零参数角度候选。

## 2026-08-03 04:43 CST：shape e8 后的几何决策

- `0803_02` e8 相对 `0801_09` e8 的 cls/det HOTA 为 `+0.680/-0.455`，双 HOTA 总和净增
  `+0.225`；双 AssA 提高 `1.271/1.651`，但 det DetA 降低 `2.046`。pair 与
  both-independent 的 mAP/AP50 四项全部转为正增益，说明 e4 的关联损伤能够恢复，原实验继续
  e12，不以 e8 直接否决。
- 结构归因进一步收敛：完整 `w/h/angle` 共享的关联收益真实存在，但 `w/h` 硬约束压低 det
  覆盖。下一候选仍为零参数 `0803_03 angle-only`，保持 x/y/w/h、分类与 DN 独立；优先等待
  `0801_09` e64 完整评测释放 252 GPU2/3，再做目标配置 deepcopy、完整构建、真实双卡 smoke
  与 formal iter-50 五项门槛。

## 2026-08-03 04:58 CST：e64 关闭纯续训分支并切换 angle-only

- `0801_09` e64 为 `54.326/62.572`，总和 `116.898`，严格目标差 `1.432`；检测 AP 也较 e60
  回落。完整产物核验后已停止 PGID `3292233`，252 GPU2/3 释放，不再继续 e68/e72 纯续训。
- `0803_03` 在 252 使用独立 checkout，避免修改 GPU0/1 的活跃 `0803_01` 仓库。部署顺序固定为：
  远端提交/哈希核对、目标 formal 与 smoke 配置 `deepcopy`、完整父/新模型构建、GPU2/3 连续空闲、
  4-iter 真数据 DDP smoke、checkpoint/有限损失审计、formal iter-50 五项启动门槛。

## 2026-08-03 05:08 CST：angle-only 启动后的推进规则

- 上述隔离部署链已全部通过，`0803_03` formal PGID 为 `3460950`；iter 50 的时间/loss/grad norm
  为 `1.3920/21.3894/130.3343`，总、DN、encoder loss 与双卡状态有限正常。
- 与 `0803_02` 的比较重点固定为：angle-only 是否保留完整 shape 在 e8 的 AssA/AP 增益，同时
  避免 det DetA `-2.046` 的覆盖损失。e4/e8 只作轨迹诊断，继续至少 e12；任何新派生必须等待
  angle-only 的完整节点，且仍禁止 class-aware、reweight、额外 loss/attention/layer 或无依据
  scale/gate 扫描。

## 2026-08-03 05:32 CST：objectness 成熟证据关闭该分支

- `0803_01` e12 cls/det HOTA 为 `43.962/52.094`，相对父线 e12 仍低
  `3.433/2.342`；e8 到 e12 已分别恢复 `4.754/5.735`，因此该判断满足 decoder 晚收敛约束，
  不是以 e4/e8 直接否决。e12 pair mAP/AP50 为 `0.222812/0.393028`，both-independent 为
  `0.265827/0.449310`，未形成可支持继续硬共享 objectness 的反证。
- 完整产物核验后精确停止 PGID `3268273`，252 GPU0/1 释放。分类 objectness 分支关闭，
  不派生 gate、scale、class-aware 或 reweight；释放资源暂不填充，优先等待 `0803_03`
  angle-only 的 e4 轨迹，再决定是否需要零参数 terminal-only angle 接替。

## 2026-08-03 05:57 CST：完整 shape 成熟证据关闭该分支

- `0803_02` e12 cls/det HOTA 为 `46.101/50.453`，相对 `0801_09` 父线 e12 仍低
  `1.294/3.983`；尽管 e8→e12 恢复 `3.449/2.730`，且 pair/both 的 mAP/AP50 全部继续
  上升，恢复仍未转化为足够的跟踪净收益。该 e4/e8/e12 轨迹满足晚收敛观察要求。
- 完整产物核验后精确停止 PGID `2857661`，178 GPU0 释放。完整 `w/h/angle` 共享不再派生
  gate、scale 或调权版本；继续以 `0803_03` 单角度共享区分“角度噪声可抑制”与“逐层 reference
  耦合本身有害”。若 angle-only e4 表现为 AssA 改善但 DetA 明显受损，才启用零参数末层-only
  angle 后备；否则保持当前结构到 e8/e12，不提前占用释放资源。

## 2026-08-03 06:27 CST：周期切空间角度共识并行验证

- 利用完整 shape e12 已确认的几何约束代价，在 178 启动 `0803_04`：普通 query 每层仅共享
  π 周期切空间中的角度增量圆周中点，而非 `0803_03` 的 raw logit residual 算术均值。
  两者均保持 x/y/w/h、分类和 DN 独立，参数/state 增量为零；这是一项坐标语义对照，不是
  residual scale、gate 或超参数扫描。
- 3 项定向测试、126 项完整 decoder 回归、配置深拷贝和父/新模型完整构建通过。首次 smoke
  因 logger 间隔不产生 4-iter loss/grad 证据，未计为通过；retry1 逐 iter 日志、有限总/DN/
  encoder loss、有限 grad、364,504,628-byte checkpoint 与语义检查全部通过。
- formal PGID `2893156` 已在 178 GPU0 通过 iter-50 五项门槛；`0803_03` 同期继续在 252 GPU2/3
  到 e4/e8/e12。两项都不按 e4/e8 直接停止，先比较 HOTA、DetA、AssA 与 AP 的轨迹，再决定
  是否把共享收缩到 terminal-only 或继续周期表示。

## 2026-08-03 06:46 CST：raw-logit angle e4 归因

- `0803_03` e4 cls/det HOTA 为 `31.076/37.040`，相对 `0801_09` 父线低
  `3.230/1.550`；双 DetA、双 AssA 与 pair/both 的 mAP/AP50 也全部下降。该结构在早期不是
  覆盖与关联的交换，而是 raw-logit residual 硬共识同时伤害两者。
- 完整 checkpoint、检测和 TrackEval 产物已核验；训练继续 e8/e12，不把 e4 当作直接否决点。
  同时保留 `0803_04` 的 π 周期切空间圆周中点对照：若其 e4 显著恢复 DetA/AP，说明主要问题
  是角度坐标；若两者都系统性退化，则后续候选转向 terminal-only 或取消逐层角度共识，仍不做
  gate、scale、class-aware、reweight 或额外 loss/attention/layer。

## 2026-08-03 06:59 CST：中心局部坐标主线接替空闲资源

- `0803_03` 的 e4 说明 raw-logit angle 共识同时伤害 DetA、AssA 与 AP；但按晚收敛约束继续
  e8/e12。为避免只围绕角度派生 scale/gate，252 GPU0/1 的正交候选改为 `0803_05`：普通 query
  的中心校正先按各帧 reference `w/h` 归一化，在 reference 局部坐标取共同增量，再分别映回；
  `w/h/angle`、分类、DN、loss 与 decoder 主计算不变。
- 该设计零参数、class-agnostic、无 reweight；3 项新不变量测试、完整 129 项 decoder 回归、配置
  深拷贝、父/新完整构建、252 双卡真数据 4-iter smoke 与 checkpoint 语义检查已通过。正式 fresh
  PGID `3549855` 的 iter50 时间/loss/grad 为 `1.1465/21.3848/102.6548`，总、DN、Encoder loss
  有限且两卡约 19.2 GiB，无致命错误。
- 当前三条结构对照分别检验 raw-logit angle、π 周期 angle 与 reference-local center 共识；均保留
  e4/e8/e12 和成熟节点观察窗口。最终只接受同一 checkpoint 的 cls/det HOTA 严格超过
  `54.437/62.393` 且总和严格大于 `118.330`。

## 2026-08-03 07:10 CST：准备帧证据分类路径

- 在几何三路继续训练期间，对分类路径做正交审计：父结构两帧 iterative classification head
  每层读取相同融合 state，而已经计算的两帧 cross-attention evidence 没有回到各自分类 residual。
  `0803_06` 只修正这一信息路由，保留共享 recurrent query、全部框更新、reference、DN 与 loss。
- 该设计参数/state 增量为零，不增加 attention、decoder 深度或损失，不使用类别感知路由和
  score reweight。2 项定向测试、131 项完整 decoder 回归、配置深拷贝、launcher 语法和完整构建
  已在 178 隔离 checkout 通过；模型为 `22,771,111` 参数、711 个 state tensor。
- 候选保持 `PREPARED` 而非 `QUEUED`。不抢占正在 178 GPU0 运行的 `0803_04`，也不热更新其
  活跃仓库；待完整节点释放授权资源后，才依次进行真数据 smoke、checkpoint 语义审计和 formal
  iter-50 五门槛。若启动，仍收集 e4/e8/e12 与成熟节点，不以 e4/e8 单点直接否决。

## 2026-08-03 07:46 CST：周期角度 e4 后的资源决策

- `0803_04` e4 的 cls/det HOTA 为 `36.024/43.788`，相对 raw-logit `0803_03` 提高
  `4.948/6.748`，相对父线 `0801_09` 提高 `1.718/5.198`。DetA、AssA 与四项 AP 也全面高于
  raw-logit 对照，证明正确的 π 周期坐标是实质设计变量，而非表示细节。
- 相对 Encoder e4，det HOTA 已高 `5.035`，但 cls HOTA 仍低 `0.185`；cls DetA 高
  `2.126` 而 AssA 低 `5.451`。这形成明确的晚收敛观察问题：周期角度能否在保持覆盖优势时恢复
  cls 关联。因此 `0803_04` 保持 178 GPU0 到 e8/e12，不在 e4 释放或停止。
- 当时 99 GPU0/1 被外部任务占用；后续用户澄清 99 只限制总计 2 卡而不固定序号。197 GPU4/5 虽空闲但 formal 吞吐约
  `80 s/iter`；252 四卡由 `0803_03/05` 使用。`0803_06` 保持 `PREPARED`、不排队、不抢占，
  待首个合适授权资源释放后再执行真数据 smoke 和 formal 五门槛。

## 2026-08-03 07:54 CST：准备周期角度与帧证据联合候选

- e4 归因给出互补缺口：`0803_04` 的周期角度显著提高 det 与检测 AP，但 cls 仍受早期关联缺口
  限制；`0803_06` 恰好只恢复被共享 state 丢弃的帧特异分类证据。`0803_07` 把两项零参数路由
  组合，框回归和分类的信息路径不交叉，不增加 attention、loss、层、class-aware 信号或 reweight。
- 组合不变量测试、完整 132 项 decoder 回归、配置深拷贝、完整构建与 launcher 语法已在 178
  隔离 checkout 通过；参数/state 与父模型相同。活动 `0803_04` 仓库和进程未修改。
- `0803_07` 与 `0803_06` 均保持 `PREPARED`。先读取 `0803_04` e8/e12 是否保持周期坐标收益；
  若保持，联合候选优先用于冲击最终同点双超目标，单因素 `0803_06` 保留为分类归因对照。任何正式
  启动仍必须先通过目标 GPU 真数据 smoke、checkpoint 语义审计和 iter-50 五门槛。

## 2026-08-03 08:31 CST：raw-angle e8 与 normalized-center e4 决策

- `0803_03` e8 cls/det HOTA 为 `40.644/47.265`，相对自身 e4 分别恢复
  `+9.568/+10.225`，确认 decoder 的明显晚收敛；但相对父线 e8 仍低
  `1.328/0.913`，所以不在 e8 停止，也不从 raw-logit 路径派生 gate、scale 或 reweight。
- `0803_05` e4 cls/det HOTA 为 `31.737/37.202`，相对 raw-angle e4 仅提高
  `0.661/0.162`，且 det AssA 下降 `0.712`。保持到 e8/e12，用成熟轨迹判断局部中心坐标
  是否只是早期等价扰动。
- 两个节点的 checkpoint、5416 条检测记录、50 序列、异步 TrackEval、28 CSV 与 108 个
  非空评估文件均完整。252 的两条任务继续运行；178 的 `0803_04` 已进入 e8，下一决策优先读取
  其周期角度收益是否延续，再决定先部署 `0803_07` 联合候选还是 `0803_06` 分类归因候选。

## 2026-08-03 09:03 CST：periodic-angle e8 强正向与下一调度

- `0803_04` e8 cls/det HOTA 为 `45.587/52.915`，相对 raw-angle e8 提高
  `4.943/5.650`，相对父线 e8 提高 `3.615/4.737`，相对 Encoder e8 提高
  `0.318/2.722`。合并同点增益为 `+3.040`，证明 π 周期坐标设计是当前最强 decoder 机制。
- 该结果仍低于严格最终阈值 `54.437/62.393`，因此 `0803_04` 继续 e12 和更成熟节点；
  不把 e8 同点双超等同于目标完成，也不停止训练。
- 下一优先候选固定为正交组合 `0803_07`，用帧证据分类路径补充周期角度的 cls 缺口。
  已准备 252 的 2×b4 配置、真数据 4-iter smoke 与 formal launcher；当前仅完成本地静态检查。
  等 `0803_03` e12 完整收口并释放 GPU2/3 后，在独立 checkout 依次执行完整构建、132 项回归、
  真数据 smoke、checkpoint 语义审计和 formal iter-50 五门槛，不热更新活动仓库。

## 2026-08-03 09:10 CST：0803_07 252 部署前验证完成

- 隔离 checkout 固定在 `f3752db`；父/新完整模型均为 `22,771,111` 参数和 711 个 state tensor，
  增量为零。132 项 decoder 回归、2 个 subtests、配置深拷贝和两份 launcher 语法全部通过。
- 保持 `PREPARED` 而非 `QUEUED/RUNNING`。GPU2/3 仍由 `0803_03` 占用；不抢占、不热更新，
  资源释放后才执行真数据 DDP smoke、checkpoint 语义检查与 formal iter-50 五门槛。

## 2026-08-03 10:08 CST：raw-angle 收口并切换联合候选

- `0803_03` e12 cls/det HOTA 为 `43.687/51.887`，虽比自身 e8 恢复
  `3.043/4.622`，但相对父线 e12 仍低 `3.708/2.549`。完整 e4/e8/e12 轨迹足以否定
  raw-logit angle 共识，09:58 停止并释放 GPU2/3；结论不是 e4/e8 早停。
- `0803_05` e8 为 `39.525/45.114`，相对 e4 恢复 `7.788/7.912`，但仍低于父线，
  保持到 e12，不提前停止。
- `0803_07` 已接替 GPU2/3：真数据 DDP smoke 四步 loss/grad 全部有限、checkpoint 语义通过；
  formal PGID `3694870` 的 iter50 时间/loss/grad norm 为 `1.2858/21.4228/119.3820`，
  五项门槛通过。下一步按同一完整评估协议持续到 e4/e8/e12 和成熟节点。

## 2026-08-03 10:17 CST：periodic-angle e12 继续长跑

- `0803_04` e12 cls/det HOTA 为 `47.913/55.257`，相对 e8 再提高
  `2.326/2.342`，相对父线 e12 仍提高 `0.518/0.821`；pair mAP/AP50 与
  both-independent 也继续升至 `0.2577/0.4395` 和 `0.3023/0.4959`。
- 相对 Encoder e12 尚低 `1.767/1.284`，故不把父线正增益误写成严格目标完成；同时因 e8→e12
  仍持续上升，不在 e12 释放 178。继续收集 e16 及后续成熟节点，与 252 的 `0803_07` 联合候选
  并行比较分类帧证据是否能补足 cls 收敛缺口。

## 2026-08-03 11:22 CST：normalized-center 收口与资源续接

- `0803_05` e12 cls/det HOTA 为 `43.161/49.396`，相对自身 e8 继续恢复
  `3.636/4.282`，但相对父线同点低 `4.234/5.040`、相对 Encoder 同点低
  `6.519/7.145`。完整 e4/e8/e12 轨迹足以否定归一化中心共识，不是 e4/e8 早停。
- checkpoint、检测、TrackEval 与 50/28/108 产物核验完整后，精确停止 PGID `3549855`；
  23 个成员退出，252 GPU0/1 释放，GPU2/3 上 `0803_07` 不受影响。
- 下一步用 GPU0/1 启动零参数 `0803_06 frame-evidence-cls` 单因素分支，回答联合候选中分类帧证据
  是否独立贡献；保持 x/y/w/h/angle、共享 recurrent query、DN、loss 与 decoder 深度不变，不引入
  class-aware 或 reweight。必须依次通过目标环境全构建、完整回归、真数据双卡 smoke、checkpoint
  语义审计与 formal iter-50 五门槛，再收集 e4/e8/e12 和成熟节点。

## 2026-08-03 11:31 CST：frame-evidence 单因素运行与 periodic-angle 延长

- `0803_06` 已在 252 GPU0/1 完成目标环境 132 项回归、零参数全构建、真数据双卡 smoke 和
  checkpoint 语义审计；formal PGID `3765372` 的 iter50 时间/loss/grad norm 为
  `1.2946/21.4356/109.5840`，五门槛通过。按 e4/e8/e12 和成熟节点协议收集轨迹。
- `0803_04` e16 为 `48.474/55.272`，e12→e16 为 `+0.561/+0.015`；检测侧单区间趋平，
  分类侧仍有缓慢增长。由于该机制此前在 e8/e12 均为正向且 decoder 可能延迟收敛，继续 e20，
  再判断平台是否持续；严格目标仍是同 checkpoint 双超 `54.437/62.393` 且合并增益 `>1.5`。

## 2026-08-03 11:45 CST：组合候选 e4 归因与下一结构方向

- `0803_07` e4 cls/det HOTA `32.535/38.723`，相对 periodic-angle 单因素 e4 为
  `-3.489/-5.065`；pair mAP/AP50 与 both-independent 分别为
  `0.1220/0.2350` 和 `0.1618/0.3021`，覆盖与关联同时下降。完整 50/28/108 产物已核验。
- 该实验保持到 e8/e12，不按 e4 直接停止。e4 只支持一个结构归因：分类 head 直接以原始帧证据
  取代共享状态会丢失公共语义。下一候选改为 `shared + swap-odd detail`：两帧分类输入的算术平均
  精确等于共享 `layer_output`，差值仅来自已有 frame evidence 的反对称部分；框回归、reference、
  DN、loss、attention 和 decoder 深度不变，零参数、class-agnostic、无 reweight。先完成实现与
  不变量测试，待活动实验成熟节点释放资源后再决定是否启动。

## 2026-08-03 11:52 CST：common-preserving frame-detail 静态门槛

- `0803_08` 已实现 `shared ± half(frame_prev-frame_curr)` 分类路由，并与 periodic-angle 正交组合。
  133 项测试证明 shared hidden/reference 不变、分类 midpoint 精确保留、swap-odd detail 生效且 direct
  路由互斥；父/新模型同为 `22,771,111` 参数和 711 state tensors，增量严格为零。
- 178 隔离 checkout `8dd19d8` 完成配置深拷贝、完整构建和 launcher 语法检查，状态为
  `PREPARED`。当前不做真数据 smoke 或排队，避免抢占仍需 e20 验证延迟收敛的 `0803_04`。

## 2026-08-03 12:02 CST：log-size tangent 几何候选静态门槛

- `0803_09` 将候选宽高相对各自 reference 的乘法变化表示成 log 增量，再共享其均值；角度继续
  使用 periodic tangent midpoint。该设计保留帧间中心运动与分类独立性，避免 `0803_02` 将尺寸和
  raw-angle 一起平均造成的坐标混杂。
- 135 项完整测试、配置深拷贝、父/新全模型零参数构建和 launcher 语法通过；状态为
  `PREPARED`，不排队、不占 GPU。资源释放时优先级由 `0803_04` e20 平台证据和 `0803_06/07`
  e4/e8 归因共同决定：若分类直路持续负向而 periodic-angle det 已平台，优先比较 0803_08 的
  common-preserving 分类与 0803_09 的尺寸几何增益，不进行 scale 或 reweight 扫描。

## 2026-08-03 12:46 CST：periodic-angle e20 延迟收敛证据

- `0803_04` e20 cls/det HOTA 为 `49.446/55.397`，相对 e16 再升 `0.972/0.125`；
  pair mAP/AP50 与 both-independent 也同步提高到 `0.2706/0.4658` 和 `0.3132/0.5154`。
- cls 的覆盖与关联均继续增长，说明 decoder 延迟收敛确实存在；det 增长已很小但 DetA 仍升，不能
  在 e20 终止整个正向分支。继续 e24，严格目标仍未达到。若 e24 det 仍平台，则优先用已经 PREPARED
  的 `0803_09` 测试 log-size 几何补益，而 `0803_08` 由 `0803_06/07` 的分类归因决定是否部署。

## 2026-08-03 13:04 CST：frame-evidence 单因素 e4 归因

- `0803_06` e4 cls/det HOTA 为 `30.698/38.350`，相对 periodic-angle 单因素 e4 为
  `-5.326/-5.438`；pair mAP/AP50 与 both-independent 为 `0.1200/0.2291` 和
  `0.1621/0.3011`。checkpoint、5416 条检测、50 序列、28 CSV、108 个非空文件与异步完成
  标志均完整，PGID `3765372` 的 23 个成员已进入 e5。
- 单因素和组合分支在 e4 的检测 AP 几乎一致，且均显著弱于 periodic-angle，支持早期损失来自
  direct frame-evidence 分类路由本身，而非周期角度交互。仍继续两项到 e8/e12，以排除 decoder
  延迟收敛；若成熟节点仍负向，分类候选只保留公共 midpoint 并注入 swap-odd detail 的 `0803_08`。

## 2026-08-03 13:22 CST：frame-evidence + periodic-angle e8 归因

- `0803_07` e8 cls/det HOTA 为 `41.380/47.515`，相对 periodic-angle 单因素同点为
  `-4.207/-5.400`，相对 Encoder 同点为 `-3.889/-2.678`。e4→e8 的双 HOTA 增量为
  `+8.845/+8.792`，确认它确实收敛，但成熟差距仍未缩回。
- 检测 pair mAP/AP50 `0.1887/0.3403`，both-independent `0.2297/0.3976`；四项相对
  periodic-angle 同点均显著下降。checkpoint、127426 条检测、50 序列、28 CSV 和 108 个
  非空文件完整，PGID `3694870` 已进入 e9。
- 继续 e12，不以 e8 早停；若 e12 仍维持同方向缺口，则释放 GPU2/3，优先部署精确保留
  shared midpoint 的 `0803_08`。direct frame-evidence 路由不再扩展新变体。

## 2026-08-03 14:05 CST：periodic-angle e24 平台与 178 切换

- `0803_04` e24 cls/det HOTA 为 `50.133/55.346`，相对 e20 为 `+0.687/-0.051`；检测
  DetA 仍升而 AssA 下滑，四项 AP 仍小幅增加。该节点证明周期角度的分类延迟收益继续存在，
  同时检测 HOTA 已进入平台，不能仅靠延长训练达到严格最终阈值。
- 397,682,100-byte checkpoint、5416 条检测、50 序列、28 CSV 与 108 个非空文件完整后，
  精确停止 PGID `2893156`，9 个成员全部退出，GPU0 已释放；e24 保留为可恢复断点。
- 178 下一任务选择 `0803_09 log-size tangent + periodic-angle`：只补宽高的 reference-local
  乘法几何共识，不改变中心、分类、DN、loss、attention 或深度。先走真数据 smoke 与 formal
  iter50 五门槛；`0803_08` 保持 PREPARED，等待 `0803_06/07` e12 的分类成熟归因。

## 2026-08-03 14:12 CST：log-size tangent 正式启动

- 178 隔离 checkout `35e18f1c` 的真数据 4-step smoke 已通过：四步 loss/grad 全部有限，
  364,505,012-byte checkpoint 与 iterative-cls/DN 语义检查完整。
- fresh formal PGID `2971994`；真实 iter50 为 `0.9750 s/iter`、loss `21.0017`、grad
  `109.5454`，9 个成员、GPU0 约 31.4 GiB 驻留、错误扫描和提交来源五门槛均通过。
- `0803_09` 状态提升为 `RUNNING`。继续收集 e4/e8/e12；它只改变宽高与角度的几何坐标共识，
  不引入参数、类别感知、重加权、额外 attention 或 decoder 深度。

## 2026-08-03 14:29 CST：frame-evidence 单因素 e8 归因

- `0803_06` e8 cls/det HOTA `40.922/46.854`，相对 periodic-angle 单因素同点为
  `-4.665/-6.061`，相对 Encoder 同点为 `-4.347/-3.339`。e4→e8 虽增长
  `10.224/8.504`，但直接分类路由的成熟差距仍未追回。
- pair mAP/AP50 `0.1906/0.3551`，both-independent `0.2334/0.4148`；四项均明显低于
  periodic-angle。同点完整 checkpoint、128933 条检测、50 序列、28 CSV 和 108 个非空文件
  已核验，PGID `3765372` 已进入 e9。
- 继续 e12 后再释放 GPU0/1。单因素与组合的 e8 证据共同排除继续扩展 direct frame-evidence
  变体；分类侧下一实验固定为保留 shared midpoint 的 `0803_08`。

## 2026-08-03 15:01 CST：frame-evidence 联合分支 e12 收口

- `0803_07` e12 cls/det HOTA `43.504/51.168`，相对 periodic-angle 同点为
  `-4.409/-4.089`，相对 Encoder 同点为 `-6.176/-5.373`。e8→e12 虽继续增长，成熟差距
  仍显著；四项 AP 相对 periodic-angle 同点也下降 `0.0543/0.0712/0.0593/0.0746`。
- 381,031,670-byte checkpoint、143610 条检测、50 序列、28 CSV 与 108 个非空文件完整后，
  精确停止 PGID `3694870`，23 个成员全部退出，GPU2/3 已释放，e12 断点保留。
- direct frame-evidence 路由完成 e4/e8/e12 成熟负向审计。GPU2/3 下一任务固定为 `0803_08`：
  保留 shared 分类 midpoint，仅注入 swap-odd 帧细节；先补 252 双卡配置、目标构建、真数据 smoke
  与 formal iter50 五门槛。

## 2026-08-03 15:10 CST：common-preserving frame-detail 双卡启动

- 新增 252 2×4 配置与 GPU2/3 smoke/formal launcher，隔离 checkout 固定 `08356f9`。135 项
  decoder 测试、2 个 subtest、脚本语法、目标配置深拷贝与完整构建通过；零参数、711 state tensors。
- 首次构建继承了旧仓库 `PYTHONPATH`，registry 类来源不正确，未进入训练；清空污染路径并校验
  target import 后构建通过。真数据 4-step DDP smoke loss/grad 全部有限，checkpoint 与语义检查完整。
- fresh formal PGID `3940521`；iter50 `1.2866 s/iter`、loss `21.4134`、grad `109.5162`，
  7 个成员、GPU2/3 各约 19.2 GiB、错误扫描与 provenance 五门槛通过。状态为 `RUNNING`，
  继续 e4/e8/e12，直接检验保留 shared midpoint 后的 swap-odd 分类细节是否避免成熟负迁移。

## 2026-08-03 15:25 CST：log-size tangent e4 双向正增益

- `0803_09` e4 cls/det HOTA `36.930/44.486`，相对 periodic-angle 单因素同点为
  `+0.906/+0.698`，合并 `+1.604`；相对 Encoder e4 为 `+0.721/+5.733`。宽高乘法几何
  共识补充了周期角度而未破坏分类侧。
- pair mAP/AP50 `0.1743/0.3157`，both-independent `0.2193/0.3845`，四项相对
  periodic-angle 同点均提高。checkpoint、114290 条检测、50 序列、28 CSV 与 108 个非空文件完整。
- 这是首个在同点相对 periodic-angle 双 HOTA 与四项 AP 全正的尺寸机制；继续 e8/e12 检验增益
  是否成熟保持。严格目标仍使用 Encoder 最终 `54.437/62.393`，不以 e4 宣告达成。

## 2026-08-03 15:57 CST：frame-evidence 单因素 e12 延迟收敛

- `0803_06` e12 cls/det HOTA `45.752/52.950`，e8→e12 为 `+4.830/+6.096`；相对
  periodic-angle 的缺口收窄到 `2.161/2.307`，相对 Encoder e12 为 `-3.928/-3.591`。
- 四项 AP 提高到 `0.2238/0.4040/0.2689/0.4633`，相对 e8 均明显增长；checkpoint、
  147478 条检测、50 序列、28 CSV 与 108 个非空文件完整。
- 单因素明显优于 `0803_07` 联合分支同点，说明 direct frame evidence 与 periodic-angle 的组合
  存在负交互，但单因素自身仍在延迟追赶。保留 GPU0/1 到 e16，不以 e12 直接否决；e16 再判断
  追赶速度是否持续。

## 2026-08-03 16:22 CST：下一尺寸候选预案

- `0803_10` 采用“共享对数面积、保留逐帧纵横比、周期角度共识”，避免 `0803_09` 对宽高两个
  自由度都强制共识。137 项回归与零参数全模型构建已通过，状态为 `PREPARED/WAITING_GPU`。
- 调度顺序：先收集 `0803_09` e8/e12；若 log-size 正增益成熟保持，则优先延长强势线；若纵横比
  强共享导致后期回落，则在下一块合法空闲 GPU 上依次执行 `0803_10` 真数据 4-iter smoke、
  checkpoint 语义检查和 formal iter50 五门槛。严格目标仍为同 checkpoint cls/det 均高于
  `54.437/62.393`，且两项绝对增益之和大于 `1.5`。

## 2026-08-03 16:41 CST：log-size e8 决策

- `0803_09` e8 cls/det HOTA `46.170/53.539`，相对 periodic-angle 同点保持
  `+0.583/+0.624`，四项 AP 也全部为正；因此保留到 e12，不能因 e8 未达最终阈值而停止。
- e4 的合计同点增益 `+1.604` 到 e8 收窄为 `+1.207`。e12 重点看增益是否继续衰减，以及
  AssA 是否受宽高两个自由度全共享限制；该分解直接决定下一块 GPU 是延长 `0803_09`，还是启动
  只共享面积、保留纵横比的 `0803_10`。

## 2026-08-03 16:53 CST：common-preserving frame-detail e4 决策

- `0803_08` e4 cls/det HOTA `32.065/39.067`，相对 periodic-angle 同点
  `-3.959/-4.721`；四项 AP 也全部下降，说明 swap-odd 帧细节即使保持分类 midpoint，早期仍明显
  干扰分类与关联。
- 继续保留到 e8/e12，遵守 decoder 慢收敛约束；若成熟节点仍负，不再扩展帧差分类路径。尺寸
  几何候选 `0803_09/10` 维持更高优先级。

## 2026-08-03 17:00 CST：后两层几何共识候选

- `0803_11` 只在 decoder 最后两层共享 log-size 与周期角度，第一层保留帧特异几何探索，用于
  对照 `0803_09` 每层强共识在 e4→e8 增益收窄的问题。138 项回归与零参数全构建通过，状态为
  `PREPARED/WAITING_GPU`。
- 若 `0803_09` e12 仍双正但增益继续衰减，优先比较 `0803_10` 的自由度放松与 `0803_11` 的
  层级放松；两者均先通过真实 4-iter smoke 和 formal iter50 五门槛，再保留到 e12 成熟判定。

## 2026-08-03 17:05 CST：252 快速接替预案

- `0803_10` 已补齐 252 双卡配置并在隔离 checkout 完成零参数全构建。若 `0803_06` e16/e20
  追赶停滞并释放 GPU0/1，优先在该卡对启动 0803_10 smoke/formal；若 06 仍显著追赶，则不为
  新候选提前终止它，继续等待 178 或 252 的下一组合法空闲卡。

## 2026-08-03 17:06 CST：197 降级处理

- 197 在隔离 clone 完成后离线；bundle 与旧提交 clone 均保留，未进入 fetch/build/smoke。暂停该
  慢资源调度，待网络恢复后从 fetch 续做，不重新 clone。
- 当前实验决策和目标完成不依赖 197；近期顺序仍为 06 e16、09 e12、08 e8，以及 252 空闲后
  的 0803_10 真数据 smoke。

## 2026-08-03 17:24 CST：frame-evidence 单因素延长到 e20

- `0803_06` e16 cls/det HOTA `47.584/55.930`，相对 e12 仍增 `+1.832/+2.980`；相对
  periodic-angle e16 已形成 `-0.890/+0.658`，四项 AP 继续增长。
- 因检测已反超且分类仍追赶，保留 GPU0/1 到 e20，不为 0803_10 提前终止。e20 再看分类缺口
  是否继续缩小以及 det 优势是否保持；若双侧进入平台，再释放卡对部署 0803_10。

## 2026-08-03 17:33 CST：长轨迹基准复核与 0803_10 接替

- 原始 `0801_09` 从 e16 `50.036/56.933` 持续晚收敛，到 e40 首次同点双超、e56 才首次绝对双超
  `54.437/62.393`；因此当前强势几何候选不能用 e12/e20 未过最终阈值直接否决。`0803_09` e8
  相对 Encoder 同点合计领先 `4.247`，若成熟期保留超过 `1.5` 即具备目标潜力，应进入长跑。
- 同一复核也表明 `0803_06` e16 已被原始 decoder 双侧严格支配，故撤销继续 e20 的旧决定并
  精确停止；这基于四个完整节点与强版本支配，不是早期 epoch 淘汰。
- GPU0/1 已由零参数 `0803_10 shared log-area + periodic-angle` 接替；真实 smoke 和 formal
  iter50 五门槛通过。收集 e4/e8/e12 后，只要相对原始 `0801_09` 或 Encoder 的优势仍有恢复趋势，
  继续到 e24/e40/e56；严格目标仍按绝对双超且合计增益大于 `1.5`。

## 2026-08-03 18:00 CST：0803_09 确认为长轨迹主候选

- e12 cls/det HOTA `49.206/56.275` 虽暂低于 Encoder 同点 `0.474/0.266`，但相对原始
  `0801_09` decoder 同点提高 `1.811/1.839`，联合 `+3.650`；相对 periodic-angle 也联合
  `+2.311`，证明 log-size tangent 不是早期偶然单侧增益。
- 保留 178 GPU0 继续 e16/e24/e40；若相对原始 decoder 的联合优势在成熟区间仍大于 `1.5`，
  继续到 e56 并直接检验绝对双超。0803_10/0803_08 继续作为约束强度和分类路径的正交对照。

## 2026-08-03 18:35 CST：0803_08 保留到成熟节点

- e8 `40.688/47.811` 相对 Encoder 同点为 `-4.581/-2.382`，相对 periodic-angle 为
  `-4.899/-5.104`；分类细节注入暂未表现为正交增益。
- 不按 e8 直接停止；继续 e12 并检查三节点差距斜率。若 e12 仍被 Encoder 与 periodic-angle
  双侧稳定支配，再精确停止并由零参数 0803_11 late geometric consensus 接替 GPU2/3。

## 2026-08-03 18:45 CST：0803_11 双卡接替路径就绪

- 252 隔离仓库、2xb4 formal/smoke 配置和 launcher 已准备；显式隔离导入后零参数全构建及目标
  单测通过，当前不占 GPU。
- 仅在 0803_08 e12 完整产物确认且成熟负向后执行接替；先做真实 DDP smoke 与 checkpoint
  语义检查，再按 iter50 五门槛宣称 formal 运行。

## 2026-08-03 19:06 CST：0803_10 早期约束信号

- e4 `32.399/39.251` 相对 Encoder 同点 `-3.810/+0.498`，相对 periodic-angle
  `-3.625/-4.537`；只共享 log-area、保留 frame-specific aspect 暂未优于全尺度切空间。
- 继续 e8/e12 检查晚收敛恢复；不以 e4 停止。0803_09 仍是长轨迹主候选，0803_10 作为
  约束强度消融保留到成熟节点。

## 2026-08-03 19:22 CST：0803_09 epoch 16 长轨迹复核

- e16 `50.732/57.218` 相对原始 decoder 同点双侧提高 `0.696/0.285`，但联合优势从 e12
  的 `3.650` 收窄到 `0.981`；相对 Encoder 同点仍差 `0.359/1.102`。
- 不作 e16 截断：继续 e20/e24，检查 log-size 共识的优势是暂时回落还是成熟期消失；同时保留
  0803_11 late schedule，避免全层几何投影若在后期形成过强约束时无替代路线。

## 2026-08-03 19:36 CST：0803_11 提前进入 197 formal

- 197 GPU 管理恢复，GPU4/5 smoke 与 formal iter50 五门槛均通过；以 `1.4287 s/iter` 的真实
  训练速度运行，不再受此前异常慢速结论限制。
- 0803_11 只在最后两层施加 log-size/periodic-angle 共识，用于检验 0803_09 e12→e16 优势
  收窄是否由首层过早投影造成。先收 e4/e8/e12，不用 e4/e8 直接否决。

## 2026-08-03 20:28 CST：成熟分类负线收口与渐进几何接替

- 0803_08 e12 `44.177/52.763` 在 e4/e8/e12 全轨迹持续被 Encoder、periodic-angle 与原始
  decoder 双侧支配，故按成熟证据停止，不属于 e4/e8 早停。
- 0803_12 将几何约束强度随 decoder 深度递增：首层自由、倒数第二层共享面积/周期角、末层
  共享完整尺度/周期角。该结构针对 0803_09 后期优势收窄与 0803_10 面积约束早期偏弱的交叉
  证据，保持零参数、class-agnostic、无 reweight 和近零计算开销。
- 252 GPU2/3 formal iter50 五门槛通过，先收 e4/e8/e12；不以 e4/e8 直接否决。

## 2026-08-03 20:32 CST：全层尺度共识转入成熟确认

- 0803_09 e20 `49.781/57.217`，相对原始 decoder 同点已转为 `-1.062/-0.816`；e12 正增益、
  e16 收窄、e20 反转支持“全层完整尺度投影后期过约束”的设计归因。
- 仍运行到 e24 形成四个连续中后期节点；若负向趋势保持，则释放 178，不再把全层尺度共识
  延长为主要候选，优先观察 0803_11 晚层投影和 0803_12 渐进投影。

## 2026-08-03 20:35 CST：面积分解继续到成熟节点

- 0803_10 e8 `41.567/48.238`，相对 Encoder 同点 `-3.702/-1.955`；共享面积、保留纵横比没有
  保住 e4 的 det 单侧微增，说明问题不只是宽高自由度耦合。
- 保持到 e12 后再决定释放 252 GPU0/1；结构探索重点继续转向晚层施加与逐层递增，而非增强
  投影幅度或增加计算模块。

## 2026-08-03 20:44 CST：准备 terminal-only 几何隔离实验

- 0803_13 将 log-size/periodic-angle 共识缩到最终输出层一次；前三层逐帧 reference 和后续
  attention 不受配对投影影响，用于直接检验 0803_09 后期回落是否源于约束污染状态传播。
- 单测与零参数全构建通过，排队在 178；0803_09 e24 完成前不抢占，释放后先 smoke 再 formal。

## 2026-08-03 21:48 CST：全层尺度成熟收口，terminal-only 接替

- 0803_09 e24 `50.256/57.489`，相对原始 decoder 同点 `-1.453/-1.292`；结合 e12→e24
  连续轨迹，停止全层尺度共识，保留完整 e24 断点并释放 178 GPU0。
- 0803_13 已进入四步真数据 smoke；只有 loss/grad 全有限、checkpoint 完整、错误扫描与
  结构语义通过后才启动 fresh formal，并同样保留到 e12 以上评估慢收敛。

## 2026-08-03 21:52 CST：terminal-only 进入正式长轨迹

- 0803_13 smoke 与 formal iter50 五门槛均通过，且正式速度 `0.9711 s/iter`，没有计算量激增。
- 先观察 e4/e8/e12 与原始 decoder/Encoder/periodic-angle 的同点差值；无论早期高低，都不以
  e4/e8 单点停止，重点检查 terminal-only 是否避免 0803_09 e12 后优势衰减。

## 2026-08-03 22:06 CST：terminal-only 面积分解补齐二维对照

- 0803_10 e12 `44.971/52.008` 成熟双负后停止，说明共享面积本身不能解决“每层都投影”的
  状态污染问题。
- 0803_14 将 0803_10 的面积/纵横比分解与 0803_13 的最终层隔离组合：最终层共享面积和
  周期角，但纵横比逐帧保留。该零参数结构在 252 GPU0/1 先 smoke，随后按同样长轨迹评估。

## 2026-08-03 22:11 CST：terminal-area 进入正式轨迹

- 0803_14 smoke 与 formal iter50 五门槛通过，正式速度 `1.2555 s/iter`，参数/层数/attention
  均无增长。
- 与 0803_13 形成最终层“完整尺度 vs 仅面积”的正交对照；两项都先收 e4/e8/e12并保留
  后期节点，检验保留纵横比是否进一步减轻输出过约束。

## 2026-08-03 22:13 CST：渐进投影保留到中期

- 0803_12 e4 `32.057/38.097`，分类相对 Encoder 低 `4.152`、det 低 `0.656`；该节点只说明
  渐进约束的早期分类收敛偏慢。
- 继续 e8/e12，重点观察分类差距是否回补，以及 det 是否因末层完整尺度投影在中期转正。

## 2026-08-03 22:29 CST：晚层投影保留到中期

- 0803_11 e4 `31.540/38.185`，与渐进方案同样表现为分类早期慢、det 接近 Encoder；该相似性
  说明首轮性能主要受晚层尺度共识影响，尚不足以区分“最后两层完整”与“面积→尺度递增”。
- 继续 e8/e12，结合 0803_12 同点差值判定渐进强度是否优于直接晚层完整投影。

## 2026-08-03 22:31 CST：准备最终层角度最小隔离

- 0803_15 只保留 terminal periodic-angle，共享调用一次，不动尺寸/面积；它与 0803_13/14
  形成“角度 / 角度+面积 / 角度+完整尺度”的最终层结构递进。
- 单测和零参数构建通过，排队在 178；只有当前 0803_13 成熟释放后才进入 smoke/formal。

## 2026-08-04 05:19 CST：几何与语义 margin 组合的早期读数

- 0803_18 e4 为 cls/det HOTA `30.440/38.288`，相对单独 terminal geometry 的 0803_13
  同点为 `-2.409/+0.969`；分类收敛被明显推迟，det 则略有改善。
- AP 与完整 TrackEval 已闭环，但 e4 不作为 decoder 的否决点；197 继续 e8/e12，以中期
  是否回补 cls 且保留 det 判断 semantic margins 能否与终层几何互补。若成熟仍双负，优先
  切换已准备的 transported semantic geometry，而不增加层数、attention 或 loss。

## 2026-08-04 05:41 CST：成熟长跑与新结构筛选并行

- 0803_13 e24 `52.841/59.322` 相对原始 decoder 同点双增益，联合 `+1.673`；相对 Encoder
  同点为 `+1.127/-0.197`。这足以把成熟曲线迁到最慢的 252 固定 GPU0/1 继续 e28+，但尚不
  是最终成功点，后续重点是 det 回补并保持 cls 优势。
- 跨机 UID 不同使旧 workdir 对 252 只读，且文件系统不支持 ACL。采用只读源 checkpoint 加
  252 自有新 workdir 的恢复方式，避免开放目录权限；epoch/iter/优化器状态从 e24 连续恢复，
  e25 iter50 已验证。
- 释放出的 178 单卡启动 0803_23：在终层 5D 乘积切空间中传输已建立的相对变换，只保留
  沿既有相对变换方向的 detail，抑制横向末层抖动。该结构零参数、帧交换等变，不改变 DN、
  loss 或计算深度；真实 smoke 与 formal iter50 通过，进入 e4/e8/e12。
- 99 的 shared semantic margins e8 `39.478/46.483` 仍低于原 decoder，但 e4→e8恢复明显，
  按慢收敛规则看到 e12；197 的 geometry+margin 同样继续。三条快线分别承担 transported
  tangent、semantic margin、geometry+margin 的正交筛选，252 只承接成熟曲线。

## 2026-08-04 05:55 CST：transported tangent 数值语义闭环

- 0803_23 首次 fresh 暴露的不是性能差距，而是极小 reference 下先 exp 再 clamp 会留下 NaN
  反向梯度。该无效尝试在 e1 iter650、无 epoch checkpoint 时停止，不等待 e4，也不进入任何
  性能比较；这不违反 decoder 不按 e4/e8 早停的约束。
- 修复仅把尺寸解码改写为数学等价的 log-domain clamp→exp，保持 transported-tangent 几何、
  零参数、class-agnostic、无 reweight 和计算深度不变。退化小框梯度测试、交换等变/DN 测试、
  零增量整模检查与真实 smoke 均通过。
- 新 `_finite_fresh` formal 已重新通过 iter50 五门槛并进入 e4/e8/e12。后续只使用该目录的
  checkpoint 和 TrackEval；旧目录仅作为错误签名与修复证据保留。

## 2026-08-04 06:05 CST：transported tangent 真实训练覆盖复核

- 修复版 0803_23 已到 epoch1 iter700，跨过旧实现 iter350 的首次非有限匹配代价位置并达到两倍
  覆盖；同类告警与致命错误均为 0，loss/grad `16.8861/229.6270`，总、DN、Encoder proposal
  分量全有限。
- 数值修复因此不再只由定向单测和四步 smoke 支撑；178 保持该 fresh 轨迹到 e4/e8/e12，性能
  判断仍等待完整 checkpoint、检测和 TrackEval，不能把数值有效等同于性能达标。

## 2026-08-04 06:20 CST：准备 transported shape-tangent 隔离对照

- 0803_24 是 0803_13 与 0803_23 之间的保守结构：中心完全保留逐帧 residual，只在终层
  log-size/周期角三维切空间保留 pair-common 更新，并沿 detached 前序相对 shape transform 传输
  detail。它检验 det 瓶颈是否来自绝对平均抹去既有相对尺寸/角度，而不引入中心耦合。
- 该操作零参数、交换等变、class-agnostic，无 reweight、新层、attention、loss 或额外主矩阵乘法。
  4 项定向测试、配置深拷贝、launcher `bash -n` 和零状态整模构建通过：`22,771,111` 参数、
  增量 0、711 tensors。178 隔离提交 `d470f96e` clean。
- 状态为 `PREPARED/NO_GPU`，不创建 workdir、不抢占 0803_23；现有 e12/e28 结果到齐后再决定它
  与 0803_21/22 的优先级，252 仍只承担成熟 0803_13 长轨迹。

## 2026-08-04 06:53 CST：semantic margin 成熟双负，transported margin 接替

- 0803_17 e12 为 cls/det HOTA `45.597/52.020`，相对原始 decoder 同点
  `-1.798/-2.416`，相对 Encoder 同点 `-4.083/-4.521`；pair mAP/AP50
  `0.238097/0.417747`，both-independent `0.282968/0.475183`。checkpoint、5416 条检测、
  50 序列、28 CSV、108 文件和异步完成标志齐全。
- e4/e8/e12 三个完整节点持续双负，因此精确停止 PGID `1357909` 并释放 GPU1/2；该决策满足
  慢收敛观察窗口，不是早期 epoch 否决。GPU0 外部任务完全不动。
- 0803_21 以沿前序 class-ranking 方向传输终层 centered-margin detail 替代完全平均。真实双卡
  四步 smoke 的 loss/grad、DN/Encoder 分量和 642 个 checkpoint 浮点 tensor 全有限；fresh
  formal PGID `1384944` 的 iter50 为 `0.9814 s/iter`、loss/grad `21.3915/104.8633`，五门槛
  通过。继续收集 e4/e8/e12，不把当前数值稳定等同于性能成功。
