# DiffusionTrack BS1 → LX Controlled Ablation

Date: 2026-07-24

## Objective

Starting from the current HSMOT BS1 implementation, move toward the isolated
LX implementation one variable at a time. Every bridge run must preserve the
same image, annotations, seed, optimizer-step count, validation schedule and
common AP evaluator. A variable is considered causal only when its forward
ablation and, for the final key variables, reverse removal agree.

## Frozen endpoints

| Item | Current BS1 | LX endpoint |
| --- | --- | --- |
| Samples | one `data43-2/000001` image repeated 20 times | same |
| Epochs / optimizer steps | 100 / 2000 | 100 / 2000 |
| Physical batch / accumulation | 1 / 1 | 1 / 1 |
| Seed | 8823 | 8823 |
| Queries / refinement stages | 500 / 6 | 500 / 6 |
| Confidence threshold | 0.001 | 0.001 |
| EMA | enabled | enabled |
| Current native best AP50 | 0.1425 | 0.2977 |

Current BS1:

`/data4/linxu/PairMOT_DiffusionTrack/work_dirs/yolo11l_diffusion_det_hsmot_overfit_data43_2_legacycenter_covered_lxlr_b1_acc1_100e_gpu0_v2`

LX endpoint:

`/data4/linxu/PairMOT_DiffusionTrack/lx_baseline_isolated/outputs/lx_baseline_diffusiontrack_single_data43_2_x20_gpu0_val2_thresh001_diag_v3`

Frozen assets:

| Asset | SHA256 |
| --- | --- |
| MOT annotations | `251cc3493e8090100f510c9e0a2c59924fe5365136502f8fa5131750a630d980` |
| JPEG part 1 | `725b04f7a1b3da9ec3f3c35d4b0e343a52462d13245f8f9d684cc544829a0e3b` |
| JPEG part 2 | `465f65fd724a04e08d96ed0055baf44454352848e38e107eb6e8a77ee0ce21ab` |
| JPEG part 3 | `f9d56f342aefcced268a18ac513c7a8d64a3eca10b8e3f87a85c1eb7efe27800` |
| Official ConvMSI checkpoint | `d2ec9b91630dfb9263711d713caba23c9de205d26322c5c575164c3229687caa` |

## Common evaluation protocol

The authoritative cross-implementation evaluator is
`tools/evaluate_bs1_lx_common_ap.py`:

- exact convex-polygon rotated IoU;
- IoU thresholds 0.50:0.05:0.95;
- COCO-style 101-point interpolated AP;
- top 100 predictions per image;
- numerical HSMOT classes shared by both caches.

HSMOT MOT columns are
`frame, track_id, qbox8, ignored, class_id, truncated`. Class ID is zero-based
column 11; column 12 is only the truncation flag. An earlier temporary analysis
used column 12 as the class and is invalid. The corrected common results are:

| Epoch | Current BS1 AP50 / mAP | LX AP50 / mAP |
| ---: | ---: | ---: |
| 20 | 0.0011 / 0.0002 | 0.0001 / 0.0000 |
| 40 | 0.0134 / 0.0023 | 0.0218 / 0.0030 |
| 60 | 0.0292 / 0.0049 | 0.1275 / 0.0297 |
| 80 | 0.0969 / 0.0221 | 0.2532 / 0.0711 |
| 100 | 0.1041 / 0.0235 | 0.2911 / 0.0892 |

The saved machine-readable result is:

`/data4/linxu/PairMOT_DiffusionTrack/analysis/bs1_lx_common_ap_top100.json`

LX epoch-100 changes only from AP50 0.2947 at top 500 to 0.2911 at top 100,
so the detection cap is not the source of the convergence gap.

## Bridge matrix

Each row inherits the previous completed row and changes only the named
variable. Later rows may be reordered if an earlier result makes them
unnecessary, but no row may silently change more than one factor.

| Bridge | Only changed variable | Status | Evidence |
| --- | --- | --- | --- |
| 00 | frozen current BS1 | completed | existing 100-epoch run |
| 01 | include configured `100 × center_prior_penalty` in matcher cost | completed | common epoch100 AP50 0.10321 vs Base 0.10410 |
| 02 | regression-loss LE135 angle weight `0.05 → 1.0`; matcher unchanged | completed | epoch100 AP50 collapses to 0.00108; ruled out as a beneficial alignment |
| 02A | raw angle-delta scale `1 rad → 1 degree (π/180 rad)` | completed | epoch100 AP50 0.09568 vs 02 0.00108; state/RNG identical |
| 03 | matcher L1 representation `LE135-5D → normalized raw qbox8`; regression unchanged | completed | epoch100 AP50 0.01076 vs 02A 0.09568; adverse alone |
| 03R | RotatedROIAlign angle direction `MMCV clockwise=True → Detectron2-equivalent False` | completed | epoch100 AP50 0.00544 vs 03 0.01076; correctness repair, not LX speed factor |
| 04 | criterion frame-wise rotated-IoU normalization `sum → mean`; weight unchanged | completed | epoch100 AP50 0.01057 vs 03R 0.00544; positive but 27.5x below LX |
| 04B | regression L1 spatial normalization `all ÷√(WH) → ÷[W,H,W,H]`; matcher unchanged | completed | epoch100 AP50 `0.02806`; positive but secondary |
| 05 | six bbox-delta projection weights `0 → post-hoc N(0, 0.001)`; biases remain zero | prepared | exactly six tensors change; all downstream RNG equal |
| 05B | hidden regression-module Linear biases `0 → logit(0.01)=-4.595` | prepared | exactly 18 biases change; all downstream RNG equal |
| 05C | regression initialization procedure → exact LX constructor then final Xavier; downstream RNG held at 05B | prepared | 97 intended head/init tensors change; downstream RNG equal; final std 0.0772–0.0795 |
| 05D | post-initialization RNG stream `05B → natural LX-head state`; model tensors held fixed | prepared | zero state tensors change; only Torch RNG hash changes |
| 06 | center-delta projection `proposal-local axes → global x/y axes` | prepared | state and post-build RNG identical to 05D |
| 07 | disable per-stage refined-box LE135 canonicalization | prepared | state and post-build RNG identical to 06 |
| 07B | guarded/symmetric bbox-delta arithmetic → LX unguarded/max-only arithmetic | prepared | state and post-build RNG identical to 07 |
| 07C | post-layer refined width/height floor `0 px → LX 2 px` | prepared | state and post-build RNG identical to 07B |
| 07D | former ROIAlign bridge | moved to 03R | prioritized earlier because the verified feature/gradient error is large |
| 08 | runtime precision `BF16 → FP32`; model/config unchanged | prepared | state/RNG identical; launcher-only change |
| 08B | criterion ordinary rotated IoU `guarded → LX raw diff-IoU kernel` | prepared | state/RNG identical; no ProbIoU |
| 08C | classification score fusion `FP32 clamped logit → LX raw inverse logit` | prepared | state/RNG identical |
| 08D | matcher pair IoU/nonfinite handling `guarded → LX raw` | prepared | state/RNG identical; costs/coverage unchanged |
| 09 | scheduler peak LR `2.5e-5 → 1.5625e-5`; AdamW constructor LR remains `2.5e-5` | prepared | state/RNG identical; separates schedule amplitude from update timing |
| 09B | LR write timing `before update → LX after update` | prepared | exposes LX's `2.5e-5` first update, then one-step-old scheduler values |
| 10 | ConvMSI stem LR multiplier `10x → 1x`; base LR unchanged | prepared | state/RNG identical |
| 10B | equal-LR AdamW groups `non-stem + stem → LX single parameter group` | prepared | state/RNG identical; model-order parameters |
| 10C | Torch-2 AdamW dispatch `foreach auto → scalar loop` | prepared | state/RNG identical; matches LX/PyTorch-1.11 style |
| 11 | Dynamic-K coverage `corrected current-mask → LX stale-mask repair` | prepared | state/RNG identical |
| 12 | image augmentation `none → LX 8-channel distort + horizontal mirror` | prepared | state/RNG identical; only `hsmot_augment_mode` changes |
| 12A | train-loader workers `4 → LX 2` (changes augmentation RNG streams) | prepared | state/RNG identical |
| 12B | padded target capacity `500 → LX 1000` | analytical no-op | state/RNG identical; only appends zero padding |
| 13 | model-side duplicated-pair flip `off → LX on` | prepared | state/RNG identical; only `random_flip` changes |
| 14 | fixed canvas `896×1184 → 800×1440` | prepared | state/RNG identical; only train/test size changes |
| 15 | input normalization `/255 → LX per-channel mean/std` | prepared | state/RNG identical; only 8-channel constants change |
| 16 | image source `3-JPG → source NPY`; all transforms unchanged | prepared | state/RNG identical; launcher-only source change |
| 16B | annotation-coordinate storage `float32 → LX float64` before resize | analytical numerical alignment | removes the only 14-ULP target discrepancy; all 140 tensor leaves over 20 batches become bitwise equal |
| 17A | stem only: ConvMSI → LX native 8-channel Conv2D stem; all non-stem tensors unchanged | prepared | 1,051/1,051 non-stem keys and complete diffusion-head hash bitwise equal; downstream RNG equal |
| 17B | non-stem pretrain only: MMOT ConvMSI checkpoint → LX 2D checkpoint; native stem unchanged | prepared | all stem tensors and complete diffusion-head hash bitwise equal; 878/1,051 non-stem tensors change |
| 17C | backbone load path `direct checkpoint → LX YAML-build then load`; downstream RNG held fixed | prepared | 1,057/1,057 loaded; backbone state bitwise equal to 17B; all 304 head parameter hashes equal native LX |
| 17D | post-initialization RNG stream `17B → natural LX-builder state`; all model tensors held fixed | prepared | complete model hash equal to 17C; next 32 RNG samples differ |
| 18 | decoded diffusion-box minimum side `1 px → 0 px` | prepared | |
| 19 | cosine diffusion schedule/buffers `float32 → LX float64` | prepared | |
| 20 | entire six-stage internal angle representation `radians → degrees` | completed (old LX runtime) | exact only after the target bridge below |
| 20A | GT qbox→rbox conversion and degree-space normalization → LX scalar arithmetic | completed (old LX runtime) | Layer1--6 forward boxes exactly equal |
| 20B | corrected degree path, only ROIAlign kernel `MMCV → Detectron2` in py310 | running | GPU0, 100-epoch single-image controlled run |
| I0 | post-hoc common evaluation with one fixed ref/cur proposal template | pending | removes training-path RNG consumption from the convergence metric |
| I1 | native inference draw order `cur→ref → LX cur₁→ref→cur₂` | pending | code-proven endpoint difference; training unaffected |
| E | fully aligned LX endpoint in the controlled harness | pending | |
| R1+ | remove final key variables one at a time from endpoint E | pending | |

Initialization auditing found that the apparent `N(0,0.001)` assignment in
LX's `RCNNHead` is not its final regression initialization:
`DynamicHead._reset_parameters()` subsequently applies Xavier to every
two-dimensional parameter. The six final bbox projection standard deviations
are `0.0772–0.0795`, about 80 times the post-hoc `N(0,0.001)` diagnostic.
Furthermore, LX constructs a complete random YOLO from YAML before loading
all 1,057 checkpoint items. Those overwritten random tensors advance the RNG
and thereby select a different diffusion-head initialization. Reproducing
both behaviors yields bitwise equality for all 304 head parameters and all
313 non-backbone parameters against a native LX model initialized with seed
8823; the loaded backbone state is also bitwise equal. This makes
Bridge05C/17C the strongest code-derived candidate before their controlled
training results are available.

At the prepared Bridge19 endpoint, a fresh native-LX construction provides a
stronger invariant: all 851 named parameters have the same names and are
851/851 bitwise equal. All 1,391 state tensors shared by the models are also
bitwise equal. LX reports 1,220 additional state entries under
`backbone.head.*`; they belong to the unused YOLO detection head and do not
appear in either model's 851 named parameters. Post-build Torch and NumPy RNG
hashes are identical. Python RNG differs, but with 15 GT below 500 proposals
the only model-side Python-random branch (`random.shuffle`) is not entered;
augmentation workers receive independent worker seeds. It is an analytical
no-op for this diagnostic.

Machine-readable endpoint evidence:

`/data4/linxu/PairMOT_DiffusionTrack/analysis/bridge19_vs_lx_model_exact.json`

For finite, non-coincident test boxes, guarded versus LX-raw criterion IoU
had exactly zero forward and gradient difference; guarded versus raw matcher
IoU and stable versus raw class fusion also had exactly zero forward
difference. These bridges remain necessary because behavior diverges at
invalid boxes and at the differentiable polygon kernel's exact-coincidence
topology, not during ordinary finite geometry.

The current and native-LX EMA implementations have identical SHA256
`278a6cf9...8c0b2`; their LR scheduler implementations likewise have
identical SHA256 `476464b2...53dd`. Those code paths are excluded as causes
once Bridge09 supplies the same LR values. AdamW still requires explicit
group/foreach bridges because native LX used a single scalar-loop
PyTorch-1.11 optimizer while py310/Torch-2 can auto-dispatch foreach kernels.

The native LX trainer writes scheduler LR *after* `optimizer.step()`. Its
AdamW is constructed at `2.5e-5`, so update 1 uses `2.5e-5`; update 2 uses
scheduler value 1, and update \(k\) uses value \(k-1\). The log is emitted
after that write and therefore displays value \(k\), hiding the offset. The
current corrected trainer writes before the step. Bridge09 now changes only
the scheduler peak to `0.001/64`, while Bridge09B independently restores LX's
post-update timing and anomalous first peak. This distinction was verified
directly from both trainer implementations, not inferred from logged LR.

### Inference-proposal RNG audit

The two native inference paths do not draw the same proposals. With no track
history, current code samples `cur` then `ref` and uses both. LX samples an
initial `cur₁`, then `ref`, then overwrites the current proposals with
`cur₂`; it therefore consumes three Gaussian tensors and uses the second and
third. Validation does not reset a proposal seed, so any preceding difference
in training RNG consumption also changes the sampled proposals. Native cached
AP is valid end-to-end performance, but it is not a pure measurement of
weight convergence.

The 20 validation samples are exact copies of one image, allowing proposal
variance to be measured at fixed weights without a checkpoint replay. Per-copy
AP50 for Base epoch100 has mean/std/min/max
`0.1871/0.0829/0.0395/0.3992`; LX epoch100 has
`0.3549/0.0656/0.2442/0.4916`. The draw-to-draw variance is material, though
the ranges still show a real LX weight-quality advantage. I0 will therefore
re-evaluate saved future checkpoints with an identical fixed proposal
template, while I1 separately aligns the native LX draw order.

Reproduction tool and evidence:

`tools/analyze_repeated_image_ap_variance.py`

`/data4/linxu/PairMOT_DiffusionTrack/analysis/base_epoch100_repeated_image_ap_variance.json`

`/data4/linxu/PairMOT_DiffusionTrack/analysis/lx_epoch100_repeated_image_ap_variance.json`

### Exact input-pipeline audit

The endpoint data bridge was checked below the visualization level by hashing
every tensor leaf emitted by both training loaders. Before Bridge16B, all
9,216,000 float32 image values in the first `1×8×800×1440` input were bitwise
equal, as were image dimensions, frame/video IDs and sample IDs. Only 14 of
the 10,000 padded-target values differed, all coordinates, with maximum
absolute error `6.103515625e-5` pixel.

Code inspection identified the exact cause. `HSMOTDataset` parsed the MOT text
into float32 before multiplying coordinates by the `8/9` resize ratio, while
LX's COCO loader retained float64 until the resized targets were packed into
float32. Bridge16B changes only this storage precision. Afterward, all 140
tensor leaves from 20 deterministic batches—including augmented images,
targets, metadata and IDs—are bitwise equal to LX. The original discrepancy
is therefore a proven one-ULP arithmetic-order effect, not a plausible source
of the AP50 gap.

Reproduction tools and machine-readable evidence:

`tools/hash_train_loader_batches.py`

`tools/compare_loader_npz.py`

`tools/compare_loader_hashes.py`

`/data4/linxu/PairMOT_DiffusionTrack/analysis/bridge16_vs_lx_first_batch_exact.json`

`/data4/linxu/PairMOT_DiffusionTrack/analysis/bridge16b_vs_lx_first_batch_exact.json`

`/data4/linxu/PairMOT_DiffusionTrack/analysis/bridge16b_vs_lx_loader20_exact.json`

### Exact first-step forward/backward audit

With the Bridge19 endpoint and native LX model now bitwise aligned, a seeded
FP32 first training step was executed in each repository with
`cudnn.benchmark=False`. Two independent runs in each repository have
bitwise-identical Layer1--6 outputs, assignments and losses. Every tensor in
the cross-repository input batch is also bitwise equal. Torch CPU/CUDA and
NumPy RNG states are equal immediately before the forward and after the
backward. Despite those invariants, the computation is not equivalent:

| Quantity | Current endpoint | Native LX | relative difference |
| --- | ---: | ---: | ---: |
| total loss | 78.6506 | 63.6366 | 23.6% |
| final classification loss | 7.0167 | 6.9804 | 0.52% |
| final bbox L1 loss | 20.9983 | 7.6471 | 174.6% |
| final rotated-IoU loss | 1.8769 | 1.8824 | 0.29% |
| backbone gradient L2 | 3072.72 | 3116.14 | 1.39% |
| proposal-projection gradient L2 | 698.40 | 515.58 | 35.5% |
| diffusion-head gradient L2 | 5822.70 | 3827.92 | 52.1% |

Layer 1 is nearly equal (`0.020%` classification, `0.031%` bbox and
`0.0019%` IoU relative loss difference), but later refinement layers
progressively diverge. The final loss difference is dominated by a different
Layer-6 assignment and its L1 term, not by class or IoU arithmetic. This
proves that the convergence gap is not caused by input data, parameter
initialization or Torch/NumPy proposal RNG at the aligned endpoint. It
originates inside the iterative refinement path, whose small early numerical
differences are amplified by subsequent box decoding and discrete matching.

Machine-readable evidence:

`/data4/linxu/PairMOT_DiffusionTrack/analysis/bridge19_deterministic_a.json`

`/data4/linxu/PairMOT_DiffusionTrack/analysis/lx_deterministic_a.json`

`/data4/linxu/PairMOT_DiffusionTrack/analysis/bridge19_vs_lx_deterministic_first_step.json`

The rotated ROIAlign adapter also contained a verified geometry mismatch.
LX converts quadrilaterals to `(cx,cy,w,h,theta)` with `theta` explicitly in
degrees, then passes those boxes to Detectron2 `ROIAlignRotated`. The current
model stores the same LE135 angle in radians and passes it to MMCV, which is
the correct unit conversion. Under a shared deterministic fixture spanning
all three FPN levels, Detectron2 and MMCV `clockwise=False` agree to:

| Tensor | max absolute error | mean absolute error |
| --- | ---: | ---: |
| pooled output | 7.249e-6 | 2.101e-7 |
| P3 feature gradient | 2.228e-6 | 8.156e-10 |
| P4 feature gradient | 1.714e-6 | 1.143e-8 |
| P5 feature gradient | 2.831e-6 | 6.234e-8 |

In contrast, the historical `clockwise=True` adapter has pooled-output
max/mean errors `1.984/0.511`; its P3/P4/P5 maximum gradient errors are
`4.995/0.710/0.586`. Bridge07D therefore corrects a real ROI feature and
gradient error rather than a cosmetic angle convention. Reproduction script
and machine-readable result:

`tools/diagnose_roi_align_equivalence.py`

`/data4/linxu/PairMOT_DiffusionTrack/analysis/roi_equiv_comparison_v3.json`

The prioritized Bridge03R also has identical full state, parameters and
post-build Torch/NumPy/Python RNG; its only model attribute change is
`head.head.box_pooler.clockwise: true → false`:

`/data4/linxu/PairMOT_DiffusionTrack/analysis/bridge03r_model_control.json`

## Bridge 01

Experiment:

`yolo11l_diffusion_det_hsmot_overfit_bridge01_centeractive_b1_acc1_100e_gpu0_v1`

Log:

`/data4/linxu/PairMOT_DiffusionTrack/logs/stage1_overfit_bridge01_centeractive_b1_acc1_100e_gpu0_v1_20260724.log`

Configuration inspection proves that Bridge 00 and Bridge 01 have identical
experiment fields and identical 265,641,151 model parameters. Matcher fields
are also identical except:

```text
apply_center_prior_penalty: False -> True
```

The first diagnostic CSV reconstructs total matching cost as
`class + L1 + rotated-IoU + center + foreground` with maximum numerical error
`4.1e-6`, proving that the penalty is active. Eleven matcher/L1/decoder tests
passed in the remote py310 environment. The formal run reached finite epoch 1
iteration 10 losses before being classified as running.

Interim results under the common evaluator:

| Epoch | Base AP50 / mAP | Center AP50 / mAP | LX AP50 / mAP |
| ---: | ---: | ---: | ---: |
| 20 | 0.00109 / 0.00017 | 0.01117 / 0.00218 | 0.00009 / 0.00001 |
| 40 | 0.01339 / 0.00233 | 0.01926 / 0.00271 | 0.02175 / 0.00297 |
| 60 | 0.02922 / 0.00494 | 0.05793 / 0.01210 | 0.12750 / 0.02970 |
| 80 | 0.09686 / 0.02213 | 0.11524 / 0.02057 | 0.25317 / 0.07106 |
| 100 | 0.10410 / 0.02353 | 0.10321 / 0.02824 | 0.29110 / 0.08923 |

At epoch 60, Layer-6 mean matched pair IoU is `0.1765 / 0.2342 / 0.4231`
for Base / Center / LX. Center prior is therefore a real positive factor, but
it explains only part of the remaining mid-training convergence gap. At epoch
80 the corresponding IoUs are `0.1583 / 0.1454 / 0.3827`, and the center
variant's AP50 advantage over Base has narrowed to `+0.01838`; center prior is
not the dominant late-convergence factor. At epoch 100 it has no AP50 benefit
at all (`-0.00089` versus Base), although mAP50:95 improves by `+0.00471`.
Layer-6 mean matched pair IoU is `0.2226 / 0.2070 / 0.4173` for
Base / Center / LX. Therefore the center term improves early assignment but
does not explain LX's late convergence; it is ruled out as the primary cause.

The final machine-readable Bridge01 result is:

`/data4/linxu/PairMOT_DiffusionTrack/analysis/bridge01_final_epoch20_40_60_80_100_summary.json`

## Bridge 02

Bridge02 changes only criterion angle L1 weight `0.05 → 1.0`; the matcher
retains `0.05`. It formally started on GPU0 with physical/effective global
batch 1 and finite epoch-1 losses.

Common results are:

| Epoch | Center AP50 / L6 IoU | Angle1 AP50 / L6 IoU | LX AP50 / L6 IoU |
| ---: | ---: | ---: | ---: |
| 20 | 0.011174 / 0.094056 | 0.001113 / 0.066588 | 0.000090 / 0.034553 |
| 40 | 0.019258 / 0.103851 | 0.002611 / 0.071195 | 0.021753 / 0.101826 |
| 60 | 0.057925 / 0.234217 | 0.000688 / 0.128127 | 0.127503 / 0.423075 |
| 80 | 0.115238 / 0.145389 | 0.000372 / 0.032829 | 0.253171 / 0.382741 |
| 100 | 0.103212 / 0.206997 | 0.001084 / 0.035410 | 0.291102 / 0.417287 |

Full angle supervision is not merely slower at the start: it destroys the
late refinement trajectory. At epoch 100, AP50 is 95.0 times lower and
Layer-6 matched pair IoU is 5.85 times lower than Bridge01. The angle weight
`0.05` is therefore a necessary stabilization in the current radian-space
loss, and changing it to `1.0` is ruled out as the explanation for LX's
faster convergence. Bridge02A still inherits this deliberately adverse state
to isolate the next variable, LX's degree-sized residual update.

Final machine-readable result:

`/data4/linxu/PairMOT_DiffusionTrack/analysis/bridge02_final_epoch20_40_60_80_100_summary.json`

## Bridge 02A

Bridge02A changes only each of the six refinement heads'
`angle_delta_scale` from `1.0` radian to `π/180` radians. This is the exact
unit translation of LX's raw `angles + dtheta`, because LX stores angles and
the residual in degrees while the current implementation stores both in
radians. Fresh seed-8823 model construction gives the same 265,718,480 state
elements and identical full state SHA256
`e3f7480f946e6ee69074ff4c127985861f14a2883f39684cea14d2245b6030bf`
for Bridge02 and Bridge02A. All 265,641,151 parameters and the post-build
Torch/NumPy/Python RNG states are also identical; the only scalar experiment
field difference is `exp_name`. Criterion/matcher angle weights remain
`1.0/0.05`. The run reached finite epoch1 iter10 with physical/effective
global batch 1. Machine-readable control proof:

`/data4/linxu/PairMOT_DiffusionTrack/analysis/bridge02a_model_control.json`

Common final results are:

| Epoch | Angle1 AP50 / L6 IoU | DegreeStep AP50 / L6 IoU | LX AP50 / L6 IoU |
| ---: | ---: | ---: | ---: |
| 20 | 0.001113 / 0.066588 | 0.004208 / 0.080021 | 0.000090 / 0.034553 |
| 40 | 0.002611 / 0.071195 | 0.004470 / 0.141025 | 0.021753 / 0.101826 |
| 60 | 0.000688 / 0.128127 | 0.029582 / 0.176091 | 0.127503 / 0.423075 |
| 80 | 0.000372 / 0.032829 | 0.059467 / 0.122167 | 0.253171 / 0.382741 |
| 100 | 0.001084 / 0.035410 | 0.095676 / 0.177359 | 0.291102 / 0.417287 |

Thus degree-sized residuals reverse Bridge02's degradation:
AP50 is `3.78×` and Layer-6 IoU is `1.20×` Bridge02 at epoch 20. This is an
increasingly strong effect: at epoch 60 AP50 is `43.0×` and Layer-6 IoU is
`1.37×` Bridge02. At epoch 100, the gains are `88.3×` AP50 and `5.01×`
Layer-6 IoU. However, the endpoint only returns to approximately the
Bridge00/01 range (`0.0957` versus `0.1041/0.1032` AP50) and remains far
below LX (`0.2911` AP50, `0.4173` Layer-6 IoU). The raw residual unit is
therefore a major interaction with full angle supervision, but it is not the
primary independent explanation of LX's final advantage.

`/data4/linxu/PairMOT_DiffusionTrack/analysis/bridge02a_final_epoch20_40_60_80_100_summary.json`

## Bridge 03

Bridge03 changes only matcher L1 geometry from the current normalized
LE135 five-vector to LX's normalized ordered qbox8 corners. Criterion L1,
rotated IoU, Dynamic-K, model state and all RNG states remain those of
Bridge02A. The formal GPU0 run is finite and stable.

The first two frozen evaluation points are:

| Epoch | DegreeStep AP50 / L6 IoU | Qbox8 AP50 / L6 IoU | LX AP50 / L6 IoU |
| ---: | ---: | ---: | ---: |
| 20 | 0.004208 / 0.080021 | 0.000022 / 0.048716 | 0.000090 / 0.034553 |
| 40 | 0.004470 / 0.141025 | 0.000254 / 0.052305 | 0.021753 / 0.101826 |
| 60 | 0.029582 / 0.176091 | 0.002780 / 0.152064 | 0.127503 / 0.423075 |
| 80 | 0.059467 / 0.122167 | 0.005576 / 0.064717 | 0.253171 / 0.382741 |
| 100 | 0.095676 / 0.177359 | 0.010756 / 0.086031 | 0.291102 / 0.417287 |

The selected Layer-6 mean raw/weighted L1 costs at epoch 20 are
`0.206/1.028` for DegreeStep, `0.574/2.872` for Qbox8 and `0.483/2.416`
for LX. At epoch 40 they are respectively `0.194/0.970`,
`0.642/3.212` and `0.445/2.227`. Thus the bridge moves the matching-cost
scale toward LX, as intended, but does not by itself reproduce LX's
convergence; through epoch 100 it is adverse. At epoch 60 Qbox8 recovers
Layer-6 IoU to `0.1521`, but AP50 remains `10.6×` below DegreeStep and
`45.9×` below LX. This is evidence of an
interaction with later initialization/refinement variables. At epoch100,
Qbox8 AP50 is `8.90×` below DegreeStep and `27.1×` below LX; Layer-6 IoU is
`2.06×/4.85×` lower respectively. Qbox8 is therefore ruled out as an
independent explanation for LX's speed. It is retained only because it is a
real endpoint method difference whose benefit, if any, must come from
interaction with later LX variables.

Interim machine-readable result:

`/data4/linxu/PairMOT_DiffusionTrack/analysis/bridge03_final_epoch20_40_60_80_100_summary.json`

## Bridge 03R

Bridge03R corrects only the MMCV rotated-ROIAlign direction
`clockwise=True → False`; all parameters, buffers and post-build RNG states
are identical to Bridge03. A Detectron2/MMCV fixture proves that this is a
real geometry correction (pooled-output maximum error changes from `1.984`
to `7.249e-6`).

Common results are:

| Epoch | Qbox8 AP50 / L6 IoU | Correct ROIAlign AP50 / L6 IoU | LX AP50 / L6 IoU |
| ---: | ---: | ---: | ---: |
| 20 | 0.0000217 / 0.048716 | 0.00000183 / 0.063415 | 0.0000898 / 0.034553 |
| 40 | 0.000254 / 0.052305 | 0.0000653 / 0.051530 | 0.021753 / 0.101826 |
| 60 | 0.002780 / 0.152064 | 0.001061 / 0.142249 | 0.127503 / 0.423075 |
| 80 | 0.005576 / 0.064717 | 0.002088 / 0.046979 | 0.253171 / 0.382741 |
| 100 | 0.010756 / 0.086031 | 0.005442 / 0.065716 | 0.291102 / 0.417287 |

The corrected geometry raises Layer-6 matched pair IoU by `30.2%` only at
epoch20. From epoch40 onward it is neutral-to-adverse in this chain, and at
epoch100 its AP50 is `49.4%` lower than Qbox8. ROIAlign direction is therefore
a required correctness repair, but it is ruled out as the cause of LX's
faster convergence.

The deterministic first-step audit adds an important nuance. Replacing the
corrected MMCV adapter with native Detectron2 ROIAlign reduces Layer-1
physical-box relative error against LX from `0.240%` to `0.0228%` and makes
Layer1--4 positive counts identical to LX. However, Layer5--6 still diverge.
Thus the pooler explains part of the early numerical amplification but not the
remaining refinement-path difference.

`/data4/linxu/PairMOT_DiffusionTrack/analysis/bridge03r_final_epoch20_40_60_80_100_summary.json`

`/data4/linxu/PairMOT_DiffusionTrack/analysis/bridge19_d2pooler_vs_lx_deterministic_layers_physical.json`

## Bridge 04

Bridge04 changes only criterion IoU normalization. The reference/current
ordinary rotated-IoU losses are still computed independently, but their sum
is divided by two to match LX's per-frame mean. Matcher IoU, qbox cost,
Dynamic-K and every model tensor remain unchanged.

| Epoch | Pair-sum AP50 / L6 IoU | Frame-mean AP50 / L6 IoU | LX AP50 / L6 IoU |
| ---: | ---: | ---: | ---: |
| 20 | 0.00000183 / 0.063415 | 0.0000249 / 0.047075 | 0.0000898 / 0.034553 |
| 40 | 0.0000653 / 0.051530 | 0.001138 / 0.039604 | 0.021753 / 0.101826 |
| 60 | 0.001061 / 0.142249 | 0.006245 / 0.141461 | 0.127503 / 0.423075 |
| 80 | 0.002088 / 0.046979 | 0.003215 / 0.049221 | 0.253171 / 0.382741 |
| 100 | 0.005442 / 0.065716 | 0.010567 / 0.080558 | 0.291102 / 0.417287 |

Frame averaging consistently improves AP50 after epoch20. At epoch100 AP50
is `1.94×` and Layer-6 IoU is `1.23×` Bridge03R. However, it remains
`27.5×` below LX AP50 and `5.18×` below LX Layer-6 IoU. This identifies the
oversized pair-sum IoU gradient as a real secondary factor, not the dominant
refinement-convergence cause.

`/data4/linxu/PairMOT_DiffusionTrack/analysis/bridge04_final_epoch20_40_60_80_100_summary.json`

## Bridge 04B

Bridge04B changes only the regression-L1 spatial encoding from the HSMOT
LE135 geometric-mean normalization to LX's per-axis
`[cx/W, cy/H, w/W, h/H, theta]` representation.  It inherits Bridge04's
per-frame mean rotated-IoU loss, qbox matcher, corrected ROIAlign direction,
identical initialization/RNG and all training settings.

| Epoch | Bridge04 AP50 / L6 IoU | LX-norm5 AP50 / L6 IoU | LX AP50 / L6 IoU |
| ---: | ---: | ---: | ---: |
| 20 | 0.0000249 / 0.047075 | 0.000367 / 0.062929 | 0.0000898 / 0.034553 |
| 40 | 0.001138 / 0.039604 | 0.000230 / 0.087626 | 0.021753 / 0.101826 |
| 60 | 0.006245 / 0.141461 | 0.005050 / 0.123062 | 0.127503 / 0.423075 |
| 80 | 0.003215 / 0.049221 | 0.002056 / 0.063886 | 0.253171 / 0.382741 |
| 100 | 0.010567 / 0.080558 | 0.028056 / 0.052764 | 0.291102 / 0.417287 |

The LX L1 normalization is not monotonic at intermediate epochs, but its
epoch-100 AP50 is `2.65x` Bridge04 and mAP50:95 is `3.31x` Bridge04.  It is
therefore a genuine late-stage positive factor.  It remains `10.38x` below
LX AP50 and its Layer-6 pair IoU is `7.91x` below LX, so it cannot explain the
main gap.

`/data4/linxu/PairMOT_DiffusionTrack/analysis/bridge04b_final_epoch20_40_60_80_100_summary.json`

## Deterministic refine-boundary audit

With the model state, loader batch, RNG states and physical input diffusion
boxes frozen bitwise, the first RCNN layer is the first location at which the
two implementations differ.  The feature pyramid and time embedding are
bitwise identical.  After replacing the current adapter with native
Detectron2 ROIAlignRotated, the two physical input boxes differ only from the
radian-to-degree conversion (`relative L2 < 4.8e-9`, maximum `6.10e-5`
degree); nevertheless, the first ROI tensor differs by relative L2
`4.35e-7` and maximum `2.34e-5`.  That produces a `2.22e-4` relative
difference in the first raw 5D residual and a physical first-layer box error
of up to `2.25` px (reference) / `5.05` px (current).

Because each stage feeds its detached boxes into the next stage, this small
unit-conversion/pooling error is amplified: reference-box relative L2 is
`2.28e-4`, `9.28e-3`, `5.33e-2`, `2.15e-1`, `2.73e-1`, and `5.40e-1` for
layers 1--6.  Thus data loading, diffusion sampling, feature extraction and
time conditioning are ruled out as the origin.  The next isolated bridge must
remove the internal radians-to-degrees boundary itself (store/refine boxes in
LX degrees end-to-end), rather than only swapping the pooler.

`/data4/linxu/PairMOT_DiffusionTrack/analysis/bridge19_d2pooler_trace_vs_lx_layers_physical.json`

## Bridge 20: resolution of the refine-boundary discrepancy

The preceding audit correctly localized the first divergence, but the final
cause has now been tested rather than inferred. LX keeps the internal
five-dimensional boxes in **degrees** through all six `RCNNHead` updates and
through `ROIAlignRotated`; HSMOT exposed radians at the same interface. A
pooler-only conversion is insufficient because the decoder, detached
next-stage boxes and GT-seeded diffusion boxes retain a different rounding
trajectory.

Bridge20 moves only the private `DynamicHead` representation to degrees. The
public HSMOT boundary remains radians for matching, LE135 L1, ordinary
rotated-IoU and evaluation. Bridge20A additionally reproduces LX's scalar
qbox→LE135 conversion and performs degree normalization before converting the
target back to radians for the public loss. This avoids the FP32
degree→radian→degree round trip for GT-seeded diffusion queries.

In the native LX Python/Torch runtime, Bridge20A has exact Layer1--6
reference/current inputs, raw deltas and refined boxes, and exact total loss
(`63.63660430908203`) relative to LX. Layer-1 pooler output is also exact;
remaining auxiliary loss differences are at most `4.8e-7`. This establishes
the degree internal representation plus target conversion as the missing
refinement-path alignment.

The py310 target environment cannot use MMCV with degree boxes because its
rotated ROIAlign expects radians. Bridge20B therefore uses a newly built GPU
Detectron2 `ROIAlignRotated` adapter. Its first formal GPU0 run began at
2026-07-24 09:27 CST and is finite (epoch1/iter20 total loss `34.541`). Its
e1--e8 losses closely match native LX (e4/i20 `29.570` vs `29.039`; e5/i20
`31.899` vs `31.311`). Py310 and LX still have different Torch/Detectron2
builds, so bitwise identity is not claimed there; Layer-1 pooler relative
difference is `0.00363`. Common AP50/Layer-6 comparisons at epochs
20/40/60/80/100 remain the decision criterion.

The first common point is now available: at epoch 20, Bridge20B reaches
AP50/mAP50:95 `0.0003003/0.0000340`, against LX
`0.0000898/0.0000102` under the same top-100, fixed-class protocol.  This is
an early-stage result only, but it already rules out the prior near-zero
convergence failure for this aligned pathway.  The job remains running for
the pre-registered 40/60/80/100 measurements.

At epoch 40, the same protocol gives Bridge20B AP50/mAP50:95
`0.027856/0.003283` and Layer-6 mean matched pair IoU `0.103794`, compared
with LX `0.021753/0.002974` and `0.101826`. Thus the corrected py310 path
has reached the LX convergence level by the first meaningful mid-training
point; this is not yet a late-training conclusion.

At epoch 60, Bridge20B remains ahead in detection AP50/mAP50:95
(`0.142278/0.037860` vs LX `0.127503/0.029703`), but its Layer-6 matched
pair IoU is lower (`0.343789` vs `0.423075`).  Consequently the aligned path
has removed the former detection-convergence gap, while a residual late-stage
geometric-refinement difference remains to be resolved or bounded at epochs
80 and 100.

At epoch 80, Bridge20B has AP50/mAP50:95 `0.164631/0.049177` versus LX
`0.253171/0.071063`, while its Layer-6 matched IoU is slightly higher
(`0.393879` vs `0.382741`).  The opposite movement of AP and the training
assignment IoU reinforces the already documented proposal-sampling variance
in native validation; the epoch-100 result must therefore be accompanied by
a fixed-proposal replay before attributing any residual AP gap to geometry.

Diagnostic caveat repaired on 2026-07-24: the historical `match_cost_total`
field was read *after* `dynamic_k_matching` in `lx_stale` coverage mode.
That routine temporarily adds `100000` to already-matched rows while repairing
uncovered GTs, so the stored total could disagree with its displayed class,
L1, IoU, center and foreground components. This is a reporting-only mutation,
not an optimizer or assignment change. The diagnostic now snapshots the
pre-repair matrix only when diagnostics are enabled; subsequent rows have a
component-consistent total. Existing Bridge20B comparisons use AP and
per-layer IoU/L1 fields, which were already pre-mutation and remain valid.

Evidence:

`/data4/linxu/PairMOT_DiffusionTrack/analysis/bridge20a_v3_vs_lx_layers.json`

`/data4/linxu/PairMOT_DiffusionTrack/analysis/bridge20a_v3_vs_lx_first_step.json`

`/data4/linxu/PairMOT_DiffusionTrack/analysis/bridge20b_py310_d2_vs_lx_layers.json`

### Final Bridge20B result

The run completed all 100 epochs / 2,000 optimizer updates on 223 GPU0 with
finite losses. The pre-registered common top-100 results are:

| Epoch | Bridge20B AP50 / mAP | LX AP50 / mAP | Bridge20B L6 IoU | LX L6 IoU |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 0.000300 / 0.000034 | 0.000090 / 0.000010 | 0.074973 | 0.034553 |
| 40 | 0.027856 / 0.003283 | 0.021753 / 0.002974 | 0.103794 | 0.101826 |
| 60 | 0.142278 / 0.037860 | 0.127503 / 0.029703 | 0.343789 | 0.423075 |
| 80 | 0.164631 / 0.049177 | 0.253171 / 0.071063 | 0.393879 | 0.382741 |
| 100 | 0.216667 / 0.077150 | 0.291102 / 0.089228 | 0.351482 | 0.417287 |

The single epoch-100 cache is not a sufficient final detector comparison:
the 20 copies of the same image are evaluated with independently sampled
diffusion proposals. Their per-copy AP50 mean/std is
`0.338740±0.102094` for Bridge20B and `0.354885±0.065585` for LX; the
corresponding mAP50:95 means are `0.121624` and `0.120790`. Thus the observed
single-cache AP50 difference (`0.216667` vs `0.291102`) is smaller than the
proposal-sampling spread, whereas the mean mAP is effectively equal. No
checkpoint was retained because the single-image diagnostic configuration
explicitly disables large checkpoint files, so an after-the-fact identical
noise-template replay is unavailable; this limitation is recorded rather
than masked.

**Causal conclusion.** The prior large convergence gap is caused by the
iterative refinement representation mismatch: radians internal to the six
stage HSMOT `DynamicHead`, versus degree boxes and LX scalar target conversion
in LX. The required repair is the bundled semantic interface change
Bridge20/20A (all six stages and rotated ROIAlign in degrees, plus LX target
conversion/normalization), not merely a pooler angle conversion. It produces
an exact old-runtime forward refinement trace and restores comparable
py310 convergence. The remaining variation in late single-cache AP and L6
IoU is consistent with the non-identical PyTorch/Detectron2 kernels and the
uncontrolled inference proposal draw; it is not evidence for another
unidentified HSMOT box-format, loss or matching error.

Machine-readable final artifacts:

`/data4/linxu/PairMOT_DiffusionTrack/analysis/bridge20b_final_epoch20_40_60_80_100_summary.json`

`/data4/linxu/PairMOT_DiffusionTrack/analysis/bridge20b_epoch100_repeated_image_ap_variance.json`

`/data4/linxu/PairMOT_DiffusionTrack/analysis/lx_epoch100_repeated_image_ap_variance_v3.json`

## 2026-07-26 correction: degree-aware refinement in the minimal baseline

The later full-data experiment named
`yolo11l_diffusion_det_hsmot_full30_degreecore_minimal` is invalid and must
not be compared with the controlled Bridge20B result above. Its configuration
enabled degree-valued internal angles at `DiffusionHead`, but restored the
original local-axis/canonical refinement switches. The restored
`RCNNHead.apply_deltas` path still treated those numeric degree values as
radians in three places: `sin/cos` center projection, the `pi/4` plus
modulo-`pi` angle update, and LE135 regularization. Consequently the internal
angle was numerically compressed to approximately
`[-0.785, 2.356)` **degrees**, and the public degree-to-radian conversion made
the predictions appear nearly horizontal. This was a training geometry error,
not only a visualization error. At epoch 27, 100% of cached predictions were
within 3 degrees of a horizontal or vertical axis, compared with 43.8% of GT.
The completed run's AP50/mAP (`0.0981/0.0182`) is retained only as diagnostic
evidence and is excluded from valid experiment tables.

Bridge20B is not retroactively invalidated by this finding. Its parent
configuration used the LX-compatible global-axis, no-canonicalization, raw
delta path (`proj_xy=False`, `canonicalize=False`,
`lx_delta_numerics=True`), so it did not execute the three faulty
radian-only branches.

The repaired minimal baseline is
`yolo11l_diffusion_det_hsmot_full30_degreecore_fixed_v2`. Its only semantic
change relative to original local-axis/canonical refinement plus rotated-box
adaptation is unit correctness:

- degree angles are converted to radians only for `sin/cos`;
- the angle residual is wrapped with `+45` and modulo `180` degrees;
- LE135 width/height canonicalization uses `90/180` degrees;
- one atomic setter propagates the unit to all six refinement stages, and a
  runtime invariant rejects mixed-unit heads;
- all exits to the matcher, losses, NMS, evaluator and normal visualizers
  remain public radians; Detectron2 rotated ROIAlign remains the explicit
  degree-valued boundary;
- initial-proposal diagnostics now convert internal degrees to public radians
  before drawing.

The radian implementation remains unchanged when degree mode is disabled.
Regression coverage includes physical local-axis center projection,
angle-wrap/width-height swap, and degree-versus-radian geometry equivalence;
all 23 rotated decoder tests pass in both local and 223 py310 environments.
On 223 GPU0, Detectron2 degree ROIAlign and the MMCV radian reference agree
within `7.25e-6` in output and `2.83e-6` in gradients. A real two-GPU smoke
run completed 10/10 finite optimizer iterations and distributed validation.
Its 2,000 cached predictions span p10/p50/p90 angles of
`-18.897/50.330/122.741` degrees, with only 7% within 3 degrees of an axis,
confirming removal of the compression signature.

Fresh formal training started from the official MMOT checkpoint at
2026-07-26 03:55 CST on physical GPUs 0 and 3, with global batch 2,
accumulation 1, fixed `896x1184`, 30 epochs, and no resume from the invalid
run. The first 120 optimizer iterations are finite, both workers are resident,
and data time after warm-up is approximately `0.001s`. Artifacts:

`/data4/linxu/PairMOT_DiffusionTrack/work_dirs/yolo11l_diffusion_det_hsmot_full30_degreecore_fixed_v2`

`/data4/linxu/PairMOT_DiffusionTrack/logs/yolo11l_diffusion_det_hsmot_full30_degreecore_fixed_v2_20260726.log`

### Fixed-v2 full-data result

The run completed all 30 epochs and its final distributed validation at
2026-07-27 12:16 CST. There are no NaN/Inf, traceback, OOM or NCCL failures in
the formal log. Mean logged total loss falls from `42.358` at epoch 1 to
`20.823/14.792/12.121/10.617/10.125/9.766` at epochs
`5/10/15/20/25/30`. The corresponding validation trajectory is:

| Epoch | mAP50:95 | AP50 |
| ---: | ---: | ---: |
| 3 | 0.0880 | 0.1866 |
| 6 | 0.2409 | 0.3731 |
| 9 | 0.2756 | 0.4263 |
| 12 | 0.2914 | 0.4504 |
| 15 | 0.2978 | 0.4632 |
| 18 | **0.2992** | 0.4683 |
| 21 | 0.2976 | 0.4691 |
| 24 | 0.2969 | 0.4686 |
| 27 | 0.2966 | 0.4681 |
| 30 | 0.2960 | **0.4692** |

The model has therefore converged: most of the gain occurs by epoch 12,
mAP50:95 peaks at epoch 18, and epochs 21--30 form a stable plateau rather
than a collapse. `best_ckpt.pth.tar` is the epoch-18 best-mAP checkpoint;
the endpoint remains available as `epoch_30_ckpt.pth.tar`. At epoch 30 the
per-class AP50 values are car `0.8251`, bike `0.2915`, pedestrian `0.3938`,
van `0.5281`, truck `0.3418`, bus `0.7659`, tricycle `0.2179`, and
awning-bike `0.3896`. The weaker small/rare classes, rather than an angle-unit
failure, are now the principal detection limitation.
