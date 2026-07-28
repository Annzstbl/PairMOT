# PairMOT Paper Mainline Experiment Report

更新时间：2026-07-28 11:17 CST

## 1. 实验目标

本报告记录论文主线的正式可复现实验。目标是在完全相同的数据、初始化、训练精度、随机种子和评测协议下，逐步验证 Liquid、Encoder、Decoder 的增益，并将 Proposal affinity 作为无需重新训练的 zero-shot 推理模块单独验证。

主线顺序：

1. `Base`；
2. `Base + Liquid`；
3. `Base + Liquid + Encoder`；
4. 第 3 步 checkpoint 加入 zero-shot Proposal；
5. `Base + Liquid + Encoder + Decoder`，fresh train；
6. 第 5 步 checkpoint 加入 zero-shot Proposal，形成完整模型。

第 4 步不进入第 5 步训练。否则 Proposal 将不再是 zero-shot 模块，无法保持论文中的方法定义。

## 2. 固定实验协议

| 项目 | 正式设置 |
| --- | --- |
| 消融骨干 | RT-DETR R18vd，3-layer decoder |
| 初始化 | 官方 COCO-only R18 checkpoint，经统一 PairMOT adapter 适配 |
| 数据 | HSMOT full train 75 sequences / 8297 unique adjacent pairs；test 50 sequences / 5416 pairs |
| 输入 | 原始 `1200x900`，`keep_ratio=True`，不做非等比拉伸 |
| Pair | train/test 均枚举唯一时间正序 `t-1 -> t`、gap=1；两帧共享 resize/flip/rotate |
| GMC | 真实 sparse-LK + RANSAC cache，train 8297 个唯一相邻变换；缺失即报错，禁止 identity fallback |
| Batch | 2 GPUs x 4 images/pairs，global batch 8 |
| 训练 | fresh 72 epochs，AdamW，base LR `1e-4`，LinearLR warmup 2000 iter |
| 精度 | BF16 through shared encoder，encoder 输出后转 FP32；decoder/head/matcher/loss 为 FP32 |
| DDP | `find_unused_parameters=False` |
| 随机性 | global seed 3407，pair sampler seed 3407，`deterministic=False` |
| 验证 | 每 4 epochs，18 个完整 validation + TrackEval 点 |
| 可视化 | 仅关闭绘图；detection evaluation 和 TrackEval 全部保留 |
| 停止规则 | 禁止 early stopping；所有训练实验完整运行 72 epochs |

所有需要训练的模块实验必须从同一个 COCO-only adapted checkpoint fresh train，禁止从前一阶段 checkpoint resume。除被消融模块及其新参数学习率外，不允许改变训练协议。

## 3. 模型定义

`Base` 固定使用 `0704_01` 的最终 PairMOT 结构：track-union all-GT、dual classification、no presence head、unique pair top-k、PairDN，以及 pair proposal supervision。Baseline 不包含 Liquid、temporal/pyramid encoder、elliptical-spectral Proposal、tristate decoder 或 long-tail 专用模块。

COCO 80类分类头不加载；HSMOT 8类 prev/curr 分类头重新初始化。RGB stem 权重原样重排至 Conv3D spectral kernel。单帧 cross-attention 同时初始化 prev/curr 分支；水平框前四维复制到旋转框 head，angle 维保持目标初始化。Pair-only 层使用模型定义的确定性初始化。

## 4. 实验矩阵

| 阶段 | 训练 | 唯一模型变化 | 作用 |
| --- | --- | --- | --- |
| A Base | 是 | 无 | 论文统一基准 |
| B +Liquid | 是 | 最终 Liquid sampler/fusion | 验证谱段动态建模 |
| C +Encoder | 是 | 最终 temporal/pyramid encoder | 验证跨帧多尺度表征 |
| C-P +Proposal | 否 | size-aware elliptical motion + spectral affinity | 验证 zero-shot proposal affinity |
| D +Decoder | 是 | 最终 pair decoder | 验证解码阶段交互 |
| D-P Full | 否 | 在 D checkpoint 上启用 Proposal | 完整 R18 方法 |

为增强因果归因，主线完成后至少补充 `Base + Encoder` 和 `Base + Decoder` 两项独立消融。Long-tail 作为解决 cls HOTA 的正交分支，先报告 `Base + Long-tail`，验证稳定后再加入完整模型，不混入主链基础结论。

## 5. 指标与选择规则

每个训练实验只按 `cls_HOTA + det_HOTA` 在全部 18 个评测点中选取唯一最佳 epoch。表格必须分别展示 `cls_HOTA` 与 `det_HOTA`，禁止合并成一个指标展示或跨 epoch 拼接。

同时报告该 epoch 的 cls/det MOTA、IDF1、pair mAP 和 pair AP50。AP 不参与
checkpoint 选择，也不允许另选 AP 最优 epoch；所有论文指标必须来自
`cls_HOTA + det_HOTA` 所确定的同一个唯一 epoch。最终论文还需记录参数量、FLOPs、
训练显存、训练吞吐和推理速度。

## 6. 当前进展

| ID | 实验 | 服务器/GPU | 配置 | 状态 |
| --- | --- | --- | --- | --- |
| `0716_02` | Paper Base R18 COCO full 1200x900 BF16 | 99 / GPU 0,1 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_coco_full_1200x900_bf16_reboot_fresh_99.py` | 已完成72 epochs和18/18异步TrackEval；唯一最佳为epoch 68 |
| `0716_04` | Paper Base + Liquid group-set-unique R18 COCO full 1200x900 BF16 | 197 / GPU 0,3 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_plus_liquid_groupsetunique_coco_full_1200x900_bf16_197.py` | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 72，HOTA和115.749，超过Base最佳0.453 |
| `0716_05` | Paper Base + Liquid group-set-unique + Encoder R18 COCO full 1200x900 BF16 | 252 / GPU 0,1 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_plus_liquid_groupsetunique_encoder_coco_full_1200x900_bf16_252.py` | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `54.635/61.488`，同epoch pair mAP/AP50为`0.3215/0.5467` |
| `0717_01` | Parallel Liquid Set-Transport structural candidate | AutoDL / GPU 0,1 | `autodl_0717_01_paper_liquid_settransport_full_1200x900_bf16.py` | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 68，HOTA和116.777，相对`0716_04`的cls/det HOTA分别提升0.828/0.200 |
| `0717_02` | Paper Base + original-hard Liquid strict control | 99 / GPU 0,1 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_plus_liquid_originalhard_coco_full_1200x900_bf16_99.py` | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 72，cls HOTA 54.335、det HOTA 61.445 |
| `0717_03` | Hard-sampled soft-context Liquid candidate | 197 / GPU 4,5 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_plus_liquid_hardsoftcontext_coco_full_1200x900_bf16_197.py` | 已按新设计主动停止于epoch 15 iter 150；epoch 12 checkpoint保留但不resume |
| `0718_01` | Independent-group Difference/Product Liquid candidate | 197 / GPU 4,5 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_plus_liquid_independent_diffproduct_coco_full_1200x900_bf16_197.py` | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 64为cls/det HOTA `55.276/61.812`，另报告epoch 72 `55.002/61.993` |
| `0718_02` | Anchor-Residual Competitive Liquid candidate | AutoDL / GPU 0,1 | `autodl_0718_02_paper_liquid_anchorcompetitive_full_1200x900_bf16.py` | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 64为cls/det HOTA `53.357/61.054`，共享盘归档和finalizer均完成 |
| `0718_03` | Evidence-consistent adaptive-anchor ARCR Liquid candidate | 99 / GPU 0,1 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_plus_liquid_anchorcompetitive_adaptiveanchor_coco_full_1200x900_bf16_99.py` | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 60为cls/det HOTA `54.689/60.969`，route未坍塌但det低于Base |
| `0718_04` | Scale-Adaptive Shared Sparse Evidence Liquid candidate | 252 / GPU 0,1 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_adaptiveanchor_sase_coco_full_1200x900_bf16_252.py` | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 68为cls/det HOTA `53.398/60.928`，低于Base及父实验0718_03 |
| `0718_05` | Adaptive-anchor + PCDP local-detail Liquid candidate | 99 / GPU 0,1 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_adaptiveanchor_pcdp_coco_full_1200x900_bf16_99.py` | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `54.142/61.317` |
| `0718_06` | Task-Preserving Adaptive Set Router Liquid candidate | 197 / GPU 4,5 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_cpas_settransport_coco_full_1200x900_bf16_197.py` | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `54.877/61.274` |
| `0719_01` | Pair-Consensus Relaxed-Set Liquid + PACDE | AutoDL / GPU 0,1 | `autodl_0719_01_paper_liquid_pairconsensus_relaxedset_full_1200x900_bf16.py` | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `53.883/61.411`；finalizer因无deploy key跳过push，共享盘完整结果已取回 |
| `0719_02` | Reliability-Weighted Pair-Consensus + PACDE | 178 / GPU 0 | `o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_liquid_pairconsensus_reliability_coco_full_1200x900_bf16_178.py` | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 64为cls/det HOTA `55.190/60.689`，另报告epoch 72 `54.933/60.912` |
| `0719_03` | Pair-Consensus Relaxed-Set Liquid, PACDE strict ablation | 252 / GPU 0,1 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_pairconsensus_relaxedset_nopacde_coco_full_1200x900_bf16_252.py` | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `54.209/61.504` |
| `0719_04` | Historical Wide-LAF + GroupMod replay | 197 / GPU 4,5 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_widelaf_groupmod_coco_full_1200x900_bf16_197.py` | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `53.932/60.908` |
| `0719_05` | Paper Base reproducibility rerun | 178 / GPU 0 | `o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_base_rerun_coco_full_1200x900_bf16_178.py` | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `52.417/61.265` |
| `0719_06` | Paper Base + long-tail positive reweight | 99 / GPU 0,1 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_longtail_reweight_coco_full_1200x900_bf16_99.py` | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `54.087/62.336`，两个HOTA均超过Paper Base |
| `0720_01` | Gate-Mass Quality-Conserving Liquid | 252 / GPU 0,1 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_diffproduct_qc_gatemass_coco_full_1200x900_bf16_252.py` | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为`54.103/61.581`，det低于Base |
| `0720_02` | Response-Mass Quality-Conserving Liquid | 197 / GPU 4,5 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_diffproduct_qc_responsemass_coco_full_1200x900_bf16_197.py` | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为`54.463/61.437`，det低于Base |
| `0720_03` | Dual-Moment Quality-Conserving Liquid | 99 / GPU 0,1 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_diffproduct_qc_dualmoment_coco_full_1200x900_bf16_99.py` | 训练准确性修复后fresh完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为`54.650/61.547` |
| `0721_01` | Response-Mass Quality-Conserving Liquid 1x8 rerun | 178 / GPU 0 | `o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_liquid_diffproduct_qc_responsemass_coco_full_1200x900_bf16_178.py` | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为`54.006/60.934` |
| `0721_02` | Accuracy-fixed `0718_01` strict rerun | 197 / GPU 4,5 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_independent_diffproduct_accuracyfix_coco_full_1200x900_bf16_197.py` | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为`54.327/61.659` |
| `0721_03` | BSR-Liquid + corrected Negative DN | 178 / GPU 0 | `o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_liquid_bsr_diffproduct_accuracyfix_dnnegativefix_coco_full_1200x900_bf16_178.py` | 运行中；epoch 64，15/18 TrackEval；阶段最佳epoch 60为`52.943/60.739`，不进入正式结果表 |
| `0721_04` | BSAC-Liquid | 252 / GPU 0,1 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_bsac_diffproduct_accuracyfix_coco_full_1200x900_bf16_252.py` | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为`54.530/61.528` |
| `0721_05` | DSE-Liquid | 99 / GPU 0,1 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_dse_diffproduct_accuracyfix_coco_full_1200x900_bf16_99.py` | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为`54.635/61.895` |
| `0722_01` | `0721_02` + corrected Negative-DN outer band and group mask | 197 / GPU 4,5 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_independent_diffproduct_dnnegativefix_coco_full_1200x900_bf16_197.py` | 运行中；epoch 7，1/18 TrackEval；同时改动噪声和attention mask，不是单变量消融 |
| `0723_01` | Pair-coherent DN noise + revised PairDN geometry/L1 | 99 / GPU 2,3 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_independent_diffproduct_pairdn_paircoherent_le180_coco_full_1200x900_bf16_99.py` | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 64为`53.955/62.032`，相对Paper Base双提升`+0.641/+0.050` |
| `0723_02` | Independent pair-side DN noise strict ablation | 252 / GPU 0,1 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_independent_diffproduct_pairdn_independent_le180_coco_full_1200x900_bf16_252.py` | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为`53.637/61.679`，相对Paper Base为`+0.323/-0.303` |
| `0723_03` | Pair-coherent PairDN + DSE | 197 / GPU 4,5 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_pairdn_paircoherent_le180_dse_coco_full_1200x900_bf16_197.py` | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 68为`55.036/61.745`，cls显著提高但det仍低于Paper Base |
| `0723_04` | Pair-coherent PairDN + CSPR | 178 / GPU 0 | `o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_liquid_pairdn_paircoherent_le180_cspr_coco_full_1200x900_bf16_178.py` | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为`54.523/61.738`，相对Paper Base为`+1.209/-0.244` |
| `0723_05` | Pair-coherent PairDN + local CP-DSE | 99 / GPU 2,3 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_pairdn_paircoherent_le180_cpdse_local_coco_full_1200x900_bf16_99.py` | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 68为`53.536/61.619`，相对Paper Base为`+0.222/-0.363`；99随后按要求让出 |
| `0723_06` | Pair-coherent PairDN + pair-global CP-DSE | 197 / GPU 4,5 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_pairdn_paircoherent_le180_cpdse_pairglobal_coco_full_1200x900_bf16_197.py` | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为`54.124/61.914`，相对Paper Base为`+0.810/-0.068` |
| `0723_07` | Pair-coherent PairDN + PECG | 252 / GPU 0,1 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_pairdn_paircoherent_le180_pecg_coco_full_1200x900_bf16_252.py` | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 52为`53.693/61.151`，相对Paper Base为`+0.379/-0.831` |
| `0723_08` | Pair-coherent PairDN + SCPD | 178 / GPU 0 | `o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_liquid_pairdn_paircoherent_le180_scpd_coco_full_1200x900_bf16_178.py` | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为`54.465/61.213`，相对Paper Base为`+1.151/-0.769` |
| `0725_01` | Pair-coherent PairDN + DSE + pair-global CP-DSE | 197 / GPU 4,5 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_pairdn_paircoherent_le180_dse_cpdse_pairglobal_coco_full_1200x900_bf16_197.py` | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为`55.126/61.998`，相对Paper Base双提升`+1.812/+0.016` |
| `0725_02` | Pair-coherent PairDN + DSE + centered pair-global CP-DSE | 178 / GPU 0 | `o2_pair_rtdetr_r18vd_1xb4acc2_72e_hsmot_paper_liquid_pairdn_paircoherent_le180_dse_cpdse_centered_coco_full_1200x900_bf16_178.py` | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 68为cls HOTA `53.730`、det HOTA `61.864`，相对Paper Base为`+0.416/-0.118` |
| `0725_03` | Pair-coherent PairDN + DSE + detection-tangent pair-global CP-DSE | 252 / GPU 0,1 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_pairdn_paircoherent_le180_dse_cpdse_dettangent_coco_full_1200x900_bf16_252.py` | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为`54.229/61.808`，相对Paper Base为`+0.915/-0.174`，相对父配置`0725_01`为`-0.897/-0.190` |
| `0726_01` | Pair-coherent PairDN + DSE + sparse-reserve pair-global CP-DSE | 197 / GPU 4,5 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_pairdn_paircoherent_le180_dse_cpdse_sparsereserve_coco_full_1200x900_bf16_197.py` | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 68为`54.230/61.634`，相对父配置`0725_01`为`-0.896/-0.364`，终止该方向 |
| `0726_02` | Base + Liquid + `0705_01` Encoder | AutoDL / GPU 0 | `autodl_0726_02_paper_base_liquid_encoder_p5temporal_pyramidlocal_pairdn_paircoherent_le180_full_1200x900_bf16_1xb8.py` | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为`54.742/61.631`，同点pair mAP/AP50为`0.3223/0.5429`。相对直接父配置`0723_01`最佳点为`+0.787/-0.401`，未达到Encoder双提升目标 |
| `0726_03` | Base + Liquid + P5 temporal + common-detail pyramid encoder | 252 / GPU 0,1 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_commondetail_pairdn_paircoherent_le180_coco_full_1200x900_bf16_252.py` | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为`54.654/62.240`，同epoch pair mAP/AP50为`0.3239/0.5380`。相对Base+Liquid双提升`+0.699/+0.208`；det DetA/AssA也同时提高`+0.039/+0.447` |
| `0727_01` | Base + Liquid + P5 temporal + dual-evidence pyramid encoder | 178 / GPU 0 | `o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_base_liquid_encoder_p5temporal_dualevidence_pairdn_paircoherent_le180_coco_full_1200x900_bf16_178.py` | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为`54.437/62.393`，同epoch pair mAP/AP50为`0.3201/0.5371`。相对Base+Liquid双提升`+0.482/+0.361`，相对Paper Base双提升`+1.123/+0.411`；det DetA/AssA同时提高`+0.526/+0.311` |
| `0727_02` | Base + Liquid + P5 temporal + spatial dual-evidence pyramid encoder | 197 / GPU 4,5 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_spatialevidence_pairdn_paircoherent_le180_coco_full_1200x900_bf16_197.py` | epoch 12后主动停止；该点为`47.411/54.593`，相对同epoch Base+Liquid双降`-0.676/-1.042`，det DetA/AssA分别下降`1.052/0.829`，pair mAP下降`0.0108` |
| `0727_03` | Base + Liquid + P5 temporal + scale-split dual-evidence encoder | 178 / GPU 0 | `o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_base_liquid_encoder_p5temporal_scalesplit_pairdn_paircoherent_le180_coco_full_1200x900_bf16_178.py` | 2026-07-27 02:40撤销等待队列；未运行smoke或正式训练。epoch 8证据表明移除P3 detail没有针对common残差的关联代价 |
| `0727_04` | Base + Liquid + P5 temporal + detail-energy-conserved common/detail encoder | 252 / GPU 0,1 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_detailenergy_pairdn_paircoherent_le180_coco_full_1200x900_bf16_252.py` | 2026-07-28 00:50 fresh启动；按输入pair-detail逐通道RMS限制signed-detail更新能量，不增加参数或loss。当前epoch 28、6/18 TrackEval |
| `0727_05` | Base + Liquid + P5 temporal + spatial-reliability detail-only encoder | 178 / GPU 0 | `o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_base_liquid_encoder_p5temporal_spatialdetail_pairdn_paircoherent_le180_coco_full_1200x900_bf16_178.py` | 已完成20项本地/178测试、完整构建及哈希；只增加57参数，以归一化局部common/detail能量调制保持pair均值的signed detail，并验证初始逐元素等价父结构。当前仅作已准备的保守回退，不占GPU |
| `0727_06` | Base + Liquid + P5 temporal + shared-scalar common evidence + signed detail | 178 / GPU 0 | `o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_base_liquid_encoder_p5temporal_sharedscalar_pairdn_paircoherent_le180_coco_full_1200x900_bf16_178.py` | 03:47撤销队列，未运行smoke或训练；epoch 12证据表明channel-mixing common已无AssA代价，改为shared scalar的设计依据消失 |
| `0727_07` | Base + Liquid + P5 temporal + spatial dual-evidence + branch-energy trust region | 197 / GPU 4,5 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_spatialbranchtrust_pairdn_paircoherent_le180_coco_full_1200x900_bf16_197.py` | 2026-07-27 05:24撤销队列，未运行smoke或正式训练；`0727_02` epoch 12已证明失败源于common空间门抑制检测覆盖，branch-energy上限没有针对该根因 |
| `0727_08` | Base + Liquid + P5 temporal + dual-evidence + branch-energy trust region | 178 / GPU 0 | `o2_pair_rtdetr_r18vd_1xb8_72e_hsmot_paper_base_liquid_encoder_p5temporal_dualbranchtrust_pairdn_paircoherent_le180_coco_full_1200x900_bf16_178.py` | 严格队列和正式训练均正常；按用户指令停在epoch 3 iter 750，未生成正式epoch checkpoint，不resume且不进入结果表 |
| `0727_12` | Base + Liquid + P5 temporal + Dual-Evidence + cross-scale evidence budget | 99 / GPU 0,1 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_crossscalebudget_pairdn_paircoherent_le180_coco_full_1200x900_bf16_99.py` | 2026-07-27 20:18 fresh启动；基于`0727_01`用32维尺度token联合预测P3/P4/P5逐通道common/detail预算，尺度维softmax保证每分支/通道总预算为3。新增37,696参数（`+0.166%`），无额外loss或高分辨率卷积；当前epoch 49、12/18 TrackEval |
| `0727_11` | Base + Liquid + P5 temporal + moment-competitive Dual-Evidence | AutoDL / GPU 0 | `autodl_0727_11_paper_base_liquid_encoder_p5temporal_momentcompetitive_pairdn_paircoherent_le180_full_1200x900_bf16_1xb8.py` | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 68为cls/det HOTA `54.853/62.050`，同epoch pair mAP/AP50为`0.3190/0.5384`，HOTA和`116.903`。相对Base+Liquid为`+0.898/+0.018`；相对`0727_01`为`+0.416/-0.343`，HOTA和仅高`0.073`。共享盘归档完成，权威结果已取回`autodl/results/0727_11` |
| `0727_10` | Base + Liquid + P5 temporal + detached mean-preserving detail redistribution | 197 / GPU 4,5 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_detailredistribute_pairdn_paircoherent_le180_coco_full_1200x900_bf16_197.py` | 2026-07-27 12:14撤销队列，未运行smoke或正式训练；其空间门父配置`0727_09`在epoch 16已相对Base+Liquid双降，故不再派生 |
| `0727_09` | Base + Liquid + P5 temporal + Dual-Evidence + detail-only spatial reliability | 197 / GPU 4,5 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_detailspatial_pairdn_paircoherent_le180_coco_full_1200x900_bf16_197.py` | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为`54.106/62.321`，同epoch pair mAP/AP50为`0.3122/0.5273`。相对`0727_01`为`-0.331/-0.072`，未明显超过 |
| `0727_10` | Base + Liquid + P5 temporal + Dual-Evidence + detached mean-preserving detail redistribution | 197 / GPU 4,5 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_detailredistribute_pairdn_paircoherent_le180_coco_full_1200x900_bf16_197.py` | 不增加参数/loss/阈值；切断空间描述到共享特征的旁路梯度，并约束detail空间门均值为1。26项本地/197测试、完整构建及精确哈希通过，2026-07-27 08:10进入严格队列 |
| `0728_01` | Base + Liquid + P5 temporal + Dual-Evidence Encoder + `0708_03` Decoder | 197 / GPU 4,5 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_liquid_encoder_p5temporal_dualevidence_decoder0708_03_pairdn_paircoherent_le180_coco_full_1200x900_bf16_197.py` | 严格继承`0727_01`，只打开tri-state decoder与零初始化frame-pointer循环耦合，保持separate FFN关闭。修复旧实现中结构性无梯度参数后，18项单测和双卡4-iter真数据smoke通过；2026-07-28 09:30 fresh启动，epoch 1 iter 50为`0.9245 s/iter`且全部关键loss/grad有限 |

当前运行工作目录：

- 99：`/data4/litianhao/PairMmot/workdir_99/0727_12_paper_base_liquid_encoder_p5temporal_crossscalebudget_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_orderedpairs_fresh`
- 197：`/data4/litianhao/PairMmot/workdir_197/0727_09_paper_base_liquid_encoder_p5temporal_detailspatial_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_orderedpairs_fresh`
- 197条件后继：`/data4/litianhao/PairMmot/workdir_197/0728_01_paper_base_liquid_encoder_p5temporal_dualevidence_decoder0708_03_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_orderedpairs_fresh`
- 252：`/data4/litianhao/PairMmot/workdir_252/0727_04_paper_base_liquid_encoder_p5temporal_detailenergy_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_orderedpairs_fresh`
- 178：当前无PairMOT训练；`0727_01`已完成，`0727_08`已按指令停止
- AutoDL：`/root/autodl-tmp/work_dirs/0727_11_paper_base_liquid_encoder_p5temporal_momentcompetitive_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_orderedpairs_autodl_1xb8_fresh`

AutoDL的`0726_02`已于2026-07-27 09:23完成并归档到
`/autodl-fs/data/PairMOT_results/0726_02`；权威Markdown/JSON副本位于
`autodl/results/0726_02`。finalizer因实例未配置GitHub deploy key跳过自动发布，但结果、
epoch 72、日志和完整TrackEval归档均已保存。

## 7. 结果表

仅纳入已完成72 epochs且18/18 TrackEval齐全的实验。正式行按唯一最大
`cls_HOTA + det_HOTA`选择；最佳点不是epoch 72时，紧随其后报告固定末轮参考，但末轮不参与
模型选点。

| 实验 | 父配置与主要改动 | epoch | cls HOTA | det HOTA | cls MOTA | cls IDF1 | det MOTA | det IDF1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A Paper Base `0716_02` | 无父实验；COCO-R18适配、full HSMOT、1200x900 ordered pair基线 | 68 | 53.314 | 61.982 | 44.690 | 62.218 | 60.599 | 72.303 |
| A `0716_02`（末轮参考） | 同上；固定epoch 72，不参与选点 | 72 | 53.035 | 61.955 | 44.490 | 61.940 | 60.598 | 72.258 |
| B Liquid unique `0716_04` | `0716_02` + final Liquid sampler/fusion + hard跨group集合唯一分配 | 72 | 54.113 | 61.636 | 47.607 | 64.198 | 60.355 | 71.992 |
| C Liquid + Encoder `0716_05` | `0716_04` + P5全局时序adapter + P3/P4/P5 pyramid-local encoder adapter | 72 | 54.635 | 61.488 | 48.512 | 65.059 | 60.620 | 71.994 |
| Set-Transport `0717_01` | `0716_04` + soft集合容量Set-Transport；hard唯一分配不变 | 68 | 54.941 | 61.836 | 47.401 | 65.140 | 60.658 | 72.523 |
| `0717_01`（末轮参考） | 同上；固定epoch 72，不参与选点 | 72 | 54.759 | 61.971 | 46.945 | 65.007 | 60.755 | 72.842 |
| Original-hard Liquid `0717_02` | `0716_02` + 原版hard Liquid；取消跨group唯一化及Set-Transport | 72 | 54.335 | 61.445 | 47.502 | 64.540 | 59.848 | 71.905 |
| Independent Diff/Product `0718_01` | Base+Liquid取消跨group唯一化；sampler/fusion显式加入difference/product | 64 | 55.276 | 61.812 | 49.551 | 65.815 | 60.786 | 72.423 |
| `0718_01`（末轮参考） | 同上；固定epoch 72，不参与选点 | 72 | 55.002 | 61.993 | 48.801 | 65.381 | 60.933 | 72.754 |
| ARCR `0718_02` | `0718_01` + anchor-residual competitive router，抑制公共route坍塌 | 64 | 53.357 | 61.054 | 44.894 | 62.722 | 58.778 | 71.450 |
| `0718_02`（末轮参考） | 同上；固定epoch 72，不参与选点 | 72 | 53.118 | 61.216 | 44.515 | 62.263 | 59.349 | 71.630 |
| Adaptive-anchor ARCR `0718_03` | `0718_02` + evidence-consistent adaptive anchor relaxation | 60 | 54.689 | 60.969 | 47.555 | 65.255 | 58.737 | 71.512 |
| `0718_03`（末轮参考） | 同上；固定epoch 72，不参与选点 | 72 | 54.645 | 60.969 | 47.892 | 64.969 | 58.636 | 71.176 |
| SASE-Liquid `0718_04` | `0718_03` + 多尺度局部稀疏谱段证据，同时修正sampler与LAF | 68 | 53.398 | 60.928 | 45.514 | 63.131 | 59.844 | 71.689 |
| `0718_04`（末轮参考） | 同上；固定epoch 72，不参与选点 | 72 | 52.986 | 60.811 | 45.263 | 62.533 | 59.721 | 71.473 |
| PCDP `0718_05` | `0718_03` + pair-consistent detail-preserving小目标局部细节保护 | 72 | 54.142 | 61.317 | 46.936 | 63.988 | 59.441 | 71.958 |
| CPAS Set-Transport `0718_06` | `0717_01` + CPAS任务保真路由与difference/product pair coupling | 72 | 54.877 | 61.274 | 48.325 | 64.782 | 60.420 | 71.737 |
| Pair-Consensus PACDE `0719_01` | pair共享route + relaxed Set-Transport + pair-aligned fusion + PACDE小目标增强 | 72 | 53.883 | 61.411 | 46.831 | 63.920 | 59.931 | 72.366 |
| Reliability Pair-Consensus `0719_02` | `0719_01` + reliability-weighted consensus residual；单卡batch 8 | 64 | 55.190 | 60.689 | 48.838 | 65.374 | 59.359 | 70.487 |
| `0719_02`（末轮参考） | 同上；固定epoch 72，不参与选点 | 72 | 54.933 | 60.912 | 47.785 | 64.878 | 59.569 | 70.751 |
| Pair-Consensus no-PACDE `0719_03` | `0719_01`严格消融，仅移除PACDE局部细节分支 | 72 | 54.209 | 61.504 | 47.854 | 63.951 | 60.531 | 72.259 |
| Wide-LAF + GroupMod replay `0719_04` | `0716_02` + 独立8-group sampler、Wide LAF和GroupMod；无pair/set机制 | 72 | 53.932 | 60.908 | 45.958 | 63.723 | 59.028 | 71.362 |
| Accuracy-fixed parent `0721_02` | `0718_01` + 共享旋转/GMC及PairDN negative target训练正确性修复 | 72 | 54.327 | 61.659 | 47.233 | 64.123 | 60.371 | 72.246 |
| BSAC-Liquid `0721_04` | `0721_02` + 24参数band-slot条件尺度，校准物理谱段进入固定Conv3D slot的分布 | 72 | 54.530 | 61.528 | 47.626 | 64.022 | 60.332 | 71.946 |
| DSE-Liquid `0721_05` | `0721_02` + mean/RMS dispersion-aware evidence，以16参数grouped 1x1融合 | 72 | 54.635 | 61.895 | 46.574 | 64.655 | 60.544 | 72.608 |
| Pair-coherent PairDN `0723_01` | `0721_02`模型 + shared pair-relative DN、2:1 pos/neg、难样本与表示一致L1 | 64 | 53.955 | 62.032 | 45.336 | 63.673 | 61.392 | 73.215 |
| `0723_01`（末轮参考） | 同上；固定epoch 72，不参与选点 | 72 | 53.594 | 62.012 | 45.317 | 62.837 | 61.646 | 73.205 |
| Independent pair-side DN `0723_02` | `0723_01`严格消融；仅将pair两帧共享相对噪声改为独立采样 | 72 | 53.637 | 61.679 | 43.505 | 62.444 | 60.111 | 72.448 |
| Pair-coherent PairDN + DSE `0723_03` | `0723_01` + 16参数mean/RMS fusion evidence mixer | 68 | 55.036 | 61.745 | 47.792 | 64.976 | 61.357 | 72.670 |
| `0723_03`（末轮参考） | 同上；固定epoch 72，不参与选点 | 72 | 54.798 | 61.753 | 47.675 | 64.459 | 61.353 | 72.619 |
| Pair-coherent PairDN + CSPR `0723_04` | `0723_01` + detached 24x32 shared-Conv3D route preview | 72 | 54.523 | 61.738 | 46.856 | 64.082 | 61.212 | 72.561 |
| Pair-coherent PairDN + local CP-DSE `0723_05` | `0723_01` + 8参数逐像素归一化色散SE-logit残差 | 68 | 53.536 | 61.619 | 44.014 | 63.252 | 60.698 | 72.797 |
| Pair-coherent PairDN + pair-global CP-DSE `0723_06` | `0723_01` + pair共享的8参数group色散SE-logit残差 | 72 | 54.124 | 61.914 | 45.383 | 63.299 | 60.593 | 72.914 |
| Pair-coherent PairDN + SCPD `0723_08` | `0723_01` + 物理谱段坐标中的pair共识色散残差 | 72 | 54.465 | 61.213 | 46.178 | 63.988 | 61.188 | 71.986 |
| Pair-coherent PairDN + DSE + pair-global CP-DSE `0725_01` | `0723_01` + DSE局部检测证据 + pair共享CP-DSE关联残差 | 72 | 55.126 | 61.998 | 47.739 | 65.626 | 61.161 | 73.125 |
| Base + Liquid + `0705_01` Encoder `0726_02` | `0723_01` + P5双向全局MHA + P3/P4/P5 pyramid-local adapter | 72 | 54.742 | 61.631 | 45.957 | 64.798 | 59.930 | 72.408 |
| Base + Liquid + common-detail Encoder `0726_03` | `0723_01` + P5双向MHA + pair均值守恒的common/detail局部适配 | 72 | **54.654** | **62.240** | 45.710 | 64.109 | 61.359 | 73.458 |
| Base + Liquid + Dual-Evidence Encoder `0727_01` | `0723_01` + P5双向MHA + common共享检测残差 + signed-detail关联残差 | 72 | **54.437** | **62.393** | 45.857 | 63.697 | 62.154 | 73.349 |

AP为同一Tracking epoch的诊断结果，不跨epoch拼接：

| 实验 | AP epoch | pair mAP | pair AP50 |
| --- | ---: | ---: | ---: |
| A Paper Base `0716_02` | 68 | 0.3149 | 0.5225 |
| A `0716_02`（末轮参考） | 72 | 0.3142 | 0.5217 |
| B Liquid unique `0716_04` | 72 | 0.3160 | 0.5301 |
| C Liquid + Encoder `0716_05` | 72 | 0.3215 | 0.5467 |
| Set-Transport `0717_01` | 68 | 0.3196 | 0.5359 |
| `0717_01`（末轮参考） | 72 | 0.3206 | 0.5356 |
| Original-hard Liquid `0717_02` | 72 | 0.3215 | 0.5429 |
| Independent Diff/Product `0718_01` | 64 | 0.3200 | 0.5417 |
| `0718_01`（末轮参考） | 72 | 0.3202 | 0.5415 |
| ARCR `0718_02` | 64 | 0.3088 | 0.5218 |
| `0718_02`（末轮参考） | 72 | 0.3094 | 0.5219 |
| Adaptive-anchor ARCR `0718_03` | 60 | 0.3152 | 0.5302 |
| `0718_03`（末轮参考） | 72 | 0.3182 | 0.5324 |
| SASE-Liquid `0718_04` | 68 | 0.3079 | 0.5212 |
| `0718_04`（末轮参考） | 72 | 0.3051 | 0.5156 |
| PCDP `0718_05` | 72 | 0.3166 | 0.5304 |
| CPAS Set-Transport `0718_06` | 72 | 0.3244 | 0.5438 |
| Pair-Consensus PACDE `0719_01` | 72 | 0.3161 | 0.5339 |
| Reliability Pair-Consensus `0719_02` | 64 | 0.3277 | 0.5492 |
| `0719_02`（末轮参考） | 72 | 0.3273 | 0.5481 |
| Pair-Consensus no-PACDE `0719_03` | 72 | 0.3165 | 0.5314 |
| Wide-LAF + GroupMod replay `0719_04` | 72 | 0.3134 | 0.5277 |
| Accuracy-fixed parent `0721_02` | 72 | 0.3161 | 0.5340 |
| BSAC-Liquid `0721_04` | 72 | 0.3211 | 0.5353 |
| DSE-Liquid `0721_05` | 72 | 0.3254 | 0.5438 |
| Pair-coherent PairDN `0723_01` | 64 | 0.3114 | 0.5268 |
| `0723_01`（末轮参考） | 72 | 0.3113 | 0.5256 |
| Independent pair-side DN `0723_02` | 72 | 0.3155 | 0.5308 |
| Pair-coherent PairDN + DSE `0723_03` | 68 | 0.3239 | 0.5402 |
| `0723_03`（末轮参考） | 72 | 0.3241 | 0.5406 |
| Pair-coherent PairDN + CSPR `0723_04` | 72 | 0.3162 | 0.5376 |
| Pair-coherent PairDN + local CP-DSE `0723_05` | 68 | 0.3148 | 0.5320 |
| Pair-coherent PairDN + pair-global CP-DSE `0723_06` | 72 | 0.3124 | 0.5241 |
| Pair-coherent PairDN + SCPD `0723_08` | 72 | 0.3152 | 0.5334 |
| Pair-coherent PairDN + DSE + pair-global CP-DSE `0725_01` | 72 | 0.3231 | 0.5449 |
| Base + Liquid + `0705_01` Encoder `0726_02` | 72 | 0.3223 | 0.5429 |
| Base + Liquid + common-detail Encoder `0726_03` | 72 | 0.3239 | 0.5380 |
| Base + Liquid + Dual-Evidence Encoder `0727_01` | 72 | 0.3201 | 0.5371 |

Base的epoch 72固定末轮`cls_HOTA + det_HOTA=114.990`，低于按主指标选出的
epoch 68的`115.296`。因此epoch 72仅用于横向观察训练末态；正式Base及所有模块消融仍须
先按`cls_HOTA + det_HOTA`确定唯一checkpoint，再报告该checkpoint的全部tracking和AP指标。

截至2026-07-23，正式选点中HOTA选择目标最高的仍是`0718_01` epoch 64；其cls HOTA相对Base
提高`1.962`，det HOTA下降`0.170`。唯一在父实验两个HOTA方向都提高的已完成结构是
`0717_01`相对`0716_04`，cls/det分别提高`0.828/0.200`；但它的det HOTA仍比Paper Base低
`0.146`。近期已完成项中，`0718_06`的HOTA和为`116.151`，相对Base的cls/det
分别变化`+1.563/-0.708`；`0718_05`为`+0.828/-0.665`，`0719_01`为`+0.569/-0.571`，
`0719_02`为`+1.876/-1.293`，`0719_03`为`+0.895/-0.478`，最新`0719_04`为
`+0.618/-1.074`。因此新结果继续显示
最新结果中，`0721_05 DSE`相对同协议父配置`0721_02`的cls/det HOTA分别提高
`0.308/0.236`，是本轮严格父子比较中的双提升；pair mAP/AP50也提高`0.0093/0.0098`。
相对Paper Base，DSE为`+1.321/-0.087`，det差距已经缩小到0.1以内，但仍不能声明绝对双超。
`0721_04 BSAC`相对父配置为`+0.203/-0.131`：cls DetA提高`0.838`，但cls/det AssA分别下降
`0.736/0.527`，说明slot校准增强了检测证据，却扰动了跨帧关联一致性。因此DSE应保留为当前
accuracy-fixed Liquid主候选，BSAC不应单独进入论文主线。

## 8. 可复现性记录

- 代码基准 commit：`3c8af7419d9a79b6251268936805c9666bd2ab99`；正式启动时工作树含未提交研究改动，MMEngine 会在 workdir 保存完整 resolved config，启动日志和代码 diff hash需一并保留。
- COCO R18 source SHA256：`3ba8b5c909c9a1c4f21e96d0a7251ab1a485093955ca327d0061fef8d33c66f0`。
- 权重适配 manifest：`/data4/litianhao/PairMmot/pretrained_weights/rtdetr_coco_pair_family_manifest.json`。
- 正式实验不得 resume，不得使用旧 COCO+Objects365 adapted checkpoint，不得回退到历史有效高度 800 的输入协议。
- 已取消 strict run log：`/data4/litianhao/PairMmot/workdir_99/0716_02_paper_base_r18_coco_full_1200x900_bf16/20260716_154409/20260716_154409.log`。
- 已取消 strict run 的 tracked diff patch SHA256：`3e28bf0a6eef22ad1dec5a72a363ae4d2b7a3d5d1ef6d7ea3d843c2f92e2b26e`。
- 已取消 strict run 的代码文件 manifest SHA256：`e38bb254283bcdbbbc202daca37c1416e31d1e3bb5b19961fcd022d5d1dcf38e`，包含1033个源码/配置文件的逐文件哈希。
- 已取消 strict run 的代码快照：`source_snapshot_20260716_1544.tar.gz`，SHA256
  `b9ced3dde98a8703bd354ad17bc659a16c37b0de7e3627b356ae4f755079b6e5`；保存在正式 workdir，可恢复本次运行使用的完整源码。

### 8.1 启动前验收

静态检查确认：train/test 为 8372/5416 samples；实际样本输入为
`(pair=2, channel=8, height=900, width=1200)`；模型为 R18、3-layer decoder，
可训练参数 21,907,028；adapted checkpoint 加载无 unexpected key。

双卡 smoke 使用正式 batch、BF16边界和
`find_unused_parameters=False` 完成4次前向/反向。峰值显存约 9166 MB/GPU，均值约
1.1813 s/iter；总 loss、decoder loss、PairDN loss 和 encoder proposal loss 均为有限值，
未出现 OOM、NaN、unused parameter 或 dtype 错误。

### 8.2 正式启动

`0716_02` 于 2026-07-16 15:44 CST 在 99 的 GPU 0/1 fresh 启动。首个正式日志点为
epoch 1 iter 50：`time=0.9545 s/iter`、`memory=9190 MB`（日志统计）、
`loss=33.0213`、`grad_norm=95.9832`；同时采样的设备显存约 15.5 GB/GPU。
日志中未发现 traceback、OOM、NaN 或 unused parameter。按训练迭代 ETA 约20小时，另需
计入每4 epoch的18次完整 validation 与 TrackEval。

该首次运行启用了 strict deterministic。根据用户指令，运行在 epoch 1 内主动终止，日志和
源码快照保留在无后缀历史 workdir，不作为论文结果。模型或数值本身没有报错。

### 8.3 Non-deterministic fresh restart

正式论文 baseline 改为固定 seed 3407、`deterministic=False`，并显式清除
`CUBLAS_WORKSPACE_CONFIG`。其余模型、数据、初始化、优化器、BF16边界、DDP、GMC和
评测设置完全不变。新运行使用 `_nondet_restart` workdir，且不加载首次运行 checkpoint。

新运行于 2026-07-16 15:57 CST 在 GPU 0/1 启动，resolved runtime 明确记录
`deterministic: False`。epoch 1 iter 50 为 `time=0.8223 s/iter`、日志显存 9146 MB、
`loss=32.9911`、`grad_norm=93.5086`；设备显存约15.3 GB/GPU，无 OOM、NaN、unused
parameter 或 traceback。其正式 run log 为
`0716_02..._nondet_restart/20260716_155757/20260716_155757.log`。

Non-deterministic restart 的复现材料：tracked patch SHA256
`0910b70069d12dabf2fb7ac9d933078403712358a406dd8bda8fc255e43c67d8`；逐文件
manifest SHA256 `180a79c733243bba52afbefaae59b86d7a799ba934e1408e0245bcec66807d42`；
完整源码快照 `source_snapshot_20260716_nondet_restart.tar.gz` SHA256
`69b51831de30ef5d9d262a63a0f6b93bd8618b4cbfce8449c34ebc04cef3ea8b`。

该运行仍沿用了历史训练采样：每个 anchor 在 `t-1` 和 `t+1` 中随机选择后再按时间排序。
虽然模型输入没有反向时间，但一个 epoch 会重复部分相邻 pair 并遗漏另一些 pair。根据用户
确认，该运行在 epoch 1 内主动终止，不作为论文结果。

### 8.4 Unique ordered-pair fresh restart

最终论文协议取消 anchor 双向随机选择：训练和测试均使用 `frame_intervals=(1,)`、
`random_interval_range=None`，直接枚举每个唯一 `t-1 -> t` pair。full train 因而严格为
8297个 pair，每个 epoch 各出现一次，不重复、不遗漏，并与 validation/test 时间方向一致。
新运行使用 `_orderedpairs_restart` workdir，保持 `deterministic=False` 且完全 fresh train。

最终运行于 2026-07-16 16:08 CST 在 99 的 GPU 0/1 启动。启动前实际构建数据集并确认
8297个 `(sequence, prev, curr)` key 全部唯一、所有 `curr-prev=1`，首个 transformed sample
为 `(2, 8, 900, 1200)`。runtime resolved config 明确记录 `deterministic=False`、
`find_unused_parameters=False` 和 `resume=False`。

epoch 1 iter 50 为 `time=0.8185 s/iter`、日志显存 `9166 MB`、`loss=34.0257`、
`grad_norm=97.2520`；iter 100 为 `0.8031 s/iter`。截至 iter 150，主损失、PairDN损失和
encoder proposal损失均为有限值，未出现 traceback、OOM、NaN、unused parameter 或 DDP
reduction错误。正式日志为
`0716_02..._orderedpairs_restart/20260716_160843/20260716_160843.log`；启动阶段训练 ETA
约17小时，另需考虑每4 epoch执行的完整 validation 与 TrackEval。

本次最终运行的 tracked patch SHA256 为
`e90cc658c62467f85db16c2cb0be237a1d2f98556b75d17533a45ee84757b564`；1284个
Python/YAML/shell文件的逐文件 manifest SHA256 为
`563afa719fb880e1644b843344f716a7d63827084b8d660517fc13f7d5103d3e`；完整源码快照
`source_snapshot_20260716_orderedpairs_restart.tar.gz` SHA256 为
`bfcbe18aadb4958664cdf4425ac0b229fe625065509da8f3809f19ea79dd9a19`，均保存在最终
workdir根目录。

### 8.5 Base + Liquid fresh run

`0716_03` 只在 `0716_02` 论文 Base 上增加已由 `0715_05` 全量实验验证有效的最终
Liquid：8个循环三谱段组、独立单帧 sampler、wide overlap-aware LAF、group modulation、
pair sampler router和pair transport；两个有向关系模块均只使用有序 `[x,y]`。COCO-only
初始化、1200x900输入、8297个唯一 `t-1 -> t` pair、BF16边界、优化器、72-epoch schedule、
GMC和完整评测协议均与 Base 相同。所有 Liquid 新参数使用 `lr_mult=1.0`。

实验于 2026-07-16 16:16 CST 在本机 GPU 2/3 fresh 启动。epoch 1 iter 50 为
`time=0.8857 s/iter`、日志显存 `10691 MB`、`loss=36.0175`、`grad_norm=169.5151`；
主损失、PairDN和encoder proposal loss均为有限值，未出现 traceback、OOM、NaN、unused
parameter或DDP reduction错误。正式日志为
`0716_03.../20260716_161637/20260716_161637.log`。

启动代码的 tracked patch SHA256 为
`fd0b4b2a7e98b25cf8179a674def5c127eadadadd7f537cbafbb34c069e1cfdc`；1286个
Python/YAML/shell文件的manifest SHA256为
`0780b1fadef05f3b6e58038f9c97effc564a6a63f3e8d3ef266f7979541b382b`；完整源码快照
`source_snapshot_20260716_base_plus_liquid.tar.gz` SHA256为
`bbafd5bc39ef5515a194c0ac606f91656bcb6af8e71285a28ba46a21bfda65c7`。

该次启动最后写入 epoch 1 iter 1000，随后 GPU 2 从驱动层掉卡，`nvidia-smi` 对设备
`0000:B1:00.0` 返回 `Unknown Error`。2026-07-16 16:41 CST 按用户指令终止并清理 GPU 2/3
对应的完整进程组。checkpoint间隔为4 epochs，因此没有正式 checkpoint；本次运行作废，
后续不得 resume，须在健康GPU上以相同配置fresh train。同期 GPU 0/1 的 `0716_02` 进程组
保持存活且训练日志持续推进，未发现NCCL、CUDA、NaN或loss异常。

随后为允许用户手动重启本机，`0716_02` 于 2026-07-16 16:44 CST 在 epoch 3 iter 600
主动停止。停止前 `time=0.7560 s/iter`、`loss=17.3561`、`grad_norm=45.4604`，没有异常。
checkpoint从epoch 4开始每4 epochs保存，因此本次没有checkpoint；重启后不得resume，须以
相同配置fresh train。

### 8.6 Base reboot-fresh run

服务器重启后四张GPU均恢复正常，`0716_02` 于 2026-07-16 17:10 CST 在 GPU 0/1以新
workdir完全fresh启动。新配置与最终Base的model、dataset、optimizer、scheduler、hooks、
randomness逐项相同，仅覆盖workdir和TrackEval输出目录；仍为8297个唯一有序相邻pair、
COCO-only初始化、1200x900、BF16、`find_unused_parameters=False`和`resume=False`。

epoch 1 iter 50 为 `time=0.8307 s/iter`、日志显存 `9166 MB`、`loss=34.0826`、
`grad_norm=121.4044`；主损失、PairDN和encoder proposal loss均为有限值，未出现CUDA、
NCCL、NaN、OOM、unused parameter或DDP reduction错误。正式日志为
`0716_02..._reboot_fresh/20260716_171004/20260716_171004.log`。

启动代码的tracked patch SHA256为
`4c3250f4323f61c5b3997ce70d04d0c8096e8863c6dda7368f5c7d1ec0ec2bb0`；源码manifest
SHA256为`0a9215dc3f3b0f1b6c2bc41d602760f9032ffd56f5f599e44f0065d714a505af`；完整源码快照
`source_snapshot_20260716_base_reboot_fresh.tar.gz` SHA256为
`645e304993e99e01f8626807656e044c0b3ed82408cd10c1c8ff200023e48188`。

该运行已完成全部72 epochs、18次validation和18/18异步TrackEval，未发现异步评测失败
或残留进程。严格按`cls_HOTA + det_HOTA`在18个点中选择，唯一最佳为epoch 68：
`cls_HOTA=53.314`、`det_HOTA=61.982`、`cls_MOTA=44.690`、
`cls_IDF1=62.218`、`det_MOTA=60.599`、`det_IDF1=72.303`，HOTA sum为
`115.296`。同一epoch 68的检测指标为`pair_mAP=0.3149`、`pair_AP50=0.5225`。
epoch 72的HOTA sum为`114.990`，因此仅作为末轮稳定性参考，不作为论文checkpoint。

### 8.7 Base + Liquid fresh run on 197

本机失败尝试不resume。同步当前代码后，`0716_03` 于 2026-07-16 17:15 CST 在197的
GPU 0/3使用新workdir完全fresh启动。197专用配置仅覆盖远端data、GMC、workdir和
TrackEval路径；model、optimizer、scheduler、hooks、randomness、COCO-only初始化、
1200x900输入、8297个唯一有序pair、BF16边界及72-epoch评测协议均与本机正式配置一致。

epoch 1 iter 50 为 `time=1.0818 s/iter`、日志显存 `10692 MB`、`loss=36.0201`、
`grad_norm=194.2549`；主损失、PairDN和encoder proposal loss均有限，未出现CUDA、NCCL、
NaN、OOM、unused parameter或DDP reduction错误。正式日志为
`0716_03..._fresh/20260716_171544/20260716_171544.log`。

远端启动代码的tracked patch SHA256为
`2cf34504aa6af3679111cd4b6562cbf597348b90f15e651257b5e42518831ed2`；源码manifest
SHA256为`7dd1c5f3cb15d109045e17668586d8c06fb91b5f0e1ec796575907a964b54e20`；完整源码快照
`source_snapshot_20260716_base_plus_liquid_197.tar.gz` SHA256为
`b79da8e6f0ef78ada288d02f11c6bbf3a1b75755c49bd8b6500d6faba5fa3b51`。

该运行在epoch 21 iter 50主动停止且不resume。soft sampler的argmax预览已经出现
`432/431`等相同无序三波段集合，说明原实现只保证group内部不重复，不能阻止跨group
坍塌，因而不再作为最终论文Liquid运行。

### 8.8 Base + Liquid group-set-unique fresh run on 197

`0716_04`保持`0716_03`的模型、数据、初始化、优化器、BF16边界和评测协议，只在hard
train/eval增加跨group的无序波段集合唯一分配。8个group从`C(8,3)=56`个候选集合中
选择互不相同的集合，集合内部保留最高分排列；采用GPU上的regret-first greedy和
straight-through反向，不增加loss。soft fusion仍保留所有连续谱段权重，不做互斥屏蔽。

本地及197均通过20项sampler/stem测试。实验于2026-07-16 23:22 CST在197 GPU 0/3以
独立workdir完全fresh启动。epoch 1 iter 50为`time=0.9771 s/iter`、日志显存
`10692 MB`、`loss=35.9972`、`grad_norm=169.3626`，主损失、PairDN和encoder proposal
loss均有限。监控为`hard=False, unique_sets=8.00, max_set_repeat=1.00`，确认hard预览的
8组集合全部唯一。正式日志位于
`/data4/litianhao/PairMmot/workdir_197/0716_04_paper_base_plus_liquid_groupsetunique_r18_coco_full_1200x900_bf16_orderedpairs_fresh/launch.log`。

实验已完成72 epochs和18/18 TrackEval。按唯一规则
`cls_HOTA + det_HOTA`选择epoch 72：`cls_HOTA=54.113`、`det_HOTA=61.636`、总和
`115.749`；同epoch其余指标为`cls_MOTA=47.607`、`cls_IDF1=64.198`、
`det_MOTA=60.355`、`det_IDF1=71.992`，检测诊断为`pair mAP=0.3160`、
`pair AP50=0.5301`。相对Base唯一最佳epoch 68，总和提升`+0.453`，其中
`cls_HOTA +0.799`、`det_HOTA -0.346`；相对Base固定epoch 72，总和提升`+0.759`，其中
`cls_HOTA +1.078`、`det_HOTA -0.319`。因此Liquid已超过论文Base的主选择指标，但增益
明确来自cls侧，det侧尚未同步提升。

同epoch曲线在epoch 4--52低于Base，epoch 52差距缩至`-0.042`，从epoch 56开始持续领先，
epoch 60/64/68/72分别领先`+0.580/+0.430/+0.212/+0.759`。后期没有回落，说明结果不是
单点波动。hard阶段所有监控点均为`unique_sets=8.00, max_set_repeat=1.00`，训练和异步评测
期间未发现CUDA、NCCL、OOM、loss NaN、grad NaN或DDP错误。

类别HOTA相对Base最佳epoch 68的主要提升为`truck +6.861`、`tricycle +4.415`和
`van +1.025`；下降为`bike -2.967`、`awning-bike -1.228`、`pedestrian -1.075`、
`bus -0.524`和`car -0.119`。结果说明Liquid的总体收益主要来自困难车辆类别的类别/关联
表达，而不是所有类别与det侧的一致改善；后续Encoder/Proposal/Decoder主线应重点验证能否
保留该cls收益并补回det HOTA。

### 8.9 Base + Liquid group-set-unique + Encoder on 252

`0716_05`严格继承`0716_04`的最终Liquid和论文协议，只加入历史encoder最佳
`0705_01 p5temporal_pyramidlocal`：shared AIFI后的P5 global pair temporal adapter，以及
CCFF/FPN后的P3/P4/P5 pyramid-local pair adapter。两条残差gate均为零初始化；adapter参数
使用`lr_mult=2.0`，gate gamma使用`lr_mult=20.0, decay_mult=0.0`，其余参数学习率不变。

252上的20项Liquid/stem测试和10项temporal adapter测试全部通过。正式启动前使用GPU 0/1
完成100 iter双卡DDP测试：`find_unused_parameters=False`无报错，框架统计显存
`11387 MB/rank`，两个gate和attention/local block均收到梯度，全部loss有限。测试进程组已
完整停止并释放显存，未写入正式目录。

正式实验于2026-07-16 23:36 CST在252 GPU 0/1 fresh启动，workdir为
`/data4/litianhao/PairMmot/workdir_252/0716_05_paper_base_plus_liquid_groupsetunique_encoder_r18_coco_full_1200x900_bf16_orderedpairs_fresh`。
epoch 1 iter 50为`time=1.5238 s/iter`、`memory=11387 MB`、`loss=35.8960`、
`grad_norm=178.3931`；Liquid hard预览为`unique_sets=8.00, max_set_repeat=1.00`，未出现
CUDA、NCCL、OOM、NaN、unused parameter或DDP错误。

该实验已完成72 epochs和18/18 TrackEval。唯一最大`cls_HOTA + det_HOTA`位于epoch 72：
`cls_HOTA=54.635`、`det_HOTA=61.488`、`cls_MOTA=48.512`、`cls_IDF1=65.059`、
`det_MOTA=60.620`、`det_IDF1=71.994`；同epoch `pair_mAP=0.3215`、
`pair_AP50=0.5467`。相对直接父实验0716_04，cls HOTA提高`0.522`、det HOTA下降
`0.148`，选择指标净提高`0.374`；相对Paper Base，cls/det HOTA分别变化
`+1.321/-0.494`。因此Encoder叠加有效提高总目标和AP，但提升仍集中在cls侧，尚未解决
Liquid相对Base的det HOTA缺口。

### 8.10 Parallel Liquid Set-Transport candidate

`0717_01`是相对`0716_04`的单变量结构探索。它不修改最终Liquid的pair router、wide LAF、
group modulation或pair transport，而是在soft sampler中增加无参数的集合容量传输：将
三slot概率映射到56个无序三波段集合及其6种排列，利用48个slack token和16次log-Sinkhorn
得到容量受限的连续group-set分配，再还原到原波段概率接口。结构强度在前12 epochs从0
增加到1，hard阶段的ST梯度也通过该投影，不增加辅助loss。

23项单测及GPU 2/3上的100 iter双卡DDP测试通过。正式运行于2026-07-17 00:15 CST fresh
启动，epoch 1 iter 50为`time=0.9347 s/iter`、`memory=10695 MB`、`loss=36.0532`、
`grad_norm=177.8087`，未出现异常。workdir为
`/data4/litianhao/PairMmot/workdir_99/0717_01_paper_base_plus_liquid_settransport_r18_coco_full_1200x900_bf16_orderedpairs_fresh`。

该实验目前作为并行Liquid候选，不改变A/B/C主链定义。完成后与`0716_04`分别按唯一最佳
`cls_HOTA + det_HOTA`比较，只有在两个HOTA方向和稳定性均有依据时才考虑替换主线B。

本机正式训练于2026-07-17在epoch 2 iter 250主动取消。取消原因是GPU 2/3存在历史掉卡
风险，而非模型异常；进程组、worker和screen均已完整清理，GPU显存释放。此次不完整运行
不进入结果表，Set-Transport实现和已通过的单元/DDP验证保留，后续在稳定服务器上应fresh
重跑。

同一科学实验保留`0717_01`编号，于2026-07-17 05:40 CST迁移到双卡AutoDL实例并从
COCO适配权重fresh重跑，不使用99上的任何checkpoint。AutoDL路径覆盖配置只修改HSMOT、
GMC、预训练和workdir位置；模型、每卡batch 4、全局batch 8、`lr=1e-4`、BF16边界、
`find_unused_parameters=False`、72 epochs和每4 epoch完整评测均与99原运行一致。workdir为
`/root/autodl-tmp/work_dirs/0717_01_paper_base_plus_liquid_settransport_r18_coco_full_1200x900_bf16_orderedpairs_autodl_fresh`。

epoch 1 iter 50为`time=0.9282 s/iter`、`memory=10703 MB/rank`、`loss=35.7344`、
`grad_norm=193.7061`，与99原运行的`0.9347 s/iter`相符。两卡`nvidia-smi`占用约
18.8 GB/GPU；Set-Transport监控为`strength=0.004`、`unique_sets=8.00`、
`max_set_repeat=1.00`、`set_max_load=0.250`，无CUDA、NCCL、OOM、NaN或unused parameter
错误。早期99运行仍不计入最终结果。AutoDL fresh run已于2026-07-18完成72 epochs和
18/18异步TrackEval。按唯一最大`cls_HOTA + det_HOTA`选择epoch 68：
`cls_HOTA=54.941`、`det_HOTA=61.836`、总和`116.777`；同一epoch的
`pair_mAP=0.3196`、`pair_AP50=0.5359`。epoch 72总和为`116.730`，只比最佳点低0.047，
后期结果稳定。

相对Paper Base最佳点，Set-Transport的cls HOTA提升`+1.627`、det HOTA下降`-0.146`、
总和提升`+1.481`。相对直接单变量对照`0716_04`，cls/det HOTA分别提升
`+0.828/+0.200`，总和提升`+1.028`，因此Set-Transport明确优于hard-only group-set
unique Liquid，但尚不能宣称两个HOTA方向都超过Paper Base。相对`0716_04`的类别变化为：
`truck +6.008`、`bike +2.128`、`pedestrian +0.437`、`awning-bike +0.166`、
`van +0.134`；`bus -1.418`、`tricycle -0.717`、`car -0.110`。cls收益主要由truck驱动，
并非所有类别一致改善。

训练末期监控始终满足`unique_sets=8.00`、`max_set_repeat=1.00`、
`set_max_load`约1.0。hard pattern仍集中在少数集合组合，说明Set-Transport解决的是
跨group集合容量和soft/hard目标一致性，而不是保证图像级route多样性。自动finalizer在
18/18评测完成后因无卡shell中`python`不在`PATH`而停止，实验本身未失败；结果已用
`/root/miniconda3/bin/python3.12`重新生成，并将epoch 68/72、日志和完整TrackEval备份到
`/root/autodl-fs/PairMOT_results/0717_01`。

### 8.11 Original-hard Liquid strict control on 99

`0717_02`用于补齐论文协议下缺失的原版Liquid严格对照。它完整继承Paper Base和最终
pair-aware Liquid，仅显式设置`hard_group_unique_sets=False`和
`soft_group_set_transport=None`：每个group的三个slot在hard阶段保持组内波段去重，但不同
group允许选择相同的无序三波段集合。该实验没有集合级贪心唯一分配，也没有Sinkhorn
Set-Transport，因而可直接判断历史`0715_05`的原版Liquid收益能否迁移到更强Paper Base。

实验于2026-07-17 13:22 CST在99 GPU 0/1以seed 3407 fresh启动，使用COCO适配初始化、
full HSMOT、原生`1200x900`、ordered pairs、每卡batch 4、BF16、
`find_unused_parameters=False`和72 epochs。epoch 1 iter 50/100分别为
`0.8598/0.9109 s/iter`，框架显存`10691 MB/rank`；iter 100的`loss=32.0397`、
`grad_norm=149.3895`，均为有限值。sampler为`hard=False`、`set_transport=0.000`，初始
pattern保持`701/012/123/234/345/456/567/670`。未发现CUDA、NCCL、OOM、NaN或DDP异常。

训练已完成72 epochs及18/18异步TrackEval。按唯一最大`cls_HOTA + det_HOTA`选中epoch 72：
`cls_HOTA=54.335`、`det_HOTA=61.445`、`cls_MOTA=47.502`、`cls_IDF1=64.540`、
`det_MOTA=59.848`、`det_IDF1=71.905`。同一epoch的pair mAP为`0.3215`、pair AP50为
`0.5429`。相对Paper Base唯一最佳epoch 68，cls HOTA提高`1.021`，但det HOTA下降
`0.537`；cls MOTA/IDF1提高`2.812/2.322`，det MOTA/IDF1下降`0.751/0.398`。

类别HOTA变化为：`truck +7.445`、`tricycle +4.225`、`bus +0.264`、`van +0.005`、
`car -0.230`、`bike -0.883`、`awning-bike -1.148`、`pedestrian -1.514`。因此原版Liquid
的收益主要来自truck和tricycle等困难类别，不是普遍检测质量改善；pedestrian下降和det
侧整体回落仍不满足论文最终Liquid“cls与det同时超过Base”的要求。

曲线在epoch 8--56均低于Base同epoch，epoch 60首次领先；epoch 64/68/72相对Base同epoch
分别变化`-0.020/+0.098/+0.790`。后期持续追赶说明Liquid收敛更慢。末期50条hard监控只
出现4种完整pattern，变化集中于第一组`514/521`和第六组`430/431`，其余6组固定；每个
pattern内部仍保持8个不同集合。故该实验避免了跨group全相同集合，但没有解决route对图像
内容响应不足的问题。

### 8.12 Hard-sampled soft-context Liquid on 197

`0717_03`严格继承已完成的`0716_04 group-set-unique Liquid`，只解耦hard阶段的采样概率和
融合上下文。Conv3D实际谱段选择继续使用全局唯一`P_hard`；GroupMod的coverage/entropy/
peak、overlap-aware LAF的pattern/overlap以及LAF内部Pair Transport改用同一次采样对应的
连续`P_soft`。soft阶段二者相同，因而epoch 36以前前向严格保持`0716_04`路径；epoch 36
以后保留离散可解释采样，同时避免融合描述退化为离散集合交集。该结构不增加参数、loss或
额外sampler前向。

本机及197端24项stem/sampler测试通过，覆盖默认路径回归、hard one-hot采样、soft context
连续性和两条梯度回传。实验于2026-07-17 21:59 CST在197 GPU 4/5 fresh启动；选择4/5是
因为GPU 0--2已有其他用户任务，避免资源争用。配置保持COCO初始化、full HSMOT、
`1200x900`、ordered pairs、每卡batch 4、BF16、`find_unused_parameters=False`和72 epochs。
epoch 1 iter 100为`time=0.8514 s/iter`、`memory=10692 MB/rank`、`loss=31.9603`、
`grad_norm=158.1667`，未发现CUDA、NCCL、OOM、NaN、unused parameter或DDP异常。

该运行于2026-07-18主动停止在epoch 15 iter 150，最近checkpoint为epoch 12。停止是实验
方向调整，不是训练错误；后续不resume。

### 8.13 Independent-group Difference/Product Liquid on 197

`0718_01`取消跨group唯一集合和Set-Transport，但保留单group内部无放回采样；因此不同
group可以复用有价值的三波段集合，组内仍不会出现重复波段。sampler router和fusion中的
PairTransport同时改用显式`[x,y,x-y,x*y]`关系，分别修正谱段route logits和LAF前的
group descriptor。其余模型、COCO初始化、full HSMOT、`1200x900`、ordered pairs、BF16、
seed、学习率与72-epoch评测协议保持不变。

本机和197端24项测试均通过。实验于2026-07-18 02:05 CST在197 GPU 4/5 fresh启动；
epoch 1 iter 100为`0.9272 s/iter`、`10690 MB/rank`、`loss=32.8708`、
`grad_norm=176.9842`，所有分支loss有限，未见训练异常。该实验同时改变唯一化和pair关系
表达，因此是候选结构验证，不作为单因素消融；正式结果仍按18个评测点中唯一最大
`cls_HOTA + det_HOTA`选epoch，并从同一epoch记录AP。

实验现已完成72 epochs和18/18 TrackEval。唯一最大`cls_HOTA + det_HOTA`出现在epoch 64，
因此主结果及AP均严格取epoch 64；同时按论文完整性单列训练终点epoch 72，不用epoch 72替换
预注册选择结果。

| experiment | epoch | cls HOTA | det HOTA | cls MOTA | cls IDF1 | det MOTA | det IDF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Paper Base | 68 | 53.314 | 61.982 | 44.690 | 62.218 | 60.599 | 72.303 |
| `0718_01` unique best | 64 | **55.276** | 61.812 | **49.551** | **65.815** | **60.786** | **72.423** |
| change vs Base | - | +1.962 | -0.170 | +4.861 | +3.597 | +0.187 | +0.120 |
| `0718_01` endpoint | 72 | **55.002** | **61.993** | **48.801** | **65.381** | **60.933** | **72.754** |
| endpoint change vs Base | - | +1.688 | +0.011 | +4.111 | +3.163 | +0.334 | +0.451 |

| experiment | epoch | pair mAP | pair AP50 |
| --- | ---: | ---: | ---: |
| Paper Base | 68 | 0.3149 | 0.5225 |
| `0718_01` unique best | 64 | **0.3200** | **0.5417** |
| change vs Base | - | +0.0051 | +0.0192 |
| `0718_01` endpoint | 72 | **0.3202** | **0.5415** |
| endpoint change vs Base | - | +0.0053 | +0.0190 |

epoch 64的cls提升主要来自truck `+12.153`、tricycle `+3.867`、bus `+2.276`和
awning-bike `+1.018`；pedestrian、bike和van分别下降`1.254/1.112/1.381`。到epoch 72，
pedestrian下降收窄到`0.590`且det HOTA略超Base，但truck相对Base的增益降至`9.685`，使
选择目标低于epoch 64。结论是独立group加difference/product显著强化了cls和困难车辆类别，
并让det侧非常接近Base；但按唯一最佳checkpoint口径，仍未实现两个HOTA同时提升。

进一步按50个序列、8个类别、原始pair detection和最终track拆分诊断：0718_01在26个序列上
取得更高det HOTA、23个序列下降、1个持平；cls HOTA则在29个序列上更高。整体det HOTA的
`-0.170`中约90%来自DetA (`-0.189`)，AssA仅`-0.029`；同时FP减少993、IDSW减少24，但
FN增加649、Frag增加225。因此它没有出现0719_02式的系统性错误关联，主要代价是若干密集
序列上的召回与轨迹碎片化。`data37-1`是按GT规模加权后的主要负贡献，`data28-2`是最明确的
局部关联退化，`data39-1`则是检测置信度/覆盖下降；相对地，`data34-1`和`data30-3`的检测与
关联均改善。逐序列表、类别表及仅针对关键序列的独立track和prev/curr pair诊断视频见
[`20260720_0718_01_vs_paper_base_analysis/report.md`](20260720_0718_01_vs_paper_base_analysis/report.md)。

最终group诊断显示，0718_01没有发生组内重复波段：末态所有group仍由3个不同band组成，
没有`227`类退化；也没有发生8个group全部选择同一集合的灾难性跨group坍塌。但最后50个
hard监控点只有2种完整pattern，7/8个group完全固定，唯一变化仅为group 5在`742/432`间
切换。窗口平均`unique_sets=7.172`、`max_set_repeat=1.828`，最终pattern为
`521/432/021/132/432/431/720/610`，其中`432`被两个group复用。因此结论是：存在轻度
跨group集合重复，以及明显的输入自适应route坍塌；difference/product没有解决末期route
趋同问题。

已完成Paper协议Liquid的末期group状态统一如下：

| experiment | last-50 full patterns | intra-group duplicate | cross-group sets | route conclusion |
| --- | ---: | --- | --- | --- |
| `0716_04` group-set unique | 23 | 无 | `8.00` unique，repeat `1.00` | 无跨组坍塌，末期pattern仍有变化 |
| `0717_01` Set-Transport | 少数集中组合 | 无 | `8.00` unique，repeat `1.00` | 无跨组坍塌，但输入自适应不足 |
| `0717_02` original hard | 4 | 无 | `8.00` unique，repeat `1.00` | 无跨组坍塌，6/8 group固定 |
| `0718_01` independent diff/product | 2 | 无 | 均值`7.172`，repeat `1.828` | 轻度集合重复，7/8 group固定 |
| `0718_02` ARCR | 50 | 无 | 均值`7.570`，repeat `1.398` | 无输入route坍塌；允许轻度跨组复用 |

以上统计来自训练末期hook每次记录的当前batch首个样本；hard训练含Gumbel随机性，因此只能
作为坍塌诊断而非完整验证集分布估计。即便存在随机扰动，0718_01仍只剩2种pattern，足以
支持其route趋同严重的判断；论文若报告精确比例，应另做checkpoint的确定性全验证集route
audit。

### 8.14 Small-target Liquid structural candidates

`0717_02`证明原版Liquid主要提升truck/tricycle，但使pedestrian下降`1.514`且det HOTA下降
`0.537`。当前并行两个互补候选：252上的`0718_04 SASE`在采样前提取局部稀疏谱段证据，并
复用于LAF；99 GPU 2/3上的`0718_05 PCDP`不改变sampler，而是在融合输出处用双帧group级
可信度控制局部细节残差。两者均从`0718_03 adaptive-anchor ARCR`单变量派生，不叠加，便于
区分route选择问题和融合细节丢失问题。

`0718_05`新增321个参数，零初始化时前向严格等价；29项单测及20 iter双卡DDP验证通过，
正式实验于2026-07-18 15:06 CST fresh启动。99同时运行四卡任务，故新实验配置了全卡温控：
GPU 0--3任一卡达到90摄氏度即仅暂停GPU 2/3的0718_05进程组。两个候选都必须同时比较
cls/det HOTA及pedestrian/bike/awning-bike，最终epoch仍由唯一最大
`cls_HOTA + det_HOTA`确定，AP取同一epoch。

0718_05于15:14:29在epoch 1 iter 450后因GPU 3达到90摄氏度被守护进程暂停，训练没有报错。
日志窗口对比显示0718_03并发前/并发中/暂停后的平均`data_time`为
`0.0335/0.0329/0.0353 s`，没有I/O竞争证据；限制来自四卡并行散热。
随后该GPU 2/3尝试被主动终止并释放资源，不resume、不作为结果。同一配置之后在99
GPU 0/1于2026-07-19 07:08 CST从COCO适配权重fresh重跑，于2026-07-20 05:50 CST完成
72 epochs和18/18 TrackEval；该fresh run的唯一最佳epoch 72已进入第7节结果表。

### 8.15 Anchor-Residual Competitive Liquid on AutoDL

`0718_02`已完成72 epochs和18/18 TrackEval。严格按全部评测点中唯一最大
`cls_HOTA + det_HOTA`选择epoch 64；epoch 72仅作为训练终点参考，AP均取对应tracking
epoch，不跨epoch拼接。

| experiment | epoch | cls HOTA | det HOTA | cls MOTA | cls IDF1 | det MOTA | det IDF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Paper Base | 68 | 53.314 | 61.982 | 44.690 | 62.218 | 60.599 | 72.303 |
| `0718_02` unique best | 64 | 53.357 | 61.054 | 44.894 | 62.722 | 58.778 | 71.450 |
| change vs Base | - | +0.043 | -0.928 | +0.204 | +0.504 | -1.821 | -0.853 |
| `0718_02` endpoint | 72 | 53.118 | 61.216 | 44.515 | 62.263 | 59.349 | 71.630 |
| endpoint change vs Base | - | -0.196 | -0.766 | -0.175 | +0.045 | -1.250 | -0.673 |

| experiment | epoch | pair mAP | pair AP50 |
| --- | ---: | ---: | ---: |
| Paper Base | 68 | 0.3149 | 0.5225 |
| `0718_02` unique best | 64 | 0.3088 | 0.5218 |
| change vs Base | - | -0.0061 | -0.0007 |
| `0718_02` endpoint | 72 | 0.3094 | 0.5219 |
| endpoint change vs Base | - | -0.0055 | -0.0006 |

epoch 64逐类cls HOTA仅truck `+10.302`和tricycle `+0.582`提升；awning-bike、bike、bus、
car、pedestrian、van分别变化`-2.562/-4.116/-1.596/-0.347/-1.785/-0.140`。这说明ARCR
保留了Liquid对truck的显著收益，但固定锚点竞争和当前内容修正同时削弱了多数类别，尤其是
bike与pedestrian，最终det HOTA及det MOTA也明显下降。因此该模型不能替换Paper Base或
当前最佳Liquid。

路由目标则已实现：末50个hard监控点有50种完整pattern，平均
`image_variant_ratio=0.875`、`changed_ratio=0.538`、`unique_sets=7.570`、
`max_set_repeat=1.398`、`content_sample_std=0.262`。8个group各自仍有6--24种无序集合，
没有组内重复波段，也没有输入route固定坍塌；跨group集合偶尔复用是独立采样的允许行为。
结论是ARCR证明了无额外loss也能恢复输入自适应route，但当前锚点/残差权衡牺牲检测质量，
后续应保留其内容条件机制而放松固定锚点对有效谱段组合的限制。

AutoDL finalizer于2026-07-18 21:59 CST确认18/18评测完成。报告、日志、epoch 64、epoch 72
及完整TrackEval归档在`/autodl-fs/data/PairMOT_results/0718_02`，共享盘为权威结果源。

### 8.16 Reliability-Weighted Shared Route on 178

`0719_02`继承0719_01的pair-shared route、relaxed Set-Transport、Pair-Aligned Fusion和
PACDE，只将两帧等权log-mean-exp改为按group/slot学习相对帧可靠性的对称加权log-mixture。
该设计针对遮挡、单侧可见和局部模糊：仍由pair共同决定唯一route，但不再要求两帧对每个
route slot贡献相同。质量头共享参数且末层zero-init，故交换帧不改变结果，训练初始也严格
等价0719_01；没有新增loss或人工阈值。

41项测试通过。178上单卡batch 8的worker profile表明8 workers最优，去掉启动前10 iter后
平均`time/data_time=1.7405/0.8530 s`，优于4 workers的`2.4447/1.5367 s`和16 workers的
`2.1036/1.2235 s`。正式实验编号`0719_02`，于2026-07-19 03:02 CST在178 GPU 0从COCO
适配权重fresh启动；full HSMOT、1200x900、ordered pairs、BF16、global batch 8、学习率和
72-epoch评测协议与Paper主线一致。iter 50为`0.9082 s`，无训练异常，GPU 1未占用。

该实验已完成72 epochs和18/18 TrackEval，唯一最佳为epoch 64。相对Paper Base，cls HOTA
提高`1.876`，但det HOTA下降`1.293`。进一步用`all_cls_summary`、逐序列
`all_seq_summary`、固定score `0.2`的原始pair detection及轨迹身份连续性审计分解：DetA
下降`0.648`、AssA下降`2.198`，按`HOTA=sqrt(DetA*AssA)`计算，AssA解释`71.5%`的负向
变化；IDSW增加`350`、Frag增加`135`。与此同时pair mAP/AP50从`0.3149/0.5225`提高到
`0.3277/0.5492`。因此det掉点主要不是检测器整体能力下降，而是模型输出的时序一致性使
固定tracker产生更多身份碎裂；固定阈值下raw precision下降`2.086`、pair-link recall下降
`0.855`是次要检测侧因素。50个序列的逐序列胜负图及最差序列分析见
[`0719_02 vs Paper Base diagnosis`](20260720_0719_02_vs_paper_base_analysis/report.md)。

### 8.17 Adaptive-Anchor ARCR Final Result

99上的`0718_03`已完成72 epochs和18/18 TrackEval。严格按唯一最大
`cls_HOTA + det_HOTA`选择epoch 60，AP取同一epoch；epoch 72仅报告为训练终点。

| experiment | epoch | cls HOTA | det HOTA | cls MOTA | cls IDF1 | det MOTA | det IDF1 | pair mAP | pair AP50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Paper Base | 68 | 53.314 | 61.982 | 44.690 | 62.218 | 60.599 | 72.303 | 0.3149 | 0.5225 |
| `0718_03` unique best | 60 | **54.689** | 60.969 | **47.555** | **65.255** | 58.737 | 71.512 | **0.3152** | **0.5302** |
| change vs Base | - | +1.375 | -1.013 | +2.865 | +3.037 | -1.862 | -0.791 | +0.0003 | +0.0077 |
| `0718_03` endpoint | 72 | **54.645** | 60.969 | **47.892** | **64.969** | 58.636 | 71.176 | **0.3182** | **0.5324** |
| endpoint change | - | +1.331 | -1.013 | +3.202 | +2.751 | -1.963 | -1.127 | +0.0033 | +0.0099 |

epoch 60相对Base的逐类cls HOTA变化为truck `+14.604`、tricycle `+5.550`、bus
`-0.045`、car `-0.517`、van `-0.969`、pedestrian `-2.476`、bike `-2.515`、
awning-bike `-2.631`。cls侧DetA提高`1.553`，但class-agnostic det侧DetRe下降`1.882`、
DetA下降`1.464`，而AssA仅下降`0.200`；主要问题是高频小目标漏检，不是关联退化。

末50个hard监控点有50种完整pattern，平均`image_variant_ratio=0.875`、
`changed_ratio=0.662`、`unique_sets=7.482`、`max_set_repeat=1.450`，说明adaptive anchor
解决了输入route坍塌。它相对原ARCR最佳点将HOTA和提高`1.247`，但仍以小目标det召回换取
truck/tricycle长尾分类收益，不能替代Paper Base或当前最佳Liquid。

### 8.18 SASE-Liquid Final Result

252上的`0718_04`已完成72 epochs和18/18 TrackEval。按唯一最大
`cls_HOTA + det_HOTA`选择epoch 68，AP严格取同一epoch；epoch 72单独报告。

| experiment | epoch | cls HOTA | det HOTA | cls MOTA | cls IDF1 | det MOTA | det IDF1 | pair mAP | pair AP50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Paper Base | 68 | 53.314 | 61.982 | 44.690 | 62.218 | 60.599 | 72.303 | 0.3149 | 0.5225 |
| `0718_03` parent best | 60 | 54.689 | 60.969 | 47.555 | 65.255 | 58.737 | 71.512 | 0.3152 | 0.5302 |
| `0718_04` unique best | 68 | **53.398** | 60.928 | **45.514** | **63.131** | 59.844 | 71.689 | 0.3079 | 0.5212 |
| change vs Base | - | +0.084 | -1.054 | +0.824 | +0.913 | -0.755 | -0.614 | -0.0070 | -0.0013 |
| `0718_04` endpoint | 72 | 52.986 | 60.811 | 45.263 | 62.533 | 59.721 | 71.473 | 0.3051 | 0.5156 |
| endpoint change | - | -0.328 | -1.171 | +0.573 | +0.315 | -0.878 | -0.830 | -0.0098 | -0.0069 |

相对0718_03父实验，SASE让pedestrian、bike、awning-bike分别恢复
`1.611/0.470/0.217`，但truck、tricycle、van分别下降`5.913/2.130/2.499`。相对Base时，
这些小目标类别仍下降`0.865/2.045/2.414`，因此局部证据只实现部分恢复，没有形成净提升。

末期稀疏router/fusion gain稳定约`0.180/0.033`，分支已实际参与优化；50个监控点仍有50种
完整pattern，`image_variant_ratio=0.875`，不存在route坍塌。det侧相对Base的
`DetRe/AssA`分别下降`1.161/1.135`，说明检测召回和关联同时受损。SASE将部分能力从长尾
车辆转移给小目标，但没有超过Base，也削弱了0718_03的主要cls收益，不作为最终Liquid。

### 8.19 PACDE Strict Ablation on 252

`0719_03`用于判断0719_01/0719_02中局部细节增强是否造成det能力下降。相对0719_01，它
只关闭PACDE，pair共享route、relaxed Set-Transport、Pair-Aligned Fusion及全部论文训练
协议保持不变；不启用SASE。实验于2026-07-19 16:45 CST在252 GPU 0,1从COCO适配权重
fresh启动，于2026-07-20 18:04 CST完成72 epochs和18/18 TrackEval。唯一最大
`cls_HOTA + det_HOTA`出现在epoch 72，AP取同一epoch。

| experiment | epoch | cls HOTA | det HOTA | HOTA sum | cls MOTA | cls IDF1 | det MOTA | det IDF1 | pair mAP | pair AP50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Paper Base | 68 | 53.314 | 61.982 | 115.296 | 44.690 | 62.218 | 60.599 | 72.303 | 0.3149 | 0.5225 |
| Parent `0719_01` with PACDE | 72 | 53.883 | 61.411 | 115.294 | 46.831 | 63.920 | 59.931 | 72.366 | 0.3161 | 0.5339 |
| `0719_03` no-PACDE | 72 | **54.209** | 61.504 | **115.713** | **47.854** | **63.951** | 60.531 | 72.259 | **0.3165** | 0.5314 |
| change vs `0719_01` | - | +0.326 | +0.093 | +0.419 | +1.023 | +0.031 | +0.600 | -0.107 | +0.0004 | -0.0025 |
| change vs Base | - | +0.895 | -0.478 | +0.417 | +3.164 | +1.733 | -0.068 | -0.044 | +0.0016 | +0.0089 |

移除PACDE后两个HOTA均相对直接父实验恢复，但det只提高`0.093`，不足以解释0719_01相对
Base的`0.571`缺口。更关键的是pedestrian和awning-bike相对0719_01反而下降
`0.553/0.833`，bike下降`0.155`；PACDE确实保护了部分小目标。no-PACDE的收益主要来自
truck `+1.471`、bus `+1.139`、van `+0.571`、car `+0.508`和tricycle `+0.462`。
因此PACDE存在轻度整体代价，但它不是小目标或det下降的根因。

相对Base，`DetA/AssA/DetRe/DetPr`分别变化`-0.588/-0.182/-1.052/+0.076`，FN增加
`1684`、FP减少`1370`、IDSW减少`182`、Frag仅增加`18`。该模型输出明显更保守：精度基本
保持甚至略升，但召回不足；det退化不是关联错乱。逐类cls HOTA仍由truck `+11.269`和
tricycle `+3.216`支撑，pedestrian/bike/van分别低于Base `1.288/2.763/2.411`。

末50个hard监控点只有5种完整pattern，平均`image_variant_ratio=0.420`、
`unique_sets=6.065`、`max_set_repeat=2.000`；6个group基本固定，且`521`和`431`被跨group
复用。它没有所有group完全相同的灾难性坍塌，但pair共享route的输入自适应性已经明显减弱。
综合消融结果，剩余det缺口主要属于pair-consensus共享route及其融合主干，而不是PACDE。
该模型的HOTA和超过Base `0.417`，但未实现两个HOTA同时提升，不替换Base或`0718_01`。

### 8.20 Historical Wide-LAF + GroupMod Replay on 197

`0719_04`在Paper协议下严格复现历史`0711_01`的核心结构：每帧、每group独立执行8-group
Liquid采样，使用Wide LAF和GroupMod；不使用pair router、PairTransport、Set-Transport或
跨group集合唯一约束。实验在full HSMOT、1200x900、ordered pairs、COCO适配初始化、BF16
和global batch 8下fresh训练，于2026-07-20 16:31 CST完成72 epochs和18/18 TrackEval。

唯一最大`cls_HOTA + det_HOTA`出现在epoch 72，因此不存在最佳点和末轮分离问题，AP也取
epoch 72。

| experiment | epoch | cls HOTA | det HOTA | HOTA sum | cls MOTA | cls IDF1 | det MOTA | det IDF1 | pair mAP | pair AP50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Paper Base | 68 | 53.314 | 61.982 | 115.296 | 44.690 | 62.218 | 60.599 | 72.303 | 0.3149 | 0.5225 |
| `0719_04` | 72 | **53.932** | 60.908 | 114.840 | **45.958** | **63.723** | 59.028 | 71.362 | 0.3134 | **0.5277** |
| change vs Base | - | +0.618 | -1.074 | -0.456 | +1.268 | +1.505 | -1.571 | -0.941 | -0.0016 | +0.0052 |

检测侧分解显示`DetA/AssA/DetRe/DetPr`分别变化`-1.143/-0.812/-1.428/-0.587`；FN增加
`2893`、FP增加`351`、Frag增加`571`，但IDSW减少`163`。因此主要退化是检测召回不足，
其次是轨迹碎片与关联质量下降，并非错误身份切换大量增加。pair mAP基本不变且AP50略升，
也说明COCO式框质量摘要不能代替tracking工作点上的召回和连续性诊断。

逐类cls HOTA相对Base的变化为truck `+10.007`、tricycle `+3.384`、bus `-0.507`、
car `-0.666`、van `-1.513`、awning-bike `-1.523`、bike `-2.056`、pedestrian
`-2.182`。truck和tricycle的长尾收益使宏平均cls HOTA提高，但pedestrian、car和bike等
高频类别的下降主导class-agnostic det指标；van虽然DetA提高`1.655`，AssA却下降`6.228`，
是明确的关联侧异常类别。

末期route监控中`image_variant_ratio`通常为`0.875`，`unique_sets`约`6.88--7.62`，
`max_set_repeat`约`1.38--1.88`，`changed_ratio`约`0.63--0.67`，没有发生所有图像或所有
group选择同一路由的灾难性坍塌。结论是该结构发生了从常见/小目标检测证据向长尾谱段判别
证据的能力重分配。它可作为历史结构的严格负向复现和长尾分析样本，但HOTA和低于Base
`0.456`，也明显低于`0718_01`的`117.088`，不应作为最终Liquid主结构。

### 8.21 Fusion Quality Conservation Experiments

`0718_01`的唯一最佳点相对Base具有`+1.962/-0.170`的cls/det HOTA变化，已经保留了主要
Liquid分类收益，剩余问题集中在检测召回。`0719_03`进一步证明移除小目标PACDE不能修复
该问题，而pair共享route会产生偏保守输出。因此本轮回到`0718_01`：独立group采样、
difference/product pair router、coverage PairTransport、Wide LAF及GroupMod全部保持不变，
只约束LAF增量进入SE gate时造成的整体融合质量漂移。

设原SE logit为`z`、LAF增量为`d`。新增参数无关投影先计算原gate的局部敏感度
`sigmoid(z)*(1-sigmoid(z))`，再从`d`中移除会整体抬高或压低门控总量的分量；Liquid仍可
在8个group之间相对重分配证据。三组实验不增加loss、类别规则、route约束或可训练参数：

| ID | server | conservation | motivation | status |
| --- | --- | --- | --- | --- |
| `0720_01` | 252 GPU 0,1 | gate-mass tangent | 最弱约束，只消除SE总门控量的一阶漂移，优先保留0718_01的cls能力 | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为`54.103/61.581` |
| `0720_02` | 197 GPU 4,5 | response-weighted mass | 以Conv3D响应加权守恒，优先保护实际有响应的目标证据，直接针对FN增加 | 4-iter DDP smoke通过；正式epoch 1 iter 50为`0.849 s/iter`、`10802 MiB/rank` |
| `0720_03` | 99 GPU 0,1 | dual moment | 同时移除总门控漂移和响应相关漂移，以更强保护换取较小的Liquid自由度 | 已排在0719_06之后；GPU释放后先自动4-iter DDP smoke，再fresh训练 |

单元测试验证三种投影的约束残差、有限梯度和非零反传；252/197真实数据smoke均在BF16、
global batch 8和`find_unused_parameters=False`下完成，所有decoder、DN及encoder proposal
loss有限。三组最终仍严格按唯一最大`cls_HOTA + det_HOTA`选epoch并从同epoch记录AP；成功
标准是cls HOTA高于`53.314`且det HOTA高于`61.982`，而不是仅提高HOTA和。

### 8.22 2026-07-22 Latest Results

本节于2026-07-22 01:08 CST从各实验`metrics.json`、TrackEval summary及MMEngine
`scalars.json`交叉复核。完成实验使用全部18个评测点；运行中实验仅保留此前已记录的阶段点。

以下结果仍按唯一最大`cls_HOTA + det_HOTA`选点，AP取同一epoch。运行中实验只列截至
2026-07-21 14:35 CST已完成TrackEval的阶段最佳点，不把阶段点当作最终选点。

| experiment | status | epoch | cls HOTA | det HOTA | HOTA sum | cls MOTA | cls IDF1 | det MOTA | det IDF1 | pair mAP | pair AP50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Paper Base | completed | 68 | 53.314 | 61.982 | 115.296 | 44.690 | 62.218 | 60.599 | 72.303 | 0.3149 | 0.5225 |
| `0719_05` Base rerun, 1x8 | completed | 72 | 52.417 | 61.265 | 113.682 | 43.776 | 61.827 | 59.213 | 72.013 | 0.3001 | 0.5069 |
| `0719_06` Base + long-tail reweight | completed | 72 | **54.087** | **62.336** | **116.423** | **45.805** | **62.949** | **61.320** | **72.989** | **0.3209** | **0.5333** |
| `0720_02` response-mass Liquid | completed | 72 | 54.463 | 61.437 | 115.900 | 46.912 | 64.190 | 60.284 | 71.989 | 0.3154 | 0.5328 |
| `0720_01` gate-mass Liquid | completed | 72 | 54.103 | 61.581 | 115.684 | 46.550 | 63.824 | 59.866 | 72.155 | 0.3159 | 0.5309 |
| `0720_03` dual-moment Liquid, accuracy-fix | completed | 72 | 54.650 | 61.547 | 116.197 | 48.078 | 64.431 | 60.311 | 72.080 | 0.3231 | 0.5402 |
| `0721_01` response-mass Liquid, 1x8 accuracy-fix | completed | 72 | 54.006 | 60.934 | 114.940 | 45.921 | 63.377 | 58.475 | 70.855 | 0.3148 | 0.5297 |
| `0721_02` independent difference/product, accuracy-fix | completed | 72 | 54.327 | 61.659 | 115.986 | 47.233 | 64.123 | 60.371 | 72.246 | 0.3161 | 0.5340 |

`0719_06`是当前第一组在预注册选点规则下同时超过Paper Base两个HOTA的结果：cls/det
分别提高`+0.773/+0.354`，HOTA和提高`+1.127`；cls/det MOTA提高`+1.115/+0.721`，
cls/det IDF1提高`+0.731/+0.686`。提升不是仅靠分类宏平均：det `DetA/AssA`也分别提高
`+0.389/+0.338`，FN减少`1795`、IDSW减少`17`，代价是FP增加`397`。逐类cls HOTA主要
由truck `+4.446`、van `+2.996`和awning-bike `+0.954`推动，pedestrian仍提高`+0.361`；
tricycle下降`-3.157`、bike近似持平`-0.072`，说明长尾权重有效但仍需解决tricycle回退。

`0720_02`的HOTA和虽高于Base `+0.604`，但cls/det变化为`+1.149/-0.545`，未满足双提升
标准。det侧`DetA/AssA`分别下降`-0.575/-0.392`，FN增加`1055`；truck和tricycle的cls
HOTA大幅提高`+11.434/+5.600`，但pedestrian、bike、van分别下降
`-1.338/-3.265/-2.063`。因此response-mass conservation仍是从常见/小目标检测向长尾
判别能力重分配，不能替换Base或`0719_06`。

`0720_01`的18个点中HOTA和持续增长，唯一最大值位于末轮epoch 72，不存在挑选中期峰值的
问题。同epoch cls/det HOTA为`54.103/61.581`，相对Paper Base变化`+0.789/-0.401`，HOTA
和仅提高`+0.388`；cls/det MOTA变化`+1.860/-0.733`，cls/det IDF1变化
`+1.606/-0.148`。det分解显示`DetA -0.697`、`DetRe -0.666`，而`AssA +0.127`、IDSW
减少`80`；FN增加`915`且FP增加`602`。因此det下降来自目标覆盖和分类前检测质量，而不是
pair关联恶化。

逐类cls HOTA相对Base变化为truck `+6.594`、tricycle `+4.900`、car `+0.225`、
awning-bike `+0.170`，但van `-1.987`、pedestrian `-1.443`、bike `-1.244`、bus
`-0.906`。与直接父配置`0718_01`的各自最佳点相比，`0720_01`的cls/det HOTA进一步下降
`1.173/0.231`；即使固定比较epoch 72，也下降`0.899/0.412`。gate-mass投影削弱了父配置
最主要的truck收益`5.559`，却没有恢复常见及小目标类别，因此不应继续作为质量守恒主线。

末50个route监控点只有3种完整pattern，主pattern占31/50；平均`unique_sets=6.750`、
`max_set_repeat=2.000`、`image_variant_ratio=0.418`。它没有所有group相同的灾难性坍塌，
但输入自适应仍明显固化，gate-mass守恒也没有解决route多样性问题。

`0720_03`已完成18/18 TrackEval，HOTA和从epoch 52的`115.066`总体上升到epoch 72的
`116.197`；唯一最大点即epoch 72，同一epoch的pair mAP/AP50为`0.3231/0.5402`。相对修复前
Paper Base的数值变化为cls/det HOTA `+1.336/-0.435`、cls/det MOTA
`+3.388/-0.288`、cls/det IDF1 `+2.213/-0.223`。det侧`DetA/AssA`下降`0.355/0.429`，
`DetRe`下降`0.511`，FN增加`777`、IDSW增加`62`，说明召回和关联均有轻度退化。

逐类cls HOTA相对旧Base由truck `+9.153`、tricycle `+4.644`和bus `+1.525`推动，bike、
pedestrian、car、van、awning-bike分别下降`2.843/0.985/0.360/0.301/0.151`。相对直接父
配置`0718_01`的各自最佳点，cls/det HOTA下降`0.626/0.265`；固定epoch 72比较也下降
`0.352/0.446`。由于两者训练代码协议不同，该差值不能作为严格消融量，但没有证据支持
dual-moment优于父结构。

末50个监控点仅3种完整route，主pattern占38/50；平均`unique_sets=7.210`、
`max_set_repeat=1.778`、`image_variant_ratio=0.310`。dual-moment既没有满足双HOTA超过Base，
也没有解决输入自适应route固化，因此不作为最终Liquid候选。

`0719_05`单卡Base复跑完成18/18评测，但最佳HOTA和比主Base低`1.614`，说明单卡1x8、
软件栈及执行环境下存在不可忽略的复现偏差。`0721_01`最终唯一最佳点为epoch 72，相对同服务器
`0719_05`提高cls HOTA `+1.589`、降低det HOTA `-0.331`，HOTA和提高`+1.258`；相对主Base
则为`+0.692/-1.048`。因此response-mass在1x8拓扑下仍强化cls但牺牲det，不能替换Base。

`0721_02`最终唯一最佳点同样位于epoch 72，cls/det HOTA为`54.327/61.659`，相对Paper Base
为`+1.013/-0.323`，HOTA和提高`0.690`，仍未达到双提升。相对修复前`0718_01`各自最佳点
则下降`0.949/0.153`；该run修复了PairDN negative标签，但尚未修复Negative-DN噪声可趋近
零以及正负块被错误隔离的问题。`0722_01`同时修正外环噪声和contrastive-group attention
mask，因此不能用它单独量化外环采样的贡献。

最后，Paper Base、`0719_05/0719_06`及`0720_01/0720_02`均在2026-07-21训练准确性修复前
启动；`0720_03/0721_01`是修复后fresh run。旧协议内部比较仍可用于结构筛选，但修复后尚无
同代码Base，不能把两组协议混作严格消融。论文最终表至少需要在修复代码上重跑Base与当前
最强的Base + long-tail，之后再决定是否叠加Liquid。

### 8.23 2026-07-24 PairDN and DSE Final Results

本节从正式workdir的72轮checkpoint、18/18 TrackEval summary和MMEngine验证日志交叉
复核。仍以`cls_HOTA + det_HOTA`的唯一最大值选epoch，pair mAP/AP50严格取同一epoch；
cls与det指标分列，不用HOTA和掩盖任一侧下降。

| experiment | parent and change | epoch | cls HOTA | det HOTA | HOTA sum | cls DetA | cls AssA | det DetA | det AssA | pair mAP | pair AP50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Paper Base `0716_02` | COCO-R18、full HSMOT、1200x900 ordered pair | 68 | 53.314 | 61.982 | 115.296 | 43.386 | 68.287 | 53.890 | 73.643 | 0.3149 | 0.5225 |
| Base + long-tail `0719_06` | Base + positive classification reweight | 72 | 54.087 | 62.336 | 116.423 | 44.291 | 68.327 | 54.279 | 73.981 | 0.3209 | 0.5333 |
| Pair-coherent PairDN `0723_01` | accuracy-fixed `0718_01` + shared pair-relative DN noise、2:1 pos/neg、hard positive、rotated-IoU negative、padding隔离和representation-consistent L1 | **64** | **53.955** | **62.032** | **115.987** | 44.032 | 68.174 | 53.751 | 73.988 | 0.3114 | 0.5268 |
| Independent pair-side DN `0723_02` | `0723_01`严格消融；仅将pair两帧共享相对噪声改为独立采样 | **72** | **53.637** | **61.679** | **115.316** | 43.760 | 67.572 | 53.272 | 73.977 | 0.3155 | 0.5308 |
| Pair-coherent PairDN + DSE `0723_03` | `0723_01` + identity初始化的per-group mean/RMS fusion evidence mixer | **68** | **55.036** | **61.745** | **116.781** | 45.933 | 67.698 | 53.977 | 73.199 | 0.3239 | 0.5402 |
| Pair-coherent PairDN + CSPR `0723_04` | `0723_01` + detached 24x32 shared-Conv3D route preview | **72** | **54.523** | **61.738** | **116.261** | 44.655 | 68.675 | 53.707 | 73.567 | 0.3162 | 0.5376 |
| Pair-coherent PairDN + local CP-DSE `0723_05` | `0723_01` + bounded per-pixel group dispersion SE-logit residual | **68** | **53.536** | **61.619** | **115.155** | 43.558 | 67.864 | 53.390 | 73.649 | 0.3148 | 0.5320 |
| Pair-coherent PairDN + pair-global CP-DSE `0723_06` | `0723_01` + bounded pair-shared group dispersion SE-logit residual | **72** | **54.124** | **61.914** | **116.038** | 44.281 | 68.134 | 53.458 | 74.288 | 0.3124 | 0.5241 |
| Pair-coherent PairDN + PECG `0723_07` | `0723_01` + pair-mean-preserving contraction of SE gates at coverage/evidence-consistent locations | **52** | **53.693** | **61.151** | **114.844** | 43.764 | 67.994 | 52.663 | 73.564 | 0.3096 | 0.5262 |
| Pair-coherent PairDN + SCPD `0723_08` | `0723_01` + physical-band-coordinate pair-consensus dispersion residual | **72** | **54.465** | **61.213** | **115.678** | 44.470 | 69.321 | 53.712 | 72.176 | 0.3152 | 0.5334 |

`0723_01`相对Paper Base的cls/det HOTA分别提高`+0.641/+0.050`，是当前修复协议下第一组
不依赖类别权重、同时超过Paper Base两侧HOTA的模型结果。其det DetA下降`0.139`，但det
AssA提高`0.345`；增益主要来自更一致的pair关联，而不是增加检测响应。逐类cls HOTA中，
pedestrian、truck和tricycle分别提高`0.495/3.717/4.667`，主要下降为bike
`-0.850`和awning-bike `-1.992`。该结果尚未超过`0719_06`的两侧指标，因此不是当前绝对
最强模型。

`0723_02`相对Paper Base为`+0.323/-0.303`，也未达到双提升。更关键的是，与共享噪声父
配置固定epoch 72比较，独立采样只令cls HOTA提高`0.043`，却令det HOTA下降`0.333`。
det AssA反而微增`0.102`，主要退化来自det DetA下降`0.552`；全序列FP从`23253`增加至
`26001`，多出`2748`。这说明pair两侧独立DN噪声破坏的是检测响应精度，而不是训练出更好
的独立扰动鲁棒性，后续保持共享pair-relative DN。

`0723_03`相对Paper Base为`+1.722/-0.237`，不能认定为双提升；但其HOTA和`116.781`及
cls HOTA `55.036`均超过`0719_06`。固定epoch 68与直接父配置`0723_01`比较，DSE带来
cls/det HOTA `+1.189/-0.315`：cls DetA增加`1.912`，det DetA增加`0.134`，但det AssA
下降`0.743`。类别上truck和bus的HOTA分别增加`5.812/3.712`，而car和van分别下降
`0.775/0.796`；关联代价集中在car、van和tricycle。

`0723_04`相对Paper Base为`+1.209/-0.244`，相对父配置固定epoch 72为
`+0.929/-0.274`，同样不能认定为双提升。truck和tricycle分别提高`4.628/3.605`，但
pedestrian和van下降`0.575/2.135`。其后期det恢复未跨过Base，CSPR方向到此终止。

检查epoch 36--56的DSE权重发现，16参数自由混合很早就稳定为5个负系数，mean主路径相对
identity的平均偏离约`0.74`。因此下一方向不是继续调DSE学习率或提高响应，而是构建
consistency-preserving DSE：保留`x.mean`到SE的完整主路径，仅由归一化通道离散度生成
零初始化、幅度有界的SE-logit残差。目标是保留DSE对truck/bus及pair AP的提升，同时避免
负系数反相和大幅改变car/van/tricycle的pair排序。根据用户约束，不使用当前修改代码重跑
Paper Base；后续正式实验仍在99/197/252使用两张GPU，在178使用一张GPU。

与CP-DSE正交的`0723_07 PECG`针对det AssA：它计算每组prev/curr谱段覆盖的
Bhattacharyya一致度，以及Conv3D mean evidence的逐像素相对一致度；两者均高时，才将
两侧SE gate向pair均值收缩。修正量在prev/curr侧严格反号，因此pair平均gate逐元素保持
不变；一致度停止梯度，避免sampler通过改变route规避约束。模块仅新增8个可学习收缩强度，
不增加loss。远端精确4-iteration DDP smoke通过后，它已于2026-07-24 06:42在252 GPU
0/1 fresh启动；screen、双rank、GPU占用、正式日志迭代和有限数值五项门槛均通过。

`0723_04`的epoch 56进一步验证该诊断：对比`0723_01`同epoch，det DetA仅下降`0.111`，
det AssA却下降`1.737`，IDSW增加`535`。因此CSPR提高cls的同时破坏了跨帧排序，不适合
继续增加route自由度；PECG必须保持route本身不变。

CP-DSE到epoch 8曾出现有利分化。local `0723_05`为`44.620/49.897`，相对父配置
同epoch为`+0.308/+0.060`，det DetA/AssA分别提高`0.027/0.151`。pair-global
`0723_06`为`44.661/50.551`，相对父配置同epoch为`+0.349/+0.714`；det DetA/AssA分别
提高`0.441/1.150`，IDSW减少`179`。但pair-global在epoch 12变为`48.092/55.124`，
相对父配置仅`+0.005/-0.511`；det DetA/AssA下降`0.360/0.784`且IDSW增加62。同点AP50
基本不变，mAP下降`0.0031`。因此epoch 8收益属于尚未稳定的早期优化差异，不能直接据此
派生下一模型，需继续观察中后期。

上述结果暴露出pair-global CP-DSE的语义缺口：它按group slot共享修正，但`0723_01`允许
两帧独立route，同一slot不保证选择相同物理谱段。`0723_08 SCPD`因此先用soft coverage将
各帧group色散投影回8个物理谱段，在谱段坐标中求pair共识，再投影到各帧自己的group。
descriptor对Conv3D和sampler停止梯度，残差跨group去均值、零初始化且最大幅度受限，仅
新增8参数。它不修改route、不增加空间分支或loss。精确4-iteration smoke通过后，已于
2026-07-24 08:53在178 GPU 0 fresh启动，五项正式启动门槛全部通过。

`0723_06`的最终曲线否定了根据epoch 16作出的持续失败外推。完成72 epochs和18/18
TrackEval后，唯一最佳点为epoch 72：cls/det HOTA `54.124/61.914`，同点pair
mAP/AP50为`0.3124/0.5241`。相对Paper Base为`+0.810/-0.068`；det AssA从`73.643`
提高到`74.288`，但det DetA从`53.890`下降到`53.458`。相对父配置同为epoch 72时，
cls/det HOTA为`+0.530/-0.098`，det AssA为`+0.413`，det DetA为`-0.366`。因此
pair-global CP-DSE后期确实增强关联一致性，只是仍付出少量检测质量代价。

这与`0723_03 DSE`形成互补：DSE最佳点相对Base的det DetA/AssA为
`+0.087/-0.444`，CP-DSE则为`-0.432/+0.645`。据此新增`0725_01`，同时保留DSE的
mean/RMS局部证据混合和pair-global CP-DSE的共享SE-logit残差，不增加loss，也不从
高epoch权重resume。56项stem测试、配置deepcopy、远端哈希及双卡真实数据4-iteration
smoke通过；2026-07-25 02:45在197 GPU 4/5 fresh启动，正式epoch 1 iter 50全部数值有限。

同期阶段结果不支持继续派生PECG或SCPD：`0723_07` epoch 48为`53.211/60.805`，相对父
配置同epoch`-0.260/-0.375`；`0723_08` epoch 56为`53.891/60.333`，相对父配置同epoch
`-0.031/-1.388`。两者继续跑完用于完整消融，但当前不作为后续父结构。

`0723_08 SCPD`最终完成72 epochs和18/18 TrackEval。唯一最佳epoch 72为cls/det HOTA
`54.465/61.213`，cls DetA/AssA `44.470/69.321`，det DetA/AssA
`53.712/72.176`，cls MOTA/IDF1 `46.178/63.988`，det MOTA/IDF1
`61.188/71.986`；同点pair mAP/AP50为`0.3152/0.5334`。相对Paper Base，cls提高
`1.151`而det下降`0.769`。det DetA只下降`0.178`，det AssA下降`1.467`，因此失败主要来自
物理谱段共识扰动了pair关联排序，而不是检测覆盖崩溃；SCPD不再继续派生。

`0725_01`到epoch 12为`50.818/56.538`，相对直接父配置同epoch提高
`2.731/0.903`，det DetA/AssA分别提高`1.089/0.437`，是当前唯一在连续早期评测中同时改善
两类det分解的候选。为降低其后期发生CP-DSE式DetA回落的风险，`0725_02`在相同DSE +
pair-global CP-DSE结构中，将8个group的CP-DSE残差去均值，只保留相对group重分配而不改变
平均SE logit。单卡physical batch 8真实数据smoke在反向峰值OOM，改为physical batch 4 +
accumulation 2后保持effective batch 8并通过4次optimizer update；2026-07-25 06:57在178
GPU 0 fresh启动，正式epoch 1 iter 100的总损失、DN、encoder proposal loss及grad norm均有限。

`0725_01`到epoch 16仍保持同方向优势：cls/det HOTA为`52.058/58.339`，相对父配置同epoch
提高`2.154/0.710`；det DetA/AssA提高`1.007/0.300`，pair mAP/AP50提高
`0.0187/0.0300`。逐类cls HOTA有6/8类提高，car也由epoch 12的基本持平转为`+0.715`；
主要负项只剩awning-bike `-0.749`和van `-0.166`。该结果说明组合收益已连续跨越
epoch 8/12/16，而非单个早期点，但最终结论仍须等待18/18评测。

epoch 20进一步达到`52.556/59.200`，相对父配置同epoch提高`+1.693/+1.039`；det
DetA/AssA分别提高`1.105/0.997`，pair mAP/AP50提高`0.0197/0.0318`。与epoch 16相比，
det AssA优势从`+0.300`扩大到`+0.997`，说明组合结构目前没有重现DSE单独路径的关联损失，
而是同时保留了检测证据和pair一致性收益。

对178单卡accumulation协议的源码审计发现，MMEngine的`ParamSchedulerHook`每个
micro-iteration调用scheduler，`EMAHook`也每个micro-iteration调用averaged model；仅设置
`accumulative_counts=2`并不能保持原bs8的warmup和EMA时间尺度。`0725_02`初始run因此在
epoch 4停止且不resume，其1/18 TrackEval不进入结果表。修复版保持physical batch 4 +
accumulation 2，将warmup改为4000 micro-iter，并使用`ExpMomentumEMA(interval=2,
gamma=4000)`；这使第`2n`个micro-iter的LR对应原协议第`n`个optimizer update，并使EMA每两
个micro-iter更新一次、指数时间常数同步翻倍。57项stem测试、配置deepcopy、远端哈希和
4次optimizer-update smoke通过，2026-07-25 08:30在178 GPU0使用新workdir fresh启动。

`0723_07 PECG`最终完成72 epochs和18/18 TrackEval。按
`cls_HOTA + det_HOTA`唯一选出的epoch 52为cls/det HOTA `53.693/61.151`，
cls DetA/AssA `43.764/67.994`，det DetA/AssA `52.663/73.564`，
cls MOTA/IDF1 `44.958/63.355`，det MOTA/IDF1 `59.512/72.250`；同点pair
mAP/AP50为`0.3096/0.5262`。相对Paper Base为`+0.379/-0.831`。其中det AssA只下降
`0.079`，det DetA下降`1.227`，说明PECG确实基本守住了关联，但对两帧SE gate的收缩抑制了
有效检测响应。该方向不再派生。

`0725_01`的epoch 24进一步达到`53.393/59.910`，相对直接父配置同epoch提高
`+2.472/+1.253`；det DetA/AssA分别提高`1.189/1.404`，pair mAP/AP50提高
`0.0181/0.0293`。从epoch 8到24的六个连续评测点均保持双HOTA优势，且最新点同时扩大
检测与关联分解，当前仍是最有潜力超过Paper Base两侧指标的候选。

`0725_02` protocol-fixed版本的首个epoch 4结果为`35.079/43.891`，相对同epoch未center
组合低`3.671/0.820`，相对父配置低`1.622/1.119`。它相对父配置令det AssA提高`1.436`，
但det DetA下降`2.974`，说明简单约束CP-DSE残差跨group和为0并未在早期实现预期的质量守恒。
由于178使用单卡梯度累积且只有一个评测点，继续到epoch 8确认趋势，不用该首点提前选型。

基于PECG保持AssA但损失DetA、centered残差又使用与检测重要性无关的全1方向这一诊断，
新增`0725_03 Detection-Tangent CP-DSE`。它在`0725_01`的DSE + pair-global CP-DSE上，
用CP-DSE已计算的Conv3D group second moment与DSE base-gate的group-pooled sigmoid敏感度
形成每个pair共享的8维检测重要性向量，并从CP-DSE残差中移除沿该向量的一阶分量。剩余
7维残差仍对两帧完全共享，可继续改善关联；投影方向停止梯度，不允许主干通过改变重要性
规避约束。该结构不增加参数或loss。

实现过程中先后消除了重复的`[B,C,G,H,W]`平方归约和完整gate-map上的第二次sigmoid，
最终只复用已有second moment、增加一次group-pooled logit及8维投影。两个预优化启动均在
epoch 1 iter 100前后停止，无checkpoint/TrackEval且不进入结果。最终`fast_fresh`版本通过
58项stem测试、formal/smoke配置deepcopy、逐文件哈希和双卡4-iteration真实数据smoke；
2026-07-25 11:05在252 GPU 0/1 fresh启动，iter 100的总损失、DN、encoder proposal loss
与grad norm均有限，显存约15.95 GiB。

新增评测继续支持主组合。`0725_01` epoch 28为`53.599/60.252`，相对父配置同epoch提高
`+1.890/+0.867`；det DetA/AssA提高`0.982/0.794`，pair mAP/AP50提高
`0.0142/0.0202`。增益较epoch 24收窄但仍同时覆盖检测和关联，且epoch 8--28七个连续
评测点均为双提升。

`0725_02` epoch 8为`44.630/51.546`。相对父配置同epoch提高`+0.318/+1.709`，
det DetA/AssA提高`0.471/2.947`，否定了根据epoch 4提前终止的判断；相对未center组合则
仍低`1.812/0.452`，其中det DetA低`1.281`但det AssA高`0.457`。这与结构预期一致：
跨group去均值强化关联约束，却暂时牺牲检测响应，需继续观察中后期DetA能否恢复。

截至epoch 32，`0725_01`为`53.758/60.621`，相对纯DSE父配置同epoch仍提高
`+0.574/+0.337`；det DetA/AssA提高`0.464/0.231`，pair mAP/AP50近似不变
`-0.0001/-0.0006`。这证明组合的双HOTA收益连续覆盖epoch 8--32，但从epoch 28的
`+1.890/+0.867`明显收窄，尚不能推断其后期必然超过Base。

`0725_02` epoch 12降至相对纯DSE `-1.759/-0.275`，其中det DetA下降`1.102`、AssA提高
`0.965`；相对未center组合则为`-2.739/-0.349`。去group均值确实把优化偏向关联，但同时
削弱检测与cls，当前证据弱于未center组合，保留训练只用于确认中后期是否恢复。

`0725_03`首个epoch 4为`39.611/45.056`，相对未投影组合提高`+0.861/+0.345`；
cls/det AssA提高`2.242/1.537`，而det DetA下降`0.241`。相对纯DSE仍低
`0.514/1.590`。Detection-Tangent投影在首点达到了预期的关联保护，但尚未恢复DSE本身的
检测优势，需至少观察epoch 8/12后再判断。

`0725_01`最终完成72 epochs和18/18 TrackEval，唯一最大`cls_HOTA + det_HOTA`位于
epoch 72：cls/det HOTA为`55.126/61.998`，同epoch pair mAP/AP50为
`0.3231/0.5449`。相对Paper Base双提升`+1.812/+0.016`；det DetA下降`0.138`，
AssA提高`0.545`。相对纯DSE父配置同epoch提高`+0.328/+0.245`，证明局部DSE与pair-global
CP-DSE的检测/关联互补在完整训练后成立，但det余量仍很小。

逐类诊断显示truck、tricycle、bus分别较Base提高`9.677/5.543/2.338` HOTA，而bike和
awning-bike下降`1.139/1.257`，其中bike DetA下降`2.568`。checkpoint中的8个CP-DSE gain
有7个为负，说明pair共享分支主要通过全局抑制提高AssA，容易同时压低稀疏小目标。为此新增
`0726_01 Sparse-Reserve CP-DSE`：以现有归一化dispersion map的空间RMS减均值表示稀疏证据，
pair两帧共享该group reserve；仅衰减负向CP残差，正向修正不变。该结构无新参数、阈值或
loss。59项本地/远端测试、配置deepcopy、哈希和真实数据双卡4-iteration smoke通过；
2026-07-26 01:57在197 GPU 4/5 fresh启动，正式iter 100五项训练信号有限。
