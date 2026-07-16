# Module Ablation Report

更新时间：2026-07-16 CST

## 1. 新基准定义

本轮模块消融不再沿用旧的 `0704_01 resume` 作为训练初始化方式，而是保留 `0704_01` 的模型结构和训练目标，重新建立 full-data 版本：

| 实验 | 配置 | workdir | 状态 |
| --- | --- | --- | --- |
| `0714_01_0704_resume_coco365_full_unique_allgt` | `projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_coco365_full_252.py` | `/data4/litianhao/PairMmot/workdir_252/0714_01_0704_resume_coco365_full_unique_allgt` | 已完成 72 epochs 和全部 18 个 TrackEval 点；唯一最佳 HOTA 和 AP 最优均在最后一个评测周期 |

这个实验作为后续 `+long-tail`、`+liquid`、`+encoder`、`+decoder` 的统一模块消融 baseline。

### 1.1 2026-07-15 后续实验统一训练基准

从 2026-07-15 起，新启动的实验统一以 99 上已完成的
`0715_01_0704_01_half_unique_allgt_bf16_encoder_findfalse` 作为训练配置模板：

- `AmpOptimWrapper(dtype='bfloat16', loss_scale=1.0)`；
- backbone、neck 和 shared RT-DETR encoder 使用 BF16，encoder 输出只转换一次 FP32；
- query initialization、decoder、head、matching 和 loss 保持 FP32；
- DDP 固定使用 `find_unused_parameters=False`；新结构必须保证所有应训练参数具有有效梯度；
- 保留 validation、detection evaluation 和 TrackEval，只关闭图像绘制；
- 默认 fresh train、`resume=False`。初始化权重和 half/full 数据集仍由具体消融任务决定，不因 AMP 模板而强制统一。

对 half-data、原始 `0704_01` 结构的后续实验，性能基准改为 `0715_01` 的唯一最佳
HOTA 点：`cls_HOTA=46.531`、`det_HOTA=58.484`、两者之和 `105.015`。历史实验及其
既有报告继续保留 `0704_01 resume` 对照，避免回溯性改写结论。

## 2. 与旧 0704_resume 的差异

| 项目 | 旧 `0704_01 resume` | 新 `0714_01` |
| --- | --- | --- |
| 初始化来源 | HSMOT 单帧检测模型适配到 PairMOT | COCO+Objects365 RT-DETR R18vd 直接适配到 PairMOT |
| 训练数据 | `train_half.txt`，29 个训练序列 | `ann_file=None` 自动扫描 full train，252 上为 75 个训练序列 |
| 模型结构 | `unique_pair_selection + PairDN + dual-cls + no-presence + all-GT` | 不变 |
| 评测 | pair AP + track eval | 不变；只关闭 val 可视化绘图 |

## 3. COCO+O365 权重适配

源权重：

`/data4/litianhao/PairMmot/pretrained_weights/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth`

输出权重：

`/data4/litianhao/PairMmot/pretrained_weights/rtdetr_r18vd_5x_coco_objects365_pair_unique_allgt_full/pair_coco365_full_adapted_pretrain.pth`

适配统计：

| 项目 | 数量 |
| --- | ---: |
| 源 state 来源 | `ema.module` |
| 源 tensor | 533 |
| 目标 tensor | 610 |
| 生成候选 tensor | 542 |
| 完全 shape 匹配并复制 | 384 |
| Conv3D stem 适配 | 1 |
| 部分复制 | 13 |
| shape mismatch 跳过 | 0 |
| target 缺失跳过 | 144 |

主要规则：

| 模块 | 适配方式 |
| --- | --- |
| Backbone R18vd | Paddle/PPHGNet key 映射到本项目 `backbone.stem` 和 `backbone.layer{1..4}` |
| Conv3D stem | `backbone.conv1.conv1_1.conv.weight` 的 RGB 2D kernel 先按输入通道求均值，再扩展为 `(32, 1, 3, 3, 3)`，temporal 维复制后除以 depth，保持响应尺度 |
| Neck input projection | `encoder.input_proj.*` 映射到 `neck.convs.*` |
| Hybrid encoder | transformer block、FPN/PAN 中命名一致且 shape 匹配的层映射到 `encoder.*` |
| Decoder | `decoder.decoder.layers.*` 映射到 `decoder.layers.*`；self-attn、FFN、norm 直接复制 |
| Pair cross-attn | 源 `cross_attn` 同时复制到 `cross_attn_prev` 和 `cross_attn_curr` |
| Regression head | `dec_bbox_head` 映射到 prev/curr `bbox_head.reg_branches`；最后 4-d box 输出部分复制到 5-d rotated box head，angle 维保留目标初始化 |
| Classification head | COCO/O365 类空间与 HSMOT 8 类不一致，分类 logits 不加载 |
| Pair-only 参数 | `pair_quality_predictor`、`pair_query_fusion`、`pair_dn_query_generator` 等保持目标初始化 |

部分复制的 key 包括 `decoder.ref_point_head.layers.0.weight` 以及前 3 个 decoder layer 的 `reg_branches.*.4.{weight,bias}` / `reg_branches_curr.*.4.{weight,bias}`。目标第 4 个 decoder/head layer 没有源模型对应层，保持初始化。

### 3.1 R18/R34/R50 COCO 统一预训练（2026-07-16）

后续正式模型族改用官方 COCO-only RT-DETR 权重，三种规模使用同一预训练数据来源：

| backbone | 官方 checkpoint | decoder | 完全复制 | Conv3D | 4D→5D 部分复制 | 未匹配候选 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| R18 | `rtdetr_r18vd_dec3_6x_coco_from_paddle.pth` | 3 | 470 | 1 | 17 | 0 |
| R34 | `rtdetr_r34vd_dec4_6x_coco_from_paddle.pth` | 4 | 588 | 1 | 21 | 0 |
| R50 | `rtdetr_r50vd_6x_coco_from_paddle.pth` | 6 | 763 | 1 | 29 | 0 |

文件统一位于 `/data4/litianhao/PairMmot/pretrained_weights`，下载来源、SHA256、目标配置、
输出 checkpoint 和逐 key 适配统计记录在
`rtdetr_coco_pair_family_manifest.json`。生成入口为
`projects/multispec_pair_rotated_rtdetr/tools/prepare_coco_pair_family_pretrain.py`。

新版适配器不写死 backbone 深度或 decoder 层数：R18/R34 BasicBlock、R50 Bottleneck、
FPN/PAN、shared encoder 和 3/4/6 层 decoder 均按目标结构适配。源 RGB stem 的
`(C,3,3,3)` 权重直接重排为 Conv3D `(C,1,3,3,3)`，不做平均或缩放；旋转框新增 angle
维保持目标初始化。Liquid sampler/fusion、temporal encoder、tristate decoder 等目标独有
扩展不伪造源参数，保持各自初始化，并在 JSON 的 `target_only_keys` 中记录。

注意：第 3 节记录的是 `0714_01` 已实际使用的旧 COCO+Objects365 适配文件，不能用新版
统计回溯替换。复查发现旧生成器曾遗漏 FPN/PAN block 映射，且 Conv3D stem 使用了均值缩放；
新版已修复并通过逐 tensor 加载检查。因此后续严格模块消融应以本节 COCO-only 新权重重新
训练统一 baseline，`0714_01` 保留为历史 full-data 性能锚点。

## 4. 启动与记录规则

252 使用路径：

| 项目 | 路径 |
| --- | --- |
| 代码 | `/data/users/litianhao01/PairMmot/ai4rs` |
| 数据 | `/data/users/litianhao01/PairMmot/data/hsmot` |
| workdir | `/data4/litianhao/PairMmot/workdir_252/0714_01_0704_resume_coco365_full_unique_allgt` |

启动前为 full train 补齐了真实 GMC cache：

| cache | 路径 | 内容 |
| --- | --- | --- |
| train gap=1 | `/data/users/litianhao01/PairMmot/workdir/aux/gmc_cache/hsmot_train_gap1` | `build_hsmot_gmc_cache.py` sparse LK + RANSAC 构建，75 个 train 序列，8297 个 pair |
| test gap=1 | `/data/users/litianhao01/PairMmot/workdir/aux/gmc_cache/hsmot_test_gap1` | 沿用既有 test cache |

注意：曾短暂尝试过 identity fallback cache，已删除，当前 train cache 中没有 `identity_fallback` 文件。

训练启动记录：

| 项目 | 值 |
| --- | --- |
| launch log | `/data4/litianhao/PairMmot/workdir_252/0714_01_0704_resume_coco365_full_unique_allgt/launch.log` |
| 当前 run log | `/data4/litianhao/PairMmot/workdir_252/0714_01_0704_resume_coco365_full_unique_allgt/20260714_104533/20260714_104533.log` |
| GPU | 252: `CUDA_VISIBLE_DEVICES=0,1` |
| 首条训练日志 | epoch 1 iter 50，loss 27.8584 |
| 最新检查 | 2026-07-14 14:02，epoch 11 iter 50，loss 14.6145，训练进程仍在运行 |

COCO/O365 直接初始化在训练初期可能让 GWD/KLD match cost 出现非有限值；`PairHungarianAssigner` 已加入数值防护：对单项或汇总 cost 中的 NaN/Inf，用当前 finite 最大 cost 加大惩罚替换，避免 scipy Hungarian 直接退出。这个防护不跳过样本、不关闭任何评测，只把无效候选设为极大代价。

### 4.1 单卡训练 step profile

2026-07-14 在 252 的空闲 GPU2 上，用临时配置 `tmp_profile_0714_coco365_full_single_gpu.py` 对当前 full-data baseline 做 40 iter 单卡 profile。该 profile 关闭验证、保留真实 train batch、打开 `PairComponentTimerHook` 和 assigner cost 子项计时，不影响正式训练。

稳定版本均值：

| 组件 | 单 iter 耗时 |
| --- | ---: |
| iter_wall | 0.9238s |
| backward_opt | 0.4712s |
| head_loss | 0.2838s |
| backbone_neck | 0.0883s |
| encoder | 0.0304s |
| query_init | 0.0325s |
| decoder | 0.0173s |
| assign_focal | 0.0089s |
| assign_chamfer_prev | 0.0244s |
| assign_chamfer_curr | 0.0234s |
| assign_gd_prev | 0.0367s |
| assign_gd_curr | 0.0359s |
| assign_cpu_copy | 0.0009s |
| assign_cpu_hungarian | 0.0030s |

结论：

| 观察 | 判断 |
| --- | --- |
| 最大项是 `backward_opt`，约占 51% | 主要瓶颈不是 dataloader，也不是 scipy Hungarian 本身 |
| `head_loss` 约占 31%，其中 assigner cost 约 0.13s | Pair OBB matching 有成本，但不是唯一主瓶颈 |
| `data_time` 约 0.03-0.08s | 增加 worker 对总时长帮助有限 |
| 尝试把 invalid side 也送入 GDCost 后再 mask | 会引入 NaN，已回退 |
| 尝试异步化 non-finite sanitize | 早期训练出现 NaN，已回退 |

### 4.2 AMP 与 find_unused_parameters 单卡尝试

继续在 252 的 GPU2 上做短程单卡测试，全部使用 40 iter 临时 profile 配置，不影响正式 `0714_01` 训练。

| 尝试 | 配置 / workdir | 结果 | 判断 |
| --- | --- | --- | --- |
| DDP 单卡，`find_unused_parameters=True` | `tmp_profile_0714_coco365_full_single_gpu.py`，`tmp_profile_0714_coco365_full_single_gpu_ddp_findunused_true` | 完成 40 iter，均值 `iter_wall=0.9497s`、`backward_opt=0.4915s` | 比非 DDP 单卡 `0.9238s/iter` 慢约 2.8%，开销不大 |
| DDP 单卡，`find_unused_parameters=False`，修复前 | `tmp_profile_0714_coco365_full_single_gpu_findunused_false.py`，`tmp_profile_0714_coco365_full_single_gpu_ddp_findunused_false` | 第 2 个 iter 报 DDP unused parameter，未参与梯度参数包括 learned query/ref 和 `pair_quality_predictor` | 当前模型存在条件分支或未激活分支，不能直接关闭 |
| DDP 单卡，`find_unused_parameters=False`，冻结 proposal head 诊断 | `tmp_profile_0714_coco365_full_single_gpu_findunused_false.py`，`tmp_profile_0714_findunused_false_after_paramfix_v2` | 完成 40 iter，均值 `iter_wall=0.9212s`、`backward_opt=0.4696s` | 只能说明 unused 来自结构路径；不能作为最终 baseline，因为会切断 encoder proposal supervision |
| DDP 单卡，`find_unused_parameters=False`，单帧 encoder aux 试验 | `tmp_profile_0714_coco365_full_single_gpu_findunused_false.py`，`tmp_profile_0714_findunused_false_encoderaux_topk_validmask` | 完成 40 iter，均值 `iter_wall=0.9753s`、`backward_opt=0.4848s` | 已废弃：该试验把 encoder aux 改成每帧独立 proposal 监督，破坏了 PairMOT 必须由 pair proposal 与 pair GT 建立联系的设定 |
| AMP 单卡 | `tmp_profile_0714_coco365_full_single_gpu_amp.py`，`tmp_profile_0714_coco365_full_single_gpu_amp` | 已修复 matcher 阶段 dtype mismatch，并强制 PairGDCost 以 FP32 计算；随后训练 loss 阶段仍在 `GDLoss` 触发 `lu_factor_cublas not implemented for Half` | AMP 不能直接接入；需要把 head 里的 GDLoss/KLD loss 全部显式 FP32 后再测 |

DDP `find_unused=True` 与非 DDP profile 对比：

| 组件 | 非 DDP 单卡 | DDP 单卡 `find_unused=True` | DDP 单卡 `find_unused=False` 单帧 aux 试验 |
| --- | ---: | ---: | ---: |
| iter_wall | 0.9238s | 0.9497s | 0.9753s |
| backward_opt | 0.4712s | 0.4915s | 0.4848s |
| head_loss | 0.2838s | 0.2891s | 0.3118s |
| backbone_neck | 0.0883s | 0.0910s | 0.0903s |
| encoder | 0.0304s | 0.0306s | 0.0304s |
| query_init | 0.0325s | 0.0296s | 0.0391s |
| decoder | 0.0173s | 0.0175s | 0.0184s |
| assign cost 合计 | 0.1541s | 0.1639s | 0.1467s |

结论更新：单帧 encoder aux 试验虽然能绕开 DDP unused 报错，但与 PairMOT 的核心监督目标不一致，已从代码中撤回。当前保留的合理修正只有 `_single_frame_topk_proposals(_v2)` 在 top-k 前屏蔽 invalid proposal；`pair_topk_v2` / `typed_pair_topk_v1` 的 encoder aux loss 仍监督 pair-selected proposal，使 proposal 与 pair GT 通过同一个 pair assigner 建立联系。

因此，当时的阶段性结论是不采用会改变数值行为的 assigner 快速路径，且在完成 FP32
island 前不把 AMP 接入正式训练。后续修正和完整训练结果见第 5 节；
`find_unused_parameters=False` 已通过目标结构验证。

后续模块消融建议按同一个新 baseline 逐项累加：

| 阶段 | 定义 |
| --- | --- |
| baseline | `0714_01`：0704_01 结构 + COCO/O365 direct init + full train |
| `+long-tail` | 在 baseline 上加入类别长尾修复模块 |
| `+liquid` | 在 baseline 或 `+long-tail` 上加入 liquid spectral 模块 |
| `+encoder` | 在前序最佳设置上加入 encoder 结构改动 |
| `+decoder` | 在前序最佳设置上加入 decoder 结构改动 |

报告比较时不合并指标展示；每个实验仍按 `cls_HOTA + det_HOTA` 选择唯一最佳 epoch。

## 5. AMP 完整训练性能核验

本节只纳入完成 72 epochs 且保持正式评测的两个实验：99 的 `0715_01` BF16-through-encoder
和 197 的 `0714_03` corrected hybrid FP16。197 的 `0714_02` 改变过 GDLoss 数学实现，
不参与比较。baseline 使用 `0704_01 resume` 高指标。

Tracking 严格按 `cls_HOTA + det_HOTA` 从每个实验全部 18 个评测点中选择唯一最佳 epoch：

| experiment | precision | unique track point | cls HOTA | det HOTA | cls+det | delta cls | delta det | delta sum |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `0704_01 resume` | FP32 baseline | async 7 / val_det epoch 67 | 45.523 | 58.120 | 103.643 | 0.000 | 0.000 | 0.000 |
| `0715_01` 99 | BF16 through encoder | async 18 / val_det epoch 71 | 46.531 | 58.484 | 105.015 | +1.008 | +0.364 | +1.372 |
| `0714_03` 197 | FP16 backbone/neck, FP32 transformer | async 16 / val_det epoch 63 | 46.271 | 58.381 | 104.652 | +0.748 | +0.261 | +1.009 |

AP 单独按 `pair_mAP50:95` 选择，不与上表的 HOTA epoch 拼接：

| experiment | AP epoch | pair mAP | pair AP50 | both mAP | both AP50 | delta pair mAP |
|---|---:|---:|---:|---:|---:|---:|
| `0704_01 resume` | 68 | 0.2383 | 0.4157 | 0.2448 | 0.4275 | 0.0000 |
| `0715_01` 99 | 72 | 0.2445 | 0.4257 | 0.2514 | 0.4382 | +0.0062 |
| `0714_03` 197 | 72 | 0.2424 | 0.4215 | 0.2492 | 0.4337 | +0.0042 |

结论：两个正式 AMP 实验在 HOTA 和 AP 上均未出现相对 baseline 的性能下降。99 BF16
观察结果最好，也支持保留 BF16 方案；197 FP16 虽然指标未下降，但由于此前的数值稳定性
风险，不恢复为默认精度。需要注意，这不是严格的同 seed 精度消融：AMP 实验是 fresh
rerun，且包含同期的 DDP/KLD 稳定性修正，因此可以得出“没有观察到退化”，不能把观察到
的提升全部归因于 AMP。

分项上唯一明确下降是 197 FP16 最佳 HOTA 点的 `cls_MOTA=34.208`，比 baseline
`34.750` 低 `0.542`；但同一点的 cls HOTA、cls IDF1 以及全部 det tracking 指标均提高。
99 BF16 的 cls/det HOTA、MOTA、IDF1 六项则全部高于 baseline。

基于完整训练的稳定性和性能核验，`0715_01` 的 BF16-through-encoder 边界从
2026-07-15 起成为所有新实验的默认精度与 DDP 配置。FP32 `0704_01 resume` 仅保留为
历史指标锚点，不再作为新实验的默认启动配置。

## 6. Full-data COCO+Objects365 baseline 完整结果

252 的 `0714_01_0704_resume_coco365_full_unique_allgt` 已完成 72 epochs。Tracking 从全部
18 个评测点中严格按 `cls_HOTA + det_HOTA` 选择唯一最佳：`val_track_0018`，对应
`val_det epoch 71`。

| track point | cls HOTA | det HOTA | cls+det | cls MOTA | cls IDF1 | det MOTA | det IDF1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `val_track_0018 / val_det epoch 71` | 52.374 | 60.318 | 112.692 | 44.159 | 62.126 | 57.407 | 70.957 |

AP 单独按 `pair_mAP50:95` 选择，最优为训练 epoch 72，不与上面的 tracking epoch 合并：

| AP epoch | pair mAP | pair AP50 | both mAP | both AP50 |
|---:|---:|---:|---:|---:|
| 72 | 0.2928 | 0.5062 | 0.3011 | 0.5209 |

相对历史 half-data `0704_01 resume`，full-data baseline 的 `cls_HOTA` 提升 `+6.851`，
`det_HOTA` 提升 `+2.198`，两者之和提升 `+9.049`。相对 99 的 half-data BF16
`0715_01`，对应提升为 `+5.843`、`+1.834` 和 `+7.677`。该增益来自 29 到 75 个训练
序列、COCO+Objects365 直接适配初始化以及 fresh 72-epoch 训练的联合变化，不能归因于
单一因素。

该历史 full-data 实验本身使用 FP32 `OptimWrapper` 和
`find_unused_parameters=True`。它是当前 full-data 性能锚点；2026-07-15 后新启动的
full-data 模块消融仍应使用统一的 BF16-through-encoder、
`find_unused_parameters=False` 训练配置，并在报告中注明这一训练配置差异。

### 6.1 Full-data Liquid 候选结果

99 的 `0715_05_liquid8_final_pairtransport_paironly_coco365_full_bf16` 已完成 72 epochs 和
全部 18 个 TrackEval 点。它使用 8-group sampler、pair-conditioned router、wide
overlap-aware LAF、group modulation 和 coverage-based pair transport；两个关系 MLP
只消费有序 `[x,y]`。Tracking 唯一最佳为最终 `val_track_0018 / step 71`：

| experiment | cls HOTA | det HOTA | cls MOTA | cls IDF1 | det MOTA | det IDF1 |
|---|---:|---:|---:|---:|---:|---:|
| full baseline `0714_01` | 52.374 | 60.318 | 44.159 | 62.126 | 57.407 | 70.957 |
| `+liquid` candidate `0715_05` | 53.472 | 60.907 | 44.951 | 62.704 | 58.652 | 71.215 |
| delta | +1.098 | +0.589 | +0.792 | +0.578 | +1.245 | +0.258 |

AP 独立最优均为 epoch 72：full baseline pair mAP `0.2928`、pair AP50 `0.5062`；
`0715_05` pair mAP `0.2988`、pair AP50 `0.5115`。

该结果支持 full liquid 候选整体优于 full baseline，但当前只能列为 `+liquid candidate`，
不能视为严格单模块消融：baseline 为 FP32 + `find_unused=True`，Liquid 为 BF16 through
encoder + `find_unused=False`，且后者包含同期稳定性修正。严格模块链结论需要补跑同一
BF16 代码版本、相同 seed 的 full-data baseline。

## 相关独立报告

Proposal affinity 的 zero-shot 分析不属于 baseline 模块逐项训练消融主线，已迁移至独立报告：

- [Proposal Zero-shot: Elliptical Motion + Spectral Similarity](20260716_proposal_zeroshot_elliptical_spectral_report.md)
