# Module Ablation Report

更新时间：2026-07-14 14:35 CST

## 1. 新基准定义

本轮模块消融不再沿用旧的 `0704_01 resume` 作为训练初始化方式，而是保留 `0704_01` 的模型结构和训练目标，重新建立 full-data 版本：

| 实验 | 配置 | workdir | 状态 |
| --- | --- | --- | --- |
| `0714_01_0704_resume_coco365_full_unique_allgt` | `projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_coco365_full_252.py` | `/data4/litianhao/PairMmot/workdir_252/0714_01_0704_resume_coco365_full_unique_allgt` | 252 训练中，2026-07-14 14:02 已到 epoch 11 iter 50 |

这个实验作为后续 `+long-tail`、`+liquid`、`+encoder`、`+decoder` 的统一模块消融 baseline。

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

因此，当前不采用会改变数值行为的 assigner 快速路径；AMP 仍不接入正式训练。`find_unused_parameters=False` 已可作为后续训练加速选项，但切换正式实验前应在目标结构配置上先做同样的 40 iter smoke test。如果后续继续推进 AMP，需要先对 `pair_rotated_rtdetr_head.py` 中所有 `loss_iou` / `GDLoss` 路径做 FP32 island。

后续模块消融建议按同一个新 baseline 逐项累加：

| 阶段 | 定义 |
| --- | --- |
| baseline | `0714_01`：0704_01 结构 + COCO/O365 direct init + full train |
| `+long-tail` | 在 baseline 上加入类别长尾修复模块 |
| `+liquid` | 在 baseline 或 `+long-tail` 上加入 liquid spectral 模块 |
| `+encoder` | 在前序最佳设置上加入 encoder 结构改动 |
| `+decoder` | 在前序最佳设置上加入 decoder 结构改动 |

报告比较时不合并指标展示；每个实验仍按 `cls_HOTA + det_HOTA` 选择唯一最佳 epoch。
