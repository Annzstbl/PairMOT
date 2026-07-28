# 20260713 Encoder Temporal Adapter HOTA Report

## 1. 对比对象

本报告汇总 encoder temporal adapter 系列实验，重点看 HOTA。Baseline 固定使用 252 上补齐的高指标 `0704_01 resume`，即非 temporal encoder 的 `unique + PairDN + all-GT` 强 baseline。

选择规则：

- 不合并不同 epoch 的单项最优。
- 每个实验统一按 `cls_HOTA + det_HOTA` 选择唯一最佳 epoch。
- AP 只作为诊断参考，不作为主排序依据。

| 实验 | 角色 | 服务器/路径 |
|---|---|---|
| `0704_01 resume` | 非 temporal encoder baseline，`unique + PairDN + all-GT`，从 epoch 40 续训到 72 | `workdir_252/0704_01...resume_from_epoch40_to72` |
| `0704_02 p5temporal` | P5 全局 pair temporal adapter | 252: `/data/users/litianhao01/PairMmot/workdir/0704_02...p5temporal` |
| `0705_01 p5temporal_pyramidlocal` | P5 全局 temporal + post-FPN pyramid-local temporal | 252: `/data/users/litianhao01/PairMmot/workdir/0705_01...p5temporal_pyramidlocal` |
| `0705_02 pyramidlocal` | 只使用 post-FPN pyramid-local temporal，levels 0/1/2 | 99: `/data4/litianhao/PairMmot/workdir_99/0705_02...pyramidlocal` |
| `0705_03 pyramidlocal_p4p5` | 只在 P4/P5 使用 post-FPN pyramid-local temporal | 252: `/data/users/litianhao01/PairMmot/workdir/0705_03...pyramidlocal_p4p5` |
| `0705_04 pyramidlocal_p4p5_slowgate` | `0705_03` 的 slow-gate / slower adapter LR 版本 | 252: `/data/users/litianhao01/PairMmot/workdir/0705_04...pyramidlocal_p4p5_slowgate` |

说明：`0705_02` 在 252 原 workdir 未找到可用 scalars，本报告使用 99 上已有的可用结果；该项作为结构对照保留。

## 2. 实验内容与结构改动

所有 encoder 实验都继承 `0704_01` 的 proposal、matching、PairDN、loss、all-GT 监督和验证设置。也就是说，这组实验只讨论 encoder 侧的时序特征交互，不把 decoder、matching 或训练目标变化混进结论。

结构关系：

```text
0704_01 resume
  -> 0704_02 P5 temporal adapter
      -> 0705_01 P5 temporal + pyramid-local
  -> 0705_02 pyramid-local only, P3/P4/P5
      -> 0705_03 pyramid-local only, P4/P5
          -> 0705_04 pyramid-local P4/P5 slowgate
```

| 实验 | 相对来源 | 具体改动 | 设计意图 | 不变项 |
|---|---|---|---|---|
| `0704_01 resume` | `0704_01` 原实验从 epoch 40 续训到 72 | 无 encoder temporal 新结构 | 提供同监督目标、同训练长度的强 baseline | proposal、matching、PairDN、loss、all-GT |
| `0704_02 p5temporal` | `0704_01 resume` | 在 shared AIFI encoder 后、FPN/CCFF 前，对 P5 加 pair temporal adapter；残差 `gamma_init=0` | 在高语义层建立前后帧全局时序交互，同时保持初始路径等价 baseline | decoder、proposal、matching、PairDN、loss、all-GT |
| `0705_01 p5temporal_pyramidlocal` | `0704_02` | 保留 P5 全局 temporal，并在 FPN/CCFF 后加入 `pyramid_local` adapter，levels `[0,1,2]` | 同时利用 P5 全局语义交互和多尺度局部时序对齐 | decoder、proposal、matching、PairDN、loss、all-GT |
| `0705_02 pyramidlocal` | `0704_01 resume` | 移除 P5 全局 MHA，只保留 post-FPN `pyramid_local` adapter，levels `[0,1,2]` | 验证多尺度局部 temporal 是否单独有效 | decoder、proposal、matching、PairDN、loss、all-GT |
| `0705_03 pyramidlocal_p4p5` | `0705_02` | 将 `pyramid_local` 限制到 levels `[1,2]`，即 P4/P5 | 测试低层 P3 局部 temporal 是否带来噪声；强调更语义化尺度 | decoder、proposal、matching、PairDN、loss、all-GT |
| `0705_04 pyramidlocal_p4p5_slowgate` | `0705_03` | 保持 P4/P5 pyramid-local 结构，但降低 adapter/gamma 的相对学习速度 | 测试更保守的 temporal gate 是否提升训练稳定性和 det-side 表现 | decoder、proposal、matching、PairDN、loss、all-GT |

## 3. 唯一最佳点

| 实验 | 最佳 epoch | cls HOTA | cls MOTA | cls IDF1 | det HOTA | det MOTA | det IDF1 | cls HOTA + det HOTA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `0704_01 resume` | 67 | 45.523 | 34.750 | 52.845 | 58.120 | 51.956 | 66.997 | 103.643 |
| `0704_02 p5temporal` | 59 | 46.839 | 35.608 | 54.971 | 58.056 | 52.734 | 67.232 | 104.895 |
| `0705_01 p5temporal_pyramidlocal` | 55 | 47.073 | 36.619 | 55.106 | 58.351 | 52.499 | 67.292 | 105.424 |
| `0705_02 pyramidlocal` | 67 | 46.722 | 34.882 | 54.351 | 58.421 | 52.761 | 67.554 | 105.143 |
| `0705_03 pyramidlocal_p4p5` | 71 | 46.999 | 35.793 | 54.723 | 58.344 | 52.762 | 67.488 | 105.343 |
| `0705_04 pyramidlocal_p4p5_slowgate` | 63 | 46.929 | 35.829 | 54.654 | 58.373 | 52.867 | 67.405 | 105.302 |

相对 `0704_01 resume`：

| 实验 | delta cls HOTA | delta det HOTA | delta sum | delta cls IDF1 | delta det IDF1 |
|---|---:|---:|---:|---:|---:|
| `0704_02 p5temporal` | +1.316 | -0.064 | +1.252 | +2.126 | +0.235 |
| `0705_01 p5temporal_pyramidlocal` | +1.550 | +0.231 | +1.781 | +2.261 | +0.295 |
| `0705_02 pyramidlocal` | +1.199 | +0.301 | +1.500 | +1.506 | +0.557 |
| `0705_03 pyramidlocal_p4p5` | +1.476 | +0.224 | +1.700 | +1.878 | +0.491 |
| `0705_04 pyramidlocal_p4p5_slowgate` | +1.406 | +0.253 | +1.659 | +1.809 | +0.408 |

按 `cls_HOTA + det_HOTA` 排序：

| rank | 实验 | epoch | sum | vs baseline |
|---:|---|---:|---:|---:|
| 1 | `0705_01 p5temporal_pyramidlocal` | 55 | 105.424 | +1.781 |
| 2 | `0705_03 pyramidlocal_p4p5` | 71 | 105.343 | +1.700 |
| 3 | `0705_04 pyramidlocal_p4p5_slowgate` | 63 | 105.302 | +1.659 |
| 4 | `0705_02 pyramidlocal` | 67 | 105.143 | +1.500 |
| 5 | `0704_02 p5temporal` | 59 | 104.895 | +1.252 |
| 6 | `0704_01 resume` | 67 | 103.643 | 0.000 |

## 4. AP 诊断

| 实验 | AP epoch | pair mAP | pair AP50 | both mAP | both AP50 |
|---|---:|---:|---:|---:|---:|
| `0704_01 resume` | 68 | 0.2383 | 0.4157 | 0.2448 | 0.4275 |
| `0704_02 p5temporal` | 52 | 0.2420 | 0.4272 | 0.2488 | 0.4398 |
| `0705_01 p5temporal_pyramidlocal` | 68 | 0.2451 | 0.4326 | 0.2519 | 0.4449 |
| `0705_02 pyramidlocal` | 64 | 0.2445 | 0.4249 | 0.2512 | 0.4369 |
| `0705_03 pyramidlocal_p4p5` | 72 | 0.2473 | 0.4311 | 0.2541 | 0.4432 |
| `0705_04 pyramidlocal_p4p5_slowgate` | 72 | 0.2484 | 0.4346 | 0.2553 | 0.4473 |

AP 侧与 HOTA 侧方向一致但排序不同。`0705_04` 的 AP 最强，说明 slowgate 对检测置信排序和框质量有帮助；但按 tracking HOTA，`0705_01` 仍是唯一最佳。

## 5. HOTA 结论

### 5.1 最佳 encoder 版本

`0705_01 p5temporal_pyramidlocal` 是当前 encoder 系列最佳点。相对 `0704_01 resume`：

- `cls_HOTA +1.550`
- `det_HOTA +0.231`
- 综合分 `+1.781`
- `cls_IDF1 +2.261`
- `det_IDF1 +0.295`

这说明 P5 全局 temporal 与 post-FPN 多尺度局部 temporal 是互补的。P5 全局分支主要提升 cls-side 的时序身份和语义一致性，pyramid-local 分支补上 det-side 与局部对齐收益。

### 5.2 P5 temporal 单独有效，但 det 侧不增

`0704_02 p5temporal` 相对 baseline 的综合分提升 `+1.252`，其中 `cls_HOTA +1.316`、`cls_IDF1 +2.126` 很明显。但 `det_HOTA -0.064`，说明只在 P5 做全局 temporal 更像是在改善类别/身份侧表达，而不是改善检测侧空间对齐。

### 5.3 pyramid-local 单独也有效

`0705_02 pyramidlocal` 相对 baseline 综合分 `+1.500`，且 `det_HOTA +0.301`、`det_IDF1 +0.557`。这说明 post-FPN 局部 temporal adapter 可以独立改善 detection-side tracking，不依赖 P5 全局 temporal。

### 5.4 P4/P5 比全尺度 pyramid-local 更适合 HOTA

`0705_03 pyramidlocal_p4p5` 比 `0705_02 pyramidlocal` 的综合分高 `+0.200`，主要来自 `cls_HOTA +0.277`。这提示低层 P3 的局部 temporal 可能引入噪声，或者对类别/身份侧帮助不如更语义化的 P4/P5。

需要注意，`0705_02` 使用的是 99 上可用结果，和 252 上的其他实验不是完全同一次运行环境，因此这条结论应作为趋势判断。

### 5.5 slowgate 改善 det/AP，但没有超过 P4/P5 原版

`0705_04 slowgate` 相对 `0705_03`：

- `det_HOTA +0.029`
- `det_MOTA +0.105`
- `pair_mAP +0.0011`
- `pair_AP50 +0.0035`
- `cls_HOTA -0.070`
- 综合分 `-0.041`

因此 slowgate 更像是 det/AP 稳定器，不是综合 HOTA 最优结构。它证明 conservative gate 有价值，但当前会略压 cls-side 上限。

## 6. 训练曲线观察

末段 HOTA 趋势：

| 实验 | epoch 51 sum | epoch 55 sum | epoch 59 sum | epoch 63 sum | epoch 67 sum | epoch 71 sum | 最佳点 |
|---|---:|---:|---:|---:|---:|---:|---|
| `0704_01 resume` | 103.080 | 103.076 | 103.436 | 103.500 | 103.643 | 103.563 | epoch 67 |
| `0704_02 p5temporal` | 104.455 | 104.498 | 104.895 | - | - | - | epoch 59 |
| `0705_01 p5temporal_pyramidlocal` | 104.925 | 105.424 | - | 105.315 | - | 105.271 | epoch 55 |
| `0705_02 pyramidlocal` | 104.790 | 104.547 | 104.331 | 104.892 | 105.143 | - | epoch 67 |
| `0705_03 pyramidlocal_p4p5` | - | - | 104.875 | 105.048 | 105.123 | 105.343 | epoch 71 |
| `0705_04 pyramidlocal_p4p5_slowgate` | 105.122 | - | - | 105.302 | 105.227 | 105.207 | epoch 63 |

观察：

- `0705_01` 较早在 epoch 55 达峰，后期基本维持在高位，说明双 temporal 分支收敛快。
- `0705_03` 后期持续上升，到 epoch 71 达到最佳，说明 P4/P5 local temporal 的收益更偏后期。
- `0705_04` 在 epoch 63 达峰后略回落，符合 slowgate 更保守、更稳定但上限略低的表现。
- `0704_02` 目前只看到可用结果到 epoch 59；它已经明显超过 baseline，但还不能判断完整 72 epoch 后是否继续提升。

## 7. 最终建议

1. encoder 系列主线推荐 `0705_01 p5temporal_pyramidlocal`，它是当前 `cls_HOTA + det_HOTA` 唯一最佳点。
2. 如果后续希望做更轻量或更稳定的 encoder temporal，优先从 `0705_03 pyramidlocal_p4p5` 继续，而不是全尺度 `0705_02`。
3. `0705_04 slowgate` 可作为 det/AP 方向的 ablation，但不建议替代 `0705_01` 作为综合 HOTA 主线。
4. encoder temporal 的收益明显高于 decoder tri-state 系列：当前 encoder 最优相对 baseline `+1.781`，decoder 最优为 `+0.988`。
5. 后续若与 liquid 组合，应优先选择 `0705_01` 或 `0705_03` 作为 encoder 侧结构；前者追求综合最优，后者结构更简洁且后期趋势更稳。

## 8. 论文全量组合复验

`0716_05`将本报告唯一最佳的`0705_01 p5temporal_pyramidlocal`叠加到论文正式
Base + Liquid group-set-unique配置。相对同步运行的`0716_04`，唯一模型变化是P5 global
temporal与post-FPN三尺度pyramid-local adapter；仍从同一个COCO-only适配权重fresh训练，
使用全量8297个正序pair、1200x900、BF16、`find_unused_parameters=False`和完整72-epoch
评测。

2026-07-16在252 GPU 0/1完成30项单测和100 iter DDP验证后启动正式训练。epoch 1 iter 50
的loss和grad均有限，两个零初始化gate已经打开且其内部模块收到梯度，框架统计显存
`11387 MB/rank`。正式目录为
`/data4/litianhao/PairMmot/workdir_252/0716_05_paper_base_plus_liquid_groupsetunique_encoder_r18_coco_full_1200x900_bf16_orderedpairs_fresh`。

### 8.1 `0723_01` Base+Liquid上的严格复验

`0726_02`将同一`0705_01` Encoder叠加到当前Base+Liquid基准`0723_01`，保持Liquid、
PairDN、proposal、decoder、loss、全量数据和选点规则不变；AutoDL单卡5090使用physical
batch 8保持global batch 8。训练完成72 epochs和18/18 TrackEval，唯一最大
`cls HOTA + det HOTA`位于epoch 72：

| 实验 | epoch | cls HOTA | cls DetA | cls AssA | det HOTA | det DetA | det AssA | pair mAP | pair AP50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0723_01` Base+Liquid | 64 | 53.955 | 44.032 | 68.174 | 62.032 | 53.751 | 73.988 | 0.3114 | 0.5268 |
| `0726_02` + `0705_01` Encoder | 72 | 54.742 | 44.941 | 68.622 | 61.631 | 53.365 | 73.689 | 0.3223 | 0.5429 |
| 变化 |  | +0.787 | +0.909 | +0.448 | -0.401 | -0.386 | -0.299 | +0.0109 | +0.0160 |

因此原始`0705_01`结构显著增强分类和AP，但det HOTA同时因DetA与AssA小幅下降而未超过
Base+Liquid。逐类HOTA相对`0723_01`最佳点：awning-bike `+1.094`、bus `+0.354`、
car `+0.017`、tricycle `+0.883`、truck `+5.327`、van `+0.417`；bike `-0.689`、
pedestrian `-1.113`。主要收益集中在tricycle/truck，pedestrian下降与det覆盖损失说明原始
方向性pyramid-local对小目标和单帧检测主路径仍不够保守。该结果支持继续探索Encoder，但不
支持直接把`0705_01`作为最终消融结构。

权威AutoDL归档位于`/autodl-fs/data/PairMOT_results/0726_02`，本仓库结果副本为
`autodl/results/0726_02/result.md`和`result.json`。

## 9. 0726_03 Pair Common-Detail Pyramid Encoder

在论文协议的`0723_01 Base + Liquid`和`0705_01 Encoder`组合上，保留FPN前P5双向全局
MHA，将FPN后P3/P4/P5的方向性local adapter替换为pair-common/detail结构：

- 每层计算`common=(prev+curr)/2`和`detail=(curr-prev)/2`；
- 只用pair顺序不敏感的`[mean(common), mean(abs(detail))]`生成共享channel gate；
- 对detail使用无bias的depthwise 3x3、`tanh`和grouped 1x1构成严格奇函数变换；
- prev/curr施加等大反向残差，因此逐位置pair均值严格不变，交换两帧只交换输出；
- 不新增loss、proposal/decoder改动或高分辨率attention。

新post-FPN adapter参数由原`280,515`降至`179,907`。13项单测覆盖零门控恒等、pair均值
保持、交换等变、梯度回传和构建；双卡4-iteration真实数据smoke完成，三个local gamma由0
更新到`0.00387/0.00357/0.00214`。2026-07-26 20:35在252 GPU 0/1 fresh启动正式训练，
epoch 1 iter 200约`1.203 s/iter`、每卡MMEngine显存约11.0 GiB，总loss、PairDN、encoder
proposal loss、grad norm及两个encoder分支参数更新均正常。正式目录为
`/data4/litianhao/PairMmot/workdir_252/0726_03_paper_base_liquid_encoder_p5temporal_commondetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_orderedpairs_fresh`。

## 10. 0727 Encoder Goal与Dual-Evidence分支

新一轮Encoder探索严格固定`0723_01 Base + Liquid`，不改变Liquid sampler/fusion、PairDN、
proposal、decoder、loss、数据、初始化或选点规则。成功门槛固定为`0723_01`唯一最佳epoch 64：

- cls HOTA必须高于`53.955`；
- det HOTA必须高于`62.032`；
- 两项必须来自同一个最大`cls_HOTA + det_HOTA`的唯一epoch。

`0726_03`只允许signed detail改变两帧，严格保持pair均值。该约束保护单帧检测主路径，但也
禁止Encoder增强两帧共同存在的目标证据。为同时覆盖det与cls，新增Dual-Evidence结构：

1. `common=(prev+curr)/2`进入轻量depthwise/grouped local分支，生成两帧同向的检测证据；
2. `detail=(curr-prev)/2`进入bias-free odd local分支，生成两帧反向的时序/关联证据；
3. `[mean(common), mean(abs(detail))]`只产生共享channel gate，不引入类别或ROI依赖；
4. common/detail分别使用零初始化gamma，初始网络严格等价父配置；
5. 交换两帧时common不变、detail和其残差同时反号，因此整个adapter严格帧交换等变。

`0727_01`不使用空间门控，用于直接检验common检测增强与detail关联增强能否互补。其
post-FPN adapter为261,318参数；16项单测、完整模型构建、远端哈希/config audit和单卡
batch-8真实4-iter smoke通过。2026-07-27 00:17在178 GPU 0 fresh启动，正式iter 50约
`1.1665 s/iter`；总loss、PairDN、encoder proposal、grad norm以及P5 MHA、common和detail
分支均已更新。正式目录为：

`/data4/litianhao/PairMmot/workdir_178/0727_01_paper_base_liquid_encoder_p5temporal_dualevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh`。

`0727_02`在相同结构上只增加每层2通道3x3空间门控，输入为逐位置common magnitude和
detail magnitude，用于抑制P3背景差异并保留小目标局部证据。新增空间门控总计仅114参数，
无额外loss和高分辨率attention。它已在197排队；只有前序`0726_01`完成epoch 72及18/18
TrackEval、GPU 4/5连续空闲并通过自己的双卡4-iter smoke后，队列才会启动正式训练。

AutoDL不加入该目标的自主调度。已有任务只监控；新任务仅在用户明确提供实例与GPU后启动。

2026-07-27首个同epoch证据：`0726_03`在epoch 8的cls HOTA为`44.414`、det HOTA为
`50.136`；相对`0723_01`同epoch分别提高`+0.102/+0.299`。det DetA提高`0.123`，
det AssA提高`0.657`，pair mAP提高`0.0022`。该点说明pair-mean-preserving detail路径当前
同时改善检测和关联，但只有epoch 4/8两个点，尚不能外推到最终门槛。

基于历史Encoder尺度消融，新增`0727_03 Scale-Split Dual-Evidence`：P3仅保留common共享
检测增强，P4/P5保留common与signed detail两条路径，避免高分辨率P3的局部帧差异干扰小目标
分类/关联。该结构不增加loss或Liquid改动，post-FPN参数为250,822。17项单测覆盖尺度选择后
全部可训练参数均获得梯度，完整构建、远端哈希和config audit通过。它已在178排于`0727_01`
之后；队列首次因前序尚无评测目录触发`pipefail`退出，未创建smoke或正式目录，计数逻辑修复
后于00:29重启并确认持续存活。

`0726_03`的epoch 12进一步给出更明确的分解证据。相对`0723_01`同epoch，其cls HOTA由
`48.087`提高到`48.860`（`+0.773`），det HOTA由`55.635`降到`55.199`（`-0.436`）；
cls DetA提高`1.353`，det DetA提高`0.155`，但det AssA下降`1.348`。同点pair mAP和AP50
分别提高`0.0012/0.0092`。因此common-detail结构并未削弱检测覆盖，主要风险是训练中期
signed-detail残差幅度增长后改变关联排序；该判断也与post adapter的gamma绝对值已达到
约`1.82`一致。

据此新增`0727_04 Detail-Energy Conservation`。它完全保留`0726_03`的Liquid、P5 MHA、
common/detail分解和训练协议，仅对每个样本、层和通道计算原始pair detail与signed update
的空间RMS；当更新RMS超过输入detail RMS时按比例缩小更新，否则保持原值。RMS统计停止梯度，
因此该约束不能通过放大输入统计被规避，不引入参数、阈值或额外loss。两帧仍施加等大反向
残差，所以pair均值守恒和帧交换等变性质不变。18项测试覆盖能量上限、恒等初始化、交换等变、
梯度和完整构建，本地与252核心文件哈希一致。严格队列于2026-07-27 01:20在252启动，只有
`0726_03`完成72 epochs、18/18 TrackEval和独立双卡4-iter smoke后才fresh训练。

`0727_01 Dual-Evidence`的首个完整epoch 4结果为`36.209/38.753`。相对`0723_01`
同epoch的`36.701/45.010`，cls/det HOTA分别下降`0.492/6.257`；det DetA下降`4.865`，
det AssA下降`8.738`，而cls AssA反而提高`0.695`。同checkpoint的P3/P4/P5
common gamma为`0.330/-1.146/-0.229`，detail gamma为
`-0.725/-0.746/-0.210`，说明自由的双残差尤其在P4过快放大，当前大幅det下降不能解释为
普通评测噪声。训练暂保留到epoch 8复核，但含同一全尺度common路径的`0727_03`不能在该证据
下无条件接续。

为此准备`0727_05 Spatial-Reliability Detail-Only`。它回到`0726_03`只修改signed detail、
严格保持pair均值的结构，不再添加shared common residual；每层从channel-mean
`abs(common)`和`abs(detail)`构造局部证据，并分别用停止梯度的空间均值归一化，再由一个
`2→1`的3×3卷积产生空间logit。实际调制为`2*sigmoid(logit)`，因此零初始化时严格为1，
范围限制在`[0,2]`，可以抑制背景/错位差异而不能产生无界更新。三层总计只增加57参数，
无额外loss。20项本地与178测试、完整构建和逐文件哈希通过，且相同父权重与非零gamma下
空间门初始输出逐元素严格等于父结构；当前只准备、不占GPU，作为detail-only保守回退候选，
不替代当前双残差主线。

为避免只凭总分猜测双残差失败原因，已准备epoch 8零训练分支诊断。工具读取MMEngine
checkpoint中用于部署/验证的EMA `state_dict`，分别生成`no_common`、`no_detail`、
`no_p4_common`和`no_post`四个analysis-only精简checkpoint，只归零对应gamma并在meta中记录
原值、修改值与来源；原checkpoint保持只读。若epoch 8仍显著落后，将先停止`0727_01`，再在
同一GPU和测试协议下串行运行四组同步TrackEval。脚本要求epoch 8 checkpoint、epoch 4/8两个
完整评测和训练进程已退出，拒绝覆盖诊断目录；5项本地/178单测、shell语法和逐文件哈希通过。
这些结果只用于确定common、detail或P4-common的责任，不作为模型或论文消融结果。

epoch 8结果改变了早期判断。`0727_01`达到cls/det HOTA `45.269/50.193`，相对
`0723_01`同epoch的`44.312/49.837`双提升`+0.957/+0.356`。分解上，cls DetA提高
`1.842`且AssA仅下降`0.042`；det DetA提高`2.710`，但det AssA下降`2.918`。pair
mAP/AP50也提高`0.0165/0.0213`。相对detail-only的`0726_03`同epoch，Dual-Evidence的
cls/det HOTA为`+0.855/+0.057`，det DetA为`+2.587`、AssA为`-3.575`。因此common分支的
检测增强是可训练且有效的，当前限制来自通道混合的shared residual破坏后续关联排序，而不是
detail信息不足。`0727_01`继续完整训练，预备的epoch 8关支路评测不再执行。

该证据也否决了`0727_03`的设计靶点：它只在P3移除detail，却保留P3/P4/P5同一common
残差，无法解决已观测到的DetA/AssA矛盾。其等待队列于02:40撤销，期间未创建smoke、正式
workdir或训练进程。

替代方案`0727_06 Shared-Scalar Common Evidence`保留P5 MHA与原signed-detail路径，但将
common分支从depthwise/grouped通道混合残差改为逐位置单标量增益。归一化的local common/detail
能量进入`2→1`的3×3卷积，增益为`1+tanh(logit)`、严格落在`(0,2)`；同一增益同时乘prev和
curr，因此不会旋转每个位置的通道向量，并以相同比例缩放原pair差异。零初始化时与父结构逐
元素一致，三层仅增加57参数，无额外loss。22项本地/178单测覆盖初始等价、几何保持、交换
等变和梯度，完整模型构建为22,677,421参数，config、shell和逐文件哈希审计通过。严格队列于
02:43启动。后续epoch 12证据推翻了该设计的前提，队列于03:47在任何smoke或正式训练前撤销；
代码保留为已验证但未运行候选。

`0726_03`在epoch 16重新恢复为干净的双侧提升：cls/det HOTA为`50.302/57.806`，
相对`0723_01`同epoch的`49.904/57.629`分别提高`+0.398/+0.177`。det DetA从
`50.204`提高到`50.426`（`+0.222`），det AssA从`68.464`提高到`68.555`
（`+0.091`），pair mAP从`0.2725`提高到`0.2770`（`+0.0045`）。因此epoch 12的
det回落更像优化过程中的暂时关联扰动，而不是该结构必然以AssA换DetA；当前四个评测点中
epoch 8和16均为双提升。该证据支持让`0726_03`完整训练，也保留其能量受控后继`0727_04`，
但最终结论仍必须按18/18评测和唯一HOTA和最佳epoch确定。

`0727_02`的首个完整epoch 4点进一步验证空间可靠性门的作用。cls/det HOTA为
`37.920/45.436`，相对`0723_01`同epoch的`36.701/45.010`提高`+1.219/+0.426`；
det DetA从`37.319`提高到`37.759`（`+0.440`），det AssA从`56.204`提高到
`56.695`（`+0.491`），pair mAP从`0.1599`提高到`0.1680`（`+0.0081`）。
相对无空间门的`0727_01`同epoch，cls/det HOTA分别恢复`+1.711/+6.683`。这说明
common/detail双残差的早期失稳并非结构不可训练，而主要来自缺少局部证据约束；2通道空间门
能够同时恢复检测覆盖和关联。当前只将其视为阶段证据，至少等待epoch 8/12连续点后再决定
197的下一后继结构。

代码审计同时发现，`0727_02`的空间门由小随机权重接sigmoid构成，初始值约为`0.5`。因此
epoch 4的大幅恢复既包含空间证据选择，也包含common/detail更新整体减半的稳定作用；后期
gamma继续增大时，单靠该门不能保证残差幅度仍受输入证据约束。

据此构建`0727_07 Spatial Branch-Energy Trust Region`。它完整保留`0727_02`结构，只在
乘上各自gamma后计算每个样本、通道的空间RMS：shared-common更新超过输入common RMS时按
比例缩小，signed-detail更新超过输入detail RMS时独立缩小，未超过上限的更新完全不变。
所有RMS参考量停止梯度，模型不能通过放大参考统计规避约束；该设计无新增参数、loss或人工
阈值，并保持帧交换等变。24项本地/197单测覆盖双分支上限、近零证据不抬高上限、交换等变、
完整梯度与历史行为，
正式模型仍为22,758,889参数；config deepcopy、完整构建、shell和6文件哈希均通过。严格队列
于2026-07-27 03:05在197启动，等待`0727_02`完成72 epochs、18/18 TrackEval和独立双卡
4-iter smoke后fresh训练。monitor会额外记录`common/detail_scale`及两分支`clip`比例；
这些均为detached诊断标量，不进入梯度图，可直接判断约束是未触发、适度介入还是过强。

epoch 4 EMA参数进一步排除了“空间门只靠固定缩小残差”的解释。`0727_02`的P3
common/detail gamma已为`1.209/-1.009`，而无空间门`0727_01`同epoch仅为
`0.330/-0.725`；尽管common幅度更大，前者det HOTA仍高`6.683`。同时`0727_02`的P3
空间门18个卷积权重全部为正（范围`0.103~0.164`），两个bias也为正
（`0.153/0.168`）；由于common/detail能量输入非负，该门已学习在高证据位置将响应推到
sigmoid `0.5`以上，而不是维持初始的统一减半。P4也以正权重为主，P5保持更弱且有正负
混合。因此当前有效因素确实包含尺度相关的空间证据选择；与此同时P3 common gamma快速超过
1也证明后期存在分支能量继续增长的真实风险，支持`0727_07`只在超出输入证据RMS时介入。

`0727_01` epoch 12继续扩大双侧优势：cls/det HOTA达到`49.680/56.541`，相对
`0723_01`同epoch的`48.087/55.635`分别提高`+1.593/+0.906`。det DetA从`48.896`
提高到`50.345`（`+1.449`），det AssA从`65.573`微升到`65.579`（`+0.006`），
pair mAP从`0.2594`提高到`0.2732`（`+0.0138`）。相对detail-only的`0726_03`同epoch，
cls/det HOTA也提高`+0.820/+1.342`，det DetA/AssA同时提高`+1.294/+1.354`。因此
epoch 8的AssA下降是可恢复的优化阶段，不再支持“channel-mixing common必然破坏关联”的
判断；Dual-Evidence目前是三条Encoder结构中能力最强且机制最完整的主线。

基于该结论，用`0727_08 Dual-Evidence Branch Trust`替代`0727_06`。它逐项保留
`0727_01`的P5 MHA、common/detail卷积分支、Liquid及完整训练协议，不使用`0727_02`空间门；
唯一改动是打开已由`0727_07`实现和验证的双分支RMS trust region。这样178直接检验能量约束
本身，197则检验空间门与能量约束的组合。该候选不增加参数、loss或阈值，模型参数仍为
22,758,775；24项本地/178测试、完整构建、config deepcopy、shell及7文件哈希通过。严格
队列于2026-07-27 03:50启动，首个心跳正确识别`0727_01`仍运行、3/18 TrackEval和GPU占用，
后续只在72 epochs、18/18 TrackEval、连续空闲及独立4-iter smoke全部通过后fresh启动。

`0727_02` epoch 8为cls/det HOTA `44.277/50.302`。相对`0723_01`同epoch，
cls HOTA微降`0.035`，det HOTA提高`0.465`；det DetA下降`0.296`，但det AssA提高
`1.669`，pair mAP提高`0.0022`。相对无空间门的`0727_01`同epoch，cls HOTA低
`0.992`、det HOTA高`0.109`，det DetA低`3.006`而AssA高`4.587`。这确认空间门已形成
明显的“降低检测覆盖、增强关联稳定性”偏置，而不是无条件提升两者。由于cls差距仅`0.035`
且epoch 4曾双提升，当前不撤销`0727_07`，但必须等待epoch 12确认该偏置是否持续。

`0726_03` epoch 20达到`50.889/58.348`，相对`0723_01`同epoch的
`50.863/58.161`严格提高`+0.026/+0.187`。det DetA提高`0.347`，AssA微降`0.062`，
pair mAP提高`0.0076`。虽然cls裕量很小，但它在epoch 8、16、20三个评测点均实现双提升，
仅epoch 12出现det暂时回落；因此pair-mean-preserving common-detail仍是当前稳定性最好的
保守Encoder路径，继续完整训练及其`0727_04` detail-energy后继。

`0727_01` epoch 12的提升也不是少数序列或单一类别拉动。TrackEval原始8类中7类HOTA
提高：tricycle `+4.920`、bus `+3.865`、bike `+2.074`、van `+1.171`、
pedestrian `+1.053`、awning-bike `+0.905`、car `+0.309`；仅truck下降`1.552`，
且pedestrian占大量GT仍保持明确正增益。263个含GT的seq-class项中152项提高、90项下降、
21项不变，
提高项覆盖约60.0%的GT。进一步按每个序列的GT数对其类别差值做诊断性加权（不是官方
TrackEval聚合）时，50个序列中34个为正、16个为负。主要回退序列为`data47-4`、
`data31-1`、`data39-2`和`data47-3`；这表明总增益具有较广覆盖，但后续完整结果仍需检查
truck及这些序列的AssA回退是否持续。

`0727_01` epoch 16延续为第三个连续双提升点：cls/det HOTA为`51.091/58.320`，
相对`0723_01`同epoch的`49.904/57.629`提高`+1.187/+0.691`。det DetA从
`50.204`提高到`51.714`（`+1.510`），det AssA从`68.464`降到`67.970`
（`-0.494`），pair mAP从`0.2725`提高到`0.2839`（`+0.0114`）。相对同期
detail-only `0726_03`，cls/det HOTA仍高`+0.789/+0.514`。因此Dual-Evidence的主要优势
已从epoch 12的DetA/AssA同时改善转为更强检测覆盖承担小幅关联代价，但总HOTA和cls仍保持
明确优势。epoch 8/12/16连续三点严格双提升，使其成为当前达到最终双超门槛概率最高的
Encoder结构；`0727_08`继续用于测试能量约束能否保留DetA增益并降低后期AssA代价。

`0727_02` epoch 12否决了“同时空间门控common/detail”的结构。其cls/det HOTA为
`47.411/54.593`，相对同epoch Base+Liquid分别下降`0.676/1.042`；det DetA与AssA分别
下降`1.052/0.829`，pair mAP下降`0.0108`。这不是仅有分类或关联侧的随机波动，而是
检测覆盖和关联共同退化。结合epoch 8尚有det `+0.465`但DetA已低`0.296`，最一致的解释是
common分支的空间门逐步压低共享目标证据；给两条分支增加RMS上限的`0727_07`没有针对根因，
因此在smoke和正式训练前取消。

据此在197启动`0727_09 Detail-Only Spatial Reliability`。它完整保留`0727_01`的P5 MHA、
channel-mixing common共享残差和signed-detail分支，只用局部common/detail绝对幅值预测
一个单通道空间因子，并且只乘到detail残差。三个3x3卷积均零初始化，门值使用
`2*sigmoid(logit)`，所以初始值严格为1，模型起点与`0727_01`逐元素相同；仅增加57参数，
不增加loss、阈值或branch-energy cap。25项本地/197测试、完整模型构建、配置深拷贝和
6文件哈希通过。真实数据4-iter双卡DDP smoke中全部total/DN/encoder loss与grad norm有限，
三个空间门都从零发生非零更新；正式训练于05:31 fresh启动，iter 50为约`1.016 s/iter`，
P5、common、detail和spatial参数均已更新。该实验直接检验局部可靠性能否只改善detail关联，
同时保留`0727_01`已验证的DetA能力。

`0726_03` epoch 24进一步增强了保守common-detail路径的可信度。该点cls/det HOTA为
`51.623/58.988`，相对同epoch Base+Liquid的`50.921/58.657`提高
`+0.702/+0.331`。不同于epoch 20仅有很小的det AssA回退，本次det DetA/AssA分别提高
`+0.442/+0.175`，cls DetA提高`1.115`而cls AssA仅低`0.210`；pair mAP/AP50分别提高
`0.0078/0.0173`。这说明pair-mean preservation没有限制后期检测能力，signed-detail的
关联代价也能随训练自行恢复。`0727_04`暂时保留到epoch 28复核，但其角色从“修复已确认的
关联退化”调整为“检验是否有必要限制偶发过强detail更新”；若后续AssA继续非负，则不应
为了早期波动牺牲当前已验证的自由优化能力。

`0727_01` epoch 20构成连续第四个严格双提升点。其cls/det HOTA为`51.514/58.922`，
相对同epoch Base+Liquid的`50.863/58.161`提高`+0.651/+0.761`。cls DetA/AssA同时提高
`+0.884/+0.358`；det DetA提高`1.626`，det AssA仍低`0.514`，pair mAP/AP50提高
`0.0136/0.0157`。这说明Dual-Evidence的主要增益仍来自更强检测覆盖，但cls关联已经恢复，
det关联代价也小于DetA收益。263个含GT的seq-class项中155项提高、94项下降、14项不变，
提升项覆盖约63.2%的GT；50个序列中35个诊断加权为正。类别HOTA中tricycle
`+4.717`、bus `+1.379`、bike `+1.143`、pedestrian `+0.984`和car `+0.723`，
而van `-1.654`、truck `-1.550`、awning-bike `-0.537`。因此`0727_08`仍有明确靶点：
在不削弱common分支DetA能力的前提下限制过强残差，尝试回收det AssA；但其结果必须与
`0727_01`自由容量路径直接比较，不能把能量上限默认视为改进。

`0727_09`首个epoch 4评测为cls/det HOTA `37.865/45.325`，相对同epoch Base+Liquid
`36.701/45.010`严格提高`+1.164/+0.315`。cls DetA/AssA分别提高`+0.790/+1.826`，
det DetA/AssA分别提高`+0.004/+0.597`，pair mAP/AP50提高`0.0025/0.0064`。相比同一父结构
`0727_01` epoch 4的`36.209/38.753`，两侧HOTA提高`+1.656/+6.572`，det DetA/AssA恢复
`+4.869/+9.335`。其结果与同时门控common/detail的`0727_02` epoch 4
`37.920/45.436`接近，但`0727_09`不压制common分支、初始函数严格等于父配置，因此避开了
`0727_02`中后期检测覆盖持续下降的机制。EMA中三个detail空间门bias已分别学习到
`0.103/0.121/0.009`，说明改善不是固定0.5缩放造成。相对Base有6/8类提高，但pedestrian
与truck分别低`1.212/1.126`；该点只证明结构没有早期失败，仍需epoch 8确认小目标、truck
及双侧HOTA持续性。

`0727_01` epoch 24进一步达到`51.714/59.519`，相对同epoch Base+Liquid提高
`+0.793/+0.862`。cls DetA/AssA分别提高`+0.986/+0.571`，det DetA提高`1.772`，
det AssA下降`0.489`，pair mAP/AP50提高`0.0099/0.0098`。这是epoch 8以来连续第五个
严格双提升点，且det HOTA裕量继续扩大；因此自由容量Dual-Evidence仍是主线。det AssA
约0.5的稳定代价与EMA分支gamma绝对值达到`2.479`共同支持保留`0727_08` branch trust，
但该约束必须证明不会消耗当前`+1.772`的DetA优势。

`0726_03` epoch 28为`52.081/59.387`，相对同epoch Base+Liquid仍是
`+0.372/+0.002`，但det优势已经接近评测精度边界。cls/det DetA分别提高
`+1.283/+0.241`，AssA分别下降`1.346/0.288`，pair mAP/AP50提高`0.0067/0.0153`。
因此epoch 24的关联恢复不是稳定趋势；该结构仍具有检测增益，但signed-detail更新会在部分
阶段换取关联。`0727_04` detail-energy cap继续保留，用于检验限制过强detail残差能否将
DetA增益转换为稳定的双侧HOTA裕量。

`0727_09` epoch 8没有延续早期双提升。该点cls/det HOTA为`44.077/50.210`，相对同epoch
Base+Liquid为`-0.235/+0.373`；det DetA/AssA分别提高`+0.385/+0.467`，但cls DetA/AssA
分别下降`0.056/0.263`，pair mAP下降`0.0023`。相对同一父配置`0727_01`，空间detail门将
det AssA回收`+3.385`，却损失det DetA `2.325`、pair mAP `0.0188`和cls HOTA `1.192`。
类别上相对Base提升awning-bike、bike、car、pedestrian和van，但tricycle/truck分别下降
`3.987/2.506`。这说明detail路径不只承担关联，也参与检测定位，不能以强空间抑制将其当成
纯关联分支。

epoch 12进一步表明无约束空间门不能稳定改善父结构。该点cls/det HOTA为
`49.210/55.433`，相对同epoch Base+Liquid为`+1.123/-0.202`；cls DetA/AssA分别提高
`1.196/1.667`，det DetA提高`0.356`但AssA下降`1.057`，pair mAP/AP50提高
`0.0041/0.0139`。相对`0727_01`同epoch，cls/det HOTA分别下降`0.470/1.108`，
det DetA/AssA均下降约`1.1`。逐类HOTA相对Base除car下降`1.278`外，awning-bike、bike、
bus、pedestrian、tricycle、truck和van均提高，其中truck提高`3.487`。因此该分支确实增强
局部类别/检测证据，但det侧总下降明确来自关联质量，而不是覆盖不足；它不能作为最终结构，
保留完整训练仅用于验证后期轨迹，结构修正由`0727_10`承担。

checkpoint进一步显示，`0727_09`三个空间卷积的权重和bias均整体偏正；同时P4 common gamma
由父配置EMA的`-1.602`变为`+1.293`。现有`2*sigmoid`门既能全局放大detail更新，其
common/detail描述也未停止梯度，因而会通过空间门旁路改变共享特征与common分支的优化轨迹。
据此构建`0727_10 Detail Spatial Redistribution`：空间门仍从局部common/detail幅值学习，
但描述停止梯度；每个样本、层的正空间调制再除以自身空间均值，使均值严格为1。因此门只能
重分配detail更新，不能改变全局平均尺度或向共享特征注入描述梯度。该改动不新增参数、loss、
阈值或高分辨率attention，零初始化时仍与`0727_01/09`父函数逐元素一致。26项本地和197
单测、配置深拷贝、完整`22,758,832`参数模型构建、shell及6文件哈希均通过；严格队列于
2026-07-27 08:10启动，等待`0727_09`完成72 epochs和18/18 TrackEval后运行独立真实数据
双卡4-iter smoke，再fresh训练。

`0727_01` epoch 28仍保持第六个连续双提升点，但优势明显收窄。cls/det HOTA为
`51.740/59.830`，相对同epoch Base+Liquid仅为`+0.031/+0.445`；cls DetA/AssA分别
为`+0.273/-0.129`，det DetA/AssA为`+1.523/-1.147`，pair mAP/AP50提高
`0.0064/0.0035`。8类中awning-bike、bike、bus、car、pedestrian和tricycle仍提升，
truck/van分别下降`3.190/0.905`。因此Dual-Evidence的检测容量仍有效，但随着EMA分支gamma
最大绝对值接近`2.87`，关联代价正在扩大并消耗cls优势；`0727_08`对两条分支分别施加
输入证据RMS上限仍是直接针对该后期趋势的必要消融。

epoch 32恢复为更明确的第七个连续双提升点：cls/det HOTA为`52.354/60.330`，相对同epoch
Base+Liquid为`+0.248/+0.553`。cls DetA/AssA变化为`+0.709/-0.581`，det DetA/AssA为
`+1.524/-0.906`，pair mAP/AP50提高`0.0078/0.0048`；pedestrian、tricycle分别提高
`0.836/2.705`，truck、van下降`2.810/1.220`。这说明自由Dual-Evidence仍具有稳定增益，
但其机制仍是以检测覆盖换取部分关联质量。EMA分支gamma最大绝对值从epoch 28约`2.87`
增至`2.926`，后继`0727_08`应被解释为后半程稳定性约束，而不是对失败父结构的修复。

`0726_03` epoch 32重新扩大了epoch 28几乎为零的det裕量，但仍是小幅双提升：
cls/det HOTA为`52.265/59.855`，相对同epoch Base+Liquid为`+0.159/+0.078`。
cls DetA提高`1.189`而AssA下降`1.937`；det DetA提高`0.340`、AssA下降`0.195`；
pair mAP/AP50提高`0.0086/0.0182`。类别上bike、car、truck和van提升，bus下降`1.889`，
pedestrian/tricycle接近持平。该结构在8个评测点中除epoch 12外均保持双提升或接近双提升，
说明pair均值守恒路径稳定但上限偏保守；其主要问题不是检测覆盖，而是signed-detail对分类
关联的阶段性扰动。`0727_04`继续检验输入detail RMS上限能否回收AssA，但必须同时证明不会
损失当前DetA和AP增益。

epoch 36给出更强且机制更健康的中期结果：cls/det HOTA为`52.361/60.261`，相对同epoch
Base+Liquid双提升`+0.688/+0.416`。cls DetA提高`1.070`、AssA下降`0.547`；det DetA/AssA
则同时提高`0.272/0.744`，pair mAP/AP50提高`0.0112/0.0217`。类别上awning-bike、car、
tricycle、truck和van提高，pedestrian基本持平，bike/bus下降`0.660/1.369`。这证明
common/detail pair-mean-preserving结构的det关联代价不是固定机制，epoch 32的AssA下降更像
训练波动；`0727_04` detail-energy cap继续保留队列，但必须在父实验完成后重新审核，不能因
早期诊断自动启动。

### 9.9 MCDE：稀疏矩竞争式Dual-Evidence（0727_11）

截至epoch 40，`0727_01 Dual-Evidence`相对同epoch Base+Liquid达到
`cls HOTA +1.304 / det HOTA +0.415`，其中cls DetA/AssA均提高，det DetA提高
`0.911`而det AssA仅下降`0.292`；它是当前最有希望最终双超基准的Encoder结构。
相反，`0727_09`在detail分支加入高分辨率空间门后，epoch 16相对基准双降
`0.075/0.399`，det DetA/AssA也同时下降；`0726_03`的严格pair均值守恒结构在epoch 40
仅为`+0.093/-0.233`。下一步不应继续压缩残差容量或增加空间抑制，而应改善Dual-Evidence
现有全局channel gate对稀疏小目标证据的感知，并约束两条分支的相对选择。

`0727_11 Moment-Competitive Dual Evidence (MCDE)`保留`0727_01`全部卷积残差路径：

1. 对common和detail分别计算逐通道均值描述；detail仍使用`mean(abs(detail))`。
2. 额外计算稀疏性矩`gap = RMS(x) - mean(abs(x))`。均匀响应的gap较小，少量强响应的
   gap较大，因此小目标或稀疏纹理不再被纯全局均值完全稀释。
3. 每层仅以两个可学习逐通道系数混合原描述和gap；gap对高分辨率输入停止梯度，避免新增
   统计旁路改变backbone/neck的优化。
4. gate MLP输出common/detail两组logit，并在分支维执行两路softmax。每个通道的两路权重
   和严格为1，使共享检测证据与帧差关联证据形成可学习竞争；它不同于能量hard cap，不会
   截断残差幅值，也不规定哪条分支必须占优。
5. 输出仍为`prev + shared_update - signed_update`和
   `curr + shared_update + signed_update`，保持帧交换等变；`gamma`零初始化保证训练起点
   与父配置严格逐元素一致。

三尺度只新增`2 * (256 + 256 + 256) = 1,536`个参数。完整模型参数由父配置
`22,758,775`增至`22,760,311`，增幅`0.00675%`；gate MLP和卷积主路径规模不变，新增计算
仅为几个逐通道全局归约及两路softmax，预期整体计算增幅低于1%。本地、197和AutoDL均完成
29项功能/梯度/等变测试、配置深拷贝及完整模型构建。原197严格队列在未运行smoke或正式训练
前取消，同一实验迁移到单卡RTX 5090 AutoDL；physical batch 8保持global batch、LR、EMA
和epoch语义。正式尺寸4-iter smoke全部loss和grad有限，MMEngine峰值显存约21.8 GB；正式
训练于2026-07-27 12:57 fresh启动，iter 50为`0.819 s/iter`，总/DN/encoder loss及
grad norm有限，MCDE参数已发生更新。finalizer已挂接，完成后要求18/18 TrackEval并按唯一
最大`cls_HOTA+det_HOTA`选epoch后归档和自动关机。

### 9.10 CSEB：跨尺度证据预算（0727_12）

`0727_01 Dual-Evidence`在17/18个阶段评测点中的唯一最佳为epoch 64：
cls/det HOTA达到`54.673/62.140`，相对Base+Liquid最佳点分别提高`0.718/0.108`。
该点已经满足当前Encoder双提升目标，但学习到的P3/P4/P5 common/detail缩放方向差异明显：
EMA checkpoint中的两路gamma约为
`[[3.604,-2.353],[-2.745,-1.774],[-2.008,-2.888]]`。这说明各尺度独立门控可以取得
增益，但缺少显式的跨尺度协调；继续叠加空间门会损伤检测覆盖，而把每层残差硬裁剪又会
削弱已验证的表达能力。

`0727_12 Cross-Scale Evidence Budget (CSEB)`严格保留`0727_01`的P5 temporal MHA、
common/detail卷积残差和输出形式，只在三尺度门控之间加入轻量预算协调：

1. 每个尺度复用现有全局描述`[mean(common), mean(abs(detail))]`，经共享
   `LayerNorm + Linear`映射成32维token，并加入P3/P4/P5尺度嵌入。
2. 每个尺度token与三个尺度token的均值拼接，通过共享小型MLP预测该尺度的common/detail
   逐通道预算；没有self-attention，也不在高分辨率特征图上增加卷积。
3. 对每个分支和通道，预算在P3/P4/P5维执行softmax并乘3，因此三尺度权重和严格为3。
   该模块只把残差容量从证据较弱尺度重分配给证据较强尺度，不改变三尺度平均门控强度。
4. 预算描述对输入特征停止梯度，避免新增统计旁路绕过原Encoder监督；输出线性层零初始化，
   初始预算逐元素为1，因此初始网络函数与`0727_01`完全一致。
5. common使用pair均值、detail使用绝对pair差，最终仍为对称共享更新和反对称细节更新，
   保持pair帧交换等变。模型不增加loss、阈值、类别规则或温度调度。

该协调器增加37,696参数，完整模型由`22,758,775`增至`22,796,471`，增幅`0.166%`。
32项功能、梯度和帧交换等变测试全部通过，配置深拷贝及完整模型构建通过；真实全尺寸2卡
4-iter DDP smoke的总loss、PairDN loss、encoder proposal loss和梯度均有限。99 GPU 0、1
上的正式fresh训练于2026-07-27 20:18启动，epoch 1 iter 50为`0.9595 s/iter`，
MMEngine峰值显存约11.25 GB/rank；参数已更新，未发现OOM、NaN、NCCL、DDP reduction或
unused-parameter错误。

### 9.11 2026-07-27晚间结果汇总

当前论文阶段固定比较基准为Base+Liquid `0723_01`：唯一最佳epoch 64的cls/det HOTA为
`53.955/62.032`。以下完整实验严格按`cls_HOTA + det_HOTA`选唯一最佳epoch，AP取同一epoch；
运行中实验只报告当前已完成评测点的阶段最佳，不作为最终结论。

| 实验 | 状态 | 评测 | 阶段/最终最佳epoch | cls HOTA | det HOTA | HOTA和 | pair mAP | pair AP50 | 相对Base+Liquid cls/det |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `0723_01` Base+Liquid | COMPLETED | 18/18 | 64 | 53.955 | 62.032 | 115.987 | 0.3114 | 0.5268 | `0/0` |
| `0726_02` 原`0705_01` Encoder | COMPLETED | 18/18 | 72 | 54.742 | 61.631 | 116.373 | 0.3223 | 0.5429 | `+0.787/-0.401` |
| `0726_03` common-detail | COMPLETED | 18/18 | 72 | **54.654** | **62.240** | **116.894** | **0.3239** | **0.5380** | **`+0.699/+0.208`** |
| `0727_01` Dual-Evidence | COMPLETED | 18/18 | 72 | **54.437** | **62.393** | **116.830** | **0.3201** | **0.5371** | **`+0.482/+0.361`** |
| `0727_09` detail spatial reliability | COMPLETED | 18/18 | 72 | 54.106 | 62.321 | 116.427 | 0.3122 | 0.5273 | `+0.151/+0.289` |
| `0727_11` MCDE | COMPLETED | 18/18 | 68 | **54.853** | **62.050** | **116.903** | **0.3190** | **0.5384** | **`+0.898/+0.018`** |
| `0727_12` CSEB | RUNNING | 12/18 | 48 | 54.294 | 61.647 | 115.941 | 0.3131 | 0.5389 | `+0.339/-0.385` |
| `0727_04` detail-energy cap | RUNNING | 6/18 | 24 | 50.222 | 58.405 | 108.627 | 0.2809 | 0.4978 | `-3.733/-3.627` |

`0726_03`和`0727_01`均已完成18/18评测，并相对Base+Liquid同时提高两个HOTA。
`0726_03`的HOTA和为`116.894`，比`0727_01`高`0.064`；`0727_01`的det HOTA则高
`0.153`。因此`0726_03`是按预注册HOTA和规则的当前Encoder最佳，`0727_01`是det侧更强的
互补候选。

`0727_01`相对
Paper Base，它的cls/det HOTA分别提高`1.123/0.411`；det DetA和AssA也同时提高
`0.526/0.311`，说明最终收益不是用关联质量换检测覆盖。相对直接父配置Base+Liquid，
HOTA和提高`0.843`。

逐类别HOTA相对Base+Liquid最佳点有6/8类提高：awning-bike `+3.862`、bike `+0.290`、
bus `+0.908`、car `+0.253`、pedestrian `+0.470`、truck `+2.937`；tricycle
`-1.992`、van `-2.873`。因此总收益不依赖单一大类，pedestrian也没有下降；剩余风险集中在
tricycle和van，后续结构应避免为追求这两类而破坏已获得的det双提升。

`0726_03`最终epoch 72相对Base+Liquid的cls/det HOTA为`+0.699/+0.208`，det
DetA/AssA也同时提高`+0.039/+0.447`，说明common-detail的pair均值守恒设计在后期同时保住
检测覆盖和关联质量。`0726_02`仍只有cls及AP提升，det HOTA低`0.401`。`0727_09`进一步加入高分辨率detail
空间门后两个HOTA均下降，当前证据足以终止空间门方向。

MCDE最终最佳epoch 68相对Base+Liquid为`+0.898/+0.018`，HOTA和`116.903`，名义上比
`0726_03`高`0.009`、比`0727_01`高`0.073`；但相对`0727_01`为
`+0.416/-0.343`，表明稀疏矩竞争主要换取cls增益，并没有保留Dual-Evidence的det优势。
因此它是有效Encoder候选，但不能称为明显优于`0727_01`。`0727_11`是AutoDL最后一个
Encoder实验；整个Encoder阶段只等待已运行的`0727_12 CSEB`和`0727_04 detail-energy`
完成，之后不再追加新Encoder结构，按统一规则选定最终版本并转入Decoder主线。当前
`0727_12`阶段最佳epoch 48为`54.294/61.647`，`0727_04`阶段最佳epoch 24为
`50.222/58.405`，二者目前都没有超过MCDE或`0727_01`，但仍等待完整18/18结果后定稿。
