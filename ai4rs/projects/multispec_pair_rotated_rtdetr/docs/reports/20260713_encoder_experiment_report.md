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
