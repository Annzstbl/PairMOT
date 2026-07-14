# 20260713 Long-tail Class Repair Report

## 1. 背景

此前对比 MOTRv2 与 PairMOT 0704 resume 发现：PairMOT 的 class-agnostic detection/association 更强，但 class-aware tracking 指标偏低。

| 方法 | cls HOTA | det HOTA | 现象 |
|---|---:|---:|---|
| MOTRv2 | 49.286 | 54.521 | cls 更强，det 较弱 |
| PairMOT 0704 resume | 45.523 | 58.120 | det 更强，cls 较弱 |

这说明 PairMOT 并非整体检测失效，而是类别判别、类别校准或细粒度类别混淆限制了 `cls_HOTA`。长尾与细粒度类别中，`bike/truck/bus/tricycle/awning-bike/van` 更容易受影响。

## 2. 实验设置

统计口径：

- baseline 固定为 `0704_01 ... resume_from_epoch40_to72`。
- 所有实验均读取 TrackEval 生成的 `*_summary.csv`。
- 最佳 epoch 只按 `cls_HOTA + det_HOTA` 选取唯一结果。
- AP 只作为检测侧旁证，不参与最佳 epoch 选择。
- per-class 分析使用该实验最佳 epoch 下的 class-aware TrackEval 结果。

实验路径：

| 实验 | 路径 |
|---|---|
| baseline | `/data4/litianhao/PairMmot/workdir_252/0704_01_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_resume_from_epoch40_to72` |
| 0713_01 longtail_reweight | `/data4/litianhao/PairMmot/workdir_252/0713_01_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_longtail_reweight` |
| 0713_02 finecls_margin | `/data4/litianhao/PairMmot/workdir_252/0713_02_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_finecls_margin` |

## 3. 模型改动

### 0713_01 longtail_reweight

Motivation：PairMOT 在 det HOTA 上已经强于 MOTRv2，因此不应通过阈值或 tracker 参数硬改模型表现；优先从训练阶段增强长尾类别正样本的分类梯度。

实现：

- 在 bbox head 中新增 `cls_pos_loss_weights`。
- 只提升真实类别正样本位置的分类 loss 权重。
- 不改 box loss、proposal、decoder、pair association、推理阈值和 tracker。

类别权重：

| class | car | bike | pedestrian | van | truck | bus | tricycle | awning-bike |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| weight | 1.00 | 1.30 | 1.00 | 1.25 | 1.80 | 1.60 | 1.70 | 1.25 |

### 0713_02 finecls_margin

Motivation：如果 cls HOTA 低主要来自细粒度类别混淆，则可以在训练阶段要求长尾/易混类别的 true-class logit 形成更大间隔。

实现：

- 保留较温和的 `cls_pos_loss_weights`。
- 新增 `cls_pos_logit_margins`，训练时对正样本真实类别 logit 减去 margin 后再计算分类 loss。
- 不改推理阈值，per-class 阈值仍只作为分析手段，不作为模型方案。

类别 margin：

| class | car | bike | pedestrian | van | truck | bus | tricycle | awning-bike |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| margin | 0.00 | 0.08 | 0.00 | 0.06 | 0.15 | 0.12 | 0.14 | 0.06 |

## 4. 总体结果

| 实验 | best epoch | cls HOTA | Δcls HOTA | cls MOTA | cls IDF1 | det HOTA | Δdet HOTA | det MOTA | det IDF1 | cls+det HOTA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0704_01 resume baseline | 68 | 45.523 | 0.000 | 34.750 | 52.845 | 58.120 | 0.000 | 51.956 | 66.997 | 103.643 |
| 0713_01 longtail_reweight | 72 | 46.822 | +1.299 | 36.561 | 54.712 | 58.347 | +0.227 | 52.964 | 67.514 | 105.169 |
| 0713_02 finecls_margin | 64 | 46.641 | +1.118 | 34.479 | 54.688 | 58.052 | -0.068 | 51.575 | 66.898 | 104.693 |

结论：

- `0713_01 longtail_reweight` 是当前最好的 long-tail 修复方案，`cls_HOTA + det_HOTA` 相比 baseline 提升 `+1.526`。
- `0713_01` 同时提升 `cls_HOTA/cls_MOTA/cls_IDF1`，且 `det_HOTA` 没有下降，说明它不是用检测侧代价换 cls 侧收益。
- `0713_02 finecls_margin` 也能提升 `cls_HOTA` 和 `cls_IDF1`，但 `cls_MOTA`、`det_HOTA` 略低于 baseline，整体不如 reweight 稳定。
- 与 MOTRv2 的 cls HOTA 差距从 `-3.763` 缩小到 `-2.464`，说明 PairMOT 的 cls 短板可以通过训练侧类别修复缓解。

## 5. Epoch 走势

| epoch | baseline cls | baseline det | 0713_01 cls | 0713_01 det | 0713_02 cls | 0713_02 det |
|---:|---:|---:|---:|---:|---:|---:|
| 44 | 44.998 | 57.596 | 46.106 | 57.593 | 46.025 | 57.733 |
| 48 | 45.292 | 57.583 | 46.310 | 57.773 | 46.182 | 57.800 |
| 52 | 45.348 | 57.732 | 46.646 | 57.953 | 46.214 | 57.867 |
| 56 | 45.353 | 57.723 | 46.572 | 58.160 | 46.367 | 57.929 |
| 60 | 45.528 | 57.908 | 46.592 | 58.040 | 46.480 | 57.939 |
| 64 | 45.499 | 58.001 | 46.647 | 58.257 | 46.641 | 58.052 |
| 68 | 45.523 | 58.120 | 46.711 | 58.259 | 46.560 | 58.041 |
| 72 | 45.526 | 58.037 | 46.822 | 58.347 | 46.465 | 58.086 |

`0713_01` 的 cls/det HOTA 在后半程仍持续上升，最佳点出现在 epoch 72。`0713_02` 的最佳点出现在 epoch 64，之后 cls HOTA 回落，说明 margin 约束可能更容易带来后期过约束或置信度校准问题。

## 6. Per-class 分析

### 0713_01 longtail_reweight vs baseline

| class | baseline HOTA | 0713_01 HOTA | ΔHOTA | baseline IDF1 | 0713_01 IDF1 | ΔIDF1 | baseline MOTA | 0713_01 MOTA | ΔMOTA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| car | 79.250 | 79.180 | -0.070 | 90.221 | 90.219 | -0.002 | 86.194 | 86.397 | +0.203 |
| bike | 36.575 | 36.844 | +0.269 | 42.375 | 43.215 | +0.840 | 18.050 | 19.209 | +1.159 |
| pedestrian | 39.055 | 39.553 | +0.498 | 50.215 | 50.956 | +0.741 | 31.495 | 32.788 | +1.293 |
| van | 57.002 | 57.111 | +0.109 | 65.803 | 65.421 | -0.382 | 41.192 | 41.849 | +0.657 |
| truck | 27.474 | 27.875 | +0.401 | 26.624 | 26.544 | -0.080 | 4.878 | 5.691 | +0.813 |
| bus | 56.876 | 59.490 | +2.614 | 71.675 | 74.506 | +2.831 | 57.564 | 56.066 | -1.498 |
| tricycle | 24.601 | 30.735 | +6.134 | 22.736 | 31.267 | +8.531 | 4.904 | 11.467 | +6.563 |
| awning-bike | 43.350 | 43.788 | +0.438 | 53.112 | 55.568 | +2.456 | 33.724 | 39.024 | +5.300 |

分析：

- `tricycle` 是最大收益类别，HOTA `+6.134`、IDF1 `+8.531`、MOTA `+6.563`，说明 reweight 明显改善了长尾类别的有效召回和轨迹一致性。
- `bus` HOTA `+2.614`、IDF1 `+2.831`，但 MOTA `-1.498`，说明关联质量和轨迹身份更好，但 FP/FN 组合仍有波动。
- `bike/pedestrian/van/truck/awning-bike` 均有 HOTA 正收益，car 基本不受影响。
- 这组结果支持“cls 短板主要来自长尾/细粒度类别训练不足或校准不足”的判断。

### 0713_02 finecls_margin vs baseline

| class | baseline HOTA | 0713_02 HOTA | ΔHOTA | baseline IDF1 | 0713_02 IDF1 | ΔIDF1 | baseline MOTA | 0713_02 MOTA | ΔMOTA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| car | 79.250 | 79.075 | -0.175 | 90.221 | 89.952 | -0.269 | 86.194 | 86.000 | -0.194 |
| bike | 36.575 | 37.393 | +0.818 | 42.375 | 43.679 | +1.304 | 18.050 | 19.683 | +1.633 |
| pedestrian | 39.055 | 38.889 | -0.166 | 50.215 | 49.798 | -0.417 | 31.495 | 30.629 | -0.866 |
| van | 57.002 | 57.851 | +0.849 | 65.803 | 66.652 | +0.849 | 41.192 | 40.834 | -0.358 |
| truck | 27.474 | 25.273 | -2.201 | 26.624 | 25.080 | -1.544 | 4.878 | 1.781 | -3.097 |
| bus | 56.876 | 60.133 | +3.257 | 71.675 | 76.819 | +5.144 | 57.564 | 59.211 | +1.647 |
| tricycle | 24.601 | 30.149 | +5.548 | 22.736 | 28.953 | +6.217 | 4.904 | 0.981 | -3.923 |
| awning-bike | 43.350 | 44.362 | +1.012 | 53.112 | 56.570 | +3.458 | 33.724 | 36.710 | +2.986 |

分析：

- margin 对 `bus/tricycle/awning-bike/bike/van` 的 HOTA 或 IDF1 有明显帮助，说明“类间边界不足”确实是问题之一。
- 但 `truck` 大幅下降，`tricycle` 的 MOTA 也下降，说明 margin 对极低样本或极难类别可能过强，导致召回或置信度排序受损。
- 因此 margin 不能直接作为当前主方案，需要更温和的 curriculum、class-adaptive margin 或只对混淆对施加 margin。

## 7. AP 旁证

训练日志中的检测 AP 最佳值如下，仅用于确认检测侧没有明显退化：

| 实验 | best AP epoch | pair mAP | pair AP50 | both mAP | both AP50 |
|---|---:|---:|---:|---:|---:|
| 0704_01 resume baseline | 68 | 0.2383 | 0.4157 | 0.2448 | 0.4275 |
| 0713_01 longtail_reweight | 72 | 0.2436 | 0.4298 | 0.2502 | 0.4420 |
| 0713_02 finecls_margin | 72 | 0.2433 | 0.4278 | 0.2500 | 0.4399 |

AP 与 HOTA 结论一致：0713 两个实验并没有牺牲检测 AP，其中 `0713_01` 最稳定。

## 8. 结论

1. long-tail class repair 是有效方向。`0713_01 longtail_reweight` 在不改 tracker、不改推理阈值的情况下，把 cls HOTA 从 `45.523` 提升到 `46.822`，同时 det HOTA 从 `58.120` 提升到 `58.347`。
2. 主要收益来自长尾和细粒度类别，尤其是 `tricycle`、`bus`、`awning-bike`；这与之前 PairMOT cls HOTA 低于 MOTRv2 的诊断一致。
3. additive margin 能证明类间边界问题存在，但目前不如 reweight 稳定；它对 `truck` 和 `tricycle` 的 MOTA 有负面影响。
4. 当前应将 `0713_01 longtail_reweight` 作为 long-tail 修复 baseline，后续探索应在它上面继续，而不是回到原始 0704 baseline。

## 9. 后续建议

- 在 `0713_01` 上继续做 class-adaptive reweight：降低 `truck` 的过高权重或改为基于 effective number 的平滑权重，观察 truck IDF1 是否恢复。
- 针对 `bus/tricycle/awning-bike` 保留较强正样本权重，因为这些类别收益最明确。
- margin 方向改为 confusion-aware margin：只对易混类别对施加约束，而不是对整个类别的 positive logit 统一加 margin。
- 增加 per-class score calibration 作为分析工具，输出混淆矩阵、score 分布和 FN/FP breakdown，但不作为最终模型推理策略。

