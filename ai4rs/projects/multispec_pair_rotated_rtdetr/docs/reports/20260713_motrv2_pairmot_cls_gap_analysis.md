# 20260713 MOTRv2 vs PairMOT cls-HOTA Gap Analysis

## 1. 问题

MOTRv2 报告与 PairMOT `0704_01 resume` 对比时出现一个分裂现象：

- PairMOT 的 `det_HOTA` 更高，说明类无关检测/跟踪质量更强。
- PairMOT 的 `cls_HOTA` 更低，说明类别敏感 tracking 被类别混淆、FP 或 class-aware association 拉低。

本报告只把 per-class threshold 作为诊断工具，不作为最终模型方案。

## 2. 指标对比

数据来源：

- MOTRv2: `ai4rs/projects/multispec_pair_rotated_rtdetr/docs/reports/motrv2/all_cls_summary.csv`
- PairMOT: `/data4/litianhao/PairMmot/workdir_252/0704_01...resume_from_epoch40_to72/val_track_eval/val_track_0008/trackers/val_pairmot_0008/eval/all_cls_summary.csv`

| metric | MOTRv2 | PairMOT 0704 resume | Pair - MOTRv2 |
|---|---:|---:|---:|
| `cls_comb_cls_av HOTA` | 49.286 | 45.526 | -3.760 |
| `cls_comb_det_av HOTA` | 54.521 | 58.037 | +3.516 |
| `cls_av DetA` | 37.885 | 36.044 | -1.841 |
| `cls_av AssA` | 67.707 | 60.484 | -7.223 |
| `det_av DetA` | 44.181 | 48.852 | +4.671 |
| `det_av AssA` | 68.832 | 71.616 | +2.784 |
| `cls_av CLR_Pr` | 90.558 | 74.025 | -16.533 |
| `det_av CLR_Re` | 57.036 | 67.588 | +10.552 |

解释：

- PairMOT 类无关 recall 和 association 更强，所以 `det_HOTA` 高。
- PairMOT 在 class-aware 统计下 precision 和 association 明显下降，所以 `cls_HOTA` 低。
- `async_track_eval_payload.json` 中 `class_aware=false`，说明跟踪阶段主要按类无关轨迹做匹配；类别噪声会在 TrackEval 的 cls-aware 口径中被放大。

## 3. 类别差异

| class | MOTRv2 HOTA | PairMOT HOTA | delta | 主要现象 |
|---|---:|---:|---:|---|
| `truck` | 43.309 | 26.894 | -16.415 | DetA、AssA、precision 全面下降 |
| `bus` | 68.349 | 57.323 | -11.026 | recall 和 MOTA 明显弱于 MOTRv2 |
| `tricycle` | 31.718 | 25.333 | -6.385 | precision 和 association 下降明显 |
| `van` | 61.357 | 57.033 | -4.324 | 细粒度 vehicle 类混淆 |
| `awning-bike` | 44.615 | 43.063 | -1.552 | 小幅下降 |
| `bike` | 36.772 | 36.382 | -0.390 | DetA 提升但 AssA/precision 下降 |
| `pedestrian` | 36.433 | 39.123 | +2.690 | PairMOT 更强 |
| `car` | 71.738 | 79.058 | +7.320 | PairMOT 明显更强 |

判断：

- 这是明显的长尾/细分类问题，但不只是长尾样本数问题。
- PairMOT 在大类 `car/pedestrian` 上更强；弱点集中在 `truck/bus/van` 和 `bike/tricycle/awning-bike` 这类细粒度类别边界。
- 如果按 GT_Dets 加权，PairMOT 的 per-class HOTA 差值约为正；但 `cls_comb_cls_av` 是类别宏平均，长尾类与 `car/pedestrian` 权重相同，因此长尾细分类会主导 cls-HOTA 短板。

## 4. 已实现的模型修复

在 `PairRotatedRTDETRHead` 中新增两个可配置训练机制：

1. `cls_pos_loss_weights`
   - 对指定类别的正类 classification loss 加权。
   - VarifocalLoss 路径通过 `weight` 矩阵只放大正类位置，不改变 IoU quality target 数值。
   - 目标是提高 `truck/bus/tricycle/van/bike/awning-bike` 的有效分类梯度。

2. `cls_pos_logit_margins`
   - 训练 loss 前对正样本 true-class logit 减去 additive margin。
   - 目标是迫使细粒度类别形成更大的 true-class 间隔，专门针对类别混淆。
   - 只影响训练 loss，不改变推理后处理阈值。

代码位置：

- `projects/multispec_pair_rotated_rtdetr/multispec_pair_rotated_rtdetr/pair_rotated_rtdetr_head.py`

验证：

- `python -m py_compile` 通过。
- 本机 `py310` 配置加载通过。
- 本机 `py310` CPU model build 通过。
- 252 `py310` CPU model build 通过。

## 5. 252 已启动实验

| exp | GPU | config | workdir | 状态 |
|---|---|---|---|---|
| `0713_01 longtail_reweight` | `0,1` | `...allgt_longtail_reweight_252.py` | `/data4/litianhao/PairMmot/workdir_252/0713_01...longtail_reweight` | 已启动，确认到 `Epoch(train) [1][150/484]` |
| `0713_02 finecls_margin` | `2,3` | `...allgt_finecls_margin_252.py` | `/data4/litianhao/PairMmot/workdir_252/0713_02...finecls_margin` | 已启动，确认到 `Epoch(train) [1][150/484]` |

实验设计：

- `0713_01`：只做 long-tail positive reweight，测试 cls-HOTA 是否受长尾正样本梯度不足限制。
- `0713_02`：做 fine-grained positive margin，并配合较温和的 positive reweight，测试细粒度类别边界是否是核心问题。
- 两个实验都不改 box loss、proposal、association、tracker 参数或后处理阈值。

## 6. 预期判读

重点不看 AP，优先看：

1. `cls_comb_cls_av HOTA` 是否超过 `45.526`。
2. `det_comb_det_av HOTA` 是否保持接近 `58.037`。
3. `truck/bus/tricycle/van/bike/awning-bike` 的 per-class HOTA/IDF1 是否提升。
4. `car/pedestrian` 是否被 reweight/margin 牺牲过多。

若 `0713_01` 有效，说明问题主要是长尾正样本分类梯度不足。
若 `0713_02` 更有效，说明细粒度类别 margin 比简单重加权更关键。
若两者都提高 cls-HOTA 但 det-HOTA 基本不掉，可以继续做 `long-tail class refinement head`，把类别修正做成结构模块。
