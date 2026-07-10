# 20260709 Liquid Unique AllGT Analysis

## 1. 对比对象

本报告比较 liquid 谱段采样实验与非 liquid 的同结构 baseline。

| 实验 | 角色 | workdir |
|---|---|---|
| `0704_01 resume` | 非 liquid baseline，`unique_pair_selection + PairDN + all-GT`，从 epoch 40 续训到 72 | `/data4/litianhao/PairMmot/workdir_252/0704_01_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_resume_from_epoch40_to72` |
| `0704_03_rerun_liquid_unique_allgt_init2_keep_lr_20260706` | 在同结构上加入 Liquid Spectral Sampling Conv3D stem，`init_logit=2` 且保持原学习率 | `/data4/litianhao/PairMmot/workdir_99/0704_03_rerun_liquid_unique_allgt_init2_keep_lr_20260706` |

选择规则：

- AP 单独按 `pair_mAP50:95` 选最佳 epoch。
- Tracking 不合并不同 epoch 的单项最优，统一按 `cls_HOTA + det_HOTA` 选唯一最佳 epoch。
- 类别 AP 的类别顺序来自 `HSMOTDataset.METAINFO`：`car, bike, pedestrian, van, truck, bus, tricycle, awning-bike`。
- TrackEval 的 per-class 指标直接使用 `track_class/<class>_*` 字段。

## 2. 总体结果

AP 结果：

| 实验 | AP epoch | pair mAP50:95 | pair AP50 | both mAP50:95 | both AP50 | new AP50 | disappear AP50 | independent AP50 | association gap AP50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0704_01 resume` | 67 | 0.2383 | 0.4157 | 0.2448 | 0.4275 | 0.0006 | 0.0005 | 0.4388 | 0.0231 |
| `liquid rerun` | 71 | 0.2395 | 0.4241 | 0.2463 | 0.4367 | 0.0004 | 0.0004 | 0.4468 | 0.0227 |
| delta | - | +0.0012 | +0.0084 | +0.0015 | +0.0092 | -0.0002 | -0.0001 | +0.0080 | -0.0004 |

Tracking 结果：

| 实验 | tracking epoch | cls HOTA | cls MOTA | cls IDF1 | det HOTA | det MOTA | det IDF1 | cls HOTA + det HOTA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `0704_01 resume` | 67 | 45.523 | 34.750 | 52.845 | 58.120 | 51.956 | 66.997 | 103.643 |
| `liquid rerun` | 63 | 46.416 | 34.243 | 54.290 | 57.876 | 51.231 | 66.950 | 104.292 |
| delta | - | +0.893 | -0.507 | +1.445 | -0.244 | -0.725 | -0.047 | +0.649 |

总体判断：

- liquid 有收益，但不是全面收益。它主要提升 `cls_HOTA`、`cls_IDF1`、`pair_AP50` 和 `both_AP50`。
- `pair_mAP50:95` 只提升 `+0.0012`，属于小幅收益。
- det-side tracking 没有提升：`det_HOTA=-0.244`，`det_IDF1=-0.047`。
- new/disappear 仍接近 0，liquid 没有解决 single-visible 召回问题。

## 3. cls tracking 类别分析

按 tracking 唯一最佳 epoch 对比：baseline 为 epoch 67，liquid 为 epoch 63。

| class | base HOTA | liquid HOTA | delta HOTA | base IDF1 | liquid IDF1 | delta IDF1 | base MOTA | liquid MOTA | delta MOTA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `awning-bike` | 43.350 | 40.573 | -2.777 | 53.112 | 50.755 | -2.357 | 33.724 | 31.066 | -2.658 |
| `bike` | 36.575 | 38.145 | +1.570 | 42.375 | 44.355 | +1.980 | 18.050 | 15.267 | -2.783 |
| `bus` | 56.876 | 58.439 | +1.563 | 71.675 | 73.899 | +2.224 | 57.564 | 55.067 | -2.497 |
| `car` | 79.250 | 78.994 | -0.256 | 90.221 | 90.126 | -0.095 | 86.194 | 86.194 | +0.000 |
| `pedestrian` | 39.055 | 38.638 | -0.417 | 50.215 | 49.985 | -0.230 | 31.495 | 30.219 | -1.276 |
| `tricycle` | 24.601 | 28.443 | +3.842 | 22.736 | 29.128 | +6.392 | 4.904 | 6.828 | +1.924 |
| `truck` | 27.474 | 29.759 | +2.285 | 26.624 | 29.096 | +2.472 | 4.878 | 6.194 | +1.316 |
| `van` | 57.002 | 58.334 | +1.332 | 65.803 | 66.975 | +1.172 | 41.192 | 43.106 | +1.914 |

按类别看，liquid 的 cls-side 增益集中在：

- `tricycle`：HOTA `+3.842`，IDF1 `+6.392`，MOTA `+1.924`，是最强增益类。
- `truck`：HOTA `+2.285`，IDF1 `+2.472`，MOTA `+1.316`。
- `bus`：HOTA `+1.563`，IDF1 `+2.224`，但 MOTA `-2.497`，说明 ID 连续性或匹配质量改善，但 FP/FN 侧没有同步改善。
- `bike`：HOTA `+1.570`，IDF1 `+1.980`，但 MOTA `-2.783`，同样是 IDF/HOTA 改善而 MOTA 下降。
- `van`：HOTA `+1.332`，IDF1 `+1.172`，MOTA `+1.914`，是较均衡的提升类。

下降类别：

- `awning-bike` 明显下降：HOTA `-2.777`，IDF1 `-2.357`，MOTA `-2.658`。
- `pedestrian` 小幅下降：HOTA `-0.417`，IDF1 `-0.230`，MOTA `-1.276`。
- `car` 基本持平略降：HOTA `-0.256`，IDF1 `-0.095`，MOTA 不变。

## 4. AP 类别分析

AP 类别对比使用各自 AP 最优点：baseline epoch 67，liquid epoch 71。

| class | base pair AP50 | liquid pair AP50 | delta | base both AP50 | liquid both AP50 | delta |
|---|---:|---:|---:|---:|---:|---:|
| `car` | 0.8885 | 0.8878 | -0.0006 | 0.9038 | 0.9037 | -0.0001 |
| `bike` | 0.3202 | 0.3402 | +0.0200 | 0.3302 | 0.3509 | +0.0206 |
| `pedestrian` | 0.3736 | 0.3606 | -0.0130 | 0.3905 | 0.3770 | -0.0136 |
| `van` | 0.5461 | 0.5607 | +0.0146 | 0.5602 | 0.5762 | +0.0160 |
| `truck` | 0.1055 | 0.1261 | +0.0206 | 0.1097 | 0.1328 | +0.0231 |
| `bus` | 0.5593 | 0.5947 | +0.0354 | 0.5779 | 0.6146 | +0.0367 |
| `tricycle` | 0.1293 | 0.1309 | +0.0016 | 0.1312 | 0.1329 | +0.0017 |
| `awning-bike` | 0.4031 | 0.3917 | -0.0114 | 0.4166 | 0.4056 | -0.0110 |

AP 侧与 tracking 侧基本一致，但不完全相同：

- AP50 提升最明显的是 `bus`、`truck`、`bike`、`van`。
- `car` 已经处在很高 AP50 区间，liquid 基本不改变它。
- `pedestrian` 和 `awning-bike` 在 AP50 与 tracking 上都下降。
- `tricycle` 的 AP50 只小幅提升，但 tracking IDF1 大幅提升，说明 liquid 对该类更可能改善时序身份一致性，而不是单帧/成对检测置信排序。

## 5. 解释与判断

liquid 是一个有创新性的方向，因为它不是在 pair head 或 matching 规则上继续堆逻辑，而是在 multispectral 输入侧学习谱段组合。当前结果支持这个方向有信号：

- 它提升了 cls-side tracking：`cls_HOTA +0.893`，`cls_IDF1 +1.445`。
- 它对中低基线类别更有帮助，尤其是 `tricycle`、`truck`、`bike`、`bus`、`van`。
- 它对 AP50 有稳定增益，说明谱段采样确实改善了一部分目标的可分性或置信排序。

但它还不能被表述为全面超过 baseline：

- `pair_mAP50:95` 只提升 `+0.0012`。
- det-side tracking 略降。
- `awning-bike` 和 `pedestrian` 被压低。
- new/disappear 仍然无效，说明 liquid 主要改善 both/survival 类目标，不解决 single-visible 建模。

## 6. 后续建议

1. 保留 liquid 作为创新主线，但不要只用总体 mAP 判断它；更应该围绕类别收益和谱段选择行为做分析。
2. 下一步优先检查 `tricycle/truck/bus/bike/van` 的 liquid sampler pattern，确认这些类别是否学到了稳定的非固定谱段组合。
3. 对 `awning-bike/pedestrian` 做失败样例可视化，判断下降来自谱段选择、尺寸尺度、遮挡，还是类别混淆。
4. 可以尝试 class-aware 或 instance-aware 的 liquid 正则，让低基线类别继续受益，同时约束 `awning-bike/pedestrian` 不被过度扰动。
5. 不建议把 liquid 与 tri-state/new-disappear 目标混为一个结论；当前 liquid 主要是 cls/survival 侧收益，single-visible 需要单独设计。
