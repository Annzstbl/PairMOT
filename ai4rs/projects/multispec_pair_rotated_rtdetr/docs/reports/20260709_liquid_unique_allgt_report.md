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

## 7. 2026-07-11 最新 multi-server 进展

本轮统计包含 99、252、197 的 liquid 相关实验。baseline 仍使用 `0704_01` resume 高指标：`cls_HOTA=45.523`，`det_HOTA=58.120`，`cls+det=103.643`；AP 对照为 `pair_mAP=0.2383`，`pair_AP50=0.4157`。

Tracking 仍按唯一规则选择最佳点：`cls_HOTA + det_HOTA` 最大的 async validation。

| exp | server | status | AP point | pair mAP | pair AP50 | both mAP | both AP50 | track point | cls HOTA | det HOTA | cls+det | vs baseline |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0709_01_liquid8` | 99 | finished epoch 72 | epoch 72 | 0.2457 | 0.4333 | 0.2526 | 0.4460 | async 18 | 46.803 | 57.899 | 104.702 | +1.059 |
| `0709_02_liquid8_liquidawarefusion` | 252 | finished epoch 72 | epoch 72 | 0.2432 | 0.4254 | 0.2501 | 0.4380 | async 18 | 46.328 | 57.994 | 104.322 | +0.679 |
| `0709_03_liquid8_laf_overlap` | 197 | finished epoch 72 | epoch 72 | 0.2419 | 0.4293 | 0.2488 | 0.4419 | async 17 | 46.573 | 58.025 | 104.598 | +0.955 |
| `0709_04_liquid8_laf_wide_overlap` | 252 | finished epoch 72 | epoch 72 | 0.2495 | 0.4367 | 0.2566 | 0.4494 | async 18 | 47.314 | 58.250 | 105.564 | +1.921 |
| `0709_05_liquid8_laf_patternbias` | 99 | finished epoch 72 | epoch 72 | 0.2414 | 0.4263 | 0.2480 | 0.4383 | async 18 | 46.346 | 58.077 | 104.423 | +0.780 |
| `0710_01_liquid8_groupmod` | 99 | finished epoch 72 | epoch 72 | 0.2423 | 0.4283 | - | - | async 18 | 46.672 | 58.214 | 104.886 | +1.243 |
| `0710_02_liquid8_laf_outputres` | 197 | finished epoch 72 | epoch 72 | 0.2434 | 0.4248 | - | - | async 18 | 46.190 | 58.275 | 104.465 | +0.822 |
| `0710_03_liquid8_sampler_bandattn` | 252 | running, epoch 48 observed | epoch 48 interim | 0.2424 | 0.4281 | - | - | async 12 interim | 46.099 | 57.445 | 103.544 | -0.099 |
| `0711_01_liquid8_laf_wide_groupmod` | 99 | finished epoch 72 | epoch 72 | 0.2479 | 0.4376 | 0.2550 | 0.4506 | async 18 | 47.484 | 58.421 | 105.905 | +2.262 |

表内实验改动与动机简述：

- `0709_01_liquid8`：将 liquid 谱段组从旧版扩展到 8 groups，动机是恢复 8 谱段解析力，并验证基础 liquid 是否比 6-group/旧 rerun 更稳。
- `0709_02_liquid8_liquidawarefusion`：加入基础 liquid-aware SE fusion，让 SE gate 感知采样 pattern，动机是解决普通 SE 对动态谱段组合不敏感的问题。
- `0709_03_liquid8_laf_overlap`：在 LAF 中加入 overlap/coverage context，动机是让模型知道不同 group 是否覆盖相似源谱段，减少谱段组合冲突。
- `0709_04_liquid8_laf_wide_overlap`：把 LAF descriptor 容量加宽到 `embed_dims=64` 并保留 overlap context，动机是增强 pattern-aware fusion 的表达力。
- `0709_05_liquid8_laf_patternbias`：弱化空间混合，偏向 pattern-only bias，动机是测试收益是否主要来自谱段 pattern，而不是空间分支。
- `0710_01_liquid8_groupmod`：加入 `LiquidGroupModulator`，用 group coverage/entropy/response 重标定每个 liquid group，动机是提升 det-side 稳定性。
- `0710_02_liquid8_laf_outputres`：让 LAF delta 直接残差注入 stem 输出，动机是测试 pattern 信息是否应绕过 SE gate 直接影响特征。
- `0710_03_liquid8_sampler_bandattn`：在 sampler 内加入 inter-band self-attention，动机是让原始谱段 descriptor 先做跨谱段对比再选择 group。
- `0711_01_liquid8_laf_wide_groupmod`：组合 wide LAF 与 groupmod，动机是叠加 cls-side pattern-aware fusion 收益和 det-side group 稳定收益。

最新判断：

- 252 的 `0709_04_laf_wide_overlap` 已成为当前 liquid 全局最佳，`cls+det=105.564`，比 baseline 高 `+1.921`，比 plain `liquid8` 高 `+0.862`。
- `0710_01_groupmod` 对 det-side 有价值：相对 plain `liquid8`，`det_HOTA +0.315`，但 `cls_HOTA -0.131`。它适合作为 wide LAF 的稳定器，而不是单独替代 wide LAF。
- `0710_02_laf_outputres` 主要提高 det-side，`det_HOTA=58.275`，但 cls-side 掉到 `46.190`，不建议继续沿 output residual 单独加深。
- `0710_03_sampler_bandattn` 目前中期偏弱，`cls+det=103.544`，还不能否定，但不应作为优先主线。
- 最新排序按 `cls_HOTA + det_HOTA` 为：`0711_01 wide LAF + groupmod` > `0709_04 wide LAF` > `0710_01 groupmod` > `0709_01 liquid8`。

解释：

1. wide LAF 的后半程明显追回并超过所有已完成变体，说明 pattern/group descriptor 需要足够容量，早期中期结果不能过早判断。
2. groupmod 证明“按采样覆盖关系调制每个 liquid group”能改善 det HOTA，这与 wide LAF 的 cls-side 强收益互补。
3. output residual 的直接特征注入可能扰动类别侧表征，虽然 det HOTA 高，但 cls HOTA 损失过大。
4. sampler band attention 的目标是改善谱段选择，但从中期看没有立刻带来 HOTA 收益，可能需要和 wide LAF 组合，或者等后半程确认。

## 8. 2026-07-11 新探索实验

基于上面的结果，本轮不再只调参数，而是组合两个有模型信号的结构：

| server | exp | model idea | launch status |
|---|---|---|---|
| 99 | `0711_01_liquid8_laf_wide_groupmod` | 以当前最佳 `wide LAF` 为主体，加入 `LiquidGroupModulator`，测试 group-level coverage modulation 是否能补上 wide LAF 的 det-side 稳定性。 | 已通过 detached `screen` 启动，GPU `1,2`，port `29813`，日志确认到 epoch 1 iter 50。 |
| 197 | `0711_02_liquid8_laf_wide_bandattn` | 以当前最佳 `wide LAF` 为主体，在 sampler 内加入 inter-band self-attention，测试谱段描述符先做跨 band 对比后是否能提升 group 选择质量。 | 已同步代码并启动，GPU `2,3`，port `29814`，日志确认到 epoch 1 iter 200。 |
| 252 | `0711_03_liquid8_laf_wide_groupmod_bandattn` | 以当前最佳 `wide LAF` 为主体，同时加入 `LiquidGroupModulator` 和 sampler inter-band attention，测试 group coverage modulation 与 band context 是否能在同一模型中叠加。 | 已同步代码并通过 detached `screen` 启动，GPU `0,1`，port `29815`，日志确认到 epoch 1 iter 50。 |
| 99 | `0712_01_liquid8_laf_wide_groupmod_outputres` | 以当前 99 最强 `wide LAF + groupmod` 为主体，加入小尺度 liquid-aware output residual，测试 output residual 的 det-side 直接注入能否在 groupmod 稳定后不再压低 cls-side。 | 已通过 detached `screen` 启动，GPU `0,1`，port `29816`，日志确认到 epoch 1 iter 50。 |

表内实验改动与动机简述：

- `0711_01`：`wide LAF + groupmod`，动机是验证 wide LAF 的语义收益与 groupmod 的 det-side 稳定性是否互补。
- `0711_02`：`wide LAF + sampler band attention`，动机是测试更强 fusion 容量下，跨谱段 descriptor 交互是否能改善 group 选择。
- `0711_03`：`wide LAF + groupmod + band attention`，动机是测试 group coverage modulation 与 band descriptor context 是否可以继续叠加。
- `0712_01`：`wide LAF + groupmod + output residual`，动机是检验直接注入 LAF delta 是否能带来 det-side 收益且不损伤 cls-side。

本轮预期：

- 如果 `0711_01` 超过 `105.564`，说明 wide LAF 的语义收益和 groupmod 的 det-side 稳定性可以叠加。
- 如果 `0711_02` 不超过 `0709_04`，而 `0710_03` 也继续偏弱，则 sampler attention 不是当前优先方向，后续应转向 class-aware/group-aware regularization。
- 如果 `0711_03` 超过 `0711_01` 和 `0711_02`，说明 coverage-aware group balancing 与 band descriptor context 不是互斥机制，可以作为下一版 liquid-aware fusion 的默认组合。
- `0712_01` 的判断重点不是 AP，而是它是否在 `0711_01` 的 `cls_HOTA=47.484` 基础上保持 cls 不掉，同时继续推高 `det_HOTA=58.421`。
- 若 `0711_03` 和 `0712_01` 都未超过 `0711_01`，当前最强结论应收敛到 `liquid8 + wide liquid-aware fusion + groupmod`。

## 9. 2026-07-12 Liquid 总结

本节按 `scalars.json` 重新抽取所有 liquid 关键实验。Tracking 仍按
`cls_HOTA + det_HOTA` 选择唯一最佳点，AP 按 `pair_mAP50:95` 选择最佳点。

| exp | status | pair mAP | pair AP50 | both mAP | both AP50 | cls HOTA | det HOTA | cls+det | vs baseline |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `0704_01 resume baseline` | finished | 0.2383 | 0.4157 | 0.2448 | 0.4275 | 45.523 | 58.120 | 103.643 | 0.000 |
| `0704_03 liquid rerun` | finished | 0.2395 | 0.4241 | 0.2463 | 0.4367 | 46.416 | 57.876 | 104.292 | +0.649 |
| `0709_01 liquid8` | finished | 0.2469 | 0.4327 | 0.2540 | 0.4456 | 46.803 | 57.899 | 104.702 | +1.059 |
| `0709_02 liquid8_laf` | finished | 0.2432 | 0.4254 | 0.2501 | 0.4380 | 46.328 | 57.994 | 104.322 | +0.679 |
| `0709_03 laf_overlap` | finished | 0.2419 | 0.4293 | 0.2488 | 0.4419 | 46.573 | 58.025 | 104.598 | +0.955 |
| `0709_04 laf_wide_overlap` | finished | 0.2495 | 0.4367 | 0.2566 | 0.4494 | 47.314 | 58.250 | 105.564 | +1.921 |
| `0709_05 laf_patternbias` | finished | 0.2414 | 0.4263 | 0.2480 | 0.4383 | 46.346 | 58.077 | 104.423 | +0.780 |
| `0710_01 groupmod` | finished | 0.2423 | 0.4283 | 0.2491 | 0.4408 | 46.672 | 58.214 | 104.886 | +1.243 |
| `0710_02 laf_outputres` | finished | 0.2434 | 0.4248 | 0.2503 | 0.4374 | 46.190 | 58.275 | 104.465 | +0.822 |
| `0710_03 sampler_bandattn` | finished | 0.2451 | 0.4314 | 0.2521 | 0.4441 | 46.892 | 57.836 | 104.728 | +1.085 |
| `0711_01 wide_groupmod` | finished | 0.2493 | 0.4390 | 0.2565 | 0.4520 | 47.484 | 58.421 | 105.905 | +2.262 |
| `0711_02 wide_bandattn` | finished | 0.2493 | 0.4363 | 0.2563 | 0.4489 | 47.092 | 58.074 | 105.166 | +1.523 |
| `0711_03 wide_groupmod_bandattn` | finished | 0.2545 | 0.4426 | 0.2618 | 0.4556 | 47.627 | 58.227 | 105.854 | +2.211 |
| `0712_01 wide_groupmod_outputres` | finished | 0.2459 | 0.4305 | 0.2530 | 0.4433 | 46.401 | 57.856 | 104.257 | +0.614 |

最终排序按 `cls_HOTA + det_HOTA`：

1. `0711_01 wide_groupmod`: `105.905`，当前 tracking 最优。
2. `0711_03 wide_groupmod_bandattn`: `105.854`，AP 最优，但 HOTA 比 `0711_01` 低 `0.051`。
3. `0709_04 laf_wide_overlap`: `105.564`，证明 wide LAF 是主收益来源。
4. `0711_02 wide_bandattn`: `105.166`，band attention 与 wide LAF 组合有效，但不如 groupmod。
5. `0710_01 groupmod`: `104.886`，单独 groupmod 提升 det-side，但缺少 wide LAF 的 cls-side 收益。

结论：

- liquid 主线成立。相对 baseline，最佳 HOTA 从 `103.643` 提升到 `105.905`，增益 `+2.262`。
- 最可靠结构是 `liquid8 + wide liquid-aware fusion + LiquidGroupModulator`。它同时达到最高 det HOTA `58.421` 和很高 cls HOTA `47.484`。
- `wide LAF` 是关键跃迁点：plain `liquid8` 到 `wide LAF`，sum 从 `104.702` 到 `105.564`，说明 pattern-aware fusion 需要足够 descriptor 容量和 overlap context。
- `LiquidGroupModulator` 是第二个有效模块：`0709_04` 到 `0711_01`，sum `+0.341`，主要强化 det-side 稳定性。
- `sampler band attention` 对 AP 有帮助，`0711_03` 给出最高 `pair_mAP=0.2545`、`pair_AP50=0.4426`，但 HOTA 没超过 `0711_01`。如果目标是论文表格里的 AP，它有价值；如果目标是 tracking，不能作为默认替代 groupmod-only。
- `output residual` 不建议继续。`0712_01` 从 `0711_01` 的 `105.905` 掉到 `104.257`，cls 和 det 都下降，说明直接把 LAF delta 注入输出特征会扰动主干表征。
- new/disappear 仍接近 0，liquid 主要改善 both-visible/survival 场景，不解决 single-visible 建模。

下一步建议：

1. 当前默认候选模型应定为 `0711_01 wide_groupmod`。
2. 如果需要 AP 表格，可以保留 `0711_03 wide_groupmod_bandattn` 作为 AP-oriented variant。
3. 后续不要继续加 output residual；更值得做的是 class-aware/group-aware regularization，目标是保住 `tricycle/truck/bus/bike/van` 的收益，同时抑制 `pedestrian/awning-bike` 的下降。
4. 做最终报告时，HOTA 主表用 `0711_01`，AP 辅表可报告 `0711_03`，但必须明确二者不是同一最佳选择规则。

## 10. 2026-07-15 Pair-aware liquid descriptor

本节补充本机 99 的 `0714_01 liquid8_pairaware_laf_wide`。Tracking 仍只按
`cls_HOTA + det_HOTA` 选择唯一最佳 epoch，不与其他 epoch 的 AP 或其他指标拼接。

### 10.1 直接 baseline 与模型改动

该实验的直接结构 baseline 是 `0709_04 liquid8_laf_wide_overlap`，不是
`0704_01 resume`，也不是包含额外 `LiquidGroupModulator` 的 `0711_01`。配置直接继承
`0709_04`，保留 8-group liquid sampler、`embed_dims=64` 的 wide LAF、overlap context
和 spatial mixer，仅新增 `PairAwareLiquidFusion`。

`PairAwareLiquidFusion` 允许 prev/curr 两帧继续独立采样。它在 sampler 之后分别提取
每个 group 的源谱段 coverage、coverage entropy、peak coverage 和响应强度，再组合
`[src, other, src-other, src*other]` 描述两帧的差异与一致性，输出逐 group 的
SE-logit residual。该分支采用 zero initialization，动机是让 fusion 显式感知两帧
采样 pattern 的变化，同时不在训练初始阶段破坏已有 wide LAF。

### 10.2 HOTA 结果

| exp | role | unique best epoch | cls HOTA | det HOTA | cls+det | vs direct baseline | vs 0704 resume |
|---|---|---:|---:|---:|---:|---:|---:|
| `0704_01 resume` | global baseline | best resume point | 45.523 | 58.120 | 103.643 | -1.921 | 0.000 |
| `0709_04 laf_wide_overlap` | direct structural baseline | epoch 72 | 47.314 | 58.250 | 105.564 | 0.000 | +1.921 |
| `0714_01 pairaware_laf_wide` | pair-aware experiment | epoch 72 | 46.782 | 58.077 | 104.859 | -0.705 | +1.216 |

相对直接 baseline，`0714_01` 的 `cls_HOTA` 下降 `0.532`，`det_HOTA` 下降
`0.173`，两者没有一项提升。相对当前 liquid tracking 最优
`0711_01 wide_groupmod` 的 `105.905`，总和低 `1.046`。因此当前证据不支持把该 pair-aware
descriptor 加入默认 liquid 模型。

### 10.3 AP 辅助观察

AP 单独按 `pair_mAP50:95` 选择，`0714_01` 的最佳点为 epoch 64：
`pair_mAP=0.2435`、`pair_AP50=0.4310`、`both_mAP=0.2505`、
`both_AP50=0.4437`。它也低于直接 baseline `0709_04` 的
`pair_mAP=0.2495`、`pair_AP50=0.4367`。此 AP 点不用于上面的 HOTA 行。

结论：pair-aware 的问题定义仍合理，但当前把紧凑 pair descriptor 作为附加
SE-logit residual，与 wide LAF 已有的 pattern-aware gate 存在功能重叠，且响应强度
在 descriptor 中被 detach，分支只能学习如何调 gate，不能反向塑造 group response。
现有实现没有形成互补增益，应保留为负结果；liquid 默认候选仍是
`0711_01 liquid8 + wide LAF + groupmod`。

## 11. Pair-Consistent Spectral Transport

针对 `0714_01` 的负结果，下一版不再向 SE gate 追加一个独立 pair residual，而是让
sampler 和 wide LAF 通过实际采样分布形成同一条 pair-aware 路径。严格 baseline 选择
`0711_01 liquid8 + wide LAF + groupmod`，其指标为 `cls_HOTA=47.484`、
`det_HOTA=58.421`、`cls+det=105.905`。

新结构包含两个耦合部分：

1. `PairCoupledSamplerRouter`：prev/curr 仍独立选择谱段，但各自的 sampler hidden 与
   paired hidden 组成 `[src, other, src-other, src*other]`，双向预测 sampler-logit
   residual。该设计让选择过程感知另一帧，同时不强迫两帧使用相同 pattern。
2. `PairTransportTokenCoupling`：根据两帧 sampler 输出的源谱段 coverage 计算
   group-to-group transport matrix。wide LAF 不按固定 group index 对齐，而是聚合另一帧
   中谱段覆盖最相关的 group token，再以同样的差异/一致性关系更新当前 token。

这两个模块的最后一层均为 zero initialization，因此训练起点严格等价于 `0711_01`；
groupmod、wide LAF、overlap context 和 spatial mixer 均保持不变。相比 `0714_01`，该设计
的关键区别是 pair sampler 的结果直接决定 pair fusion 的跨帧对齐关系，而不是在已有
fusion 后重复增加 SE bias。soft sampling 保持连续融合和可导 transport，hard/eval-hard
仍通过原有去重逻辑保证每个 group 内谱段唯一。

实验配置为 `0715_02 liquid8_laf_wide_groupmod_pairtransport`。首次运行按要求在 epoch 1
iter 200 后停止，没有生成 checkpoint。随后于 2026-07-15 01:52 在本机 99 的 GPU
`2,3` fresh restart，不使用 resume，并改为 `setsid + nohup` 独立会话。rerun 已通过
epoch 1 iter 50，loss 正常。该运行随后到达 epoch 70 iter 400，但物理 GPU2
（PCI `0000:b1:00.0`）掉卡；驱动当前返回 `Unknown Error`，PCI 设备显示 `rev ff` 且
VBIOS 不可读。DDP 两个 rank 等待 30 分钟后触发 NCCL ALLREDUCE timeout。故障前 loss、
grad norm 和显存均正常，没有 OOM、NaN 或模型 traceback，因此归因为 GPU/PCIe 硬件
故障。最后可恢复 checkpoint 为 epoch 68；epoch-67 validation 与 async TrackEval 17
已经完成。服务器重启后四张 GPU 已恢复正常识别。该历史运行启动早于统一 AMP 约定，
实际配置为 FP32 `OptimWrapper` 和 `find_unused_parameters=True`，不能误记为 BF16 实验。
重启后使用完整 `epoch_68.pth` 在 GPUs `0,1` 显式 resume；model、optimizer、scheduler、
message hub、EMA 和 early-stopping 状态均成功恢复。续训已验证至 epoch 69 iter 100，
loss `10.9303`、grad norm `40.2505`，速度 `0.8573 s/iter`。续训随后完成 epoch 72、
最终 AP 验证和 TrackEval。由于 resume 后异步评测计数从 1 重新开始，最终 epoch 72
（payload `step=71`）结果写入并覆盖了 `val_track_0001`，不能把该目录误认为 epoch 4。

按 `cls_HOTA + det_HOTA` 选择唯一最佳点，最终 epoch 72 同时也是该实验最佳点：
`cls_HOTA=47.520`、`det_HOTA=58.600`、总和 `106.120`。相对严格结构 baseline
`0711_01 wide_groupmod`，分别变化 `+0.036`、`+0.179`、总和 `+0.215`；相对
`0704_01 resume` 总和提高 `+2.477`。最终 AP 为 `pair_mAP=0.2540`、
`pair_AP50=0.4448`、`both_mAP=0.2611`、`both_AP50=0.4575`，仅作为辅助指标。
因此 Pair Transport 是当前 liquid 的 HOTA 新最优，并且预期的 det-side 改善已经出现，
但相对 `0711_01` 的增幅较小，应视为边际正收益，后续最好在统一 BF16 基准上复验。

效率验证使用 RTX 3090、每卡 4 pairs（展平 8 帧）和 `400x600` 输入，对 baseline 与
新结构各重复三轮 CUDA event 测时并取中位数。stem 前向由 `10.349 ms` 增至
`10.463 ms`（`+1.10%`），stem 前向反向由 `33.878 ms` 增至 `34.903 ms`
（`+3.03%`），峰值显存增加不超过 `0.7 MiB`。新增计算只有约 6.27 万参数，主要作用于
每帧全局 hidden、`8x8` transport matrix 和 64 维 group token，不在高分辨率特征图上
执行跨帧 attention；因此放到完整模型和实际更大输入中，占总迭代时间的比例预计低于
约 `0.2%`。

实际训练日志报告显存 `13412 MiB`；直接 baseline `0711_01` 的历史显存中位数为
`13453 MiB`、最大值为 `13481 MiB`。`nvidia-smi` 显示的约 `18.4 GiB` 还包含 CUDA
caching allocator、cuDNN workspace 和上下文保留，不能视为新增模块的有效激活占用。

## 12. Band-Aligned Pair Context

为避免只依赖 `0715_02 Pair Transport`，新增结构互补的 `0715_03`。严格 baseline 仍为
`0711_01 liquid8 + wide LAF + groupmod`，不叠加 Pair Transport。

`PairBandContextEncoder` 在 sampler 的物理谱段 token 上逐 band 对齐 prev/curr，以
`[src, common, src-other, src*other]` 建模稳定谱段信息和帧间变化。该 context 通过两条
共享路径进入模型：一条修正 sampler band descriptor 并直接产生 sampler-logit residual；
另一条由实际采样 coverage 从 band context 池化为 group context，再注入 wide LAF token。
因此 sampler 和 fusion 使用同一份谱段级 pair 表征，而不是分别学习无关联的 pair 分支。

该设计相对 Pair Transport 的潜在优势是：pair 交互发生在具有固定物理含义的 8 个源谱段
上，不需要等待 sampler 先形成稳定 group pattern 才能可靠对齐；同时保留每帧独立选择，
不会强制 prev/curr 使用相同谱段。sampler descriptor、sampler logits 和 LAF token 的注入
均为 zero initialization，训练起点严格等价于 baseline。

新增参数为 `24384`，只操作 `8x32` band token 和 `8x64` group token，不在空间特征图上
做跨帧运算。15 个单元测试已通过，包括 baseline 前向等价、sampler/fusion 首步梯度、
单帧回退以及现有 hard 去重逻辑。实验尚未启动：原 GPU `2,3` 队列曾被 `0715_02` 的
残留进程阻塞，服务器重启后该队列与残留进程均已消失。后续如启动，需要按当前统一的
BF16-through-encoder、`find_unused_parameters=False` 配置重新建立启动任务。
停止新结构后，又在同一 GPU `2,3`、同一 batch 和 FP32 设置下临时运行 `0711_01`
baseline 到 epoch 1 iter 50。baseline 的 `nvidia-smi` 为 `18575/18469 MiB`，训练日志
为 `13457 MiB`；新结构对应为 `18437/18451 MiB` 和 `13411--13464 MiB`。两者等价，
没有观察到可归因于 pair-aware 模块的显存增长；临时 baseline 随后也已停止。

## 13. Pair Change-Gated Liquid Fusion

`0715_04` 从 `0711_01 liquid8 + wide LAF + groupmod` 结构出发，但不叠加 Pair Transport
或 Band Context。它解决的问题是：两帧中稳定谱段适合共享，而真实运动、遮挡或光谱变化
不应被无条件平均。模块先按相同 group index 计算两帧采样 coverage 的 histogram
intersection、L1 distance，以及 conv3d group 响应均值/方差的相对变化，再产生逐 group
reliability gate。高可靠性 group 偏向共享 token，低可靠性 group 偏向带方向的
frame-specific change token，最后以 zero-initialized residual 注入 wide LAF。

该结构不做新的 self-attention、group-to-group 矩阵或高分辨率跨帧卷积，新增计算只作用于
8 个 64 维 group token。新增参数 `12833`。RTX 3090、8 帧、`400x600` 输入的三轮相邻
stem 前向微基准中，baseline 为 `9.112--9.137 ms`，新结构为
`9.234--9.251 ms`，增幅 `1.25%--1.34%`；放到完整模型中的占比更低。单元测试验证了
zero-init 时与 `0711_01` 输出严格一致，并确认所有新增参数均接入梯度，可使用
`find_unused_parameters=False`。

实验遵循 2026-07-15 后的新训练基准：BF16 through encoder、后续 FP32、
`find_unused_parameters=False`、fresh train，并保留 validation 与 TrackEval，只关闭绘图。
197 GPUs `2,3` 上的正式运行已到 epoch 1 iter 100，loss `24.7166`、grad norm
`58.0278` 均有限，8-group pattern 正常。iter 50 含启动预热为 `2.251 s/iter`，iter 100
恢复到 `1.037 s/iter`；服务器同时存在 GPU0 满载任务，因此该总迭代速度不用于估计模块
自身开销。截至 2026-07-15 21:50，训练已进入 epoch 61，约 `1.04 s/iter`，loss 和 grad
norm 正常，训练 ETA 约 1 小时 45 分。按 `cls_HOTA + det_HOTA` 选择的当前唯一最佳点为
epoch 52 / payload `step=51`：`cls_HOTA=46.298`、`det_HOTA=57.768`、总和 `104.066`。
epoch 56 回落到 `45.989 + 57.701 = 103.690`；epoch 60 的 TrackEval 正在异步执行。
当前最好点高于 `0704_01 resume`，但仍比历史 FP32 `0711_01` 最终总和 `105.905` 低
`1.839`。后半程仍有 12 个 epoch，是否能证明 change gate 有正收益必须等待最终唯一最佳点。

### 13.1 与 Pair Transport 的有限对比

将 `0715_04 change gate` 与本机 `0715_02 pair transport` 按相同 epoch 对齐。该比较只能
判断候选竞争力，不能解释为单模块消融：`0715_02` 同时包含 pair sampler router 和
pair transport，使用 FP32 与 `find_unused_parameters=True`；`0715_04` 只加入 change
gate，使用 BF16 与 `find_unused_parameters=False`。两者 sampler seed 均为 `3407`，数据、
训练轮数和评测间隔一致。

| Epoch | Transport cls_HOTA | Transport det_HOTA | Change gate cls_HOTA | Change gate det_HOTA | 选择分数差值 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 46.756 | 57.568 | 45.531 | 56.800 | -1.993 |
| 40 | 46.603 | 57.790 | 45.982 | 57.353 | -1.058 |
| 48 | 46.870 | 57.802 | 45.995 | 57.582 | -1.095 |
| 52 | 46.808 | 57.988 | 46.298 | 57.768 | -0.730 |
| 56 | 46.787 | 57.953 | 45.989 | 57.701 | -1.050 |

epoch 8--56 的 13 个共同评测点中，change gate 的 `cls_HOTA + det_HOTA` 均未超过
pair transport。差距在后期缩小，说明模型仍在收敛，但 epoch 56 再次回落，当前轨迹不支持
其作为 pair transport 的替代方案。epoch 52 的逐类 HOTA 显示 change gate 对 bike
`+1.064`、tricycle `+0.635`、van `+0.593`、pedestrian `+0.085`，但 awning-bike
`-3.074`、bus `-2.450`、truck `-0.544`、car `-0.392`。因此它有有限的类别互补潜力，
尤其是 bike/tricycle/van，但现有 reliability gate 存在类别偏置。合理定位是保留其机制
用于后续“transport 后的轻量置信门控”设计，而不是把当前 `0715_04` 升为默认 Liquid。

## 14. Final Pair-Only Liquid On Full Data

`0715_05` 将 liquid 收敛为完整的 pair-aware stem：8-group sampler、pair-conditioned
sampler router、Conv3D group encoding、group modulator、wide overlap-aware LAF 和
coverage-based pair transport。sampler router 与 transported-token relation 均只使用有序
`[x,y]`，不再显式拼接 `x-y` 或 `x*y`。历史 relation 模式仍保留为默认选项以保证旧配置
和 checkpoint 可复现；本实验在两处显式选择 `relation_mode='pair'`。

实验使用全部 75 个训练序列、COCO+Objects365 direct adapted checkpoint、全局 batch 8、
72 epochs、2000-iter warmup、BF16 through encoder、后续 FP32，以及
`find_unused_parameters=False`。普通 backbone 保持 `lr=1e-5`；Conv3D/SE 和全部新
liquid 参数均通过最终解析配置确认使用 `lr=1e-4`。

PyTorch 2.0.1/CUDA 11.8 不实现 BF16 `bilinear` 或 `nearest` interpolate。首次启动在
sampler low-resolution gradient correction 处报 dtype 错误，未完成任何 iteration。
短暂的 FP32-bilinear fallback 随后按要求停止。代码保留了经过前向和反向对照测试的
自定义纯 BF16 bilinear 作为可选项，但为避免自实现插值成为正式实验的风险变量，最终配置
显式选择 `lowres_grad_upsample_mode='nearest'`。该方案保留 full-resolution
`P.detach()` 前向和 1/4-resolution sampler 概率梯度，再用纯 BF16 `index_select` 将
零值 correction 最近邻展开到原尺寸。correction 前向严格为零，因此真实 sampled
feature 不变；nearest 只决定 sampler 近似梯度的空间聚合方式。

固定压力测试张量 `8x8x800x1200` 的 sampler BF16 前后向测试通过，head gradient 和输出
均有限。正式 HSMOT pipeline 对 `1200x900` 原图等比例缩放到约 `1067x800`，再按 32
对齐 pad 到 `1088x800`；该压力测试尺寸不代表正式数据的实际 resize 输出。
正式 fresh run 于 2026-07-15 18:35 在 99 GPUs `0,1` 启动，已验证到 epoch 1 iter 150；
iter 100/150 分别为 `0.8195/0.8087 s/iter`，训练日志显存约 `8.36 GiB`，loss/grad
finite，初始 pattern 为
`701 / 012 / 123 / 234 / 345 / 456 / 567 / 670`，没有 unused-parameter 或 dtype 错误。
与临时 FP32-bilinear 路径的早期速度处于同一波动区间；nearest 方案的主要收益是实现
简单、纯 BF16 执行和去除 full-resolution FP32 correction 临时张量，目前不能声称有
显著速度提升。

### 14.1 Full-data 完整结果

`0715_05` 已完成 72 epochs、18 个 validation 和对应的 18 个 TrackEval 点。严格按
`cls_HOTA + det_HOTA` 选择唯一最佳，最佳点为最终 `val_track_0018`，payload
`step=71`，对应 val_det epoch 71：

| experiment | cls HOTA | det HOTA | cls MOTA | cls IDF1 | det MOTA | det IDF1 |
|---|---:|---:|---:|---:|---:|---:|
| full baseline `0714_01` | 52.374 | 60.318 | 44.159 | 62.126 | 57.407 | 70.957 |
| full liquid `0715_05` | 53.472 | 60.907 | 44.951 | 62.704 | 58.652 | 71.215 |
| delta | +1.098 | +0.589 | +0.792 | +0.578 | +1.245 | +0.258 |

用于唯一最佳点选择的两项 HOTA 之和从 `112.692` 提高到 `114.379`，变化 `+1.687`。
指标展示仍保持 cls/det 分离；该和只用于选择唯一 checkpoint。

AP 独立按 pair mAP 选择，双方最优均为 epoch 72：

| experiment | pair mAP | pair AP50 | both mAP | both AP50 |
|---|---:|---:|---:|---:|
| full baseline `0714_01` | 0.2928 | 0.5062 | 0.3011 | 0.5209 |
| full liquid `0715_05` | 0.2988 | 0.5115 | 0.3070 | 0.5256 |
| delta | +0.0059 | +0.0052 | +0.0058 | +0.0047 |

最佳 HOTA 点的逐类 cls HOTA：

| class | full baseline | full liquid | delta |
|---|---:|---:|---:|
| car | 80.004 | 81.115 | +1.111 |
| bike | 41.266 | 41.597 | +0.331 |
| pedestrian | 42.192 | 42.325 | +0.133 |
| van | 61.745 | 62.166 | +0.421 |
| truck | 39.610 | 40.150 | +0.540 |
| bus | 71.302 | 71.378 | +0.076 |
| tricycle | 37.831 | 42.903 | +5.072 |
| awning-bike | 45.040 | 46.140 | +1.100 |

八个类别的 cls HOTA 均提高，最大收益来自 tricycle `+5.072`，其次是 car `+1.111` 和
awning-bike `+1.100`。这说明最终 Liquid 不只是提高总体 detection association，也改善了
多个类别的分类一致性；但 bike、pedestrian 和 bus 的增益较小，仍有继续优化空间。

该对比的数据集、COCO+Objects365 adapted initialization、72 epochs 和评测协议一致，
但不是严格的单变量 Liquid 消融：`0714_01` 使用 FP32 `OptimWrapper` 和
`find_unused_parameters=True`，`0715_05` 使用 BF16 through encoder、
`find_unused_parameters=False`，并包含同期的 DDP/KLD 稳定性修正。因此可以确认当前
full liquid 系统稳定超过 full baseline 性能锚点，不能把全部 `+1.098/+0.589` HOTA
增益都归因于 Liquid 模块本身。严格归因仍需要同一 BF16 代码基线的 full-data rerun。

## 15. Pair-Only Band Context On Full Data

`0715_06` 将未运行的 `0715_03` 更新为当前统一规范并在 252 上执行全量实验。模型以
`liquid8 + wide LAF + groupmod` 为主体，在 8 个具有固定物理含义的源谱段上构建双向
pair context；同一 context 一路修正 sampler descriptor 与 logits，另一路按实际 sampling
coverage 池化为 group context 后注入 wide LAF。这样 sampler 与 fusion 共享同一份 pair
证据，同时仍允许两帧独立选择谱段。

`PairBandContextEncoder` 新增可复现的 `relation_mode`。历史默认
`pair_diff_product` 保留用于旧配置，本实验显式使用 `relation_mode='pair'`，关系输入仅为
有序 `[src, other]`。实验不包含 pair sampler router、pair transport 或 change gate；
band-context fusion 只消费已经编码的 context，不再次构造差值或乘积。两个注入出口保持
zero initialization，单元测试确认初始输出与 wide-groupmod baseline 一致，并确认 sampler
与 fusion 两条新增路径均有非零梯度。

训练使用全部 75 个序列、COCO+Objects365 direct adapted checkpoint、全局 batch 8、
72 epochs、2000-iter warmup、BF16 through encoder、后续 FP32、nearest sampler gradient
expansion 和 `find_unused_parameters=False`。基础 LR 为 `1e-4`；实际构建优化器后确认
Conv3D、SE、sampler、pair-band encoder、band-context fusion、wide LAF 与 groupmod 均为
`1e-4`，普通 backbone 参数为 `1e-5`。

252 的第一次 22:01 启动在模型构建前退出，因为远端仍停留在不支持当前 BF16 边界参数的
旧 detector 代码。同步本机稳定的 detector、head、RT-DETR layer 和 GDLoss 实现后，
正式 fresh run 于 2026-07-15 22:05 在 GPUs `0,1`、port `29878` 启动，没有 resume。
实验已完成 72 epochs、18 个 validation 和全部 18 个 TrackEval 点，训练期间没有
unused-parameter、dtype、GMC 或 DDP 错误。

### 15.1 完整结果

严格按 `cls_HOTA + det_HOTA` 选择唯一最佳，最佳点为 `val_track_0017`，payload
`step=67`，对应训练 epoch 68 / val_det epoch 67。最终 epoch 72 的选择分数从
`112.194` 回落到 `112.050`，因此不使用最后一个 checkpoint。指标保持 cls/det 分离
展示，二者之和只用于选择 checkpoint：

| experiment | unique best epoch | cls HOTA | det HOTA | cls MOTA | cls IDF1 | det MOTA | det IDF1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| full baseline `0714_01` | 72 | 52.374 | 60.318 | 44.159 | 62.126 | 57.407 | 70.957 |
| pair transport `0715_05` | 72 | 53.472 | 60.907 | 44.951 | 62.704 | 58.652 | 71.215 |
| band context `0715_06` | 68 | 51.817 | 60.377 | 43.162 | 60.815 | 56.502 | 70.348 |
| `0715_06` vs baseline | - | -0.557 | +0.059 | -0.997 | -1.311 | -0.905 | -0.609 |
| `0715_06` vs `0715_05` | - | -1.655 | -0.530 | -1.789 | -1.889 | -2.150 | -0.867 |

用于唯一 checkpoint 选择的分数分别为 baseline `112.692`、`0715_05` `114.379`、
`0715_06` `112.194`。因此 band context 略微提高 det HOTA，但 cls HOTA 的下降更大，
综合结果比 baseline 低 `0.498`，比 pair transport 低 `2.185`。其后期轨迹仍在上升，
epoch 60/64/68 的选择分数为 `111.717/111.774/112.194`，但 epoch 72 已开始回落，
不存在通过继续选择最后 checkpoint 扭转结论的依据。

AP 独立按 pair mAP 选择，`0715_06` 的 AP 最佳也为 epoch 68：

| experiment | AP epoch | pair mAP | pair AP50 | both mAP | both AP50 |
|---|---:|---:|---:|---:|---:|
| full baseline `0714_01` | 72 | 0.2928 | 0.5062 | 0.3011 | 0.5209 |
| pair transport `0715_05` | 72 | 0.2988 | 0.5115 | 0.3070 | 0.5256 |
| band context `0715_06` | 68 | 0.2920 | 0.4957 | 0.3001 | 0.5097 |
| `0715_06` vs baseline | - | -0.0009 | -0.0105 | -0.0010 | -0.0112 |
| `0715_06` vs `0715_05` | - | -0.0068 | -0.0157 | -0.0068 | -0.0159 |

唯一最佳 HOTA 点的逐类 cls HOTA 为：

| class | full baseline | `0715_05` | `0715_06` | `0715_06` vs baseline | `0715_06` vs `0715_05` |
|---|---:|---:|---:|---:|---:|
| car | 80.004 | 81.115 | 80.633 | +0.629 | -0.482 |
| bike | 41.266 | 41.597 | 40.584 | -0.682 | -1.013 |
| pedestrian | 42.192 | 42.325 | 41.844 | -0.348 | -0.481 |
| van | 61.745 | 62.166 | 59.655 | -2.090 | -2.511 |
| truck | 39.610 | 40.150 | 35.824 | -3.786 | -4.326 |
| bus | 71.302 | 71.378 | 70.405 | -0.897 | -0.973 |
| tricycle | 37.831 | 42.903 | 38.795 | +0.964 | -4.108 |
| awning-bike | 45.040 | 46.140 | 46.796 | +1.756 | +0.656 |

Band context 对 awning-bike、tricycle 和 car 有局部收益，其中 awning-bike 还超过
`0715_05`；但 truck、van、bus、bike 和 pedestrian 均下降。尤其 truck `-3.786` 和
van `-2.090` 主导了 cls-side 回落。该结果说明逐物理波段的 pair context 具有类别互补
信息，但当前“同一 context 同时修正 descriptor、sampler logits 和 LAF token”的注入
过强或存在偏置，不能替代 `0715_05` 的 pair sampler router + coverage transport。

### 15.2 Sampler 多样性诊断

两个实验都只保证单个三波段 group 内无重复，不约束不同 group 选择相同波段集合。
`0715_06` 后期 epoch 69--72 平均只有 `6.19/8` 个不同有序组合；忽略排列后只有
`5.19/8` 个不同波段集合，且所有监控点均存在跨 group 重复。`0715_05` 同期分别为
`5.89/8` 和 `4.89/8`，塌缩更严重。`0715_06` 的较高多样性没有转化为更高 tracking
指标，说明“不同组合更多”本身不足以证明 pair context 有效；但两个模型均未真正实现
8 个有效不同谱段组，这也是当前 Liquid 仍需修正的共同结构缺陷。

后续不增加辅助 loss，而是在 hard train/eval 中使用全局集合唯一的 straight-through
分配：8 个 group 从 `C(8,3)=56` 个无序三波段集合中选择互不相同的集合，集合内部保留
得分最高的排列；soft fusion 阶段保持连续融合不变。该改动应首先在当前最强的
`0715_05` 结构上做严格复验，`0715_06` 保留为类别互补分析分支。

## 16. 论文严格 Base + Liquid 消融

`0716_03` 使用论文统一协议重新验证最终 Liquid：COCO-only初始化、原生1200x900输入、
全量数据的8297个唯一有序 `t-1 -> t` pair、BF16、`find_unused_parameters=False`和完整
72-epoch评测。相对同步运行的`0716_02`，唯一模型变化是8-group最终Liquid，包括独立
sampler、wide overlap-aware LAF、group modulation、pair sampler router和pair transport。

本机GPU 2故障导致首次运行在epoch 1作废且不resume。代码同步后，正式fresh run于
2026-07-16 17:15 CST在197的GPU 0/3启动，workdir为
`/data4/litianhao/PairMmot/workdir_197/0716_03_paper_base_plus_liquid_r18_coco_full_1200x900_bf16_orderedpairs_fresh`。
epoch 1 iter 50为`1.0818 s/iter`、日志显存`10692 MB`，loss和grad有限，无CUDA、NCCL、
NaN、OOM、unused parameter或DDP错误。该运行在epoch 21 iter 50主动停止，不resume；
原因是soft阶段的argmax预览已经出现`432/431`等相同无序波段集合，继续运行不能解决
跨group坍塌。

`0716_04`在完全相同的最终Liquid结构和论文训练协议上只增加hard group-set unique
assignment。每个样本先对`C(8,3)=56`个无序三波段集合枚举内部6种排列，以三个slot的
logit和选择最佳排列，再使用regret-first GPU greedy为8个group分配互不相同的集合。
hard train/eval使用该离散结果；straight-through反向仍经过对应soft概率。soft fusion阶段
不屏蔽任何group且数学路径保持不变，不增加辅助loss。

新实验于2026-07-16 23:22 CST在197 GPU 0/3 fresh启动，独立workdir为
`/data4/litianhao/PairMmot/workdir_197/0716_04_paper_base_plus_liquid_groupsetunique_r18_coco_full_1200x900_bf16_orderedpairs_fresh`。
远端20项sampler/stem测试全部通过；epoch 1 iter 50为`0.9771 s/iter`、loss和grad有限，
监控为`hard=False, unique_sets=8.00, max_set_repeat=1.00`。完成后将与`0716_02`按唯一最佳
`cls_HOTA + det_HOTA` epoch进行严格比较。

### 16.1 `0716_04`完整结果

实验已完成72 epochs、18次validation及18/18 TrackEval。唯一最佳为epoch 72：

| experiment | epoch | cls HOTA | det HOTA | cls+det | cls MOTA | cls IDF1 | det MOTA | det IDF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Paper Base `0716_02` | 68 | 53.314 | 61.982 | 115.296 | 44.690 | 62.218 | 60.599 | 72.303 |
| group-set-unique Liquid `0716_04` | 72 | 54.113 | 61.636 | 115.749 | 47.607 | 64.198 | 60.355 | 71.992 |
| Liquid vs Base | - | +0.799 | -0.346 | **+0.453** | +2.917 | +1.980 | -0.244 | -0.311 |

同一epoch 72比较时，Base总和为`114.990`，Liquid提升`+0.759`。Liquid同一最佳epoch的
`pair mAP=0.3160`、`pair AP50=0.5301`，相对Base正式epoch 68分别为`+0.0011`和
`+0.0076`。主结论仍以HOTA为准：Liquid已经超过论文Base，但提升集中在cls侧。

曲线上，Liquid到epoch 52仍比同epoch Base低`0.042`，从epoch 56开始连续领先；
epoch 60/64/68/72的同epoch优势为`+0.580/+0.430/+0.212/+0.759`。hard监控始终保持
`unique_sets=8.00, max_set_repeat=1.00`，没有重新出现完全相同集合的跨group坍塌。

类别HOTA相对Base最佳点的变化为：`truck +6.861`、`tricycle +4.415`、`van +1.025`；
`car -0.119`、`bus -0.524`、`pedestrian -1.075`、`awning-bike -1.228`、
`bike -2.967`。该结构显著改善truck/tricycle等困难类别，但det HOTA和bike类仍是后续
组合模块需要补偿的主要短板。

## 17. Set-Transport Liquid

### 17.1 动机与结构

`0716_04`只在hard train/eval保证8个group使用不同无序三波段集合，soft阶段仍按
`[group, slot, band]`独立归一化。因此模型可能先在soft阶段形成相同集合偏好，到hard
切换时才把低置信的次优集合分配给其他group，存在明显的soft/hard目标错位。

`0717_01`新增无参数的Set-Transport Liquid Sampler。它在pair sampler router之后将slot
概率提升为`C(8,3)=56`个无序集合，每个集合内部显式保留6种排列；再加入48个slack
token，将8个真实group与56个集合构成方阵，通过16次log-domain Sinkhorn得到行质量为1、
每个集合容量不超过1的连续分配。集合分配与排列概率最后还原为原接口`[B,8,3,8]`，因此
Conv3D、wide LAF、group modulation和pair transport均不改变。该结构不增加loss和可学习
参数，pair router产生的双帧条件logits直接参与集合竞争。

Set-Transport强度从epoch 0的0线性增加到epoch 12的1，初始前向严格等价`0716_04`；
epoch 12后soft梯度完全经过集合容量投影，epoch 36进入hard时straight-through梯度仍经过
同一投影。这样在不禁止soft谱段融合的前提下，使soft学习逐步接近hard唯一分配流形。

### 17.2 正确性、开销与实验状态

23项sampler/stem测试全部通过，包括零强度精确等价、极端8-group同集合输入下容量上限、
概率归一化、soft梯度和hard straight-through梯度。正式初始化下仍保持
`701/012/123/234/345/456/567/670`，平均概率改变量仅`0.0086`。纯sampler前反向增加约
`20 ms/iter`，完整训练预计增加约2%--2.5%，参数量增加0。

本机GPU 2/3的100-iter双卡DDP测试通过，`find_unused_parameters=False`无报错，显存
`10695 MB/rank`，loss和grad有限。正式`0717_01`于2026-07-17 00:15 CST fresh启动；
epoch 1 iter 50为`0.9347 s/iter`、`loss=36.0532`、`grad_norm=177.8087`，监控为
`set_transport=0.004, set_max_load=0.250`。workdir为
`/data4/litianhao/PairMmot/workdir_99/0717_01_paper_base_plus_liquid_settransport_r18_coco_full_1200x900_bf16_orderedpairs_fresh`。

该实验是相对`0716_04`的单变量Liquid结构候选，不在结果出来前替代论文主线。两者均按
各自全部18个评测点的唯一最佳`cls_HOTA + det_HOTA`进行比较，指标分别展示。

2026-07-17按用户要求在epoch 2 iter 250主动取消本机正式训练，原因是GPU 2/3存在历史
掉卡风险，不是模型loss、显存或DDP错误。对应进程组和screen已全部清理，GPU 2/3显存均
释放到约10 MiB；GPU 0/1上的Base训练不受影响。该不完整运行不产生性能结论，后续若转移
到稳定服务器，应使用新workdir fresh训练，不从本机早期状态续跑。

### 17.3 AutoDL完整结果

AutoDL fresh run已完成72 epochs和18/18 TrackEval。按唯一最大
`cls_HOTA + det_HOTA`选择epoch 68：

| experiment | epoch | cls HOTA | det HOTA | cls+det | cls MOTA | cls IDF1 | det MOTA | det IDF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Paper Base `0716_02` | 68 | 53.314 | 61.982 | 115.296 | 44.690 | 62.218 | 60.599 | 72.303 |
| group-set-unique `0716_04` | 72 | 54.113 | 61.636 | 115.749 | 47.607 | 64.198 | 60.355 | 71.992 |
| Set-Transport `0717_01` | 68 | 54.941 | 61.836 | **116.777** | 47.401 | 65.140 | 60.658 | 72.523 |
| Set-Transport vs `0716_04` | - | +0.828 | +0.200 | **+1.028** | -0.206 | +0.942 | +0.303 | +0.531 |
| Set-Transport vs Base | - | +1.627 | -0.146 | **+1.481** | +2.711 | +2.922 | +0.059 | +0.220 |

同一epoch 68的`pair mAP=0.3196`、`pair AP50=0.5359`。epoch 72总和为116.730，
距最佳点仅0.047，说明提升不是孤立波动。Set-Transport相对直接对照`0716_04`同时提高
cls和det HOTA，验证soft集合容量约束及soft/hard一致性有效；但det HOTA仍比Paper Base低
0.146，因此当前最准确结论是“Liquid内部明确改进，整体det侧基本恢复但尚未超过Base”。

相对`0716_04`，类别HOTA为`truck +6.008`、`bike +2.128`、`pedestrian +0.437`、
`awning-bike +0.166`、`van +0.134`、`car -0.110`、`tricycle -0.717`、`bus -1.418`。
truck是cls提升的主要来源。末期hard route始终保持8个不同无序集合，但完整pattern仍集中
于少数组合；该结果支持集合覆盖约束的有效性，不足以证明route已具备强图像自适应性。

## 18. Hard-sampled Soft-context Liquid

`0716_04`进入hard后，GroupMod、overlap-aware LAF和Pair Transport只能看到one-hot谱段
集合，丢失选择置信度和次优谱段概率。`0717_03`不改变全局唯一hard采样，而是让实际谱段
采样与Conv3D继续使用`P_hard`，GroupMod descriptor、LAF pattern/overlap及Pair Transport
coverage改用同一次采样对应的`P_soft`。该改动不增加参数和loss；soft阶段两种概率相同，
所以epoch 36前严格等价`0716_04`，只有hard阶段的融合上下文发生变化。

本机和197端24项测试通过，包括默认路径回归、hard one-hot前向、soft context连续性及两条
梯度有限。`0717_03`于2026-07-17 21:59 CST在197 GPU 4/5 fresh启动，workdir为
`/data4/litianhao/PairMmot/workdir_197/0717_03_paper_base_plus_liquid_hardsoftcontext_r18_coco_full_1200x900_bf16_orderedpairs_fresh`。
epoch 1 iter 100为`0.8514 s/iter`、`10692 MB/rank`、`loss=31.9603`、
`grad_norm=158.1667`，训练无异常。正式判断必须观察epoch 36 hard切换以后，并按18个
TrackEval点中唯一最佳`cls_HOTA + det_HOTA`与`0716_04`和Paper Base比较。

2026-07-18按新实验设计主动停止该运行，停止点为epoch 15 iter 150，最近周期checkpoint为
epoch 12。进程组已完整退出，GPU 4/5释放；checkpoint保留作诊断记录，但该实验不resume，
也不作为性能结论。

## 19. Independent-group Difference/Product Liquid

### 19.1 动机与唯一变量

`0718_01`以已完成的AutoDL Set-Transport模型路径为结构参照，保留8-group Liquid sampler、
pair sampler router、wide overlap-aware LAF、GroupMod和PairTransport，但撤销跨group集合
唯一化：`hard_group_unique_sets=False`、`soft_group_set_transport=None`。各group重新独立
选择三波段，允许不同group学习同一集合；单个group内部仍采用无放回采样，因此不会产生
`227`这类组内重复波段。

同时，sampler router和fusion内PairTransport都从仅拼接`[x,y]`改为显式关系
`[x,y,x-y,x*y]`。有符号差分直接表达帧间变化，乘积表达共同响应，目的是恢复历史half
实验中更强的图像条件route变化；该关系只作用于轻量token/descriptor，不经过全分辨率
特征图，计算增量很小。实验不增加loss，也不修改Conv3D、检测器、训练数据或评测协议。

这是一个联合结构假设，不应被解释为“取消唯一化”或“difference/product”任一单因素的
严格消融。它直接检验当前目标：在不强制8组全不相同的情况下，显式pair关系能否同时获得
更高HOTA和更强route自适应性。route分析需同时报告跨图像pattern变化、group重复数量和
最终HOTA，不能只以无重复作为成功标准。

### 19.2 启动状态

配置在本机与197均完整解析，24项sampler/stem测试在两端通过。`0718_01`于
2026-07-18 02:05 CST在197 GPU 4/5 fresh启动，沿用Paper Base协议：COCO适配初始化、
full HSMOT、原生`1200x900`、ordered gap-1 pairs、每卡batch 4、BF16、
`find_unused_parameters=False`和72 epochs。epoch 1 iter 100为`0.9272 s/iter`、
`10690 MB/rank`、`loss=32.8708`、`grad_norm=176.9842`，所有decoder、DN与encoder loss
均有限，未发现CUDA、NCCL、OOM、NaN或DDP错误。workdir为：

`/data4/litianhao/PairMmot/workdir_197/0718_01_paper_base_plus_liquid_independent_diffproduct_r18_coco_full_1200x900_bf16_orderedpairs_fresh`。

## 20. Anchor-Residual Competitive Liquid

### 20.1 目标与结构

`0718_02`的主要目标按优先级定义为：首先超过Paper Base；同时避免group route退化为公共
固定pattern；跨group集合唯一只作为次要诊断，不作为硬约束。为此不使用额外diversity
loss、Set-Transport或hard全局唯一分配，而是在sampler内部加入Anchor-Residual
Competitive Router（ARCR）。

原始sampler由一个全局hidden通过共享head一次产生全部`8x3x8` logits，任意数据集级公共
波段偏置都能同时覆盖8个group的初始锚点。ARCR保留固定循环锚点
`701/012/123/234/345/456/567/670`，并将所有可学习logits分解为：

1. 跨group公共修正：限制在`[-0.5,0.5]`，只能微调总体谱段偏好，不能压过强度2.0的锚点；
2. 零均值group-specific residual：容量为`[-2,2]`，允许有证据的单个group改变选择；
3. pair content evidence：复用sampler已经计算的每波段mean/std/max，在每张图内跨band标准化，
   以`[x,y,x-y,x*y]`编码双帧关系，再由共享band encoder与group-slot query做余弦匹配；
   该分支不含band/group输出偏置，不能退化为纯数据集常量。

content score先对band去均值、再对group去均值，以强度0.35加入group-specific residual。
sampler router和LAF PairTransport也继续使用显式difference/product。不同group允许重复真正
有价值的集合，单group内部仍无放回采样，因而不会出现`227`。结构新增1,488个可训练参数，
只在`8 bands x 8 groups x 3 slots`小张量上计算，不增加全分辨率特征操作。

### 20.2 99正确性与效率验证

26项sampler/stem测试全部通过。新增测试包括：向所有group注入相同的
`+100/+80/+60`三波段公共偏置后仍保留至少6个不同集合；不同pair输入产生不同content
score；输入、content encoder和group-slot query均获得有限非零梯度。

99 GPU 2/3上完成两组同样本、full `1200x900`、每卡batch 4、BF16、
`find_unused_parameters=False`的50-iter DDP对照。ARCR为`0.912 s/iter`、
`10663 MiB/rank`，关闭ARCR为`0.933 s/iter`、`10663 MiB/rank`；速度差属于短程波动，结论
是没有可测开销。ARCR末次诊断为`common_abs=0.001`、`specific_abs=0.042`、
`content_sample_std=0.039`，说明公共偏置受控且内容证据随样本变化。全部detector、DN和
encoder loss及梯度有限，未出现NaN、OOM、unused parameter或DDP错误。

正式配置为
`o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_plus_liquid_anchorcompetitive_coco_full_1200x900_bf16_99.py`。
正式训练完成前不能提前声称已经超过Base。实验除HOTA外还必须报告
`image_variant_ratio`、`unique_sets`、`max_set_repeat`、
`common_abs`、`specific_abs`和`content_sample_std`；主要成功条件是HOTA超过Base且route
保持输入条件变化，是否8组完全唯一仅作次要观察。

### 20.3 AutoDL正式训练

AutoDL升级为两张RTX 4080 SUPER后，远端配置解析、26项测试、两次训练迭代及单批推理
smoke全部通过；推理输出300个有限预测。正式实验于2026-07-18 02:44 CST fresh启动，
epoch 1 iter 50为`0.9379 s/iter`、框架显存`10699 MiB/rank`、`loss=34.3786`、
`grad_norm=134.8329`，全部decoder、DN及encoder loss有限。此时
`common_abs=0.001`、`specific_abs=0.041`、`content_sample_std=0.112`，说明内容证据已随
样本变化；`image_variant_ratio=0`且pattern仍为初始8组，属于锚点占优的warmup初期，不能
据此判断最终route是否自适应。

训练和唯一finalizer均已脱离SSH运行。finalizer将等待72 epochs及18/18异步TrackEval，按
唯一最大`cls_HOTA + det_HOTA`选择checkpoint，以`0716_02` Paper Base作比较，将报告、
日志、TrackEval及选中checkpoint保存到`/autodl-fs/data/PairMOT_results/0718_02`后关机。
上一轮关机已删除GitHub deploy key，因此本轮不自动push，shared FS为权威结果源。

### 20.4 最终结果与route诊断

训练及18/18 TrackEval均已完成。唯一最大`cls_HOTA + det_HOTA`位于epoch 64，正式结果为
`cls_HOTA=53.357`、`det_HOTA=61.054`、`cls_MOTA=44.894`、`cls_IDF1=62.722`、
`det_MOTA=58.778`、`det_IDF1=71.450`；同epoch `pair_mAP=0.3088`、
`pair_AP50=0.5218`。相对Paper Base，两个HOTA分别变化`+0.043/-0.928`，没有同时提升。

固定终点epoch 72为`cls_HOTA=53.118`、`det_HOTA=61.216`、`cls_MOTA=44.515`、
`cls_IDF1=62.263`、`det_MOTA=59.349`、`det_IDF1=71.630`；同epoch
`pair_mAP=0.3094`、`pair_AP50=0.5219`。该点不替换预注册的epoch 64结果。

epoch 64逐类只有truck `+10.302`和tricycle `+0.582`提升；awning-bike、bike、bus、car、
pedestrian、van分别变化`-2.562/-4.116/-1.596/-0.347/-1.785/-0.140`。ARCR保留truck
收益，但对bike、pedestrian和多数常见类别不利，是cls收益有限且det侧下降的主要表现。

末50个hard监控点包含50种完整pattern，平均`image_variant_ratio=0.875`、
`changed_ratio=0.538`、`unique_sets=7.570`、`max_set_repeat=1.398`、
`content_sample_std=0.262`。各group有6--24种无序集合，组内无重复波段，未发生route固定
坍塌；偶发跨group集合复用符合独立采样设计。ARCR因此解决了route自适应性问题，但没有
转化为检测收益，不作为最终Liquid结构。

finalizer已完成归档，权威结果位于`/autodl-fs/data/PairMOT_results/0718_02`。

### 20.5 Confidence-Preserving Adaptive Set Router

#### 历史实验结论

`0716_04`证明hard集合唯一化能阻止跨group重复并提高cls，但det HOTA仍下降；`0717_01`
Set-Transport进一步改善soft/hard一致性，使cls/det HOTA相对`0716_04`同时提高，是目前最稳
的Liquid主体。`0718_01`取消集合唯一化并加入difference/product后cls提升最大，但末50个
监控点只剩2种pattern，说明高性能任务logit仍会退化为数据集级固定route。`0718_02` ARCR
反过来获得50/50种末期pattern，却使det HOTA下降`0.928`，bike、pedestrian分别下降
`4.116/1.785`。

ARCR掉点的核心不是输入自适应本身，而是以下三个结构因素叠加：固定强度2.0的循环锚点在
整个训练期持续参与决策；原任务logit被拆成common/specific后分别截断到`0.5/2.0`，强任务
证据无法完整保留；内容分支基于全图mean/std/max，背景变化也能推动hard route。由此产生的
route变化很多，但不一定对应目标，尤其容易伤害占图面积小的bike和pedestrian。truck仍大幅
提升说明Liquid的谱段条件建模有效，问题在于路由校准而不是模块方向错误。

#### CPAS结构

新候选`0718_06`采用Confidence-Preserving Adaptive Set Router（CPAS）。性能主体继承
`0717_01`的hard group-set uniqueness和soft Set-Transport，并吸收`0718_01`在sampler
router和LAF PairTransport中验证出高cls潜力的显式`[x,y,x-y,x*y]`关系；不使用ARCR固定
锚点或logit截断。检测任务学习到的原logit完整保留，pair-conditioned内容分支只输出残差：

1. 每个band复用已有mean/std/max，以`[x,y,x-y,x*y]`构造双帧内容key；
2. 内容分数分别沿band和group去均值，公共数据集偏置不能同时推动所有group；
3. 用任务logit归一化top-2 margin估计置信度，只在候选接近时开放较大内容残差；
4. 高置信选择只保留5%的门控下限，不强制翻转任务已经确定的谱段；
5. 残差按任务logit标准差缩放并限制尺度，避免训练后期logit放大使内容分支失效或反客为主。

跨group集合重复由Set-Transport在结构上禁止。跨图像soft route始终保留无bias、双中心化的
输入条件残差；hard pattern只在任务低置信候选间改变，不以制造随机pattern为目标。这避免
了ARCR“route变化强但任务无关”的问题。新增参数1,440个，计算仅发生在
`8 groups x 3 slots x 8 bands`张量上，不增加全分辨率操作、额外loss或新超参数调度。

该组合的优先级是明确的：Set-Transport与difference/product负责超过Base，依据分别是
`0717_01`相对直接对照的det HOTA `+0.200`和`0718_01`相对Base仅`-0.170`的det差距及
`+1.962`的cls收益；CPAS只负责在低margin处恢复输入条件变化。即使CPAS最终没有显著增加
hard pattern数量，它也不应压过前两条已验证性能路径。该设计比ARCR类“先保证route变化，
再补性能”的顺序更符合当前实验事实。

正式配置为
`o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_cpas_settransport_coco_full_1200x900_bf16_99.py`。
模型验收除HOTA和逐类结果外，必须联合观察`preserving_residual`、`preserving_gate`、
`preserving_margin`、`preserving_content_std`及确定性验证集route分布。成功标准仍是cls与det
HOTA均不低于Paper Base，并保持bike/pedestrian不出现ARCR式显著下降。

本机32项stem/sampler单测全部通过，包括高置信logit不翻转、低置信残差更强、pair输入
敏感性、band/group双中心化、内容分支梯度、Set-Transport联合反传和router互斥。GPU2上的
BF16前向/反向smoke同样通过，所有残差与梯度有限。`4x8x256x320`纯stem短程微基准中，
Set-Transport为`55.32 ms`，加入CPAS为`60.08 ms`；约`4.76 ms`增量主要来自小张量kernel
launch。相对现有正式整网约`0.9 s/iter`，预期总体开销约0.5%，仍需正式DDP短测确认。

2026-07-18 23:16补充检查表明，继续沿ARCR加结构的成功概率较低：`0718_03`
adaptive-anchor ARCR到epoch 44为`cls_HOTA=53.676`、`det_HOTA=60.380`，末50个监控点虽有
50种pattern，但det仍明显低于Base；`0718_04 SASE`到epoch 28为`49.832/57.512`，局部
证据分支尚未带来更好的收敛轨迹。这进一步支持`0718_06`以高性能Set-Transport和
difference/product为主体、仅弱化使用CPAS的设计顺序。

优化器解析确认CPAS位于`backbone.stem.0.liquid_sampler`，匹配现有`lr_mult=1.0`，实际
学习率为`1e-4`；普通预训练backbone仍为`1e-5`。因此不为新分支增加独立学习率或额外调度。
正式训练必须fresh加载COCO adapted权重，并保持Paper Base的full HSMOT、`1200x900`、
ordered pairs、BF16、全局batch 8和72 epochs协议。

## 21. Evidence-Consistent Adaptive-Anchor Liquid

### 21.1 动机与结构

`0718_02`用固定循环锚点和有界公共修正阻止所有group被数据集级公共偏置覆盖，但强度2.0的
固定锚点也可能让已经随图像变化的content score长期无法改变hard route。`0718_03`保留
ARCR的抗坍塌边界，只增加Evidence-Consistent Anchor Relaxation：对每个group-slot分别计算
学习到的group-specific residual与当前pair content delta在band维的正余弦一致度，并以
residual RMS衡量任务证据强度。只有二者同向且强度超过0.08时，才连续减弱该位置的锚点；
反向或弱证据保持原锚点不变。

最大松弛比例限制为0.45，因此锚点至少保留55%；该模块没有新增参数、loss或全分辨率操作，
也不强制跨group唯一。其目标不是制造随机route变化，而是允许检测任务已经学到、并由当前
双帧光谱内容支持的选择越过固定锚点，同时继续抑制公共模式坍塌。监控新增
`anchor_scale_mean/min`，需与`image_variant_ratio`、HOTA和group重复度共同判断效果。

### 21.2 验证与运行状态

27项sampler/stem测试全部通过，其中定向测试验证：强且同向的证据可将锚点缩放到约0.55，
反向证据严格保持1.0，弱证据保持0.98以上。配置核对确认full HSMOT、原生`1200x900`、
ordered gap-1 pairs、COCO适配初始化、BF16、`find_unused_parameters=False`和72 epochs。

99 GPU 2/3的50-iter双卡DDP smoke完成，末次为`0.9401 s/iter`、`10663 MiB/rank`、
`loss=34.5873`、`grad_norm=121.4444`，全部detector、DN和encoder loss有限。初始
`anchor_scale mean/min=0.999/0.994`，符合warmup早期证据较弱时保持锚点的设计预期。

正式实验编号为`0718_03`。带文件锁的队列同时确认`0717_02`精确训练进程不存在、两卡显存
均低于1024 MiB后，于2026-07-18 10:19 CST在99 GPU 0/1自动fresh启动，证明排队逻辑按设计
生效。epoch 9训练约`0.92 s/iter`，loss和grad有限；`image_variant_ratio`已达到
`0.25--0.50`，锚点最小缩放约`0.58--0.61`，说明输入条件路径已开始越过部分固定锚点。
最终结果仍按18个评测点中唯一最大`cls_HOTA + det_HOTA`选epoch，并从同一epoch记录AP。

### 21.3 最终结果与route诊断

`0718_03`已完成72 epochs和18/18 TrackEval。唯一最大`cls_HOTA + det_HOTA`位于epoch 60；
epoch 56/60/72的HOTA和分别为`115.648/115.658/115.614`，最佳点在后期较稳定，但仍严格
选择epoch 60并从同一epoch取AP。

| experiment | epoch | cls HOTA | det HOTA | cls MOTA | cls IDF1 | det MOTA | det IDF1 | pair mAP | pair AP50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Paper Base | 68 | 53.314 | 61.982 | 44.690 | 62.218 | 60.599 | 72.303 | 0.3149 | 0.5225 |
| `0718_02` ARCR best | 64 | 53.357 | 61.054 | 44.894 | 62.722 | 58.778 | 71.450 | 0.3088 | 0.5218 |
| `0718_03` unique best | 60 | **54.689** | 60.969 | **47.555** | **65.255** | 58.737 | 71.512 | **0.3152** | **0.5302** |
| change vs Base | - | +1.375 | -1.013 | +2.865 | +3.037 | -1.862 | -0.791 | +0.0003 | +0.0077 |
| `0718_03` endpoint | 72 | **54.645** | 60.969 | **47.892** | **64.969** | 58.636 | 71.176 | **0.3182** | **0.5324** |

相对父实验0718_02，adaptive anchor使cls HOTA提高`1.332`，det HOTA仅下降`0.085`，HOTA和
提高`1.247`，说明松弛固定锚点明显改善了ARCR的任务适配。但相对Paper Base，det侧
`DetRe/DetA/AssA`分别变化`-1.882/-1.464/-0.200`：损失主要来自目标召回，而非关联。

epoch 60逐类cls HOTA相对Base：truck `+14.604`、tricycle `+5.550`；pedestrian、bike、
awning-bike分别`-2.476/-2.515/-2.631`，car、van、bus分别`-0.517/-0.969/-0.045`。
因此macro cls提升由稀有truck/tricycle拉动，而class-agnostic det被高频小目标漏检主导，
仍是长尾收益与小目标召回之间的结构性权衡。

route侧没有坍塌。末50个hard监控点有50种完整pattern，8个group各有9--33种ordered
pattern；平均`image_variant_ratio=0.875`、`changed_ratio=0.662`、
`unique_sets=7.482`、`max_set_repeat=1.450`、`content_sample_std=0.159`。结论是adaptive
anchor成功实现内容相关route，但route多样性不能自动转化为检测质量；该模型不替代Base，
后续结构必须直接保护小目标证据或采用pair-shared稳定route，而不能继续单独放宽锚点。

## 22. Original-Hard Liquid Strict-Control Result

99上的`0717_02`已完成72 epochs和18/18异步TrackEval。它不使用跨group集合唯一约束或
Set-Transport；每个group内部仍无放回采样，不会产生`227`，不同group之间允许重复集合。
这是Paper Base协议下对原版Liquid路径的严格验证。

按全部评测点中唯一最大`cls_HOTA + det_HOTA`选中epoch 72，且AP严格取同一epoch：

| experiment | epoch | cls HOTA | det HOTA | cls MOTA | cls IDF1 | det MOTA | det IDF1 | pair mAP | pair AP50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Paper Base `0716_02` | 68 | 53.314 | 61.982 | 44.690 | 62.218 | 60.599 | 72.303 | 0.3149 | 0.5225 |
| Original-hard Liquid `0717_02` | 72 | 54.335 | 61.445 | 47.502 | 64.540 | 59.848 | 71.905 | 0.3215 | 0.5429 |
| change | - | +1.021 | -0.537 | +2.812 | +2.322 | -0.751 | -0.398 | +0.0066 | +0.0204 |

类别HOTA相对Base唯一最佳点变化如下：

| class | Base | `0717_02` | change |
| --- | ---: | ---: | ---: |
| truck | 33.852 | 41.297 | +7.445 |
| tricycle | 39.283 | 43.508 | +4.225 |
| bus | 71.483 | 71.747 | +0.264 |
| van | 62.825 | 62.830 | +0.005 |
| car | 81.516 | 81.286 | -0.230 |
| bike | 43.807 | 42.924 | -0.883 |
| awning-bike | 49.109 | 47.961 | -1.148 |
| pedestrian | 44.640 | 43.126 | -1.514 |

该模型直到epoch 60才首次超过Base同epoch，epoch 64/68/72的选择目标相对同epochBase分别
变化`-0.020/+0.098/+0.790`，表现出比Base更慢的后期收敛。最终cls提升主要由truck和
tricycle驱动，而pedestrian、bike和awning-bike下降；det HOTA、MOTA和IDF1均回落。因此
它证明原版Liquid具有长尾识别价值，但不能作为要求cls/det同时提升的论文最终结构。

路由方面，末期50条hard监控只有4种完整pattern，且变化基本只发生在第一组`514/521`和
第六组`430/431`，其余6组固定。各完整pattern内部仍有8个不同无序集合，所以没有出现
所有group选择同一集合的跨组坍塌；但图像条件route变化明显不足。监控只采样各rank周期
batch，不能代替全验证集route统计，但已足以说明后续ARCR类结构应重点解决输入自适应性，
而不是仅增加硬唯一约束。

## 23. Scale-Adaptive Shared Sparse Evidence Liquid

### 23.1 动机与结构

`0717_02`表明原版Liquid能显著提高truck和tricycle，却使pedestrian、bike、awning-bike及
det侧指标下降。现有sampler和LAF token主要依赖全图mean/std/max；稀疏小目标对全局统计
贡献弱，而且LAF位于采样之后，无法恢复sampler已经丢弃的关键谱段。`0718_04`因此在
`0718_03 adaptive-anchor ARCR`上加入Scale-Adaptive Shared Sparse Evidence（SASE），使
同一份局部证据在采样前影响route，并在采样后增强LAF。

SASE先按波段做图像内标准化，并按stem stride下采样2倍；分别计算3x3和9x9中心-周围绝对
响应。共享线性层根据两个尺度各自的mean/std为每张图、每个波段预测二路softmax权重，不
设置额外温度。融合contrast在band维计算RMS saliency并做空间和归一化，由此提取每个波段
的有符号响应及局部对比强度。band evidence再做跨波段标准化，使用
`[prev,curr,prev-curr,prev*curr]`构建pair关系，与group-slot query生成band去均值的sampler
logit residual。

LAF不重新计算证据：按当前谱段coverage聚合相同band evidence，zero-init投影后加入原group
token；同时由coverage聚合contrast得到group-specific局部图，通过每组zero-init可学习增益
调制现有spatial mixer。ARCR、GroupMod、overlap-aware LAF、Pair Transport、SE及检测器均
保持不变。不增加loss、类别规则、box监督、阈值、top-k比例或连续手工系数；两个固定尺度是
结构定义。所有新增输出为zero-init，训练初始前向严格等价`0718_03`。

### 23.2 正确性、开销与运行状态

本机和252均通过28项sampler/stem测试。新增测试覆盖：紧凑局部目标对应波段具有更高RMS
contrast；二路尺度权重归一化；SASE group map形状正确；初始化输出与无SASE模型严格一致；
sampler evidence、pair projection、router gain、LAF token projection和局部空间增益全部进入
计算图并获得有限梯度。配置解析确认full HSMOT、原生`1200x900`、ordered gap-1 pair、
COCO适配初始化、BF16、`find_unused_parameters=False`和72 epochs。

新增参数精确为1,402个。99 GPU 2/3的50-iter完整DDP smoke为`0.9484 s/iter`、
`11236 MiB/rank`，相对`0718_03`短程对照约增加0.8%时间和573 MiB/rank；全部loss和grad
有限。252 GPU 2/3的20-iter环境smoke也通过，显存`11231 MiB/rank`，无CUDA、NCCL、OOM、
NaN或unused-parameter错误。

正式`0718_04`于2026-07-18 13:12 CST在252 GPU 0/1 fresh启动，screen为
`pairmot_0718_04_sase_252`，workdir为
`/data4/litianhao/PairMmot/workdir_252/0718_04_paper_liquid_adaptiveanchor_sase_r18_coco_full_1200x900_bf16_orderedpairs_fresh`。
正式epoch 1 iter 100为`1.1228 s/iter`、`11262 MiB/rank`、`loss=31.5313`、
`grad_norm=146.8564`；所有分支有限，初始`scale3=0.500`且两类sparse gain仍接近zero-init，
未发现训练异常，ETA约23小时39分。
除HOTA外需跟踪pedestrian/bike/awning-bike、`sparse_scale3`、sampler/LAF sparse gain、
`image_variant_ratio`和group重复度。最终仍按18个评测点中唯一最大
`cls_HOTA + det_HOTA`选epoch，并从同一epoch记录AP。

### 23.3 最终结果

`0718_04`已完成72 epochs和18/18 TrackEval。唯一最佳为epoch 68，epoch 72仅作为训练终点：

| experiment | epoch | cls HOTA | det HOTA | cls MOTA | cls IDF1 | det MOTA | det IDF1 | pair mAP | pair AP50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Paper Base | 68 | 53.314 | 61.982 | 44.690 | 62.218 | 60.599 | 72.303 | 0.3149 | 0.5225 |
| `0718_03` parent best | 60 | 54.689 | 60.969 | 47.555 | 65.255 | 58.737 | 71.512 | 0.3152 | 0.5302 |
| `0718_04` unique best | 68 | **53.398** | 60.928 | **45.514** | **63.131** | 59.844 | 71.689 | 0.3079 | 0.5212 |
| change vs Base | - | +0.084 | -1.054 | +0.824 | +0.913 | -0.755 | -0.614 | -0.0070 | -0.0013 |
| `0718_04` endpoint | 72 | 52.986 | 60.811 | 45.263 | 62.533 | 59.721 | 71.473 | 0.3051 | 0.5156 |

SASE相对父实验使pedestrian、bike、awning-bike恢复`1.611/0.470/0.217`，证明局部稀疏
证据方向有针对性；但truck、tricycle、van下降`5.913/2.130/2.499`，HOTA和反而下降
`1.332`。相对Base时，pedestrian/bike/awning-bike仍为`-0.865/-2.045/-2.414`，且
truck/tricycle虽仍提升`8.691/3.420`，幅度不足以弥补其他类别。

末50个hard监控有50种完整pattern，平均`image_variant_ratio=0.875`、
`changed_ratio=0.573`、`unique_sets=7.715`、`max_set_repeat=1.265`；route没有坍塌。
`sparse_scale3=0.527`，router/fusion sparse gain约`0.180/0.033`，说明新增分支并非未训练。
det侧相对Base的`DetRe/AssA`分别下降`1.161/1.135`，即召回与关联同时变差。结论是当前
SASE发生能力重分配而非整体增强，不替代Base、0718_03或当前最佳Liquid。

## 24. Pair-Consistent Detail Preservation Liquid

### 24.1 动机与结构

`0718_05`针对与SASE相同的小目标问题给出互补结构验证。SASE在采样前提取局部谱段证据并
复用于LAF；PCDP不修改sampler或SE gate，而是在谱段融合输出处显式保护可能被全局SE抑制的
紧凑空间细节。因此它与`0718_04`不是叠加实验，可以分别判断“选择更合适的谱段”和“保留
已选谱段中的局部细节”哪条路径更有效。

PCDP对每个Conv3D group的通道均值计算3x3中心-周围绝对响应，以每group RMS归一化后得到
有界的frame-specific detail mask。它不在双帧间逐像素对齐，以免目标运动或GMC残差破坏
细节；双帧关系仅在group级使用当前/配对帧的detail mean/std、difference/product，以及当前
采样谱段coverage的intersection/distance。共享两层MLP和group embedding由这些紧凑描述符
预测每帧每group的可信gain，再将`group feature x SE gate x detail mask x gain`跨group求和，
作为原融合输出的细节残差。

最终gain层严格零初始化，因而训练初始前向与`0718_03`完全一致；没有额外loss、类别规则、
目标尺寸阈值或像素级pair对齐。新增参数321个，仅增加3x3平均池化和stem分辨率上的逐元素
运算。该结构的主要判断标准是能否保留ARCR的cls收益，同时改善det HOTA及
pedestrian/bike/awning-bike，而不是只提高总和。

### 24.2 验证、温控与运行状态

29项sampler/stem测试全部通过。新增测试验证初始化输出严格等价、双帧输入接口正确，且
所有PCDP参数均有有限梯度。99 GPU 2/3的20 iter完整DDP smoke在
`find_unused_parameters=False`下通过，iter 20为`0.9699 s/iter`、`11100 MiB/rank`、
`loss=31.2928`、`grad_norm=91.6567`；全部decoder、DN及encoder proposal loss有限，未出现
CUDA、NCCL、OOM、NaN或unused parameter错误。短程数据只能证明开销可接受，不能作为稳定
吞吐量结论。

正式实验编号`0718_05`，于2026-07-18 15:06 CST在99 GPU 2/3从COCO适配权重fresh启动。
配置为full HSMOT、原生`1200x900`、ordered gap-1 pair、BF16、全局batch 8、
`find_unused_parameters=False`和72 epochs。workdir为
`/data4/litianhao/PairMmot/workdir_99/0718_05_paper_liquid_adaptiveanchor_pcdp_r18_coco_full_1200x900_bf16_orderedpairs_fresh`。

考虑99四卡并行导致的机箱热累积，训练使用独立进程组并由温控守护每10秒检查GPU 0--3；
任一卡达到90摄氏度即对0718_05整个进程组发送`SIGSTOP`、写入`THERMAL_PAUSED`和温度快照，
但不向GPU 0/1上的`0718_03`发送信号。监控同时记录`detail_gain_abs/std`，最终仍按唯一最大
`cls_HOTA + det_HOTA`选epoch，并从同一epoch记录AP。

正式epoch 1 iter 100为`0.9613 s/iter`、`11139 MiB/rank`、`loss=32.1263`、
`grad_norm=136.7609`，所有分支有限。iter 100采样时四卡温度为`71/75/81/85`摄氏度，未达到
暂停阈值，温控守护保持armed状态。

训练于15:14:29在epoch 1 iter 450之后由温控守护暂停，触发原因是GPU 3瞬时达到90摄氏度；
紧随其后的快照为GPU 2/3 `84/89`摄氏度。进程收到`SIGSTOP`而非终止，可继续运行，且没有
CUDA、NCCL、OOM或训练数值错误。并发阶段9个日志点平均`0.9465 s/iter`，其中
`data_time=0.0312 s`；同期0718_03的`data_time=0.0329 s`，相对并发前`0.0335 s`没有
增加。当前证据排除常规训练I/O竞争，暂停原因是四卡散热。

确认99当前散热条件不适合两个双卡任务长期并行后，0718_05已按决定主动终止，所有训练子进程
和screen均已退出，GPU 2/3显存释放。该运行仅保留正确性、速度及温控验证，不resume且不进入
性能结果表；模型实现和正式配置保留，待其他资源空闲时可fresh重跑。

## 25. 2026-07-18 21:50 Latest Liquid Snapshot

当前`0718_xx`最新结构实验共5组，其中1组完成、1组训练结束等待最后评测、2组训练中、1组
主动终止。

1. `0718_01 independent-group difference/product`已在197完成72 epochs和18/18 TrackEval。
   按唯一最大`cls_HOTA + det_HOTA`选择epoch 64：`cls_HOTA=55.276`、
   `det_HOTA=61.812`、总和`117.088`。相对Paper Base最佳点分别变化
   `+1.962/-0.170`，总和`+1.792`。epoch 72为`55.002/61.993`，虽两个HOTA都略高于
   Base，但不能违背预注册规则改选；最终报告必须使用epoch 64及其同epoch AP。末期50个
   监控点仅2种完整pattern，7/8个group固定；平均`unique_sets=7.172`、
   `max_set_repeat=1.828`。组内无重复波段，也未发生所有group完全相同，但存在轻度跨group
   重复和明显的输入自适应route坍塌。
2. `0718_02 ARCR`在AutoDL已完成72 epochs和18/18 TrackEval。唯一最佳epoch 64为
   `cls_HOTA=53.357`、`det_HOTA=61.054`；epoch 72为`53.118/61.216`。相对Paper Base
   最佳点分别变化`+0.043/-0.928`，不具备替换价值。末50个监控点有50种完整pattern，
   平均`image_variant_ratio=0.875`、`unique_sets=7.570`、`max_set_repeat=1.398`，说明
   输入route未坍塌；问题已经从route固定转为检测质量下降。
3. `0718_03 adaptive-anchor ARCR`已完成72 epochs和18/18 TrackEval。唯一最佳epoch 60为
   `54.689/60.969`，相对Base为`+1.375/-1.013`；末50个监控点有50种完整pattern，route
   未坍塌，但pedestrian/bike/awning-bike下降且det召回不足，不能替代Base。
4. `0718_04 SASE`已在252完成72 epochs和18/18 TrackEval。唯一最佳epoch 68为
   `53.398/60.928`，相对Base为`+0.084/-1.054`；小目标相对0718_03部分恢复，但truck和
   tricycle收益明显回落，最终低于Base及父实验，不作为最终Liquid。
5. `0718_05 PCDP`在99通过29项测试及双卡smoke后启动，但四卡并行8分钟触发90摄氏度温控，
   在epoch 1 iter 450后主动终止。无checkpoint和评测，不resume、不进入结果表；需换资源
   fresh重跑。

## 26. 0719_01 Pair-Consensus Relaxed-Set Liquid

### 26.1 性能优先的共享route

`0719_01`以高性能Liquid路径为主体，将两帧独立采样改为pair共同生成group。两帧基础任务
logits通过log-mean-exp聚合，保留任一帧的强谱段证据；对称描述符
`[(prev+curr)/2, abs(prev-curr), prev*curr]`经zero-init MLP产生共享残差。每个pair仅执行
一次Gumbel采样，soft概率、hard索引和采样噪声严格共享，但两帧像素与Conv3D特征分别计算。

0717_01的56集合Set-Transport被改为confidence-gated relaxed形式：强度在前12 epochs由0
增加至0.25，任务top-2 margin高时只保留0.05门控下限，低margin时才奖励不同group选择
互补集合。hard group仍独立决策，允许跨group重复；仅组内保持无放回，不增加diversity loss。

共享route后，原coverage-based PairTransport被删除，替换为同编号group直接耦合；同帧
group overlap-aware LAF继续保留，因为它负责识别允许重复后的组间冗余，而非跨帧匹配。

### 26.2 Pair-Aligned Compact Detail Enhancement

为改善pedestrian、bike和awning-bike等小目标，0719_01增加轻量PACDE。每帧每group先计算
3x3中心-周围绝对响应，再减去5x5局部细节密度，抑制大面积重复纹理并保留稀疏紧凑细节；
空间mask保持逐帧独立，不假设移动目标像素对齐。两帧同group的detail mean/std通过
`[pair mean, absolute difference, product]`生成共享group gain。

细节残差乘以`4 * SE * (1-SE)`，只修正SE不确定的group，SE已明确接受或拒绝的group保持
原路径。gain末层严格zero-init，无类别规则、面积阈值、额外loss或跨帧coverage特征。

完整PACDE模型已在AutoDL通过39项测试、单卡两步BF16完整训练与推理smoke，以及正式
1200x900、全局batch 8、`find_unused_parameters=False`的4步双卡DDP训练。所有loss、grad
和推理输出有限，无未使用参数或NCCL错误。正式配置为
`autodl_0719_01_paper_liquid_pairconsensus_relaxedset_full_1200x900_bf16.py`，已于
2026-07-19 02:08 CST在AutoDL GPU 0,1从COCO适配权重fresh启动72-epoch训练。启动验收显存
约`18861/21071 MiB`，环境保留镜像PyTorch `2.8.0+cu128`，真实GMC覆盖train `8297`和
test `5416` pairs。自动收尾按唯一最大`cls_HOTA + det_HOTA`选epoch，并使用`0716_02`
Paper Base做同协议比较；训练完成前不声称性能提升。

### 26.3 PACDE等价加速

初版PACDE在LAF已计算`x_se = groups.mean(dim=1)`后内部重复相同归约，并将基础SE输出与
detail残差分别执行一次大张量乘法和group求和。优化版直接复用`x_se`生成compact mask，
再以`fusion_gate = se_gate + detail_gate`一次完成融合。该变换在实数域严格等价；调试态
`return_sampling=True`仍返回原定义的SE-gated groups，不改变接口。

包含非零detail gain输出和梯度对比在内的40项测试全部通过。AutoDL 100-iter、1200x900、
全局batch 8双卡短测中，iter 100由初版`0.9340 s`降至`0.8931 s`，显存约从
`11212`降至`11153 MiB/rank`。初版正式run在epoch 1停止并隔离保存，不resume；优化版于
2026-07-19 02:22 CST使用同一实验ID和COCO适配权重fresh重启，后续仅统计优化版结果。

## 27. 0719_02 Reliability-Weighted Pair Consensus

0719_01共享router用等权log-mean-exp融合两帧logits，隐含两帧在每个group/slot同等可靠。
0719_02保持pair只采样一次且最终route严格共享，但为每帧、每group、每slot增加同权重共享的
轻量质量头。两帧质量经pair维softmax后，使用加权log-mixture聚合原始route logits；交换
两帧只交换权重，不改变聚合结果，因此时间顺序对称。单侧可见、遮挡或模糊时，模型可以保留
较可靠帧的谱段证据，而无需回到两帧独立route。

质量头末层zero-init，初始权重严格为`0.5/0.5`，初始前向逐元素等价0719_01。原对称共享
残差、relaxed Set-Transport、Pair-Aligned Fusion和PACDE均不变；不增加loss、类别规则或
采样阈值。新增监控记录frame reliability相对0.5的偏移与二帧熵，用于判断该分支是否真正
学习到内容相关质量，而不是静态偏置。

41项backbone/sampler测试通过，覆盖初始等价、pair交换对称、共享route及可靠性头非零梯度。
178单卡worker profile选择8 workers；正式配置为全量HSMOT、1200x900、BF16、batch 8，
与原2卡x4保持相同global batch及学习率调度。`0719_02`于2026-07-19 03:02 CST在178
GPU 0 fresh启动，iter 50为`0.9082 s/iter`、`data_time=0.0758 s`、
`memory=21833 MiB`，运行正常。训练完成前不作性能提升结论，最终仍按唯一最大
`cls_HOTA + det_HOTA`选epoch并从同一epoch记录AP。

## 28. 0719_03 PACDE Strict Ablation

0719_03完整继承0719_01的pair共享route、confidence-gated relaxed Set-Transport、
soft-context、overlap-aware LAF、groupmod和same-index Pair-Aligned Fusion，只将
`pair_aligned_compact_detail_enhancement`设为`None`。它不使用SASE，也不改变采样、学习率、
数据、初始化或评测协议，用于严格区分pair-consensus主干与PACDE局部细节分支的贡献。
实验已完成72 epochs和18/18 TrackEval，唯一最佳为epoch 72。

| experiment | epoch | cls HOTA | det HOTA | HOTA sum | cls MOTA | cls IDF1 | det MOTA | det IDF1 | pair mAP | pair AP50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Paper Base | 68 | 53.314 | 61.982 | 115.296 | 44.690 | 62.218 | 60.599 | 72.303 | 0.3149 | 0.5225 |
| `0719_01` with PACDE | 72 | 53.883 | 61.411 | 115.294 | 46.831 | 63.920 | 59.931 | 72.366 | 0.3161 | 0.5339 |
| `0719_03` no-PACDE | 72 | 54.209 | 61.504 | 115.713 | 47.854 | 63.951 | 60.531 | 72.259 | 0.3165 | 0.5314 |
| change vs `0719_01` | - | +0.326 | +0.093 | +0.419 | +1.023 | +0.031 | +0.600 | -0.107 | +0.0004 | -0.0025 |
| change vs Base | - | +0.895 | -0.478 | +0.417 | +3.164 | +1.733 | -0.068 | -0.044 | +0.0016 | +0.0089 |

PACDE移除后det HOTA仅恢复`0.093`，而pedestrian、awning-bike、bike相对0719_01还下降
`0.553/0.833/0.155`，说明PACDE确实对小目标有轻度保护作用，并非det缺口的主因。相对
Base，no-PACDE的FN增加`1684`、FP减少`1370`、IDSW减少`182`，DetRe下降`1.052`而
DetPr提高`0.076`：问题是pair-consensus主干产生了偏保守、低召回的检测，而不是关联错误。

末50个hard监控只有5种完整pattern，`image_variant_ratio=0.420`、`unique_sets=6.065`、
`max_set_repeat=2.000`；没有所有group完全相同，但pair共享route已明显趋同。0719_03不能
替换Base或0718_01，且后续不应继续围绕PACDE做修补；若保留pair-consensus路线，需要直接
解决共享route与融合造成的召回损失。

## 29. 0719_04 Wide-LAF + GroupMod Paper Replay

`0719_04`用于检验历史half-data阶段表现最强的`0711_01`核心能否迁移到当前Paper协议。模型
仅使用独立8-group Liquid sampler、Wide LAF和GroupMod，不包含pair router、
PairTransport、Set-Transport或跨group集合唯一约束。197于2026-07-20 16:31 CST完成
72 epochs和18/18 TrackEval；按唯一最大`cls_HOTA + det_HOTA`选择epoch 72，同epoch
`pair_mAP/pair_AP50=0.3134/0.5277`。

| experiment | epoch | cls HOTA | det HOTA | HOTA sum | cls MOTA | cls IDF1 | det MOTA | det IDF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Paper Base | 68 | 53.314 | 61.982 | 115.296 | 44.690 | 62.218 | 60.599 | 72.303 |
| `0719_04` | 72 | 53.932 | 60.908 | 114.840 | 45.958 | 63.723 | 59.028 | 71.362 |
| change vs Base | - | +0.618 | -1.074 | -0.456 | +1.268 | +1.505 | -1.571 | -0.941 |

逐类cls HOTA中truck和tricycle分别提高`10.007/3.384`，但pedestrian、bike、
awning-bike、van和car分别下降`2.182/2.056/1.523/1.513/0.666`。检测侧FN增加`2893`，
`DetRe/DetA/AssA`下降`1.428/1.143/0.812`；IDSW反而减少`163`，所以det掉点主要来自
高频目标召回不足和碎片增加，而非pair身份错配激增。

末期`image_variant_ratio`通常为`0.875`，`unique_sets`约`6.88--7.62`，完整pattern仍随
输入变化，没有灾难性route坍塌。该实验说明Wide LAF + GroupMod在full/native-resolution
协议下仍偏向提升长尾类别判别，但不足以保护pedestrian/bike等高频小目标检测。相较历史
half实验对旧Base的`+1.961/+0.301` cls/det收益，本次相对更强Paper Base变为
`+0.618/-1.074`，历史优势没有迁移。最终Liquid应保留`0718_01`所证明有效的显式pair
difference/product耦合，而不应退回纯Wide-LAF + GroupMod路径。

## 30. 0720 Fusion Quality Conservation

本轮以当前HOTA和最高的`0718_01`为唯一父结构，保留独立group、difference/product pair
router和PairTransport，不使用pair共享route、跨group唯一化、额外loss或小目标附加分支。
动机是`0718_01`的det HOTA仅低于Base `0.170`，而近期消融反复指向检测召回不足；最小风险
的改进位置是LAF增量进入SE gate的边界，而不是重新设计sampler。

新增`FusionQualityConservation`不含可训练参数。它根据原SE gate的局部导数，把LAF增量
投影到不改变融合质量统计量的一阶切空间，同时保留不同group之间的相对重分配。三组互补
版本为：`0720_01 gate_mass`守恒总门控量；`0720_02 response_mass`按Conv3D响应加权守恒；
`0720_03 dual_moment`同时守恒总量和响应相关一阶矩。

252和197的双卡真实数据4-iter smoke均通过，显存约`10.78 GB/rank`，loss、grad及完整DDP
路径有限。正式`0720_01/0720_02`分别于2026-07-20 18:19 CST在252 GPU 0,1和197 GPU 4,5
fresh启动，首个正式监控点分别为`1.165/0.849 s/iter`。`0720_03`已在99可靠排到0719_06
之后；队列通过flock防重复，连续检查GPU低显存，并在正式启动前自动执行4-iter smoke。

三个实验都使用Paper Base的COCO适配初始化、full HSMOT、1200x900、ordered gap-1 pairs、
BF16、global batch 8和72-epoch/18-point TrackEval协议。最终成功必须同时满足
`cls_HOTA>53.314`与`det_HOTA>61.982`；不能用HOTA和掩盖任一主指标下降。

## 31. 2026-07-21 Fusion-Conservation Results

本节于2026-07-21 14:41 CST重新从共享实验目录的原始TrackEval结果和同epoch AP标量复核并
恢复；完成实验使用全部18个评测点，运行中实验只使用已经落盘的阶段评测点。

`0720_02 response-mass`已完成72 epochs和18/18 TrackEval，唯一最佳点与末轮均为epoch 72。
其cls/det HOTA为`54.463/61.437`，相对Paper Base为`+1.149/-0.545`；同epoch pair
mAP/AP50为`0.3154/0.5328`。虽然HOTA和提高`0.604`，det HOTA仍下降，因此不满足Liquid
必须双指标超过Base的目标。

检测分解为`DetA -0.575`、`AssA -0.392`、FN `+1055`、FP `-375`、IDSW `-62`。模型输出
更保守，det退化主要是召回不足而不是身份切换。逐类cls HOTA中truck/tricycle提高
`11.434/5.600`，但pedestrian/bike/van下降`1.338/3.265/2.063`；response-mass投影没有
阻止Liquid继续把能力转移到长尾类别。末期route通常固定为
`431/026/120/132/431/436/521/610`，`unique_sets`约7、`max_set_repeat`约2，且多个监控点
`image_variant_ratio=0`。不存在所有group相同的坍塌，但输入自适应route仍明显固化。

截至2026-07-22 01:13 CST，`0720_01/0720_03`均已完成；运行中实验仍只报告已完成评测点中的
阶段最佳，不作为最终选点：

| experiment | progress | epoch | cls HOTA | det HOTA | HOTA sum | pair mAP | pair AP50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Paper Base | completed | 68 | 53.314 | 61.982 | 115.296 | 0.3149 | 0.5225 |
| `0720_01` gate-mass | completed, 18/18 | 72 | 54.103 | 61.581 | 115.684 | 0.3159 | 0.5309 |
| `0720_02` response-mass | completed, 18/18 | 72 | 54.463 | 61.437 | 115.900 | 0.3154 | 0.5328 |
| `0720_03` dual-moment accuracy-fix | completed, 18/18 | 72 | 54.650 | 61.547 | 116.197 | 0.3231 | 0.5402 |
| `0721_01` response-mass 1x8 accuracy-fix | completed, 18/18 | 72 | 54.006 | 60.934 | 114.940 | 0.3148 | 0.5297 |
| `0721_02` independent difference/product accuracy-fix | completed, 18/18 | 72 | 54.327 | 61.659 | 115.986 | 0.3161 | 0.5340 |

gate-mass的唯一最佳点为epoch 72，相对Base cls/det HOTA变化`+0.789/-0.401`，仍未双提升。
dual-moment修复版的唯一最佳点同样为epoch 72，相对旧Base数值变化`+1.336/-0.435`；
约束更强但det仍落后。178修复版在epoch 24
相对同服务器`0719_05` Base同epoch为`+0.593/+0.942`，阶段走势有价值，但主Base与该run的
代码版本不同，不能作为严格提升声明。

`0720_01`相对Base的HOTA和提高`0.388`，但det `DetA/DetRe`下降`0.697/0.666`，FN/FP分别
增加`915/602`；`AssA`反而提高`0.127`且IDSW减少`80`，证明主要问题是检测覆盖而非关联。
逐类cls HOTA中truck/tricycle提高`6.594/4.900`，但van/pedestrian/bike/bus分别下降
`1.987/1.443/1.244/0.906`。相对直接父配置`0718_01`的唯一最佳点，cls/det HOTA下降
`1.173/0.231`；同为epoch 72时也下降`0.899/0.412`，所以gate-mass不是有效增量。

末50个监控点只有3种完整route pattern，主pattern出现31次；平均`unique_sets=6.750`、
`max_set_repeat=2.000`、`image_variant_ratio=0.418`。不存在所有group相同，但route仍明显
固化，质量守恒没有带来route多样性改善。

`0720_03`相对旧Base的HOTA和提高`0.901`，同一epoch的pair mAP/AP50提高`0.0082/0.0177`，但
det HOTA仍下降`0.435`。det `DetA/AssA/DetRe`分别下降`0.355/0.429/0.511`，FN增加
`777`、IDSW增加`62`，属于召回与关联共同退化。逐类cls HOTA中truck/tricycle/bus提高
`9.153/4.644/1.525`，bike/pedestrian/car/van/awning-bike分别下降
`2.843/0.985/0.360/0.301/0.151`，仍是向长尾车辆类别重分配能力。

与`0718_01`各自最佳点相比，`0720_03`的cls/det HOTA下降`0.626/0.265`；固定epoch 72
也下降`0.352/0.446`。由于accuracy-fix使训练协议不同，这不是严格消融，但结果不支持
dual-moment优于父结构。末50个监控点仅3种完整route，主pattern占38/50，平均
`unique_sets=7.210`、`max_set_repeat=1.778`、`image_variant_ratio=0.310`；route仍明显固化。

训练准确性修复包括成对几何变换/GMC组合及PairDN negative目标修正。`0720_01/0720_02`在
修复前启动，`0720_03/0721_01`在修复后fresh启动；因此本节只对各自协议内趋势作判断。
当前最明确的双提升来自非Liquid的`0719_06 long-tail reweight`，但它同样属于修复前协议。
下一轮正式确认应先在修复代码上重跑Base和Base + long-tail，再以胜者作为Liquid叠加基线。

截至2026-07-22 12:20 CST，`0721_01/0721_02`均已完成。`0721_01`相对同服务器1x8 Base
`0719_05`的cls/det HOTA变化为`+1.589/-0.331`；`0721_02`相对Paper Base变化为
`+1.013/-0.323`。两者都再次表现为cls增强、det轻度下降，未达到Liquid双提升目标。
`0721_02`仍使用会产生近零位移的旧Negative-DN噪声，并将正、负DN块作为彼此隔离的
attention group。运行中的`0722_01`同时修正外环采样和group mask，因此不能作为单独的
box-noise消融。

## 32. 0721_03 BSR-Liquid Block Route Descriptor

`0721_03`代号为BSR-Liquid（Blockwise Spectral Recurrent Liquid），继续使用accuracy-fixed
`0718_01`的独立group、difference/product pair router和PairTransport，但完全替换原sampler
的全图route descriptor，不保留global statistics或global hidden残差，也不增加额外loss。

对于`1200x900`输入，BSR将每个谱段划分为`12x16`个`75x75`区块，计算每块的
`mean/std/max`，得到`[B,8,192,3]`。共享`3->32`投影后形成`[B,8,192,32]`，将192个
区块并入batch维，并行执行原有8步spectral recurrent，得到`[B,192,32]`局部光谱状态。
recurrent始终只沿8个物理谱段运行，不沿192个区块串行。最后对block hidden计算
`mean/std/max`并拼接为`[B,96]`，经`LayerNorm + Linear(96,32) + tanh`形成唯一图像级
route hidden，再由原head预测`[B,8,3,8]`，现有pair router在该聚合hidden之后工作。

该结构将每个区块视为局部八谱段序列，使常见局部模式、场景异质性和稀有强响应分别通过
block hidden的mean/std/max进入route，同时保持整图只生成一套路由。新增正式配置为
`o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_liquid_bsr_diffproduct_accuracyfix_coco_full_1200x900_bf16_99.py`。
代码默认关闭BSR，因此所有既有配置继续使用原global descriptor。

新增测试验证`[B,8,192,3] -> [B,192,32] -> [B,32]`形状、所有sampler和pair-router参数
梯度有限，并证明全局mean/max相同但空间分布不同的输入产生不同route hidden。完整backbone
测试44项通过，配置解析确认为full HSMOT、`1200x900`、BF16、global batch 8和
pair difference/product路径。当前仅完成代码和配置，尚未启动训练或GPU smoke。

## 33. 0721_04--0721_06 BSAC, DSE and CSPR

为避免只依赖BSR，本轮从Conv3D输入适配、稀疏检测证据和更强route语义三个互补位置实现
三种默认关闭的Liquid模块。三者均以accuracy-fixed `0718_01`为父配置，保留独立group、
difference/product pair router、PairTransport、full HSMOT、`1200x900`、BF16和global
batch 8，不增加额外loss。

`0721_04 BSAC`（Band-Slot Adaptive Calibration）在谱段采样后、共享Conv3D前，根据
`[slot, physical band]`的`3x8`参数表计算每个采样槽的条件尺度：soft采样使用概率期望，
hard采样退化为查表。尺度为`1+tanh(evidence)`，24个参数全零初始化，因此初始输出与父
配置严格一致。该结构解决任意物理谱段进入COCO三通道Conv3D固定kernel slot时的条件分布
失配，新增参数使用`1.0x base LR=1e-4`。

`0721_05 DSE`（Dispersion-aware Spectral Evidence）将SE/LAF原有的channel mean扩展为
每个group的channel mean和RMS，再用grouped `1x1 Conv`从`2->1`混合。16个参数初始化为
`[1,0]`，初始严格恢复mean路径；RMS用于保留会被有符号通道均值抵消的局部强响应，目标是
减少pedestrian、bike及密集序列FN。实现使用channel norm reduction，避免显式保存完整
`x.square()`中间张量；新增参数同样使用`1e-4`。

`0721_06 CSPR`（Coarse Spectral Preview Router）将输入下采样到`24x32`，沿谱段轴循环
补边后使用正式stem的共享Conv3D权重生成8个低分辨率预览组，再从每组Conv3D响应的
mean/RMS/absolute peak构建8步spectral recurrent route。预览输入和共享权重均停止梯度，
所以不增加参数、不执行预览Conv3D反向，也不会通过第二条路径扰动正式stem；检测loss仍可
正常训练descriptor projection、recurrent、route head和pair router。CSPR与BSR互斥，
当前作为BSR失败后的替代候选，不进入正式训练队列。

本地及252端完整backbone单元测试均为48/48通过，覆盖identity等价、所有新增参数有限非零
梯度、CSPR共享权重无预览梯度、stem集成和`find_unused_parameters=False`所需的全参数
可达性。三份正式配置均已解析并构建完整模型：BSAC/DSE/CSPR分别新增`24/16/0`个参数。

按2026-07-21 17:39日志ETA，`0721_04 BSAC`排在252的`0720_01`之后，`0721_05 DSE`排在
99的`0720_03`之后。252队列在前序实验完成后正确识别GPU空闲，并于19:27完成4-iteration
双卡smoke且生成`epoch_1.pth`；随后正式launcher因`set -u`下Conda脚本读取未定义`PS1`
而退出，并非模型、数据、显存或DDP失败。launcher和队列均已在Conda激活期间临时关闭
`nounset`。`0721_04`于2026-07-22 00:52在252 GPU 0,1 fresh恢复，模型构建、预训练权重
加载和首轮计算均已开始；screen为`train_0721_04`。epoch 1 iter 50/100分别为
`1.1234/1.0864 s/iter`，日志显存`11107 MB/rank`，总loss、PairDN loss、encoder proposal
loss和grad norm均有限，未发现NaN、OOM、unused parameter或DDP错误。

## 34. 0718_01--0722_01 PairDN Audit

本节于2026-07-23从三个正式workdir、运行时config、源码修改时间、逐epoch TrackEval、
`scalars.json`及导出的pair detections交叉检查。`0722_01`尚未完成，相关数字只作为epoch 48
阶段诊断，不进入正式结果选点。

三次运行的DN语义并非单变量变化：

| run | DN目标 | DN噪声 | DN self-attention | 同期非DN变化 |
| --- | --- | --- | --- | --- |
| `0718_01` | 两个block都按正类和GT框监督；所谓negative实际是远距离positive DN；padding不监督 | `uniform(-1,1) * uniform(1,2)` | 每个正、负block分别隔离 | 旧训练几何/GMC路径 |
| `0721_02` | positive为真实类别和框；negative为background且无box监督；padding也作为background监督 | 仍使用可趋近零的旧乘积噪声 | 正、负block仍彼此隔离 | 同时修复共享旋转/GMC；推理公共pair label改由证据更强的一侧决定 |
| `0722_01` | 与`0721_02`相同 | 每个rbox参数使用独立符号乘`uniform(1,2)` | 同一contrastive group内正负block可互相注意 | 仅有dtype兼容性修改，无预期数值影响 |

因此，`0718_01 -> 0721_02`只能评价一组训练正确性修复的综合效果；`0721_02 -> 0722_01`
也同时改变了噪声和attention mask，不能把差值全部归因于外环采样。

关键训练点如下。`0722_01`在epoch 4领先，但epoch 8--40明显落后，epoch 44后重新接近
`0721_02`；其最终趋势仍需等待剩余6个TrackEval点。

| epoch | `0718_01` cls HOTA | `0718_01` det HOTA | `0721_02` cls HOTA | `0721_02` det HOTA | `0722_01` cls HOTA | `0722_01` det HOTA |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 36.529 | 46.043 | 38.385 | 46.614 | 39.020 | 47.754 |
| 8 | 44.075 | 50.859 | 45.302 | 51.271 | 44.443 | 51.192 |
| 24 | 51.435 | 58.544 | 51.446 | 58.161 | 49.492 | 57.257 |
| 40 | 53.416 | 60.320 | 53.261 | 60.093 | 50.381 | 59.199 |
| 48 | 54.046 | 60.983 | 53.618 | 60.755 | 53.775 | 60.531 |
| 72 | 55.002 | 61.993 | 54.327 | 61.659 | - | - |

固定epoch 72比较，`0721_02`相对`0718_01`的cls HOTA下降`0.675`，det HOTA下降`0.334`。
cls侧`DetA`下降`1.195`而`AssA`提高`0.150`；det侧`DetA/AssA`分别下降`0.326/0.360`。
TP减少`674`、FN增加`674`、FP增加`422`，说明最终退化主要是检测和分类覆盖，而不是明显的
ID switch恶化。逐类cls HOTA中tricycle和van提高`0.657/0.350`，car、pedestrian、bike、
truck、awning-bike、bus分别下降`0.172/0.539/0.845/0.958/1.027/2.869`。

DN loss进一步表明“更低的训练loss”不等于“更有效的DN”。epoch 41--48平均
`dn_loss_cls`从`0721_02`的`0.15776`降至`0722_01`的`0.14363`，下降约`9.0%`；同期两侧
DN bbox和IoU loss几乎不变。导出的epoch 48 pair detections中，`pair_score>0.2`的数量从
`201837`降为`197582`，减少`4255`，与det召回和det HOTA下降一致。严格外环使negative分类
更容易，但目前没有转化为更好的匹配查询。

当前PairDN还有两个数据相关问题。第一，HSMOT训练集相邻帧track union中位数为27，raw
pair均值为35.60；`num_dn_queries=100`的动态分组实际产生每图中位数180、均值172.49个DN
slots。按每rank batch 4随机组合估计，约42.2%的DN slots是为了对齐batch最大目标数而产生的
padding；`0721_02/0722_01`把这些padding全部作为background监督，使background slots平均约为
真实positive slots的2.62倍。该行为来自DINO式padding，但对目标密度变化很大的HSMOT pair
可能形成过强背景压力。第二，DINO在`xyxy`角点上构造外环，当前实现却直接要求
`(cx,cy,w,h,theta)`五个参数全部进入外环，会同时改变中心、尺度和角度，强度比原始DINO更大。
轴对齐proxy Monte Carlo在`box_noise_scale=0.4`下显示，旧乘积噪声平均IoU约`0.544`，约
62.5%样本高于0.5；当前五维全外环平均IoU约`0.319`，高于0.5的比例接近0。前者包含大量
矛盾的近GT背景，后者又可能过于容易。

### 34.1 PairDN query与来源GT的旋转IoU统计

由于训练日志和checkpoint不保存每次随机生成的DN reference，本统计按三个实验启动时的
generator实现复现采样。使用全部`292179`个HSMOT train标注框，模拟正式pipeline的
`PairSharedRandomFlip(prob=0.5)`和`PairSharedRandomRotate(prob=0.5, angle_range=180)`，过滤旋转后
中心出界的框，并以固定随机种子对每个保留框采样一次DN reference；共得到`270822`个有效侧
样本。IoU使用`mmcv.ops.box_iou_rotated`计算。这里的`old negative`是`0718_01/0721_02`使用的
可趋近零乘积噪声，`strict negative`是`0722_01`使用的五维signed `[1,2)`外环噪声。

| 几何采样 | 对应实验与监督语义 | mean IoU | median | P10 | P90 | IoU<0.1 | 0.1--0.3 | 0.3--0.5 | 0.5--0.7 | IoU>=0.7 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| positive | 三个实验相同；均按正样本监督 | 0.680 | 0.718 | 0.376 | 0.860 | 0.013% | 4.686% | 10.968% | 28.942% | 55.391% |
| old negative | `0718_01`按正样本监督；`0721_02`按background监督 | 0.480 | 0.487 | 0.294 | 0.652 | 0.256% | 10.507% | 42.880% | 42.010% | 4.347% |
| strict negative | `0722_01`按background监督 | 0.286 | 0.295 | 0.160 | 0.398 | 3.446% | 48.749% | 47.752% | 0.053% | 0.000% |

因此单侧`IoU>=0.5`的比例分别为positive `84.333%`、old negative `46.357%`和strict negative
`0.053%`。`0721_02`约有一半所谓negative仍与来源GT达到常用正匹配IoU范围，却被强制监督为
background；`0722_01`几乎完全消除了这一冲突，但也把negative变成了明显更容易的背景样本。
`0718_01`则没有真正的DN negative：每个group中positive block和old-negative block均指向同一
GT并按正类/GT框监督，相当于每个GT有一个普通positive和一个更宽、更难的positive。

PairDN的prev/curr噪声是两次独立调用，因此又在`284205`个原始both-visible pair上统计两侧联合
情况；模拟增强后保留`263130`个两侧仍有效的pair：

| 几何采样 | 两侧IoU均值 | `min(prev,curr)`均值 | 两侧均>=0.5 | 仅一侧>=0.5 | 两侧均<0.5 | 两侧均>=0.7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| positive | 0.679 | 0.634 | 81.648% | 4.957% | 13.395% | 41.017% |
| old negative | 0.480 | 0.418 | 25.669% | 41.340% | 32.991% | 0.270% |
| strict negative | 0.286 | 0.241 | 0.002% | 0.076% | 99.922% | 0.000% |

该表说明`0721_02`的old negative中，`25.669%`在两侧都仍像正框，另有`41.340%`只在一侧仍像
正框；把这两类统一标成双侧background会制造大量与PairDN可见性语义冲突的监督。严格外环则
使`99.922%`的pair两侧IoU都低于0.5，基本不再形成hard boundary附近的pair样本。

统计过程中还发现一个独立于三次实验差异的共同问题：generator的`_to_rbox()`直接返回
`qbox2rbox`结果，没有执行head构造loss target时使用的
`regularize_boxes(width_longer=True, start_angle=0)`。正式增强后的rbox角度可能为负或超出
`[0, pi)`，随后`_noisy_refs()`却对五个归一化参数统一执行`clamp(1e-4, 1-1e-4)`，从而把一部分
角度截到边界。若先按head口径正则化角度、其余采样完全不变，纯噪声统计为：positive mean IoU
`0.769`、`IoU>=0.5`为`98.137%`；old negative为`0.498/52.827%`；strict negative为
`0.271/0.077%`。这说明当前正样本在正确角度表示下本来非常容易，同时实际训练还混入了不应
存在的角度截断误差。后续DN改动应先统一generator与target的角度表示，再讨论正样本加难，
否则会把表示错误误认为有效的hard positive。

下一步DN探索按优先级建议如下，均不增加推理计算或额外loss：

1. **Padding-masked PairDN**：保留正确positive/negative目标和当前group mask，只让真实
   positive及真实negative slot参与分类，padding的`label_weights`恢复为0。这是最小、最有
   可能恢复det召回的改动，也能直接验证过量背景监督假设。
2. **Pair-coherent DN**：对两帧共享同一组相对几何扰动，而不是分别调用独立噪声；对
   both-visible目标保留真实运动，只模拟共同定位误差，single-visible缺失侧继续使用neutral
   reference、background分类和零box权重。该改动更符合pair query语义，主要目标是改善AssA。
3. **Geometry-consistent hard negative**：不要让五个rbox参数同时进入外环。可让一个局部中心
   轴进入`[1,2)`、其余尺度和角度保持inner jitter，或向量化采样少量候选并选择IoU低于0.5
   的最高IoU候选。这样保留有梯度价值的hard negative，同时避免旧实现的近GT伪负样本。
4. **仅衰减negative分类**：若前述结构修复后仍出现“前8 epoch领先、后期落后”，保持
   positive DN及box denoising权重`0.2`不变，只在中后期降低negative分类权重。直接衰减整个
   `dn_loss_weight`会同时移除有价值的正样本定位监督，不作为首选。

推荐先完成`0722_01`，然后以其正确目标和group mask为基础做“padding mask only”严格消融；
若det召回恢复，再叠加pair-coherent noise。外环难度和negative权重不应同时修改，否则仍无法
确定收益来源。
