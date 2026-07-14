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
