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
已验证到 epoch 1 iter 200：`0.9636 s/iter`、日志显存 `8444 MiB`，loss/grad finite，
初始 pattern 为 `701 / 012 / 123 / 234 / 345 / 456 / 567 / 670`，没有 unused-parameter、
dtype、GMC 或 DDP 错误。当前 ETA 约 20 小时 29 分。

## 16. 论文严格 Base + Liquid 消融

`0716_03` 使用论文统一协议重新验证最终 Liquid：COCO-only初始化、原生1200x900输入、
全量数据的8297个唯一有序 `t-1 -> t` pair、BF16、`find_unused_parameters=False`和完整
72-epoch评测。相对同步运行的`0716_02`，唯一模型变化是8-group最终Liquid，包括独立
sampler、wide overlap-aware LAF、group modulation、pair sampler router和pair transport。

本机GPU 2故障导致首次运行在epoch 1作废且不resume。代码同步后，正式fresh run于
2026-07-16 17:15 CST在197的GPU 0/3启动，workdir为
`/data4/litianhao/PairMmot/workdir_197/0716_03_paper_base_plus_liquid_r18_coco_full_1200x900_bf16_orderedpairs_fresh`。
epoch 1 iter 50为`1.0818 s/iter`、日志显存`10692 MB`，loss和grad有限，无CUDA、NCCL、
NaN、OOM、unused parameter或DDP错误。该实验完成后将与`0716_02`按唯一最佳
`cls_HOTA + det_HOTA` epoch进行严格比较。
