# PairMOT decoder 实验状态（2026-07-30）

更新时间：2026-07-30 16:32 CST

## 当前研究原则

- 论文主线保持为：`0719_05 Base 52.417/61.265` → `0723_01 Base+Liquid 53.955/62.032` → `0727_01 +Encoder 54.437/62.393`。
- decoder 目标是同时超过 `0727_01` 的 cls HOTA 54.437 和 det HOTA 62.393。
- 不再进行 class-specific reweight、long-tail reweight 或大规模 residual-scale 扫描；优先验证有明确时序归纳偏置的模型结构。
- AutoDL 实例均处于关机状态，不纳入当前调度。

## 运行中

| 服务器 | 实验 | 状态 | 结构与判定方式 |
| --- | --- | --- | --- |
| 99 GPU 0,1 | `0730_05 ... decoder_commonmotion ... fresh` | `RUNNING`，保留至 epoch 8 复核 | epoch 4 cls/det HOTA `36.424/41.733`，相对 `0727_01` 同点为 `+0.215/+2.980`；cls DetA/AssA `26.059/54.005`，det DetA/AssA `30.636/58.011`。pair mAP/AP50 `0.1535/0.2962`，检测精度基本未增，但 det AssA 同点提高 `+10.545`，呈现明确的运动关联信号。该信号与 shared-evidence 的检测增益互补，因此不在 epoch 4 停止，但必须在 epoch 8 排除早期 AssA 虚高。 |
| 178 GPU 0 | `0730_06 ... decoder_sharedevidence ... fresh` | `RUNNING`，保留至 epoch 8 复核 | epoch 4 cls/det HOTA `39.547/42.546`，相对 `0727_01` 同点为 `+3.338/+3.793`；cls DetA/AssA `31.490/53.776`，det DetA/AssA `37.899/49.203`。pair mAP/AP50 `0.1791/0.3655`，同点提高 `+0.0219/+0.0694`，both-independent AP50 提高约 `+0.0720`。分类、检测和 AP 同时改善，是当前最强的 decoder 结构信号，至少训练至 epoch 8 复核。 |
| 252 GPU 0,1 | `0730_07 ... decoder_commonmotion_sharedevidence ... fresh` | `PREPARING` | 组合两个已独立验证且在 epoch 4 呈互补信号的结构：shared-evidence 改善共享 query 与检测，common-motion 提供反对称两帧运动修正。两支均为零初始化，父配置仍为 `0727_01`；使用 2×batch4，与 99/252 上的同拓扑父实验可严格对比。 |

## 已完成或释放

| 服务器 | 实验 | 状态 | 说明 |
| --- | --- | --- | --- |
| 252 GPU 0,1 | `0730_02 decoder_boxonly_gradisolated_reg0p25 ... fresh` | `STOPPED`；2026-07-30 16:17 CST 主动停止，保留 epoch 4 checkpoint、检测结果和完整首轮 TrackEval | epoch 4 为 cls/det HOTA `35.883/42.045`，相对 `0727_01` 同点为 `-0.326/+3.292`；cls DetA/AssA 为 `27.176/51.060`，det DetA/AssA 为 `32.601/55.041`。pair AP50 `0.3245`（`+0.0284`），但 pair mAP `0.1545`（`-0.0027`）；det 表面增益主要来自早期 AssA 激增 `+7.575`，与旧 dual-output 在 epoch 4 det 虚高、epoch 12 落后的轨迹一致，因此不展开 residual-scale 扫描或长跑。停止后 GPU 0/1 连续三次检查均为 1 MiB、0% 利用率。 |
| 178 GPU 0 | `0729_05 ... pairdn_easyhardpositive ... fresh` | `COMPLETED`，训练与 epoch 72 检测验证完成；当前保留 15/18 个 TrackEval 点 | 该轨迹在 epoch 12 后曾恢复训练，异步 TrackEval 计数器重置并覆盖了 epoch 4/8/12 的三个目录；现存 15 点的暂定最佳为 epoch 68：cls/det HOTA `55.319/61.515`、DetA `46.002/53.761`、AssA `68.268/72.846`，同点 pair mAP/AP50 `0.3264/0.5494`。epoch 72 的 `both_independent_AP50=0.5781`。结论仍是 positive-only DN 能恢复较高 cls，但 det/AssA 不足以替代论文主线；不得把 15 点汇总误写为完整 18/18。 |
| 252 GPU 0,1 | `0730_01 legacy 0728_03 resume` | `STOPPED`，保留至 epoch 56 checkpoint | 旧 decoder 初始化轨迹，仅作历史诊断，不与修复后正式实验合并解释。 |
| AutoDL | 所有实例 | `OFF` | 用户确认均已关机；没有后台训练。 |

## 代码一致性

- 178、252、99 以及 197 的干净运维副本在配置 `0730_07` 前均位于提交 `2027f8d`；其中 `eb9c440` 为公共运动 decoder，`25fe052` 为共享证据 decoder，`a8d8ded` 增加两者组合验证，`2027f8d` 记录了 box-only 方向的首轮否定结论。本次在该共同历史上追加 `0730_07` 配置、启动脚本和首轮结构结论。
- 99 原有实验提交 `c104193` 在共同历史中被完整保留；未跟踪目录未覆盖或删除。
- 两项新结构及其组合共通过 36 项 decoder 单元测试；组合在零初始化时精确等于父模型，第一步反向中两组 adapter 均获得非零梯度。组合已分配为 `0730_07`，计划使用 252 GPU 0,1；各单项结构均在配置展开、路径检查和真数据 smoke 后启动。
- 197 后续运维使用既有且干净的 `/data/users/litianhao/PairMOT_sync_3cb888d`；该副本已在 197 环境通过 36/36 decoder 测试，并随本次共同提交继续同步。历史目录 `/data/users/litianhao/PairMOT` 保持在原状态（39 个已修改、238 个未跟踪项），未覆盖、未清理，也不作为新实验代码源。

## 下一步

1. 保留 `0730_05` 与 `0730_06` 到 epoch 8；重点检查前者 AssA 增益是否延续、后者 cls/det HOTA 与 AP 的全面增益是否保持。
2. 在 252 GPU 0,1 启动 `0730_07` 组合实验，以 epoch 4 为首个正式判断点；若组合未继承 shared-evidence 的检测提升或 common-motion 的关联提升，则及时停止。
3. 论文主线暂不改变；只有 decoder 候选在中后期同时超过 `0727_01` 的 cls/det HOTA，才进入论文递进表。
