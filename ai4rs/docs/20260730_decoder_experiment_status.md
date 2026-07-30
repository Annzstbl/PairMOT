# PairMOT decoder 实验状态（2026-07-30）

更新时间：2026-07-30 15:43 CST

## 当前研究原则

- 论文主线保持为：`0719_05 Base 52.417/61.265` → `0723_01 Base+Liquid 53.955/62.032` → `0727_01 +Encoder 54.437/62.393`。
- decoder 目标是同时超过 `0727_01` 的 cls HOTA 54.437 和 det HOTA 62.393。
- 不再进行 class-specific reweight、long-tail reweight 或大规模 residual-scale 扫描；优先验证有明确时序归纳偏置的模型结构。
- AutoDL 实例均处于关机状态，不纳入当前调度。

## 运行中

| 服务器 | 实验 | 状态 | 结构与判定方式 |
| --- | --- | --- | --- |
| 99 GPU 0,1 | `0730_05 ... decoder_commonmotion ... fresh` | `RUNNING`；epoch 1 iter 1000，约 0.996 s/iter，loss 11.9415、grad_norm 34.3289 均有限，显存约 19.2 GB/rank | 以 `0727_01` 为严格父配置；共享 query/分类路径不变，用两帧 cross-attention 差分和周期角度 reference 位移预测零初始化的 5D 反对称运动修正。4-iter 真数据 DDP smoke 已通过，三层新权重均从 0 更新到约 `3.98e-4`。首个正式判断点为 epoch 4。 |
| 252 GPU 0,1 | `0730_02 decoder_boxonly_gradisolated_reg0p25 ... fresh` | `RUNNING`；epoch 2 iter 1000，约 1.045 s/iter，loss 11.9464、grad_norm 32.6970 均有限 | 仅作为短期结构基线，不展开 scale sweep。首个 TrackEval 点为 epoch 4；若 cls/det 同点轨迹没有优于 `0727_01` 的信号，则停止并释放资源。 |
| 178 GPU 0 | `0730_06 ... decoder_sharedevidence ... fresh` | `RUNNING`；epoch 1 iter 550，约 0.868 s/iter，loss 17.2908、grad_norm 126.6569 均有限，MMEngine 峰值约 21.7 GB | 以 `0727_01` 为严格父配置；将两帧 cross-attention 的相对不一致度作为交换不变证据，零初始化地注入共享 query，直接服务 cls 与两帧框。4-iter 真数据 smoke 已通过，三层新权重均从 0 更新到约 `4.00e-4`。首个正式判断点为 epoch 4。 |

## 已完成或释放

| 服务器 | 实验 | 状态 | 说明 |
| --- | --- | --- | --- |
| 178 GPU 0 | `0729_05 ... pairdn_easyhardpositive ... fresh` | `COMPLETED`，训练与 epoch 72 检测验证完成；当前保留 15/18 个 TrackEval 点 | 该轨迹在 epoch 12 后曾恢复训练，异步 TrackEval 计数器重置并覆盖了 epoch 4/8/12 的三个目录；现存 15 点的暂定最佳为 epoch 68：cls/det HOTA `55.319/61.515`、DetA `46.002/53.761`、AssA `68.268/72.846`，同点 pair mAP/AP50 `0.3264/0.5494`。epoch 72 的 `both_independent_AP50=0.5781`。结论仍是 positive-only DN 能恢复较高 cls，但 det/AssA 不足以替代论文主线；不得把 15 点汇总误写为完整 18/18。 |
| 252 GPU 0,1 | `0730_01 legacy 0728_03 resume` | `STOPPED`，保留至 epoch 56 checkpoint | 旧 decoder 初始化轨迹，仅作历史诊断，不与修复后正式实验合并解释。 |
| AutoDL | 所有实例 | `OFF` | 用户确认均已关机；没有后台训练。 |

## 代码一致性

- 178、252、99 以及 197 的干净运维副本当前均位于提交 `6e72228`；其中 `eb9c440` 为公共运动 decoder，`25fe052` 为共享证据 decoder，`a8d8ded` 增加两者组合验证。
- 99 原有实验提交 `c104193` 在共同历史中被完整保留；未跟踪目录未覆盖或删除。
- 两项新结构及其组合共通过 36 项 decoder 单元测试；组合在零初始化时精确等于父模型，第一步反向中两组 adapter 均获得非零梯度。组合仅处于测试就绪状态，尚未分配实验编号、配置或算力；各单项结构均在配置展开、路径检查和真数据 smoke 后启动。
- 197 后续运维使用既有且干净的 `/data/users/litianhao/PairMOT_sync_3cb888d`；该副本已 fast-forward 至 `6e72228` 并在 197 环境通过 36/36 decoder 测试。历史目录 `/data/users/litianhao/PairMOT` 保持在原状态（39 个已修改、238 个未跟踪项），未覆盖、未清理，也不作为新实验代码源。

## 下一步

1. 在 epoch 4 获取 `0730_05` 与 `0730_02` 的 cls/det HOTA、DetA、AssA 和 `both_independent_AP50`。
2. 对 `0730_06` 同样在 epoch 4 做首轮 TrackEval，与 `0727_01` 的同点结果比较。
3. 仅当独立结构分别显示 cls 或 det 的可靠改善时，才构建组合模型；否则保留论文现有三段递进主线。
