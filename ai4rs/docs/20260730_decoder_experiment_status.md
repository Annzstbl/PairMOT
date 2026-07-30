# PairMOT decoder 实验状态（2026-07-30）

更新时间：2026-07-30 15:12 CST

## 当前研究原则

- 论文主线保持为：`0719_05 Base 52.417/61.265` → `0723_01 Base+Liquid 53.955/62.032` → `0727_01 +Encoder 54.437/62.393`。
- decoder 目标是同时超过 `0727_01` 的 cls HOTA 54.437 和 det HOTA 62.393。
- 不再进行 class-specific reweight、long-tail reweight 或大规模 residual-scale 扫描；优先验证有明确时序归纳偏置的模型结构。
- AutoDL 实例均处于关机状态，不纳入当前调度。

## 运行中

| 服务器 | 实验 | 状态 | 结构与判定方式 |
| --- | --- | --- | --- |
| 99 GPU 0,1 | `0730_05 ... decoder_commonmotion ... fresh` | `RUNNING`；epoch 1 iter 50，约 1.052 s/iter，loss 21.4343、grad_norm 130.5463 均有限，显存约 19.2 GB/rank | 以 `0727_01` 为严格父配置；共享 query/分类路径不变，用两帧 cross-attention 差分和周期角度 reference 位移预测零初始化的 5D 反对称运动修正。4-iter 真数据 DDP smoke 已通过，三层新权重均从 0 更新到约 `3.98e-4`。首个正式判断点为 epoch 4。 |
| 252 GPU 0,1 | `0730_02 decoder_boxonly_gradisolated_reg0p25 ... fresh` | `RUNNING`；epoch 2 iter 150，约 1.120 s/iter，loss 11.7349、grad_norm 25.5711 均有限 | 仅作为短期结构基线，不展开 scale sweep。首个 TrackEval 点为 epoch 4；若 cls/det 同点轨迹没有优于 `0727_01` 的信号，则停止并释放资源。 |

## 已完成或释放

| 服务器 | 实验 | 状态 | 说明 |
| --- | --- | --- | --- |
| 178 GPU 0 | `0729_05 ... pairdn_easyhardpositive ... fresh` | `COMPLETED`，训练与 epoch 72 验证完成 | 终点 `both_independent_AP50=0.5781`；该实验说明 positive-only DN 能恢复较高 cls，但 det/AssA 仍不足以替代论文主线。GPU 0 已释放，下一项仅安排新的 decoder 结构，不恢复 scale 队列。 |
| 252 GPU 0,1 | `0730_01 legacy 0728_03 resume` | `STOPPED`，保留至 epoch 56 checkpoint | 旧 decoder 初始化轨迹，仅作历史诊断，不与修复后正式实验合并解释。 |
| AutoDL | 所有实例 | `OFF` | 用户确认均已关机；没有后台训练。 |

## 代码一致性

- 178、252、99 当前均包含提交 `eb9c440`（`Add antisymmetric common-motion decoder`）。
- 99 原有实验提交 `c104193` 在共同历史中被完整保留；未跟踪目录未覆盖或删除。
- `0730_05` 通过 30 项 decoder 单元测试、配置展开、路径检查和双卡真数据 smoke 后启动。

## 下一步

1. 在 epoch 4 获取 `0730_05` 与 `0730_02` 的 cls/det HOTA、DetA、AssA 和 `both_independent_AP50`。
2. 178 GPU 0 用于互补的共享证据 decoder 结构短测，不做参数 scale 重复实验。
3. 仅当独立结构分别显示 cls 或 det 的可靠改善时，才构建组合模型；否则保留论文现有三段递进主线。
