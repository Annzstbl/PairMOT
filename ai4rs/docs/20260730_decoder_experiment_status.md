# PairMOT decoder 实验状态（2026-07-30）

更新时间：2026-07-30 17:43 CST

## 当前研究原则

- 论文主线保持为：`0719_05 Base 52.417/61.265` → `0723_01 Base+Liquid 53.955/62.032` → `0727_01 +Encoder 54.437/62.393`。
- decoder 目标是同时超过 `0727_01` 的 cls HOTA 54.437 和 det HOTA 62.393。
- 不再进行 class-specific reweight、long-tail reweight 或大规模 residual-scale 扫描；优先验证有明确时序归纳偏置的模型结构。
- AutoDL 实例均处于关机状态，不纳入当前调度。

## 运行中

| 服务器 | 实验 | 状态 | 结构与判定方式 |
| --- | --- | --- | --- |
| 99 GPU 0,1 | `0730_05 ... decoder_commonmotion ... fresh` | `RUNNING`；epoch 8 checkpoint 已保存，正在等待 epoch 8 检测与 TrackEval 完成 | epoch 4 cls/det HOTA `36.424/41.733`，相对 `0727_01` 同点为 `+0.215/+2.980`；cls DetA/AssA `26.059/54.005`，det DetA/AssA `30.636/58.011`。pair mAP/AP50 `0.1535/0.2962`，检测精度基本未增，但 det AssA 同点提高 `+10.545`。epoch 8 用于排除早期 AssA 虚高；评估完成后按双 HOTA 与 DetA/AssA 分解决定去留。 |
| 252 GPU 0,1 | `0730_07 ... decoder_commonmotion_sharedevidence ... fresh` | `RUNNING`；2026-07-30 17:40 到 epoch 4 iter 200，约 `1.20 s/iter`，loss `10.7489`、grad_norm `57.4364` 均有限 | 组合 shared-evidence 与 common-motion；两支均为零初始化，父配置为 `0727_01`，使用 2×batch4。首次启动因继承 99 绝对数据路径而在数据加载前退出、未产生参数更新；失败目录保留为 `0730_07_failed_99_path_before_iter_20260730_1635`，修复为继承 252 父配置后从全新目录启动。epoch 4 为首个判断点。 |
| 197 GPU 4,5 | `0730_08 ... decoder_competitiveevidence ... fresh` | `RUNNING`；2026-07-30 17:40 到 epoch 4 iter 350，约 `0.87 s/iter`，loss `10.7384`、grad_norm `39.2360` 均有限 | 将两帧 cross-attention 显式分成 common/detail，以无 bias 奇函数门做逐通道 detail 竞争；帧交换时门和 detail 同时反号，乘积不变，`tanh` 限制修正幅度，输入停止梯度。新增 `196,608` 参数。41 项 decoder 测试、配置深拷贝和双卡 4-iter 真数据 smoke 已通过；epoch 4 为首个判断点。 |

## 已完成或释放

| 服务器 | 实验 | 状态 | 说明 |
| --- | --- | --- | --- |
| 178 GPU 0 | `0730_06 ... decoder_sharedevidence ... fresh` | `STOPPED`；2026-07-30 17:42 CST 按 epoch 8 门槛主动停止，保留 epoch 4/8 checkpoint、检测结果及 2/2 TrackEval | epoch 8 cls/det HOTA `45.218/50.225`，相对 `0727_01` 同点仅 `-0.051/+0.032`；早期 HOTA 优势已经消失。cls DetA/AssA 为 `38.087/56.460`，相对父实验 `+0.424/-0.841`；det DetA/AssA 为 `46.507/56.028`，相对父实验 `-0.554/+0.883`，属于检测与关联之间的指标搬运，不是双提升。pair mAP/AP50 `0.2414/0.4505`，相对父实验 `+0.0036/+0.0196`，both-independent AP50 `0.4839`（`+0.0179`），不足以覆盖 HOTA 门槛失败。停止后精确清理该工作目录的 orphan DataLoader，GPU 0 连续采样最终为 `1 MiB/0%`，未影响其它任务。 |
| 252 GPU 0,1 | `0730_02 decoder_boxonly_gradisolated_reg0p25 ... fresh` | `STOPPED`；2026-07-30 16:17 CST 主动停止，保留 epoch 4 checkpoint、检测结果和完整首轮 TrackEval | epoch 4 cls/det HOTA `35.883/42.045`，相对 `0727_01` 同点 `-0.326/+3.292`。det 表面增益主要来自早期 AssA 激增，pair AP50 `+0.0284` 但 pair mAP `-0.0027`；与旧 dual-output 的早期虚高轨迹一致，因此不展开 residual-scale 扫描或长跑。 |
| 178 GPU 0 | `0729_05 ... pairdn_easyhardpositive ... fresh` | `COMPLETED`；训练及 epoch 72 检测完成，现存 15/18 个 TrackEval 点 | 暂定最佳 epoch 68 为 cls/det HOTA `55.319/61.515`，同点 pair mAP/AP50 `0.3264/0.5494`。positive-only DN 能恢复较高 cls，但 det 不足以替代论文主线；不得把 15 点汇总误写为完整 18/18。 |
| 252 GPU 0,1 | `0730_01 legacy 0728_03 resume` | `STOPPED`，保留至 epoch 56 checkpoint | 旧 decoder 初始化轨迹，仅作历史诊断，不与修复后正式实验合并解释。 |
| AutoDL | 所有实例 | `OFF` | 用户确认均已关机；没有后台训练。 |

## 代码一致性

- 99、252、178 以及 197 的干净运维副本均位于提交 `bfda295`。其中 `eb9c440` 为 common-motion decoder，`25fe052` 为 shared-evidence decoder，`a8d8ded` 为两者组合，`0517066` 为 competitive-evidence decoder，`bfda295` 记录 `0730_08` 正式启动。
- 99 原有实验提交 `c104193` 在共同历史中完整保留；各服务器未跟踪目录未覆盖或删除。
- common-motion、shared-evidence、competitive-evidence 及组合结构共通过 41 项 decoder 单元测试；零初始化时精确等于父模型，第一步反向中 adapter 均有非零梯度。所有正式运行均在配置展开、路径检查和真数据 smoke 后启动。
- 197 使用干净副本 `/data/users/litianhao/PairMOT_sync_3cb888d`。历史目录 `/data/users/litianhao/PairMOT` 保持原状，不作为新实验代码源。

## 下一步

1. 等待 `0730_05` epoch 8 的检测与 TrackEval；若 AssA 优势同样坍缩或不能形成 cls/det HOTA 双提升，则停止并释放 99 GPU 0,1。
2. `0730_07` 与 `0730_08` 均以 epoch 4 为首个判断点。除 cls/det HOTA 外，同时检查 DetA/AssA、pair mAP/AP50 与 both-independent AP50，避免把单一 AssA 早期尖峰当成结构收益。
3. 178 GPU 0 暂时空闲，不立即填入参数扫描；待两个新结构的 epoch 4 结果后，只为有明确诊断价值的下一结构使用。
4. 论文主线暂不改变；只有 decoder 候选在中后期同时超过 `0727_01` 的 cls/det HOTA，才进入论文递进表。
