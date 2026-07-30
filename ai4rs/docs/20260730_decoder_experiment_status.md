# PairMOT decoder 实验状态（2026-07-30）

更新时间：2026-07-30 21:13 CST

## 当前研究原则

- 论文主线保持为：`0719_05 Base 52.417/61.265` → `0723_01 Base+Liquid 53.955/62.032` → `0727_01 +Encoder 54.437/62.393`。
- decoder 目标是同时超过 `0727_01` 的 cls HOTA 54.437 和 det HOTA 62.393。
- 不再进行 class-specific reweight、long-tail reweight 或大规模 residual-scale 扫描；优先验证有明确时序归纳偏置的模型结构。
- AutoDL 实例均处于关机状态，不纳入当前调度。

## 运行中

| 服务器 | 实验 | 状态 | 结构与判定方式 |
| --- | --- | --- | --- |
| 197 GPU 4,5 | `0730_09 ... decoder_motiontrust ... fresh` | `RUNNING`；epoch 4 完整门控通过，20:27 已进入 epoch 5 iter 350，约 `1.61 s/iter`，loss `11.3962`、grad_norm `40.1396` 均有限 | 严格继承 `0727_01`。以双帧检测置信度门控有界反对称运动修正，并将修正限制在现有帧间位移包络内。epoch 4 结构审计通过；cls HOTA/DetA/AssA `37.878/27.722/55.486`，det `44.102/34.288/58.241`，相对父配置 HOTA `+1.669/+5.349`、DetA `+0.654/+1.834`；pair mAP/AP50 `0.16257/0.31578`，both-independent mAP/AP50 `0.18872/0.34231`，全部通过固定门槛，继续到 epoch 8。 |
| 252 GPU 0,1 | `0730_11 ... decoder_sharedrouting ... fresh` | `RUNNING`；五项启动门槛通过，21:04 到 epoch 1 iter 800，约 `1.1014 s/iter`，loss `13.4689`、grad_norm `19.3992` 均有限 | 每层两帧 deformable cross-attention 只共享 `sampling_offsets` 与 `attention_weights`，保留独立 `value_proj/output_proj`。真数据 4-iter smoke 的 12 组 routing 误差为零，projection 已分化；首判 epoch 4。 |
| 178 GPU 0 | `0730_12 ... decoder_motiontrust_sharedevidence ... fresh` | `RUNNING`；真数据 smoke 与结构检查通过，五项正式启动门槛通过，21:12 到 epoch 1 iter 100，约 `0.8798 s/iter`，loss `20.1321`、grad_norm `90.7448` 均有限 | 将 `0730_09` 的检测保护框修正与 `0730_06` 的共享 query 证据组合；两条路径正交且均为零起点。smoke checkpoint 中两组三层 adapter 均有限、非零；首判 epoch 4。 |

## 已完成或释放

| 服务器 | 实验 | 状态 | 说明 |
| --- | --- | --- | --- |
| 252 GPU 0,1 | `0730_10 ... decoder_symmetricpair ... fresh` | `STOPPED`；2026-07-30 20:12 CST 在 epoch 4 完整检测和 2/2 TrackEval 后按固定门槛停止 | cls HOTA/DetA/AssA `36.750/26.756/54.632`，det `42.604/33.160/55.890`；虽然 det HOTA/DetA 相对父配置提高 `3.851/0.706`，pair AP50 与 both-independent AP50 分别提高约 `0.0147/0.0169`，但 pair mAP `0.1496` 相对父配置下降 `0.00765`，不满足检测保护门槛。结构语义审计通过：24 组共享 attention 参数误差为零；fusion 半矩阵最大误差 `1.79e-7` 是 FP32 更新漂移，不影响显式正反序平均的交换等变计算。保留 epoch 4 checkpoint 和全部评估，GPU 已释放。 |
| 99 GPU 0,1 | `0730_05 ... decoder_commonmotion ... fresh` | `STOPPED`；2026-07-30 17:49 CST 按 epoch 8 门槛主动停止，保留 epoch 4/8 checkpoint、检测结果及 2/2 TrackEval | epoch 8 cls/det HOTA `44.250/49.285`，相对 `0727_01` 同点 `-1.019/-0.908`。cls DetA/AssA `35.225/58.698`，相对父实验 `-2.438/+1.397`；det DetA/AssA `42.215/59.308`，相对父实验 `-4.846/+4.163`。关联增益持续存在，但以明显损害检测覆盖为代价。pair mAP/AP50 `0.2253/0.4141`，相对父实验 `-0.0124/-0.0168`；both-independent AP50 `0.4470`（`-0.0189`）。停止后精确清理两个残留 worker，GPU 0/1 连续采样最终均为 `12 MiB/0%`。 |
| 178 GPU 0 | `0730_06 ... decoder_sharedevidence ... fresh` | `STOPPED`；2026-07-30 17:42 CST 按 epoch 8 门槛主动停止，保留 epoch 4/8 checkpoint、检测结果及 2/2 TrackEval | epoch 8 cls/det HOTA `45.218/50.225`，相对 `0727_01` 同点仅 `-0.051/+0.032`；早期 HOTA 优势已经消失。cls DetA/AssA 为 `38.087/56.460`，相对父实验 `+0.424/-0.841`；det DetA/AssA 为 `46.507/56.028`，相对父实验 `-0.554/+0.883`，属于检测与关联之间的指标搬运，不是双提升。pair mAP/AP50 `0.2414/0.4505`，相对父实验 `+0.0036/+0.0196`，both-independent AP50 `0.4839`（`+0.0179`），不足以覆盖 HOTA 门槛失败。停止后精确清理该工作目录的 orphan DataLoader，GPU 0 连续采样最终为 `1 MiB/0%`，未影响其它任务。 |
| 252 GPU 0,1 | `0730_02 decoder_boxonly_gradisolated_reg0p25 ... fresh` | `STOPPED`；2026-07-30 16:17 CST 主动停止，保留 epoch 4 checkpoint、检测结果和完整首轮 TrackEval | epoch 4 cls/det HOTA `35.883/42.045`，相对 `0727_01` 同点 `-0.326/+3.292`。det 表面增益主要来自早期 AssA 激增，pair AP50 `+0.0284` 但 pair mAP `-0.0027`；与旧 dual-output 的早期虚高轨迹一致，因此不展开 residual-scale 扫描或长跑。 |
| 178 GPU 0 | `0729_05 ... pairdn_easyhardpositive ... fresh` | `COMPLETED`；训练及 epoch 72 检测完成，现存 15/18 个 TrackEval 点 | 暂定最佳 epoch 68 为 cls/det HOTA `55.319/61.515`，同点 pair mAP/AP50 `0.3264/0.5494`。positive-only DN 能恢复较高 cls，但 det 不足以替代论文主线；不得把 15 点汇总误写为完整 18/18。 |
| 252 GPU 0,1 | `0730_01 legacy 0728_03 resume` | `STOPPED`，保留至 epoch 56 checkpoint | 旧 decoder 初始化轨迹，仅作历史诊断，不与修复后正式实验合并解释。 |
| AutoDL | 所有实例 | `OFF` | 用户确认均已关机；没有后台训练。 |

## 代码一致性

- 99、252、178 以及 197 的干净运维副本均位于提交 `f1ffbd7`。其中 `c43635c` 引入 motion-trust + shared-evidence 组合结构，`f1ffbd7` 修复 PyTorch 2.6+ 对可信本地 smoke checkpoint 的显式加载；更早的 motion-trust、shared-routing、symmetric-pair、common-motion、shared-evidence、competitive-evidence 历史仍完整保留。
- 99 原有实验提交 `c104193` 在共同历史中完整保留；各服务器未跟踪目录未覆盖或删除。
- common-motion、shared-evidence、competitive-evidence 及组合结构共通过 41 项 decoder 单元测试；零初始化时精确等于父模型，第一步反向中 adapter 均有非零梯度。所有正式运行均在配置展开、路径检查和真数据 smoke 后启动。
- 197 使用干净副本 `/data/users/litianhao/PairMOT_sync_3cb888d`。历史目录 `/data/users/litianhao/PairMOT` 保持原状，不作为新实验代码源。

## 下一步

1. `0730_09` 继续到 epoch 8，并按相同的 HOTA、DetA、pair mAP 与 both-independent AP50 门槛判断早期共同改善能否保持。
2. 99 GPU 0,1、252 GPU 0,1 与 178 GPU 0 暂时空闲，不用参数扫描填卡；只为具有独立结构诊断价值的下一候选使用。
3. motion-trust 是当前第一次同时改善 HOTA、DetA、AssA 与 AP 的 decoder 结构。下一候选优先解决跨帧 cross-attention 的参数偏置，同时保留各帧的 value/output 投影容量；不再重复无保护的 common-motion 或完全共享。
4. 论文主线暂不改变；只有 decoder 候选在中后期同时超过 `0727_01` 的 cls/det HOTA，才进入论文递进表。

## 2026-07-30 18:25 CST 状态覆盖

- `0730_07 common-motion + shared-evidence` 已在 epoch 4 完成检测和 2/2 TrackEval 后停止。cls/det HOTA 为 `36.926/42.702`，相对 `0727_01` 同点为 `+0.717/+3.949`；但 cls/det DetA 均约下降 `1.0`，收益主要来自 AssA，pair mAP 与 both-independent AP50 也未形成一致提升，因此不继续消耗训练资源。
- `0730_08 competitive-evidence` 已在 epoch 4 完成检测和 2/2 TrackEval 后停止。cls/det HOTA 为 `36.277/43.729`，相对 `0727_01` 同点为 `+0.068/+4.976`；det AssA 提高 `9.441`，但 pair mAP 下降 `0.0041`，仍是早期关联尖峰而非检测与关联的共同改善。
- 197 的两张实验卡固定为 GPU 4、5。`0730_09 motion-trust decoder` 于 18:23 fresh 启动，父配置严格为 `0727_01`。该结构以双帧检测置信度门控反对称运动修正，并将修正限制在现有帧间位移包络内，用于直接保护 DetA；不含类别重权、loss 重权或 residual-scale 扫描。
- `0730_09` 已通过配置深拷贝、47 项 decoder 单元测试和双卡真实数据 4-iter smoke；smoke 的总 loss、DN loss、encoder loss 和 grad norm 均有限，三个 motion-trust adapter 均产生非零更新。正式训练已通过 epoch 1 iter 50 启动门槛：`0.9085 s/iter`、loss `21.3915`、grad norm `120.0892`，GPU 4/5 各约 `19.2 GiB`，无 Traceback、OOM、NaN、NCCL 或 unused-parameter 错误。
- 当前资源池为 99 双卡、197 双卡、252 双卡、178 单卡；AutoDL 保持关机。空闲资源只分配给具有结构诊断价值的 decoder 改动，不用参数扫描填满。
- 代码提交为 `f231b01`；99 canonical 和 197 运维副本已精确同步。GitHub HTTPS 连通但远端未提供非交互认证，故本轮先使用带 prerequisite 的 Git bundle 完成一致性同步，未误报为已推送。

## 2026-07-30 18:46 CST 资源扩展与 symmetric-pair 实验

- 197 的两张实验卡 GPU 4、5 已正式计入资源池，并正在运行 `0730_09 motion-trust decoder`，不是空闲资源。18:45 到 epoch 1 iter 900，约 `1.5280 s/iter`、loss `12.1506`、grad norm `21.8685`；总 loss、DN loss、encoder loss 均有限，GPU 各约 `19.2 GiB`，未见 Traceback、OOM、NaN、NCCL 或 unused-parameter 错误。首个判断点仍为 epoch 4。
- 252 GPU 0、1 已启动 `0730_10 symmetric-pair decoder`。它针对原 pair decoder 中两帧同模态、同 encoder 却使用独立 cross-attention 和有序拼接融合的问题：共享 prev/curr deformable cross-attention 权重，并对 feature fusion 与 pair-position fusion 的正反帧序输出取平均。该结构不增加参数，也不改 encoder、proposal、PairDN、head、loss 和训练协议。
- `0730_10` 的 52 项 decoder 单元测试、配置深拷贝、launcher shell 审计和双卡真实数据 4-iter smoke 全部通过。预训练权重中 24 组 prev/curr attention 参数精确相等；smoke checkpoint 中 4 个 fusion 矩阵的帧交换误差最大为 `8.45e-09`，说明帧交换等变约束在真实反向更新后仍成立。
- 正式 `0730_10` 已通过五项启动门槛。18:45 到 epoch 1 iter 200，约 `1.0837 s/iter`、loss `18.9054`、grad norm `56.8993`，DN 与 encoder loss 均有限；GPU 0/1 各约 `19.2 GiB`、采样时利用率 100%，无异常。首个判断点为 epoch 4。
- 当前实际运行：197 双卡 `0730_09`、252 双卡 `0730_10`；99 双卡与 178 单卡暂时保留。后两者只在 epoch 4 结果给出明确结构诊断后用于下一模型改动，不用于 residual-scale、loss 权重或类别重权扫描。
- 新结构代码提交为 `5341c32`，已通过 Git bundle 精确同步到 99、197、252、178，保留各服务器原有未跟踪文件。GitHub 尚缺非交互认证，本轮未误报为已推送。

## 2026-07-30 20:29 CST epoch 4 门控结果

- `0730_10 symmetric-pair decoder` 已于 20:12 CST 停止。epoch 4 cls HOTA/DetA/AssA 为 `36.750/26.756/54.632`，det 为 `42.604/33.160/55.890`；pair mAP/AP50 为 `0.1496/0.3108`，both-independent mAP/AP50 为 `0.1786/0.3400`。其 det HOTA、DetA 与 AP50 有增益，但 pair mAP 相对 `0727_01` epoch 4 的 `0.157253` 下降 `0.00765`，超过 `0.003` 保护线，因此完成 checkpoint、检测和 2/2 TrackEval 后停止，252 GPU 0/1 已释放。
- `0730_10` 的共享 attention 参数在 epoch 4 checkpoint 中仍逐项相等。fusion 半矩阵最大差异为 `1.788e-7`，略高于原检查脚本的 `1e-7` 绝对阈值；进一步 FP32 数值审计确认 24 组 attention 最大误差为零、4 个 fusion 最大半矩阵误差仅为该数值。由于前向显式平均正反两种帧序，计算上的交换等变性不依赖两半权重继续相等，故这是数值漂移而非结构失效。
- `0730_09 motion-trust decoder` 的 epoch 4 已完成 checkpoint、检测及 2/2 TrackEval。结构审计通过，三层 adapter 最大绝对权重为 `0.14190/0.13821/0.12903`，均有限且非零。cls HOTA/DetA/AssA 为 `37.878/27.722/55.486`，相对父配置为 `+1.669/+0.654/+3.392`；det 为 `44.102/34.288/58.241`，相对父配置为 `+5.349/+1.834/+10.775`。pair mAP/AP50 为 `0.16257/0.31578`，相对父配置为 `+0.00531/+0.01965`；both-independent mAP/AP50 为 `0.18872/0.34231`，相对父配置为 `+0.00425/+0.01916`。所有预先固定的门槛均通过，不是单独 AssA 尖峰。
- `0730_09` 保持在 197 GPU 4/5 继续到 epoch 8。20:27 已到 epoch 5 iter 350，训练 loss、DN loss、encoder loss 与 grad norm 均有限，无 Traceback、OOM、NaN、NCCL 或 unused-parameter 错误。当前空闲资源为 99 GPU 0/1、252 GPU 0/1 和 178 GPU 0；197 其他 GPU 不属于本实验资源池。

## 2026-07-30 21:13 CST 三路正式并行

- 197 `0730_09` 已到 epoch 6 iter 750，继续按 epoch 8 门控判断。
- 252 `0730_11 shared-routing` 于 20:49 fresh 启动，21:04 到 epoch 1 iter 800。
  它只共享两帧 deformable cross-attention 的几何路由，不共享 value/output projection，
  首判 epoch 4。
- 178 `0730_12 motion-trust + shared-evidence` 于 21:10 fresh 启动，21:12 到
  epoch 1 iter 100。57 项单测、配置/路径审计与单卡 physical batch 8 真数据 4-iter
  smoke 通过；smoke checkpoint 的三层 motion-trust 最大绝对权重为
  `3.7246e-4/3.6952e-4/3.6620e-4`，三层 shared-evidence 为
  `3.9646e-4/3.9722e-4/3.9787e-4`，均有限且非零。
- 99 当前不能形成双卡正式训练：GPU 0/2 被其他用户的 UNet 任务各占约 21 GiB，仅
  GPU 1 空闲。没有清理或抢占外部任务，也没有用参数扫描填卡。
- 四台服务器代码一致位于 `f1ffbd7`；AutoDL 保持全部关机。
