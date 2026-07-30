# PairMOT decoder 实验状态（2026-07-30）

更新时间：2026-07-31 01:54 CST

## 当前研究原则

- 论文主线保持为：`0719_05 Base 52.417/61.265` → `0723_01 Base+Liquid 53.955/62.032` → `0727_01 +Encoder 54.437/62.393`。
- decoder 目标是同时超过 `0727_01` 的 cls HOTA 54.437 和 det HOTA 62.393。
- 不再进行 class-specific reweight、long-tail reweight 或大规模 residual-scale 扫描；优先验证有明确时序归纳偏置的模型结构。
- AutoDL 实例均处于关机状态，不纳入当前调度。

## 运行中

| 服务器 | 实验 | 状态 | 结构与判定方式 |
| --- | --- | --- | --- |
| 197 GPU 4,5 | `0731_04 ... decoder_orthogonalevidence ... fresh` | `RUNNING`；01:52 fresh 启动，01:53 到 epoch 1 iter 50 | 将公共证据旁路与受包络约束的反对称帧差作为两个正交、零起点的 head-only 分量；不改变 recurrent query。78 项 decoder 单测、完整模型构建、配置深拷贝、launcher 审计和双卡真实数据 4-iter smoke 通过；两组三层门控均非零且有限。iter 50 约 `0.9036 s/iter`，loss `21.5469`、grad norm `111.8402`，总/DN/encoder loss 有限。 |
| 252 GPU 0,1 | `0731_03 ... decoder_commonevidencebypass ... fresh` | `RUNNING`；01:44 fresh 启动，01:46 到 epoch 1 iter 50 | recurrent query 保持父模型；仅让 heads 通过零起点、交换不变且受限的残差恢复可能被 fusion/FFN 抑制的两帧公共证据。4-iter smoke 和 checkpoint 结构检查通过。iter 50 约 `1.1786 s/iter`，loss `21.3837`、grad norm `130.9290`，总/DN/encoder loss 有限。 |
| 178 GPU 0 | `0731_01 ... decoder_sharedattention_antisymmetricdetail ... fresh` | `RUNNING`；01:11 fresh 启动，01:12 到 epoch 1 iter 50，约 `0.9344 s/iter`，loss `20.9701`、grad_norm `96.9294` 均有限 | 组合当前最强早期主效应 shared-attention 与只作用于逐帧 head 的中点守恒 `-detail/+detail`。真实数据 4-iter smoke 的总、DN、encoder loss 与梯度均有限；checkpoint 同时通过 `SHARED_ATTENTION_CHECKPOINT_OK` 和 `ANTISYMMETRIC_DETAIL_CHECKPOINT_OK`；首判 epoch 4。 |
| 99 GPU 0,1 | `0731_02 ... decoder_envelopeddetail ... fresh` | `RUNNING`；01:44 fresh 启动，01:45 到 epoch 1 iter 50 | recurrent query 保持父模型；逐帧 heads 只接收存在于真实双帧 cross-attention 差异内的交换反对称校正，幅值逐元素受观测帧差包络约束。4-iter smoke 和 checkpoint 结构检查通过。iter 50 约 `0.9785 s/iter`，loss `21.4400`、grad norm `112.8147`，总/DN/encoder loss 有限。 |

## 已完成或释放

| 服务器 | 实验 | 状态 | 说明 |
| --- | --- | --- | --- |
| 197 GPU 4,5 | `0730_15 ... decoder_sharedevidence_sharedattention ... fresh` | `STOPPED`；epoch 8 完整评估和结构审计后于 01:35 精确停止 | epoch 8 cls HOTA/DetA/AssA `43.386/34.314/57.656`，det `49.340/42.544/59.013`；pair mAP/AP50 `0.207499/0.401020`，both-independent mAP/AP50 `0.241319/0.431198`。早期优势未保持，明显低于 `0727_01` 同点；6 组共享 attention 误差为零、两类结构参数均已学习，故结论是结构方向失败而非未生效。 |
| 99 GPU 0,1 | `0730_14 ... decoder_motiontrust_sharedattention ... fresh` | `STOPPED`；epoch 8 完整评估和结构审计后按固定门槛停止 | epoch 8 cls HOTA/DetA/AssA `44.366/35.594/57.867`，det `50.647/44.289/60.035`；pair mAP/AP50 `0.221063/0.416210`，both-independent mAP/AP50 `0.255138/0.449079`。相对父配置 HOTA `-0.903/+0.454`、DetA `-2.069/-2.772`，检测覆盖损失明显。 |
| 252 GPU 0,1 | `0730_13 ... decoder_sharedattention ... fresh` | `STOPPED`；epoch 8 完整评估和结构审计后按固定门槛停止 | epoch 8 cls HOTA/DetA/AssA `43.310/34.380/57.487`，det `49.011/43.032/57.581`；pair mAP/AP50 `0.207615/0.397715`，both-independent mAP/AP50 `0.242288/0.429662`。相对父配置 HOTA `-1.959/-1.182`、DetA `-3.283/-4.029`；epoch 4 的关联增益在 epoch 8 退化为检测覆盖损失。 |
| 178 GPU 0 | `0730_16 ... decoder_antisymmetricdetail ... fresh` | `STOPPED`；epoch 4 checkpoint、检测、完整 TrackEval 与结构审计后按固定门槛停止 | epoch 4 cls HOTA/DetA/AssA `36.684/27.590/52.398`，det `39.221/31.788/49.436`；pair mAP/AP50 `0.1700/0.3110`，both-independent mAP/AP50 `0.1992/0.3428`。相对父配置 HOTA 和 AP 均提高，但 det DetA 下降 `0.666`，超过允许的 `0.5`；仅超限 `0.166`，因此保留其细节分支并与 shared-attention 组合验证。 |
| 178 GPU 0 | `0730_12 ... decoder_motiontrust_sharedevidence ... fresh` | `STOPPED`；epoch 8 checkpoint、检测、结构审计及 2/2 TrackEval 完成后按固定门槛停止 | epoch 8 cls HOTA/DetA/AssA `44.387/33.262/61.885`，det `49.824/40.708/63.485`；pair mAP/AP50 `0.2144/0.3809`，both-independent mAP/AP50 `0.2457/0.4098`。相对父配置 cls/det HOTA `-0.882/-0.369`，cls/det DetA `-4.401/-6.353`，pair mAP `-0.02333`、both-independent AP50 `-0.05615`；epoch 4 的共同增益未保持，属于以检测覆盖换 AssA，23:30 前精确停止并释放 GPU 0。 |
| 197 GPU 4,5 | `0730_09 ... decoder_motiontrust ... fresh` | `STOPPED`；epoch 8 checkpoint、检测、结构审计及 2/2 TrackEval 完成后按固定门槛停止 | epoch 8 cls HOTA/DetA/AssA `45.498/36.788/58.984`，det `51.160/45.018/60.159`；pair mAP/AP50 `0.230058/0.430536`，both-independent mAP/AP50 `0.265862/0.464984`。相对父配置 HOTA 仅 `+0.229/+0.967`，但 cls/det DetA 下降 `0.875/2.043`、pair mAP 下降 `0.007676`；属于 AssA 搬运，结构虽有效但保护线失败。23:00 前已精确停止并释放 GPU 4/5。 |
| 252 GPU 0,1 | `0730_11 ... decoder_sharedrouting ... fresh` | `STOPPED`；epoch 4 checkpoint、检测、结构审计及 2/2 TrackEval 完成后按固定门槛停止 | cls HOTA/DetA/AssA `36.504/27.255/51.884`，det `42.163/33.695/53.992`；pair mAP/AP50 `0.149753/0.305401`，both-independent mAP/AP50 `0.178556/0.335452`。虽然 det HOTA 与 DetA 提升，但 pair mAP 相对父配置下降 `0.007499`，超过 `0.003` 保护线。共享 routing 结构审计通过，停止后 GPU 0/1 已释放。 |
| 252 GPU 0,1 | `0730_10 ... decoder_symmetricpair ... fresh` | `STOPPED`；2026-07-30 20:12 CST 在 epoch 4 完整检测和 2/2 TrackEval 后按固定门槛停止 | cls HOTA/DetA/AssA `36.750/26.756/54.632`，det `42.604/33.160/55.890`；虽然 det HOTA/DetA 相对父配置提高 `3.851/0.706`，pair AP50 与 both-independent AP50 分别提高约 `0.0147/0.0169`，但 pair mAP `0.1496` 相对父配置下降 `0.00765`，不满足检测保护门槛。结构语义审计通过：24 组共享 attention 参数误差为零；fusion 半矩阵最大误差 `1.79e-7` 是 FP32 更新漂移，不影响显式正反序平均的交换等变计算。保留 epoch 4 checkpoint 和全部评估，GPU 已释放。 |
| 99 GPU 0,1 | `0730_05 ... decoder_commonmotion ... fresh` | `STOPPED`；2026-07-30 17:49 CST 按 epoch 8 门槛主动停止，保留 epoch 4/8 checkpoint、检测结果及 2/2 TrackEval | epoch 8 cls/det HOTA `44.250/49.285`，相对 `0727_01` 同点 `-1.019/-0.908`。cls DetA/AssA `35.225/58.698`，相对父实验 `-2.438/+1.397`；det DetA/AssA `42.215/59.308`，相对父实验 `-4.846/+4.163`。关联增益持续存在，但以明显损害检测覆盖为代价。pair mAP/AP50 `0.2253/0.4141`，相对父实验 `-0.0124/-0.0168`；both-independent AP50 `0.4470`（`-0.0189`）。停止后精确清理两个残留 worker，GPU 0/1 连续采样最终均为 `12 MiB/0%`。 |
| 178 GPU 0 | `0730_06 ... decoder_sharedevidence ... fresh` | `STOPPED`；2026-07-30 17:42 CST 按 epoch 8 门槛主动停止，保留 epoch 4/8 checkpoint、检测结果及 2/2 TrackEval | epoch 8 cls/det HOTA `45.218/50.225`，相对 `0727_01` 同点仅 `-0.051/+0.032`；早期 HOTA 优势已经消失。cls DetA/AssA 为 `38.087/56.460`，相对父实验 `+0.424/-0.841`；det DetA/AssA 为 `46.507/56.028`，相对父实验 `-0.554/+0.883`，属于检测与关联之间的指标搬运，不是双提升。pair mAP/AP50 `0.2414/0.4505`，相对父实验 `+0.0036/+0.0196`，both-independent AP50 `0.4839`（`+0.0179`），不足以覆盖 HOTA 门槛失败。停止后精确清理该工作目录的 orphan DataLoader，GPU 0 连续采样最终为 `1 MiB/0%`，未影响其它任务。 |
| 252 GPU 0,1 | `0730_02 decoder_boxonly_gradisolated_reg0p25 ... fresh` | `STOPPED`；2026-07-30 16:17 CST 主动停止，保留 epoch 4 checkpoint、检测结果和完整首轮 TrackEval | epoch 4 cls/det HOTA `35.883/42.045`，相对 `0727_01` 同点 `-0.326/+3.292`。det 表面增益主要来自早期 AssA 激增，pair AP50 `+0.0284` 但 pair mAP `-0.0027`；与旧 dual-output 的早期虚高轨迹一致，因此不展开 residual-scale 扫描或长跑。 |
| 178 GPU 0 | `0729_05 ... pairdn_easyhardpositive ... fresh` | `COMPLETED`；训练及 epoch 72 检测完成，现存 15/18 个 TrackEval 点 | 暂定最佳 epoch 68 为 cls/det HOTA `55.319/61.515`，同点 pair mAP/AP50 `0.3264/0.5494`。positive-only DN 能恢复较高 cls，但 det 不足以替代论文主线；不得把 15 点汇总误写为完整 18/18。 |
| 252 GPU 0,1 | `0730_01 legacy 0728_03 resume` | `STOPPED`，保留至 epoch 56 checkpoint | 旧 decoder 初始化轨迹，仅作历史诊断，不与修复后正式实验合并解释。 |
| AutoDL | 所有实例 | `OFF` | 用户确认均已关机；没有后台训练。 |

## 代码一致性

- 四台服务器当前代码已统一至 `3d65dc4`；同步未重启 99、252、178 的在途训练。
- 99 原有实验提交 `c104193` 在共同历史中完整保留；各服务器未跟踪目录未覆盖或删除。
- 当前 decoder 测试集共 78 项通过。`0731_02/03/04` 均在零初始化时精确等于父模型，第一步反向中目标门控非零；所有正式运行均在配置展开、完整模型构建、路径检查和真数据 smoke 后启动。
- 197 使用干净副本 `/data/users/litianhao/PairMOT_sync_3cb888d`。历史目录 `/data/users/litianhao/PairMOT` 保持原状，不作为新实验代码源。

## 下一步

1. `0731_01` 继续到 epoch 4，检验 shared-attention 与 antisymmetric detail 的组合能否同时保护 DetA、AP 和 HOTA。
2. `0731_02/03/04` 先做 epoch 4 同点诊断；分别判断帧差、公共证据及二者正交组合的主效应与交互。
3. 四台服务器维持四路结构实验并行；不以参数扫描、loss scale 或类别 reweight 填充资源。
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

## 2026-07-30 23:45 CST 0730_12 epoch-8 门控与 0730_16

- `0730_12 motion-trust + shared-evidence` 的 epoch 8 完整结果为：cls HOTA/DetA/AssA
  `44.387/33.262/61.885`，det `49.824/40.708/63.485`，pair mAP/AP50
  `0.2144/0.3809`，both-independent mAP/AP50 `0.2457/0.4098`。相对父配置
  `0727_01` 同点，cls/det HOTA 为 `-0.882/-0.369`，cls/det DetA 为
  `-4.401/-6.353`，pair mAP 为 `-0.02333`，both-independent AP50 为
  `-0.05615`。虽然 AssA 提高 `+4.584/+8.340`，但这是明显的检测覆盖向关联搬运，
  epoch 4 的共同优势没有保持，因此在 checkpoint、检测、2/2 TrackEval 和结构审计
  完整后精确停止，178 GPU 0 释放。
- 178 的接替候选为 `0730_16 antisymmetric frame-detail decoder`。它完全保留
  `0727_01` 的 recurrent shared query 与下一层 decoder 路径，仅从本层两帧真实
  cross-attention 输出构造归一化、detach 的有符号帧间证据，通过每层一个共享、
  零初始化的 `Linear(D,D)` 和 `tanh` 产生有界细节，再以 `-detail/+detail`
  只注入两帧分类/回归 head 特征。两帧特征中点严格等于父模型共享输出，帧交换时修正
  严格变号；不直接修改共享 query 或框更新，针对前述 DetA/AP 损失设计。
- 该候选已通过 69 项 decoder 单元测试、正式/4-iter smoke 配置深拷贝、
  launcher shell 审计和 `git diff --check`。178 单卡真实数据 4-iter smoke 的四步
  总 loss 为 `21.1809/20.2144/20.1859/20.2035`，DN、encoder loss 和 grad norm
  均有限；checkpoint 三层 adapter 最大绝对权重约为
  `3.994e-4/3.998e-4/4.001e-4`，结构检查通过。
- 正式 fresh 训练于 23:48 启动，23:49 到 epoch 1 iter 50：`0.9458 s/iter`、
  loss `21.1606`、grad norm `107.4478`，GPU 0 设备占用约 `31.4 GiB`，
  总、DN、encoder loss 均有限，无 Traceback、OOM、NaN、NCCL 或 unused-parameter
  错误。五项启动门槛通过，四机并行恢复，首判 epoch 4。

## 2026-07-31 00:02 CST 0730_13 epoch-4 门控

- `0730_13 shared-attention decoder` 的 epoch 4 已完成 checkpoint、检测、完整
  TrackEval 与结构审计。cls HOTA/DetA/AssA 为 `37.559/27.119/55.846`，
  det 为 `43.257/33.530/56.895`；相对 `0727_01` 同点分别为
  `+1.350/+0.051/+3.752` 和 `+4.504/+1.076/+9.429`。
- pair mAP/AP50 为 `0.1547/0.3182`，both-independent mAP/AP50 为
  `0.1839/0.3467`。pair mAP 相对父配置下降约 `0.00255`，仍在预设 `0.003`
  容差内；both-independent AP50 提高约 `0.0235`。因此 HOTA、DetA 与 AP 保护线
  同时通过，不是只依靠 AssA 尖峰。
- checkpoint 中 6 组共享 `attention_weights` 最大误差为零，18 组应独立的
  sampling-offset/value/output 参数最大差异 `0.03846`，说明共享与逐帧自由度均按
  设计保持。实验继续到 epoch 8，再判断优势能否持续。

## 2026-07-31 00:04 CST 0730_14 epoch-4 门控

- `0730_14 motion-trust + shared-attention decoder` 的 epoch 4 已完成 checkpoint、
  检测、完整 TrackEval 与联合结构审计。cls HOTA/DetA/AssA 为
  `37.075/27.355/53.989`，det 为 `42.159/32.966/55.263`；相对父配置分别为
  `+0.866/+0.287/+1.895` 和 `+3.406/+0.512/+7.797`。
- pair mAP/AP50 为 `0.1625/0.3105`，both-independent mAP/AP50 为
  `0.1887/0.3369`；相对父配置 pair mAP 提高约 `0.00525`，both AP50 提高约
  `0.0138`。全部固定保护线通过，继续到 epoch 8。
- 三层 motion-trust adapter 最大绝对权重为 `0.1450/0.1337/0.1198`，6 组共享
  attention 权重误差为零，18 组独立参数最大差异 `0.03538`。但相对 `0730_13`
  同点，cls/det HOTA 分别低 `0.484/1.098`，因此当前证据更支持 shared-attention
  主效应，尚不支持 motion-trust 形成正交增益。

## 2026-07-31 00:24 CST 0730_15 epoch-4 门控

- `0730_15 shared-evidence + shared-attention` 已完成 epoch 4 checkpoint、检测、
  完整 TrackEval 与联合结构审计。cls HOTA/DetA/AssA 为
  `36.732/27.680/51.849`，det 为 `41.818/33.239/53.766`；相对父配置同点，
  cls/det HOTA 提高 `0.523/3.065`，DetA 提高 `0.612/0.785`。
- pair mAP/AP50 为 `0.157715/0.320595`，both-independent mAP/AP50 为
  `0.186071/0.345886`；pair mAP 提高约 `0.000462`，both AP50 提高约
  `0.022737`，全部固定保护线通过，实验继续到 epoch 8。
- 三层 shared-evidence adapter 最大绝对权重为
  `0.031539/0.029112/0.017190`，6 组共享 attention 权重误差为零，18 组应独立
  参数最大差异 `0.036013`。但相对 `0730_13 shared-attention` 同点，cls/det HOTA
  分别低 `0.827/1.439`；相对 `0730_14` 也低 `0.343/0.341`。现有证据支持
  shared-attention 是主要早期增益来源，尚不支持 shared-evidence 与其形成正交互补。

## 2026-07-31 01:13 CST 0730_16 门控与 0731_01 接替

- `0730_16 antisymmetric frame-detail decoder` 的 epoch 4 已完成 checkpoint、检测、
  完整 TrackEval 与结构审计。cls HOTA/DetA/AssA 为 `36.684/27.590/52.398`，
  det 为 `39.221/31.788/49.436`；pair mAP/AP50 为 `0.1700/0.3110`，
  both-independent mAP/AP50 为 `0.1992/0.3428`。
- 相对 `0727_01` 同点，cls/det HOTA 提高 `0.475/0.468`，AP 也共同提高；
  唯一失败项是 det DetA 下降 `0.666`，超过固定上限 `0.5` 共 `0.166`。
  因此 00:59 精确停止，不把 AssA 增益误写成完整通过。
- 接替实验 `0731_01` 将 `0730_13` 的 shared-attention 与该中点守恒帧细节组合。
  两者位于不同作用位置：前者约束双帧 cross-attention 的定位权重，后者只在逐帧
  cls/reg head 前注入反对称细节；不含 loss、类别权重或 residual-scale 调参。
- 四台服务器已统一到提交 `5c556e4`。178 单卡真实数据 4-iter smoke 生成
  `iter_4.pth`，全部关键训练数值有限，同时通过两项 checkpoint 结构检查。
  正式 fresh 训练于 01:11 启动，01:12 到 epoch 1 iter 50；GPU0 使用约
  `31.4 GiB`，日志持续更新且无 Traceback、OOM、NaN 或 NCCL 错误。
