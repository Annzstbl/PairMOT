# PairMOT decoder 实验状态（2026-07-30）

更新时间：2026-08-01 18:05 CST

## 当前研究原则

- 论文主线保持为：`0719_05 Base 52.417/61.265` → `0723_01 Base+Liquid 53.955/62.032` → `0727_01 +Encoder 54.437/62.393`。
- decoder 目标是同时超过 `0727_01` 的 cls HOTA 54.437 和 det HOTA 62.393。
- 不再进行 class-specific reweight、long-tail reweight 或大规模 residual-scale 扫描；优先验证有明确时序归纳偏置的模型结构。
- AutoDL 实例均处于关机状态，不纳入当前调度。
- 调度采用“快服务器首轮、慢服务器确认”：99/178/197 负责新结构快速筛选，252 只延续已投入实验或复验明确候选；同一时间只并行严格正交、能回答不同机制问题的候选。

## 运行中

| 服务器 | 实验 | 状态 | 结构与判定方式 |
| --- | --- | --- | --- |
| 252 GPU 0,1 | `0801_07 ... decoder_iterativeclsresidual ... fresh` | `RUNNING` | 普通 query 以 detached encoder proposal logits 为基底逐层学习分类残差；DN prefix 也沿用统一残差路径。既有投入保留到 e4，18:00 位于 epoch 3 iter 900，约 `1.08 s/iter`，loss/梯度有限。该实验给出统一路径对照，但 252 不再用于新结构首轮筛选。 |
| 99 GPU 0,1 | `0801_08 ... decoder_iterativeclsdnisolated ... fresh` | `RUNNING` | 普通 query 保留逐层分类残差；DN prefix 改用已有绝对分类头，消除“普通 query 有 encoder 基底、DN query 无同语义基底”的冲突。相对 `0801_07` 不增加总参数、attention 或 loss。真实 smoke、checkpoint 双类分类头更新审计通过；18:04 iter 50 为约 `1.02 s/iter`、loss `21.3868`、grad norm `99.5376`，无异常。 |
| 178 GPU 0 | `0801_09 ... decoder_iterativeclsdnisolatede2e ... fresh` | `RUNNING` | 在 `0801_08` 的 DN 隔离上，仅允许最终 decoder 分类损失贯通早期残差层；encoder proposal 基底仍 detached。与 `0801_08` 参数量完全相同，构成严格梯度路由因子比较。真实 smoke 与 checkpoint 更新审计通过；18:04 iter 50 为约 `0.95 s/iter`、loss `21.0149`、grad norm `137.5518`，无异常；验证 loader 改为单进程仅规避 178 `/dev/shm` 外部占用，不改预测或指标。 |
| 197 GPU 4,5 | 无 | `IDLE` | 保留给 e4/e8 出现明确优势后的确认实验，不启动第三个低信息候选。 |

## 已完成或释放

| 服务器 | 实验 | 状态 | 说明 |
| --- | --- | --- | --- |
| 252 GPU 0,1 | `0801_03 ... decoder_terminaldiagonalcentermotiondetailonly ... fresh` | `STOPPED`；epoch 8 全量评估与结构审计后于 08:54 精确停止 | epoch 8 cls HOTA/DetA/AssA `44.183/35.231/58.370`，det `50.011/44.289/58.441`；相对 Encoder 同点 HOTA `-1.086/-0.182`、DetA `-2.432/-2.772`，仅 AssA 提高。pair mAP/AP50 `0.210025/0.406177`，both-independent `0.245219/0.438068`，四项均明显下降。唯一 256 维 gate 最大值 `0.337337`、独立 attention 最大差异 `0.059180`，结构确已学习；完整 checkpoint、检测、TrackEval、50 序列与 108 文件保留，GPU 已释放。 |
| 178 GPU 0 | `0731_16 ... decoder_terminalcommonevidencebypass ... fresh` | `STOPPED`；epoch 8 全量评估与结构审计后于 15:03 精确停止 | epoch 8 cls HOTA/DetA/AssA `43.972/32.412/62.419`，det `49.378/39.704/63.985`；相对父模型 HOTA `-1.297/-0.815`、DetA `-5.251/-7.357`，而 AssA `+5.118/+8.840`，属于强烈的 DetA→AssA 搬运。pair mAP/AP50 下降 `0.032405/0.064222`，both-independent mAP/AP50 下降 `0.039642/0.070239`；terminal gate 最大权重 `0.032061`。完整产物保留，GPU0 已释放。 |
| 252 GPU 0,1 | `0731_13 ... decoder_sharedattention_terminalmidpointenvelopeddetail ... fresh` | `STOPPED`；epoch 8 全量评估与结构审计后于 12:11 精确停止 | epoch 8 cls HOTA/DetA/AssA `43.170/34.514/57.338`，det `48.484/42.910/56.678`；相对父 encoder 同点 HOTA `-2.099/-1.709`、DetA `-3.149/-4.151`，而 AssA `+0.037/+1.533`，属于检测覆盖下降而非双提升。pair mAP/AP50 `0.212921/0.391949`，both-independent mAP/AP50 `0.248771/0.430138`，诊断指标同样下降。结构检查通过：6 组共享 attention 误差为零、18 组独立参数最大差异 `0.058077`、terminal-midpoint gate 从 smoke 的约 `3.9e-4` 学到 `0.119707`，排除模块未学习。epoch 8 checkpoint、检测和完整 TrackEval 均保留；GPU 0/1 已释放。 |
| 99 GPU 0,1 | `0731_12 ... decoder_sharedattention_terminalenvelopeddetail ... fresh` | `STOPPED`；epoch 8 全量评估与结构审计后停止 | epoch 8 cls HOTA/DetA/AssA `43.178/35.246/55.355`，det `50.010/43.700/59.276`；相对父 encoder 同点 HOTA `-2.091/-0.183`，DetA `-2.417/-3.361`，cls AssA `-1.946`，仅 det AssA `+4.131`。pair mAP/AP50 `0.217536/0.410164`，both-independent AP50 `0.448156`，检测诊断也下降。结构检查通过，证明末层分类与回归同时专门化仍会损伤覆盖；GPU 0/1 已释放。 |
| 252 GPU 0,1 | `0731_05 ... decoder_sharedattention_envelopeddetail ... fresh` | `STOPPED`；epoch 16 全量评估后于 09:07 精确停止 | epoch 16 cls HOTA/DetA/AssA `51.007/42.964/62.130`，det `57.940/50.762/68.403`；相对父 encoder 同点 HOTA `-0.084/-0.380`。虽然 pair mAP 和 both-independent AP50 提高，但双 HOTA 未通过预设门槛，故不再消耗 epoch 17 以后资源。checkpoint、检测、TrackEval、原始 CSV 与结构检查均保留。 |
| 99 GPU 0,1 | `0731_09 ... decoder_sharedattention_regressionenvelopeddetail ... fresh` | `STOPPED`；epoch 8 全量评估后停止 | epoch 8 cls/det HOTA `44.043/49.271`，相对父 encoder 同点 `-1.226/-0.922`，双 HOTA 同时失败；结构 checkpoint 验收通过，说明是方向失败而非模块未学习。 |
| 197 GPU 4,5 | `0731_10 ... decoder_sharedattention_midpointregressionenvelopeddetail ... fresh` | `STOPPED`；epoch 8 全量评估后于 09:47 精确停止 | epoch 8 cls HOTA/DetA/AssA `45.254/36.908/58.207`，det `50.846/44.718/59.651`；相对父 encoder 同点 HOTA `-0.015/+0.653`。结合 178 同结构 epoch 12 双降，不再重复消耗；checkpoint、检测、完整 TrackEval 与原始 CSV 保留。 |
| 178 GPU 0 | `0731_11 ... decoder_sharedattention_midpointregressionenvelopeddetail ... fresh` | `STOPPED`；epoch 12 全量评估后于 09:47 精确停止 | epoch 12 cls HOTA/DetA/AssA `49.478/40.961/62.396`，det `56.451/50.351/65.494`；相对父 encoder 同点 HOTA `-0.202/-0.090`。epoch 8 的 det 优势未延续，midpoint-regression 方向未形成稳定双提升。 |
| 197 GPU 4,5 | `0731_08 ... decoder_sharedattention_classificationenvelopeddetail ... fresh` | `STOPPED`；epoch 4 后按 HOTA 优先规则恢复，epoch 8 完整评估后于 07:14 精确停止 | epoch 8 cls HOTA/DetA/AssA `43.801/35.436/56.745`，det `49.318/43.792/57.563`；相对父配置 HOTA `-1.468/-0.875`、DetA `-2.227/-3.269`。pair mAP `0.213045`、both-independent AP50 `0.436989` 也明显下降。共享 attention 误差为零，独立参数和三层分类门控均充分学习，结论是分类专用细节引起中期检测覆盖退化。 |
| 178 GPU 0 | `0731_01 ... decoder_sharedattention_antisymmetricdetail ... fresh` | `STOPPED`；epoch 8 全部 artifacts 和两项结构审计完成后精确停止 | cls HOTA/DetA/AssA `45.152/37.611/57.666`，det `50.817/46.745/57.206`；相对父配置 cls HOTA `-0.117`、det HOTA `+0.624`，cls/det DetA `-0.052/-0.316`。pair mAP `0.246292`、both-independent AP50 `0.479887`，分别提高 `0.008558/0.013936`。结构确已学习：6 组共享 attention 误差为零，18 组独立参数最大差异 `0.060115`，三层 detail 权重有限非零。因 cls HOTA 唯一低于父配置，未将关联/AP 增益误判为全门槛通过。 |
| 197 GPU 4,5 | `0730_15 ... decoder_sharedevidence_sharedattention ... fresh` | `STOPPED`；epoch 8 完整评估和结构审计后于 01:35 精确停止 | epoch 8 cls HOTA/DetA/AssA `43.386/34.314/57.656`，det `49.340/42.544/59.013`；pair mAP/AP50 `0.207499/0.401020`，both-independent mAP/AP50 `0.241319/0.431198`。早期优势未保持，明显低于 `0727_01` 同点；6 组共享 attention 误差为零、两类结构参数均已学习，故结论是结构方向失败而非未生效。 |
| 99 GPU 0,1 | `0730_14 ... decoder_motiontrust_sharedattention ... fresh` | `STOPPED`；epoch 8 完整评估和结构审计后按固定门槛停止 | epoch 8 cls HOTA/DetA/AssA `44.366/35.594/57.867`，det `50.647/44.289/60.035`；pair mAP/AP50 `0.221063/0.416210`，both-independent mAP/AP50 `0.255138/0.449079`。相对父配置 HOTA `-0.903/+0.454`、DetA `-2.069/-2.772`，检测覆盖损失明显。 |
| 252 GPU 0,1 | `0730_13 ... decoder_sharedattention ... fresh` | `STOPPED`；epoch 8 完整评估和结构审计后按固定门槛停止 | epoch 8 cls HOTA/DetA/AssA `43.310/34.380/57.487`，det `49.011/43.032/57.581`；pair mAP/AP50 `0.207615/0.397715`，both-independent mAP/AP50 `0.242288/0.429662`。相对父配置 HOTA `-1.959/-1.182`、DetA `-3.283/-4.029`；epoch 4 的关联增益在 epoch 8 退化为检测覆盖损失。 |
| 178 GPU 0 | `0730_16 ... decoder_antisymmetricdetail ... resume epoch 4` | `STOPPED`；按 HOTA 主规则恢复至 epoch 8，完整评估和结构审计后停止 | epoch 8 cls HOTA/DetA/AssA `44.591/37.157/56.398`，det `49.685/46.466/54.883`；相对父 encoder 同点 HOTA `-0.678/-0.508`，DetA `-0.506/-0.595`，AssA `-0.903/-0.262`。pair mAP/AP50 `0.241645/0.435966`，both-independent mAP/AP50 `0.279185/0.474553`。三层结构权重最大值 `0.035530/0.036084/0.034844`，方向失败而非未学习；结果归档为 `val_track_epoch08_resume_20260731`，GPU 0 已释放。 |
| 178 GPU 0 | `0730_12 ... decoder_motiontrust_sharedevidence ... fresh` | `STOPPED`；epoch 8 checkpoint、检测、结构审计及 2/2 TrackEval 完成后按固定门槛停止 | epoch 8 cls HOTA/DetA/AssA `44.387/33.262/61.885`，det `49.824/40.708/63.485`；pair mAP/AP50 `0.2144/0.3809`，both-independent mAP/AP50 `0.2457/0.4098`。相对父配置 cls/det HOTA `-0.882/-0.369`，cls/det DetA `-4.401/-6.353`，pair mAP `-0.02333`、both-independent AP50 `-0.05615`；epoch 4 的共同增益未保持，属于以检测覆盖换 AssA，23:30 前精确停止并释放 GPU 0。 |
| 197 GPU 4,5 | `0730_09 ... decoder_motiontrust ... resume epoch 8` | `STOPPED`；按 HOTA 主规则恢复至 epoch 12，完整评估和结构审计后停止 | epoch 12 cls HOTA/DetA/AssA `48.877/40.354/61.309`，det `55.819/48.907/65.923`；相对父 encoder 同点 HOTA `-0.803/-0.722`，DetA `-1.006/-1.438`，仅 det AssA `+0.344`。pair mAP/AP50 `0.263196/0.479269`，both-independent mAP/AP50 `0.300922/0.513250`。三层 motion-trust 权重最大值 `0.453352/0.294524/0.256910`，结构充分学习但双 HOTA 失败；结果归档为 `val_track_epoch12_resume_20260731`，GPU 4/5 已释放。 |
| 252 GPU 0,1 | `0730_11 ... decoder_sharedrouting ... fresh` | `STOPPED`；epoch 4 checkpoint、检测、结构审计及 2/2 TrackEval 完成后按固定门槛停止 | cls HOTA/DetA/AssA `36.504/27.255/51.884`，det `42.163/33.695/53.992`；pair mAP/AP50 `0.149753/0.305401`，both-independent mAP/AP50 `0.178556/0.335452`。虽然 det HOTA 与 DetA 提升，但 pair mAP 相对父配置下降 `0.007499`，超过 `0.003` 保护线。共享 routing 结构审计通过，停止后 GPU 0/1 已释放。 |
| 252 GPU 0,1 | `0730_10 ... decoder_symmetricpair ... fresh` | `STOPPED`；2026-07-30 20:12 CST 在 epoch 4 完整检测和 2/2 TrackEval 后按固定门槛停止 | cls HOTA/DetA/AssA `36.750/26.756/54.632`，det `42.604/33.160/55.890`；虽然 det HOTA/DetA 相对父配置提高 `3.851/0.706`，pair AP50 与 both-independent AP50 分别提高约 `0.0147/0.0169`，但 pair mAP `0.1496` 相对父配置下降 `0.00765`，不满足检测保护门槛。结构语义审计通过：24 组共享 attention 参数误差为零；fusion 半矩阵最大误差 `1.79e-7` 是 FP32 更新漂移，不影响显式正反序平均的交换等变计算。保留 epoch 4 checkpoint 和全部评估，GPU 已释放。 |
| 99 GPU 0,1 | `0730_05 ... decoder_commonmotion ... fresh` | `STOPPED`；2026-07-30 17:49 CST 按 epoch 8 门槛主动停止，保留 epoch 4/8 checkpoint、检测结果及 2/2 TrackEval | epoch 8 cls/det HOTA `44.250/49.285`，相对 `0727_01` 同点 `-1.019/-0.908`。cls DetA/AssA `35.225/58.698`，相对父实验 `-2.438/+1.397`；det DetA/AssA `42.215/59.308`，相对父实验 `-4.846/+4.163`。关联增益持续存在，但以明显损害检测覆盖为代价。pair mAP/AP50 `0.2253/0.4141`，相对父实验 `-0.0124/-0.0168`；both-independent AP50 `0.4470`（`-0.0189`）。停止后精确清理两个残留 worker，GPU 0/1 连续采样最终均为 `12 MiB/0%`。 |
| 178 GPU 0 | `0730_06 ... decoder_sharedevidence ... fresh` | `STOPPED`；2026-07-30 17:42 CST 按 epoch 8 门槛主动停止，保留 epoch 4/8 checkpoint、检测结果及 2/2 TrackEval | epoch 8 cls/det HOTA `45.218/50.225`，相对 `0727_01` 同点仅 `-0.051/+0.032`；早期 HOTA 优势已经消失。cls DetA/AssA 为 `38.087/56.460`，相对父实验 `+0.424/-0.841`；det DetA/AssA 为 `46.507/56.028`，相对父实验 `-0.554/+0.883`，属于检测与关联之间的指标搬运，不是双提升。pair mAP/AP50 `0.2414/0.4505`，相对父实验 `+0.0036/+0.0196`，both-independent AP50 `0.4839`（`+0.0179`），不足以覆盖 HOTA 门槛失败。停止后精确清理该工作目录的 orphan DataLoader，GPU 0 连续采样最终为 `1 MiB/0%`，未影响其它任务。 |
| 252 GPU 0,1 | `0730_02 decoder_boxonly_gradisolated_reg0p25 ... fresh` | `STOPPED`；2026-07-30 16:17 CST 主动停止，保留 epoch 4 checkpoint、检测结果和完整首轮 TrackEval | epoch 4 cls/det HOTA `35.883/42.045`，相对 `0727_01` 同点 `-0.326/+3.292`。det 表面增益主要来自早期 AssA 激增，pair AP50 `+0.0284` 但 pair mAP `-0.0027`；与旧 dual-output 的早期虚高轨迹一致，因此不展开 residual-scale 扫描或长跑。 |
| 178 GPU 0 | `0729_05 ... pairdn_easyhardpositive ... fresh` | `COMPLETED`；训练及 epoch 72 检测完成，现存 15/18 个 TrackEval 点 | 暂定最佳 epoch 68 为 cls/det HOTA `55.319/61.515`，同点 pair mAP/AP50 `0.3264/0.5494`。positive-only DN 能恢复较高 cls，但 det 不足以替代论文主线；不得把 15 点汇总误写为完整 18/18。 |
| 252 GPU 0,1 | `0730_01 legacy 0728_03 resume` | `STOPPED`，保留至 epoch 56 checkpoint | 旧 decoder 初始化轨迹，仅作历史诊断，不与修复后正式实验合并解释。 |
| AutoDL | 所有实例 | `OFF` | 用户确认均已关机；没有后台训练。 |

## 代码一致性

- 四台服务器当前代码已统一至 `885134e`；同步未重启任何在途训练。
- 99 原有实验提交 `c104193` 在共同历史中完整保留；各服务器未跟踪目录未覆盖或删除。
- 当前 decoder 测试集共 78 项通过。`0731_02/03/04` 均在零初始化时精确等于父模型，第一步反向中目标门控非零；所有正式运行均在配置展开、完整模型构建、路径检查和真数据 smoke 后启动。
- 197 使用干净副本 `/data/users/litianhao/PairMOT_sync_3cb888d`。历史目录 `/data/users/litianhao/PairMOT` 保持原状，不作为新实验代码源。

## 下一步

1. `0731_06` 在 178 完成单测、完整模型构建和真数据 4-iter smoke 后 fresh 启动，首判 epoch 4。
2. `0731_02/04` 继续到 epoch 8；`0731_05` 首判 epoch 4，分别判断受包络帧差、公共证据交互与 shared-attention 组合的可持续性。
3. 四台服务器维持四路结构实验并行；不以参数扫描、loss scale 或类别 reweight 填充资源。
4. 论文主线暂不改变；只有 decoder 候选在中后期同时超过 `0727_01` 的 cls/det HOTA，才进入论文递进表。

## 2026-08-01 09:11 CST：0731_01 低复杂度续跑

- 选择已有 `0731_01 shared-attention + antisymmetric detail` 从可信的 `epoch_8.pth` 原轨迹恢复，而不是构造更复杂的新模块或重训前 8 个 epoch。
- 结构效率审计：Encoder 为 `22,758,775` 参数，0731_01 为 `22,881,367`，增量 `122,592`（`0.539%`）；178 同机相近协议 epoch 9 日志均值约 `0.919 s/iter`，对照约 `0.898 s/iter`，下降约 `2.3%`，符合参数不超过约 1%、吞吐下降不超过 5% 的约束。
- 首次恢复因 PyTorch 2.6 默认 `weights_only=True` 拒绝旧 MMEngine checkpoint 而在加载阶段退出，未进入训练且 GPU 自动释放。随后仅在该受信任本地 checkpoint 的专用 launcher 中设置 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`，未修改模型或训练协议。
- 09:09 再次恢复，09:10 已通过正式五项门槛：epoch 9 iter 50、`0.9572 s/iter`、loss `10.2507`、grad norm `46.6676`，DN 与 encoder proposal losses 均有限；GPU0 约 `31.4 GiB`，无 Traceback/OOM/NaN/NCCL/DDP 异常。计划只续至 epoch 12 完整检测与 TrackEval 后判定。

## 2026-08-01 09:21 CST：0731_05 低成本补证

- `0731_05 shared-attention + enveloped detail` 是除 0731_01 外唯一保留到中期仍接近双门槛的简单候选：e8 相对 Encoder `+0.072/+1.396`，e12 `+0.491/-0.111`，e16 `-0.084/-0.380`，且检测 AP 未系统性下降。
- 完整模型为 `22,881,367` 参数，相对 Encoder 只增加 `122,592`（`0.539%`）；沿用原 252 双卡 physical `2xb4` 协议，从可信的 `epoch_16.pth` 恢复，只补到 epoch 20，不做参数扫描。
- 09:20 正式到达 epoch 17 iter 50：`1.1663 s/iter`、loss `10.4376`、grad norm `61.8605`，总/DN/encoder losses 均有限；两卡各约 `19.4 GiB`，无 Traceback/OOM/NaN/NCCL/DDP 异常。

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

## 2026-07-31 02:24 CST 0731_01 epoch-4 全门槛

- checkpoint、`val_det/epoch_03/metrics.json`、完整
  `val_track_eval/val_track_0001/metrics.json` 和 TrackEval 原始 CSV 均已落盘。
- cls HOTA/DetA/AssA 为 `37.590/28.607/52.920`，det 为
  `40.313/33.923/49.759`。相对 `0727_01` epoch 4，cls 三项分别
  `+1.381/+1.539/+0.826`，det 三项分别 `+1.560/+1.469/+2.293`；
  不是用 AssA 搬运 DetA。
- pair mAP 为 `0.173430`，both-independent AP50 为 `0.356102`，
  相对父配置分别 `+0.016177/+0.032953`，明显通过 `-0.003` 保护线。
- checkpoint 结构审计确认 6 组共享 attention 权重误差为零，18 组应独立参数
  最大差异 `0.042917`；三层 antisymmetric-detail 权重范数分别为
  `0.024836/0.021828/0.023927`。两项结构均已真实学习。
- 该组合是当前首个在统一 epoch-4 门槛上同时提高 HOTA、DetA、AssA 与 AP 的
  decoder 候选，已进入 epoch 5 并保留到 epoch 8；此时仍不改写论文最终主线。

## 2026-07-31 03:08 CST 0731_02 epoch-4 全门槛

- checkpoint、检测指标、完整 TrackEval 和原始 CSV 均已落盘。
- cls HOTA/DetA/AssA 为 `37.859/28.112/54.688`，det 为
  `42.873/34.173/55.071`；相对 `0727_01` epoch 4，cls 三项分别
  `+1.650/+1.044/+2.594`，det 三项分别 `+4.120/+1.719/+7.605`。
- pair mAP/AP50 为 `0.167896/0.317776`，both-independent mAP/AP50 为
  `0.196745/0.349436`；相对父配置 pair mAP 和 both-independent AP50 分别
  `+0.010643/+0.026287`，AP 保护线明显通过。
- checkpoint 结构审计通过；三层 enveloped-detail 权重最大绝对值分别为
  `0.066020/0.042205/0.063736`，均有限且非零，排除结构未生效。
- 该 head-only 帧差包络结构在 epoch 4 同时提高 HOTA、DetA、AssA 与 AP，
  且 det 侧早期增益强于 `0731_01`；继续到 epoch 8，重点判断是否避免既有
  decoder 候选的中期检测覆盖退化。

## 2026-07-31 03:12 CST 0731_04 epoch-4 全门槛

- cls HOTA/DetA/AssA 为 `36.831/27.861/52.246`，det 为
  `43.581/34.573/56.147`；相对 `0727_01` epoch 4，cls 三项分别
  `+0.622/+0.793/+0.152`，det 三项分别 `+4.828/+2.119/+8.681`。
- pair mAP/AP50 为 `0.161207/0.315535`，both-independent mAP/AP50 为
  `0.189718/0.346363`；相对父配置 pair mAP 和 both-independent AP50 分别
  `+0.003954/+0.023214`，AP 保护线通过。
- checkpoint 结构审计确认 enveloped-detail 三层权重
  `0.063819/0.046991/0.041138`，common-evidence-bypass 三层权重
  `0.025945/0.037553/0.027217`，全部有限非零。
- 正交组合在 epoch 4 通过完整门槛并继续到 epoch 8；其 det 侧早期增益为当前
  四路候选最高，但 cls 增益低于 `0731_01/02`，中期需重点观察共同证据是否再次
  造成检测覆盖退化。

## 2026-07-31 03:16 CST 0731_03 停止与 0731_05 接替

- `0731_03 common-evidence-bypass` 的 cls HOTA/DetA/AssA 为
  `36.564/27.324/52.415`，det 为 `43.279/34.694/55.415`；HOTA 与 DetA
  均高于父配置，结构检查也确认三层门控最大权重
  `0.030437/0.037983/0.040347`，均有限非零。
- pair mAP/AP50 为 `0.153565/0.313624`，both-independent mAP/AP50 为
  `0.182670/0.341695`。pair mAP 相对父配置下降 `0.003688`，超过保护线
  `0.000688`，因此完成全部 artifacts 后精确停止，252 GPU 0/1 已释放。
- 接替候选 `0731_05 shared-attention + enveloped-detail` 不再引入已失败的
  common-evidence bypass，而是组合 `0730_13` 的共享聚合几何与 `0731_02`
  的受包络帧细节。两者作用位置不同：前者约束 cross-attention 的权重预测，
  后者只调整逐帧 head 输入；recurrent query 与训练协议保持父配置。

## 2026-07-31 03:25 CST 0731_05 正式启动

- 79 项 decoder 单测全部通过；新增组合测试确认零起点时逐元素等于
  shared-attention 父路径，并保持 6 组 attention 参数共享、18 组
  sampling/value/output 参数独立。
- 正式配置和 4-iter smoke 配置均通过深拷贝与完整模型构建。双卡真实数据 smoke
  的四步总/DN/encoder loss 与 grad norm 均有限；checkpoint 中共享 attention
  最大误差为零、独立参数最大差异 `0.000785`，三层 enveloped-detail 最大权重
  `0.000396/0.000393/0.000394`。
- 03:23 在 252 GPU 0/1 fresh 启动，03:25 达到 epoch 1 iter 50：
  `1.1589 s/iter`、loss `21.4509`、grad norm `104.0930`，两卡约
  `19.2 GiB` 且 100% 利用，无 Traceback、OOM、NaN、NCCL 或未使用参数错误。
  五项正式启动门槛通过，首判 epoch 4。

## 2026-07-31 03:40 CST 0731_01 epoch-8 门控与 0731_06

- `0731_01 shared-attention + antisymmetric-detail` 的 epoch 8 checkpoint、检测、
  完整 TrackEval、原始 CSV 和结构审计均已完成。cls HOTA/DetA/AssA 为
  `45.152/37.611/57.666`，det 为 `50.817/46.745/57.206`；pair mAP/AP50
  `0.246292/0.441537`，both-independent mAP/AP50 `0.284940/0.479887`。
- 相对 `0727_01` 同点，det HOTA 提高 `0.624`，pair mAP 与 both AP50 分别提高
  `0.008558/0.013936`，但 cls HOTA 下降 `0.117`。cls/det DetA 只下降
  `0.052/0.316`，均在保护线内；因此结论不是检测崩坏，而是分类与逐帧 detail
  耦合造成的轻微 cls 退化。按预先固定的“双 HOTA 不低于父配置”门槛精确停止。
- checkpoint 中 6 组共享 attention 最大误差为零，18 组独立参数最大差异
  `0.060115`；三层 antisymmetric-detail 权重范数
  `0.033877/0.031303/0.031082`，排除结构未生效。
- 接替候选 `0731_06` 保留 shared-attention 的分类共享特征，只把零起点、
  受观测帧差逐元素包络约束的 swap-odd correction 注入两帧框回归及 reference
  更新。它是分类/几何作用路径的结构性解耦，不是 scale、loss 权重或类别重权调整。

## 2026-07-31 03:47 CST 0731_06 正式启动

- 80 项 decoder 单测、正式/烟测配置深拷贝、完整模型构建和 launcher 审计通过。
  178 单卡真数据 4-iter smoke 的四步 loss 为
  `21.3239/20.1028/20.0323/20.0772`，总、DN、encoder loss 与 grad norm 均有限。
- smoke checkpoint 中 6 组共享 attention 最大误差为零，18 组逐帧独立参数最大差异
  `0.000779`；三层 regression-only enveloped-detail 门控最大权重
  `0.000389/0.000389/0.000390`，确认新回归路径获得真实梯度。
- 03:45 在 178 GPU 0 fresh 启动；03:46 到 epoch 1 iter 50：
  `0.9610 s/iter`、loss `20.9556`、grad norm `111.7257`，显存约
  `31.4 GiB`，无 Traceback、OOM、NaN、NCCL 或未使用参数错误。五项正式启动门槛
  通过，四机结构实验重新并行，首判 epoch 4。

## 2026-07-31 04:35 CST epoch-8 淘汰与分类专用细节

- `0731_02` epoch 8：cls HOTA/DetA/AssA `45.128/37.117/57.200`，det
  `50.223/44.663/58.189`，pair mAP/AP50 `0.236938/0.434026`，
  both-independent mAP/AP50 `0.275284/0.473473`。cls HOTA、cls DetA 和
  det DetA 均触发淘汰线，说明 full-path bounded detail 仍会伤害检测覆盖。
- `0731_04` epoch 8：cls `44.760/36.328/57.530`，det
  `49.732/44.183/57.992`，pair mAP/AP50 `0.226459/0.424496`，
  both-independent `0.263142/0.463058`。共同证据旁路没有修复问题，反而使
  pair mAP 相对父配置下降 `0.011275`。完整结果落盘后停止；误继续的 epoch 9
  迭代不参与结论。
- 新增 `classification_enveloped_detail_decoder`：frame correction 只作用于
  `hidden_states_prev/curr` 的分类分支，regression heads 与 reference updates
  均使用共享 `layer_output`。零起点时输出逐元素等于父路径；非零时 reference
  仍与父路径一致；门控在首步即可获得分类梯度。
- 82 项单测通过。99 `0731_07` 是无 shared-attention 的分类专用版本；197
  `0731_08` 是 shared-attention + 分类专用版本。两者的真数据 DDP smoke、
  checkpoint 门控审计和正式 iter-50 门槛全部通过，提交为 `7dee533`。
- 现有四路形成同一问题的可归因结构矩阵：252 full-path、178 regression-only、
  99 classification-only、197 shared-attention + classification-only。后续优先看
  det DetA 是否由分类隔离得到保护，以及 197 是否在保留 shared-attention det
  增益的同时把 cls HOTA 拉回父配置以上。

## 2026-07-31 04:55 CST 0731_05 epoch-4 门控

- 252 `0731_05 shared-attention + full-path enveloped-detail` 的 epoch 4
  checkpoint、检测、完整 TrackEval、原始 CSV 和组合结构检查均已收齐。
  cls HOTA/DetA/AssA 为 `36.593/27.412/52.831`，det 为
  `43.208/34.366/55.990`；相对父配置分别为 HOTA `+0.384/+4.455`、
  DetA `+0.344/+1.912`、AssA `+0.737/+8.524`。
- pair mAP/AP50 为 `0.1617/0.3118`，both-independent mAP/AP50 为
  `0.1910/0.3450`；相对父配置 pair mAP 和 both-independent AP50
  分别提高约 `0.0044/0.0219`，不存在由关联分数掩盖的检测覆盖下降。
- checkpoint 中 6 组共享 attention 最大误差为零、18 组逐帧独立参数最大差异
  `0.036059`；三层 enveloped-detail 门控最大权重
  `0.063013/0.045573/0.054088`。结构确已生效且所有固定门槛通过，
  因此保留训练继续到 epoch 8。
- 178 `0731_06 shared-attention + regression-only enveloped-detail` 的 epoch 4
  cls HOTA/DetA/AssA 为 `37.693/27.731/54.370`，det 为
  `40.205/32.703/50.871`；相对父配置 HOTA `+1.484/+1.452`、
  DetA `+0.663/+0.249`、AssA `+2.276/+3.405`。pair mAP `0.1700`、
  both-independent AP50 `0.3494`，检测代理同步上升。结构检查、完整 TrackEval
  和原始 CSV 均通过，因此同样保留到 epoch 8。

## 2026-07-31 05:44 CST 0731_08 epoch-4 淘汰

- 197 `0731_08 shared-attention + classification-only enveloped-detail` 的
  epoch 4 checkpoint、检测、完整 TrackEval、原始 CSV 和组合结构检查均已收齐。
  cls HOTA/DetA/AssA 为 `36.312/27.192/52.399`，det 为
  `42.336/33.389/54.669`；双 HOTA 与双 DetA 均不低于父配置。
- pair mAP/AP50 为 `0.153285/0.312909`，both-independent mAP/AP50 为
  `0.182831/0.343821`。pair mAP 相对父配置下降 `0.003968`，超过固定
  `0.003` 保护线 `0.000968`，因此不能因 HOTA/AssA 增益判为通过。
- 6 组共享 attention 最大误差为零，18 组逐帧独立参数最大差异
  `0.035560`；三层 classification-only 门控最大权重
  `0.045196/0.065476/0.063087`。结构确已学习，结论是分类专用细节造成轻微
  pair AP 搬运。05:44 精确停止，GPU 4/5 已释放；等待99 `0731_07`
  同点结果后再选择有因果依据的替代结构。

## 2026-07-31 06:02 CST HOTA 主门槛与四机并行恢复

- 99 `0731_07 classification-only enveloped-detail` 的 epoch 4 完整结果为：
  cls HOTA/DetA/AssA `35.533/26.341/51.534`，det
  `42.384/33.122/55.289`，pair mAP/AP50 `0.151026/0.309108`，
  both-independent mAP/AP50 `0.180451/0.338591`。相对父配置，cls HOTA
  与 cls DetA 分别下降 `0.676/0.727`，pair mAP 下降 `0.006227`；
  虽然 det HOTA 提高 `3.631`，但未满足双 HOTA 主目标，完成全部 artifacts
  后于 05:50 精确停止。
- 决策门槛调整为：cls HOTA 与 det HOTA 是一级目标，DetA/AssA 用于解释增益；
  pair mAP 与 both-independent AP50 降为诊断项。小于约 `0.005` 的单点 AP
  波动不再单独淘汰；只有 AP 明显、持续下降并与 HOTA/DetA 同向恶化时才参与停止。
  最终论文目标仍是 decoder 的 cls HOTA 与 det HOTA 同时超过 encoder
  `0727_01` 的 `54.437/62.393`。
- 按新门槛重新审视，197 `0731_08` 在 epoch 4 的双 HOTA 与双 DetA 均高于父配置，
  pair mAP 仅下降 `0.003968`，不应淘汰。06:00 已从原
  `epoch_4.pth` 原位恢复；日志确认 resumed epoch 4、iter 4152，并已到 epoch 5
  iter 100，GPU 4/5 正常。
- 99 接替为 `0731_09`，即 178 `0731_06 regression-only` 的双卡 2xb4
  复现。84 项 decoder 单测、完整模型构建、双卡真数据 4-iter smoke 与 checkpoint
  结构检查全部通过；smoke 的三层门控均获得非零更新。06:02 在 GPU 0/1 fresh
  启动。此时 252 `0731_05`、178 `0731_06`、197 `0731_08`、99 `0731_09`
  四路正式结构实验恢复并行。
- 新的 `midpoint_regression_enveloped_detail_decoder` 已完成实现与测试：把帧细节
  转换为 5D box-logit residual 空间的一对严格反对称修正，使新增细节的 pair
  midpoint 精确为零，分类路径继续共享。它是结构改动，不含 scale、loss 重权或
  类别重权；当前排队等待下一可用资源，不抢占上述四条 HOTA 主线。

## 2026-07-31 06:08 CST 0731_06 epoch-8 淘汰与 0731_11 接替

- 178 `0731_06 regression-only enveloped-detail` 的 epoch 8 完整结果为：
  cls HOTA `44.398`，det HOTA `48.552`；相对 `0727_01` 同点
  `45.269/50.193` 分别下降 `0.871/1.641`。pair mAP `0.221531`、
  both-independent AP50 `0.430058`，也分别低于父配置
  `0.237734/0.465951`。
- checkpoint、检测 metrics、完整 TrackEval、原始 CSV 与结构审计均已完成。
  6 组共享 attention 误差为零，18 组独立参数最大差异 `0.060327`，三层
  detail gate 最大权重 `0.087576/0.072578/0.059405`。结构确实学习但双 HOTA
  同时下降，故淘汰；异步评估期间训练误入 epoch 9，06:08 精确停止，epoch 9
  不参与任何结论。
- 178 接替实验定为 `0731_11 shared-attention +
  midpoint-regression-enveloped-detail`。它针对 0731_06 的关键缺陷：
  两个独立非线性回归头会把特征域的 `-detail/+detail` 映射成不对称的框残差；
  新结构在框 logit residual 空间显式构造 `-box_detail/+box_detail`，保证新增
  细节不移动 pair midpoint。完成配置构建与真数据 smoke 后 fresh 启动。
- `0731_11` 已通过完整模型构建、84 项 decoder 单测（代码级）、178 真数据
  4-iter smoke 与 checkpoint 结构检查。smoke 的 6 组共享 attention 误差为零，
  18 组独立参数最大差异 `0.000786`，三层 detail gate 最大权重
  `0.000389/0.000390/0.000390`。06:11 fresh 启动，06:12 达到 epoch 1
  iter 50：`0.9405 s/iter`、loss `20.9608`、grad norm `109.0985`，
  GPU0 约 `31.4 GiB`，无 Traceback/OOM/NaN/NCCL。

## 2026-07-31 06:17 CST 0731_05 epoch-8 双 HOTA 通过

- 252 `0731_05 shared-attention + full-path enveloped-detail` 的 epoch 8
  checkpoint、检测 metrics、完整 2/2 TrackEval、原始 CSV 与结构审计均已完成。
  cls HOTA/DetA/AssA 为 `45.341/37.690/56.836`，det 为
  `51.589/45.176/60.817`。
- 相对 `0727_01` epoch 8，cls HOTA `+0.072`、det HOTA `+1.396`，
  因而这是当前首个在 epoch 8 同时超过父 encoder 两项 HOTA 的 decoder 候选。
  但 cls 优势很窄；cls DetA `+0.027`、AssA `-0.465`，det DetA
  `-1.885`、AssA `+5.672`，说明 det HOTA 增益主要来自关联，而非检测覆盖全面增强。
- pair mAP `0.2380`、both-independent AP50 `0.4738`，相对父配置约
  `+0.0003/+0.0078`，没有 AP 崩塌。checkpoint 中 6 组共享 attention 误差为零，
  18 组独立参数最大差异 `0.059736`，三层 detail gate 最大权重
  `0.099468/0.084183/0.087388`，结构确已学习。
- 按 HOTA 主门槛保留继续训练。该结果只证明 epoch 8 的阶段性双提升，最终仍必须
  同时超过 `54.437/62.393`；后续重点观察 cls 微弱优势能否保持，以及 det DetA
  是否回升，不能把单独 AssA 搬运写成完整检测能力提升。

## 2026-07-31 06:35 CST terminal-only decoder detail

- 当前四路训练均稳定：99 `0731_09` epoch 2、197 `0731_08` epoch 7、252
  `0731_05` epoch 10、178 `0731_11` epoch 2。同步前后日志连续，不存在重启或
  checkpoint 轨迹改变。
- 新候选只在最终 decoder 层把 bounded swap-odd detail 送入逐帧 cls/reg heads；
  前两层分类输出、框更新和下一层 reference 均严格留在 shared-attention 父路径。
  这直接检验“递归污染”而非缩小残差强度，因此不是 residual-scale 或超参数扫描。
- 单测覆盖零起点全输出等价、非零时前两层 hidden/reference 逐元素不变、最终层确实
  分化、首步梯度进入唯一 terminal gate，以及 detector 初始化后门控重新归零。
  完整 86 项 decoder 测试、模型构建和初始化检查通过，代码提交为 `764ff7d`。
- 独立 checkpoint 验收器同时要求 6 组 attention 权重严格共享、18 组
  sampling/value/output 参数已分化、唯一 terminal gate 有限非零；合成 checkpoint
  正向验收通过。正式 smoke 仍必须用真实 `iter_4.pth` 再执行同一验收器。
- 该候选尚未创建正式配置、workdir 或实验编号；只有现有 midpoint/full-path
  路径在完整 HOTA 证据下需要接替时才部署，避免把预案误报为正在运行的实验。

## 2026-07-31 07:25 CST classification-only 淘汰与 midpoint 主线

- `0731_08` epoch 8 的 cls/det HOTA 为 `43.801/49.318`，相对 encoder 同点
  下降 `1.468/0.875`；双 DetA 和 AP 同向明显下降。分类专用细节只隔离了
  reference 直接污染，却仍通过分类分数改变匹配与筛选，因此无法保护检测覆盖。
- 197 已切换到 `0731_10 midpoint-regression` 双卡复现。该结构把反对称约束放到
  最终框 residual 空间，不靠 residual scale、loss 重权或类别重权。真实数据 smoke
  和正式 iter 100 的全部门槛通过。
- 99 `0731_09 regression-only` 与 178 `0731_11 midpoint-regression` 的 epoch 4
  都在 HOTA、DetA、AssA 和 AP 上同步提高；其中 midpoint 的 cls/det HOTA 为
  `38.668/43.586`，早期覆盖增益最完整。两者均继续到 epoch 8。
- 当前结构判断是：直接分类细节不可取；回归细节必须进一步约束 pair midpoint。
  252 full-path 仍是 epoch 8 阶段最强，midpoint 路径是更有希望避免中期
  reference/覆盖退化的接替主线。

## 2026-07-31 07:45 CST 0731_05 epoch-12 近门槛结果

- epoch 12 的 cls HOTA/DetA/AssA 为 `50.171/42.372/61.238`，det 为
  `56.430/49.540/66.445`；父 encoder 同点分别为
  `49.680/41.360/61.810` 与 `56.541/50.345/65.579`。
- 结果不是严格双提升：cls HOTA `+0.491`，det HOTA `-0.111`。其机制仍是
  cls 检测覆盖增强，而 det 侧以 AssA `+0.866` 抵消 DetA `-0.805`，说明
  full-path detail 的关联收益与覆盖损失尚未完全解耦。
- 与 epoch 8 相比，det DetA 相对差距从 `-1.885` 收窄到 `-0.805`；
  pair mAP `0.276140` 和 both-independent AP50 `0.534695` 也未显示检测崩塌。
  因此不把 `-0.111` 的单点差距当作系统性失败，继续到 epoch 16 验证趋势。
- 最终标准保持不变：必须同时超过 encoder 最终 `54.437/62.393`。若 epoch 16
  仍未双超越同点，则停止 full-path 路径，252 转入已验证代码的 terminal-only
  结构；99/178/197 的 regression-only 与 midpoint 实验继续独立判定。

## 2026-07-31 07:58 CST terminal-only 配置准备

- `0731_12` 已补齐面向 252 的正式 2xb4 配置、4-iter 真数据 smoke 配置及两套
  launcher。配置显式关闭 full-path、classification-only、regression-only 与
  midpoint detail，只保留 shared attention 和唯一 terminal detail。
- 配置深拷贝、完整模型构建、launcher 语法和 detector 初始化后的零门控均通过；
  252 上两个目标目录均不存在。该状态不代表实验已启动，当前不占用任何 GPU。
- 预先准备只缩短失败接替时间，不改变决策顺序：继续等待 99/178 epoch 8、
  197 epoch 4 和 252 epoch 16 的完整 HOTA；只有证据支持且资源释放后才运行 smoke。

## 2026-07-31 09:20 CST full-path 淘汰与 terminal-midpoint 接替

- `0731_05` epoch 16 cls HOTA/DetA/AssA 为
  `51.007/42.964/62.130`，det 为 `57.940/50.762/68.403`；相对父 encoder
  同点 HOTA `-0.084/-0.380`。其 mAP/AP50 提升只能说明检测分数诊断未崩塌，
  不能替代双 HOTA 主目标。按预设门槛在 09:07 精确停止，epoch 17 不参与结论。
- `0731_09 regression-only` epoch 8 cls/det HOTA
  `44.043/49.271`，相对父配置 `-1.226/-0.922`，证明不约束 box midpoint
  的全层回归细节会在中期同时损伤两个 HOTA，完成结构审计后停止。
- `0731_10` epoch 4 双 HOTA `38.794/44.142`，相对父配置
  `+2.585/+5.389`，继续到 epoch 8。`0731_11` epoch 8 为
  `45.173/51.263`，相对父配置 `-0.096/+1.070`；保留到 epoch 12，验证
  midpoint 约束能否在后期恢复 cls 的极窄差距。
- 99 的 `0731_12 terminal-only` 已正式运行，专门验证“去掉递归污染”是否足够。
  252 则运行结构更严格的 `0731_13 terminal-midpoint`：分类只在末层专门化，
  末层回归额外保证两帧新增 box-logit residual 之和严格为零。两者形成可解释的
  结构对照，而不是 residual scale 或 loss/类别重权扫描。
- `0731_13` 通过 4 项针对性单测、两份配置深拷贝和完整模型构建、双 launcher
  语法检查、252 真数据双卡 4-iter smoke 与 checkpoint 验收。smoke 确认
  6 组 attention 严格共享、18 组采样/投影参数保持独立、唯一 terminal gate
  有限非零。09:18 fresh 启动，09:19 正式 iter 50 的 loss `22.1957`、
  grad norm `115.9534`，总/DN/encoder loss 全部有限，正式状态为 `RUNNING`。

## 2026-07-31 09:50 CST midpoint 淘汰与 HOTA 优先恢复

- `0731_10 midpoint-regression` epoch 8 已形成 checkpoint、检测 metrics、
  完整 TrackEval 和原始 CSV。cls HOTA/DetA/AssA 为
  `45.254/36.908/58.207`，det 为 `50.846/44.718/59.651`；相对 encoder
  同点 HOTA 为 `-0.015/+0.653`。cls 的极窄差距不足以单独判死，但也不是双过。
- `0731_11` 的同结构单卡轨迹继续到 epoch 12 后，cls HOTA/DetA/AssA 为
  `49.478/40.961/62.396`，det 为 `56.451/50.351/65.494`；相对 encoder
  同点 HOTA `-0.202/-0.090`。epoch 8 的 det 优势消失，且 cls 差距扩大，
  因而不再把 197 的重复轨迹继续到 epoch 12；两项在 09:47 精确停止。
- 用户已明确将 cls/det HOTA 作为推进主指标，并放松 mAP。按这一新规则重新审视
  历史结构，`0730_09 motion-trust` epoch 8 曾相对 encoder 双升
  `+0.229/+0.967`，`0730_16 antisymmetric-detail` epoch 4 曾双升
  `+0.475/+0.468`；它们此前均因次要 DetA/mAP 保护线停止，不是 HOTA 失败。
- 197 于 09:48 从 `0730_09 epoch_8.pth` 恢复，09:49 到 epoch 9 iter 50；
  loss `10.0724`、grad norm `40.8031`，总/DN/encoder loss 均有限。178 于
  09:49 从 `0730_16 epoch_4.pth` 恢复，PyTorch 2.6 仅在可信旧 checkpoint
  恢复脚本中显式关闭 weights-only 默认值；09:50 到 epoch 5 iter 50，loss
  `10.3146`、grad norm `40.9181`，无 Traceback/OOM/NaN。
- 当前四路结构互补：99/252 验证 terminal 隔离是否避免递归污染，178 验证纯
  antisymmetric detail，197 验证 motion-trust。下一决策点依次为 99 epoch 4、
  178 epoch 8、252 epoch 4、197 epoch 12；仍以同点 cls/det HOTA 双过为准。

## 2026-07-31 10:04 CST 0731_12 epoch-4 决策

- 99 `0731_12 terminal-only` epoch 4 的 checkpoint、检测 metrics、完整
  TrackEval、原始 CSV 与结构检查均已完成。cls HOTA/DetA/AssA 为
  `37.799/28.644/53.288`，det 为 `43.483/34.398/56.345`。
- 相对 encoder 同点，cls HOTA/DetA/AssA 为 `+1.590/+1.576/+1.194`，
  det 为 `+4.730/+1.944/+8.879`；两种 HOTA 及四个组成量全部提高，
  不属于以检测覆盖换关联的搬运。
- pair mAP `0.165773`，相对父配置同点约 `+0.008520`；
  both-independent AP50 为 `0.355169`。AP 仅作诊断，但这里也没有显示覆盖崩塌。
- checkpoint 结构检查确认 6 组 shared attention 最大误差为零、18 组独立参数
  最大差异 `0.038689`，唯一 terminal gate 最大权重 `0.080521`，模块确已学习。
  该结构明确继续到 epoch 8；10:03 已进入 epoch 5。

## 2026-07-31 10:49 CST 0731_13 epoch-4 决策

- 252 `0731_13 terminal-midpoint` 已完成 epoch 4 checkpoint、检测 metrics、
  完整 TrackEval、原始 CSV 与 checkpoint 结构检查。cls HOTA/DetA/AssA 为
  `36.148/27.020/51.933`，det 为 `43.247/33.958/56.389`。
- 相对 encoder 同点，cls 三项为 `-0.061/-0.048/-0.161`，det 三项为
  `+4.494/+1.504/+8.923`。这不是双 HOTA 通过，但 cls 落后仅 `0.061`，
  det 增益同时来自检测与关联，不属于单纯 AssA 搬运。
- pair mAP `0.153978`、both-independent AP50 `0.335487`，相对父配置约
  `-0.003275/+0.012338`。AP 仅作诊断，当前没有与 det HOTA 同向的覆盖崩塌。
- 结构检查确认 6 组 shared attention 最大误差为零、18 组独立参数最大差异
  `0.037019`，唯一 terminal-midpoint gate 最大权重 `0.081010`。同 epoch
  它比 99 `0731_12 terminal-only` 低 `1.651/0.236`，说明 midpoint 约束没有
  早期互补优势；但末层隔离的核心假设是中期稳定性，因此仅继续到 epoch 8 做最终
  稳定性判定。epoch 8 若仍非双 HOTA 通过，立即停止该方向，不再延长。

## 2026-07-31 11:12 CST 恢复实验收口

- 197 `0730_09 motion-trust` 从旧 epoch 8 恢复后已完成 epoch 12 的 checkpoint、
  检测、TrackEval、原始 CSV 与结构检查。cls HOTA/DetA/AssA 为
  `48.877/40.354/61.309`，det 为 `55.819/48.907/65.923`；相对 encoder
  同点分别为 HOTA `-0.803/-0.722`、DetA `-1.006/-1.438`、AssA
  `-0.501/+0.344`。早期双 HOTA 优势没有延续，故停止并释放 GPU 4/5。
- 197 同点 pair mAP/AP50 为 `0.263196/0.479269`，both-independent
  mAP/AP50 为 `0.300922/0.513250`。三层 motion-trust 权重最大值为
  `0.453352/0.294524/0.256910`，排除结构未学习。恢复评估复用了
  `val_track_0001`，因此当前结果另行保存在
  `val_track_epoch12_resume_20260731`。
- 178 `0730_16 antisymmetric-detail` 从旧 epoch 4 恢复后已完成 epoch 8 的全部
  产物。cls HOTA/DetA/AssA 为 `44.591/37.157/56.398`，det 为
  `49.685/46.466/54.883`；相对 encoder 同点 HOTA `-0.678/-0.508`，
  DetA `-0.506/-0.595`，AssA `-0.903/-0.262`，已是组成量全面退化，故停止。
  pair mAP/AP50 为 `0.241645/0.435966`，both-independent mAP/AP50 为
  `0.279185/0.474553`；三层结构权重最大值
  `0.035530/0.036084/0.034844`。结果另存于
  `val_track_epoch08_resume_20260731`，GPU 0 已释放。
- 共享存储元数据时钟与登录节点存在约 25 分钟偏移，以上两个恢复评估通过
  checkpoint、检测 metrics、TrackEval metrics 和原始 CSV 的实际写入顺序与内容
  交叉确认，不依赖绝对 mtime。99 `0731_12` 已完成 epoch 8 训练，完整 TrackEval
  即将形成；178/197 暂时空闲，等该同点结果决定下一轮结构，避免无依据并行。

## 2026-07-31 11:33 CST terminal-only 淘汰与回归专用结构

- 99 `0731_12 terminal-only` epoch 8 的 cls HOTA/DetA/AssA 为
  `43.178/35.246/55.355`，det 为 `50.010/43.700/59.276`。相对 encoder
  同点 HOTA `-2.091/-0.183`，DetA `-2.417/-3.361`，cls AssA
  `-1.946`，仅 det AssA `+4.131`。早期六个组成量共同提高的轨迹没有保持。
- pair mAP/AP50 为 `0.217536/0.410164`，both-independent AP50
  `0.448156`，相对父配置也明显下降。checkpoint、检测、完整 TrackEval、原始
  CSV 与结构检查均完整；6 组 shared attention 误差为零，18 组独立参数最大差异
  `0.063149`，唯一 terminal gate 最大权重 `0.117809`。结构确已学习，故按
  HOTA 主门槛停止，GPU 0/1 释放。
- 结合历史结果可把失败来源进一步分离：classification-only 在 epoch 8 失败，
  全层 regression-only 也因递归 reference 污染失败，而 terminal-only 同时暴露
  分类和回归细节仍失败。因此下一组不是 scale 调参，而是两个结构对照：
  `0731_14` 只在最终框输出注入细节、分类完全共享；`0731_15` 在此基础上进一步
  令新增 5D box-logit residual 严格反对称并保持 pair midpoint。
- 两个新模式的 89 项 decoder 单测全部通过；其中验证了分类 hidden state 在所有层
  与父路径逐元素相同、前两层 reference 不变、最终框真实分化、midpoint 残差和为零，
  且 detector 初始化后唯一 gate 仍为零。正式与 smoke 配置均通过深拷贝，两个完整
  模型构建通过，四个 launcher 通过 `bash -n`。下一步分别在 99 与 197 做真实双卡
  4-iter smoke，只有 smoke checkpoint 结构验收和有限 loss 全部通过后才正式启动。

## 2026-07-31 11:45 CST 0731_14/15 正式启动

- 提交 `1575d96` 已同步到四台本地服务器；99 与 197 的正式和 smoke workdir
  均相互隔离，启动前为空。99 使用 GPU 0/1，197 仅使用 GPU 4/5。
- `0731_14` smoke 完成 4 个真实训练 iter，iter 4 loss/grad norm
  `19.8231/87.0932`，DN 与 encoder loss 有限。checkpoint 检查为：
  attention share 6 组、最大误差 `0`；18 组独立参数最大差异 `7.85518e-4`；
  terminal gate 最大值 `3.92929e-4`。
- `0731_15` smoke 完成 4 个真实训练 iter，iter 4 loss/grad norm
  `19.9996/94.1014`，DN 与 encoder loss 有限。checkpoint 检查为：
  attention share 6 组、最大误差 `0`；18 组独立参数最大差异 `7.79018e-4`；
  midpoint terminal gate 最大值 `3.92011e-4`。
- 两项正式 fresh 训练均通过五项启动门槛。99 `0731_14` 于 11:45 达到
  epoch 1 iter 50，`0.9528 s/iter`、loss `22.1461`、grad norm
  `126.9356`；197 `0731_15` 同时达到 iter 50，`0.8932 s/iter`、
  loss `22.1811`、grad norm `124.8561`。两者总、DN、encoder loss
  均有限，目标 GPU 各约 `19.2 GiB`，没有训练异常。
- 当前并行实验为 252 `0731_13`、99 `0731_14`、197 `0731_15`；
  178 空闲。三项均以同 epoch cls/det HOTA 为主判据，首判新实验的 epoch 4。

## 2026-07-31 12:15 CST 0731_13 epoch-8 淘汰

- 252 `0731_13 terminal-midpoint` 的 epoch 8 checkpoint、检测 metrics、完整
  TrackEval、原始 CSV 与结构审计均已完成。cls HOTA/DetA/AssA 为
  `43.170/34.514/57.338`，det 为 `48.484/42.910/56.678`。
- 相对 `0727_01` 同点，cls HOTA/DetA/AssA 为
  `-2.099/-3.149/+0.037`，det 为 `-1.709/-4.151/+1.533`。它从 epoch 4
  的 det 全组成量增益退化为明显 DetA 损失，仅保留 AssA 搬运；不是随机小波动，
  也不满足 HOTA 双过。
- pair mAP/AP50 为 `0.212921/0.391949`，both-independent mAP/AP50 为
  `0.248771/0.430138`，检测诊断同样低于父配置。checkpoint 检查确认 6 组
  shared attention 最大误差为零、18 组独立参数最大差异 `0.058077`，
  terminal-midpoint gate 最大权重 `0.119707`，证明结构已充分学习。
- 按预设 epoch-8 门槛于 12:11 对该 workdir 的唯一训练进程组发送 SIGTERM；
  目标进程已完全退出，GPU 0/1 均为 `1 MiB/0%`，epoch 8 checkpoint 与全部
  评估产物保留。当前仅 99 `0731_14` 和 197 `0731_15` 运行，二者均在 epoch 2
  稳定训练并等待 epoch 4 首次 HOTA 判定；178 与 252 空闲。

## 2026-07-31 12:26 CST 按 HOTA 主规则恢复 0731_03

- 重新审计历史 Decoder 后，252 `0731_03 common-evidence bypass` 在 epoch 4
  相对 `0727_01` 同点的 cls/det HOTA 为 `+0.355/+4.526`，DetA 也分别提高
  `+0.256/+2.240`。它此前停止的唯一原因是 pair mAP 下降 `0.003688`，超过旧
  保护线 `0.003`；这与当前“Cls/Det HOTA 为主、mAP 只作诊断”的规则不一致。
- 该结构保持 recurrent decoder query 不变，只在每层预测头之前通过零起点、
  frame-swap invariant 门控恢复两帧共同 cross-attention 证据，属于模型结构改动，
  不是 loss/类别重权或 residual-scale 扫描。三层 gate 在 epoch 4 的最大绝对权重
  为 `0.030437/0.037983/0.040347`，结构检查通过。
- 252 代码已同步到 `17ef55f`。恢复前核验了干净 tracked 状态、`epoch_4.pth`
  的 epoch/优化器状态、检测和完整 TrackEval、配置深拷贝、完整模型构建、
  launcher 语法、数据/GMC/预训练权重及 GPU 0/1 空闲状态。
- 12:24 从可信 `epoch_4.pth` 恢复；日志确认 `resumed epoch: 4, iter: 4152`。
  12:25 到 epoch 5 iter 50，学习率 `1.0e-4`、time `1.1729 s/iter`、
  loss `11.6087`、grad norm `34.1805`；总 loss、DN loss、encoder loss
  均有限，GPU 0/1 各约 `19.4 GiB`，无 Traceback/OOM/NaN/NCCL。
  该轨迹只继续到 epoch 8 做双 HOTA 可持续性判断，不把旧 epoch 5 的部分训练日志
  当作当前恢复轨迹。

## 2026-07-31 12:44 CST 0731_16 terminal common-evidence 正式启动

- 历史对照表明，`0731_03` 的共同证据旁路在 epoch 4 同时提高 cls/det HOTA 与
  DetA，但逐层注入会改变后续 decoder reference。`0731_16` 因此不是调 scale，
  而是把同一结构假设严格隔离到最终输出：保持 recurrent query、全部 auxiliary
  output 和供后续层消费的 reference 与父路径一致，只在最后一层双帧预测头前加入
  swap-invariant、零起点且有界的共同 cross-attention 证据。
- 提交 `8a663bd` 已同步到 99、197、252、178。新增模式通过 93 项 decoder 单测，
  包括零起点严格等价、仅最终输出变化、帧交换不变性/有界性和 gate 梯度；正式及
  smoke 配置深拷贝、完整模型构建和 launcher 语法检查均通过。
- 178 GPU 0 的真实数据 4-iter smoke 最终 loss/grad norm 为
  `20.1619/63.3459`，总、DN、encoder loss 全部有限；`iter_4.pth` 中唯一 terminal
  gate 最大绝对权重 `3.98724e-4`，结构确已获得更新，且没有训练异常。
- 正式 fresh 训练于 12:42 启动，12:43 达到 epoch 1 iter 50：
  `0.9353 s/iter`、loss `21.4237`、grad norm `94.8035`；GPU 0 约
  `31.4 GiB`，总、DN、encoder loss 有限，无 Traceback/OOM/NaN/NCCL。
- 当前四路并行为：99 `0731_14`、197 `0731_15`、252 恢复的 `0731_03`、
  178 `0731_16`。前三者分别等待 epoch 4/4/8 完整结果；`0731_16` 首判
  epoch 4。决策仍以同 checkpoint 的 cls/det HOTA 为主，mAP 只作检测退化诊断。

## 2026-07-31 12:59 CST terminal regression epoch-4 对照

- `0727_01` 同点基准为：cls HOTA/DetA/AssA
  `36.209/27.068/52.094`，det `38.753/32.454/47.466`。
- 99 `0731_14 terminal regression-only` 达到 cls
  `37.209/27.864/53.684`、det `43.646/34.679/56.188`；相对基准双 HOTA
  `+1.000/+4.893`、双 DetA `+0.796/+2.225`。pair mAP/AP50
  `0.163485/0.323237`，both-independent mAP/AP50
  `0.192363/0.351268`，检测诊断也全面提高。
- 197 `0731_15 terminal midpoint-regression` 达到 cls
  `38.153/27.923/55.246`、det `44.506/34.409/59.052`；相对基准双 HOTA
  `+1.944/+5.753`、双 DetA `+0.855/+1.955`。pair mAP/AP50
  `0.161120/0.318966`，both-independent mAP/AP50
  `0.189558/0.346678`。
- 两项 checkpoint 均确认 6 组 shared attention 严格共享、18 组独立参数已分化、
  唯一 terminal gate 已获得非零更新；完整 TrackEval 与原始 CSV 齐全。两者均
  继续到 epoch 8。197 当前 HOTA 更强，99 当前 DetA 保持更强；后续以是否出现
  DetA 向 AssA 搬运作为中期定性分界，而最终成功标准仍是
  cls/det HOTA 同时超过 `54.437/62.393`。

## 2026-07-31 14:08 CST 三条旧路径的 epoch-8 收口

- 252 `0731_03 all-layer common-evidence bypass` 的 epoch 8 cls/det HOTA 为
  `44.798/50.415`，相对 `0727_01` 同点为 `-0.471/+0.222`。cls
  DetA/AssA 分别变化 `-1.030/-0.209`，det 分别变化 `-2.602/+3.680`。
  虽然 det HOTA 仍略高，但 cls 未通过，而且 det 已明显依靠 AssA 补偿 DetA；
  checkpoint、检测、完整 TrackEval、原始 CSV 与结构检查齐全后停止，GPU 0/1
  释放。
- 99 `0731_14 terminal regression-only` 的 epoch 8 cls/det HOTA 为
  `44.321/49.059`，相对父配置 `-0.948/-1.134`；cls DetA/AssA 变化
  `-1.901/+0.232`，det 变化 `-3.480/+1.981`。197 `0731_15 terminal
  midpoint-regression` 的 epoch 8 为 `43.918/49.071`，相对父配置
  `-1.351/-1.122`；cls DetA/AssA 变化 `-2.925/+0.607`，det 变化
  `-4.146/+2.977`。
- 两个独立 terminal regression 对照都从 epoch 4 的双 HOTA、双 DetA 增益退化为
  epoch 8 的 DetA 系统性下降，且检测 AP 诊断也同步下降。两项结构检查均确认模块
  已学习，因此不是初始化未生效或随机小波动；均已停止并保留 epoch 4/8 全部产物。
  定性结论是：只消除 recurrent reference 污染仍不够，末层 box 专门化本身就会在
  中期破坏检测覆盖。

## 2026-07-31 14:38 CST 严格正交结构接替

- 178 `0731_16 terminal common-evidence bypass` 的 epoch 4 cls
  HOTA/DetA/AssA 为 `37.750/26.503/57.185`，det 为
  `43.723/31.613/62.366`。相对父配置 HOTA `+1.541/+4.970`，但 DetA
  `-0.565/-0.841`，AssA `+5.091/+14.900`。它通过 HOTA 主门槛但已经显示明显
  关联搬运，故只继续到 epoch 8 做持续性判定；14:38 已进入 epoch 8 iter 100，
  尚未形成 epoch 8 checkpoint 或评估。
- `0731_17` 最初把分类共同证据与反对称框细节分离，并通过 98 项单测、完整构建和
  2 卡真实数据 smoke；正式训练到 epoch 1 iter 300 时复核 forward，发现框预测仍以
  `common_output` 为基底，因此共同证据仍能改变 boxes，不满足“严格正交”定义。该
  run 在首个正式 epoch 完成前停止，只保留为实现审计记录，不进入科学结果。
- 提交 `ad99b0d` 修正为：分类只使用 `common_output`，框以原始
  `layer_output` 为基底且只接收反对称 detail；单测证明任意非零 common gate 下所有
  box references 与父模型逐位相等，detail midpoint 为零。252 `0731_18` 已通过
  98 项单测、配置深拷贝、完整构建、2 卡 4-iter smoke 和正式训练五项门槛；
  14:38 到 epoch 2 iter 50，loss/grad norm `12.7175/23.6045`。
- 提交 `ac9d629` 新增更纯的 classification-only common-evidence 模式：只改变末层
  classification，boxes、auxiliary output 与 recurrent references 全部严格等于父模型；
  同时覆盖 shared/non-shared attention。100 项单测、两份配置深拷贝、完整构建和两台
  2 卡真实数据 smoke 全部通过。99 `0731_19` 使用独立 attention，14:38 到 epoch 1
  iter 400；197 `0731_20` 使用 shared attention，14:38 到 epoch 1 iter 250。
- 四项当前正式训练均无 Traceback/OOM/NaN/NCCL，总、DN、encoder loss 与 grad norm
  有限。四台仓库 tracked HEAD 均为干净的 `ac9d629`；已启动进程仍按各自启动时载入的
  代码运行，178 为 `d78500d`、252 为 `ad99b0d`，后续仓库快进不改变运行时语义。

## 2026-07-31 15:03 CST 0731_16 epoch-8 决策

- epoch 8 完整对比为：cls HOTA/DetA/AssA
  `43.972/32.412/62.419`，det `49.378/39.704/63.985`；相对
  `0727_01` 同点 HOTA `-1.297/-0.815`，DetA `-5.251/-7.357`，
  AssA `+5.118/+8.840`。
- pair mAP/AP50 `0.205329/0.366688`，both-independent mAP/AP50
  `0.236540/0.395712`，四项分别低于父配置
  `0.032405/0.064222/0.039642/0.070239`。54 个 TrackEval 原始文件、
  checkpoint、检测与 TrackEval metrics 均完整。
- terminal-common gate 已学到 `0.032061`，所以失败不是模块未更新。该结构把共同
  证据同时送入分类和框，即使隔离到末层仍会造成严重 DetA→AssA 搬运；15:03 已精确
  停止并释放 178 GPU0。后续不再派生“共同证据直接改框”的 decoder。

## 2026-07-31 15:14 CST 0731_21 独立 attention 因子对照

- `0731_18/19/20` 已覆盖 shared/non-shared attention 与 classification common
  evidence，但缺少 non-shared attention 下是否加入 antisymmetric box detail 的
  因子单元。`0731_21` 补齐该严格结构对照，不修改 loss、类别权重、residual scale、
  encoder、proposal、PairDN 或训练协议。
- 提交 `a77e135` 将 factorized evidence 与 shared attention 解耦；新增测试证明在
  attention-weight 参数独立时，共同证据仍完全不改变 boxes，detail 对 box-logit
  的新增 pair midpoint 仍严格为零。100 项 decoder 单测、配置深拷贝、完整构建及
  178 目标环境预检全部通过。
- 真实单卡 4-iter smoke 的最终 loss/grad norm `19.8328/67.2125`；6 组
  attention-weight 最大差异 `7.67466e-4`，common/detail gate 分别获得
  `3.97146e-4/3.92367e-4` 的非零更新。15:13 fresh 正式启动，15:14 到
  epoch 1 iter 50，loss `20.9349`，总、DN、encoder loss 有限，GPU0 正常，
  无训练异常；GPU1 未触碰。

## 2026-07-31 15:50 CST 严格因子对照首批 epoch-4 结果

- 252 `0731_18 shared attention + classification common + antisymmetric box
  detail` 的 cls HOTA/DetA/AssA 为 `37.315/26.954/55.487`，det 为
  `41.743/31.826/56.276`。相对 `0727_01` 同点，双 HOTA 为
  `+1.106/+2.990`，DetA 为 `-0.114/-0.628`，AssA 为
  `+3.393/+8.810`。pair mAP/AP50 `0.158907/0.304586`，
  both-independent mAP/AP50 `0.185215/0.329253`，检测诊断均提高。
- checkpoint 审计确认 6 组 shared attention 严格相同、18 组独立参数最大差异
  `0.033962`，common/detail gate 最大权重 `0.031387/0.090002`。因此双 HOTA
  增益并非模块未学习造成的偶然等价；但 box detail 当前主要表现为 AssA 增益并伴随
  轻度 DetA 损失，继续到 epoch 8 检查是否重现历史中期搬运。
- 99 `0731_19 independent attention + classification common only` 的 cls
  HOTA/DetA/AssA 为 `36.810/27.099/53.921`，det 为
  `43.194/33.533/56.890`；相对父配置双 HOTA `+0.601/+4.441`，DetA
  `+0.031/+1.079`，AssA `+1.827/+9.424`。pair mAP/AP50
  `0.156129/0.309147`，both-independent mAP/AP50
  `0.183573/0.334282`。唯一 common gate 已学到 `0.034268`。
- 两项均通过当前双 HOTA 主门槛并继续到 epoch 8。现阶段 `0731_19` 的检测组成量
  更干净，表明在独立 attention 下仅给分类恢复共同证据，可能比同时向 boxes 注入
  反对称 detail 更能保护 DetA；该判断仍需 `0731_20/21` 的另两个因子单元和四项
  epoch-8 结果共同确认。

## 2026-07-31 16:38 CST 2×2 epoch-4 因子分析完成

- 197 `0731_20 shared attention + classification common only` 的 cls
  HOTA/DetA/AssA 为 `37.345/28.037/52.923`，det 为
  `43.627/34.822/56.234`。相对 `0727_01` 同点，双 HOTA
  `+1.136/+4.874`、DetA `+0.969/+2.368`。pair mAP/AP50
  `0.156468/0.318016`，both-independent mAP/AP50
  `0.185581/0.346145`。checkpoint 中 common gate `0.033425`，
  6 组 attention 严格共享，18 组其余逐帧参数最大差异 `0.027972`。
- 178 `0731_21 independent attention + classification common +
  antisymmetric box detail` 的 cls HOTA/DetA/AssA 为
  `38.031/29.960/52.258`，det 为 `41.072/36.731/47.129`。相对父配置
  双 HOTA `+1.822/+2.319`、DetA `+2.892/+4.277`；pair mAP/AP50
  `0.174049/0.350164`，both-independent mAP/AP50
  `0.205622/0.379277`，均为四个单元中最强的检测覆盖改善。独立 attention
  最大差异 `0.036236`，common/detail gate `0.033932/0.054266`，
  严格结构审计通过。
- 四单元 epoch-4 排列为：
  - independent + classification common only（`0731_19`）：
    HOTA `36.810/43.194`；
  - shared + classification common only（`0731_20`）：
    HOTA `37.345/43.627`；
  - independent + common/detail factorized（`0731_21`）：
    HOTA `38.031/41.072`；
  - shared + common/detail factorized（`0731_18`）：
    HOTA `37.315/41.743`。
- 因子不是简单可加。classification-only 条件下，共享 attention 带来
  cls/det HOTA `+0.535/+0.433` 与 DetA `+0.938/+1.289`；factorized
  条件下，共享 attention 却使 cls/det DetA `-3.006/-4.905`。独立 attention
  下加入 antisymmetric detail 虽使 cls/det DetA `+2.861/+3.198`，却使
  det AssA `-9.761`、det HOTA `-2.122`。因此当前不扩展更多组合，四项继续到
  epoch 8，以双 HOTA 是否保持和 DetA/AssA 是否发生中期搬运决定方向。

## 2026-07-31 17:39 CST epoch-8 结论与梯度隔离对照

- `0731_19` 在 epoch 8 的 cls/det HOTA 为 `43.480/50.152`，相对
  `0727_01` 同点 `-1.789/-0.041`；`0731_18` 为 `43.681/51.660`，
  相对父配置 `-1.588/+1.467`。两者都出现系统性 DetA 损失且 cls 未通过，
  完整产物与结构审计齐全后已停止。
- `0731_21 independent attention + factorized evidence` 在 epoch 8 达到
  cls HOTA/DetA/AssA `46.642/38.691/59.237`，det
  `52.107/46.230/60.830`；相对父配置双 HOTA `+1.373/+1.914`。
  pair mAP/AP50 与 both-independent mAP/AP50 也分别提高
  `0.010728/0.035152` 和 `0.010185/0.033446`。这说明独立 attention 下的
  common/detail 正交分解在中期仍有净增益，目前继续训练。
- 失败单元暴露的关键结构问题不是 boxes 前向本身，而是 common/detail gate 输入
  evidence 仍把分类或回归梯度送回两帧共享 decoder 表征。提交 `2c8b13b` 增加
  `terminal_detach_gate_evidence`：仅停止 gate evidence 输入的反向路径，
  residual/detail 已有停止梯度保持不变，gate 参数仍正常学习。
- 99 `0731_22` 对照 `0731_19`，252 `0731_23` 对照 `0731_21`，构成
  “attached/detached × classification-only/factorized”的严格结构对照。两项
  均通过 103 项 decoder 单测、配置深拷贝、完整模型构建、真实双卡 4-iter smoke
  和正式 iter 50 五项门槛；首个 HOTA 判定点为 epoch 4。

## 2026-07-31 18:33 CST 0731_20 epoch-8 淘汰与梯度解释纠正

- 197 `0731_20 shared attention + classification common only` 的 epoch 8
  checkpoint、检测 metrics、完整 TrackEval、54 个原始结果和结构审计均已完成。
  cls HOTA/DetA/AssA 为 `43.670/34.495/58.446`，det 为
  `49.863/43.705/58.916`；相对 `0727_01` 同点 HOTA
  `-1.599/-0.330`、DetA `-3.168/-3.356`，而 AssA
  `+1.145/+3.771`。pair mAP/AP50 下降 `0.027697/0.027872`，
  both-independent mAP/AP50 下降 `0.031427/0.030375`。
- checkpoint 审计确认唯一 common gate 已学到 `0.052896`，6 组共享
  attention 最大误差为零，18 组其余逐帧参数最大差异 `0.056359`。因此失败是
  中期 DetA→AssA 搬运，不是模块未学习。完整产物保留后于 18:31 精确停止，
  197 GPU 4,5 已释放。
- 对提交前代码的逐行复核发现：`_normalized_shared_evidence()` 从这些实验开始前就以
  `.detach()` 返回，`_normalized_motion_evidence()` 也同样停止梯度。因此
  `terminal_detach_gate_evidence=True` 只是对已 detached 张量再次 detach；
  `0731_22` 与 `0731_19`、`0731_23` 与 `0731_21` 在前向和反向上分别严格等价。
  此前“gate evidence 梯度污染共享 decoder 表征”的解释错误，现正式撤销。
- `0731_22/23` 不再作为科学实验或有效消融。两项均在首个 checkpoint 前于 18:32
  精确停止，日志保留，99 GPU 0,1 与 252 GPU 0,1 已释放。后续不得把它们的
  smoke gate 更新或训练 loss 当成支持梯度隔离假设的证据。

## 2026-07-31 18:55 CST object-reliable factorization 与 epoch-12 复核

- `0731_21` epoch 12 达到 cls HOTA/DetA/AssA `49.159/40.613/61.861`，
  det `56.341/49.482/66.346`。相对 `0727_01` 同点 HOTA 为
  `-0.521/-0.200`，DetA 为 `-0.747/-0.863`，AssA 为 `+0.051/+0.767`。
  epoch 8 的双 HOTA 增益没有在 epoch 12 保持，表现为轻度 DetA→AssA 搬运；但差距
  仍小、pair/both AP50 仍分别提高 `0.006536/0.005902`，故继续至 epoch 16 确认趋势。
- 新结构不再调 loss 或 residual scale，而是用两条父分类路径最大 object confidence 的
  detached 几何均值限制 terminal common/detail 修正只作用于更可信的 object query。
  `0731_24/25/26` 分别只约束 common、只约束 detail、同时约束两者，形成直接定位
  DetA 回落来源的三单元结构对照；无类别重加权、阈值或新增监督。
- 提交 `737210b` 已同步四机。102 项单测、配置深拷贝、完整构建、launcher 语法及
  三机双卡真实数据 smoke 全部通过。99 `0731_24`、197 `0731_25`、252 `0731_26`
  于 18:49 fresh 启动，18:52 均通过正式迭代五项门槛，无 Traceback/OOM/NaN/NCCL；
  首个 HOTA 判定点均为 epoch 4。

## 2026-07-31 19:52 CST 0731_21 epoch-16 与非早停规则

- `0731_21` epoch 16 完整结果为 cls HOTA/DetA/AssA
  `50.273/41.786/62.674`，det `58.085/51.049/68.372`；相对 Encoder 同点
  HOTA `-0.818/-0.235`、DetA `-0.818/-0.665`。pair/both mAP 仅分别低
  `0.000938/0.001007`，AP50 反而高 `0.000355/0.001311`。
- 它继 epoch 12 后第二次小幅双降，但并非 HOTA、DetA、AP 全面崩坏。按用户要求，
  epoch 4 不再用作性能淘汰点，epoch 8/12 主要观察收敛与指标搬运；单点小幅落后不停止，
  有竞争力的结构至少看到 epoch 16/20。故 `0731_21` 继续到 epoch 20。
- 后继模型必须保持简单、可解释和高效：不增加 decoder 深度、额外 attention、高分辨率
  分支或额外 loss；参数、显存基本不变，并在同卡同温条件下比较训练和推理速度。

## 2026-07-31 20:35 CST object-confidence 三路 epoch-4 对照

- `0731_24 confident-common`：cls HOTA/DetA/AssA `36.689/26.771/53.796`，
  det `42.562/32.498/56.889`；相对 Encoder 同点 HOTA `+0.480/+3.809`，
  DetA `-0.297/+0.044`。HOTA 增益主要来自 AssA，mAP 略降但 AP50 提高。
- `0731_26 confident-common+detail`：cls HOTA/DetA/AssA `36.958/26.877/54.801`，
  det `41.792/32.810/54.722`；HOTA `+0.749/+3.039`，DetA `-0.191/+0.356`。
  其 cls HOTA 为三条新实验最高，但仍需验证中期 DetA 是否恢复。
- `0731_25 confident-detail`：cls HOTA/DetA/AssA `35.824/25.781/53.847`，
  det `41.591/30.920/57.109`；HOTA `-0.385/+2.838`，DetA `-1.287/-1.534`，
  pair/both mAP 同时下降，是当前最弱候选。
- 三条结构均为 query 级乘法路由，没有新增可学习参数、decoder 层、attention 或 loss，
  满足复杂度约束。epoch 4 只用于排除灾难性退化，不作性能淘汰；24/26 继续到
  epoch 8/12，25 先到 epoch 8。任何论文主线候选还必须通过同卡同温速度实测，不能
  以跨服务器的训练日志估算效率。

## 2026-07-31 21:02 CST 0731_21 epoch-20 结果

- 完整结果为 cls HOTA/DetA/AssA `51.475/43.145/63.273`，det
  `58.956/51.645/69.703`；相对 Encoder 同点 HOTA `-0.039/+0.034`。
- pair mAP/AP50 提高 `0.003474/0.007421`，both-independent mAP/AP50 提高
  `0.004285/0.008951`，并未出现 AP 搬运式崩塌。与 epoch 16 的双 HOTA
  `-0.818/-0.235` 相比已经明显恢复。
- 严格双提升尚未成立，但该零额外层级的简单结构正在恢复且训练已进入 epoch 21，继续到
  epoch 24 是比 e20 停止更合理的判定；e24 将检验近同点是否转为稳定双增益。

## 2026-07-31 21:47 CST confidence epoch-8 结论与复杂度

- `0731_24 confident-common` e8 为 cls/det HOTA `44.103/50.795`，相对 Encoder
  `-1.166/+0.602`；DetA `-2.648/-2.241`，AssA `+1.807/+4.659`，四项 AP
  下降 `0.017332/0.017646/0.022292/0.021760`。
- `0731_26 confident-common+detail` e8 为 `44.283/50.390`，相对 Encoder
  `-0.986/+0.197`；DetA `-2.028/-3.075`，AssA `+0.648/+4.312`，四项 AP
  也全部下降。两者都明显低于无 confidence 的 `0731_21` e8 `46.642/52.107`。
- 定性结论是双边分类 confidence 过度衰减末层检测修正，造成 DetA→AssA 搬运；两项按宽松
  规则看到 e12，但不再扩展该系列。
- `0731_21` 相对 Encoder 新增 `131,072` 参数，约占模型状态 `0.575%`；confidence
  变体不新增参数，但会多执行末层父分类置信度计算。参数轻量不等于已经证明无速度代价，
  主线候选仍需同卡同温训练和推理测速。

## 2026-07-31 22:19 CST e24、confidence 收口与轻量门控

- `0731_21` epoch 24 达到 cls HOTA/DetA/AssA `52.141/43.566/64.279`，det
  `59.381/51.920/70.336`。相对 Encoder 同点 HOTA 为 `+0.427/-0.138`：cls 已超过，
  det 仍差 `0.138`；pair mAP/AP50 提高 `0.007922/0.015433`，both-independent
  提高 `0.008977/0.017255`。该点不是系统性恶化，继续观察到 epoch 28。
- `0731_25 confident-detail` epoch 8 为 cls/det HOTA `43.629/50.129`，相对
  Encoder 同点 `-1.640/-0.064`，DetA 与四项 AP 全面下降。完整 checkpoint、评估和
  54 个 TrackEval 原始文件确认后已停止，197 GPU 4,5 已释放。结合 `0731_24/26` e8，
  三种 confidence 路由均弱于无 confidence 的 `0731_21`，不再扩展该系列。
- 新建 `0731_27 terminal diagonal factorized evidence`：保留 `0731_21` 的末层
  common/detail 语义、独立 attention、零初始化与严格 box midpoint，仅把两个
  `256×256` 稠密门改成两个 256 维逐通道门。新增参数从 `131,072` 降至 `512`，完整模型
  为 `22,759,287` 参数；不增加 decoder 层、attention、分支或 loss。
- 103 项 decoder 单测、配置构建、双卡真实数据 4-iter smoke 与 checkpoint 结构检查均通过。
  `0731_27` 已在 197 GPU 4,5 fresh 正式启动，并于 22:17 到 epoch 1 iter 50；总损失、
  DN/encoder loss 与梯度均有限，无 Traceback/OOM/NaN/NCCL。首个结构检查点为 epoch 4。
- 后续候选采用联合门槛：先看同点 cls/det HOTA，再核对 DetA/AssA；论文候选还必须在同卡
  同温下证明训练与推理吞吐没有大幅下降。依赖堆叠深度、额外 attention、高分辨率分支或
  新增 loss 才取得的小幅收益不进入主线。

## 2026-07-31 22:35 CST 0731_24 epoch-12 最终结论

- `0731_24 confident-common` e12 为 cls HOTA/DetA/AssA
  `48.271/39.526/61.431`，det `56.179/48.821/66.894`；相对 Encoder 同点 HOTA
  `-1.409/-0.362`，DetA `-1.834/-1.524`。
- pair mAP/AP50 下降 `0.014159/0.007145`，both-independent mAP/AP50 下降
  `0.014924/0.007281`。这与 e8 的 DetA/AP 退化方向一致，排除单点随机波动或仅 AssA
  指标搬运的解释。
- 完整 e12 checkpoint、检测评估、TrackEval metrics 和 54 个原始文件验证后，99 上该实验
  已精确停止并释放 GPU 0,1。三种 confidence 路由均不再扩展；下一步只考虑轻量、结构性且
  与 confidence/scale 调参不同的 decoder 改进。

## 2026-07-31 22:49 CST 0731_28 center-motion factorization

- 机制依据：`0731_21` e24 的 det AssA 相对 Encoder 提高 `0.814`，但 det DetA 下降
  `0.735`。完整五维 antisymmetric detail 可能保留关联收益的同时扰动 width/height/angle。
- `0731_28` 仅允许该 detail 修正旋转框中心 `x/y`；`w/h/angle` 使用未修改的父模型几何。
  classification common、独立 attention、零初始化、严格 pair midpoint、数据、PairDN 和 loss
  全部不变。这是运动几何约束，不是 confidence、residual scale 或类别重加权。
- 相对 `0731_21` 参数和主要计算完全不变：完整模型 `22,889,847` 个参数，两个稠密 gate
  仍共 `131,072` 参数；仅对已有 5D detail 固定屏蔽后三维。
- 提交 `8a24666`；104 项单测、配置与完整构建、双卡真实 4-iter smoke、checkpoint gate/
  attention 审计以及正式 iter 50 五项门槛均通过。99 `0731_28` 已 RUNNING，首看 e4。

## 2026-08-01 03:47 CST 0731_28 e16 final

- e16 cls HOTA/DetA/AssA 为 `48.845/39.751/62.295`，det 为
  `57.176/48.934/69.124`；相对 Encoder 同点双 HOTA `-2.246/-1.144`、双 DetA
  `-2.853/-2.780`。pair mAP/AP50 为 `0.2635/0.4778`，both-independent 为
  `0.3017/0.5115`，四项相对父轨迹均下降 `0.0145–0.0206`。
- e12/e16 的连续结果否定“稠密 classification common + 中心运动 detail”绑定结构。
  epoch 16 checkpoint、检测与完整 TrackEval 核验后停止，99 GPU0/1 已释放；该分支不再 resume。

## 2026-08-01 04:14 CST 0801_02 epoch-4 Gate

- 分类路径完全保持 Encoder、仅最终框中心使用反对称 detail 的 `0801_02`，e4 cls/det HOTA 为
  `36.757/42.974`，相对 Encoder 同点提高 `0.548/4.221`。cls DetA/AssA 变化
  `-0.153/+2.052`，det 为 `+0.937/+9.054`，检测覆盖基本受保护且 det 覆盖有所提高。
- pair mAP/AP50 为 `0.1555/0.3083`，both-independent 为 `0.1828/0.3358`；相对父轨迹
  mAP 仅轻微下降约 `0.0017–0.0018`，AP50 提高约 `0.0122–0.0127`。完整 checkpoint、
  检测和 50 序列 TrackEval 已核验，继续 e8/e12，不把 e4 强信号外推成最终结论。
- 该候选新增 `65,536` 参数（约 `0.29%`），无新 decoder 层、attention、分支或 loss，符合
  简洁与效率硬约束；只有持续双 HOTA 增益成立后才进入同卡同温速度测试。

## 2026-08-01 04:36 CST 0801_01 epoch-8 复核

- 256 参数共享逐通道门 e8 cls/det HOTA `43.546/50.488`，相对 Encoder 同点
  `-1.723/+0.295`；cls/det DetA 下降 `3.092/2.941`，AssA 提高 `1.070/4.721`。
  e4 的双提升没有持续，det 微增主要来自关联而不是检测覆盖。
- pair mAP/AP50 下降 `0.0259/0.0294`，both-independent 下降 `0.0308/0.0337`。
  完整 e8 产物已核验。因 det HOTA 仍略高于父轨迹，按放宽规则保留到 e12 最终复核；
  当前为弱候选，不扩展共享门组合，不进行低价值速度测试。

## 2026-08-01 05:45 CST 0801_02 epoch-8 与 0801_03

- `0801_02` e8 cls HOTA/DetA/AssA `44.632/35.582/58.513`，det
  `49.687/43.373/59.043`；相对 Encoder 同点 HOTA `-0.637/-0.506`、DetA
  `-2.081/-3.688`，AssA 虽提高 `1.212/3.898`，但未抵消检测覆盖损失。
- pair mAP/AP50 为 `0.2158/0.4179`，相对父轨迹下降 `0.0219/0.0130`；
  both-independent 为 `0.2519/0.4518`，下降 `0.0243/0.0142`。HOTA、DetA、AP
  已形成系统性下降，完整 e8 产物核验后于 05:38 精确停止，252 GPU0/1 已释放。
- `0801_03` 只把上述 detail-only 路径的 `256×256` 稠密 gate 换成 256 维逐通道 gate。
  它不改变分类、reference、辅助输出、宽高角或训练协议，不新增层、attention、分支或 loss；
  新增参数降为 256，并去掉矩阵乘法。该最小对照检验稠密跨通道混合是否是 e8 覆盖退化来源。
- 107 项 decoder 单测、配置加载与深拷贝已通过；完成提交同步和真实双卡 smoke 后，按五项门槛
  决定是否 formal fresh。即使性能通过，也必须同卡同温确认吞吐下降不超过 5% 才能进入论文主线。

## 2026-08-01 05:58 CST 0801_03 正式启动

- `9a18a0c` 已同步四机。252 的真实双卡 4-iter smoke 产生完整 checkpoint；唯一 detail gate
  形状为 `(256,)` 且已非零，6 组 prev/curr attention 均保持独立，总/DN/encoder loss 与
  grad norm 有限，无分布式或数值异常。
- formal fresh 于 05:56 启动并通过 iter 50：`1.1619 s/iter`、loss `21.5299`、grad norm
  `106.3036`，两卡各约 19.2 GiB。该候选正式记为 RUNNING；首个 e4 仅用于结构检查，e8/e12
  决定逐通道门能否避免 `0801_02` 的中期 DetA 与 AP 退化。

## 2026-08-01 06:46 CST 0801_01 epoch-12 最终结论

- 仅 256 参数的 common/detail 共用逐通道 gate 在 e12 得到 cls HOTA/DetA/AssA
  `47.158/38.483/59.975`，det `55.516/47.987/66.488`。相对 Encoder 同点双 HOTA
  `-2.522/-1.025`、双 DetA `-2.877/-2.358`，仅 det AssA `+0.909`。
- pair mAP/AP50 `0.2468/0.4538`，both-independent `0.2834/0.4872`；已确认 pair mAP
  与 both AP50 相对父配置下降 `0.02637/0.02455`。e8/e12 连续证明共享幅度约束不能阻止
  DetA→AssA 搬运，且 cls HOTA 进一步恶化。
- e12 checkpoint、检测、TrackEval、50 序列与 108 个评估文件完整后，197 已精确停止并释放
  GPU4/5。该机制不再扩展；当前唯一运行候选是更小且分类完全保持 Encoder 的 `0801_03`。

## 2026-08-01 07:26 CST 0801_03 epoch-4 结构结果

- e4 cls HOTA/DetA/AssA `36.944/27.076/54.129`，det `42.493/33.391/55.143`；
  相对 Encoder 同点 HOTA `+0.735/+3.740`，DetA `+0.008/+0.937`。早期增益没有通过
  明显压低检测覆盖获得。
- pair mAP/AP50 `0.1493/0.3078`，both-independent `0.1789/0.3365`；相对 Encoder
  mAP 略降而 AP50 提高。相比 `0801_02` e4，cls HOTA 提高 `0.187`、det HOTA 下降
  `0.481`，但 det DetA 同为 `33.391`，差异来自 AssA 而不是覆盖。
- 完整 e4 checkpoint、检测、TrackEval `async_done=1`、50 序列与 108 个文件已核验。
  该 256 参数候选继续到 e8/e12；不在 e4 增益上追加结构，也不提前做主线速度结论。

## 2026-08-01 11:02 CST 恢复 0730_10 symmetric-pair

- 全部近期 evidence/detail 门控实验共同表现为 DetA→AssA 搬运，因此不再继续增加门控复杂度。
  回查发现 `0730_10 symmetric-pair decoder` 在 e4 的 cls/det HOTA 为
  `36.750/42.604`，两项均高于 Encoder 同点；它当时只因 pair mAP 下降 `0.00765`
  触发旧保护线，在 epoch 5 中途被停止，没有得到 e8/e12 证据。
- 该结构共享两帧 decoder deformable cross-attention，并将 frame-evidence fusion 与
  pair-position fusion 显式交换对称化；它不新增参数、decoder 层、attention、分支或 loss，
  因而比新增 residual/gate 变体更符合当前简洁和效率要求。
- 252 GPU0/1 已从原 `epoch_4.pth` 恢复。当前提交相对原启动提交只增加其他互斥 decoder
  选项，symmetric-pair 的前向语义未变；配置深拷贝、checkpoint、HSMOT、GMC 和双卡空闲
  均已核验。11:01 到 epoch 5 iter 50：`1.0891 s/iter`、loss `11.6265`、grad norm
  `37.3829`，总/DN/encoder loss 均有限，无 Traceback/OOM/NaN/NCCL。e8 是首个正式补验点，
  mAP 只作诊断，不再覆盖 Cls/Det HOTA 主判据；若非系统性退化则继续 e12。

## 2026-08-01 11:20 CST 准备 0801_04 symmetric-position

- `0730_10` 同时共享 cross-attention、交换对称化 frame-feature fusion 与 pair-position，若结果变化，三个约束难以单独归因。`0801_04` 只消除共享 self-attention position 中的帧序偏置：两帧位置先取均值，再经现有 `pair_pos_fusion` 一次；其余 decoder 路径严格保留父配置。
- 该改动不新增参数、层、attention、分支、loss 或矩阵乘法，参数量与父配置均为 `22,758,775`；独立 prev/curr deformable cross-attention 与原有有序 frame-feature fusion 均保留，因此不是完整 symmetric-pair 的重复实验。
- 110 项 decoder 单测、formal/smoke 配置加载与深拷贝、父/新模型完整构建和初始函数等价检查已在 99 通过。当前只登记为 `PREPARED`；197 GPU4/5 仍需通过真实双卡 4-iter smoke、有限 loss/grad、checkpoint 标志与独立 attention 审计，之后才允许 formal fresh。

## 2026-08-01 11:28 CST 0801_04 启动验收

- 提交 `d6e8c9a` 已在 197 精确 fast-forward。GPU4/5 连续空闲且 smoke/formal 目标目录均不存在后，真实双卡 4-iter smoke 产生 `iter_4.pth`；4 次总/DN/encoder loss 与 grad norm 全部有限，无 Traceback/OOM/NaN/NCCL/DDP 异常。
- smoke 有效配置只启用 `symmetric_position_decoder`；checkpoint 中 24 组 prev/curr cross-attention 张量均独立，训练后最大差异 `0.00078709`，排除误用完整共享 attention。formal fresh 于 11:25 启动。
- 11:27 到 epoch 1 iter 50：`1.7400 s/iter`、loss `21.5687`、grad norm `115.9326`，DN 与 encoder proposal loss 均有限，两卡各约 19.2 GiB，五项正式启动门槛通过。当前只记为训练稳定，不宣称性能成功；e4 结构检查后继续以 e8/e12 的双 HOTA 持续性为主判据。

## 2026-08-01 11:59 CST 0801_05 feature-only symmetry 启动验收

- `0801_05` 只约束每层两帧 cross-attention 输出到共享 recurrent query 的融合顺序：
  两帧输出先取均值，再重复为双输入，经现有 `cross_fusion` 一次。独立 cross-attention、
  原有有序 pair-position、Encoder、proposal、PairDN、head、loss 与训练协议均保持父配置。
  与 `0801_04 position-only` 一起可分离 `0730_10` 全对称结构的两个融合因素。
- 该路径不新增参数、层、attention、分支、loss 或矩阵乘法；父/新模型参数量均为
  `22,758,775`，state_dict 键和形状完全一致。提交 `9bda2ed` 已同步四机和 GitHub；
  113 项 decoder 单测、两份配置深拷贝、完整模型构建与 launcher 语法检查通过。
- 178 真实单卡 4-iter smoke 产生 `iter_4.pth`；最终 loss/grad norm
  `20.1570/185.1551`，总、DN、encoder proposal loss 有限，24 组独立 attention
  最大训练差异 `0.00076836`。formal fresh 于 11:57 启动，11:58 iter 50 为
  `0.9385 s/iter`、loss `21.1807`、grad norm `98.8892`，五项门槛通过。e4 只作
  结构信号；e8/e12 与 Encoder 同点双 HOTA 决定是否继续，AP 仅作系统性退化诊断。

## 2026-08-01 12:32 CST 0730_10 symmetric-pair epoch-8 结论

- e8 cls HOTA/DetA/AssA 为 `42.890/33.328/58.017`，det 为
  `48.228/41.431/57.793`。相对 Encoder e8，双 HOTA 为 `-2.379/-1.965`，双 DetA
  为 `-4.335/-5.630`；AssA 的提高不能抵消检测覆盖损失。
- pair mAP/AP50 `0.2034/0.3969`，both-independent `0.2390/0.4301`。检测 AP、HOTA
  与 DetA 同向退化，说明 e4 的早期双增益没有持续；问题不是旧 AP 保护线过严，而是共享
  cross-attention 加完整交换对称约束在中期削弱了帧特异检测证据。
- `epoch_8.pth`、检测 metrics、50 序列、TrackEval `async_done=1`、28 个 CSV 与 108 个
  评估文件完整。12:31 精确停止并释放 252 GPU0/1，不继续 e12。保留独立 cross-attention
  的 `0801_04 position-only` 与 `0801_05 feature-only` 继续，以区分位置和特征融合约束本身
  是否有效；在两者 e4/e8 出来前不启动新的复杂 decoder。

## 2026-08-01 13:12 CST 0801_05 feature-only symmetry epoch-4

- e4 cls HOTA/DetA/AssA `34.947/25.960/49.970`，det `38.300/30.836/48.870`；相对
  Encoder e4，双 HOTA `-1.262/-0.453`、双 DetA `-1.108/-1.618`，cls/det AssA
  `-2.124/+1.404`。因此 det 的接近不是检测覆盖改善，而是部分 DetA→AssA 搬运。
- pair mAP/AP50 `0.1399/0.2951`，both-independent `0.1685/0.3240`。checkpoint、检测、
  50 序列、TrackEval `async_done=1` 与 108 个评估文件完整。该点是负向但非灾难性的
  结构信号，按放宽后的 e4 规则继续到 e8，不提前停止。
- 13:11 已进入 epoch 5 iter 500，loss `10.1890`、grad norm `48.9170`，训练与显存正常。
  feature fusion 的帧序可能承载有用的时序证据；在 e8 前不扩展该机制，也不增加模型复杂度。
## 2026-08-01 13:59 CST：0801_04 e4 与 0801_06 启动

- `0801_04 symmetric-position` 的完整 e4 结果为：cls HOTA/DetA/AssA
  `35.531/26.057/52.181`，det `41.704/32.604/54.431`。相对 Encoder 同点，
  cls/det HOTA 为 `-0.678/+2.951`，DetA 为 `-1.011/+0.150`，AssA 为
  `+0.087/+6.965`。det 增益并非纯粹由 DetA 搬运到 AssA，但 cls 检测覆盖仍有损失；
  按放宽规则继续到 e8。
- `0801_05 symmetric-feature` e4 为 cls/det HOTA `34.947/38.300`，双 HOTA、
  双 DetA 均低于 Encoder；feature-only 对称化当前不支持扩展，仍保留到 e8 做持续性复核。
- 新建 `0801_06 symmetric-position + residual-preserving fusion`：每层先保留
  `shared_query`，再融合两帧相对 shared query 的 cross-attention innovation。
  它不增加参数、decoder 层、attention、分支、loss 或主矩阵乘法；完整模型仍为
  `22,758,775` 参数。
- 117 项 decoder 单测、配置深拷贝、完整构建、252 双卡真实数据 4-iter smoke 和
  checkpoint 结构检查均通过。formal fresh 于 13:57 在 252 GPU0/1 启动，iter 50
  的 loss/DN/encoder loss/grad norm 均有限，两卡各约 19.2 GiB。
- 同一台 252、同为 2xb4 的 iter-50 日志对照：旧 `0730_10` 为 `1.2052 s/iter`，
  `0801_06` 为 `1.2082 s/iter`，仅慢约 `0.25%`，显存少约 7 MiB，满足吞吐下降
  不超过 `5%` 的效率硬约束。首看 e4 结构信号，e8/e12 判断持续性。
## 2026-08-01 14:16 CST：0801_05 e8 淘汰

- `0801_05 symmetric-feature` e8 的 cls HOTA/DetA/AssA 为
  `43.178/36.079/55.376`，det 为 `49.175/45.687/54.586`。相对 Encoder e8，
  HOTA `-2.091/-1.018`，DetA `-1.584/-1.374`，AssA `-1.925/-0.559`。
- pair mAP/AP50 为 `0.2215/0.4305`，both-independent 为 `0.2618/0.4665`；
  两项 mAP 相对 Encoder 下降约 `0.014–0.016`，AP50 基本持平。结合双 HOTA、双 DetA、
  双 AssA 和 mAP 的一致下降，feature-only symmetry 是系统性退化，而非 DetA/AssA 搬运。
- e8 checkpoint、检测结果、`async_done=1`、50 序列和 108 个 TrackEval 文件均完整。
  14:16 精确终止训练进程组 `2550260`，178 GPU0 已释放；该分支不继续 e12，也不派生
  参数、scale、loss 或更复杂结构。

## 2026-08-01 15:24 CST：0801_06 epoch-4 结构信号

- `0801_06 symmetric-position + residual-preserving fusion` 的完整 e4 结果为：
  cls HOTA/DetA/AssA `36.223/26.836/52.292`，det
  `42.937/34.036/55.255`。相对 Encoder e4 的
  `36.209/27.068/52.094` 与 `38.753/32.454/47.466`，cls/det HOTA
  分别为 `+0.014/+4.184`，DetA 为 `-0.232/+1.582`，AssA 为
  `+0.198/+7.789`。cls 基本持平且未出现明显覆盖崩塌，det 同时提高 DetA 与 AssA，
  不是单纯把检测覆盖搬运为关联收益。
- 相对只做位置对称化的 `0801_04` 同点，`0801_06` 的 cls/det HOTA 又提高
  `+0.692/+1.233`；这支持“显式保留 recurrent query identity，只融合两帧
  cross-attention innovation”的结构假设，但 e4 只作为早期信号，不宣称最终成功。
- 检测诊断为 pair mAP/AP50 `0.1479/0.3029`，both-independent
  `0.1767/0.3320`；50 个序列、5416 条记录、28 个 CSV、`metrics.json` 中
  `track/async_done=1` 均已核验。训练已自动进入 epoch 5，继续到 e8 判断双 HOTA、
  DetA 与 AssA 的持续性；最终标准仍为同一 checkpoint 同时超过
  `54.437/62.393`。

## 2026-08-01 15:49 CST：0801_04 epoch-8 最终结论

- `0801_04 symmetric-position decoder` 的完整 epoch-8 结果为：cls HOTA/DetA/AssA
  `43.936/35.319/57.280`，det `50.186/44.544/58.474`。相对 Encoder 同点
  `45.269/37.663/57.301` 与 `50.193/47.061/55.145`，cls/det HOTA 分别为
  `-1.333/-0.007`，DetA 为 `-2.344/-2.517`，AssA 为 `-0.021/+3.329`。
- det HOTA 表面接近持平，但由明显 DetA 损失与 AssA 补偿构成；cls HOTA、DetA 从 e4 到 e8
  的差距继续扩大。检测评估 pair mAP/AP50 为 `0.2068/0.4058`，both-independent
  mAP/AP50 为 `0.2428/0.4403`，未提供反向支持。
- `epoch_4.pth`、`epoch_8.pth`、50 序列、5416 条记录、28 个 CSV 与
  `track/async_done=1` 均已核验。该结构满足中止条件，15:49 精确终止工作目录对应进程组，
  不继续 e12，也不派生参数、scale、loss 或复杂结构变体。
- 197 GPU0/1 当前由其他用户进程占用，各约 13 GiB；PairMOT 无残留进程，因此197暂不调度。
  当前唯一保留的 decoder 候选为252上的 `0801_06`，继续到 e8 验证残差保留融合是否维持双 HOTA。

## 2026-08-01 16:47 CST：0801_06 epoch-8 最终结论

- `0801_06 symmetric-position + residual-preserving fusion` 的完整 epoch-8 结果为：
  cls HOTA/DetA/AssA `42.599/34.046/55.870`，det `48.910/42.651/57.782`。
  相对 Encoder e8，cls/det HOTA 为 `-2.670/-1.283`，DetA 为 `-3.617/-4.410`，
  AssA 为 `-1.431/+2.637`；相对 `0801_04` e8，双 HOTA 又下降 `1.337/1.276`。
- 检测诊断 pair mAP/AP50 为 `0.2031/0.3842`，both-independent 为
  `0.2373/0.4167`。HOTA、DetA 与四项 AP 构成系统性退化，e4 的
  `+0.014/+4.184` 早期信号没有持续；显式保留 shared query 仍无法避免中期帧特异检测证据衰减。
- `epoch_4.pth`、`epoch_8.pth`、50 序列、5416 条记录、28 个 CSV、108 个评估文件与
  `track/async_done=1` 均已核验。16:47 精确终止对应进程组，252 GPU0/1完全释放；
  不继续 e12，也不再扩展 symmetric-position/feature/fusion 系列。
- 下一候选只考虑已经过只读路径审计的逐层分类 logit 残差：3 层双帧共 6 个 `256→8`
  零初始化线性头，新增 `12,336` 参数（约 `0.054%`），不增加 decoder 深度、attention、
  loss 或主计算。普通 query 以 detached Encoder proposal logits 为基值，DN query 以零为基值，
  层间像 box reference 一样 detach；实现前后必须证明默认配置严格等价，并通过真实数据 smoke。

## 2026-08-01 17:07 CST：0801_07 轻量逐层分类残差正式启动

- 新候选只修改 decoder 分类输出方式：以 detached Encoder proposal logits 作为普通 query
  的初始分类值，DN query 前缀严格补零；现有三层 decoder 各自预测 prev/curr 的 `256→8`
  零初始化 residual，层间分类基值像 box reference 一样 detach。回归、Encoder、Liquid、
  PairDN、loss、decoder 深度与 attention 均不改变；禁止与 class reweight、prototype gate、
  residual scale adapter 组合。
- 六个线性头共新增 `12,336` 参数，总参数从 `22,758,775` 增至 `22,771,111`
  （`+0.0542%`）。旧 decoder 分类头保留在 state dict 以兼容 checkpoint，但被冻结并由新头
  等量替换，因此可训练参数净增为 `0`。32 项 head 单测、10 项模型等价性测试、配置深拷贝和
  完整构建均通过。
- 252 双卡真实数据 4-iter smoke 完成，所有 loss、DN/Encoder loss 与 grad norm 有限，
  无 Traceback/OOM/NaN/NCCL；`iter_4.pth` 中 12 个 residual 张量均有限，六个头全部非零更新。
  与同机 `0801_06` smoke 的 iter 2–4 对比，总时间约慢 `3.7%`，扣除 data time 后模型时间约慢
  `1.9%`，显存 `11122` 对 `11121 MiB`，满足吞吐下降不超过 `5%` 的硬约束。
- 提交 `14f8bce` 已固定并同步到 252。formal fresh 于 17:05 在 GPU0/1 启动，
  17:06 到 epoch 1 iter 50：`1.1511 s/iter`、loss `27.6245`、grad norm `118.7157`，
  两卡各约 `19.2 GiB`，无数值或分布式异常。当前只认定为稳定运行；e4 仅作结构信号，
  e8/e12 判断持续性，最终成功仍要求同一 checkpoint 的 cls/det HOTA 同时超过
  Encoder `54.437/62.393`。

## 2026-08-01 18:28 CST：0801_07 epoch-4 结构结果

- 完整 TrackEval 得到 cls HOTA/DetA/AssA `32.950/26.338/44.533`，det
  `37.629/32.051/45.030`。相对 Encoder e4 的 cls
  `36.209/27.068/52.094` 与 det `38.753/32.454/47.466`，双 HOTA 分别下降
  `3.259/1.124`，双 DetA 下降 `0.730/0.403`，双 AssA 下降 `7.561/2.436`。
- 检测诊断 pair mAP/AP50 为 `0.1346/0.2551`，相对 Encoder 同点
  `0.157253/0.296134` 同时下降；both-independent mAP/AP50 为
  `0.1780/0.3269`，其中 mAP 低于 Encoder `0.184465`，AP50 略高于
  `0.323149`。因此当前不是单纯 DetA→AssA 搬运，而是统一残差头早期同时损失
  分类、检测和关联质量。
- `epoch_4.pth`、50 序列、5416 条检测记录、两类 combined summary CSV 与完整
  `trackeval_stdout.log` 已核验；TrackEval 50 个序列在 `53.72 s` 完成，无
  Traceback/OOM/NaN。e4 仍只作结构筛选点，实验已自动进入 epoch 5，保留到 e8
  检查是否存在延迟恢复，但不把该弱信号扩散到 197。
- 99 `0801_08` 与 178 `0801_09` 分别隔离 DN 的绝对分类语义，并对层间分类残差采用
  detach / end-to-end 两种梯度路径；两项 formal 均已通过真数据 smoke 与 iter-50
  五项门槛。它们到 e4 前不再增加新候选；若 e8 仍复现双 DetA/HOTA 系统性下降，
  下一结构只考虑末层 Encoder-anchored 分类修正，不继续三层累计或参数扫描。

## 2026-08-01 19:35 CST：0801_08/09 epoch-4 与 0801_10 决策

- `0801_08 DN-isolated + layer-detach` 完整 TrackEval 为 cls
  HOTA/DetA/AssA `32.381/26.695/42.728`，det
  `38.380/33.911/44.366`；both-independent mAP/AP50 为
  `0.1898/0.3413`。相对 Encoder e4，cls/det HOTA 为
  `-3.828/-0.373`，DetA 为 `-0.373/+1.457`，AssA 为
  `-9.366/-3.100`。DN 绝对分类隔离能改善 det 覆盖，但层间断梯度使 cls
  关联损失进一步扩大。
- `0801_09 DN-isolated + end-to-end` 完整 TrackEval 为 cls
  HOTA/DetA/AssA `34.306/27.645/44.607`，det
  `38.590/33.639/45.922`；both-independent mAP/AP50 为
  `0.1896/0.3496`。相对 Encoder e4，cls/det HOTA 为
  `-1.903/-0.163`，DetA 为 `+0.577/+1.185`，AssA 为
  `-7.487/-1.544`。相对 `0801_08`，双 HOTA 提高
  `+1.925/+0.210`，说明端到端层间梯度优于 detach，但三层累计分类改写仍持续
  破坏 AssA。两项均继续到 e8 检查持续性，暂不在 197 做重复复现。
- 基于上述机制差异，新增 `0801_10 terminal encoder cls residual`：辅助 decoder
  分类头完全保持原实现，只有最终 normal query 在 detached Encoder proposal logits
  上做一次零初始化线性残差；最终 DN prefix 继续使用原 absolute classifier。该设计
  不改变回归、reference、attention、decoder 深度、loss、类别权重或 scale，仅新增两个
  `256→8` 线性头，目标是保留 `0801_09` 的 DetA/AP 改善，同时避免三层累计改写导致的
  AssA 崩塌。计划通过单测、构建和 197 双卡真数据 smoke 后启动 formal。

## 2026-08-01 19:50 CST：0801_10 验证与正式启动

- 38 项 head 单测、配置深拷贝和完整模型构建通过。相对不启用该功能的同配置 Encoder
  parent，只新增 prev/curr 两个末层 `256→8` 线性头，共 `4,112` 参数；总参数
  `22,762,887`，相对 `22,758,775` 增加 `0.0181%`。新增 state dict 仅有四个
  weight/bias 张量，没有 decoder layer、attention、回归分支或 loss 变化。
- 197 GPU4/5 真数据 2xb4、4-iter smoke 完成。四个迭代的主分类/回归、DN、Encoder
  loss 与 grad norm 全部有限，无 Traceback/OOM/NaN/NCCL；峰值训练显存约
  `11.15 GiB/卡`。`iter_4.pth` 中四个新张量全部有限，prev/curr weight 各
  `2048/2048` 个元素非零，证明两个末层残差头均获得真实梯度更新。
- 197 的默认 Python 路径曾指向另一份旧目录；构建门槛据此失败并在正式启动前纠正。
  核验后的实际导入路径为
  `/data/users/litianhao/PairMOT_sync_3cb888d/ai4rs/.../pair_rotated_rtdetr_head.py`，
  与 launch script 的 `PYTHONPATH` 一致，不存在混用旧实现。
- 提交 `936b1ae` 已从 99 以增量 bundle 快进同步至 178、252、197，四台仓库 HEAD
  一致。formal fresh 于 19:48 在 197 GPU4/5 启动；19:49 到 epoch 1 iter 50：
  `1.6436 s/iter`、data time `0.0357 s`、loss `21.3908`、grad norm
  `111.9423`、训练显存 `11,162 MiB`，无异常。GPU 利用率受同机其他用户作业影响有
  波动，因此当前速度只作为运行状态，不与不同温度/负载的历史 smoke 做百分比结论；
  参数和主计算图增量远低于 1%，继续以稳定窗口核验不超过 5% 的效率约束。
- e4 仍只作为结构信号，重点检查 `0801_09` 已改善的 DetA/AP 是否保留、cls/det
  AssA 是否明显恢复；e8 判断持续性。最终成功标准不变：同一 checkpoint 的 cls/det
  HOTA 同时严格超过 Encoder `54.437/62.393`。

## 2026-08-01 22:55 CST：目标口径更新与 epoch-12/16 复核

- 新一轮持续探索明确以 `0727_01` 为模型基底；DN 等训练细节允许调整，但不得采用
  class-aware routing、类别/样本 reweight 或显著增加主计算量的结构。成功门槛按权威归档值
  执行：同一 checkpoint 的 cls/det HOTA 同时严格超过 `54.437/62.393`。decoder 可能慢收敛，
  因此 e4/e8 只保留为诊断点，不再作为单独否决条件；后续去留依据更长轨迹、退化分解和资源
  机会成本共同决定。本条规则覆盖本文较早记录中的 e4/e8 提前终止策略。
- `0801_07 iterative cls residual` e12 为 `46.274/53.442`，相对 Encoder 同点
  `-3.406/-3.099`；e16 恢复至 `49.482/56.340`，相对 Encoder 同点
  `-1.609/-1.980`。e16 的 cls/det DetA 差值为 `-2.188/-2.776`，AssA 差值已收窄至
  `-0.452/-0.738`；pair mAP/AP50 差值也由 e12 的 `-0.0394/-0.0557` 收窄至
  `-0.0229/-0.0314`。由于 e12→e16 出现明确的全面恢复，不在 e16 停止，至少继续到 e24
  检查晚收敛斜率。
- `0801_08 DN-isolated + layer-detach` e12 为 `47.398/54.092`，相对 Encoder
  `-2.282/-2.449`；`0801_09 DN-isolated + end-to-end` e12 为 `47.395/54.436`，
  相对 Encoder `-2.285/-2.105`。两项当时的 DetA、AssA 与四项检测 AP 均低于 Encoder；
  仍保留到至少 e16 完成评测，再结合 e12→e16 斜率判断是否释放其中的冗余资源。
- `0801_10 terminal Encoder-anchored cls residual` e4 为 cls/det HOTA
  `33.754/39.446`，相对 Encoder `-2.455/+0.693`；cls DetA/AssA 差值
  `+0.308/-7.445`，det 为 `+2.010/-1.063`。它呈现覆盖提升但关联损失，且当前 197
  受同机负载影响较慢；按新口径继续跨越 e8 观察，不以 e4 结果否决。

## 2026-08-01 22:55 CST：下一低成本 decoder 已完成静态验证

- `0801_11 terminal pair-common cls residual` 保留 parent decoder 的完整初始函数，只在最终层
  normal query 上把同一个零初始化 `256→8` residual 同时加到 prev/curr logits；DN prefix 与
  所有辅助层仍走原 absolute classifier。共享修正严格保持每一类别的 prev/curr logit 差，意图
  只校正 pair-common 置信度而不直接扰动两帧身份判别。
- 该结构不增加 attention、decoder 深度、回归分支、loss、类别路由、reweight 或 residual scale；
  总参数从 `22,758,775` 增至 `22,760,831`，仅 `+2,056`（`+0.00903%`）。41 项 head 单测、
  parent/DN/auxiliary 初始等价性、共享差分不变性、梯度与互斥检查、配置深拷贝、完整构建和 launcher
  语法均通过。待快速 GPU 合理释放后执行真实数据 smoke，再决定 formal；当前不抢占或中断已有
  长收敛实验。
- 历史上最接近双增益的 `0731_05 shared-attention + enveloped detail` 已准备从 `epoch_20.pth`
  原目录续跑至 72 epoch。其 e8 为 `+0.072/+1.396`，e12 为 `+0.491/-0.111`，e20 为
  `+0.126/-0.431`；由于只差 det HOTA 且旧结论可能受短窗口影响，保留为释放 252 后的晚收敛
  优先复核项，但不在 `0801_07` 正明显恢复时抢占该机器。

## 2026-08-01 23:12 CST：三条迭代式分类残差 epoch-16 对照

- `0801_07 unified iterative residual` 为 cls/det HOTA `49.482/56.340`，相对 Encoder
  `-1.609/-1.980`；`0801_08 DN-isolated + layer-detach` 为 `49.201/56.755`，相对
  `-1.890/-1.565`；`0801_09 DN-isolated + end-to-end` 为 `50.036/56.933`，相对
  `-1.055/-1.387`。三项均未通过同点门槛，但从 e12 到 e16 都在恢复，故不把 e16 解释为
  decoder 方向的直接失败。
- `0801_09` 同时具有三者最高的双 HOTA，并以 cls/det DetA `41.796/50.733`、pair mAP/AP50
  `0.269704/0.473090` 和 both mAP `0.314116` 主导其余两项；只有 `0801_08` 的 both AP50
  高 `0.001155`。因此端到端层间梯度与 DN 绝对语义隔离是当前迭代式分支的主线，继续至少到
  e24。`0801_07/08` 不再作为同等优先级复制；先让已接近的下一评估点完成，再释放双卡给机制
  独立的候选或历史近门槛晚收敛复核。
- `0801_08` e16 相对 Encoder 的 HOTA 差距已从 e12 的 `-2.282/-2.449` 收窄至
  `-1.890/-1.565`，且 both AP50 为 `+0.000476`；`0801_09` 从 `-2.285/-2.105` 收窄至
  `-1.055/-1.387`。这些轨迹直接支持继续观察，而不是沿用旧的 e4/e8 单点淘汰规则。

## 2026-08-01 23:12 CST：0801_12 class-agnostic objectness residual 准备

- `0801_12` 从最终共享 decoder state 只预测每个 query 的一个标量，并把它广播到全部 8 类和
  prev/curr 两帧。由此同时严格保持“同类跨帧 logit 差”和“同帧跨类 logit margin”，只允许
  decoder 校正 pair-common objectness；DN prefix 与所有辅助层保持 parent 路径。这是结构性的
  置信度分解，不是 class-aware routing、样本/类别 reweight 或 loss scale。
- 新分支为单个 `256→1` 零初始化线性层，仅增加 `257` 参数：总参数
  `22,759,032`，相对 Encoder 为 `+0.001129%`。44 项 head 单测、两类 margin 不变性、DN/aux
  初始等价性、梯度与互斥检查、正式/smoke 配置深拷贝、完整模型构建以及 launcher 语法均通过。
  当前仅为 `PREPARED`，尚未占用 GPU；优先级在更一般的 `0801_11 256→8 pair-common residual`
  之后，用作类别自由度是否必要的最小消融。

## 2026-08-02 00:12 CST：0801_10 epoch-8 与 0801_07 epoch-20

- `0801_10 terminal Encoder-anchored cls residual` e8 的 cls/det HOTA 为
  `41.840/48.644`，相对 Encoder 同点 `-3.429/-1.549`；cls/det DetA 为
  `-2.104/-2.552`，AssA 为 `-5.653/+0.017`。pair mAP/AP50 下降
  `0.041760/0.056862`，both-independent mAP/AP50 下降 `0.032582/0.018702`。
  e4 的 det HOTA/DetA 正增益没有保持，且 e8 出现 HOTA、DetA 与 AP 同向下降；但按当前目标
  约束，e8 仍不单独否决 decoder，保留到 e12 检查晚恢复，不复制该机制。
- `0801_07 unified iterative cls residual` e20 的 cls/det HOTA 为
  `50.470/57.537`，相对 Encoder 同点 `-1.044/-1.385`；较 e16 的
  `-1.609/-1.980` 再次收窄。cls/det AssA 已转为 `+0.557/+0.168`，主要差距集中在
  DetA `-1.893/-2.414`；pair mAP/AP50 仍低 `0.021732/0.032611`，both-independent
  mAP/AP50 低 `0.013799/0.010339`。
- 由于 e16→e20 的双 HOTA、AssA 与 AP 差距都继续恢复，`0801_07` 不在 e20 停止，继续到
  至少 e24。若 e24 DetA/AP 仍收窄则保留长收敛；若出现恢复停滞或反转，再释放 252 给
  `0731_05 epoch-20` 晚收敛复核。该决策依据完整 e12/e16/e20 轨迹，而非早期单点。

## 2026-08-02 00:32 CST：0801_08/09 epoch-20

- `0801_08 DN-isolated + layer-detach` e20 的 cls/det HOTA 为 `50.903/58.019`，相对
  Encoder 同点 `-0.611/-0.903`；相较 e16 的 `-1.890/-1.565` 显著恢复。cls DetA 只差
  `0.383`，det AssA 已高 `0.451`；pair mAP/AP50 仍低 `0.014601/0.020120`，但
  both-independent AP50 已高 `0.005046`。
- `0801_09 DN-isolated + end-to-end` e20 的 cls/det HOTA 为 `50.843/58.033`，相对
  Encoder `-0.671/-0.889`；cls/det DetA 只低 `0.081/0.744`，pair mAP/AP50 低
  `0.011749/0.016424`，both-independent mAP 低 `0.005715`、AP50 高 `0.002026`。
- 两项的双 HOTA 仅相差 `0.060/0.014`：detach 版本具有更高 AssA，end-to-end 版本更保护
  DetA 与主要 AP，当前互不支配；两者从 e16 到 e20 都出现全面晚恢复，因此均继续至 e24。
  `0801_07` e20 的 HOTA、DetA 与 AP 暂弱于两条 DN-isolated 路径，但 AssA 已超父模型，仍按
  既定判据观察 e24 是否把关联优势转化为检测覆盖恢复。e24 后再做第一次中后期资源收敛。

## 2026-08-02 01:36 CST：0801_07 epoch-24 收口

- `0801_07 unified iterative cls residual` e24 的 cls/det HOTA 为 `51.309/58.209`，相对
  Encoder 同点 `-0.405/-1.310`。cls HOTA 较 e20 的 `-1.044` 继续恢复，但 det HOTA 仅从
  `-1.385` 改善至 `-1.310`；cls/det AssA 已高 `0.993/0.579`，而 DetA 仍低
  `1.292/2.549`，其中 det DetA 较 e20 的 `-2.414` 反向扩大。
- pair mAP/AP50 差值从 e20 的 `-0.021732/-0.032611` 收窄至
  `-0.016942/-0.026663`，both-independent 从 `-0.013799/-0.010339` 收窄至
  `-0.008979/-0.004226`；但检测覆盖恢复速度已经明显落后于关联，且绝对双 HOTA
  `51.309/58.209` 被历史 `0731_05` e20 的 `51.640/58.491` 同时超过。
- 该结论来自 e4/e8/e12/e16/e20/e24 六个完整节点，不属于早期 epoch 淘汰。01:35 对精确
  PGID `1743885` 发送 TERM；二次核验确认全部目标进程退出、GPU0/1 为 `0%/1 MiB`，e24
  checkpoint、检测、TrackEval、50 序列与 108 个评估文件完整保留。252 接替为
  `0731_05 shared-attention + enveloped detail` 从可信 `epoch_20.pth` 原轨迹继续晚收敛复核。

## 2026-08-02 01:40 CST：252 恢复 0731_05 至 72 epoch

- 252 运行仓库在 `0801_07` 完全退出后，由干净 `c294b6c` 通过已验证增量 bundle 精确
  fast-forward 到 `5fc0b3e`；未在旧训练存活时热更新。resume launcher 通过 `bash -n`，
  `epoch_20.pth` 非空，GPU0/1 连续空闲，配置深拷贝和完整模型构建通过：模型参数
  `22,881,367`、`max_epochs=72`、workdir 保持原目录。
- 首次准备脚本的 `pgrep -af` 自匹配导致保护性退出，训练尚未启动；修正为只匹配唯一 workdir
  下的 `torchrun/tools/train.py` 后全部门槛通过。01:34:37 在 screen
  `resume_0731_05_e20_252` 启动，日志明确加载原 `epoch_20.pth` 并恢复到
  `epoch=20, iter=20760`，不是 fresh 或错误 checkpoint。
- 01:36 到 epoch 21 iter 100：iter 50 为 `1.1471 s`、loss `9.9780`、grad norm
  `53.5289`；iter 100 为 `1.1032 s`、loss `9.1609`、grad norm `46.7177`。主、DN、
  Encoder proposal losses 全部有限，两 rank 显存约 `19.4 GiB`；当前增量日志无
  Traceback/OOM/NaN/NCCL/DDP 异常。进程组 PGID `2161450` 含 torchrun、2 个训练 rank 和
  4 个 data worker；一个由超时只读审计遗留的 bash PID `2162241` 已精确清理，训练 PGID
  保持不变。该实验据此登记为 `RUNNING`，下一完整节点为 e24/`val_track_0006`。

## 2026-08-02 01:43 CST：0801_08/09 epoch-24 晚收敛复核

- `0801_08 DN-isolated + layer-detach` e24 的 cls/det HOTA 为 `51.326/58.495`，相对
  Encoder 同点 `-0.388/-1.024`。cls DetA 已从 e20 的 `-0.383` 收窄至 `-0.013`，det
  AssA 为 `+0.219`；pair mAP/AP50 仍为 `-0.010344/-0.011295`，但 both-independent
  AP50 已为 `+0.013104`。其 cls 差距继续恢复，det 差距略有反复，尚不能视为过门槛或收敛完成。
- `0801_09 DN-isolated + end-to-end` e24 的 cls/det HOTA 为 `51.709/58.781`，相对
  Encoder 仅 `-0.005/-0.738`；cls DetA 已为 `+0.095`，cls AssA 只差 `0.264`，det
  DetA/AssA 分别差 `0.751/0.568`。pair mAP/AP50 为 `-0.009758/-0.011840`，
  both-independent AP50 为 `+0.004938`。它从 e16 的 `-1.055/-1.387`、e20 的
  `-0.671/-0.889` 连续恢复到当前结果，直接证明该 decoder 分支存在明显晚收敛。
- e24 时 `0801_09` 的双 HOTA 比 `0801_08` 高 `0.383/0.286`，覆盖和 cls 关联也更强；
  `0801_08` 仍保留 det AssA 与部分 both AP 优势，二者尚非严格支配关系。两项均已自动进入 e25，
  因轨迹仍有有效恢复且未达到绝对目标，继续到后续四 epoch 评测点，不以 e24 的未过门槛直接否决。

## 2026-08-02 02:16 CST：0801_10 收口与 0801_11 正式启动

- `0801_10 terminal Encoder-anchored cls residual` e12 的 cls/det HOTA 为
  `45.875/54.476`，相对 Encoder 同点 `-3.805/-2.065`；cls DetA/AssA 为
  `-3.103/-5.076`，det DetA/AssA 为 `-2.584/-0.975`。pair mAP/AP50 为
  `-0.051749/-0.063691`，both-independent mAP/AP50 为 `-0.042793/-0.028763`。
  它在 e4/e8/e12 三个完整点上没有晚恢复，且 e12 比 e8 的双 HOTA 差距进一步扩大，构成覆盖、
  关联和 AP 一致退化；该路线据此收口，不再复制同机制或继续占用 197。
- 完整 e12 检测和 50 序列 TrackEval 均落盘后，对唯一 PGID `1965422` 发送 TERM；torchrun、
  两 rank 和全部 data worker 均退出，197 GPU4/5 回到 `0%/1 MiB`。checkpoint 与三组完整评测保留。
  该结论来自 e4/e8/e12 轨迹，而非用 e4/e8 单点否决。
- 197 的 tracked worktree 无修改后，从 `c294b6c` 经已验证增量 bundle 快进到 `c8bc1ef`。
  新的 `0801_11 terminal pair-common cls residual` 保留 parent decoder 初始函数，仅在最终 normal
  query 上把同一个零初始化 `256→8` 残差加入 prev/curr；DN 和辅助层仍使用原分类器，不增加
  attention、decoder 深度、回归、loss、class-aware routing 或 reweight。完整构建确认参数从
  `22,758,775` 增至 `22,760,831`，仅 `+2,056`（约 `+0.00903%`）。
- 197 GPU4/5 的真实 4-iter DDP smoke 正常退出，`iter_4.pth` 中 632 个浮点模型张量全部有限；
  新残差 weight `2048/2048`、bias `8/8` 元素均从零更新，日志无 Traceback/OOM/NaN/NCCL。
  formal 于 02:14 在 screen `decoder_0801_11_197` 启动；epoch 1 iter 50 为
  `1.6317 s/iter`、loss `21.2437`、grad norm `112.9867`，DN 与 Encoder loss 均有限，
  双卡各约 `19.2 GiB`，目标进程 7 个且无致命信号。该实验登记为 `RUNNING`，下一完整节点为 e4。

## 2026-08-02 02:22 CST：0731_21 epoch-40 长程后备恢复入口

- 历史 `0731_21 terminal orthogonal factorized evidence` 已完整评估到 e40，绝对 cls/det HOTA
  为 `53.655/60.379`；相对 Encoder 同点为 `-0.142/-0.684`，但四项检测 AP 均提高，且
  e28/e32/e36/e40 形成了远晚于早期节点的可靠轨迹。共享盘上的 `epoch_40.pth` 为
  `421,537,702` bytes，`last_checkpoint` 精确指向它，10 个 checkpoint 与 21 个 metrics 文件均保留。
- 新增 `launch_resume_0731_21_from_epoch40_178.sh`，只提供从该可信 checkpoint 到 72 epoch 的
  精确恢复入口；它校验 checkpoint、`last_checkpoint`、GMC/预训练资源、唯一 workdir 进程和 GPU0
  空闲状态，并设置 PyTorch 2.6 可信 optimizer 恢复兼容变量。launcher 已通过 `bash -n`，当前仅为
  `PREPARED`，不会热更新或抢占正在运行且同点更接近的 `0801_09`。

## 2026-08-02 03:02 CST：0801_08/09 epoch-28 与 0801_13 静态验证

- `0801_08 DN-isolated + layer-detach` e28 的 cls/det HOTA 为 `51.719/58.933`，相对
  Encoder 同点 `-0.021/-0.897`。cls DetA 已为 `+0.481`，det AssA 为 `+0.441`；pair
  mAP/AP50 仍为 `-0.008635/-0.006310`，both-independent AP50 为 `+0.014838`。
  det HOTA 差距由 e24 的 `-1.024` 收窄到 `-0.897`，cls 已基本持平，继续到 e32。
- `0801_09 DN-isolated + end-to-end` e28 的 cls/det HOTA 为 `52.177/59.280`，相对
  Encoder 同点 `+0.437/-0.550`。cls DetA/AssA 已为 `+0.398/+0.486`，det DetA/AssA
  仍为 `-0.716/-0.215`；pair mAP/AP50 为 `-0.008833/-0.010193`，both-independent
  AP50 为 `+0.003575`。cls 已全面超过同点父轨迹，det HOTA 差距从 e24 的 `-0.738`
  继续收窄，直接支持 decoder 晚收敛假设，继续到 e32 而不提前收口。
- 新增 `0801_13 terminal pair-differential objectness residual`：最终共享 decoder state 只预测
  每 query 一个零初始化标量，并以 `-s/+s` 加到 prev/curr 的全部类别 logits。因此 DN/辅助层、
  每帧类别 margin、两帧平均 logits、回归和全部 decoder 计算保持不变，仅允许时间方向 objectness
  偏差学习；它是 class-agnostic 加性残差，不使用 routing、reweight、scale、额外 attention 或深度。
- 252 的活动训练 worktree 未修改；在独立共享对象测试副本 `PairMmot_test_0801_13_4de300a`
  上完成 3 项定向测试和 47 项完整 head 回归，均通过。正式/烟测配置深拷贝和父/新完整模型构建
  通过：参数从 `22,758,775` 增至 `22,759,032`，仅 `+257`（`+0.001129%`），新增 state
  仅一个 `(1,256)` weight 与 `(1,)` bias。当前登记为 `PREPARED`，尚未做真实数据 smoke 或占用 GPU。

## 2026-08-02 03:08 CST：0731_05 epoch-24 晚收敛结果

- `0731_05 shared-attention + enveloped detail` 从可信 e20 恢复后的首个完整点 e24，cls/det
  HOTA 为 `51.974/59.168`，相对 Encoder 同点为 `+0.260/-0.351`。cls DetA/AssA 为
  `+0.665/-0.940`，det DetA/AssA 为 `-0.892/+0.507`；当前唯一 HOTA 未过项是 det，且
  瓶颈明确为检测覆盖而非关联。
- 检测诊断不存在崩塌：pair mAP/AP50 为 `+0.001687/+0.016308`，both-independent
  mAP/AP50 为 `+0.003824/+0.018868`，四项全部高于父轨迹。该点较 e20 的
  `+0.126/-0.431` 进一步把 det HOTA 差距收窄，同时保持 cls 正增益，是当前最强同点 decoder。
- 完整 checkpoint、5416 条记录、50 序列、54 个 TrackEval 原始文件和检测 metrics 均已核验；
  当前恢复增量日志仍无 Traceback/OOM/NaN/NCCL。实验已经进入后续 epoch，继续到 e28，不能因
  单项差 `0.351` 在仍改善且 AP 全升时提前终止。

## 2026-08-02 03:12 CST：0801_12 增加 197 可调度版本

- 99 的直接 SSH 仍在 banner 前超时，虽然共享训练和评测可正常监控，但不能把下一候选的启动能力
  绑定在该主机。为此给 `0801_12 terminal pair-common objectness residual` 增加 197 双卡正式/烟测
  配置和 launcher；当前只登记为 `PREPARED`，不修改或中断正在运行的 `0801_11`。
- 该版本继承 197 的数据、GMC、TrackEval 与 2xb4 资源路径，只把最终 normal query 的同一标量加到
  两帧全部类别 logits；DN、辅助层、类别 margin、跨帧同类差、框和 decoder 均不变。独立测试副本
  上的配置深拷贝、launcher 语法和父/新完整模型构建通过，参数为 `22,758,775→22,759,032`，
  仅 `+257`（`+0.001129%`）。真实数据 smoke 仍须等授权 GPU 合理释放后执行。

## 2026-08-02 04:16 CST：0801_08/09 epoch-32

- `0801_08 DN-isolated + layer-detach` e32 的 cls/det HOTA 为 `52.257/59.442`，相对
  Encoder 同点为 `-0.097/-0.888`。cls DetA 为 `+0.393`，但 cls AssA 为 `-1.361`；
  det DetA/AssA 为 `-1.722/+0.481`。pair mAP/AP50 为 `-0.007854/-0.003782`，
  both-independent mAP/AP50 为 `-0.001232/+0.016091`。该点较 e28 的 HOTA
  `51.719/58.933` 绝对值仍上升，但相对父轨迹的双 HOTA 没有继续收窄，故不把它作为当前主候选；
  训练已自然进入 e33，保留至 e36 复核晚收敛而不按单点终止。
- `0801_09 DN-isolated + end-to-end` e32 的 cls/det HOTA 为 `52.566/59.955`，相对
  Encoder 同点为 `+0.212/-0.375`。cls DetA/AssA 均为正增益 `+0.198/+0.306`；
  det DetA/AssA 分别为 `-0.406/-0.241`。pair mAP/AP50 为 `-0.008109/-0.009569`，
  both-independent mAP/AP50 为 `-0.004586/+0.002965`。det HOTA 差距从 e24 的
  `-0.738`、e28 的 `-0.550` 连续收窄到 `-0.375`，同时 cls 保持超过父轨迹，仍是迭代式分支主线，
  继续到 e36。该 checkpoint 的绝对值尚低于最终目标 `54.437/62.393`，不得误报达标。
- 两项 e32 都核验了 `54` 个 TrackEval 原始文件，并使用第 8 个检测/跟踪评测点与 Encoder e32
  精确对齐；不是用最终值或错位 epoch 做比较。当前所有长跑保持运行，下一完整节点为 e36。

## 2026-08-02 04:26 CST：0801_11 epoch-4 首个完整节点

- `0801_11 terminal pair-common cls residual` e4 的 cls/det HOTA 为 `36.449/42.527`，
  相对 Encoder 同点为 `+0.240/+3.774`，同一 checkpoint 的两项同点门槛均通过。cls
  DetA/AssA 为 `+0.411/-0.176`；det DetA/AssA 为 `+1.934/+6.427`，早期 det 增益同时来自
  覆盖与关联，而不是单一分量交换。
- pair mAP/AP50 为 `-0.005578/+0.023509`，both-independent mAP/AP50 为
  `-0.002781/+0.026497`；两个 AP50 均提高，但 mAP 尚未同步。评测核验了 `54` 个 TrackEval
  原始文件，检测与跟踪均精确对应 e4。
- 该点只证明结构在早期已有有效信号，绝对 HOTA `36.449/42.527` 仍远低于最终目标
  `54.437/62.393`，因此不得登记为成功或按 e4 外推最终结果。正式训练保持运行，至少继续检查
  e8/e12 以及后续晚收敛轨迹；下一候选不会抢占或中断本任务。
- 对 `epoch_4.pth` 中新增 `8×256` 权重做结构分解：八行共同均值分量占权重平方能量
  `29.07%`，bias 的共同分量占 `50.25%`；第一奇异值占总权重能量 `40.13%`，但其余七个
  奇异值均非退化。说明早期增益既含 class-agnostic objectness 成分，也含不可忽略的低成本
  类别通道自由度；因此保留 `0801_12` 作为最小消融，而不据此用 objectness-only 替换当前主跑。

## 2026-08-02 04:30 CST：0731_05 epoch-28

- `0731_05 shared-attention + enveloped detail` e28 的 cls/det HOTA 为 `52.157/59.493`，
  相对 Encoder 同点为 `+0.417/-0.337`。cls DetA/AssA 为 `+1.083/-1.230`；det
  DetA/AssA 为 `-0.875/+0.516`，所以未过项继续明确受检测覆盖限制，而不是关联崩塌。
- pair mAP/AP50 为 `-0.000386/+0.011328`，both-independent mAP/AP50 为
  `+0.001103/+0.014289`；四项中三项提高，唯一负项已接近数值持平。det HOTA 差距较 e24
  的 `-0.351` 小幅收窄，cls 正增益扩大。该轨迹尚未达到绝对目标，也未形成同点双通过，
  但瓶颈仍在改善且 AP 诊断健康，继续到 e32；不因 e28 尚差 `0.337` 提前收口。
- 本次对照使用第 7 个检测/跟踪评测点并核验 `54` 个 TrackEval 原始文件；e28 checkpoint
  与当前训练进程均保持正常，252 已自然进入 e29。

## 2026-08-02 05:34 CST：0801_08/09 epoch-36

- `0801_09 DN-isolated + end-to-end` e36 的 cls/det HOTA 为 `52.985/60.410`，相对
  Encoder 同点为 `+0.073/-0.297`。cls DetA/AssA 为 `+0.186/-0.074`；det
  DetA/AssA 为 `-0.292/-0.331`。pair mAP/AP50 为 `-0.007950/-0.010089`，
  both-independent mAP/AP50 为 `-0.003323/+0.003584`。det HOTA 差距从 e24
  `-0.738`、e28 `-0.550`、e32 `-0.375` 连续收窄到 `-0.297`，且 cls 仍略高于父轨迹；
  绝对值尚低于 `54.437/62.393`，因此继续到 e40 而不误报成功。
- `0801_08 DN-isolated + layer-detach` e36 的 cls/det HOTA 为 `52.698/59.654`，相对
  Encoder 同点为 `-0.214/-1.053`。cls DetA/AssA 为 `+0.113/-1.042`；det
  DetA/AssA 为 `-1.701/-0.015`。pair mAP/AP50 为 `-0.008160/-0.003739`，
  both-independent mAP/AP50 为 `-0.002345/+0.013352`。它从 e28 到 e36 的相对双 HOTA
  已停滞/反转，并在 e36 被 `0801_09` 的绝对 HOTA 同时严格超过 `+0.287/+0.756`。
- 对 `0801_08` 的资源收口依据是 e4–e36 九个完整节点、后期相对趋势和同机制强版本支配，
  不是 e4/e8 淘汰。99 直连 SSH 及经 252 的 ProxyJump 均在 banner exchange 超时；共享盘日志
  显示作业仍正常进入 e37，但当前无法安全发送精确 PGID 信号。暂不采用共享文件等非受控方式
  强停，待管理连接恢复后释放该双卡给 `0801_12/13`。
- 两项 e36 均核验 `54` 个 TrackEval 原始文件，并使用第 9 个检测/跟踪评测点与 Encoder e36
  精确对齐。

## 2026-08-02 05:57 CST：0731_05 epoch-32

- `0731_05 shared-attention + enveloped detail` e32 的 cls/det HOTA 为 `52.022/59.938`，
  相对 Encoder 同点为 `-0.332/-0.392`。cls DetA/AssA 为 `+0.706/-2.511`；det
  DetA/AssA 为 `-0.882/+0.404`。相对 e28，cls 主要因关联出现回撤；det 绝对值从
  `59.493` 升至 `59.938`，但父轨迹同时变强，使同点差距从 `-0.337` 略扩大到 `-0.392`。
- pair mAP/AP50 为 `-0.001905/+0.008710`，both-independent mAP/AP50 为
  `-0.000820/+0.011440`；两项 AP50 仍为正，mAP 仅小幅低于父轨迹。该点未达到同点双通过或
  最终绝对目标，但不能把单个 e32 回撤等同于 decoder 已收敛失败：历史 `0731_21` 到 e40 仍有
  有效改善，且本轨迹 det 绝对值仍上升。因此保留至 e36 复核，再结合 e24–e36 的完整斜率决定资源。
- 本次核验第 8 个检测/跟踪评测点及 `54` 个 TrackEval 原始文件；checkpoint 与训练进程正常，
  252 已进入 e33。

## 2026-08-02 06:31 CST：0801_11 epoch-8

- `0801_11 terminal pair-common cls residual` e8 的 cls/det HOTA 为 `43.486/49.394`，
  相对 Encoder 同点为 `-1.783/-0.799`。e4 的同点双增益未在 e8 保持；cls
  DetA/AssA 为 `-3.183/+0.340`，det DetA/AssA 为 `-3.501/+2.527`，说明当前主要问题是
  检测覆盖下降，而关联仍优于父轨迹。
- pair mAP/AP50 为 `-0.027994/-0.031200`，both-independent mAP/AP50 为
  `-0.031737/-0.033942`，四项检测 AP 均转负，与 DetA 诊断一致。该点未通过同点门槛，
  绝对值也远低于最终目标。
- 按既定约束不以 e8 单点否决 decoder：正式训练已进入 e9，继续至少到 e12，检查覆盖与 AP
  是否像其他迭代式 decoder 一样在中后期恢复。若 e12/e16 仍保持 DetA/AP 系统退化，再结合完整
  e4–e16 轨迹决定是否给 `0801_12/13` 释放 197。评测核验第 2 个检测/跟踪点和 `54` 个
  TrackEval 原始文件。

## 2026-08-02 07:00 CST：0801_09 epoch-40 首次成熟同点双通过

- `0801_09 DN-isolated + end-to-end` e40 的 cls/det HOTA 为 `54.059/61.102`，相对
  Encoder 同点为 `+0.262/+0.039`；这是该长轨迹首次在同一 checkpoint 上同时严格超过父模型。
  cls DetA/AssA 为 `+0.486/-0.100`，det DetA/AssA 为 `+0.177/-0.224`，双 HOTA 通过同时
  伴随两路检测覆盖提升，不是纯 AssA 交换。
- pair mAP/AP50 为 `-0.001953/-0.003197`，both-independent mAP/AP50 为
  `+0.002518/+0.009286`。同点双 HOTA 已通过，但 pair AP 仍有轻微负差，说明继续训练仍有必要。
- 该 checkpoint 的绝对值距最终严格目标 `54.437/62.393` 仍差 `0.378/1.291`，因此只登记为
  `SAME-POINT PASS`，不得登记为最终成功。det 差距从 e24 `-0.738`、e28 `-0.550`、e32
  `-0.375`、e36 `-0.297` 到 e40 `+0.039` 连续跨零，构成明确晚收敛证据；继续到 e44 及更晚节点。
- 本次使用第 10 个检测/跟踪评测点，并核验 `54` 个 TrackEval 原始文件；训练已自然进入 e41，
  checkpoint、异步 TrackEval 与主进程均正常。

## 2026-08-02 07:03 CST：0801_08 epoch-40 与 99 资源状态

- `0801_08 DN-isolated + layer-detach` e40 的 cls/det HOTA 为 `53.880/60.432`，相对
  Encoder 同点为 `+0.083/-0.631`。cls DetA/AssA 为 `+0.773/-1.527`；det
  DetA/AssA 为 `-1.167/+0.186`。pair mAP/AP50 为 `-0.000736/+0.007134`，
  both-independent mAP/AP50 为 `+0.005930/+0.023822`。AP 诊断较 e36 改善，但 det HOTA
  仍受覆盖限制，未形成同点双通过或绝对目标通过。
- 同一 e40 checkpoint 上，`0801_09` 的绝对 cls/det HOTA 同时高 `0.179/0.670`，且它已同点
  双通过；`0801_08` 因此在 e4–e40 十个完整节点后被同机制强版本严格支配。管理优先级改为
  释放 99 给 `0801_12/13`，不再主动把该路线视为主候选。
- 99 SSH 再次以 `ConnectTimeout=20` 直连仍在 TCP 连接阶段超时，经 252 跳板此前也在 banner
  exchange 超时；共享盘显示作业继续正常运行。当前无法精确核验并终止 PGID，故不使用非受控
  替代方式强停；等待管理通道恢复。e40 仍核验了第 10 个检测/跟踪点和 `54` 个 TrackEval 原始文件。

## 2026-08-02 07:12 CST：0801_13 增加 252 可调度版本

- 为避免 99 管理连接长期不可用阻塞下一机制，给 `0801_13 terminal pair-differential
  objectness residual` 增加 252 双卡 formal/smoke 配置与 launcher；只适配 252 数据、GMC、
  TrackEval、Conda 和 workdir 路径，不改变该 `-s/+s` class-agnostic 机制。
- 两个 Python 配置通过语法编译，两个 launcher 通过 `bash -n`。在 252 的独立测试副本
  `/data/users/litianhao01/PairMmot_test_0801_13_4de300a` 上完成配置深拷贝和父/新完整模型构建：
  参数 `22,758,775→22,759,032`，仅 `+257`（`+0.001129%`），新增 state 仍严格只有
  `(1,256)` weight 和 `(1,)` bias。测试副本更新到 `cc852db`，活动训练仓库仍保持 `5fc0b3e`
  未被热更新。
- 当前状态为 `PREPARED`，尚未执行真实数据 GPU smoke。只有在 `0731_05` 完成既定 e36 复核且
  252 双卡安全释放后，才同步活动仓库、先跑 4-iter DDP smoke，再依据迭代 50 门槛登记 formal。

## 2026-08-02 07:24 CST：0731_05 收口并启动 0801_13

- `0731_05 shared-attention + enveloped detail` e36 的 cls/det HOTA 为 `52.694/60.428`，
  相对 Encoder 同点为 `-0.218/-0.279`。cls DetA/AssA 为 `+0.653/-2.072`；det
  DetA/AssA 为 `-0.735/+0.393`。pair mAP/AP50 为 `-0.001863/+0.006683`，
  both-independent mAP/AP50 为 `-0.000463/+0.009689`。它较 e32 有恢复，但 e24–e36 始终
  未双通过，且在绝对值上被 `0801_09` e40 同时超过 `1.365/0.674`。
- 收口依据为 e4–e36 九个完整节点、后期相对平台和已有强版本，不是早期 epoch 淘汰。第 9 个
  检测/跟踪点及 `54` 个 TrackEval 原始文件完成后，对精确 PGID `2161450` 发送 TERM；复核无
  残留目标进程，252 GPU0/1 均回到 `0%/1 MiB`。全部 checkpoint、检测和 TrackEval 产物保留。
- 252 活动仓库在旧任务完全退出、tracked 状态干净后，从 `5fc0b3e` 经已验证 bundle 快进到
  `8947b00`；没有在训练存活时热更新。`0801_13` 真实数据 2-GPU 4-iter smoke 正常退出：四个
  iteration 的总损失、DN、Encoder proposal loss 与 grad norm 均有限，无
  Traceback/OOM/NaN/NCCL；`iter_4.pth` 为 `364,262,824` bytes，新 `(1,256)+(1,)` 分支有限且
  获得非零更新，smoke 后双卡重新空闲。
- 第一次 formal 外壳因 SSH 引号拆分使训练挂在临时 sshd 而非 screen；在尚未到 iter 50 时对其
  精确 PGID `2449347` 发送 TERM，GPU 完全释放，并把 `130,258` bytes 的日志与配置移到可恢复
  目录 `0801_13_failed_shell_20260802_0720`，未删除。随后用脚本方式在持久会话
  `decoder_0801_13_252` 重新 fresh 启动，torchrun PGID 为 `2451495`。
- 正式任务于 e1 iter 50 达到 `1.0978 s/iter`、loss `21.3898`、grad norm `98.8906`；主、DN、
  Encoder proposal loss 全部有限，双卡利用率约 `83%/81%`，各约 `19.2 GiB`，无致命信号。
  据此登记为 `RUNNING`，下一完整节点为 e4；最终成功门槛仍是同一 checkpoint 绝对
  `cls/det HOTA > 54.437/62.393`。

## 2026-08-02 08:19 CST：0801_09 epoch-44 接近绝对门槛

- `0801_09 DN-isolated + end-to-end` e44 的 cls/det HOTA 为 `54.415/61.737`，相对
  Encoder 同点为 `+0.213/+0.545`，连续第二个成熟节点同点双通过。cls DetA/AssA 为
  `+0.579/-0.507`；det DetA/AssA 均为正 `+0.603/+0.470`，det 增益已从 e40 的边缘
  `+0.039` 扩大为稳定双分量增益。
- pair mAP/AP50 为 `-0.002012/-0.001499`，both-independent mAP/AP50 为
  `+0.002880/+0.010765`；pair 检测 AP 仍轻微低于父轨迹，但差距已小于 `0.0021`。
- 相对最终严格绝对目标 `54.437/62.393`，该 checkpoint 只差 cls `0.022`、det `0.656`。
  `54.415` 不做四舍五入达标处理，当前仍为 `SAME-POINT PASS / ABSOLUTE FAIL`。训练已进入
  e45，继续到 e48；本次核验第 11 个检测/跟踪点及 `54` 个 TrackEval 原始文件。

## 2026-08-02 08:34 CST：0801_11 epoch-12 恢复

- `0801_11 terminal pair-common cls residual` e12 的 cls/det HOTA 为 `48.966/56.120`，
  相对 Encoder 同点为 `-0.714/-0.421`。相比 e8 的 `-1.783/-0.799`，两项差距均显著收窄，
  直接证明 e8 不能作为该 decoder 的否决点。
- cls DetA/AssA 为 `-1.167/-0.226`；det DetA/AssA 为 `-1.656/+1.279`，当前主要瓶颈仍是
  检测覆盖，det 关联保持明显正增益。pair mAP/AP50 为 `-0.017076/-0.002863`，
  both-independent mAP/AP50 为 `-0.015952/-0.001198`；两项 AP50 已接近持平，mAP 仍需恢复。
- 该点未通过同点或绝对门槛，但 e8→e12 的 HOTA、DetA 与 AP50 均恢复，故继续到 e16；
  不在 e12 释放 197。评测核验第 3 个检测/跟踪点和 `54` 个 TrackEval 原始文件，训练已进入 e13。

## 2026-08-02 08:55 CST：0801_13 epoch-4 首点

- `0801_13 terminal pair-differential objectness residual` e4 的 cls/det HOTA 为
  `36.904/42.377`，相对 Encoder 同点为 `+0.695/+3.624`，同一 checkpoint 双通过。cls
  DetA/AssA 为 `+0.265/+1.569`；det DetA/AssA 为 `+1.030/+7.403`，四个 HOTA 分量全部
  正增益，早期 det 提升不是覆盖/关联交换。
- pair mAP/AP50 为 `-0.007057/+0.017851`，both-independent mAP/AP50 为
  `-0.004754/+0.020362`；两项 AP50 提高，mAP 尚未同步。第 1 个检测/跟踪评测点及 `54` 个
  TrackEval 原始文件完整。
- 该点只证明 class-agnostic `-s/+s` 结构有早期有效信号；绝对值仍远低于最终目标，且
  `0801_11` 已证明 e4 增益可能在 e8 反转。因此训练保持运行，继续 e8/e12 与后续晚收敛，
  不按 e4 外推或宣布成功。

## 2026-08-02 09:39 CST：0801_09 epoch-48 单项跨过绝对门槛

- `0801_09 DN-isolated + end-to-end` e48 的 cls/det HOTA 为 `54.609/62.091`，相对
  Encoder 同点为 `+0.606/+0.802`，连续第三个成熟节点同点双通过。cls DetA/AssA 为
  `+0.615/+0.268`；det DetA/AssA 为 `+0.763/+0.883`，四个 HOTA 分量全部为正。
- pair mAP/AP50 为 `+0.001332/+0.003852`，both-independent mAP/AP50 为
  `+0.004320/+0.012240`；四项检测 AP 首次在该主线同一节点全部超过父轨迹，当前没有检测诊断
  交换项。
- cls 已严格超过最终绝对门槛 `54.437`，但 det 相对 `62.393` 仍差 `0.302`；因此当前状态为
  `CLS ABSOLUTE PASS / DET ABSOLUTE FAIL`，目标仍未完成。训练已进入 e49，继续到 e52；不得因
  单项只差 `0.302` 或同点增益较大而提前宣布成功。核验第 12 个检测/跟踪点及 `54` 个
  TrackEval 原始文件。

## 2026-08-02 10:09 CST：0801_13 epoch-8 不作早停

- `0801_13 terminal pair-differential objectness residual` e8 的 cls/det HOTA 为
  `44.198/51.062`，相对 Encoder 同点为 `-1.071/+0.869`。cls DetA/AssA 为
  `-2.724/+1.153`；det DetA/AssA 为 `-3.286/+6.680`：pair-differential 信号仍明显提高
  关联，但当前检测覆盖下降，使 cls HOTA 暂时未通过同点门槛。
- pair mAP/AP50 为 `-0.028485/-0.030045`，both-independent mAP/AP50 为
  `-0.032269/-0.032924`，四项 AP 诊断同样指向覆盖恢复不足。第 2 个检测/跟踪评测点及
  `54` 个 TrackEval 原始文件完整。
- 该结果只登记为 `SAME-POINT/ABSOLUTE FAIL`，不作为 e8 直接否决：此前 `0801_11` 已在
  e8→e12 明确恢复，而 decoder 的后期收敛是本轮必须保留的观察窗口。252 训练继续到 e12
  及后续节点；是否收口将依据成熟节点轨迹、同机制支配关系和绝对目标，而非早期 epoch。

## 2026-08-02 10:38 CST：0801_11 epoch-16 继续恢复

- `0801_11 terminal pair-common cls residual` e16 的 cls/det HOTA 为 `50.873/58.037`，
  相对 Encoder 同点为 `-0.218/-0.283`。相比 e12 的 `-0.714/-0.421`，两项差距继续收窄；
  cls DetA/AssA 为 `-0.033/-0.794`，det DetA/AssA 为 `-1.304/+1.232`，det 仍表现为覆盖
  下降、关联提升。
- pair mAP/AP50 为 `-0.011530/+0.004735`，both-independent mAP/AP50 为
  `-0.010036/+0.007061`。两项 AP50 已转正，mAP 负差也较 e12 进一步收窄；第 4 个检测/跟踪
  节点及 `54` 个 TrackEval 原始文件完整。
- 该点仍为 `SAME-POINT/ABSOLUTE FAIL`，但 e8→e12→e16 连续恢复，尚没有成熟期平台证据；
  训练继续到 e20，避免把 decoder 的较慢收敛误判为结构失败。

## 2026-08-02 11:14 CST：0801_09 epoch-52 距绝对目标 0.005

- `0801_09 DN-isolated + end-to-end` e52 的 cls/det HOTA 为 `54.695/62.388`，相对
  Encoder 同点为 `+0.459/+0.917`，连续第四个成熟节点同点双通过。cls DetA/AssA 为
  `+0.465/-0.006`，AssA 仅近似持平；det DetA/AssA 均为正 `+0.787/+1.100`。
- pair mAP/AP50 为 `+0.003456/+0.005767`，both-independent mAP/AP50 为
  `+0.007388/+0.015760`，连续第二个节点四项 AP 全部超过父轨迹。第 13 个检测/跟踪节点及
  `54` 个 TrackEval 原始文件完整。
- cls 已严格超过绝对门槛 `54.437`，但 det `62.388` 比严格门槛 `62.393` 低 `0.005`；不按
  三位小数接近或四舍五入处理，状态仍为 `CLS ABSOLUTE PASS / DET ABSOLUTE FAIL`。训练已
  异步继续，保留 e56 作为下一成熟节点。
