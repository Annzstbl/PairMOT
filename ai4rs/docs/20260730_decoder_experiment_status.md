# PairMOT decoder 实验状态（2026-07-30）

更新时间：2026-08-05 18:45 CST

## 当前研究原则

- 论文主线保持为：`0719_05 Base 52.417/61.265` → `0723_01 Base+Liquid 53.955/62.032` → `0727_01 +Encoder 54.437/62.393`。
- decoder 目标是同时超过 `0727_01` 的 cls HOTA 54.437 和 det HOTA 62.393。
- 不再进行 class-specific reweight、long-tail reweight 或大规模 residual-scale 扫描；优先验证有明确时序归纳偏置的模型结构。
- AutoDL 实例均处于关机状态，不纳入当前调度。
- 资源边界为：252 固定 GPU0/1；99 总计 2 卡、178 总计 1 卡、197 总计 2 卡但不固定序号。每台机器同一时间至多使用该总卡数。252 最慢，只延续成熟路线或复验明确候选；新结构优先在 99、178、197 筛选。

## 运行中

| 服务器 | 实验 | 状态 | 结构与判定方式 |
| --- | --- | --- | --- |
| 252 固定 GPU 0,1 | `0804_01 ... factorized product-tangent ... resume from e12` | `RUNNING/TO_E60+` | e56 完整 `54.574/61.316`，同点和 `115.890`、严格仍差 `2.440`；较 e52 双升 `+0.260/+0.338`，已到 e57 iter450，GPU2/3 不动。 |
| 178 当前 GPU 0 | `0804_14 ... hemisphere-boundary center + log-shape consensus ... fresh` | `RUNNING/TO_E4+` | 0804_12 e12 成熟为 `47.019/52.807` 后停止；新线真实 smoke、checkpoint 与 formal iter50 五门槛通过，PGID `4074056`。GPU1 外部任务不动。 |
| 99 当前 GPU 0,1 | `0804_13 ... hemisphere-fold center + log-shape consensus ... fresh` | `RUNNING/TO_E12+` | e8 完整 `41.261/47.265`，相对强父线 e8 `-3.741/-1.818`；不以 e8 直接否决，已到 e9 iter350，GPU2 不动。 |
| 197 动态 GPU 0,1 | `0804_09 ... norm-preserving Householder product-tangent ... fresh` | `STOPPED/HOST_CPU_THROTTLED` | e8 完整 `42.596/47.448`；e12 step12418 后主机 80 核降至约 118–167 MHz 并持续自旋，精确停止原/恢复 PGID，GPU0/1 释放，保留 e8 等待主机恢复后续跑。 |

`0803_14 terminal log-area` 在 252 的 PGID `77558` 已停止且正式目录尚无 epoch checkpoint；smoke、正式 iter50 证据和隔离提交保留。资源序号澄清后改迁 99 的空闲 GPU1/2，重新执行 smoke 后 fresh 启动。252 不再使用 GPU2/3。

`0804_04 body-frame` 与 `0804_05 SE(2)` 均在 e4/e8/e12 三个完整节点后成熟停止；
`0804_07 axis-Frenet` 和 `0804_08 shared-metric` 已分别接替 178 与 99 并通过五项动态门槛。

`0804_10 covariant-Frenet product-tangent` 已在 e4/e8/e12 三个完整节点后成熟停止；其接替者
`0804_12 spherical-midpoint center + log-shape consensus` 已在 e4/e8/e12 完整窗口后成熟停止；接替者
`0804_14 hemisphere-boundary center + log-shape consensus` 已在 GPU0 通过真实单卡 smoke、
checkpoint 与 formal iter50 五项动态门槛，当前登记 `RUNNING/TO_E4+`，不触碰 GPU1 外部任务。

## 2026-08-05 18:45 CST：252 e56 延续双升；99 e8 中期负差但不早停

- 252 固定 GPU0/1 的 `0804_01 factorized product-tangent` e56 同一 checkpoint 的 cls
  HOTA/DetA/AssA 为 `54.574/45.498/67.605`，det 为 `61.316/54.146/71.984`。cls 已严格超过
  Encoder `0.137`，det 仍低 `1.077`；同点和 `115.890`，距严格 `>118.330` 仍差 `2.440`，
  因而未达标。相对 e52 HOTA 双升 `+0.260/+0.338`，DetA/AssA 也全部上升，曲线未平台，
  固定 GPU0/1 继续 e60+；18:45 已到 e57 iter450，GPU2/3 保持 `1 MiB/0%`。
- e56 pair mAP/AP50 `0.313371/0.528904`、both-independent
  `0.353368/0.567596`，较 e52 四项仅轻微回落，但 tracking 的 DetA/AssA 同步上升。441,451,318-byte
  checkpoint 的 iterative-cls/DN 已训练，642 个浮点张量有限；5416/50、28 CSV、108 非空文件、
  50 preds、`async_done=1` 完整，TrackEval 用时 382.3 秒。
- 99 动态 GPU0/1 的 `0804_13 hemisphere-fold center + log-shape consensus` e8 cls
  HOTA/DetA/AssA `41.261/34.161/52.333`，det `47.265/42.563/54.363`；相对强父线 `0803_13`
  e8 HOTA `-3.741/-1.818`，cls DetA/AssA `-4.976/-1.664`，det `-4.162/+1.009`。折返仍表现为
  定位损伤，只在 det AssA 留下小幅正差；但 e8 不是 decoder 的直接否决点，继续到 e12+。
- e8 pair mAP/AP50 `0.199485/0.367706`、both-independent `0.244864/0.430787`，相对父线四项
  `-0.033994/-0.057097/-0.041002/-0.063699`。375,530,998-byte checkpoint 的 iterative-cls/DN
  已训练且 642 张量有限；5416/50、28 CSV、108 非空文件、50 preds、`async_done=1` 完整，
  TrackEval 用时 280.1 秒。18:45 已到 e9 iter350，GPU2 未触碰。
- 178 的 `0804_14` 同期只用 GPU0 到 e2 iter600，total/DN/Encoder/grad 有限；GPU1 外部任务
  未动。三线均保留各自下一成熟节点，不将“checkpoint 有效”或“单侧过线”误写为目标成功。

## 2026-08-05 18:22 CST：0804_12 e12 成熟停止，0804_14 五门槛接替

- `0804_12 spherical-midpoint center + log-shape consensus` e12 同一 checkpoint 的 cls
  HOTA/DetA/AssA 为 `47.019/37.990/60.804`，det 为 `52.807/46.629/61.984`；相对强父线
  `0803_13` e12 分别低 `1.270/1.732`，其中 DetA 低 `3.445/3.162`，而 AssA 为
  `+2.639/+0.174`。pair mAP/AP50 `0.2436/0.4206`、both-independent
  `0.2865/0.4717`，相对父线四项低 `0.017939/0.044163/0.023210/0.051163`。e4 的关联型
  早期增益到 e8/e12 已转为稳定定位损伤，因此这是完整成熟窗口后的停止，不是 e4/e8 早停。
- e12 checkpoint 为 381,071,860 bytes，iterative-cls/DN 已训练，642 个浮点张量全部有限；
  5416 records/50 sequences、50 detection files、28 CSV、108 nonempty files、50 preds、
  `async_done=1` 完整，TrackEval 用时 241.2 秒。确认 9 个成员属于 PGID `3968124` 后仅对该
  进程组发送 TERM，成员 `9→0`，GPU0 连续空闲；GPU1 外部负载未触碰。
- `0804_14 hemisphere-boundary center + mature log-shape consensus` 随后在隔离 clean HEAD
  `6666085` 的 GPU0 运行四步真数据 smoke：loss `21.3728/20.6239/20.8790/21.1618`、grad
  `59.6014/67.0472/75.8907/78.8500`，DN/Encoder 全有限；364,506,676-byte checkpoint 的
  iterative-cls/DN 已训练且 642 个浮点张量有限，致命扫描 0。
- fresh formal screen/PGID `4074054/4074056` 于 18:19 启动；iter50 `0.9456 s/iter`、
  loss/grad `21.0279/103.0120`，total、DN、Encoder proposal 全有限，GPU0 约 31.4 GiB，
  GPU1 外部任务保持不动，致命扫描 0。screen/process、GPU、正式日志、iter50、有限值五门槛
  全部通过后才登记 `RUNNING/TO_E4+`；继续收集 e4/e8/e12+，不得以早期节点直接否决。
- 并行审计：252 固定 GPU0/1 到 e55 iter300，GPU2/3 仍为 `1 MiB/0%`；99 使用正确的
  `WY10.106.14.99` 入口，动态 GPU0/1 到 e8 iter350，GPU2 未触碰；197 仍保持
  `STOPPED/HOST_CPU_THROTTLED`。

## 2026-08-05 17:36 CST：0804_14 的 178 隔离动态 checkout 就绪但不占卡

- 在不修改存活 `0804_12` 仓库、也不创建任何 smoke/formal workdir 的前提下，从静态副本建立
  `/data1/users/litianhao01/PairMOT_hemisphereboundarycenterlogshape_0804_14_178`；它是 clean
  detached HEAD `6666085`，与 launcher 默认路径一致，状态为 `PREPARED/NO_GPU`。
- 远端副本的 formal/smoke 配置均通过 `copy.deepcopy`，两份 launcher 通过 `bash -n`；定向
  unittest `1/1 OK`，父/新完整构建通过，均为 `22,771,111` 参数、711 state tensors，增量 0、
  smoke 4 iter。首次构建检查因 editable 环境误加载旧 `/data1/users/litianhao01/PairMOT/ai4rs`
  而不认识新 decoder 参数；固定 `PYTHONPATH` 到隔离 checkout 后立即通过，属于路径污染而非
  模型构建失败。178 环境没有 pytest，故使用测试文件自身的 unittest 入口。
- 17:36 活跃 `0804_12` 仅用 GPU0 到 e11 iter500，total/DN/Encoder/grad 有限；候选仍不做
  真实 smoke。只有 e12 checkpoint、检测与 TrackEval 成熟闭环且支持停止、原 PGID 精确退出、
  授权单卡连续空闲后，才依次执行真实 smoke、checkpoint 检查与 formal iter50 五门槛。
- 197 同期只读命令仍约 24 秒，12 核仅 `132-147 MHz`；即使 GPU0-5 当前空闲，也继续登记
  `STOPPED/HOST_CPU_THROTTLED`，不得迁移候选。

## 2026-08-05 17:28 CST：99 hemisphere-fold e4 完整闭环，继续慢收敛窗口

- 99 `0804_13 hemisphere-fold center + mature log-shape consensus` e4 同一 checkpoint 的 cls
  HOTA/DetA/AssA 为 `30.896/25.987/39.169`，det 为 `37.806/33.666/43.476`。相对强父线
  `0803_13` e4 HOTA 为 `-1.953/+0.487`：折返只保住 det 微增，但分类明显受损；相对 178
  `0804_12` e4 的 `34.909/42.639` 也更弱。
- 检测 AP 为 pair mAP/AP50 `0.137848/0.264067`、both-independent
  `0.180727/0.333931`；369,969,782-byte checkpoint 的 iterative-cls/DN 语义已训练且 642
  个浮点张量全有限。5416 records/50 sequences、28 CSV、108 非空文件、50 preds、
  `async_done=1` 与 258.3 秒 TrackEval 完整。
- e4 仅作机制诊断，不作为 decoder 直接否决点；PGID `1891973` 与动态 GPU0/1 继续 e8/e12+，
  17:28 已到 e5 iter300，GPU2 外部任务未动。同期 252 到 e53 iter1000、178 到 e10
  iter1000，关键 total/DN/Encoder/grad 均有限。`0804_14` 保持
  `STATIC_VALIDATED/NOT_DEPLOYED/NO_GPU`，不以早期结果抢卡。

## 2026-08-05 17:15 CST：252 e52 延续双升但未达标；178 e8 早期优势反转

- 252 `0804_01 factorized product-tangent` e52 同一 checkpoint 的 cls
  HOTA/DetA/AssA 为 `54.314/45.363/67.175`，det 为 `60.978/53.997/71.336`；同点和
  `115.292`，距严格 cls/det/总和门槛仍差 `0.123/1.415/3.038`，不得登记成功。相对 e48
  HOTA 继续双升 `+0.146/+0.369`，成熟曲线未停滞，因此固定 GPU0/1 继续到 e56+；17:15 已到
  e53 iter350，GPU2/3 保持 `1 MiB/0%`。
- e52 检测 AP 为 pair mAP/AP50 `0.313803/0.530176`、both-independent
  `0.354068/0.569205`；435,968,374-byte checkpoint 的 iterative-cls/DN 语义已训练，642
  个浮点张量全有限。5416 records/50 sequences、28 CSV、108 非空文件、50 preds 与
  `async_done=1` 完整，TrackEval 用时 394.2 秒。
- 178 `0804_12 spherical-midpoint center + log-shape consensus` e8 的 cls
  HOTA/DetA/AssA 为 `43.401/32.729/60.312`，det 为 `48.520/39.772/62.131`；相对强父线
  `0803_13` e8 HOTA 为 `-1.601/-0.563`，说明 e4 的早期双正没有保持。pair mAP/AP50
  `0.200536/0.356655`、both-independent `0.240331/0.408671` 虽较 e4 继续上升，但不足以替代
  轨迹差距；按慢收敛规则继续 e12+，不以 e8 直接否决，17:15 已到 e10 iter150，仅用 GPU0。
- e8 的 375,558,452-byte checkpoint 通过 iterative-cls/DN 与 642 张量全有限检查；5416/50、
  28 CSV、108 非空文件、50 preds、`async_done=1` 与 233.3 秒 TrackEval 完整。99 `0804_13`
  同期到 e4 iter850，GPU0/1 活跃且关键 loss/DN/Encoder/grad 有限；GPU2 外部任务未动。
  `0804_14` 继续保持 `STATIC_VALIDATED/NOT_DEPLOYED/NO_GPU`，等待成熟 e12 与 99 e4/e8 证据。

## 2026-08-05 16:26 CST：0804_14 最近球面半空间投影完成双端口静态闭环

- `0804_11` 的 rank-one 删除会同时损伤 DetA/AssA/AP，而 `0804_12` 的球面中点在 e4 给出
  双正信号；`0804_13` 的折返虽保留范数和横向细节，但会把负纵向分量等幅翻到正侧。因此
  下一单因素 `0804_14 hemisphere-boundary center + mature log-shape consensus` 对已经位于
  运动一致闭半空间的 detail 完全恒等，只把运动反向 detail 移到球面半空间边界：去掉负纵向
  分量后将横向分量归一到原范数；严格反平行时采用确定性的二维垂直方向。这是球面测地距离下
  的最近可行方向，而不是 scale/gate/reweight 扫描。
- 结构仍仅在 terminal normal query 生效，保留成熟 log-size/周期角共识；DN、分类、loss、
  attention、层数、递归 reference 与辅助输出不变。实现零参数/state、交换等变、完整范数
  保持、class-agnostic，额外计算只有二维内积、投影与归一化，无明显计算量增长。
- 99 隔离静态 checkout
  `/data/users/wangying01/lth/PairMOT_hemisphereboundarycenterlogshape_0804_14_static99`
  clean detached HEAD `66e38e8`；178 等价物理 `1x8` 端口在
  `/data1/users/litianhao01/PairMOT_hemisphereboundarycenterlogshape_0804_14_static178`
  clean detached HEAD `6666085`。两端定向测试 `1/1 OK`、配置 deepcopy、launcher `bash -n`
  与父/新完整构建通过：`22,771,111` 参数、711 states、增量 0、smoke 4 iter；99/178 的
  smoke/formal 目标 workdir 均不存在，状态严格为 `STATIC_VALIDATED/NOT_DEPLOYED/NO_GPU`。
- 当前不抢占任何训练卡：99 `0804_13` 到 e1 iter900；178 `0804_12` 到 e7 iter800，仅用
  GPU0；252 `0804_01` 到 e51 iter250，固定 GPU0/1。关键 total/DN/Encoder/grad 全有限，
  继续优先收集 e8/e52 与同 checkpoint 异步评测，再决定 `0804_14` 的动态部署。

## 2026-08-05 16:13 CST：99 控制恢复，0804_11 成熟停止并由 0804_13 接替

- 99 的实际 SSH 入口是配置别名 `WY10.106.14.99`（端口 `2367`）；此前
  `RUNNING/CONTROL_UNREACHABLE` 来自误探测默认端口 22，并非训练机宕机。恢复控制后先核实
  `0804_11` 唯一训练进程组仍为 PGID `1791967`、占用动态 GPU0/1，而 GPU2 为外部任务；随后
  仅向该 PGID 发送 TERM。进程成员归零、screen 消失，GPU0/1 连续两次均为 `12 MiB/0%`，
  GPU2 外部任务保持约 `21081 MiB/100%`，未触碰。
- `0804_11` 的停止依据仍是 e12/e16 成熟窗口以及相对强父线在 HOTA、DetA/AssA 和四项 AP
  上的全面支配，不属于 e4/e8 早停。其 checkpoint、检测和 TrackEval 产物全部保留。
- `0804_13 hemisphere-fold center + mature log-shape consensus` 部署到新的隔离 checkout
  `/data/users/wangying01/lth/PairMOT_hemispherefoldcenterlogshape_0804_13_99_v2`，clean detached
  HEAD `84ad131`；首次普通 clone 因仓库缺失 LFS 对象导致 checkout 未完成，保留为失败构建
  证据，未用于训练，也未热更新任何存活仓库。`GIT_LFS_SKIP_SMUDGE=1` 的 v2 checkout 完整。
- 目标配置 deepcopy、两份 launcher `bash -n`、定向测试与父/新完整构建通过：参数
  `22,771,111`、state tensors `711`、增量 0。真实 GPU0/1 DDP smoke 四步 loss/grad、DN、
  Encoder 均有限，364,505,654-byte `iter_4.pth` 的 iterative-cls/DN 语义已训练且 642 个
  浮点张量全有限，致命扫描 0。
- fresh formal 于 16:10:57 启动，screen/PGID `1891971/1891973`、两 rank 与动态 GPU0/1
  驻留一致；iter50 为 `1.0030 s/iter`、loss/grad `21.4104/121.9437`，total、DN、Encoder
  proposal 全有限，致命扫描 0。五项动态门槛全部通过后才登记 `RUNNING/TO_E4+`；继续收集
  e4/e8/e12+，不得由早期节点直接淘汰。
- 同时只读复核：178 `0804_12` 仅用 GPU0，已到 e6 iter1000；252 `0804_01` 固定 GPU0/1，
  已到 e50 iter650。两线 total/DN/Encoder/grad 均有限且无致命错误；178 等待 e8，252 等待
  e52 同 checkpoint 检测与 TrackEval，252 GPU2/3 保持未用。

## 2026-08-05 15:53 CST：0804_12 e4 早期双正；252 e48 继续逼近；99 e16 成熟父线负差

- 178 `0804_12 spherical-midpoint center + log-shape consensus` e4 同一 checkpoint 的 cls
  HOTA/DetA/AssA 为 `34.909/27.213/47.152`，det 为 `42.639/32.932/57.756`。相对成熟
  terminal log-shape 父线 `0803_13` e4 的 cls `32.849/26.682/43.921`、det
  `37.319/34.948/41.243`，HOTA 为 `+2.060/+5.320`，cls DetA/AssA 为
  `+0.531/+3.231`，det 为 `-2.016/+16.513`。det 的早期提升主要来自关联，但不是以 cls
  或检测 AP 同步退化换取：pair mAP/AP50 `0.1459/0.2774`、both-independent
  `0.1905/0.3501`，相对父线四项约 `+0.0060/+0.0161/+0.0024/+0.0110`。
- e4 只作为机制诊断，不能直接通过或淘汰。369,970,612-byte checkpoint 的 12 个
  iterative-cls residual 最大绝对值 `0.0550922`，DN 隔离头已训练，642 个浮点张量全有限；
  5416 records、50 sequences、28 CSV、108 个非空文件、50 preds、`async_done=1` 完整，
  TrackEval 224.3 秒。PGID `3968124` 已恢复 e5 iter750，仅使用 GPU0，继续 e8/e12+；即使
  GPU1 当前空闲，也严格遵守 178 总计 1 卡上限。
- 252 `0804_01 product-tangent` e48 cls HOTA/DetA/AssA
  `54.168/45.451/66.613`，det `60.609/53.693/70.819`，同点和 `114.777`；距严格
  cls/det/总和仍差 `0.269/1.784/3.553`，不得登记成功。相对 e44 HOTA
  `+0.284/+0.415`，cls DetA/AssA `+0.246/+0.242`，det `+0.171/+0.761`；pair
  mAP/AP50 `0.3142/0.5321`、both-independent `0.3556/0.5735`，四项也继续小升。
  430,483,510-byte checkpoint 的 12 个 residual 最大绝对值 `0.1297024`，DN 与 642 个
  浮点张量全有限；5416/50、28 CSV、108 非空文件、50 preds、`async_done=1` 与 383.8 秒
  TrackEval 完整。成熟曲线仍在 DetA、AssA、AP 上同步上升，固定 GPU0/1 已到 e49 iter650，
  因此继续 e52，而不是在最接近门槛且仍双升时截断；GPU2/3 保持未用。
- 99 `0804_11 center-tangent` e16 cls HOTA/DetA/AssA `46.103/37.228/59.905`，det
  `53.390/47.668/61.908`；相对自身 e12 回升 `+2.003/+2.376`，证明此前决定不是 e4/e8
  早停，但相对强父线 `0803_13` e16 `50.415/57.456` 仍低 `4.312/4.066`。pair
  mAP/AP50 `0.2312/0.4084`、both-independent `0.2731/0.4600`，相对父线四项低约
  `0.04396/0.07773/0.04755/0.07743`，成熟定位、关联与 AP 仍被全面支配。
  386,506,998-byte checkpoint 的 residual 最大绝对值 `0.0991428`，iterative-cls/DN 与
  642 个浮点张量全有限；5416/50、28 CSV、108 非空文件、50 preds、`async_done=1` 与
  285.9 秒 TrackEval 完整。共享日志已到 e17 iter250；99 直连再次超时，保持
  `RUNNING/CONTROL_UNREACHABLE`，不通过共享盘注入控制，SSH 恢复后精确 TERM PGID
  `1791967` 并连续核验两张动态卡释放。
- 197 于 15:56 再次只读复核：短命令仍需约 24–32 秒，load `16.95`，抽样 12 核仅
  `131–157 MHz`；GPU0–3 空闲而 GPU4/5 有外部任务。主机级 CPU 降频未恢复，继续登记
  `STOPPED/HOST_CPU_THROTTLED`，不因空闲 GPU 启动/恢复实验，也不为 `0804_13` 创建动态
  workdir；现有 99/178 两个静态端口保持 `STATIC_VALIDATED/NOT_DEPLOYED/NO_GPU`。

## 2026-08-05 15:04 CST：三线真实进度复核；0804_13 补齐 178 等价端口

- 252 `0804_01` 的 screen、PGID `823929` 与 7 个进程成员仍一致，正式日志到 e47 iter450；
  GPU0/1 分别约 `21620/21644 MiB`，GPU2/3 均 `1 MiB/0%`。e48 checkpoint 尚未出现，
  因此不读取半成品指标，继续等待同点检测与 TrackEval 完整闭环。
- 178 `0804_12` 的 screen/PGID `3968121/3968124` 与 9 个成员一致，正式日志到 e3 iter350，
  total、DN、Encoder 与 grad 有限；GPU0 约 `31475 MiB/95%`，GPU1 当前空闲，但严格遵守
  178 总计 1 卡上限，不在 GPU1 并发新实验。e4 checkpoint 尚未出现，继续收集 e4/e8/e12+。
- 99 的共享正式日志 15:03 仍更新到 e14 iter850，有限 loss 与 grad 表明原 PGID 未静默退出；
  控制机直连及经 252 的独立探针仍在 `10.106.14.99:22` 超时。继续登记
  `RUNNING/CONTROL_UNREACHABLE`，不伪报 TERM、不通过共享盘注入控制；链路恢复后第一动作仍是
  精确停止 PGID `1791967` 并连续验证动态双卡释放。
- `0804_13` 新增 178 物理 `1x8`、全局 batch 8 的等价配置与 formal/smoke launcher；隔离静态
  checkout 已由增量 bundle 更新为 clean HEAD `4bf8964`。目标单测仍 `1/1 OK`，两份 launcher
  `bash -n`、配置 deepcopy 与完整父/新构建均通过：`22,771,111` 参数、711 states、增量 0、
  smoke 4 iter；目标 workdir 均不存在。99 与 178 两条端口都只登记
  `STATIC_VALIDATED/NOT_DEPLOYED/NO_GPU`，待合法资源释放后才依次执行真实 smoke、checkpoint
  语义/有限性检查和 fresh formal iter50 五门槛。

## 2026-08-05 14:43 CST：99 center-tangent e12 成熟负差；控制链路待恢复

- 99 `0804_11 center-tangent + log-shape consensus` e12 同一 checkpoint 的 cls
  HOTA/DetA/AssA 为 `44.100/36.004/56.845`，det 为 `51.014/46.188/58.474`，同点和
  `95.114`，距严格 cls/det/总和仍差 `10.337/11.379/23.216`。相对成熟 terminal
  log-shape 父线 `0803_13` e12 的 cls `48.289/41.435/58.165`、det
  `54.539/49.791/61.810`，HOTA 为 `-4.189/-3.525`，cls DetA/AssA 为
  `-5.431/-1.320`，det 为 `-3.603/-3.336`。pair mAP/AP50
  `0.219747/0.394447`、both-independent `0.262616/0.449127`，相对父线四项也低
  `0.041793/0.070316/0.047094/0.073736`。虽然自身 e8→e12 HOTA 继续回升
  `+4.690/+4.971`，e8/e12 的定位、关联与 AP 均被强父线同步支配，e4/e8/e12 完整窗口已
  足以形成成熟负结论；这不是 e4/e8 直接否决。
- `epoch_12.pth` 为 381,012,982 bytes，meta `12/12456`；12 个 iterative-cls residual
  最大绝对值 `0.0877105`，DN 两帧绝对头差异最大值 `0.0846600`，iterative-cls/DN 已训练，
  642 个浮点张量全有限。检测与 TrackEval 为 5416 records、50 sequences、28 CSV、
  108 个非空文件、50 preds、`async_done=1`，用时 279.2 秒。
- 99 正式共享日志仍在 14:36 更新至 e13 iter300，说明训练未静默退出；但从控制机及 178
  到 `10.106.14.99:22` 的两条独立 SSH 探针均在 banner/连接阶段超时，无法安全核验 PGID/GPU
  或执行精确 TERM。因此状态登记为 `RUNNING/CONTROL_UNREACHABLE`，不伪报已停止，也不通过
  共享盘注入控制操作；链路恢复后优先精确停止 PGID `1791967` 并连续核验动态卡释放。
- 197 可连接但每次短命令约 26 秒；抽样 CPU 仍仅约 `129–140 MHz`、load 约 14，GPU0–3
  空闲而 GPU4 有外部任务。该主机仍不登记为安全可用资源，不在异常 CPU 状态下恢复训练或
  启动新候选。

## 2026-08-05 14:55 CST：0804_13 hemisphere-fold 后继完成隔离静态闭环

- `0804_11` e12 表明 rank-one center tangent 同时损伤 DetA、AssA 与 AP；既有 Householder
  与 spherical-midpoint 又会对每个非退化 detail 做完整轴对齐或半程旋转。因此下一单因素
  `0804_13 hemisphere-fold center + mature log-shape consensus` 只在中心反对称 detail 与
  已建立运动方向内积为负时，反射其纵向分量；内积非负的 detail 完全不动，所有横向分量始终
  保留，范数严格不变。它不是幅度/scale 扫描，而是投影到运动一致闭半球的最小离散几何操作。
- 结构仍只作用于 terminal normal query，并保留成熟 log-size/周期角共识；分类、DN、loss、
  attention、层数、递归 reference 与辅助输出不变。实现零参数/state 增量、swap-equivariant、
  class-agnostic、无 reweight，计算仅为二维内积与条件反射，无明显计算量增长。
- 本地隔离 commit `84ad131` 已通过 Python 语法和两份 launcher `bash -n`；增量 bundle 部署到
  178 全新静态 checkout
  `/data1/users/litianhao01/PairMOT_hemispherefoldcenterlogshape_0804_13_static178`，clean HEAD
  `84ad131`，未修改活跃 `0804_12` 仓库。定向测试 `1/1 OK`，覆盖 terminal 调用顺序、DN
  精确保留、只折返负纵向分量、范数保持、交换等变、有限梯度与互斥。配置 deepcopy、父/新
  完整构建通过：均为 `22,771,111` 参数、711 states、增量 0；smoke 仍为 4 iter。
- 99 的目标 smoke/formal workdir 均确认不存在，当前严格登记
  `STATIC_VALIDATED/NOT_DEPLOYED/NO_GPU`：99 SSH 未恢复，未在 99 创建 checkout，未运行
  smoke、未产生 checkpoint、更未启动 formal。只有控制恢复、`0804_11` 精确停止且动态双卡
  连续空闲后，才部署并依次通过真实 DDP smoke、checkpoint、fresh formal iter50 五项门槛。
- 同步现场：178 `0804_12` 正式日志 14:54 到 e2 iter800，total、DN、Encoder 与 grad 有限；
  其 GPU0 活跃、GPU1 外部任务未动。252 与 99 的现有正式长线均未热更新。

## 2026-08-05 14:32 CST：covariant-Frenet e12 成熟停止；spherical-midpoint 五门槛接替；252 e44 继续上升

- 178 `0804_10 covariant-Frenet product-tangent` e12 同一 checkpoint 的 cls
  HOTA/DetA/AssA 为 `46.788/38.193/59.523`，det 为 `53.443/47.724/61.963`，同点和
  `100.231`。相对直接 product-tangent 父线 e12 的 cls `49.784/41.865/61.515`、det
  `56.243/50.021/65.603`，HOTA 为 `-2.996/-2.800`，cls DetA/AssA 为
  `-3.672/-1.992`，det 为 `-2.297/-3.640`；pair mAP/AP50
  `0.237823/0.416006`、both-independent `0.281755/0.469516`，也分别低约
  `0.03798/0.06259/0.03965/0.06148`。尽管相对自身 e8 HOTA 回升
  `+4.997/+5.318`，e4/e8/e12 三个完整节点始终被强父线支配，故这是成熟停止而非 e4/e8
  早停。381,103,476-byte checkpoint meta `12/12456`，12 个 residual 最大绝对值
  `0.0877797`，PairDN 与 642 个浮点张量全有限；5416/50、28 CSV、108 个非空文件、
  50 preds、`async_done=1` 与 241.2 秒 TrackEval 完整。TERM PGID `3856480` 后成员
  `9→0`，GPU0 连续两轮回到 `1 MiB/0%`，GPU1 外部任务未动。
- `0804_12 spherical-midpoint center + mature log-shape consensus` 只将中心反对称 detail
  的方向旋转到“学习方向与最短符号运动方向”的球面中点，精确保留原 detail 范数；其余成熟
  log-size/周期角共识不变。该操作零参数/state 增量、swap-equivariant、class-agnostic、无
  reweight、无新 layer/attention/loss。178 隔离 checkout clean HEAD `d85e837`；配置
  deepcopy、完整父/新构建 `22,771,111` 参数、711 states、增量 0 与 launcher 语法均通过。
  动态 GPU0 的四步真实 smoke loss `21.3696/20.6405/20.9079/21.2148`、grad
  `59.7293/76.8835/97.5604/94.5484`，total、DN、Encoder 全有限；364,506,164-byte
  `iter_4.pth` 的 iterative-cls/DN 与 642 个浮点张量审计通过。
- fresh formal screen `3968121.pm_0804_12_formal_178`、PGID `3968124` 保持 9 个成员；
  iter50 为 `0.9509 s/iter`、loss/grad `21.0208/137.1233`，正式 workdir/日志持续更新，
  total、DN、Encoder 与 grad 有限、fatal 0，GPU0 约 31.4 GiB。五门槛通过后才登记
  `RUNNING/TO_E4+`；GPU1 外部进程约 6.9 GiB，始终未触碰。
- 252 `0804_01 product-tangent` e44 cls HOTA/DetA/AssA
  `53.884/45.205/66.371`，det `60.194/53.522/70.058`，同点和 `114.078`；严格 cls、det、
  总和仍分别差 `0.553/2.199/4.252`，不得登记成功。相对 e40 HOTA 双升
  `+0.434/+0.545`、总和 `+0.979`；pair mAP/AP50 `0.312939/0.531436`、
  both-independent `0.354629/0.573133`，四项也较 e40 继续上升。424,999,478-byte
  checkpoint 的 iterative-cls/DN 与 642 个浮点张量全有限；5416/50、28 CSV、108 个
  非空文件、50 preds、`async_done=1`，TrackEval 392.3 秒。PGID `823929` 已到 e45，
  固定只用 GPU0/1，GPU2/3 保持 `1 MiB/0%`；因成熟曲线仍双升，继续 e48 复核。

## 2026-08-05 13:25 CST：center-tangent e8 完整闭环，继续 e12

- 99 `0804_11 center-tangent + log-shape consensus` e8 同一 checkpoint 的 cls
  HOTA/DetA/AssA 为 `39.410/32.941/49.304`，det 为 `46.043/41.679/52.688`，同点和
  `85.453`，距严格总和 `118.330` 仍差 `32.877`。相对成熟 terminal log-shape 父线
  `0803_13` e8 的 `45.002/39.137/53.997` 与 `49.083/46.725/53.354`，HOTA 为
  `-5.592/-3.040`，cls DetA/AssA 为 `-6.196/-4.693`，det 为 `-5.046/-0.666`；中心
  product-tangent 在 e8 的主要问题是检测覆盖/定位损伤，且分类关联也下降，不是只把
  DetA 搬运到 AssA。
- 相对自身 e4，HOTA 回升 `+8.977/+8.393`，cls DetA/AssA 回升
  `+7.979/+9.885`，det 回升 `+9.164/+7.924`，模型仍处于正常追赶段，不能用 e8
  直接否决。pair mAP/AP50 `0.188469/0.346839`、both-independent
  `0.235004/0.416619`；相对父线 e8 四项为
  `-0.045010/-0.077964/-0.050862/-0.077867`，与 DetA/HOTA 损伤方向一致。
- `epoch_8.pth` 为 375,519,798 bytes，meta `8/8304`；12 个 iterative-cls residual
  最大绝对值 `0.0704144`，PairDN 已训练，642 个浮点张量全有限。检测为 5416 records/
  50 sequences；TrackEval `val_track_0002` 产出 28 CSV、108 个非空文件、50 个非空预测，
  `async_done=1`，payload→metrics 用时 275.8 秒。正式 PGID `1791967` 已恢复 e9，13:24
  到 iter350，继续 e12 及成熟节点；GPU0/1 为本任务，GPU2 外部任务不动。
- 同步资源审计：252 固定 GPU0/1 到 e42 iter900，GPU2/3 均 `1 MiB/0%`；178 仅本任务
  GPU0 到 e10 iter500，GPU1 外部任务不动；197 13:08 短连接仍超时，不将其登记为可用。
  三条活跃正式日志 total/DN/Encoder/grad 有限，未热更新存活训练仓库；`0804_12`
  继续保持 PREPARED/NO_GPU，等待合法资源释放后再走真实动态五门槛。

## 2026-08-05 13:06 CST：covariant-Frenet e8 完整闭环，继续 e12

- 178 `0804_10 covariant-Frenet product-tangent` e8 同一 checkpoint 的 cls
  HOTA/DetA/AssA 为 `41.791/35.146/52.091`，det 为 `48.125/44.414/54.070`；
  相对直接 product-tangent e8 `46.673/53.922`，HOTA 为 `-4.882/-5.797`，
  cls DetA/AssA 为 `-4.745/-4.504`，det 为 `-2.848/-10.066`。共同 Frenet 帧在 e8
  仍同时损伤定位与关联，尤其 det AssA，尚未表现出比直接分块切空间更好的中期归纳偏置。
- 相对自身 e4，HOTA 回升 `+7.602/+11.513`，cls DetA/AssA 回升
  `+7.516/+7.004`，det 回升 `+11.295/+12.250`，说明模型仍在正常收敛，不能把 e8
  的父线负差直接当作成熟否决。pair mAP/AP50 `0.2068/0.3707`，both-independent
  `0.2538/0.4346`；相对直接 product-tangent 四项分别低约
  `0.0456/0.0736/0.0501/0.0793`。
- 375,572,468-byte checkpoint meta `8/8304`，12 个 residual 最大绝对值
  `0.0757188`，iterative-cls/DN 已训练且有限，642 个浮点张量全有限；5416/50、
  28 CSV、108 个非空文件、50 个非空预测与 `async_done=1` 完整，TrackEval
  payload→metrics 约 234 秒。正式 PGID `3856480` 已恢复 e9，继续 e12；仍仅使用
  动态 GPU0，GPU1 外部任务不动，`0804_12` 保持 PREPARED/NO_GPU。

## 2026-08-05 12:55 CST：product-tangent e40 完整闭环，继续 e44 成熟复核

- 252 `0804_01 product-tangent` e40 同一 checkpoint 的 cls HOTA/DetA/AssA 为
  `53.450/45.021/65.541`，det 为 `59.649/53.232/69.133`，同点和 `113.099`；
  距严格 `54.437/62.393/118.330` 三门槛仍差 `0.987/2.744/5.231`，不得登记成功。
  相对 e36 HOTA 双升 `+0.124/+0.356`、总和 `+0.480`；pair mAP/AP50
  `0.3087/0.5282`，both-independent `0.3508/0.5711`，较 e36 四项分别提高约
  `0.0030/0.0040/0.0029/0.0033`。成熟曲线仍有小幅恢复，因此保留到 e44，而不是把
  e36 的近平台单点外推成停线结论。
- `epoch_40.pth` 为 419,513,398 bytes，meta `40/41520`；iterative-cls residual 与 DN
  已训练且有限，642 个浮点张量全有限。检测为 5416 records/50 sequences；TrackEval
  `val_track_0007` 产出 28 CSV、108 个非空文件、50 个非空预测，`async_done=1`，
  payload→metrics 约 356 秒。正式 PGID `823929` 已恢复 e41，仍固定 GPU0/1，
  GPU2/3 保持 `1 MiB/0%`。
- 同步节点：178 `0804_10` e8 checkpoint 为 375,572,468 bytes，meta `8/8304`，
  12 个 residual 最大绝对值 `0.0757188`，iterative-cls/DN 与 642 个浮点张量审计通过；
  e8 检测正在执行，尚无完整 HOTA，不作半成品判断，也不以 e8 直接否决。

## 2026-08-05 12:27 CST：球面中点候选补齐 178 单卡接替路径

- 为避免 99 双卡长线占用使下一候选闲置，`0804_12` 新增与 99 物理 `2x4`、全局 batch 8
  等价的 178 物理 `1x8` 配置、四步 smoke 配置和 formal/smoke launcher；训练日程、EMA、
  scheduler、模型、DN、loss 与评估协议均不变，仅批拓扑沿用已验证的 178 单卡协议。
- 新隔离 checkout
  `/data1/users/litianhao01/PairMOT_sphericalmidpointcenterlogshape_0804_12_178`
  固定 clean HEAD `d85e837`，未修改正在运行的 `0804_10` 仓库。目标球面中点单测
  `1/1 OK`；178 配置 deepcopy、父/新完整构建和两份 launcher 语法通过，父/新均为
  `22,771,111` 参数、711 states、增量 0，smoke 为 4 iter。
- 该路径仍严格为 `PREPARED/NO_GPU`：没有抢占 178 当前 GPU0，没有创建 smoke/formal
  workdir，也没有 checkpoint 或 formal iter50 证据。只有 `0804_10` 至少完成 e8/e12
  成熟窗口并合法释放单卡后，才依次执行真实 smoke、checkpoint 审计和 formal iter50 五门槛。
- 12:27 现场复核：252 固定 GPU0/1 到 e40 iter400、GPU2/3 不用；178 仅本线 GPU0 到
  e7 iter600，GPU1 外部任务不动；两线 total/DN/Encoder/grad 有限。

## 2026-08-05 12:15 CST：center-tangent e4 闭环；球面中点候选静态就绪

- 99 `0804_11 center-tangent + log-shape consensus` e4 同一 checkpoint 的 cls
  HOTA/DetA/AssA 为 `30.433/24.962/39.419`，det 为 `37.650/32.515/44.764`。相对其成熟
  terminal log-shape 父线 `0803_13` e4 的 `32.849/37.319`，HOTA 为 `-2.416/+0.331`；
  cls DetA/AssA 为 `-1.720/-4.502`，det 为 `-2.433/+3.521`。这说明 center tangent 在
  det 侧发生 DetA→AssA 交换，同时损伤 cls，而不是整体优于父线。pair mAP/AP50
  `0.1336/0.2565`、both-independent `0.1748/0.3227`，相对父线分别为
  `-0.0063/-0.0048/-0.0133/-0.0164`；相对直接 product-tangent e4，cls/det HOTA 亦低
  `4.841/6.199`。同点和仅 `68.083`，距严格总和 `118.330` 仍差 `50.247`。
- `epoch_4.pth` 为 369,965,878 bytes，meta `4/4152`；12 个 iterative-cls residual 最大
  绝对值 `0.0673615`，iterative-cls/DN 已训练且有限，642 个浮点张量全有限。检测为
  5416 records/50 sequences；TrackEval 产出 28 CSV、108 个非空文件、50 个非空预测和
  `async_done=1`，用时 242.9 秒。该点仅作早期机制诊断，不以 e4 否决；正式 PGID
  `1791967` 已恢复到 e5 iter500，GPU0/1 正常，继续收集 e8/e12 及成熟节点，GPU2 外部任务不动。
- 下一单因素候选 `0804_12 spherical-midpoint center + mature log-shape consensus` 已在 99
  全新隔离 checkout `/data/users/wangying01/lth/PairMOT_sphericalmidpointcenterlogshape_0804_12_99`
  固定 clean HEAD `4f6d563`。它保留成熟 log-size/周期角共识，只把 center 的反对称 detail
  方向改为学习方向与既有运动方向的最短符号球面中点，并精确保留原 detail 范数；零参数/state、
  交换等变、class-agnostic、无 reweight，且不增加层、attention、loss 或主矩阵计算。
  定向单测通过；配置 deepcopy、完整构建与 launcher 语法通过，父/新均为 `22,771,111`
  参数、711 states，增量 0。当前严格登记 `PREPARED/NO_GPU`：尚未占用 GPU、尚未通过真实
  DDP smoke、checkpoint 与 formal iter50，不能写作 RUNNING；等现有 99 双卡路线到成熟节点
  并释放后，再按五项动态门槛决定是否接替。
- 12:15 资源复核：252 固定 GPU0/1 到 e39 iter800，GPU2/3 均 `1 MiB/0%`；178 仅本线
  GPU0 到 e6 iter800，GPU1 外部任务不动；99 仅本线 GPU0/1 到 e5 iter500，GPU2 外部任务
  不动。三线正式日志的 total/DN/Encoder/grad 均有限，未热更新任何存活训练仓库。

## 2026-08-05 11:55 CST：covariant-Frenet e4 闭环；197 主机降频保护停线

- 178 `0804_10 covariant-Frenet product-tangent` e4 同一 checkpoint 的 cls
  HOTA/DetA/AssA 为 `34.189/27.630/45.087`，det 为 `36.612/33.119/41.820`。相对直接
  product-tangent e4 `35.274/43.849`，HOTA 为 `-1.085/-7.237`；cls DetA/AssA 为
  `-0.899/-0.038`，det 为 `-1.214/-16.280`，早期损失主要来自 det 关联崩落，而不是单纯
  定位交换。pair mAP/AP50 `0.1436/0.2761`，both-independent `0.1884/0.3473`，相对父线
  四项分别低约 `0.0157/0.0204/0.0185/0.0231`。
- 369,974,324-byte checkpoint meta `4/4152`；12 个 iterative-cls residual 最大绝对值
  `0.0634495`，iterative-cls/DN 已训练且有限，642 个浮点张量全有限。5416 records/
  50 sequences、28 CSV、108 个非空文件、50 个非空预测和 `async_done=1` 完整，TrackEval
  227.1 秒。该点只作早期归因，不以 e4 否决；PGID `3856480` 已到 e5 iter450，继续 e8/e12，
  仍仅使用动态 GPU0，GPU1 外部任务不动。
- 197 `0804_09` 原训练在 e12 step12418 后超过 39 分钟无 scalar/checkpoint 更新，GPU0/1
  降至 0–2%，两个 rank 在用户态持续自旋。进一步只读审计确认全机 80 核仅约
  `118–167 MHz`、温度 `44–46°C`，独立 `import torch` 20 秒超时，属于主机级异常降频而非
  模型、checkpoint 或共享盘错误。原 PGID `2390925` 经 TERM 后成员归零；GPU0/1 连续为
  `1 MiB/0%` 后曾从可信 e8 checkpoint 尝试隔离恢复，但 torchrun 同样因降频无法及时生成
  rank，未达到 formal iter50，故精确停止恢复 PGID `4191760`/screen `4191757`，未登记 RUNNING。
  e8 checkpoint、截至 step12418 的 scalars 与已有 e4/e8 全量评估均保留；外部 GPU4/5 任务
  未触碰，不修改整机 governor，等待主机恢复后再续跑 e12。

## 2026-08-05 11:27 CST：product-tangent e36 完整闭环，继续 e40 成熟复核

- 252 `0804_01 product-tangent` e36 同一 checkpoint 的 cls HOTA/DetA/AssA 为
  `53.326/44.846/65.452`，det 为 `59.293/52.962/68.706`，同点和 `112.619`；距离严格
  `54.437/62.393/118.330` 三门槛仍差 `1.111/3.100/5.711`，不得登记成功。相对 e32 HOTA
  仅 `+0.017/-0.027`、总和 `-0.010`，说明成熟轨迹曲线已经接近平台；但 pair mAP/AP50
  升至 `0.3057/0.5242`，both-independent 升至 `0.3479/0.5678`，因此保留 e40+ 复核，
  不按单点波动提前停止。
- `epoch_36.pth` 为 414,028,918 bytes，meta `36/37368`；12 个 iterative-cls residual
  均已训练且有限，最大绝对值 `0.1192054`，642 个浮点张量全有限。检测为 5416 records/
  50 sequences；TrackEval 产出 28 个 CSV、108 个非空文件、50 个非空预测，
  `async_done=1`，用时 415.6 秒。正式 PGID `823929` 已恢复 e37，仍固定 GPU0/1，
  GPU2/3 保持空闲。
- 同步审计：178 `0804_10` 到 e4 iter350，99 `0804_11` 到 e2 iter800；两线日志的
  total/DN/Encoder/grad 有限。197 `0804_09` 主进程及 GPU0/1 仍活跃，但共享盘 I/O 拥塞，
  尚未稳定产出 e12 checkpoint；不重启、不迁移，继续等待同点完整评估。各机外部 GPU 任务
  均未触碰。

## 2026-08-05 10:58 CST：center-tangent + log-shape consensus 通过 99 五项门槛

- 现有归因显示：直接 product-tangent 的 shape transport 在 e12 相对原 decoder 为
  `+0.117/-0.679`，而 terminal log-size/周期角共识在成熟 e56 达到 `54.980/62.009`；
  shared-metric、axis-Frenet 与 Householder 又分别因尺度混合、横向 detail 删除或过度旋转同步
  损伤 DetA/AP/AssA。因此 `0804_11` 以成熟 terminal log-shape consensus 为父结构，只在
  最后一层 normal query 新增原 product-tangent 的 2D center detail 投影；分类、DN、loss、
  attention、层数、递归 reference 与辅助输出全不变。它是单因素、零参数/state、交换等变、
  class-agnostic、无 reweight 的轻量候选。
- 本地 commit `eec6fc9` 部署到 99 全新隔离 checkout
  `/data/users/wangying01/lth/PairMOT_terminalcenterlogshape_0804_11_99`；定向测试覆盖末层调用顺序
  `center→log-size→periodic-angle`、DN 精确保留、交换等变、有限梯度与互斥。配置 deepcopy、
  两份 launcher 语法及父/新完整构建通过：均为 `22,771,111` 参数、711 states，增量 0。
- 99 GPU0/1 连续两次为 `10 MiB/0%` 后执行真数据 DDP smoke，四步 loss
  `12.9358/19.4522/19.5966/21.1587`、grad
  `105.6080/162.7252/154.4735/142.0009`，total、DN、Encoder proposal 全有限；
  364,505,078-byte `iter_4.pth` 的 iterative-cls/DN 更新与 642 个浮点张量有限性审计通过。
- fresh formal screen `1791965.formal_0804_11_99`、PGID `1791967`；iter50 为
  `0.9719 s/iter`、loss/grad `21.4245/134.0124`，7 个进程组成员，GPU0/1 各约
  19.2 GiB，正式 total/DN/Encoder 有限且无 Traceback/OOM/NaN/NCCL error。五项门槛齐全，
  登记 `RUNNING/TO_E12+`；e4/e8 只作诊断，不作直接否决，GPU2 约 6 GiB 外部任务始终未动。
- 10:58 同步审计：252 固定 GPU0/1 已到 e36、GPU2/3 空闲；197 Householder 已到 e12，仍仅
  使用动态 GPU0/1；178 `0804_10` 已到 e2，仍仅 GPU0；各机外部进程不触碰。继续优先闭环
  252 e36、197 e12，并按计划收集 178/99 的 e4/e8/e12 与更晚成熟节点。

## 2026-08-05 10:38 CST：axis/shared-metric e12 成熟收口；covariant-Frenet 正式接替 178

- 178 `0804_07 axis-Frenet` e12 cls HOTA/DetA/AssA `48.628/40.912/59.925`，det
  `54.069/48.455/62.596`，同点和 `102.697`，距离严格总和仍差 `15.633`。相对直接
  product-tangent e12 HOTA 为 `-1.156/-2.174`，DetA/AssA 分别为
  `-0.953/-1.590` 与 `-1.566/-3.007`；pair mAP/AP50 `0.2617/0.4600`、
  both-independent `0.3092/0.5180`，也分别低 `0.0141/0.0186` 与 `0.0122/0.0130`。
  381,099,572-byte checkpoint meta `12/12456`，12 个 residual 最大绝对值 `0.1036803`，
  642 个浮点张量全有限；5416/50/28/108、50 个非空预测、`async_done=1` 与 254.8 秒
  TrackEval 完整。e4/e8/e12 三节点相对强父线均负，精确 TERM PGID `3752856`，成员
  `9→0`；GPU0 回到 `1 MiB`，全部产物保留。
- 99 `0804_08 shared-metric` e12 cls HOTA/DetA/AssA `44.241/36.331/56.477`，det
  `52.755/47.148/61.146`，相对直接 product-tangent e12 HOTA `-5.543/-3.488`；pair
  mAP/AP50 `0.2256/0.4012`、both-independent `0.2676/0.4553`，定位、关联和 AP 均未恢复。
  381,050,294-byte checkpoint meta `12/12456`，12 个 residual 最大绝对值 `0.0989550`，
  642 个浮点张量全有限；5416/50/28/108、50 个非空预测、`async_done=1` 与 287.7 秒
  TrackEval 完整。e4/e8/e12 成熟双负后精确 TERM PGID `1751255`，成员 `23→0`；GPU0/1
  回到 `10 MiB`，GPU2 约 7 GiB 外部任务始终未动。
- `0804_10` 只在 `0804_07` 完整闭环且 GPU0 释放后上机。真实单卡四步 smoke loss
  `21.3690/20.7041/20.9412/21.2466`、grad
  `60.4018/65.3105/88.5642/89.0179`，总、DN、Encoder 均有限；364,507,636-byte
  `iter_4.pth` 的 iterative-cls/DN 语义与 642 个浮点张量全有限。fresh formal screen
  `3856478.formal_0804_10_178`、PGID `3856480` 在 iter50 为 `0.9524 s/iter`、loss/grad
  `21.0102/93.6656`，9 个成员，GPU0 约 31.4 GiB；正式 total/DN/Encoder 有限、fatal 0。
  配置 deepcopy、完整构建、真实 smoke、checkpoint 与 formal iter50 五门槛齐全，现登记
  `RUNNING/TO_E12+`。GPU1 约 5.9 GiB/92% 的外部任务不触碰。
- 10:40 复核：252 固定 GPU0/1 到 e35，GPU2/3 空闲；178 `0804_10` 仅 GPU0 到 e1
  iter450，GPU1 外部任务不动；197 `0804_09` 仅动态 GPU0/1 到 e11，GPU5 新外部任务不动；
  99 GPU0/1 已释放，GPU2 外部任务不动。所有授权卡与正式进程、日志一致。

## 2026-08-05 10:03 CST：product-tangent e32 与 Householder e8 闭环；covariant-Frenet 静态就绪

- 252 `0804_01 product-tangent` e32 cls HOTA/DetA/AssA `53.309/44.791/65.462`，det
  `59.320/52.822/68.981`，同 checkpoint 和 `112.629`。距离严格
  `54.437/62.393/118.330` 三门槛仍差 `1.128/3.073/5.701`，不得登记成功；但相对 e28
  双升 `+0.668/+0.334`、联合缩小 `1.002`，cls/det DetA 分别升 `+0.323/+0.309`，
  pair mAP/AP50 升至 `0.3037/0.5206`、both-independent 升至 `0.3472/0.5663`，成熟曲线
  尚未平台。408,537,142-byte checkpoint meta 为 `32/33216`，12 个 residual 最大绝对值
  `0.1143389`，642 个浮点张量全有限；5416/50/28/108、50 个非空预测文件和
  `async_done=1` 完整，TrackEval 用时 393.7 秒。PGID `823929` 已恢复 e33，固定 GPU0/1
  继续 e36+，GPU2/3 保持 `1 MiB`。
- 197 `0804_09 Householder product-tangent` e8 cls HOTA/DetA/AssA
  `42.596/34.975/54.658`，det `47.448/43.295/53.888`，相对直接 product-tangent e8
  HOTA 为 `-4.077/-6.474`，DetA/AssA 为 `-4.916/-1.937` 与 `-3.967/-10.248`。
  pair mAP/AP50 `0.2045/0.3731`、both-independent `0.2503/0.4358`，同样明显低于父结构。
  375,529,191-byte checkpoint meta `8/8304`，12 个 residual 最大绝对值 `0.0788621`，
  642 个浮点张量全有限；5416/50/28/108、50 个非空预测与 `async_done=1` 完整，TrackEval
  用时 321.3 秒。该点只说明保范数传输未修复中期检测与关联，不作为 e8 早停理由；PGID
  `2390925` 已恢复 e9并继续 e12。
- 新 `0804_10` 只改变 `0804_01` 的终层 center transport：以 detached 前后参考朝向的半程
  旋转把两帧完整 2D update 搬到共同 Frenet 坐标，在该坐标仅把反对称不一致投影到 detached
  chord，再旋回各帧；shape product-tangent、分类、DN、loss、层数与 attention 全不变。零转角
  时严格退化为 `0804_01`，保持交换等变、class-agnostic、无 reweight、零参数/零 state 增量，
  只增加逐元素三角运算。隔离 clean HEAD `15213f6` 的 2 项定向测试通过，完整父/候选构建均为
  `22,771,111` 参数、711 states，配置 deepcopy 与两份 launcher 语法通过；等待 178 e12 后按
  五项动态门槛验证，当前不是 `RUNNING`。
- 10:02 四机复核：252 仅 GPU0/1 到 e33、178 仅动态 GPU0 到 e11、99 动态 GPU0/1 到
  e10且 GPU2 外部任务不动、197 动态 GPU0/1 到 e9；四线正式日志 total/DN/Encoder/grad
  均有限，无 fatal，未热更新存活仓库。

## 2026-08-05 09:25 CST：axis-Frenet 与 shared-metric e8 同点闭环

- 178 `0804_07 axis-Frenet product-tangent` e8 cls HOTA/DetA/AssA
  `44.638/36.575/57.023`，det `51.044/44.642/60.973`，同点和 `95.682`，距离严格
  `54.437/62.393/118.330` 三门槛分别仍差 `9.799/11.349/22.648`。相对直接
  product-tangent e8，cls/det HOTA 为 `-2.035/-2.878`；cls DetA/AssA 为
  `-3.316/+0.428`，det 为 `-2.620/-3.163`。方向分解只在 cls AssA 保留轻微正差，
  并未修复主要定位与 det 关联缺口。
- axis-Frenet e8 pair mAP/AP50 `0.2308/0.4073`、both-independent
  `0.2765/0.4691`，相对直接 product-tangent 分别下降 `0.0216/0.0370` 和
  `0.0274/0.0448`。375,568,692-byte checkpoint 的 12 个 iterative-cls residual
  张量已训练且有限，最大绝对值 `0.0921249`；642 个浮点张量全有限。5416/50/28/108、
  50 个预测文件及 `async_done=1` 闭环，TrackEval 用时 242.4 秒。它相对原 decoder e8
  仍为 `+2.666/+2.866`，所以不在 e8 淘汰，PGID `3752856` 已恢复 e9并继续 e12。
- 99 `0804_08 shared-metric product-tangent` e8 cls HOTA/DetA/AssA
  `41.011/33.708/53.262`，det `46.607/42.601/52.615`，同点和 `87.618`，严格总和仍差
  `30.712`。相对直接 product-tangent e8，cls/det HOTA 为 `-5.662/-7.315`；
  cls DetA/AssA 为 `-6.183/-3.333`，det 为 `-4.661/-11.521`。共享几何均值度量同时
  损伤定位与关联，并非只改变一个侧面的尺度噪声。
- shared-metric e8 pair mAP/AP50 `0.2001/0.3607`、both-independent
  `0.2432/0.4210`，相对直接 product-tangent 分别下降 `0.0523/0.0836` 与
  `0.0607/0.0929`。375,547,190-byte checkpoint meta 为 epoch/iter `8/8304`，
  12 个 iterative-cls residual 张量最大绝对值 `0.0797748`，642 个浮点张量全有限；
  5416/50/28/108、50 个预测文件及 `async_done=1` 闭环，TrackEval 用时 261.9 秒。
  按慢收敛规则仍保留 e12 成熟节点，PGID `1751255` 已恢复 e9。
- 09:25 四机复审：252 固定 GPU0/1 到 e31 iter400，GPU2/3 为 `1 MiB/0%`；178 仅本任务
  GPU0 到 e9，GPU1 空闲；99 本任务动态 GPU0/1 到 e9 iter350，GPU2 的约 6 GiB 外部任务
  保持不动；197 本任务动态 GPU0/1 到 e6 iter1000。四条正式日志的 total/DN/Encoder/grad
  均有限，未热更新存活仓库或改变外部任务。

## 2026-08-05 08:51 CST：Householder e4 完整闭环，范数保持仍未修复早期关联

- 197 `0804_09 norm-preserving Householder product-tangent` e4 cls HOTA/DetA/AssA
  `31.605/25.498/41.950`，det `37.701/31.744/45.689`。相对原 decoder e4
  `34.306/38.590` 为 `-2.701/-0.889`，相对 Encoder e4 `36.209/38.753` 为
  `-4.604/-1.052`；该早期点不满足最终绝对三门槛，但不据此淘汰。
- 相对直接 product-tangent e4 `35.274/43.849`，cls/det HOTA 为
  `-3.669/-6.148`；cls DetA/AssA 分别 `-3.031/-3.175`，det 分别
  `-2.589/-12.411`。pair mAP/AP50 `0.1351/0.2542`、both-independent
  `0.1753/0.3184`，相对父结构分别下降 `0.0242/0.0423` 与 `0.0316/0.0520`。
  因此早期瓶颈仍以 det 关联和检测 AP 同时受损为主，单纯把 rank-one 投影换成保范数
  正交传输尚未恢复父结构的收敛速度。
- 369,968,615-byte checkpoint 的 12 个 iterative-cls residual 张量已训练且全有限，
  最大绝对值 `0.0611005`；642 个浮点张量全有限。5416 条检测、50 序列、28 CSV、
  108 个非空评测文件、50 个预测文件及 `async_done=1` 完整，TrackEval 用时 273.9 秒。
  该结果只用于早期归因；PGID `2390925` 已恢复到 e5 iter350，继续 e8/e12。
- 08:50 四机复审：252 固定 GPU0/1 到 e30 iter150，GPU2/3 空闲；178 仅本任务 GPU0
  到 e8 iter650；99 本任务动态 GPU0/1 到 e7 iter650，GPU2 不用；197 本任务动态
  GPU0/1 到 e5 iter350。四条正式日志的 total/DN/Encoder/grad 均有限，未改变外部任务。

## 2026-08-05 08:34 CST：252 product-tangent e28 继续检测增长，严格门槛未达

- 252 `0804_01` e28 cls HOTA/DetA/AssA `52.641/44.468/64.147`，det
  `58.986/52.513/68.606`，同 checkpoint 和 `111.627`。相对严格门槛
  `54.437/62.393/118.330` 分别仍差 `1.796/3.407/6.703`，因此不得登记成功。
- 相对 e24，cls/det HOTA `+0.163/+0.215`，DetA `+0.386/+0.437`，AssA
  `-0.330/-0.071`；pair mAP/AP50 从 `0.2973/0.5127` 升至 `0.3001/0.5156`，
  both-independent 从 `0.3412/0.5591` 升至 `0.3442/0.5622`。检测与定位仍在增长，
  关联仅轻微回落，故保留固定 GPU0/1 到 e32+，不在 e28 截断成熟线。
- 403,042,998-byte checkpoint 的 iterative-cls/DN 语义通过，642 个浮点张量全有限；
  5416 条检测、50 序列、28 CSV、108 个非空文件、50 个预测文件和 `async_done=1`
  全部闭环，TrackEval 用时 391.2 秒。训练在异步评估期间持续并已进入 e29/e30，GPU2/3
  始终未用于本任务。

## 2026-08-05 08:11 CST：shared-metric e4 完整闭环，保留慢收敛窗口

- 99 `0804_08 shared-metric product-tangent` e4 cls HOTA/DetA/AssA
  `31.368/26.157/40.232`，det `38.646/33.488/45.640`。相对原 decoder e4
  `34.306/38.590` 为 `-2.938/+0.056`，相对 Encoder e4 `36.209/38.753` 为
  `-4.841/-0.107`；该点显然未满足最终绝对三门槛。
- 相对直接 product-tangent e4 `35.274/43.849`，cls/det HOTA 为
  `-3.906/-5.203`；cls DetA/AssA 分别 `-2.372/-4.893`，det 分别
  `-0.845/-12.460`。共同 geometric-mean metric 在早期尤其损伤轨迹关联；pair mAP/AP50
  `0.1409/0.2654`、both-independent `0.1843/0.3349` 也相对父结构分别下降
  `0.0184/0.0311` 与 `0.0226/0.0355`。
- 369,971,894-byte checkpoint 的 12 个 iterative-cls residual 张量均已训练且有限，
  642 个浮点张量全有限；5416 条检测、50 序列、28 CSV、108 个非空评测文件及
  `async_done=1` 完整，TrackEval 用时 234.4 秒。该结果只用于早期归因，不作为 e4
  淘汰理由；PGID `1751255` 已恢复到 e5 iter250，继续 e8/e12。
- 08:10 复审：252 固定 GPU0/1 已到 e28 iter500；178 本任务 GPU0 到 e6 iter50；99
  本任务动态 GPU0/1 到 e5 iter250；197 本任务动态 GPU0/1 到 e3 iter100。GPU2/3（252）、
  178 GPU1 与 99 GPU2 均未被本任务占用或改动。

## 2026-08-05 08:01 CST：axis-Frenet e4 完整闭环，仅作早期归因

- 178 `0804_07 axis-Frenet product-tangent` e4 cls HOTA/DetA/AssA
  `35.434/28.624/45.484`，det `44.417/35.116/57.988`。相对原 decoder e4
  `34.306/38.590` 为 `+1.128/+5.827`，相对 Encoder e4 `36.209/38.753` 为
  `-0.775/+5.664`；仍远未满足最终绝对门槛，不能登记成功。
- 相对直接 product-tangent e4 `35.274/43.849`，axis-Frenet 的 cls/det HOTA
  `+0.160/+0.568`：cls DetA/AssA 分别 `+0.095/+0.359`，det 分别
  `+0.783/-0.112`。方向分解在早期把一部分 det 关联优势换成了更明显的定位回补；但
  pair mAP/AP50 `0.1584/0.2908`、both-independent `0.2036/0.3605` 相对父结构仍为
  `-0.0009/-0.0057` 与 `-0.0033/-0.0099`，因此不能由 e4 的 HOTA 微升推断成熟优势。
- 369,976,116-byte checkpoint 的 iterative-cls/DN 语义与 642 个浮点张量全有限；
  5416 条检测、50 序列、28 CSV、108 个非空评测文件及 `async_done=1` 完整，TrackEval
  用时 221.4 秒。按慢收敛规则不在 e4 淘汰，PGID `3752856` 已恢复到 e5 iter400，继续
  收集 e8/e12。
- 08:00 四机审计：252 固定 GPU0/1、PGID `823929` 到 e27 iter1000，GPU2/3 空闲；178
  仅本任务 GPU0，GPU1 外部任务保持；99 本任务动态 GPU0/1 到 e4 iter1000，GPU2 外部
  任务保持；197 本任务动态 GPU0/1 到 e2 iter450。正式日志的 total/DN/Encoder/grad
  均有限，进程、GPU、正式目录一致，未触碰外部任务。

## 2026-08-05 07:40 CST：Frenet e12 成熟停止；Householder 保能传输接替 197

- 197 `0804_06 Frenet` e12 cls HOTA/DetA/AssA `44.810/36.342/57.539`，det
  `51.879/45.809/61.000`；相对原 decoder e12 `47.395/54.436` 为
  `-2.585/-2.557`，相对 Encoder e12 `49.680/56.541` 为 `-4.870/-4.662`。
  pair mAP/AP50 `0.2243/0.3983`、both-independent `0.2672/0.4546`；
  381,030,887-byte checkpoint 的 iterative-cls/DN 语义与 642 个浮点张量全有限，
  5416 条检测、50 序列、28 CSV、108 个非空文件及 `async_done=1` 完整，TrackEval
  用时 326.4 秒。
- 该线已具备 e4/e8/e12 三个完整节点且持续双负，故在 e12 异步评测闭环后精确 TERM
  PGID `482699`，成员 `23→0`；六张 GPU 连续检查均回到 `1 MiB/0%`。这是成熟窗口
  判定，不是 e4/e8 早停。
- e12 的 DetA 与检测 AP 同时明显下降，证据指向 rank-one 投影删除横向几何 detail，
  而不只是坐标系选择错误。新 `0804_09` 保持 center 2D / shape 3D product bundle 与
  所有分类、DN、辅助输出、递归 reference 不变，只把两个 rank-one detail 投影替换为
  最小正交 Householder 平行传输：detail 对齐既有运动轴但范数严格保持。结构零参数、
  零 state 增量、交换等变、class-agnostic，无 prediction reweight、新层、attention 或 loss。
- 独立 clean HEAD `84fa6cc` 的定向测试、配置/烟测 deepcopy、两份 launcher 语法与完整
  父/候选构建通过：均为 `22,771,111` 参数、711 states、增量 0。动态 GPU0/1 上真实
  DDP smoke 四步 loss `12.9412/19.4368/19.6046/21.1094`、grad
  `102.5677/89.2892/81.2947/90.7800`；364,505,255-byte checkpoint 的 iterative-cls/DN
  语义与 642 个浮点张量全有限，fatal 0。
- fresh formal screen/PGID `2390923/2390925` 在 iter50 为 `0.9251 s/iter`、loss/grad
  `21.4022/108.1453`，7 个成员，GPU0/1 各约 19.2 GiB，其他 GPU 空闲，正式日志中
  total/DN/Encoder 均有限且 fatal 0。五项动态门槛全部通过，登记 `RUNNING/TO_E12+`。
- 07:38 四机复审：252 固定 GPU0/1 到 e26 iter950；178 动态 GPU0 到 e4 iter800，
  GPU1 外部任务不动；99 动态 GPU0/1 到 e3 iter650，GPU2 外部任务不动；197 动态
  GPU0/1 为新线 iter50。四条正式线进程、GPU、日志一致且 fatal 均为 0。

## 2026-08-05 07:08 CST：e24 继续双升；两条成熟负线由 0804_07/08 接替

- 252 `0804_01` e24 cls HOTA/DetA/AssA `52.478/44.082/64.477`，det
  `58.771/52.076/68.677`，同 checkpoint 和 `111.249`；严格门槛
  `54.437/62.393/118.330` 分别仍差 `1.959/3.622/7.081`，尚未达标。
  相对原 decoder e24 `51.709/58.781` 为 `+0.769/-0.010`、联合 `+0.759`；
  相对 Encoder e24 `51.714/59.519` 为 `+0.764/-0.748`、联合 `+0.016`。
  e20→e24 仍双升 `+0.280/+0.639`，pair mAP/AP50 `0.2973/0.5127`、
  both-independent `0.3412/0.5591` 也继续上升，因此固定 GPU0/1 保留到 e28+，
  不把 e24 尚未过线当作停止理由；GPU2/3 不用于本任务。
- 397,549,046-byte e24 checkpoint 的 iterative-cls/DN 语义和 642 个浮点张量全有限；
  5416 条检测、50 序列、28 CSV、108 个非空评测文件与 `async_done=1` 全部闭环，
  TrackEval 用时 399.5 秒。07:07 PGID `823929` 已到 e25 iter350，fatal 0。
- 178 `0804_04 body-frame` e12 cls HOTA/DetA/AssA `47.220/37.937/61.436`，det
  `53.882/47.177/63.450`；pair mAP/AP50 `0.2414/0.4132`、both-independent
  `0.2853/0.4658`。相对原 decoder e12 `-0.175/-0.554`、相对 Encoder
  `-2.460/-2.659`。e4/e8/e12 三点完整后才精确停止 PGID `3652382`，成员 `8→0`，
  不是 e4/e8 早停；381,084,660-byte checkpoint、5416/50/28/108 与异步完成均完整。
- 178 GPU0 连续两轮为 `1 MiB/0%` 后，`0804_07 axis-Frenet` 真实单卡 smoke 四步
  loss `21.3688/20.6416/20.8939/21.1817`、grad
  `60.1890/67.1206/97.9977/90.9295`，364,507,572-byte checkpoint 的 642 个浮点张量
  全有限。fresh formal screen/PGID `3752854/3752856`，iter50 `0.9441 s/iter`、
  loss/grad `20.9969/156.2116`，DN/Encoder 有限、fatal 0；五门槛通过，登记
  `RUNNING/TO_E12+`。07:07 已到 e2 iter800，GPU0 约 31.5 GiB，GPU1 未动。
- 99 `0804_05 SE(2)` e12 cls HOTA/DetA/AssA `45.531/37.305/57.963`，det
  `51.727/46.529/59.516`；pair mAP/AP50 `0.2304/0.4116`、both-independent
  `0.2739/0.4678`。相对原 decoder e12 `-1.864/-2.709`、相对 body-frame
  `-1.689/-2.155`；e4/e8/e12 完整持续双负后精确停止 PGID `1715384`，成员 `23→0`。
- 99 全机三卡两轮均为 `10 MiB/0%` 后，本次动态选择 GPU0/1 验证 `0804_08`；真实双卡
  smoke 四步 loss `12.9369/19.5109/19.6569/21.2423`、grad
  `103.4712/141.1132/120.5418/136.1262`，364,504,438-byte checkpoint 的 642 个浮点张量
  全有限。fresh formal screen/PGID `1751253/1751255` 已到 iter100，loss/grad
  `20.5120/112.2396`，DN/Encoder 有限、fatal 0；五门槛通过，登记 `RUNNING/TO_E12+`。
  07:07 已到 e1 iter700；随后出现的 GPU2 外部任务约 6.0 GiB/97%，本任务不触碰。
- 197 `0804_06 Frenet` e8 cls HOTA/DetA/AssA `40.641/33.920/50.843`，det
  `47.268/42.797/54.147`；pair mAP/AP50 `0.1977/0.3616`、both-independent
  `0.2427/0.4267`。相对原 decoder e8 `-1.331/-0.910`、相对 Encoder
  `-4.628/-2.925`；375,534,503-byte checkpoint、5416/50/28/108 与 322.4 秒异步完成
  均完整。按规则继续 e12，07:07 PGID `482699` 已到 e12 iter450，GPU2/3 动态占用，
  其他卡不动。

## 2026-08-05 06:00 CST：shared-metric product tangent 静态闭环；四线健康推进

- 现有最强 `0804_01` 对前后帧 center update 分别用各自宽高归一化，却把二者 detail
  与 geometric-mean pair-size 归一化的 chord 比较；尺寸变化时三者不在同一度量基。
  新 `0804_08` 只把前帧 update、后帧 update、chord 与重构统一到同一个几何均值宽高度量，
  shape tangent、分类、DN、辅助输出、递归 reference、层、attention 和 loss 均不变。
- 该结构零参数、零 state 增量、class-agnostic、无 reweight、交换等变；两帧尺寸相同时严格
  退化为 `0804_01`。定向测试覆盖末层唯一调用、DN 精确保留、交换等变、有限梯度、互斥、
  等尺寸退化和非等尺寸因子激活，2/2 通过。正式/烟测配置 deepcopy 与 launcher 语法通过；
  父/候选完整构建均为 `22,771,111` 参数、711 states、增量 0。
- 隔离提交 `f9b923b` 已放入 99 新 checkout
  `/data/users/wangying01/lth/PairMOT_terminalsharedmetric_0804_08_99`；活跃 `0804_05`
  仓库保持 clean。99 GPU0/1 仍由 `0804_05` 占用，所以未创建 smoke/formal workdir，
  只登记 `PREPARED/NO_GPU`；释放后仍须依次通过真实双卡 smoke、有限 checkpoint 与 formal
  iter50 三道动态门槛，绝不把静态构建写成 RUNNING。
- 05:58 实审：252 仅 GPU0/1、已到 e22 iter250；178 仅 GPU0、已到 e11 iter600；
  99 动态 GPU0/1、已到 e10 iter400；197 动态 GPU2/3、已到 e8 iter450。四条正式日志
  total/DN/Encoder/grad 有限且 fatal 扫描均为 0；尚无新的 e24/e12/e8 checkpoint，继续等待
  完整 checkpoint、检测、AP 与 TrackEval 闭环，不用中间 epoch 直接否决。

## 2026-08-05 05:41 CST：product-tangent e20 继续上升但严格未达标

- 252 `0804_01` e20 cls HOTA/DetA/AssA `52.198/43.979/63.843`，det
  `58.132/51.652/67.820`；同 checkpoint 和 `110.330`，严格绝对门槛
  `54.437/62.393/118.330` 分别仍差 `2.239/4.261/8.000`，不能登记成功。
- 相对原 decoder e20 `50.843/58.033` 为 `+1.355/+0.099`，联合 `+1.454`；
  相对 Encoder e20 `51.514/58.922` 为 `+0.684/-0.790`。e16→e20 继续双升
  `+0.978/+0.762`，pair mAP/AP50 由 `0.2914/0.5039` 升至 `0.2958/0.5113`，
  both-independent 由 `0.3379/0.5551` 升至 `0.3395/0.5575`，曲线尚未平台。
- 392,051,702-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件
  与 393.1 秒异步完成标记完整。固定 GPU0/1 的 PGID `823929` 已进入 e21并继续 e24+；
  GPU2/3 不用于本任务，不以 e20 未过绝对线停止仍在上升的成熟轨迹。

## 2026-08-05 05:38 CST：SE(2) Lie-twist e8 完整闭环

- 99 `0804_05` e8 cls HOTA/DetA/AssA `41.230/33.612/53.634`，det
  `46.977/42.481/54.032`，同 checkpoint 和 `88.207`；pair mAP/AP50
  `0.1972/0.3621`、both-independent `0.2415/0.4251`。
- 相对 body-frame e8，cls HOTA/DetA/AssA 为 `-0.417/-0.058/-0.597`，det 为
  `-0.473/+0.793/-2.168`：有限旋转把少量 det 定位换成更大的关联损失。相对原 decoder
  e8 `41.972/48.178` 为 `-0.742/-1.201`，相对 Encoder `45.269/50.193`
  为 `-4.039/-3.216`。
- 375,526,710-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件
  与 266.9 秒异步完成标记完整。e4→e8 双 HOTA 已恢复 `+9.251/+9.049`，因此动态
  GPU0/1 的 PGID `1715384` 继续 e12 成熟窗口，不以 e8 直接否决。

## 2026-08-05 05:23 CST：body-frame e8 完整闭环

- 178 `0804_04` e8 cls HOTA/DetA/AssA `41.647/33.670/54.231`，det
  `47.450/41.688/56.200`，同 checkpoint 和 `89.097`；pair mAP/AP50
  `0.2004/0.3618`、both-independent `0.2436/0.4184`。
- 相对原 decoder e8 `41.972/48.178` 为 `-0.325/-0.728`，相对 Encoder
  `45.269/50.193` 为 `-3.622/-2.743`，也低于轴归一化 product-tangent e8
  `46.673/53.922` 为 `-5.026/-6.472`。body-frame 从 e4 到 e8 虽双升
  `+7.530/+8.211`，但去掉 width/height 轴向 metric 后没有保留主要收益。
- 375,567,924-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件
  与 253.3 秒异步完成标记完整。该点只作为中期归因；PGID `3652382` 继续 e12 成熟窗口，
  不以 e8 直接否决，`0804_07` 仍为 `PREPARED/NO_GPU` 且不抢占单卡。

## 2026-08-05 05:09 CST：Frenet endpoint-tangent e4 完整闭环

- 197 `0804_06` e4 cls HOTA/DetA/AssA `31.656/26.584/39.932`，det
  `36.905/32.714/42.549`；同 checkpoint 和 `68.561`。pair mAP/AP50
  `0.1384/0.2648`、both-independent `0.1825/0.3378`。
- 相对 SE(2) e4，cls/det DetA 为 `+0.975/+1.170`，但 AssA 为
  `-2.152/-4.330`，HOTA 为 `-0.323/-1.023`；相对 body-frame e4 HOTA
  `-2.461/-2.334`。早期证据说明前后端点切向旋转没有换来定位优势，反而削弱关联一致性。
- 369,968,743-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件
  与 305.2 秒异步完成标记完整。该结果只作 e4 结构归因；动态 GPU2/3 的 PGID `482699`
  已进入 e5并继续 e8/e12，不以 e4 直接否决，其他 GPU 未动。

## 2026-08-05 04:40 CST：axis-Frenet product tangent 静态就绪

- 证据表明轴向 `0804_01` 在 e16 相对原 decoder 保留联合 `+1.621`，而 body-frame
  e4 丢失大部分早期优势、SE(2) e4 的 DetA/AP 又继续下降。因此准备 `0804_07`：
  保留 `0804_01` 的 width/height 轴向归一化与 shape tangent，只把单一共享弦方向按
  参考框 π 周期转角旋转为前/后 constant-turn endpoint tangent；零转角严格退化为
  `0804_01`。
- 该结构零参数、class-agnostic、无 reweight，不改分类、DN、辅助输出、递归 reference、
  layer、attention 或 loss，只在终层增加逐元素三角函数。交换等变、DN 精确保留、
  有限梯度、终层唯一调用、零转角退化两项定向测试全部通过。
- 本地提交 `452b629` 与审计修复 `fe7e9fe` 已送入 178 全新隔离 checkout
  `/data1/users/litianhao01/PairMOT_terminalaxisfrenet_0804_07`；活跃
  `0804_04` 仓库未修改。正式/烟测配置均完成 `copy.deepcopy`，两份 launcher
  `bash -n` 通过，父/候选完整构建均为 `22,771,111` 参数、711 states、增量 0。
- 当前只登记 `PREPARED/NO_GPU`：没有 smoke/formal workdir 或训练进程。先等待
  `0804_04` e8/e12 成熟证据；只有动态单卡真实释放后才执行 smoke、checkpoint 与
  formal iter50 五门槛，绝不把静态构建写成 RUNNING。

## 2026-08-05 04:23 CST：SE(2) Lie-twist e4 完整归因

- 99 `0804_05` e4 cls HOTA/DetA/AssA `31.979/25.609/42.084`，det
  `37.928/31.544/46.879`；相对原 decoder e4 `34.306/38.590` 为
  `-2.327/-0.662`，相对 Encoder e4 `36.209/38.753` 为 `-4.230/-0.825`。
- pair mAP/AP50 `0.1290/0.2500`、both-independent `0.1702/0.3194`；
  369,967,414-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件
  与 248.4 秒异步完成标记完整。相对 body-frame e4 `34.117/39.239` 为
  `-2.138/-1.311`，且 DetA/AP 下滑没有换来 AssA 优势，早期证据指向角耦合有限运动
  对中心定位过强。
- 该结果仅用于结构归因，不构成 e4 淘汰。动态 GPU0/1 的 PGID `1715384` 已进入 e5，
  total/DN/Encoder/grad 有限，继续 e8/e12；GPU 序号仍是本次动态选择而非固定分配。

## 2026-08-05 04:15 CST：product-tangent e16 联合优势超过原 decoder 1.5

- 252 `0804_01` e16 cls HOTA/DetA/AssA `51.220/43.809/61.567`，det
  `57.370/51.202/66.638`；同 checkpoint 和为 `108.590`，因此严格绝对门槛
  `54.437/62.393/118.330` 分别仍差 `3.217/5.023/9.740`，尚未达标。
- 相对原 decoder e16 `50.036/56.933` 为 `+1.184/+0.437`，联合 `+1.621`；
  相对 Encoder e16 `51.091/58.320` 为 `+0.129/-0.950`，相对 full-tangent
  e16 `49.627/56.820` 为 `+1.593/+0.550`。e12→e16 又提高
  `+1.436/+1.127`，pair AP/AP50 与 both-independent 也分别提高
  `+0.0156/+0.0253`、`+0.0165/+0.0241`。
- 386,552,438-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件均已核验；
  该节点虽未达最终目标，却确认分块 product tangent 在成熟区间保留明确联合增益。
  固定 GPU0/1 的 PGID `823929` 继续 e20/e24+；不因单个 e16 未过绝对线而停止。

## 2026-08-05 04:12 CST：178 e4 闭环；252 e16 TrackEval 存活

- 178 `0804_04` e4 cls HOTA/DetA/AssA `34.117/27.146/45.226`，det
  `39.239/34.041/46.965`；相对原 decoder e4 `34.306/38.590` 为
  `-0.189/+0.649`，相对 Encoder e4 `36.209/38.753` 为 `-2.092/+0.486`。
  pair mAP/AP50 `0.1398/0.2697`、both-independent `0.1870/0.3463`，
  5416 条检测、50 序列和 TrackEval 完整。该早期点显示 det 侧轻微正信号、cls 基本持平，
  PGID `3652382` 已健康进入 e5，继续保留 e8/e12 与成熟窗口，不以 e4 否决。
- 252 `0804_01` e16 pair mAP/AP50 `0.2914/0.5039`、both-independent
  `0.3379/0.5551`；同 checkpoint 异步 TrackEval 进程 PID `833515` 存活，
  评测目录文件数已由 2 增至 19，属于正在生成轨迹而非失败。PGID `823929` 已进入
  e17，GPU0/1 各约 21.6 GiB、GPU2/3 均为 1 MiB；待 summary CSV 落盘后严格核验
  `cls>54.437`、`det>62.393`、`sum>118.330`，同时继续成熟节点。
- 99 `0804_05` 已到 e4 iter900，197 `0804_06` 已到 e2 iter50；两条正式日志的
  total/DN/Encoder/grad 均有限。99 当前动态 GPU0/1、197 当前动态 GPU2/3，均未扩大卡数。

## 2026-08-05 04:02 CST：252 e16 与 178 e4 进入评估闭环

- 252 固定 GPU0/1 的 `0804_01` 已完成 epoch16 训练并生成 386,552,438-byte
  `epoch_16.pth`，PGID `823929` 的正式恢复日志在 e16 全程 total/DN/Encoder/grad 有限、fatal 为 0，
  04:01 进入 validation；GPU2/3 仍为 `1 MiB/0%`。等待 AP 与同 checkpoint TrackEval 完整后，
  再按 `54.437/62.393/118.330` 三重门槛判断。
- 178 动态 GPU0 的 `0804_04` e4 已完成 1354-batch validation；pair mAP/AP50
  `0.1398/0.2697`，both-independent `0.1870/0.3463`，5416 条检测与 50 序列完整，
  `val_track_eval/val_track_0001` 已异步启动。当前只登记 AP 与评测进行态，HOTA 未出前不做性能结论，
  更不会以 e4 直接否决；GPU1 外部任务未动。
- 99 动态 GPU0/1 的 `0804_05` 已进入 epoch4 iter50，iter50 loss/grad
  `11.6851/30.7542`，总、DN、Encoder proposal 有限，继续等待 e4 checkpoint/AP/TrackEval。

## 2026-08-05 03:54 CST：log-size-only e12 成熟收口；Frenet 通过五门槛

- `0804_03` e12 cls HOTA/DetA/AssA `44.836/36.679/57.472`，det
  `51.366/46.167/59.307`；相对原 decoder e12 `47.395/54.436` 为
  `-2.559/-3.070`。pair mAP/AP50 `0.2332/0.4164`、both-independent
  `0.2674/0.4572`；381,026,151-byte checkpoint、5416 条检测、50 序列、28 CSV、
  108 个非空评测文件和 `async_done=1` 全部闭环。e4/e8/e12 三个完整节点持续双负后，
  精确停止 PGID `2540932`；独立复核时已无成员，GPU2/3 连续两次为 `1 MiB/0%`。
  该决策来自 e12 成熟窗口，不是以 e4/e8 直接否决 decoder；GPU5 外部任务始终未动。
- 197 隔离仓库 clean HEAD `2c45640` 的 `0804_06` 在动态 GPU2/3 完成真实双卡 smoke：
  四步 loss `12.9369/19.5816/19.7056/21.2506`、grad
  `103.5813/99.5912/91.1271/93.4536`，总损失、DN、Encoder proposal 全有限；
  364,504,615-byte checkpoint 的 642 个浮点 tensor 全有限，致命错误扫描为 0。
- fresh formal screen `482698.pm_0804_06_formal_197`、PGID `482699`；iter50
  `1.0850 s/iter`、loss/grad `21.3925/110.7751`，7 个进程组成员、双 rank、GPU2/3
  各约 19.2 GiB、正式日志/workdir 与 clean commit 对齐，DN/Encoder proposal 有限且 fatal 为 0。
  五门槛通过，正式登记 `RUNNING/TO_E4+`；继续收 e4/e8/e12 和后续成熟节点，不以 e4/e8 早停。

## 2026-08-05 03:10 CST：Frenet endpoint-tangent 后继完成静态闭环

- 根据 product/body-frame 单弦投影可能压制转弯轨迹横向定位 detail 的风险，准备 `0804_06`
  constant-turn Frenet product tangent：shape tangent 不变；center detail 不再让前后帧共用弦方向，
  而是由参考 π 周期角速度把弦旋转 `-turn/2` 与 `+turn/2`，分别投影到圆弧前/后端点切线。
  交换帧序时两端 projector 对调且 detail 反号，零转角严格退化为 body-frame。该结构零参数、
  class-agnostic、无 reweight，不增加 layer/attention/loss，仅有终层逐元素三角函数和点积。
- 197 隔离仓库 clean HEAD `2c45640`；交换等变/有限梯度/DN 精确保留、零转角退化两项定向测试，
  正式/烟测配置 deepcopy、两份 launcher `bash -n` 和完整父/候选构建通过：
  `22,771,111` 参数、增量 0、711 states。当前严格登记 `PREPARED/NO_GPU`，没有 smoke/formal
  workdir；先等待 0804_03 e12 完整 checkpoint/AP/TrackEval，且只在动态两卡真正释放后再决定
  是否进入五门槛，绝不热更新当前活跃 197 仓库。

## 2026-08-05 03:04 CST：SE(2) Lie-twist 在 99 通过五门槛；四线并行进入有效轨迹

- `0804_05` 是 product-tangent 的一个单因素几何后继：shape tangent 先保持 0804_01 的运输投影，
  随后仅把最终 center correction 改写为由已运输角增量驱动的 SE(2) midpoint Lie twist，沿参考轨迹
  投影后用匹配 Jacobian 回缩。分类、DN、辅助输出、递归 reference、层、attention、loss 均不变；
  零参数、零 state 增量、class-agnostic、无 reweight，终层只增加逐元素 `sinc`/三角运算。
- 99 隔离仓库 clean HEAD `f2c60a9`；交换等变、零角退化为 body-frame、整体旋转等变三项定向测试，
  正式/烟测配置 deepcopy、两份 launcher `bash -n` 和完整父/候选构建均通过：
  `22,771,111` 参数、增量 0、711 states。动态 GPU0/1 两轮均为 `10 MiB/0%` 后，真实双卡 smoke
  loss `12.9369/19.4810/19.5814/21.1211`、grad
  `103.5530/98.0642/155.6795/142.9127`；total、DN、Encoder proposal、642 个 checkpoint
  浮点 tensor 全有限。
- fresh formal screen `1715383.pm_0804_05_formal_99`、PGID `1715384`，iter50
  `0.9695 s/iter`、loss/grad `21.3647/108.6043`；双 rank、动态 GPU0/1 各约 19.2 GiB，
  正式目录/日志持续更新且无 fatal，五门槛通过，登记 `RUNNING/TO_E4+`。e4/e8 仅作中间归因，
  不直接否决 decoder；先收齐 e4/e8/e12，再结合 DetA/AssA/AP 与轨迹决定延长。

## 2026-08-05 02:53 CST：product-tangent 成熟迁移；angle-only 收口；body-frame 启动

- 178 `0804_01` e12 cls HOTA/DetA/AssA `49.784/41.865/61.515`，det
  `56.243/50.021/65.603`；相对原 decoder e12 `47.395/54.436` 为 `+2.389/+1.807`，联合
  `+4.196`，相对 Encoder `49.680/56.541` 为 `+0.104/-0.298`，相对 full-tangent
  `50.145/56.375` 为 `-0.361/-0.132`。pair mAP/AP50 `0.2758/0.4786`、both-independent
  `0.3214/0.5310`；checkpoint、5416/50/28/108 与 async 标志完整。它尚未严格达标，但保留了
  显著成熟优势，因此不是淘汰，而是把 e12 迁到最慢资源继续观察晚收敛。
- 252 隔离仓库 `f356593` 从同一 e12 checkpoint 恢复到固定 GPU0/1，screen
  `823928.pm_0804_01_resume252`、PGID `823929`；日志确认 `resumed epoch:12, iter:12456`，
  正式 iter50 `1.2182 s/iter`、loss/grad `9.9167/39.2891`，total/DN/Encoder 全有限、
  GPU2/3 保持 1 MiB，五门槛通过。继续收 e16/e20/e24+，每个 checkpoint 必须同点核验严格
  `cls>54.437`、`det>62.393`、`sum>118.330`。
- 178 原 PGID `3555710` 在 e12 产物完整后精确停止并释放动态 GPU0；随即对 `0804_04`
  body-frame product tangent 完成真实单卡 smoke，loss
  `21.3731/20.6355/20.9715/21.1433`、grad
  `60.1883/78.9025/84.5692/78.5934`，checkpoint 642 tensor 有限。fresh formal screen
  `3652381.pm_0804_04_formal_178`、PGID `3652382`，iter50 `0.9726 s/iter`、loss/grad
  `21.0216/101.0078`，五门槛通过并运行到 e4/e8/e12；GPU1 外部任务不动。
- 99 `0804_02` angle-only e12 `45.595/53.257`，DetA/AssA 为
  `37.536/57.720` 与 `47.802/61.450`，相对原 decoder `-1.800/-1.179`；pair mAP/AP50
  `0.2336/0.4131`、both-independent `0.2754/0.4665`。e4/e8/e12 三个完整节点均无净优势，
  所以在成熟窗口后停止 PGID `1673454` 并释放 99，而不是因 e4/e8 早停。
- 197 `0804_03` log-size-only e8 `41.299/46.767`，DetA/AssA 为
  `34.582/52.056` 与 `42.695/53.068`，相对原 decoder `-0.673/-1.411`；pair mAP/AP50
  `0.2043/0.3736`、both-independent `0.2487/0.4357`。这是中期负信号，PGID `2540932`
  继续 e12，不以 e8 直接否决；动态 GPU2/3 使用不意味着固定序号，GPU5 外部任务不动。

## 2026-08-05 01:46 CST：0804_01 的 252 成熟接力端口通过烟测

- 为避免 e12 到齐后再临时修改活跃仓库，在 252 隔离 checkout clean HEAD `f356593` 静态准备
  `0804_01 product-tangent` 的 2x4 接力配置；它直接导入 178 权威 1x8 模型，仅改固定 GPU0/1、
  per-rank batch 4、worker 与 252 数据/GMC/TrackEval 物理路径。配置 deepcopy、launcher
  `bash -n` 与双模型完整构建通过：科学 `model` 完全相等，参数 `22,771,111`、711 states，
  全局 batch 仍为 8。
- 252 GPU0/1 两轮均为 `1 MiB/0%` 后完成真实双卡四步 smoke：loss
  `12.9389/19.5019/19.6050/21.1636`、grad
  `103.0192/97.8254/91.1546/101.3375`，total、DN、Encoder proposal 全有限；
  `iter_4.pth` 的 iterative-cls/DN 语义与 642 个浮点 tensor 检查通过，无 fatal。
- 当前仍严格登记 `PREPARED/WAIT_E12`，未创建 formal workdir、未启动 252 长训。只有 178 e12
  保持明确同点优势时才从该 checkpoint 接力到固定 GPU0/1，并释放 178 验证 `0804_04`；否则
  不在最慢资源上延长弱路线。

## 2026-08-05 01:36 CST：运行健康与 252 历史点名任务复核

- 178 `0804_01` 在动态 GPU0、PGID `3555710` 健康进入 epoch10 iter50；99 `0804_02`
  在动态 GPU0/1、PGID `1673454` 到 epoch9 iter700；197 `0804_03` 在动态 GPU2/3、
  PGID `2540932` 到 epoch6 iter750。三线正式日志的 total/DN/Encoder/grad 均有限且无 fatal，
  继续分别等待 e12、e12 和 e8/e12；外部 GPU1（178）、GPU2（99）、GPU5（197）未被改动。
- 252 四卡均为 `1 MiB/0%`，没有本任务训练或 screen。点名历史 `0803_01 fresh` 仍无进程，
  `last_checkpoint=epoch_12.pth`，e4/e8/e12 checkpoint 与三组 TrackEval 保留；`0801_09`
  e56 resume 仍无进程，`last_checkpoint=epoch_64.pth`，e60/e64 checkpoint 与两组 TrackEval
  保留。252 继续只允许固定 GPU0/1，并保留给成熟路线或严格复验。

## 2026-08-05 01:31 CST：178/99 epoch 8 完整评估；body-frame 后继静态就绪

- `0804_01` product-tangent e8 cls HOTA/DetA/AssA `46.673/39.891/56.595`，det
  `53.922/47.262/64.136`；相对原 decoder e8 `41.972/48.178` 为
  `+4.701/+5.744`，相对 Encoder `45.269/50.193` 为 `+1.404/+3.729`。它也超过
  full-tangent `0803_23` e8 `46.283/53.755` 为 `+0.390/+0.167`；相对 full-tangent 的
  cls DetA/AssA 为 `+0.093/+0.486`、det 为 `-0.141/+0.519`，说明分块切空间已基本修复 e4 的
  定位损失并进一步提高关联质量。
- product-tangent pair mAP/AP50 `0.252434/0.444294`、both-independent
  `0.303859/0.513938`；375,558,772-byte checkpoint、5416/50、28 CSV、108 文件和 async
  标志完整。其 AP50 比 full-tangent 低 `0.009917`、det DetA 低 `0.141`，但 HOTA 双正且训练
  已在 e9；动态 GPU0 的 PGID `3555710` 继续 e12，GPU1 外部任务不动。e12 若保持强同点优势，
  该线升级为长轨迹候选，而不是在 e8 直接宣告最终达标。
- `0804_02` angle-only e8 cls/det HOTA `41.415/48.105`，DetA/AssA 为
  `34.045/53.460` 与 `43.261/55.420`；相对原 decoder `-0.557/-0.073`，相对 Encoder
  `-3.854/-2.088`，相对 terminal mean geometry `-3.587/-0.978`。pair mAP/AP50
  `0.203140/0.366031`、both-independent `0.248462/0.429392`；375,528,630-byte checkpoint、
  5416/50/28/108 与 async 标志完整。动态 GPU0/1 的 PGID `1673454` 已在 e9并继续 e12，
  GPU2 外部任务不动；不以 e8 直接否决。
- 根据 product-tangent e4 的“AssA 高、DetA 低”和 e8 剩余 det DetA/AP50 小缺口，准备
  `0804_04 body-frame product tangent`：只把中心切空间从图像轴对齐坐标换成双帧中间朝向与
  几何均值尺度定义的物体局部坐标，shape tangent、分类、DN、loss、层数与 attention 不变。
  它零参数、交换等变、class-agnostic、无 reweight，仅增加终层逐元素三角函数。178 独立
  checkout clean HEAD `e7ef507`；模块来源核验、2 项定向测试、正式/烟测配置 deepcopy、两份
  launcher `bash -n` 和完整父/候选构建通过：`22,771,111` 参数、增量 0、711 states。
  当前严格登记 `PREPARED/NO_GPU`，未做 smoke、未创建正式 workdir，不抢占 0804_01。

## 2026-08-05 01:14 CST：197 log-size-only epoch 4 完整评估

- `0804_03` e4 cls HOTA/DetA/AssA `31.938/26.971/40.785`，det
  `38.765/34.129/45.148`；相对原 decoder e4 `34.306/38.590` 为
  `-2.368/+0.175`，相对 terminal mean geometry `32.849/37.319` 为
  `-0.911/+1.446`，相对 Encoder `36.209/38.753` 为 `-4.271/+0.012`。与 99 的
  angle-only e4 `33.265/38.716` 相比为 `-1.327/+0.049`，说明两个 shape 单因素都只给出
  极弱 det 正信号，而尺度共识的分类收敛更慢；该点只作正交归因，不作停止理由。
- pair mAP/AP50 `0.143726/0.272730`、both-independent `0.187306/0.343500`；
  369,969,511-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件和
  `async_done=1` 完整。TrackEval 于 01:12 闭环，动态 GPU2/3 的 PGID `2540932` 已在 e5，
  total/DN/Encoder/grad 有限并继续 e8/e12；GPU5 外部任务未动。

## 2026-08-05 00:49 CST：252 transport-plane epoch 12 成熟收口

- `0803_30` e12 cls HOTA/DetA/AssA `45.089/37.490/56.626`，det
  `51.741/47.123/58.773`；相对原 decoder e12 `47.395/54.436` 为
  `-2.306/-2.695`，相对 terminal mean geometry `0803_13` e12 `48.289/54.539` 为
  `-3.200/-2.798`，相对 Encoder e12 `49.680/56.541` 为 `-4.591/-4.800`。几何平面没有
  把 e8 的双负转成优势。
- pair mAP/AP50 `0.228720/0.414872`、both-independent `0.272100/0.470852`；
  381,043,830-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件和
  `async_done=1` 完整。e4/e8/e12 三个完整节点均被原 decoder、terminal mean 与 Encoder
  双侧支配后，精确 TERM PGID `798989`，成员 `23→0`；固定 GPU0/1 回落至 1 MiB，GPU2/3
  全程未用于本任务。该决策来自成熟三节点证据，不是 e4/e8 早停；252 作为最慢资源暂留空，
  只接成熟路线或严格复验。

## 2026-08-05 00:19 CST：99 periodic-angle-only epoch 4 完整评估

- `0804_02` e4 cls HOTA/DetA/AssA `33.265/27.315/42.945`，det
  `38.716/34.172/45.208`；相对原 decoder e4 `34.306/38.590` 为
  `-1.041/+0.126`，相对 terminal mean geometry `0803_13` e4 `32.849/37.319` 为
  `+0.416/+1.397`，相对 Encoder e4 `36.209/38.753` 为 `-2.944/-0.037`。角度单因素目前只
  给出轻微 det 增益，分类仍慢；与 178 product-tangent e4 `35.274/43.849` 相比低
  `2.009/5.133`，但该点只作尺度/朝向归因，不作为停止理由。
- pair mAP/AP50 `0.143965/0.274645`、both-independent `0.187432/0.346096`；
  369,965,814-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件和
  `async_done=1` 完整。TrackEval 于 00:17 闭环，动态 GPU0/1 的 PGID `1673454` 已到 e5，
  total/DN/Encoder/grad 有限并继续 e8/e12；GPU2 外部任务不动，不以 e4 直接否决 decoder。

## 2026-08-05 00:07 CST：178 product-tangent epoch 4 完整评估

- `0804_01` e4 cls HOTA/DetA/AssA `35.274/28.529/45.125`，det
  `43.849/34.333/58.100`；相对原 decoder e4 `34.306/38.590` 为
  `+0.968/+5.259`，相对 terminal mean geometry `0803_13` e4 `32.849/37.319` 为
  `+2.425/+6.530`。相对 full-tangent `0803_23` e4 `36.342/44.739` 则为
  `-1.068/-0.890`；其中 det DetA 低 `3.776`、AssA 高 `3.388`，说明 center/shape 分块在早期
  强化关联稳定性但削弱检测定位，尚需 e8/e12 判断这种交换是否随收敛回补。
- pair mAP/AP50 `0.159255/0.296499`、both-independent `0.206860/0.370432`；
  369,973,108-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件和
  `async_done=1` 完整。TrackEval 于 00:05 闭环，动态 GPU0 的 PGID `3555710` 已到 e5，
  total/DN/Encoder/grad 有限并继续 e8/e12；GPU1 外部任务不动，不以 e4 直接否决 decoder。

## 2026-08-04 23:29 CST：197 成熟收口并启动 log-size-only；252 收齐 e8

- `0803_28` e12 cls HOTA/DetA/AssA `43.953/36.592/55.255`，det
  `50.679/46.150/57.353`；相对原 decoder e12 `47.395/54.436` 为
  `-3.442/-3.757`，相对 position/product `0803_27` e12 `46.017/52.755` 为
  `-2.064/-2.076`。pair mAP/AP50 `0.222433/0.403522`、both-independent
  `0.267528/0.461563`；381,032,167-byte checkpoint、5416/50、28 CSV、108 个非空文件和
  `async_done=1` 完整。e4/e8/e12 三个完整节点均双负后精确 TERM PGID `1016336`，成员
  `23→0`；这是成熟判定而非 e4/e8 早停，GPU5 外部任务未动。
- 新 `0804_03` 只在最终 normal queries 共享宽高的 log-domain 乘法增量；中心、周期角、分类、
  DN、辅助输出与递归 reference 保持父线。它与 99 的 angle-only 构成尺度/朝向正交对照，结构
  零参数/状态、交换等变、class-agnostic，无 reweight、新 layer、attention 或 loss。隔离仓库
  clean HEAD `c73e19a`；定向单测、正式/烟测配置深拷贝、launcher 语法与整模构建通过：
  `22,771,111` 参数、增量 0、711 states。
- 动态 GPU2/3 两轮均为 `1 MiB/0%`；真实 DDP smoke loss
  `12.9355/19.3070/19.4537/21.0277`、grad
  `106.2667/108.1143/96.9112/103.3604`，DN/Encoder 与 642 个 checkpoint 浮点 tensor
  全有限。fresh formal screen `2540930.pm_0804_03_formal_197`、PGID `2540932`；iter50
  `1.0330 s/iter`、loss/grad `21.4643/103.1342`，7 个成员、GPU2/3 各约 19.2 GiB，五门槛
  全部通过，登记 `RUNNING/TO_E4+`。
- 252 `0803_30` e8 cls HOTA/DetA/AssA `40.934/33.759/52.305`，det
  `47.531/42.486/55.014`；相对原 decoder e8 `41.972/48.178` 为 `-1.038/-0.647`，相对
  terminal mean geometry `0803_13` e8 `45.002/49.083` 为 `-4.068/-1.552`。pair mAP/AP50
  `0.195488/0.359509`、both-independent `0.238468/0.419121`；375,538,934-byte checkpoint、
  5416/50、28/108 和异步完成标志完整。固定 GPU0/1 的 PGID `798989` 已到 e9，继续 e12，
  不因 e8 直接否决；GPU2/3 保持未使用。

## 2026-08-04 23:05 CST：position-plane e12 成熟停止，periodic-angle-only 接替 99

- `0803_29` e12 cls HOTA/DetA/AssA `45.384/37.106/57.832`，det
  `51.334/46.020/59.456`；相对原 decoder e12 `47.395/54.436` 为 `-2.011/-3.102`，
  相对 position/product `0803_27` e12 `46.017/52.755` 仍低 `0.633/1.421`。
- pair mAP/AP50 `0.221299/0.400565`、both-independent `0.265863/0.459146`；
  381,025,910-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件和
  `async_done=1` 完整。e4/e8/e12 三个完整节点均无优势，精确 TERM PGID `1582836` 后成员
  `23→0`；这是成熟窗口收口，不是 e4/e8 早停。
- 新 `0804_02` 只在最终 normal queries 的 π 周期角商空间取双帧中点，中心、尺度、分类、DN、
  辅助输出与递归 reference 保持逐帧自由；用于检验 terminal shape consensus 的成熟收益是否来自
  orientation，同时避免尺度过约束。结构零参数/状态、交换等变、class-agnostic，无 reweight、
  新 layer、attention 或 loss。
- 首次隔离 clone 复现非必要 83MB LFS GIF 缺失，失败目录保留；`GIT_LFS_SKIP_SMUDGE=1` 在
  `_retry1` 重建 clean HEAD `2e3fe7e`。定向 unittest、模块来源、配置/smoke deepcopy、launcher
  语法与整模构建通过：`22,771,111` 参数、增量 0、711 states。
- 动态 GPU0/1 连续两次为 `10 MiB/0%`；真实 DDP smoke loss
  `12.9371/19.5296/19.6306/21.2079`、grad `102.8095/107.6873/92.3313/90.3473`，
  DN/Encoder 与 642 个 checkpoint tensor 全有限。fresh formal screen
  `1673453.pm_0804_02_formal_99`、PGID `1673454`，iter50 `0.9654 s/iter`、loss/grad
  `21.3937/115.9512`，五门槛通过，状态 `RUNNING/TO_E4+`。

## 2026-08-04 22:44 CST：full-tangent e52 成熟收口，纯 product-tangent 接替 178

- `0803_23` e52 cls HOTA/DetA/AssA `54.197/44.752/67.727`，det
  `60.991/53.567/71.845`；较 e48 仅增 `+0.253/+0.103`，相对原 decoder e52
  `54.695/62.388` 为 `-0.498/-1.397`，相对最终门槛仍低 `0.240/1.402`。
- pair mAP/AP50 `0.308716/0.526239`、both-independent `0.349509/0.565064`；
  436,330,612-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件和
  `async_done=1` 完整。结合 e44→e48→e52 的成熟 det 增量 `+0.335→+0.103`，精确 TERM
  PGID `3151184` 后成员 `9→0`；这不是 e4/e8 早停。
- 新 `0804_01` 只把 terminal 5D detail 的单内积拆成 center 2D 与 shape 3D 两个独立切丛，
  防止平移能量跨维混入 log-size/angle；分类、DN、辅助输出、递归 reference 不变，零参数/状态、
  交换等变、class-agnostic，无 reweight、新 layer、attention 或 loss。
- 178 隔离 checkout clean HEAD `e2f5e7d`；定向 unittest、配置/smoke deepcopy、launcher
  语法和整模构建通过：`22,771,111` 参数、增量 0、711 states。GPU0 两次空闲检查均为
  `1 MiB/0%`；真实 smoke 四组 loss `21.3692/20.6455/20.9246/21.2007`、grad
  `60.2135/67.4355/115.0544/105.6304`，DN/Encoder 与 642 个 checkpoint tensor 全有限。
- fresh formal screen `3555709.pm_0804_01_formal_178`、PGID `3555710`；iter50
  `0.9700 s/iter`、loss `20.9975`、grad `132.7340`，进程、GPU0、正式日志、iter50 与有限性
  五门槛全部通过，状态 `RUNNING/TO_E4+`。GPU1 外部任务保持不动。

## 2026-08-04 21:58 CST：0803_30 epoch 4 完整评估

- e4 cls HOTA/DetA/AssA `31.119/24.852/41.221`，det `37.046/30.516/45.751`；相对原 decoder
  e4 `34.306/38.590` 为 `-3.187/-1.544`，相对 terminal mean geometry `0803_13` e4
  `32.849/37.319` 为 `-1.730/-0.273`。这是 terminal transport-plane 的早期慢收敛信号，不能由
  e4 直接否决。
- pair mAP/AP50 `0.132184/0.250904`、both-independent `0.172747/0.315071`；369,970,998-byte
  checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件和 `async_done=1` 完整。
- TrackEval 于 21:50:54 启动、21:56:31 闭环；训练同时已恢复到 e5，PGID `798989` 存活，
  total/DN/Encoder/grad 有限，GPU0/1 各约 19.2 GiB、GPU2/3 仍为 1 MiB。继续 e8/e12，
  不因 e4 双负停止最慢资源上的已验证成熟候选。

## 2026-08-04 21:44 CST：0803_29 epoch 8 完整评估

- e8 cls HOTA/DetA/AssA `40.865/34.345/50.731`，det `46.308/41.571/53.385`；较自身 e4 HOTA
  `30.658/38.402` 明显回升 `+10.207/+7.906`，说明 position-tangent 的分类慢收敛正在恢复，不能由
  e4/e8 直接否决。
- 相对原 decoder e8 `41.972/48.178` 仍低 `1.107/1.870`，相对 position/product `0803_27` e8
  `41.889/48.165` 仍低 `1.024/1.857`；pair mAP/AP50 `0.197487/0.363752`、both-independent
  `0.244215/0.429793`，尚无 AP 与轨迹指标互相背离的收益证据。
- 375,531,702-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件和
  `async_done=1` 完整；21:41 正式训练已到 e9 iter300，GPU0/1 动态占用、PGID `1582836`
  存活且 total/DN/Encoder/grad 均有限。继续 e12 形成成熟窗口，不提前切换到排队候选。

## 2026-08-03 23:22 CST：0803_14 迁移 99 GPU1/2 并正式运行

- 新增 99 端正式/四步 smoke 配置与安全启动器，训练/验证数据、GMC、TrackEval 和 workdir 均改为 99 路径；两配置加载和深拷贝通过。隔离仓库固定提交 `ed7823d`，零参数构建仍为 `22,771,111` 参数、增量 0、711 个状态张量。
- 真数据 DDP smoke 四次 loss `12.9438/19.3917/19.5249/21.1545`，grad norm `106.5528/91.3914/80.1813/83.8108`；DN 与 encoder loss 全有限，`iter_4.pth` 中 642 个浮点张量全有限，错误扫描为空。
- formal fresh 于 23:20 启动，PGID `1327092`；23:21 到 epoch1 iter50：`0.9957 s/iter`、loss `21.4082`、grad norm `149.8043`，GPU1/2 各约 19.2 GiB，进程、显存、正式日志、iter50、有限数值五门槛全部通过，状态为 `RUNNING/TO_E12`。

## 2026-08-03 23:28 CST：0803_16 terminal normalized-center 已设计

- 只在最后一层输出执行中心共识：先把两帧各自的候选中心位移除以对应 reference 的宽高，得到 reference-local 位移，再共享该局部中心增量并映射回各自 reference。所有递归 reference、宽高、角度、分类和 DN 保持原路径。
- 该投影零参数、swap-equivariant、class-agnostic，无 reweight、额外 loss、attention 或 decoder 层；与 `0803_15 terminal angle-only` 构成“中心 vs 角度”的正交终层对照。
- 新增 178 1xb8 formal/smoke 配置、零状态增量构建检查和只调用一次终层投影的定向测试。隔离仓库 `/data1/users/litianhao01/PairMOT_terminalcenter_0803_16` 固定 `c05cd21`；远端定向测试通过，完整构建为 `22,771,111` 参数、增量 0、711 个状态张量。状态 `PREPARED/WAITING_AFTER_0803_15`，不占用正在运行的 GPU。

## 2026-08-03 23:40 CST：0803_15/16 接替启动链已补齐

- 两项均新增 178 单卡 formal/smoke 安全启动器：Conda `nounset` 防护、目标 GPU 环境变量、fresh workdir、真实 GMC/预训练检查、错误扫描、iterative-cls/DN 审计和全浮点 checkpoint 有限性审计。
- 四个启动器 Bash 语法通过；仅完成准备，不创建 workdir、不占 GPU。178 释放后按 `0803_15` 再 `0803_16` 的顺序执行真实 smoke→formal 五门槛。
- 两个隔离仓库均已安全快进到启动提交 `e9f56dc` 并保持 clean；未触碰正在运行的 `0803_13` 仓库。后续 formal provenance 以 `e9f56dc` 为准。

## 2026-08-03 23:52 CST：0803_17 terminal semantic-margin consensus 已准备

- 新候选仅在迭代分类的最后一层作用于 normal queries：分别保留两帧分类残差的 class mean，
  只把去均值后的 class-margin 方向替换为双帧平均。前序 decoder 层、每帧 objectness 均值、
  DN absolute 分类与全部框回归保持原路径。
- 该运算零参数、class-permutation-equivariant，不使用类别身份、class-aware 逻辑、reweight、
  新 attention 或 decoder 层。定向测试确认两帧均值精确保持、终层 centered margins 一致、
  DN 隔离以及跨帧梯度只发生在终层。
- 178 隔离仓库 `/data1/users/litianhao01/PairMOT_terminalmargin_0803_17` 固定提交 `e245127`；
  远端目标源码核验、3 项定向测试、配置/整模构建均通过：`22,771,111` 参数、增量 0、711 个
  状态张量。formal/smoke 启动器 Bash 语法通过，状态 `PREPARED/WAITING_AFTER_0803_16`，
  未创建 workdir、未占用 GPU。
- 23:50 实测四条 formal 数值均有限：99 `0803_14` 到 e2 iter850，252 `0803_12` 恢复线到
  e7 iter250，178 `0803_13` 到 e7 iter950，197 `0803_11` 到 e8 iter100；继续等待完整
  checkpoint、检测和 TrackEval 后再做资源切换。

## 2026-08-04 00:27 CST：0803_13 epoch 8 完整评估

- e8 cls HOTA/DetA/AssA `45.002/39.137/53.997`，det `49.083/46.725/53.354`；相对原始
  `0801_09` decoder 同点 HOTA `41.972/48.178` 提高 `+3.030/+0.905`，合计 `+3.935`。
  相对 Encoder 同点 `45.269/50.193` 仍低 `0.267/1.110`，尚未达到最终门槛。
- pair mAP/AP50 `0.233479/0.424803`、both-independent `0.285866/0.494486`；相对原始
  decoder 同点四项提高约 `+0.023959/+0.037624/+0.035905/+0.056677`，HOTA、DetA/AssA
  与 AP 不存在仅靠指标搬运的迹象。
- 375,568,500-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件和
  `async_done=1` 完整。terminal-only 尺度/周期角在 e8 对强父线形成双正交增益，保持 PGID
  `3062903` 继续 e12，并观察该优势是否避免全层 `0803_09` 在 e16 后的衰减；不释放 178。

## 2026-08-04 00:42 CST：0803_11/12 epoch 8 完整评估

- `0803_11 late geometry` e8 cls HOTA/DetA/AssA `40.377/32.826/52.463`，det
  `45.730/40.871/52.928`；相对原始 decoder 同点 HOTA `-1.595/-2.448`。pair mAP/AP50
  `0.185399/0.329767`、both-independent `0.230288/0.393352`，四项同样低于父线。
- `0803_12 progressive geometry` e8 cls `40.430/33.681/51.565`，det
  `46.542/42.862/52.411`；相对原始 decoder 同点 HOTA `-1.542/-1.636`。pair mAP/AP50
  `0.192018/0.355406`、both-independent `0.237795/0.421425`，四项也均低于父线。
- 两项各自的 epoch8 checkpoint、5416 条检测、50 序列、28 CSV、108 个非空文件和
  `async_done=1` 完整；它们均被 terminal-only `0803_13` 同点双侧明显支配，但遵守不能用 e8
  直接否决 decoder 的约束，197 与 252 固定 GPU0/1 继续 e12。结构派生优先级停止扩展
  “最后两层/渐进几何”，新候选集中到 terminal-only 与语义方向。

## 2026-08-04 00:45 CST：0803_14 epoch 4 完整评估

- e4 cls HOTA/DetA/AssA `30.813/24.668/40.889`，det `36.985/31.609/44.441`；相对
  Encoder 同点 HOTA `-5.396/-1.768`，相对 terminal full-size `0803_13` 同点
  `-2.036/-0.334`。
- pair mAP/AP50 `0.129617/0.246465`、both-independent `0.170361/0.313179`，也均低于
  `0803_13` e4；369,970,934-byte checkpoint、50 序列、28 CSV、108 非空文件和
  `async_done=1` 完整。
- 该点只登记为 terminal-area 早期收敛更慢，不作直接否决；99 GPU1/2 保持 PGID
  `1327092` 到 e8/e12，GPU0 外部任务不受影响。

## 2026-08-04 00:55 CST：语义后继迁移到 99/197 快速通道

- `0803_17 terminal semantic margins` 新增 99 2xb4 formal/smoke 配置与安全启动器；隔离仓库
  `/data/users/wangying01/lth/PairMOT_terminalmargin_0803_17_99` 固定提交 `ac02fc2`。
  目标配置/四步配置整模比较通过：`22,771,111` 参数、增量 0、711 状态张量；两个启动器
  Bash 语法通过。未创建 workdir、未占 GPU，排在 99 `0803_14` e12 成熟释放之后。
- 新增 `0803_18 terminal geometry + semantic margins`：组合 `0803_13` 已在 e8 验证为正的
  终层 log-size/周期角，与 `0803_17` 的终层 centered-margin 共识；中心、每帧 residual class
  mean、递归 reference 与 DN 保持独立。仍是零参数、类别置换等变、无 class-aware/reweight、
  新 attention 或层。
- 197 隔离仓库 `/data/users/litianhao/PairMOT_terminalgeommargin_0803_18_197` 固定 `ac02fc2`；
  目标/smoke 配置、组合相对原始 decoder 的零状态增量整模比较及两个启动器语法均通过：
  `22,771,111` 参数、增量 0、711 状态张量。排在 `0803_11` e12 成熟释放之后，当前不占卡。
- 252 继续只用固定 GPU0/1 跑成熟确认 `0803_12`，不部署上述新结构筛选。

## 2026-08-04 01:45 CST：0803_13 epoch 12 完整评估

- e12 cls HOTA/DetA/AssA `48.289/41.435/58.165`，det `54.539/49.791/61.810`；相对原始
  `0801_09` decoder 同点 `47.395/54.436` 提高 `+0.894/+0.103`，合计 `+0.997`。相对
  Encoder 同点 `49.680/56.541` 仍低 `1.391/2.002`，尚未达到最终门槛。
- pair mAP/AP50 `0.261539/0.464763`、both-independent `0.309710/0.522863`；
  381,093,940-byte checkpoint、50 序列、28 CSV、108 个非空评测文件与 `async_done=1` 完整。
- 与原始 decoder 的双正优势从 e8 合计 `+3.935` 收窄到 e12 `+0.997`，说明 terminal-only
  投影减轻了早期过约束但尚未证明后期不反转。遵守慢收敛约束，178 保持 PGID `3062903`
  继续 e16；只有完成 e16 全量评估后再决定是否释放给 `0803_15/16`。

## 2026-08-04 01:55 CST：0803_19 terminal full-tangent geometry 已准备

- 既有 symmetric-feature/shared-attention 轨迹在 e8 系统性损伤 DetA/AP，因此不再共享 decoder
  hidden 或 attention。新候选改为只在最终 normal-query box 输出做自然坐标投影：中心使用
  reference-local 位移，尺寸使用 log-ratio，角度使用 π 周期切空间；三者均只平均一次。
- 早中层递归 reference、分类 residual、DN、loss 和 attention 路径不变；结构零参数、
  class-agnostic、无 reweight 或额外计算层。它是 `0803_13` terminal size/angle 的中心扩展，
  也为 `0803_16` center-only 提供组合对照。
- 178 隔离仓库 `/data1/users/litianhao01/PairMOT_terminalfulltangent_0803_19` 固定 `dc0e958`；
  定向测试 `1 passed`，两个 launcher Bash 语法通过，完整父/新模型均为 22,771,111 参数、
  711 状态张量、增量 0。状态 `PREPARED`，未建 smoke/formal workdir、未占 GPU。

## 2026-08-04 01:57 CST：0803_14 epoch 8 完整评估

- e8 cls HOTA/DetA/AssA `41.384/34.365/52.257`，det `47.315/42.578/54.312`；相对原始
  `0801_09` decoder 同点 `41.972/48.178` 为 `-0.588/-0.863`，相对 terminal full-size
  `0803_13` 同点低 `3.618/1.768`。相对 Encoder 同点仍低 `3.885/2.878`。
- pair mAP/AP50 `0.206968/0.367917`、both-independent `0.251401/0.427578`，四项均低于
  0803_13；375,534,774-byte checkpoint、50 序列、28 CSV、108 非空文件和
  `async_done=1` 完整。
- 保留逐帧纵横比、只共享面积没有形成终层几何的正增益，但不以 e8 直接否决；99 GPU1/2
  保持 PGID `1327092` 到 e12，成熟判定后优先切换 `0803_17 semantic margins`。

## 2026-08-04 02:05 CST：0803_12 epoch 12 完整评估并停止

- e12 cls HOTA/DetA/AssA `45.677/36.931/59.103`，det `52.131/46.532/60.474`；相对原始
  `0801_09` decoder 同点 `47.395/54.436` 为 `-1.718/-2.305`，相对 Encoder 同点
  `49.680/56.541` 为 `-4.003/-4.410`。
- pair mAP/AP50 `0.227110/0.412989`、both-independent `0.269190/0.466195`；
  381,003,318-byte checkpoint、50 序列、28 CSV、108 非空文件与 `async_done=1` 完整。
- e4/e8/e12 三个完整节点均双负，属于成熟轨迹停止而非 e4/e8 早停。精确 TERM PGID
  `123974` 后成员 `23→0`，252 四卡均为 `1 MiB/0%`；最慢的 252 保持空闲，仅用于后续
  快速通道已证明候选的成熟确认。

## 2026-08-04 02:10 CST：0803_20 terminal full-tangent + semantic margins 已准备

- 新组合在最终 normal queries 同时使用 0803_19 的自然坐标几何一致化，以及 0803_17 的
  centered class-margin 一致化；保留各帧 residual class mean、早中层递归 reference 与 DN
  absolute 语义。零参数、类别置换等变、无 class-aware/reweight 或额外主计算层。
- 197 隔离仓库 `/data/users/litianhao/PairMOT_terminalfulltangentmargin_0803_20_197` 固定
  `f179249`；两个 launcher Bash 语法和完整父/新模型比较通过，均为 22,771,111 参数、
  711 状态张量、增量 0。状态 `PREPARED`，排在 0803_18 之后且未占 GPU。

## 2026-08-04 02:50 CST：0803_11 epoch 12 成熟停止

- e12 cls HOTA/DetA/AssA `45.409/37.139/57.863`，det `50.665/45.754/57.944`；相对原始
  decoder 同点 `47.395/54.436` 为 `-1.986/-3.771`，相对 Encoder 同点 `49.680/56.541`
  为 `-4.271/-5.876`。e4/e8/e12 三个完整节点持续双负。
- pair mAP/AP50 `0.225726/0.405185`、both-independent `0.270304/0.461162`；
  381,030,375-byte checkpoint、50 序列、28 CSV、108 个非空文件与 `async_done=1` 完整。
- 依据成熟三节点证据精确 TERM PGID `53708`，成员 `23→0`；GPU4/5 回到 `1 MiB`，断点保留。
  该停止不是 e4/e8 早停，释放的 197 两卡转给已准备的 0803_18。

## 2026-08-04 02:58 CST：0803_18 smoke 通过并正式运行

- 0803_18 在最终 normal queries 组合终层 log-size/π 周期角几何共识与 centered semantic-margin
  共识；保留中心、每帧 class mean、递归 reference 和 DN absolute 路径。结构零参数、类别置换
  等变、无 class-aware/reweight、新 attention 或层。
- 首次空闲检查包装器因把 GPU CSV `1,0` 作为单值解析而在创建 workdir 前退出，未占 GPU；修正
  为逐卡查询后连续两次确认 GPU4/5 为 `1 MiB/0%`。真实四步 DDP smoke loss
  `12.9355/19.5166/19.6231/21.1658`、grad `106.2062/125.5042/105.4083/99.3739`；checkpoint
  642 个浮点张量、iterative-cls residual 和 DN absolute 语义均有限，错误扫描为空。
- 隔离仓库 `/data/users/litianhao/PairMOT_terminalgeommargin_0803_18_197` 固定 clean HEAD
  `ac02fc2`。fresh formal screen `387856.pm_0803_18_formal_197`、PGID `387859`；iter50
  `1.7440 s/iter`、loss `21.3900`、grad `107.1625`，DN/encoder loss 全有限，7 个进程，
  GPU4/5 各约 19.2 GiB，错误扫描、资源、进程组、provenance 与 workdir 五门槛通过。
  状态 `RUNNING`，继续完整 e4/e8/e12 及更晚节点，不以 e4/e8 直接否决。

## 2026-08-04 03:00 CST：0803_13 epoch 16 保持双正

- e16 cls/det HOTA `50.415/57.456`，相对原始 decoder 同点 `50.036/56.933` 仍提高
  `+0.379/+0.523`，合计 `+0.902`；相对 Encoder 同点 `51.091/58.320` 尚低
  `0.676/0.864`，未达到最终门槛。
- pair mAP/AP50 `0.275158/0.486134`、both-independent `0.320648/0.537430`；
  386,615,796-byte checkpoint、50 序列、28 CSV、108 个非空文件与异步完成证据齐全。
- 虽然对原始 decoder 的联合优势较 e8/e12 收窄，但 e16 仍为双正且绝对值继续上升；保持
  178 单卡 PGID `3062903` 到 e20，按成熟后期轨迹复核，不为后继候选提前终止。

## 2026-08-04 03:05 CST：0803_14 epoch 12 成熟停止

- e12 cls HOTA/DetA/AssA `46.987/38.382/59.800`，det `52.992/47.227/61.487`；相对原始
  decoder 同点 `47.395/54.436` 为 `-0.408/-1.444`，相对 Encoder 同点为 `-2.693/-3.549`。
- pair mAP/AP50 `0.241566/0.426861`、both-independent `0.286042/0.482380`；
  381,037,430-byte checkpoint、50 序列、28 CSV、108 个非空文件与 `async_done=1` 完整。
- e4/e8/e12 三个完整节点均未形成双正，精确 TERM PGID `1327092`，成员 `23→0`；GPU1/2
  回到 `10 MiB/0%`，GPU0 外部任务保持原状。该成熟停止释放两卡给 0803_17。

## 2026-08-04 03:09 CST：0803_17 smoke 通过并正式运行

- 终层 semantic-margin 只平均两帧去 class mean 后的分类 margin，分别保留每帧 class mean；
  DN、框回归、递归 reference 和前序 decoder 层不变。结构零参数、类别置换等变、无
  class-aware/reweight、新 attention 或层。
- GPU1/2 连续两次为 `10 MiB/0%` 后启动四步真数据 smoke；loss
  `12.9371/19.4836/19.6450/21.2109`、grad `102.9283/109.1407/105.1020/100.4872`，
  364,502,518-byte checkpoint、642 个浮点张量及 iterative-cls/DN 语义均有限，错误扫描为空。
- 隔离仓库固定 clean HEAD `ac02fc2`。fresh formal screen `1357907.pm_0803_17_formal_99`、
  PGID `1357909`；iter50 `0.9994 s/iter`、loss `21.3978`、grad `110.7768`，DN/encoder loss
  全有限，7 个进程，GPU1/2 各约 19.2 GiB，五门槛通过。状态 `RUNNING`，已建立 e4 完整评估
  监控并继续 e8/e12 及后期节点，不以 e4/e8 直接否决。

## 2026-08-04 03:16 CST：预留 0803_21 terminal transported margins

- 直接终层 margin 平均会删除全部帧差，包括早中层已稳定积累的类别排序变化。0803_21 将终层
  centered-margin 残差分成 pair common/detail，只保留 detail 在“前序累计 logits 的双帧
  centered difference”方向上的投影；允许延续已建立的类别排序轨迹，但禁止终层新引入横向
  class switch。
- 前序 transport 方向显式 detach；每帧 residual class mean、pair residual mean、DN absolute
  分类及早中层梯度路径保持不变。运算零参数、类别置换等变、frame-swap 等变，无 class identity、
  class-aware、reweight、新 attention/layer/loss，仅增加少量点积。
- 已预留全局 ID `0803_21`，新增 99 2xb4 formal/smoke 配置、构建审计与安全启动器；本地语法和
  launcher `bash -n` 通过，不触碰运行中的 0803_17。
- 99 隔离 checkout `/data/users/wangying01/lth/PairMOT_terminaltransport_0803_21_99` 已固定 clean
  HEAD `a7b37ef`。首次定向测试命令发现既有 py310 环境未安装 pytest，未修改环境；改用标准
  unittest 加载同一测试文件后 3 项定向测试全部通过。正式/smoke 配置加载与深拷贝、远端两个
  launcher `bash -n`、父/新整模比较均通过：`22,771,111` 参数、增量 0、711 状态张量。
  状态为 `PREPARED/NO_GPU`；未创建 smoke/formal workdir，等待 0803_17 成熟资源决策。

## 2026-08-04 03:26 CST：预留 0803_22 geometry + transported margins

- 0803_18 用终层 log-size/周期角结合“完全平均”的 semantic margins；0803_22 保持同一几何
  投影，只把语义侧替换为 0803_21 的 transported margins。这样可直接判断后期若出现分类或检测
  回落，来源是终层语义过平滑还是几何共识本身，而不同时更换中心/尺寸机制。
- 保留每帧 residual class mean、pair residual mean、中心、递归 reference 与 DN；零参数、类别
  置换/帧交换等变，无 class-aware、reweight、新 attention/layer/loss。相对 0803_18 只增加少量
  centered reduction 与点积，计算量近零变化。
- 已预留全局 ID `0803_22`，新增 197 2xb4 formal/smoke 配置、零状态构建审计和安全启动器；
  本地 Python 语法与 launcher `bash -n` 通过。
- 197 隔离 checkout `/data/users/litianhao/PairMOT_terminalgeomtransport_0803_22_197` 固定 clean
  HEAD `41c08d8`；3 项 transported-margin 定向测试、正式/smoke 配置加载与深拷贝、远端 launcher
  语法以及父/新整模零状态比较通过：`22,771,111` 参数、增量 0、711 状态张量。状态
  `PREPARED/NO_GPU`；未建 smoke/formal workdir，不抢占运行中的 0803_18。

## 2026-08-04 03:42 CST：预留 0803_23 transported full-tangent geometry

- `0803_13` 的终层几何完全平均在 e8/e12/e16 对原始 decoder 保持双正但优势收窄；新候选不再
  删除所有帧间末层几何差，而是在 reference-local center、log-size、π 周期角组成的 5D 自然
  切空间中，只保留终层 pair detail 沿“前三层 reference 已积累相对变换”方向的投影。
- transport 方向显式 detach；pair-common 切空间增量、真实已建立的平移/尺度/旋转趋势和 DN
  原路径保留，正交末层抖动被抑制。结构零参数、class-agnostic、frame-swap equivariant，无
  class-aware、置信度 reweight、新 attention/layer/loss，仅增加常数规模解码、点积与重编码。
- 已预留 ID `0803_23` 和 178 1xb8 formal/smoke 配置、零状态构建审计、安全启动器；隔离仓库
  `/data1/users/litianhao01/PairMOT_terminaltransporttangent_0803_23` 固定 clean HEAD `d6af6d32`。
  两项定向测试（既有轨迹投影、帧交换/DN 保持）、配置整模构建和 launcher 语法全部通过：
  `22,771,111` 参数、增量 0、711 状态张量。状态 `PREPARED/NO_GPU`，不抢占 0803_13；待 e20
  成熟判定后，与完全平均的 0803_19 按机制优先级选择，而非同时占用 178。

## 2026-08-04 04:13 CST：0803_13 epoch 20 双正扩大

- e20 cls/det HOTA `51.791/58.526`；相对原始 `0801_09` decoder 同点 `50.843/58.033`
  提高 `+0.948/+0.493`，联合优势 `+1.441`，较 e16 的 `+0.902` 再次扩大。相对 Encoder
  同点 `51.514/58.922` 为 `+0.277/-0.396`，已超过分类侧、检测侧尚未同点双超。
- pair mAP/AP50 `0.288615/0.506941`、both-independent `0.333302/0.555375`，四项均较 e16
  `0.275158/0.486134/0.320648/0.537430` 继续提高，不是 HOTA 与检测 AP 的指标搬运。
- 392,138,804-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评估文件和
  `async_done=1` 完整。PGID `3062903` 的 9 个成员继续运行，04:12 已进入 e21；因 e8/e12/e16/e20
  四节点均相对强父线双正且 e20 优势回升，保留至 e24，不切换已准备候选，也不以中期未过最终
  `54.437/62.393` 门槛提前否决。

## 2026-08-04 04:25 CST：0803_17 epoch 4 完整评估

- terminal shared semantic margins e4 cls HOTA/DetA/AssA `32.203/26.308/42.066`，det
  `37.822/32.233/45.135`；相对原始 decoder e4 `34.306/38.590` 为 `-2.103/-0.768`，相对
  Encoder e4 `36.209/38.753` 为 `-4.006/-0.931`。语义平均未改善早期 cls，det 较 cls 更接近。
- pair mAP/AP50 `0.140801/0.262022`、both-independent `0.183663/0.330727`；369,970,486-byte
  checkpoint、5416 条检测、50 序列、28 CSV、108 非空文件与 `async_done=1` 完整。
- 初始 e4 监控按未补零的 `val_det/epoch_3` 等待，而实际双卡目录为 `epoch_03`；训练与评估均
  正常，已由权威实际目录完成核验并终止错误等待器，后续监控使用补零路径。该节点只登记早期
  负向信号，PGID `1357909` 已进入 e5，严格继续 e8/e12，不以 e4 直接否决。

## 2026-08-04 04:53 CST：0803_13 成熟轨迹的 252 迁移通道通过

- 为提高并行吞吐，预案是在 178 e24 仍相对原 decoder 双正时，将成熟 0803_13 从该断点迁到
  最慢但空闲的 252 固定 GPU0/1 继续晚期确认，并把更快的 178 单卡释放给 transported-tangent
  新结构。当前只完成端口与 smoke，不提前停止 178 或启动 252 formal。
- 252 2x4 配置与权威 178 1x8 配置的有效 `model`、optimizer、scheduler、train loop、hooks
  逐项严格相等；每 rank batch 4 × world 2 保持 global batch 8。首次比较发现 252 父配置多携带
  若干显式 `False` 开关，已在 GPU 启动前清理并重新证明 model dict 完全一致。
- 隔离仓库 `/data/users/litianhao01/PairMmot_terminal_0803_13_resume252` 固定 clean `bec1a1c`；
  整模比较为 22,771,111 参数、零增量、711 状态张量，两个 launcher 语法通过。固定 GPU0/1
  连续两次为 `1 MiB/0%` 后执行真数据 DDP smoke，loss
  `12.9372/19.5473/19.6326/21.1872`、grad
  `106.2705/100.9766/89.2521/88.4672`，DN/encoder 项和 642 个浮点 checkpoint 张量均有限；
  364,501,750-byte `iter_4.pth` 完整，错误扫描为空，退出后 GPU0/1 回到 `1 MiB/0%`。
- 状态 `PREPARED/NO_FORMAL_GPU`；只有 e24 完整 checkpoint、检测、TrackEval 继续支持双正后，
  才会先精确停止 178 PGID，再从 `epoch_24.pth` 单点恢复，禁止两机同时写同一 workdir。

## 2026-08-03 23:12 CST：0803_13 epoch 4 与四机资源核验

- e4 cls HOTA/DetA/AssA `32.849/26.682/43.921`，det `37.319/34.948/41.243`；相对 Encoder e4 HOTA `-3.360/-1.434`。pair mAP/AP50 `0.1399/0.2613`，both-independent `0.1881/0.3391`。
- 该节点只说明终层 log-size+周期角没有改善早期收敛，不作为否决；保持模型和正式目录不变，继续 e8/e12。
- 资源实测：252 仅 GPU0/1 为 `0803_12`、GPU2/3 空闲；178 当前用 GPU0 跑 `0803_13`；197 当前用 GPU4/5 跑 `0803_11`；99 GPU0 外部占用而 GPU1/2 空闲。只有 252 固定序号，其余机器仅限制卡数，因此 99 GPU1/2 可承接 `0803_14`。

`0803_06 iterative-cls frame-evidence decoder` 已在 e16 完整评估后按四节点成熟轨迹与原始
`0801_09` 同点支配关系精确停止；GPU0/1 接替为 `0803_10 shared log-area + periodic-angle`。
所有正式实验均在隔离 checkout 中固定提交，不热更新活动仓库。

`0803_08 common-preserving frame-detail + periodic-angle` 已从 178 单卡静态候选迁移为 252
GPU2/3 双卡 formal；`0803_09 log-size tangent + periodic-angle` 已在 `0803_04` e24 平台确认
并释放后接管 178 GPU0。两者均使用隔离 checkout，不热更新活动仓库。

## 已完成或释放

| 服务器 | 实验 | 状态 | 说明 |
| --- | --- | --- | --- |
| 252 GPU 0,1 | `0803_05 ... iterativecls pair-shared-normalized-center ... fresh` | `STOPPED`；epoch 12 全量评估后于 11:18 精确停止 | e12 cls HOTA/DetA/AssA `43.161/36.061/53.834`，det `49.396/44.754/56.154`；相对父线同点 HOTA `-4.234/-5.040`，相对 Encoder 同点 `-6.519/-7.145`。e4/e8/e12 完整轨迹持续负向，不属于早停。pair mAP/AP50 `0.2142/0.3842`，both-independent `0.2592/0.4419`；checkpoint、5416 条检测、50 序列、28 CSV 与 108 个非空文件完整保留。精确 PGID `3549855` 的 23 个成员全部退出，GPU0/1 释放。 |
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

## 2026-08-02 11:35 CST：0801_13 epoch-12 仍受覆盖限制

- `0801_13 terminal pair-differential objectness residual` e12 的 cls/det HOTA 为
  `47.428/56.381`，相对 Encoder 同点为 `-2.252/-0.160`。cls DetA/AssA 为
  `-3.552/+0.608`；det DetA/AssA 为 `-1.879/+2.394`：关联仍保持正增益，但覆盖下降抵消了
  HOTA 收益。
- pair mAP/AP50 为 `-0.027821/-0.031902`，both-independent mAP/AP50 为
  `-0.030122/-0.032928`；四项 AP 仍未恢复。第 3 个检测/跟踪节点及 `54` 个 TrackEval
  原始文件完整。
- 相比 e8，该点 cls 同点差从 `-1.071` 扩大到 `-2.252`，det 从 `+0.869` 变为 `-0.160`，
  目前没有恢复证据。但 e12 仍处于 decoder 早中期，252 训练已自然续跑；保留到 e16 再结合
  轨迹和强主线支配关系判断，不以 e8 或单个早期节点直接否决。

## 2026-08-02 11:56 CST：0801_08 epoch-56 成熟期收口依据

- `0801_08 DN-isolated + layer-detach` e56 的 cls/det HOTA 为 `54.491/61.172`，相对
  Encoder 同点为 `-0.149/-0.738`。cls DetA/AssA 为 `+0.470/-1.860`；det DetA/AssA 为
  `-1.207/-0.022`。pair mAP/AP50 为 `-0.002456/+0.001322`，both-independent mAP/AP50
  为 `+0.001410/+0.013534`；第 14 个检测/跟踪节点及 `54` 个 TrackEval 原始文件完整。
- cls 虽严格超过绝对门槛 `54.437`，det 比 `62.393` 低 `1.221`；同一时间主线 `0801_09`
  e52 的绝对 cls/det HOTA 同时高 `0.204/1.216`。在 e4–e56 十四个完整节点后，该路线仍被
  同机制的 end-to-end 版本严格支配，后续不再视为主候选。
- 99 SSH 直连再次在 `ConnectTimeout=20` 的 TCP 连接阶段超时，无法核验精确 PGID；因此不使用
  非受控方式强停。训练和全部产物暂时保留，待管理通道恢复后再安全释放 99。

## 2026-08-02 12:32 CST：0801_09 epoch-56 严格绝对双通过（目标完成）

- `0801_09 DN-isolated + end-to-end iterative cls residual` e56 的同一 checkpoint 得到
  cls/det HOTA `54.653/62.456`，严格超过 Encoder 最终绝对门槛 `54.437/62.393`，绝对增量
  为 `+0.216/+0.063`。这不是四舍五入通过；e52 的 det `62.388` 仍按低 `0.005` 记为失败，
  只有 e56 首次满足双绝对门槛。
- 相对 Encoder e56 同点，cls/det HOTA 为 `+0.013/+0.546`；cls DetA/AssA 为
  `+0.073/-0.487`，det DetA/AssA 为 `+0.691/+0.341`。pair mAP/AP50 为
  `+0.000543/-0.000571`，both-independent mAP/AP50 为 `+0.003767/+0.008041`。第 14 个
  检测/跟踪节点、同点 AP 和 `54` 个 TrackEval 原始文件全部完成。
- 该 decoder 只增加 6 个逐层双帧 `256→8` 零初始化分类 residual 头，共 `12,336` 参数
  （`+0.0542%`，完整模型 `22,771,111`）；旧分类头冻结并等量替换，可训练参数净增为零。
  它不增加 decoder 层、attention、loss、class reweight、prototype/class-aware gate 或主矩阵
  计算；DN 使用绝对分类语义隔离，最终 decoder loss 对普通 query 的早期 residual 保持端到端
  梯度，Encoder proposal 基底仍 detached。
- 178 本机和 252 跨节点使用两个 Python 环境独立复读，均得到完全相同的指标、目录映射、
  `RAW_FILES=54` 和 `HOTA_GATE PASS`。`last_checkpoint` 精确指向 `epoch_56.pth`；文件大小
  `441,830,836` bytes，SHA256 为
  `d64df7628ac2fb200663f28067b6679a85a31482fc92eae494d12b48aeb352ab`。检测指标来自
  `val_det/epoch_55/metrics.json`，跟踪指标来自 `val_track_eval/val_track_0014/metrics.json`。
- 达标后对 178/197/252 的训练进程组 `2597803/1985233/2451495` 分别发送 TERM，并逐一确认
  进程组退出；178 和 252 实验卡均为 `0%/1 MiB`。197 上本实验 GPU 进程已退出，GPU0/1
  随即由外部用户进程 `1949805/1949806` 占用；只保留已启动的 e20 CPU 异步 TrackEval 自行
  收尾。99 因 SSH 连接超时仍无法受控停止。所有实验 checkpoint、检测和 TrackEval 产物保留。

## 2026-08-03 01:20 CST：目标提高并启动 0803_01

- 新的严格门槛改为同一 checkpoint 的 cls/det HOTA 均分别超过 Encoder 绝对基线
  `54.437/62.393`，并且两项绝对增量之和严格大于 `1.5`，等价于
  `cls HOTA + det HOTA > 118.330`。此前 `0801_09` e56 的 `54.653/62.456` 只提高
  `0.216+0.063=0.279`，因此只是新一轮基底候选，不再视为当前目标完成。
- 晚期补评显示 `0801_08 layer-detach` e60/e64/e68/e72 均未使 det 同点通过；e72 为
  `54.309/61.618`。`0801_11 terminal pair-common cls residual` e20 为
  `51.488/58.972`，绝对值仍远低于门槛。两者均不是按 e4/e8 早停，而是依据完整晚期轨迹
  与更强同机制版本支配关系降级。
- 新实验 `0803_01 iterative pair-shared objectness` 保留每帧独立分类 residual 的全部类别
  margin，只把每个 query 的 residual 类均值替换成两帧均值：
  `r'_p=r_p-mean(r_p)+(mean(r_p)+mean(r_c))/2`，curr 对称。该运算不使用类别身份、学习权重、
  reweight 或额外 loss/attention/layer；模型参数和 state tensor 相对 `0801_09` 均零增量。
  DN prefix 继续使用原绝对分类头，Encoder proposal 基底继续 detached，逐层梯度保持端到端。
- 提交 `bd1c329` 已同步到 252/197。252 Linux 环境新增 3 项语义/梯度测试全部通过；完整模型
  为 `22,771,111` 参数，delta 为 `0`。252 双卡真实数据 4-iter smoke 正常完成，total、DN、
  Encoder proposal loss 和 grad norm 全部有限，checkpoint 中六个 residual 头均得到非零更新。
- 252 formal 已超过 e1 iter 600：iter 600 为 `1.2027 s/iter`、loss `15.0387`、grad norm
  `42.2699`，双卡各约 `19.2 GiB` 且持续满载，无 Traceback/OOM/NaN/NCCL。正式 iter-50
  启动门槛已通过，状态为 `RUNNING`；不会使用 e4/e8 直接否决，按成熟轨迹持续评测。
- 同时准备在 197 以物理 `2x4`、全局 batch 8 从 `0801_09 epoch_56.pth` 恢复到 e60/e64；
  模型、优化器/EMA checkpoint 与科学 ID 不变。197 的正式续训必须等待对应双卡 4-iter
  portability smoke 真正完成；截至本记录，两 rank 正在慢速首次初始化，尚未登记为可续训。

## 2026-08-03 01:38 CST：0801_09 恢复旁路通过

- 197 portability smoke 最终完成 4 个真实 iteration，但该机异常慢：首 iter `154.7560 s`，
  到第 4 iter 的滑动均值仍为 `80.0246 s/iter`；四步 total、DN、Encoder proposal loss 与
  grad norm 均有限。由于速度比 252 同配置约慢两个数量级，不把正式续训部署到 197。
- 252 GPU2/3 使用 `0803_01` 已验证配置并仅关闭
  `iterative_cls_pair_shared_objectness`，得到与 `0801_09` 完全一致的模型语义；真实双卡
  4-iter smoke 在约 60 秒内通过，checkpoint 中六个 iterative residual 头均有限且非零更新。
- `0801_09 epoch_56.pth` 已在 252 的新 workdir 恢复 optimizer、scheduler、EMA 和 epoch 状态。
  恢复后的 e57 iter 50 为 `1.3730 s/iter`、loss `8.6312`、grad norm `46.7629`；主、DN、
  Encoder loss 均有限，GPU2/3 各约 `19.4 GiB`。因此晚期续训正式状态为 `RUNNING`，首个
  新完整评测节点为 e60；与 0803_01 并行占用 252 四卡。

## 2026-08-03 02:12 CST：0803_02 pair-shared shape 正式启动

- 新候选在每层 decoder 的普通 query 中保留两帧独立的中心 `x/y` residual，只把
  `w/h/angle` residual 替换为两帧均值；DN prefix 完全不改。该几何先验交换等变、
  class-agnostic，不含 learned gate、reweight、额外 loss/attention/layer，参数和 state
  相对 `0801_09` 均为零增量，完整模型仍为 `22,771,111` 参数。
- 252 隔离 clone 的 3 项语义/梯度测试全部通过；178 单卡配置又通过配置深拷贝、完整模型构建、
  真实数据 4-iter smoke 与 checkpoint 更新审计。正式训练于 178 GPU0 启动，唯一主 PGID
  为 `2857661`；epoch 1 iter 50 为 `0.9497 s/iter`、loss `21.0192`、grad norm
  `104.8574`，主、DN、encoder proposal loss 全部有限，GPU 约 `31.4 GiB`，无
  Traceback/OOM/NaN。
- 状态登记为 `RUNNING`，按 e4/e8/e12 及后续成熟轨迹连续评估；e4/e8 只作诊断，不作为直接
  否决。最终门槛仍是同一 checkpoint 的 cls/det HOTA 均严格超过 `54.437/62.393` 且
  总和严格大于 `118.330`。

## 2026-08-03 02:36 CST：0803_01 epoch-4 完整诊断

- e4 同一 checkpoint 的 cls HOTA/DetA/AssA 为 `30.075/23.879/41.043`，det 为
  `36.992/31.686/44.213`。相对 Encoder e4 的 cls `36.209/27.068/52.094` 与 det
  `38.753/32.454/47.466`，HOTA 分别下降 `6.134/1.761`，DetA 下降
  `3.189/0.768`，AssA 下降 `11.051/3.253`；六个跟踪分量全部下降。
- pair mAP/AP50 为 `0.121391/0.234509`，相对 Encoder e4 下降
  `0.035862/0.061625`；both-independent mAP/AP50 为 `0.160915/0.299571`，下降
  `0.023550/0.023578`。相对同机制父线 `0801_09` e4，pair mAP/AP50 又下降
  `0.022731/0.042984`，both-independent 下降 `0.028644/0.050069`。
- checkpoint、5416 条检测记录、50 个序列、TrackEval `async_done=1`、28 个 CSV 与 54 个
  原始评估文件均完整。该点说明 pair-shared objectness 当前造成检测与关联同步退化，不是
  DetA/AssA 指标搬运；但 e4 只登记为早期负向证据，训练已进入 e5，继续到 e8/e12 检查
  decoder 的延迟恢复，不作 e4 直接否决。

## 2026-08-03 02:49 CST：0803_03 仅角度共享候选完成隔离验证

- 基于 `0803_01` 的分类/objectness 耦合同步损伤检测与关联，不再派生任何分类耦合、scale
  或 gate。后备候选 `0803_03` 收缩为最局部的几何先验：每层普通 query 仅把两帧 angle
  residual 替换为均值，`x/y/w/h`、全部分类 residual 与 DN prefix 均保持独立。
- 该操作交换等变、class-agnostic，不含 reweight、额外 loss/attention/layer 或可训练权重。
  252 临时隔离 clone 的 3 项定向测试全部通过，覆盖 DN/非角度维逐位保持、交换等变、梯度
  只跨帧流入 angle，以及与完整 shape 共享互斥。配置深拷贝与完整父/新模型构建通过：两者
  均为 `22,771,111` 参数、`711` 个 state tensor，参数与 state 增量均为零。
- commit `9d90733` 只保留在本地主线；252/178 活动仓库没有热更新。候选状态为 `PREPARED`，
  尚未启动、未占用 GPU。先等待 `0803_02` 完整节点判断共享 `w/h/angle` 的作用，再在授权资源
  释放后做真实数据 smoke 与正式 iter-50 五项门槛，避免无依据并行消耗。

## 2026-08-03 03:20 CST：0801_09 epoch-60 成熟平台节点

- e60 同一 checkpoint 的 cls/det HOTA 为 `54.489/62.422`，均仍严格超过 Encoder 最终绝对
  基线 `54.437/62.393`，但绝对增量只有 `+0.052/+0.029`，合计 `0.081`；总 HOTA
  `116.911`，距严格目标 `118.330` 仍差 `1.419`，并低于 e56 的合计增益 `0.279`。
- cls HOTA/DetA/AssA 为 `54.489/44.611/68.680`，相对 e56 为
  `-0.164/-0.391/+0.360`；det 为 `62.422/54.809/73.543`，相对 e56 为
  `-0.034/-0.029/+0.010`。晚四个 epoch 主要把 cls 覆盖换成关联，没有形成净 HOTA 增益。
- e60 pair mAP/AP50 为 `0.3147/0.5313`，both-independent 为 `0.3550/0.5690`；相对 e56
  分别约 `-0.00083/-0.00118` 与 `-0.00073/-0.00096`，检测同样轻微进入平台。checkpoint、
  5416 条记录、50 序列、TrackEval `async_done=1`、54 个评估文件与总计 108 个 raw 文件完整。
- 训练已无缝进入 e61，保留 e64 作为成熟平台确认，不因 e60 单点停止；但当前证据不支持仅靠
  延长 `0801_09` 达到合并增益 `>1.5`，后续资源优先级转向已验证的结构候选。

## 2026-08-03 03:28 CST：0803_02 epoch-4 完整诊断

- e4 cls HOTA/DetA/AssA 为 `33.322/27.476/42.883`，det 为
  `37.485/33.886/42.930`。相对同机制父线 `0801_09` e4，cls 分别为
  `-0.984/-0.169/-1.724`，det 为 `-1.105/+0.247/-2.992`；相对 Encoder e4，det
  DetA 提高 `1.432`，但 det AssA 下降 `4.536`。完整 shape 共享主要损伤关联，而不是同步
  压低所有检测分量。
- pair mAP/AP50 为 `0.146883/0.272219`，相对 Encoder e4 为
  `-0.010370/-0.023915`；both-independent 为 `0.189566/0.338169`，相对 Encoder e4
  为 `+0.005101/+0.015020`。相对 `0801_09` e4，pair 为 `+0.002760/-0.005274`，
  both-independent 为 `+0.000007/-0.011471`，同样呈现单帧覆盖与配对质量分离。
- checkpoint、5416 条记录、50 序列、TrackEval `async_done=1`、54 个 eval 文件与总计 108 个
  raw 文件完整；训练已进入 e5。e4 只作“`w/h/angle` 共同约束偏强”的结构诊断，继续 e8/e12
  检查延迟恢复，不在 e4 停止。若成熟节点仍是 DetA→AssA 失配，已验证的 `0803_03` 仅角度
  共享将作为更局部接替，而不是对 shape 共享做 scale/gate 调参。

## 2026-08-03 04:08 CST：0803_01 epoch-8 完整诊断

- e8 cls HOTA/DetA/AssA 为 `39.208/32.700/49.369`，det 为
  `46.359/41.539/53.625`。相对 `0801_09` 父线 e8，cls 分别为
  `-2.764/-2.481/-3.763`，det 为 `-1.819/-3.195/-0.293`；相对 Encoder e8 的
  cls/det HOTA 为 `-6.061/-3.834`。逐层共享 objectness 在 e8 仍同时损伤覆盖与关联，
  没有形成优于保留两帧独立分类 residual 的证据。
- pair mAP/AP50 为 `0.196815/0.351435`，both-independent 为
  `0.235883/0.408144`。相对本实验 e4 分别恢复 `+0.075424/+0.116926` 与
  `+0.074968/+0.108573`，证明 e4 不能直接外推；但相对父线 e8 仍分别低
  `0.012705/0.035744` 与 `0.014078/0.029665`，相对 Encoder 的四项差距更大。
- checkpoint、5416 条检测记录、50 序列、TrackEval `async_done=1`、54 个 eval 文件与总计
  108 个 raw 文件完整。异步 TrackEval 因同机评测排队于 03:56 获得执行机会，385.2 秒正常
  完成，不是卡死。训练已进入 e9，继续到 e12；e8 只用于确认 objectness 硬共享的负向机制，
  不按早期节点停止，也不从该方向派生 scale、gate、class-aware 或 reweight 变体。

## 2026-08-03 04:43 CST：0803_02 epoch-8 完整诊断

- e8 同一 checkpoint 的 cls HOTA/DetA/AssA 为 `42.652/35.433/54.403`，det 为
  `47.723/42.688/55.569`。相对同机制父线 `0801_09` e8，cls 分别为
  `+0.680/+0.252/+1.271`，det 为 `-0.455/-2.046/+1.651`；双 HOTA 总和净增
  `+0.225`，但 det 仍表现为覆盖下降与关联改善的交换。
- pair mAP/AP50 为 `0.218642/0.387668`，both-independent 为
  `0.262145/0.443668`；相对父线 e8 四项分别提高
  `+0.009122/+0.000490/+0.012184/+0.005860`。这证明完整 shape 共享从 e4 到 e8
  已形成检测侧恢复，不能按早期节点否决；同时 det DetA 的明确损失支持把下一正交候选收缩到
  不约束 `w/h`、只共享 angle residual 的 `0803_03`。
- checkpoint、5416 条记录、50 序列、TrackEval `async_done=1`、54 个 eval 文件和总计
  108 个 raw 文件完整。训练继续到 e12 检查延迟收敛；不从完整 shape 共享派生 gate、scale、
  class-aware、reweight、额外 loss/attention/layer 变体。

## 2026-08-03 04:58 CST：0801_09 epoch-64 平台上限

- e64 cls HOTA/DetA/AssA 为 `54.326/44.326/68.761`，det 为
  `62.572/54.840/73.891`。相对严格最终 Encoder，cls/det HOTA 为 `-0.111/+0.179`，
  总和 `116.898`，合并增益仅 `0.068`；既未双通过，也未达到 `>118.330`。
- pair mAP/AP50 `0.313163/0.528518`、both-independent `0.352795/0.564884`，四项均较 e60
  回落。checkpoint、5416 条记录、50 序列、TrackEval `async_done=1`、54 个 eval 文件与
  总计 108 个 raw 文件完整。
- e56/e60/e64 构成明确平台/回落证据，04:57 精确停止 PGID `3292233`；不再把纯 epoch 延长
  作为主要探索手段。释放的 252 GPU2/3 转给零参数、非 class-aware、无 reweight 的
  `0803_03 angle-only`。

## 2026-08-03 05:08 CST：0803_03 angle-only 正式运行

- 使用隔离 checkout `/data/users/litianhao01/PairMmot_angle_0803_03`，提交 `0989cfd`，避免
  热更新 GPU0/1 的 `0803_01` 活跃仓库。目标 252 formal/smoke 配置 deepcopy、父/新模型完整
  构建与参数/state 对照通过：`22,771,111` 参数、`711` 个 state tensor，均为零增量。
- 真实双卡 4-iter smoke 的 loss `12.9387–20.2840`、grad norm `100.7381–109.7766`，
  总/DN/encoder loss 全部有限；364 MB checkpoint 更新与分类/DN 语义检查通过，无数值或
  DDP 错误。
- formal fresh 于 05:06 启动，PGID `3460950`；iter 50 为 `1.3920 s/iter`、loss
  `21.3894`、grad norm `130.3343`，双卡各约 19.2 GiB，五项启动门槛通过。该结构仅共享
  normal-query angle residual，x/y/w/h、分类、DN、loss、attention 和 decoder 深度不变；
  继续收集 e4/e8/e12 与成熟节点，不按 e4/e8 直接否决。

## 2026-08-03 05:32 CST：0803_01 epoch-12 成熟节点并停止

- e12 同一 checkpoint 的 cls HOTA/DetA/AssA 为 `43.962/35.877/57.193`，det 为
  `52.094/46.748/60.144`。相对 `0801_09` 父线 e12，cls/det HOTA 仍低
  `3.433/2.342`；相对 Encoder e12 仍低 `5.718/4.447`。从本实验 e8 到 e12，双 HOTA
  已恢复 `+4.754/+5.735`，因此不是按 e8 过早否决，而是成熟节点仍呈系统性退化。
- e12 pair mAP/AP50 为 `0.222812/0.393028`，both-independent 为
  `0.265827/0.449310`；相对本实验 e8 分别恢复 `+0.025997/+0.041594` 与
  `+0.029944/+0.041167`，但没有扭转 HOTA、DetA 与 AssA 同时落后父线的结论。
- `epoch_12.pth`、检测 metrics、5416 条记录、50 序列、TrackEval `async_done=1`、28 个 CSV
  与总计 108 个评估文件均核验完整。05:31 对精确进程组 PGID `3268273` 发送 TERM，全部成员
  退出，252 GPU0/1 均回到 `1 MiB/0%`。不再派生 objectness gate、scale、class-aware 或
  reweight 版本；资源保持空闲，先等 angle-only 的首个完整节点形成设计依据。

## 2026-08-03 05:57 CST：0803_02 epoch-12 成熟节点并停止

- e12 cls HOTA/DetA/AssA 为 `46.101/38.593/57.475`，det 为
  `50.453/46.095/57.200`。相对 `0801_09` 父线 e12 的 cls/det HOTA 仍低
  `1.294/3.983`；从本实验 e8 到 e12 虽继续恢复 `+3.449/+2.730`，但完整 shape 共享在
  成熟节点仍显著压低 det HOTA，不能解释为 e4/e8 早期慢收敛。
- e12 pair mAP/AP50 为 `0.233785/0.420219`，both-independent 为
  `0.277695/0.473812`；相对本实验 e8 分别增加 `+0.015143/+0.032550` 与
  `+0.015550/+0.030144`。AP 持续恢复但未转化为 HOTA，结合 det AssA 仅从 e8 增加
  `1.631`、小于 DetA 的 `3.407`，说明共享 `w/h/angle` 没有形成足够的时序净收益。
- `epoch_12.pth`、检测 metrics、5416 条记录、50 序列、TrackEval `async_done=1`、28 个 CSV
  与总计 108 个评估文件均核验完整。05:56 精确停止 PGID `2857661`，9 个进程成员全部退出，
  178 GPU0 回到 `1 MiB/0%`。该分支不派生 gate、scale、class-aware 或 reweight；几何主线继续
  由只共享角度且零参数的 `0803_03` 提供更局部的机制检验。

## 2026-08-03 06:27 CST：0803_04 周期切空间角度共识正式运行

- 新结构不在 sigmoid/logit 坐标直接平均回归 residual，而是先把每帧候选角度相对各自
  reference 的最短增量映射到 π 周期切空间，以圆周中点形成共享增量，再相对每帧原 reference
  重新编码。普通 query 的 x/y/w/h、分类与 DN prefix 均保持独立；结构交换等变、class-agnostic，
  无 reweight、loss、attention、decoder 层或可训练参数。
- 178 目标环境的 3 项定向测试与完整 126 项 decoder 回归通过；正式/短测配置深拷贝、父/新
  完整模型构建通过，二者均为 `22,771,111` 参数、711 个 state tensor，增量严格为零。第一次
  4-iter smoke 生成 checkpoint 但 logger 间隔 50，不能证明有限 loss/grad，故保留为不充分短测；
  retry1 改为逐迭代记录后重新 fresh 执行。
- retry1 四次 loss 为 `21.3730/20.6351/20.9249/21.2029`，grad norm 为
  `61.5507/73.7742/81.8450/75.9738`，总、DN、encoder loss 均有限；364,504,628-byte
  checkpoint 与 iterative-classification/DN-absolute 语义检查通过，GPU0 随后释放。
- formal fresh 于 06:25 启动，PGID `2893156`；iter 50 为 `0.9421 s/iter`、loss
  `21.0028`、grad norm `102.4835`，DN/encoder loss 有限，GPU0 约 31.4 GiB，9 个进程成员
  存活，无 Traceback/OOM/NaN/DDP 错误，五项启动门槛全部通过。与 `0803_03` 并行比较的唯一
  结构变量是角度增量坐标：raw logit residual 均值对 π 周期切空间圆周中点。

## 2026-08-03 06:46 CST：0803_03 raw-logit angle epoch-4 完整诊断

- e4 cls HOTA/DetA/AssA 为 `31.076/24.972/41.528`，det 为
  `37.040/31.623/44.375`。相对 `0801_09` 父线 e4，cls/det HOTA 约低
  `3.230/1.550`，DetA 低 `2.673/2.016`，AssA 低 `3.079/1.547`；两种 HOTA 分量
  同时下降，早期问题不是 DetA→AssA 搬运。
- pair mAP/AP50 为 `0.127673/0.238560`，both-independent 为
  `0.170334/0.309608`；相对父线 e4 分别约低 `0.016450/0.038933` 与
  `0.019225/0.040032`，与 TrackEval 的覆盖警报一致。
- `epoch_4.pth`、检测 metrics、5416 条记录、50 序列、TrackEval `async_done=1`、28 个 CSV
  与 108 个评估文件完整。该点只否定“raw-logit angle 共享有早期净收益”，不作为 decoder
  直接淘汰点；训练已进入 e5，继续到 e8/e12。`0803_04` 的 π 周期切空间表示保留为严格
  坐标对照，不从 e4 结果派生 scale、gate、class-aware 或 reweight 版本。

## 2026-08-03 06:59 CST：0803_05 参考框局部坐标中心共识正式运行

- 现有实验已证明分类 objectness、完整 `w/h/angle` 与 angle-only 的硬共享均会在早期损伤覆盖；
  新候选转向此前未约束的中心校正，并避免直接平均 raw logits。每层普通 query 先解码两帧候选
  中心，分别以各自 reference 的 `w/h` 将中心增量归一化，在局部坐标中取共同校正，再映回各自
  reference。`w/h/angle`、分类、DN、loss、attention 与 decoder 深度均不变。
- 该投影零参数、交换等变、class-agnostic，不含 reweight 或类别路由。252 隔离 checkout
  `/data/users/litianhao01/PairMmot_center_0803_05` 使用提交 `aa40da7`；3 项定向测试与完整
  129 项 decoder 回归通过，父/新完整模型参数和 state 均一致（`22,771,111` 参数、711 tensors）。
- 真数据双卡 4-iter smoke 的 loss 为
  `12.8954/19.1886/19.1859/19.8717`，grad norm 为
  `91.1080/90.8830/82.2923/83.7991`；总、DN、Encoder losses 均有限，364,473,270-byte
  checkpoint 与分类/DN 语义检查通过，短测后 GPU0/1 回收。
- formal fresh PGID `3549855`；epoch 1 iter50 为 `1.1465 s/iter`、loss `21.3848`、grad norm
  `102.6548`，两卡各约 19.2 GiB，进程组、日志与全部 loss group 正常，无
  Traceback/OOM/NaN/NCCL/DDP 错误，五项启动门槛通过。与 `0803_03/04` 同样收集
  e4/e8/e12 及成熟节点，不以 e4/e8 直接否决。

## 2026-08-03 07:10 CST：0803_06 帧证据分类 decoder 已准备

- 代码审计确认 `0801_09` 的两帧 iterative classification residual head 在每层都读取同一个
  融合后 decoder state；两帧已有的 prev/curr cross-attention evidence 虽已计算，却没有进入
  各自分类 residual。`0803_06` 只把这两份既有帧证据分别送回对应分类 head；共享 recurrent
  query、全部框回归、reference 更新、DN、loss 和训练协议保持父路径。
- 结构不新增参数、attention、loss、decoder 层、class-aware 路由或 score reweight。178 隔离
  checkout `/data1/users/litianhao01/PairMOT_framecls_0803_06` 固定在提交 `fd790e9`，活动中的
  `0803_04` 仓库未热更新。
- 2 项针对性测试与完整 131 项 decoder 回归通过；配置深拷贝、launcher 语法及完整模型构建通过：
  `22,771,111` 参数、参数增量 `0`、711 个 state tensor。状态为 `PREPARED`，尚未运行真数据
  smoke、未创建 formal workdir、未排队且不占 GPU；等待 `0803_04` 完整节点和资源决策后再执行
  真数据 smoke 与 formal iter-50 五门槛。

## 2026-08-03 07:46 CST：0803_04 周期角度 epoch-4 完整诊断

- 同一 e4 checkpoint 的 cls HOTA/DetA/AssA 为 `36.024/29.194/46.643`，det 为
  `43.788/34.870/57.251`。相对 `0803_03` raw-logit angle e4，双 HOTA 提高
  `+4.948/+6.748`，双 DetA 提高 `+4.222/+3.247`，双 AssA 提高 `+5.115/+12.876`；
  证明 π 周期切空间圆周中点不是等价重写，而是显著修复了角度坐标失真。
- 相对同机制父线 `0801_09` e4，cls/det HOTA 为 `+1.718/+5.198`，DetA 为
  `+1.549/+1.231`，AssA 为 `+2.036/+11.329`。相对 Encoder e4 则为 HOTA
  `-0.185/+5.035`、DetA `+2.126/+2.416`、AssA `-5.451/+9.785`：det 已强正向，cls
  仍受早期关联缺口限制，因此不能把 e4 写成目标达成，也不能在该点停止。
- pair mAP/AP50 为 `0.1634/0.3034`，both-independent 为 `0.2103/0.3774`；相对
  `0803_03` 分别提高 `0.035727/0.064840` 与 `0.039966/0.067792`，相对 Encoder e4
  也分别提高约 `0.006147/0.007266` 与 `0.025835/0.054251`。检测恢复与 TrackEval 一致。
- `epoch_4.pth` 为 369,971,828 bytes；5416 条检测、50 序列、TrackEval
  `async_done=1`、28 个 CSV 与 108 个评估文件完整。训练已自然进入 e5，继续 e8/e12 和成熟
  节点，不以 e4 直接决策停止。178 继续由本实验占用；`0803_06` 保持 `PREPARED`、未排队。

## 2026-08-03 07:54 CST：0803_07 正交组合候选已准备

- `0803_04` e4 表明 π 周期角度共识能同时恢复 DetA、AssA 与 AP，但 cls 相对 Encoder e4 仍有
  `0.185` HOTA 缺口；`0803_06` 则针对分类 head 丢弃已有帧特异 cross-attention evidence。
  `0803_07` 将两者组合：分类只读取各自帧证据，回归只对普通 query 的 angle residual 做周期
  切空间圆周中点；共享 recurrent query、x/y/w/h、DN、loss 与 decoder 主路径均不变。
- 两个开关在代码路径上严格正交：帧证据路由发生于分类输入，周期角度投影发生于随后生成的框
  residual。新增组合不变量测试证明，在相同权重与输入下，组合版本的 shared hidden state 和全部
  periodic references 与纯 `0803_04` 逐元素相同，同时输出两帧不同的分类 evidence。
- 178 隔离 checkout `/data1/users/litianhao01/PairMOT_framecls_0803_06` 固定在提交 `ee36e33`；
  组合定向测试 1/1、完整 132 项 decoder 回归、配置深拷贝、两份 launcher 语法和完整模型构建
  均通过。完整模型为 `22,771,111` 参数、参数增量 `0`、711 个 state tensor。
- 状态仅为 `PREPARED`：未运行真数据 smoke、未建立 formal workdir、未排队、不占 GPU。
  活动 `0803_04` 仓库仍为提交 `9fb501a` 且 9 个进程成员存活。待 `0803_04` e8/e12 与资源释放后，
  再在 `0803_06` 单因素和 `0803_07` 联合候选间依据成熟轨迹选择部署顺序。

## 2026-08-03 08:22 CST：0803_03 raw-logit angle epoch-8 完整诊断

- e8 cls HOTA/DetA/AssA 为 `40.644/33.308/52.868`，det 为
  `47.265/43.143/53.748`。相对自身 e4，HOTA 恢复 `+9.568/+10.225`、DetA
  恢复 `+8.336/+11.520`、AssA 恢复 `+11.340/+9.373`，证明 e4 不能直接外推
  decoder 的成熟性能。
- 相对 `0801_09` 父线 e8，cls/det HOTA 仍低 `1.328/0.913`，DetA 低
  `1.873/1.591`，AssA 低 `0.264/0.170`；raw-logit 共识已全面接近父线，但尚未带来净收益。
- pair mAP/AP50 `0.1899/0.3504`、both-independent `0.2340/0.4138`；checkpoint、
  5416 条检测、50 序列、TrackEval `async_done=1`、28 CSV 和 108 个非空文件完整。
  训练继续 e12，不在 e8 早停。

## 2026-08-03 08:28 CST：0803_05 normalized-center epoch-4 完整诊断

- e4 cls HOTA/DetA/AssA 为 `31.737/25.308/42.669`，det 为
  `37.202/32.484/43.663`。相对 raw-angle e4，双 HOTA 只提高 `0.661/0.162`，
  det AssA 反而降低 `0.712`；相对父线 e4 双 HOTA 仍低 `2.569/1.388`。
- pair mAP/AP50 `0.1266/0.2417`、both-independent `0.1684/0.3118`；checkpoint、
  5416 条检测、50 序列、TrackEval `async_done=1`、28 CSV 与 108 个非空文件完整。
  该节点仅用于早期归因，训练继续 e8/e12，不据 e4 否决局部中心坐标共识。

## 2026-08-03 09:03 CST：0803_04 periodic-angle epoch-8 强正向

- e8 cls HOTA/DetA/AssA 为 `45.587/38.410/56.277`，det 为
  `52.915/46.571/62.716`。相对自身 e4，双 HOTA 恢复 `+9.563/+9.127`；相对
  raw-angle e8，双 HOTA `+4.943/+5.650`、双 DetA `+5.102/+3.428`、双 AssA
  `+3.409/+8.968`，周期坐标优势已跨 e4/e8 两个完整节点成立。
- 相对父线 e8 双 HOTA 为 `+3.615/+4.737`；相对 Encoder e8 为
  `+0.318/+2.722`，合并同点增益 `+3.040`。这证明机制强正向，但严格最终阈值仍是
  `54.437/62.393`，所以训练继续 e12 和成熟节点，当前不写成目标达成。
- pair mAP/AP50 `0.2423/0.4242`、both-independent `0.2917/0.4922`；checkpoint、
  5416 条检测、50 序列、TrackEval `async_done=1`、28 CSV 与 108 个非空文件完整。
- `0803_07 frame-evidence + periodic-angle` 提升为下一优先候选。252 双卡配置、4-iter smoke
  配置及两份安全 launcher 已新增并通过本地语法检查；当前没有 smoke/formal workdir、队列或
  GPU 占用，待 `0803_03` e12 释放 GPU2/3 后在独立 checkout 完成全套目标环境验证。

## 2026-08-03 09:10 CST：0803_07 252 双卡目标环境验证

- 新隔离 checkout `/data/users/litianhao01/PairMmot_framecls_periodic_0803_07` 固定在 `f3752db`；
  活动 angle-only 仓库保持 `0989cfd`，未热更新。
- 显式使用 252 formal/smoke 配置的完整构建检查通过：`22,771,111` 参数、参数增量 0、
  711 个 state tensor、4 个 smoke iter。完整 decoder 回归为 `132 passed`，另有 2 个
  subtests 通过；两份 launcher 语法通过。
- 状态为 `PREPARED`。没有真数据 smoke、formal workdir、队列或 GPU 占用；等待
  `0803_03` e12 后释放 GPU2/3，再按真实 DDP smoke → checkpoint 语义审计 → formal iter-50
  五门槛顺序启动。

## 2026-08-03 09:58 CST：0803_03 raw-angle epoch-12 收口

- e12 cls HOTA/DetA/AssA `43.687/35.906/55.761`，det
  `51.887/46.564/59.976`。e8→e12 双 HOTA 继续恢复 `3.043/4.622`，但相对父线
  e12 仍低 `3.708/2.549`、相对 Encoder e12 仍低 `5.993/4.654`；成熟负向结论来自
  e4/e8/e12 完整轨迹，不是早停。
- pair mAP/AP50 `0.2209/0.4002`、both-independent `0.2618/0.4528`；checkpoint、
  5416 条检测、50 序列、TrackEval `async_done=1`、28 CSV 与 108 个非空文件完整。
  09:58 终止 PGID `3460950`，23 个成员退出，GPU2/3 释放给 `0803_07`。

## 2026-08-03 09:55 CST：0803_05 normalized-center epoch-8

- e8 cls HOTA/DetA/AssA `39.525/32.693/50.508`，det
  `45.114/40.486/51.873`；相对自身 e4 双 HOTA 恢复 `7.788/7.912`，相对父线 e8
  仍低 `2.447/3.064`。pair mAP/AP50 `0.1849/0.3383`、both-independent
  `0.2299/0.4011`，50/28/108 完整；继续 e12，不按 e8 停止。

## 2026-08-03 10:05 CST：0803_07 252 双卡正式运行

- 4-iter smoke loss `12.9405/19.2554/19.2316/20.1093`，grad norm
  `103.2723/94.9056/82.0348/79.9432`；总、DN、Encoder loss、checkpoint 更新与
  iterative-cls/DN-absolute 语义 checker 全部通过。
- formal fresh PGID `3694870`；iter50 时间/loss/grad norm 为
  `1.2858 s/iter`、`21.4228`、`119.3820`，GPU2/3 各约 19.2 GiB，进程、资源、
  正式日志、真实迭代、有限 loss 和错误扫描五项门槛通过。继续 e4/e8/e12 与成熟节点。

## 2026-08-03 10:17 CST：0803_04 periodic-angle epoch-12

- e12 cls HOTA/DetA/AssA `47.913/39.775/60.546`，det
  `55.257/49.050/64.762`；相对 e8 双 HOTA `+2.326/+2.342`，相对父线 e12
  `+0.518/+0.821`，周期角度机制保持稳定正增益且未平台。
- 相对 Encoder e12 双 HOTA 仍低 `1.767/1.284`，尚未达到严格目标。pair mAP/AP50
  `0.2577/0.4395`、both-independent `0.3023/0.4959`；checkpoint、5416 条检测、
  50 序列、TrackEval `async_done=1`、28 CSV 和 108 个非空文件完整。
- PGID `2893156` 继续运行到 e16 与更成熟节点；不因 e12 尚未超过 Encoder 而停止，重点观察
  后续 cls 关联是否延迟追上，并与 252 `0803_07` 的帧证据分类路径形成正交对照。

## 2026-08-03 11:22 CST：0803_05 normalized-center epoch-12 收口

- e12 cls HOTA/DetA/AssA 为 `43.161/36.061/53.834`，det 为
  `49.396/44.754/56.154`；相对 e8 双 HOTA 虽继续恢复 `+3.636/+4.282`，但相对父线 e12
  仍低 `4.234/5.040`，相对 Encoder e12 仍低 `6.519/7.145`。e4/e8/e12 三个完整节点均为系统性负向，
  因而此结论不是以 e4/e8 直接否决。
- pair mAP/AP50 为 `0.2142/0.3842`，both-independent 为 `0.2592/0.4419`；
  381,025,078-byte checkpoint、5416 条检测、50 序列、28 CSV 和 108 个非空评估文件完整。
- 11:18 对精确 PGID `3549855` 做配置路径预检后发送 TERM，23 个成员全部退出；GPU0/1 回落至
  1 MiB，GPU2/3 上的 `0803_07` 未受影响。归一化中心共识收口为成熟负向，空闲双卡转给零参数
  `0803_06 frame-evidence-cls` 单因素归因。

## 2026-08-03 11:31 CST：0803_06 frame-evidence-cls 正式运行

- 新建 252 GPU0/1 双卡配置及 smoke/formal 启动器；隔离 checkout
  `/data/users/litianhao01/PairMmot_framecls_0803_06` 固定在提交 `0585d9a`。132 项 decoder 回归与
  2 个 subtests 通过；父/新模型均为 `22,771,111` 参数、711 state tensors，参数增量严格为零。
- 真数据 4-iter smoke loss `12.9405/19.3125/19.1891/20.0475`，grad norm
  `102.9508/90.0784/88.7070/84.7921`；总、DN、Encoder loss 有限，364,473,270-byte
  checkpoint 与 iterative-cls/DN-absolute 语义检查通过。
- formal fresh PGID `3765372`；iter50 为 `1.2946 s/iter`、loss `21.4356`、grad norm
  `109.5840`，7 个进程成员存活，GPU0/1 各约 19.2 GiB，五项门槛通过。继续 e4/e8/e12 与成熟节点。

## 2026-08-03 11:31 CST：0803_04 periodic-angle epoch-16

- e16 cls HOTA/DetA/AssA `48.474/40.377/60.513`，det
  `55.272/49.385/64.188`；相对 e12 双 HOTA为 `+0.561/+0.015`，cls 仍缓慢上升，det 在单个
  四 epoch 区间接近平台，但尚不能由一个区间判定长期收敛结束。
- pair mAP/AP50 `0.2628/0.4523`，both-independent `0.3057/0.5042`；386,622,452-byte
  checkpoint、5416 条检测、50 序列、28 CSV 与 108 个非空文件完整。
- 相对严格最终阈值仍低 `5.963/7.121`，目标未达到。PGID `2893156` 继续至 e20，检验 det 平台
  是否持续及 cls 的延迟关联增益；不因单个 e12→e16 区间收窄就停止正向分支。

## 2026-08-03 11:45 CST：0803_07 组合分支 epoch-4

- e4 cls HOTA/DetA/AssA `32.535/24.868/45.326`，det
  `38.723/31.579/49.105`；相对 periodic-angle 单因素 `0803_04` e4，双 HOTA
  `-3.489/-5.065`，双 DetA `-4.326/-3.291`，双 AssA `-1.317/-8.146`。相对父线 e4 为
  `-1.771/+0.133`，说明 periodic-angle 的早期正收益被直接 frame-evidence 分类路由基本抵消。
- pair mAP/AP50 `0.1220/0.2350`，both-independent `0.1618/0.3021`；相对 periodic-angle
  单因素分别低 `0.0414/0.0684` 和 `0.0485/0.0753`。369,968,374-byte checkpoint、
  5416 条检测、50 序列、28 CSV 与 108 个非空文件完整。
- PGID `3694870` 已进入 e5，继续 e8/e12，不能以 e4 直接淘汰。下一结构候选不再用原始帧证据
  替换共享分类状态，而保留共享状态作为精确 pair midpoint，仅注入两帧证据相对均值的 swap-odd
  细节；零参数、class-agnostic、无 reweight，先静态准备、不抢占三条活动训练。

## 2026-08-03 11:52 CST：0803_08 common-preserving frame-detail 已准备

- 两帧分类输入定义为 `shared ± 0.5 * (frame_prev - frame_curr)`；因此算术平均精确回到共享
  `layer_output`，差值严格 swap-odd。框回归、reference 更新、共享 recurrent query、DN、loss、
  attention 数量和 decoder 深度与 periodic-angle 单因素一致；结构零参数、class-agnostic、无 reweight。
- 新增不变量测试覆盖：父/新共享 hidden 与全部 references 逐元素一致、两帧分类输入 midpoint 等于
  shared hidden、帧输入确实不同、与 direct frame-evidence 路由互斥。178 隔离 checkout
  `/data1/users/litianhao01/PairMOT_framedetail_0803_08` 固定在提交 `8dd19d8`。
- 该环境无 `pytest` 包，首次验证在导入前退出；改用同一测试文件的标准库入口后 133 项完整测试通过。
  父/新完整模型均为 `22,771,111` 参数、711 state tensors，参数增量为零；配置、launcher 和
  4-iter smoke 设置通过。状态为 `PREPARED`：未运行真数据 smoke、未建 formal workdir、不占 GPU。

## 2026-08-03 12:02 CST：0803_09 log-size tangent + periodic-angle 已准备

- 宽高是正物理量；新投影先解码每帧候选尺寸，计算相对各自 reference 的 log 比例增量，平均后
  再映回各自 reference。角度沿用已验证正向的 π 周期切空间 midpoint；中心、分类、recurrent query、
  DN、loss、attention 与 decoder 深度不变。该结构零参数、交换等变、class-agnostic、无 reweight。
- 首次随机大残差不变量测试触及 sigmoid 尺寸上界，边界 clamp 使 log 比例无法保持严格相等；这不是
  运行错误。测试改为非饱和正常增量以验证解析不变量，同时实现继续保留边界 clamp。178 隔离 checkout
  `/data1/users/litianhao01/PairMOT_logshape_0803_09` 固定在修正提交 `35e18f1`。
- 完整 135 项 decoder 测试通过；父/新模型均为 `22,771,111` 参数、711 state tensors，增量为零；
  配置、launcher 与 4-iter 设置通过。状态为 `PREPARED`，未运行真数据 smoke、未排队、不占 GPU。

## 2026-08-03 12:46 CST：0803_04 periodic-angle epoch-20

- e20 cls HOTA/DetA/AssA `49.446/41.202/61.747`，det
  `55.397/49.953/63.587`；相对 e16 双 HOTA `+0.972/+0.125`，cls DetA/AssA
  `+0.825/+1.234`，det DetA `+0.568`、AssA `-0.601`。分类延迟增益继续出现，检测关联趋平但
  覆盖仍上升，不能把 e16 的单区间平台当作最终收敛。
- pair mAP/AP50 `0.2706/0.4658`，both-independent `0.3132/0.5154`；相对 e16 分别提高
  `0.0078/0.0135` 和 `0.0075/0.0112`。392,152,244-byte checkpoint、5416 条检测、
  50 序列、28 CSV 与 108 个非空文件完整。
- 相对严格最终阈值仍低 `4.991/6.996`，目标未达到。PGID `2893156` 已自然进入 e21，继续 e24；
  同时保留 `0803_09` 作为 det 平台的零参数尺寸几何补充，不抢占当前正向长跑。

## 2026-08-03 13:04 CST：0803_06 frame-evidence-cls epoch-4

- e4 cls HOTA/DetA/AssA `30.698/24.130/41.666`，det
  `38.350/30.533/49.530`；相对 periodic-angle 单因素 e4 为 `-5.326/-5.438`，相对
  Encoder e4 为 `-5.511/-0.403`。直接帧证据分类路径在早期同时降低 DetA 与 AssA。
- pair mAP/AP50 `0.1200/0.2291`，both-independent `0.1621/0.3011`；369,968,182-byte
  checkpoint、5416 条检测、50 序列、28 CSV 与 108 个非空文件完整，异步 TrackEval 正常结束。
- PGID `3765372` 的 23 个成员已进入 e5。该实验与 `0803_07` 都继续 e8/e12，不以 e4 早停；
  当前因果证据降低 direct frame-evidence 路由优先级，并提高精确保留 shared midpoint 的
  `0803_08 common-preserving frame-detail` 优先级。

## 2026-08-03 13:22 CST：0803_07 frame-evidence-cls + periodic-angle epoch-8

- e8 cls HOTA/DetA/AssA `41.380/32.175/56.046`，det
  `47.515/39.879/58.497`；相对 periodic-angle 单因素同点为 `-4.207/-5.400`，相对
  Encoder 同点为 `-3.889/-2.678`。虽然 e4→e8 双 HOTA 增长 `+8.845/+8.792`，直接分类
  路由的差距在成熟节点仍然显著。
- pair mAP/AP50 `0.1887/0.3403`，both-independent `0.2297/0.3976`；相对 periodic-angle
  分别下降 `0.0536/0.0839` 和 `0.0620/0.0946`。375,531,382-byte checkpoint、127426 条
  检测、50 序列、28 CSV 与 108 个非空文件完整，异步评测正常完成。
- PGID `3694870` 的 23 个成员已进入 e9，继续 e12 后再释放。该结果不支持扩展 direct
  frame-evidence 路由，下一分类结构只考虑精确保留公共 midpoint 的 `0803_08`。

## 2026-08-03 14:05 CST：0803_04 periodic-angle epoch-24 与资源释放

- e24 cls HOTA/DetA/AssA `50.133/41.785/62.533`，det
  `55.346/50.198/63.103`；相对 e20 双 HOTA `+0.687/-0.051`。cls 延迟收益仍在，但 det 的
  DetA 增加 `0.245`、AssA 下降 `0.484`，检测 HOTA 已平台。
- pair mAP/AP50 `0.2754/0.4742`，both-independent `0.3181/0.5229`；相对 e20 分别提高
  `0.0048/0.0084` 和 `0.0049/0.0075`。397,682,100-byte checkpoint、5416 条检测、
  50 序列、28 CSV 与 108 个非空文件完整。
- 严格阈值仍差 `4.304/7.047`。精确停止 PGID `2893156` 后 9 个成员全部退出，GPU0 为
  `1 MiB/0%`；e24 checkpoint 保留。下一步在 178 启动零参数 `0803_09 log-size tangent +
  periodic-angle`，检验宽高乘法几何能否突破 det 平台。

## 2026-08-03 14:12 CST：0803_09 log-size tangent + periodic-angle 正式启动

- 隔离 checkout `/data1/users/litianhao01/PairMOT_logshape_0803_09` 固定 `35e18f1c`。真数据
  4-step smoke loss `21.3700/20.6566/20.9046/21.1935`，grad
  `117.3254/104.8011/100.5948/101.3451`，全部有限；364,505,012-byte checkpoint 与
  iterative-cls/DN 语义检查通过。
- fresh formal PGID `2971994`；iter50 `0.9750 s/iter`、loss `21.0017`、grad `109.5454`，
  9 个成员存活，GPU0 约 31.4 GiB，错误扫描、provenance 与目标 workdir 五门槛通过。
- 状态为 `RUNNING`，继续 e4/e8/e12。结构零参数、class-agnostic、无 reweight，不增加 attention
  或 decoder 深度；normal query 只对宽高采用 reference-local log 比例共识并保留周期角度共识。

## 2026-08-03 14:29 CST：0803_06 frame-evidence-cls epoch-8

- e8 cls HOTA/DetA/AssA `40.922/32.859/53.835`，det
  `46.854/41.568/54.556`；相对 periodic-angle 同点为 `-4.665/-6.061`，相对 Encoder
  同点为 `-4.347/-3.339`。e4→e8 双 HOTA 虽增长 `10.224/8.504`，成熟差距仍显著。
- pair mAP/AP50 `0.1906/0.3551`，both-independent `0.2334/0.4148`；相对 periodic-angle
  分别下降 `0.0517/0.0691` 和 `0.0583/0.0774`。375,533,238-byte checkpoint、128933 条
  检测、50 序列、28 CSV 与 108 个非空文件完整。
- PGID `3765372` 的 23 个成员已进入 e9，继续 e12。单因素和组合的 e8 证据一致，不再扩展
  direct frame-evidence 路由；下一分类候选只保留 `0803_08` 的 exact shared midpoint。

## 2026-08-03 15:01 CST：0803_07 frame-evidence-cls + periodic-angle epoch-12

- e12 cls HOTA/DetA/AssA `43.504/34.301/58.469`，det
  `51.168/44.130/61.359`；相对 periodic-angle 同点为 `-4.409/-4.089`，相对 Encoder
  同点为 `-6.176/-5.373`。e4/e8/e12 三点成熟轨迹均不支持 direct frame-evidence 路由。
- pair mAP/AP50 `0.2034/0.3683`，both-independent `0.2430/0.4213`；相对 periodic-angle
  同点分别下降 `0.0543/0.0712` 和 `0.0593/0.0746`。381,031,670-byte checkpoint、143610 条
  检测、50 序列、28 CSV 与 108 个非空文件完整。
- 精确停止 PGID `3694870` 后 23 个成员全部退出，GPU2/3 为 `1 MiB/0%`；e12 checkpoint
  保留。下一步部署 `0803_08 common-preserving frame-detail + periodic-angle` 双卡候选。

## 2026-08-03 15:10 CST：0803_08 common-preserving frame-detail 双卡正式启动

- 252 隔离 checkout `/data/users/litianhao01/PairMmot_framedetail_0803_08` 固定 `08356f9`。
  135 项回归、2 个 subtest、launcher 语法与目标配置构建通过；父/新模型均为 `22,771,111`
  参数、711 state tensors，增量为零。
- 首次全构建受继承 `PYTHONPATH` 污染而解析到旧 registry 类，未进入训练；清空路径并验证目标类
  来源后通过。真数据 DDP smoke loss `12.9402/19.2481/19.2302/20.1285`，grad
  `103.2465/94.2024/82.7476/82.7848`，checkpoint 与 iterative-cls/DN 语义检查通过。
- fresh formal PGID `3940521`；iter50 `1.2866 s/iter`、loss `21.4134`、grad `109.5162`，
  7 个成员、GPU2/3 各约 19.2 GiB，五门槛通过。状态为 `RUNNING`，继续 e4/e8/e12。

## 2026-08-03 15:25 CST：0803_09 log-size tangent + periodic-angle epoch-4

- e4 cls HOTA/DetA/AssA `36.930/29.828/47.715`，det
  `44.486/36.346/56.708`；相对 periodic-angle 单因素同点双 HOTA `+0.906/+0.698`、
  双 DetA `+0.634/+1.476`，相对 Encoder e4 双 HOTA `+0.721/+5.733`。
- pair mAP/AP50 `0.1743/0.3157`，both-independent `0.2193/0.3845`；相对 periodic-angle
  同点提高 `0.0109/0.0123` 和 `0.0090/0.0071`。369,973,748-byte checkpoint、114290 条
  检测、50 序列、28 CSV 与 108 个非空文件完整。
- 该零参数尺寸几何机制在 HOTA、DetA 和 AP 上一致正向，继续 e8/e12；严格最终阈值尚未达到，
  PGID `2971994` 的 9 个成员已进入 e5。

## 2026-08-03 15:57 CST：0803_06 frame-evidence-cls epoch-12

- e12 cls HOTA/DetA/AssA `45.752/36.629/59.854`，det
  `52.950/46.363/62.645`；e8→e12 双 HOTA `+4.830/+6.096`，相对 periodic-angle 同点
  为 `-2.161/-2.307`，相对 Encoder e12 为 `-3.928/-3.591`。
- pair mAP/AP50 `0.2238/0.4040`，both-independent `0.2689/0.4633`；相对 e8 分别提高
  `0.0332/0.0489` 和 `0.0355/0.0485`。381,033,910-byte checkpoint、147478 条检测、
  50 序列、28 CSV 与 108 个非空文件完整。
- 单因素比 `0803_07` 联合分支 e12 高 `2.248/1.782` HOTA，表明组合存在负交互；单因素自身
  缺口继续快速收窄。PGID `3765372` 已进入 e13，继续 e16，不以 e12 直接否决。

## 2026-08-03 16:22 CST：0803_10 shared log-area + periodic-angle 完成准备

- 新 decoder 只共享 normal query 的各向同性对数面积增量，同时原样保留两帧各自的对数纵横比
  增量；尺寸仍相对各自 reference 解码，角度沿用 pi-periodic tangent 共识。中心、分类、DN、
  loss、attention 数和 decoder 深度不变，结构 class-agnostic、无 reweight。
- 178 隔离 checkout `/data1/users/litianhao01/PairMOT_logarea_0803_10_repo` 固定 `8998142`；
  137 项完整 decoder 回归通过。目标配置完整构建为 `22,771,111` 参数、711 state tensors，
  参数增量严格为零；4-iter smoke 配置深拷贝与 launcher 语法检查通过。
- 状态为 `PREPARED/WAITING_GPU`。不抢占正在训练的 `0803_09`；先看其 e8/e12 是否保持
  log-size 全共享的正增益，再决定在下一块合法空闲 GPU 上启动 `0803_10` smoke/formal。

## 2026-08-03 16:41 CST：0803_09 log-size tangent + periodic-angle epoch-8

- e8 cls HOTA/DetA/AssA `46.170/38.770/57.150`，det
  `53.539/47.234/63.398`；相对 periodic-angle 同点双 HOTA `+0.583/+0.624`，合计
  `+1.207`；相对 Encoder e8 为 `+0.901/+3.346`。
- pair mAP/AP50 `0.2470/0.4398`，both-independent `0.2926/0.4984`；相对
  periodic-angle 同点四项仍全正：`+0.0047/+0.0156/+0.0009/+0.0062`。优势较 e4 收窄，
  但 HOTA、DetA 与 AP 方向一致，不能在 e8 否决。
- 375,564,404-byte checkpoint、153665 条检测、50 序列、28 CSV 与 108 个非空文件完整；
  异步 TrackEval 正常结束。相对严格最终阈值仍差 `8.267/8.854`，PGID `2971994` 已恢复 e9，
  继续 e12 并建立 `val_track_0003` 完整性监控。

## 2026-08-03 16:53 CST：0803_08 common-preserving frame-detail epoch-4

- e4 cls HOTA/DetA/AssA `32.065/25.138/43.440`，det
  `39.067/31.078/50.405`；相对 periodic-angle 同点双 HOTA `-3.959/-4.721`，相对
  Encoder e4 为 `-4.144/+0.314`。精确保留 shared midpoint 仍未避免早期分类与关联损失。
- pair mAP/AP50 `0.1332/0.2541`，both-independent `0.1721/0.3160`；相对
  periodic-angle 同点四项为 `-0.0302/-0.0493/-0.0382/-0.0614`。
- 369,968,886-byte checkpoint、95613 条检测、50 序列、28 CSV 与 108 个非空文件完整；异步
  TrackEval 正常结束。PGID `3940521` 的 23 个成员继续运行，已建立 e8 `val_track_0002`
  完整性监控；该负向 e4 只降低优先级，不作为直接停止理由。

## 2026-08-03 17:00 CST：0803_11 late geometric consensus 完成准备

- 新候选保留第一层逐帧独立几何，只在 decoder 最后两层执行 reference-local log-size 共识与
  pi-periodic angle tangent 共识，目标是保留早层帧特异探索并减少每层强约束造成的后期增益衰减。
  中心、分类、DN、loss、attention 和深度不变，结构 class-agnostic、无 reweight。
- 178 隔离 checkout `/data1/users/litianhao01/PairMOT_lategeom_0803_11_repo` 固定 `4c58c57`；
  定向测试确认 4 层测试模型只调用最后两层投影，138 项完整 decoder 回归通过。目标模型为
  `22,771,111` 参数、711 state tensors，参数增量为零；配置深拷贝、全模型构建与 launcher
  语法均通过。
- 状态为 `PREPARED/WAITING_GPU`。与 `0803_10` 一起等待合法空闲 GPU；部署优先级由
  `0803_09` e12 的增益衰减和 AssA 分解决定。

## 2026-08-03 17:05 CST：0803_10 完成 252 双卡部署准备

- 新增 252 2x4 formal/smoke 配置与 GPU0/1 默认 launcher；可通过受控环境变量切换到另一对合法
  空闲卡。隔离 checkout `/data/users/litianhao01/PairMmot_logarea_0803_10_252` 固定
  `1f0147c`，未更新正在训练的 `0803_06/08` 仓库。
- 252 目标配置深拷贝、launcher 语法和完整模型构建通过：`22,771,111` 参数、711 state tensors、
  参数增量 0。状态为 `PREPARED/WAITING_GPU`；四卡仍被 06/08 合法占用，未启动 smoke。

## 2026-08-03 17:06 CST：197 迁移支线暂停

- 197 在 114 MB bundle 完整落盘并完成隔离 clone 后主动断开；后续依次出现两次 connection
  refused 和一次 `No route to host`。隔离目录 `/data/users/litianhao/PairMOT_logarea_0803_10_197`
  仍停在旧提交 `2024222`，尚未 fetch/checkout、未构建、未启动 GPU。
- 不删除现场、不高频重连。197 标记为 `UNREACHABLE/RECOVERABLE_PREP`；主线继续使用 252/178，
  目标并未因该慢资源离线而阻塞。

## 2026-08-03 17:24 CST：0803_06 frame-evidence-cls epoch-16

- e16 cls HOTA/DetA/AssA `47.584/37.952/62.296`，det
  `55.930/47.734/67.966`；相对 e12 双 HOTA `+1.832/+2.980`。相对 periodic-angle e16
  为 `-0.890/+0.658`：检测已反超，分类缺口继续收窄，晚收敛追赶仍存在。
- pair mAP/AP50 `0.2363/0.4165`，both-independent `0.2795/0.4724`；相对 e12 四项继续增长
  `+0.0125/+0.0125/+0.0106/+0.0091`，但增速较 e8→e12 放缓。
- 386,533,238-byte checkpoint、153698 条检测、50 序列、28 CSV 与 108 个非空文件完整；
  异步 TrackEval 正常结束。相对严格最终阈值仍差 `6.853/6.463`，PGID `3765372` 的 23 个成员
  已恢复 e17；继续 e20，并建立 `val_track_0005` 完整性监控。

## 2026-08-03 17:33 CST：0803_06 成熟收口并启动 0803_10

- 回读权威长轨迹后，Encoder e16 为 `51.091/58.320`，原始 `0801_09` decoder e16 为
  `50.036/56.933`。`0803_06` e16 `47.584/55.930` 被同模型原始 decoder 严格支配
  `2.452/1.003`，pair mAP/AP50 也低 `0.0334/0.0566`。收口依据为 e4/e8/e12/e16 四个完整
  节点与强版本支配，不是 e4/e8 单点淘汰。
- 对精确 PGID `3765372` 发送 TERM 后 23 个成员全部退出，GPU0/1 为 `0%/1 MiB`；e16
  checkpoint 和完整评估保留，e20 监控撤销。
- `0803_10` 首次 smoke 在训练前因 cache 路径大小写错误退出，只产生 147-byte 失败日志、未占
  GPU；修正为实际 `PairMmot` 路径并使用 `_retry1` workdir。真实 4-iter smoke loss
  `12.9228/19.5098/19.6027/21.1315`，grad `103.6034/98.2696/107.9666/114.7782`，
  364,501,942-byte checkpoint 与 iterative-cls/DN 语义检查通过。
- formal PGID `4053545`；iter50 `1.2074 s/iter`、loss `21.3965`、grad `114.6571`，7 个
  成员、GPU0/1 各约 19.2 GiB，错误扫描、资源、进程组、provenance 与 workdir 五门槛通过。
  状态 `RUNNING`，已建立 e4 `val_track_0001` 监控，后续不以 e4/e8 直接否决。

## 2026-08-03 17:49 CST：197 部分恢复审计

- 197 SSH 已恢复并返回主机时间；隔离仓库仍为旧 HEAD `2024222`，113,797,579-byte 主 bundle
  与 2,560-byte 补丁 bundle 均在原位。
- GPU 管理查询经显式 5 秒边界仍以 `124` 超时，不能验证空闲卡、显存或训练进程；因此不宣称
  GPU 可用，也不在该机启动 smoke/formal。状态从网络不可达收窄为 SSH 可达、GPU 子系统不可用。

## 2026-08-03 18:00 CST：0803_09 epoch 12 完整评估

- cls HOTA/DetA/AssA `49.206/41.546/60.406`，det `56.275/49.874/66.109`；相对 Encoder
  同点 `49.680/56.541` 暂低 `0.474/0.266`，但相对原始 `0801_09` decoder 同点
  `47.395/54.436` 提高 `1.811/1.839`，联合优势 `3.650`。
- 相对 periodic-angle e12 `47.913/55.257` 提高 `1.293/1.018`，联合优势 `2.311`；pair
  mAP/AP50 `0.2687/0.4703`、both-independent `0.3145/0.5247`，四项均较 e8 继续增长。
- 381,087,092-byte checkpoint、5416 条记录、50 序列、28 CSV、108 个非空评估文件与
  `async_done=1` 完整。该结构是在同模型原始 decoder 上形成强正交增益，且原始 decoder 要到
  e40/e56 才完成双超；因此明确保留长轨迹，继续 e16/e24/e40，不以 e12 未过最终阈值停止。

## 2026-08-03 18:35 CST：0803_08 epoch 8 完整评估

- cls HOTA/DetA/AssA `40.688/33.752/51.391`，det `47.811/42.903/55.164`；相对 Encoder
  同点 `45.269/50.193` 低 `4.581/2.382`，相对 periodic-angle 同点 `45.587/52.915`
  低 `4.899/5.104`。
- e4 到 e8 仍增长 `8.623/8.744`，但结构差距未收窄；pair mAP/AP50 `0.1995/0.3624`、
  both-independent `0.2402/0.4175`，也明显低于 periodic-angle e8。
- 375,532,918-byte checkpoint、5416 条记录、50 序列、28 CSV、108 个非空文件完整，异步
  评估 408.8 秒正常结束。遵守不以 e4/e8 直接否决的约束，继续 e12；e12 后再按完整三节点
  轨迹判断是否释放 GPU2/3 给已准备的 0803_11。

## 2026-08-03 18:45 CST：0803_11 252 双卡隔离准备

- 新增 252 `2xb4` formal/smoke 配置和双卡 launcher，提交 `e09efb9`；默认 GPU2/3，端口
  `29857/29858`，workdir 与 178 单卡预案严格隔离。
- 252 新 checkout `/data/users/litianhao01/PairMmot_lategeom_0803_11_252` 固定 HEAD
  `e09efb9`。首次构建暴露登录环境残留 `PYTHONPATH` 指向活动主仓库；未改动模型或活动仓库，
  显式固定隔离导入根后重跑通过。
- 真实隔离导入路径已核验；full build 为 22,771,111 参数、相对父线零增量、711 state tensors，
  4-iter 配置有效。后两层投影单测 `1 passed/137 deselected`；当前只做静态准备，不占 GPU，
  等 0803_08 e12 成熟判定后再执行 DDP smoke。

## 2026-08-03 19:06 CST：0803_10 epoch 4 完整评估

- cls HOTA/DetA/AssA `32.399/26.881/41.726`，det `39.251/34.057/46.221`；相对 Encoder
  同点 `36.209/38.753` 为 `-3.810/+0.498`，相对 periodic-angle `36.024/43.788`
  为 `-3.625/-4.537`。
- pair mAP/AP50 `0.1351/0.2571`、both-independent `0.1799/0.3309`；仅比 0803_08 e4
  HOTA 高 `0.334/0.184`，共享面积、保留 frame-specific aspect 的早期约束尚未形成正向收益。
- 369,970,422-byte checkpoint、5416 条记录、50 序列、28 CSV、108 个非空文件完整；异步
  评估 362.1 秒结束。按约束继续 e8/e12，不因 e4 直接停止。

## 2026-08-03 19:22 CST：0803_09 epoch 16 完整评估

- cls HOTA/DetA/AssA `50.732/42.845/62.101`，det `57.218/50.913/66.744`；相对 Encoder
  同点 `51.091/58.320` 为 `-0.359/-1.102`，相对原始 `0801_09` decoder 同点
  `50.036/56.933` 为 `+0.696/+0.285`，联合优势 `0.981`。
- 相对 periodic-angle e16 `48.474/55.272` 仍提高 `2.258/1.946`，联合 `+4.204`；e12→e16
  自身增长 `1.526/0.943`。pair mAP/AP50 `0.2789/0.4860`、both-independent
  `0.3251/0.5393`，四项较 e12 继续提高。
- 386,613,748-byte checkpoint、5416 条记录、50 序列、28 CSV、108 个非空文件完整；异步
  评估 253.1 秒结束。对原始 decoder 的优势较 e12 收窄，但仍为双侧领先；原始 decoder 到
  e40/e56 才双超，故继续 e20/e24 并观察成熟期优势，不在 e16 截断。

## 2026-08-03 19:36 CST：197 恢复并启动 0803_11 formal

- 低频复查确认 `nvidia-smi` 恢复；GPU0/1 有外部负载，GPU2--5 空闲。197 隔离 checkout
  `/data/users/litianhao/PairMOT_lategeom_0803_11_197` 固定提交 `38ae0d4`，未修改任何活动仓库。
- GPU4/5 真实 4-iter DDP smoke loss `12.9516/19.2974/19.3485/20.0987`、grad
  `108.6312/136.9732/114.6392/107.1473` 全有限；364,473,447-byte checkpoint、错误扫描与
  iterative-cls residual/DN absolute 语义检查通过。总 wall 110.4 秒含初始化与 checkpoint 检查，
  不再出现此前约 80 秒/iter 的 GPU 异常。
- fresh formal 精确 PGID `53708`；iter50 `1.4287 s/iter`、loss `21.3917`、grad
  `120.7769`，7 个进程，GPU4/5 各 19,226 MiB，错误扫描干净，HEAD/config/workdir 一致，
  五门槛通过。状态 `RUNNING`，首个完整判断点为 e4。

## 2026-08-03 20:28 CST：0803_08 成熟停止并由 0803_12 接替

- 0803_08 e12 cls HOTA/DetA/AssA `44.177/36.085/57.044`，det
  `52.763/46.795/61.568`；相对 Encoder 同点低 `5.503/3.778`，相对 periodic-angle 低
  `3.736/2.494`，相对原始 decoder 低 `3.218/1.673`。e8→e12 相对差距继续恶化。
- pair mAP/AP50 `0.2245/0.3954`、both-independent `0.2653/0.4479`；381,032,310-byte
  checkpoint、5416 条记录、50 序列、28 CSV、108 个非空文件完整。依据 e4/e8/e12 三节点
  成熟负向轨迹精确 TERM PGID `3940521`，23→0，GPU2/3 为 `0%/1 MiB`。
- 新增零参数 `0803_12 progressive geometric consensus`：首层自由，倒数第二层只共享
  log-area 与周期角，末层共享完整 log-size 与周期角；无 class-aware、reweight、额外层或 attention。
  目标单测 `1 passed/138 deselected`，模型 22,771,111 参数、零增量、711 state tensors。
- 4-iter DDP smoke 四步 loss/grad 全有限，364,502,134-byte checkpoint、错误扫描与语义检查
  通过。fresh formal PGID `4189798`；iter50 `1.2941 s/iter`、loss `21.3858`、grad
  `113.3648`，7 个进程、GPU2/3 各 19,192 MiB，五门槛通过，状态 `RUNNING`。

## 2026-08-03 20:32 CST：0803_09 epoch 20 后期回落

- e20 cls HOTA/DetA/AssA `49.781/41.730/61.763`，det `57.217/50.699/67.029`；相对 e16
  为 `-0.951/-0.001`，相对原始 decoder 同点为 `-1.062/-0.816`，全层 log-size 共识的
  e12 优势已在后期反转。
- pair mAP/AP50 `0.2732/0.4751`、both-independent `0.3152/0.5221`；392,131,764-byte
  checkpoint、50 序列、28 CSV、108 个非空文件及异步完成证据齐全。
- 不以该单点立即停止；保持 PGID `2971994` 到 e24，结合 e12→e16→e20 的连续趋势作成熟判定。

## 2026-08-03 20:35 CST：0803_10 epoch 8 完整评估

- e8 cls HOTA/DetA/AssA `41.567/34.618/52.385`，det HOTA/DetA/AssA
  `48.238/43.861/54.777`；相对 Encoder 同点
  `-3.702/-1.955`，相对 periodic-angle 同点 `-4.020/-4.677`。e4 的 det 单侧微增未保持。
- pair mAP/AP50 `0.1986/0.3565`、both-independent `0.2430/0.4191`；375,538,358-byte
  checkpoint、50 序列、28 CSV、108 个非空文件与异步完成指标齐全。
- 面积共享目前双侧落后，但按晚收敛约束继续到 e12，不以 e8 直接否决。

## 2026-08-03 20:44 CST：0803_13 terminal geometry 已准备

- 新增 terminal-only log-size + periodic-angle 共识：前三层 reference 更新完全逐帧独立，只在
  最终输出层投影一次，因此配对几何约束不再反馈到后续 cross-attention。
- 结构为零参数、class-agnostic、无 reweight、无新增层或 attention；目标单测通过，确认
  log-size/angle 各只调用一次，互斥检查和有限值检查通过。
- 隔离仓库 `/data1/users/litianhao01/PairMOT_terminal_0803_13` 固定提交 `1b7f904`；完整构建
  22,771,111 参数、零增量、711 state tensors。状态 `PREPARED/WAITING_178_AFTER_0803_09`。

## 2026-08-03 21:48 CST：0803_09 epoch 24 成熟停止并切换 smoke

- e24 cls HOTA/DetA/AssA `50.256/42.210/61.971`，det `57.489/50.920/67.374`；相对
  原始 decoder 同点 `51.709/58.781` 为 `-1.453/-1.292`。e12 双正、e16 收窄、e20 反转、
  e24 继续双负，确认全层 log-size 共识后期过约束。
- pair mAP/AP50 `0.2742/0.4787`、both-independent `0.3165/0.5268`；397,650,228-byte
  checkpoint、50 序列、28 CSV、108 个非空文件齐全。精确 TERM PGID `2971994`，成员
  `9→0`，GPU0 `0%/1 MiB`，断点保留。
- 0803_13 随后在隔离提交 `1b7f904` 启动四步真数据 smoke，PGID `3061443`；只在最终输出
  层施加尺度/周期角共识，等待 loss/grad/checkpoint/语义四项验证。

## 2026-08-03 21:52 CST：0803_13 smoke 通过并正式启动

- 四步真数据 smoke loss `21.373/20.676/20.895/21.162`，grad
  `59.765/65.665/103.829/106.195`，全有限；364,505,716-byte checkpoint、711 个 state
  tensor 全有限，错误扫描与语义检查通过，GPU0 释放。
- fresh formal PGID `3062903`；iter50 `0.9711 s/iter`、loss `20.9881`、grad `102.8326`，
  9 个进程，GPU0 正常占用，错误扫描干净，HEAD/config/workdir 精确一致。
- 状态 `RUNNING`，先收 e4/e8/e12，并按 decoder 慢收敛原则保留更长轨迹，不以 e4/e8 直接否决。

## 2026-08-03 22:06 CST：0803_10 e12 成熟停止，0803_14 接替

- 0803_10 e12 cls HOTA/DetA/AssA `44.971/37.182/56.588`，det
  `52.008/46.294/60.418`；相对 Encoder 同点 `-4.709/-4.533`，相对原始 decoder 同点
  `-2.424/-2.428`。e4/e8/e12 成熟双负后精确停止 PGID `4053545`，成员 `23→0`。
- 381,037,558-byte e12 checkpoint、50 序列、28 CSV、108 个非空文件完整；GPU0/1 均
  `0%/1 MiB`，断点保留。
- 新增 0803_14 terminal-only log-area + periodic-angle：前三层逐帧独立，最终层只共享面积与
  周期角、保留逐帧纵横比。目标单测和 22,771,111 参数零增量构建通过；DDP smoke PGID
  `75663` 已启动。

## 2026-08-03 22:11 CST：0803_14 smoke 通过并正式启动

- DDP smoke 四步 loss `12.946/19.487/19.571/21.118`，grad
  `106.686/97.954/91.740/91.942`，全有限；364,502,454-byte checkpoint、711 个 state
  tensor 全有限，错误扫描与语义检查通过，进程退出后 GPU0/1 各 1 MiB。
- fresh formal PGID `77558`；iter50 `1.2555 s/iter`、loss `21.3640`、grad `102.0121`，
  7 个进程、双卡各 19,192 MiB，错误扫描、HEAD/config/workdir 五门槛通过。
- 状态 `RUNNING`，先收 e4/e8/e12；不以 e4/e8 直接否决。

## 2026-08-03 22:13 CST：0803_12 epoch 4 完整评估

- e4 cls HOTA/DetA/AssA `32.057/25.684/42.947`，det `38.097/32.294/45.679`；相对
  Encoder 同点 `-4.152/-0.656`。分类早期偏慢，det 接近但未越过。
- pair mAP/AP50 `0.1351/0.2517`、both-independent `0.1783/0.3199`；369,972,406-byte
  checkpoint、50 序列、28 CSV、108 个非空文件完整。
- 保持 PGID `4189798` 继续 e8/e12；不以 e4 直接否决渐进投影。

## 2026-08-03 22:29 CST：0803_11 epoch 4 完整评估

- e4 cls HOTA/DetA/AssA `31.540/25.303/42.259`，det `38.185/32.746/45.734`；相对
  Encoder 同点 `-4.669/-0.568`。分类早期偏慢，det 接近 Encoder。
- pair mAP/AP50 `0.1279/0.2455`、both-independent `0.1709/0.3172`；369,965,543-byte
  checkpoint、50 序列、28 CSV、108 个非空文件完整。197 TrackEval 用时 711.3 秒。
- 保持 PGID `53708` 继续 e8/e12；不以 e4 直接否决晚层投影。

## 2026-08-03 22:31 CST：0803_15 terminal-angle 已准备

- 新增只在最终输出层共享周期角的候选；中心、宽高与前三层 reference 全部逐帧独立，用于
  隔离 terminal-only 收益是否被尺度/面积共识抵消。
- 结构零参数、class-agnostic、无 reweight、无额外层或 attention；目标单测确认角度投影只
  调用 1 次，完整构建 22,771,111 参数、零增量、711 state tensors。
- 隔离仓库 `/data1/users/litianhao01/PairMOT_terminalangle_0803_15` 固定 `d181a98`；状态
  `PREPARED/WAITING_178_AFTER_0803_13`，不抢占当前 formal。

## 2026-08-04 05:19 CST：0803_18 epoch-4 完整评估

- cls HOTA/DetA/AssA `30.440/24.784/40.480`，det
  `38.288/33.547/44.847`；相对原始 decoder e4 HOTA `-3.866/-0.302`，相对 Encoder
  e4 为 `-5.769/-0.465`。与单独 terminal geometry 的 `0803_13` e4 相比为
  `-2.409/+0.969`，说明 shared semantic margins 在早期牺牲分类、略抬高 det。
- pair mAP/AP50 `0.125500/0.238364`，both-independent
  `0.169556/0.314817`；369,966,375-byte checkpoint、5416 条检测、50 序列、28 CSV、
  108 个评测文件和 `async_done=1` 均完整。
- e4 只登记为慢收敛信号，不作直接否决。197 当前两卡继续同一 formal 到 e8/e12，e8
  监控使用 `epoch_8.pth`、`val_det/epoch_07` 与 `val_track_0002`；资源序号仍仅为当前分配，
  只有 252 固定 GPU0/1。

## 2026-08-04 05:41 CST：0803_13 e24 成熟迁移、0803_17 e8 与 0803_23 启动

- 0803_13 e24 cls HOTA/DetA/AssA `52.841/43.966/65.407`，det
  `59.322/52.304/69.604`。相对原始 decoder e24 `51.709/58.781` 为
  `+1.132/+0.541`，联合优势 `+1.673`；相对 Encoder e24 `51.714/59.519` 为
  `+1.127/-0.197`。它已通过成熟迁移门槛，但尚未满足最终双超 Encoder 与最终联合增益门槛。
- e24 pair mAP/AP50 `0.295953/0.516788`，both-independent
  `0.340356/0.563297`，均高于 e20；397,659,956-byte checkpoint、5416 条检测、50 序列、
  28 CSV、108 文件与异步完成标志完整。178 原 PGID `3062903` 精确停止，成员 `9→0`，
  GPU0 连续三次 `1 MiB/0%`。
- 252 与 178 同名账号 UID 不同，旧 workdir 在 252 只读；共享文件系统又不支持 ACL。未放宽
  旧目录权限，而是保留原 e24 checkpoint 为只读恢复源，在
  `/data4/litianhao/PairMmot/workdir_252/0803_13_terminal_log_size_periodic_angle_resume252_from_epoch24`
  创建 252 自有续跑目录。启动器修复提交在 252 为 `a414cc1`；明确恢复到 epoch24/iter24912，
  e25 iter50 `1.1754 s/iter`、loss `9.0711`、grad `53.3604`，DN/encoder 全有限，固定
  GPU0/1 各约 19.4 GiB，PGID `419164` 五门槛通过。
- 0803_17 e8 cls HOTA/DetA/AssA `39.478/33.917/48.692`，det
  `46.483/42.353/52.855`；相对原始 decoder e8 `-2.494/-1.695`。pair mAP/AP50
  `0.199659/0.363509`，both-independent `0.244129/0.426328`；checkpoint、50 序列、
  28 CSV、108 文件完整。继续 e12，不按 e8 停止。
- 0803_23 transported full-tangent 在 178 GPU0 的四步 smoke loss
  `21.3727/20.6123/20.9003/21.2177`，grad
  `60.6826/96.6635/127.6287/129.1387`；364,506,420-byte checkpoint 的 642 个浮点
  tensor 全有限，iterative-cls/DN 语义通过。fresh formal PGID `3144617` 在 iter50 为
  `0.9593 s/iter`、loss `21.0341`、grad `100.6837`，总、DN、encoder 全有限；零参数、
  class-agnostic、无 reweight/额外层/attention/loss，继续 e4/e8/e12。

## 2026-08-04 05:55 CST：0803_23 数值审计与 finite-fresh 重启

- 首次 formal 在 epoch1 iter350 出现 1 次 `assign_gd_curr` 非有限匹配代价保护，而 0803_13
  父轨迹 24 epochs 同类警告为 0。退化小框构造进一步证明：当 reference 尺寸约 `1e-4` 时，
  旧的 `reference_size * exp(transported_scale_tangent)` 前向虽会被 clamp 成有限值，反向梯度
  却会因 `inf*0` 全部成为 NaN。
- 该问题属于实现数值无效，不是 e4/e8 性能否决。旧 PGID `3144617` 在 epoch1 iter650 精确
  停止，成员 `9→0`，无正式 epoch checkpoint；GPU0 连续三次 `1 MiB/0%`。旧 workdir 仅保留
  审计，不纳入性能表。
- 提交 `6072e76`（178 隔离仓库 `e2b399b2`）把尺度目标改为 log-domain 先 clamp 再 exp；
  数学目标与原最终尺寸 clamp 等价，不增加参数、层、attention、loss 或显著计算。三项定向测试
  全通过，包括旧实现可复现 NaN 的极小 reference 有限前向/反向用例；整模仍为
  `22,771,111` 参数、零增量、711 tensors。
- 修复版四步真实 smoke loss `21.3727/20.6766/20.9629/21.2487`，grad
  `60.7448/68.9937/72.3075/108.7309`；364,506,484-byte checkpoint 的 642 个浮点 tensor
  全有限，无非有限匹配代价。新的 `_finite_fresh` formal PGID `3151184` 在 iter50 为
  `0.9841 s/iter`、loss `21.0123`、grad `105.5777`，总、DN、encoder 有限且同类警告为 0；
  五项启动门槛重新通过。

## 2026-08-04 06:05 CST：0803_23 修复版达到 iter700 覆盖

- finite-fresh 正式轨迹到 epoch1 iter700，已经跨过旧实现首次告警 iter350 并达到两倍训练覆盖；
  非有限匹配代价告警 `0`、致命错误 `0`。iter700 loss `16.8861`、grad `229.6270`，总、DN、
  encoder proposal 所有记录分量均有限。
- 数值修复获得正式数据长于 smoke 的直接证据，PGID `3151184` 继续 e4/e8/e12。该结论只确认
  实现有效性，尚无权替代完整 HOTA/AP 评测或最终目标审计。

## 2026-08-04 06:20 CST：0803_24 transported shape tangent 已准备

- 新候选只在最终 normal queries 的 log-size/周期角三维切空间保留 pair-common 更新，并把 detail
  投影到 detached 前序相对尺寸/角度变换；中心 residual 完全逐帧保留。它是 0803_13 的保守传输
  版本，也是 0803_23 去掉中心传输与跨维投影后的正交对照。
- 4 项定向测试覆盖终层单次调用、中心逐元素保持、既有 shape detail 投影、交换等变、DN 前缀和
  极小 reference 有限梯度；配置深拷贝、两份 launcher 语法和整模构建通过。参数
  `22,771,111`、增量 0、711 tensors，无 class-aware、reweight、新层、attention 或 loss。
- 178 隔离仓库 `/data1/users/litianhao01/PairMOT_terminaltransportshape_0803_24` 固定 clean HEAD
  `d470f96e`。状态 `PREPARED/NO_GPU`，未创建 smoke/formal workdir，不抢占 0803_23；是否启动
  等待现有 e12/e28 完整闭环。

## 2026-08-04 06:53 CST：0803_17 e12 成熟停止，0803_21 formal 运行

- 0803_17 e12 cls HOTA/DetA/AssA `45.597/38.179/56.731`，det
  `52.020/47.161/59.329`。相对原始 decoder 同点 HOTA `-1.798/-2.416`，相对 Encoder
  `-4.083/-4.521`；pair mAP/AP50 `0.238097/0.417747`，both-independent
  `0.282968/0.475183`。
- 381,044,662-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空文件及异步完成
  标志完整。e4/e8/e12 持续双负后停止 PGID `1357909`，成员归零，GPU1/2 回到 10 MiB；GPU0
  外部任务不变。这是三节点成熟停止，不是 e4/e8 直接否决。
- 0803_21 四步真实 DDP smoke loss `12.9371/19.5029/19.6174/21.1605`，grad
  `102.8686/100.6335/94.7345/94.9512`；总、DN、encoder 全有限，364,502,646-byte
  checkpoint 的 642 个浮点 tensor 全有限，iterative-cls/DN 语义检查通过。
- 连续空闲检查后 fresh formal screen `1384942.pm_0803_21_formal_99`、PGID `1384944`；iter50
  `0.9814 s/iter`、loss `21.3915`、grad `104.8633`，GPU1/2 各约 19.2 GiB，错误扫描为空，
  五门槛通过。状态 `RUNNING`，继续 e4/e8/e12。

## 2026-08-04 07:09 CST：0803_13 epoch-28 完整评估

- e28 cls HOTA/DetA/AssA `53.114/44.428/65.313`，det
  `59.729/52.623/70.150`。相对原始 decoder e28 `52.177/59.280` 为
  `+0.937/+0.449`，联合优势 `+1.386`；相对 Encoder e28 `51.740/59.830` 为
  `+1.374/-0.101`。结构仍在同点双正地改善原始 decoder，但尚未双超 Encoder，也未达到最终
  `>118.330` 门槛。
- pair mAP/AP50 `0.3017/0.5230`，both-independent mAP/AP50
  `0.3449/0.5666`；403,126,774-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个
  非空评测文件和 `async_done=1` 完整。
- 该轨迹在 e24/e28 都保持相对原始 decoder 双正，且 det 与 Encoder 只差 `0.101`，因此不在
  e28 停止。252 固定 GPU0/1、PGID `419164` 继续到 e32；252 仍只承担这一条成熟路线。

## 2026-08-04 07:20 CST：0803_23 epoch-4 完整评估

- 修复后的 transported full-tangent e4 cls HOTA/DetA/AssA `36.342/30.409/45.421`，det
  `44.739/38.109/54.712`。相对原始 decoder e4 `34.306/38.590` 为
  `+2.036/+6.149`，相对 Encoder e4 `36.209/38.753` 也为 `+0.133/+5.986`；相对
  terminal mean geometry `0803_13` e4 `32.849/37.319` 为 `+3.493/+7.420`。
- pair mAP/AP50 `0.1649/0.3108`，both-independent mAP/AP50 `0.2139/0.3859`；
  369,970,164-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件和
  `async_done=1` 完整。修复后的正式轨迹到 e5 仍无非有限匹配告警。
- 这是当前最强的早期结构信号，但 e4 不作为最终通过依据。178 当前 GPU0、PGID `3151184`
  保持同一 fresh 轨迹继续 e8/e12；GPU 序号只是动态分配，178 只固定总计 1 卡。

## 2026-08-04 07:29 CST：0803_24 补齐 197 动态双卡后继

- 0803_23 e4 表明 transported tangent 明显优于 terminal mean geometry。为在 0803_18 成熟后并行
  分离“中心传输”和“形状传输”的贡献，给既有 `0803_24 transported shape tangent` 补齐 197
  的 2xb4 formal/smoke 配置和安全启动器；它不含语义 margins，因此也能隔离 0803_18 的早期
  分类损失来源。
- 197 隔离仓库 `/data/users/litianhao/PairMOT_terminaltransportshape_0803_24_197` 固定 clean
  HEAD `44395ea`。完整构建为 `22,771,111` 参数、增量 0、711 状态张量；配置为 batch 4、
  72 epochs，formal/smoke 启动器语法通过。
- 两个启动器要求显式传入当时两张空闲 GPU，未写死 197 序号。状态为 `PREPARED/NO_GPU`，
  未创建 smoke/formal workdir，不抢占当前 `0803_18`；是否启动等待 e12 与 0803_23 e8 闭环。

## 2026-08-04 07:32 CST：0803_18 epoch-8 完整评估

- e8 cls HOTA/DetA/AssA `42.014/34.349/54.039`，det
  `47.865/43.079/55.209`。相对原始 decoder e8 `41.972/48.178` 为
  `+0.042/-0.313`，相对 Encoder e8 `45.269/50.193` 为 `-3.255/-2.328`；相对
  单独 terminal geometry `0803_13` e8 `45.002/49.083` 为 `-2.988/-1.218`。
- pair mAP/AP50 `0.199896/0.368410`，both-independent mAP/AP50
  `0.244107/0.431063`；375,527,527-byte checkpoint、5416 条检测、50 序列、28 CSV、
  108 个非空评测文件和 `async_done=1` 完整。
- shared semantic margin 的早期分类损失从 e4 已明显恢复，但 e8 仍未形成可靠增益。按 decoder
  晚收敛约束保持 PGID `387859` 继续 e12，不用 e8 直接否决；`0803_24` 保持 PREPARED，
  不并发抢占 197 当前两卡。

## 2026-08-04 07:44 CST：0803_25 transported center tangent 已准备

- 0803_23 full tangent e4 的 det 大幅提升可能来自中心运动、形状运动或二者跨维投影。新增
  center-only 正交候选：仅在最后 normal-query 层把局部中心更新的 pair detail 投影到 detached
  前序相对平移方向；宽高、角度、分类、DN 与递归 reference 都保持逐帧原样。
- 该投影零参数、交换等变、class-agnostic，无 reweight、新层、attention 或 loss。三项定向
  测试覆盖终层单次调用与互斥、形状 residual 精确保留及既有平移 detail、交换等变/DN 保留/
  有限反向，全部通过；完整构建为 `22,771,111` 参数、增量 0、711 状态张量。
- 178 隔离仓库 `/data1/users/litianhao01/PairMOT_terminaltransportcenter_0803_25` 固定 clean
  HEAD `09a0d2f`；1xb8、72 epochs 配置与 formal/smoke 启动器通过。启动器要求显式传入当时
  一张空闲 GPU，不固定序号。状态 `PREPARED/NO_GPU`，排在正在运行的 0803_23 与 shape-only
  0803_24 之后。

## 2026-08-04 08:06 CST：0803_21 epoch-4 完整评估

- transported semantic margin e4 cls HOTA/DetA/AssA `30.158/23.503/41.345`，det
  `37.094/29.928/47.230`。相对原始 decoder e4 `34.306/38.590` 为
  `-4.148/-1.496`，相对 Encoder e4 `36.209/38.753` 为 `-6.051/-1.659`；相对
  shared-margin `0803_18` e4 `30.440/38.288` 也为 `-0.282/-1.194`。
- pair mAP/AP50 `0.121050/0.227453`，both-independent mAP/AP50
  `0.160944/0.293398`；369,968,054-byte checkpoint、5416 条检测、50 序列、28 CSV、
  108 个非空评测文件和 `async_done=1` 完整。
- e4 不作为直接否决，99 当前 GPU1/2、PGID `1384944` 继续 e8/e12；但在它取得成熟正增益
  之前，不把 semantic margin 与 0803_23 强几何分支组合，避免无证据扩散。

## 2026-08-04 08:29 CST：0803_23 epoch-8 保持大幅双正

- transported full-tangent e8 cls HOTA/DetA/AssA `46.283/39.798/56.109`，det
  `53.755/47.403/63.617`。相对原始 decoder e8 `41.972/48.178` 为
  `+4.311/+5.577`，合计 `+9.888`；相对 Encoder e8 `45.269/50.193` 也为
  `+1.014/+3.562`。e4 的强 det 信号没有消失，且 e8 分类也转为明确领先。
- pair mAP/AP50 `0.252926/0.454211`，both-independent mAP/AP50
  `0.301879/0.515712`，四项均高于 terminal mean geometry `0803_13` e8；375,559,796-byte
  checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件和 `async_done=1` 完整。
- `0803_23` 升级为当前第一优先主线，178 当前 GPU0、PGID `3151184` 继续 e12 以满足成熟
  轨迹约束；不在 e8 宣布最终达标。`0803_24 shape-only` 与 `0803_25 center-only` 保持
  PREPARED/NO_GPU，作为归因和后续风险备份。

## 2026-08-04 08:36 CST：0803_13 epoch-32 成熟双超同点 Encoder

- terminal mean geometry e32 cls HOTA/DetA/AssA `53.642/44.778/65.968`，det
  `60.531/53.028/71.463`。相对原始 decoder e32 `52.566/59.955` 为
  `+1.076/+0.576`，联合优势 `+1.652`；相对 Encoder e32 `52.354/60.330` 为
  `+1.288/+0.201`，首次在成熟 e32 同时越过两项同点 Encoder。
- pair mAP/AP50 `0.307277/0.528711`，both-independent mAP/AP50
  `0.349227/0.570345`；408,623,606-byte checkpoint、5416 条检测、50 序列、28 CSV、
  108 个非空评测文件和 `async_done=1` 完整。
- 当前绝对值之和 `114.173`，仍低于最终严格门槛 `>118.330`，且 det 绝对值尚未超过
  Encoder 最终 `62.393`，不得误报完成。252 固定 GPU0/1、PGID `419164` 保持成熟路线
  继续 e36；252 仍不承接新结构筛选。

## 2026-08-04 08:40 CST：0803_25 补齐 99 动态双卡后继

- 为在不占用 178 第一主线的前提下并行归因，给 `0803_25 center-only transported tangent`
  补齐 99 的 2xb4 formal/smoke 配置。它与 178 版保持同一结构和全局 batch 8，只改变服务器
  适配；启动器要求显式传入当时两张空闲 GPU，不固定 99 序号。
- 99 隔离仓库 `/data/users/wangying01/lth/PairMOT_terminaltransportcenter_0803_25_99` 固定
  clean HEAD `345a59c`；完整构建为 `22,771,111` 参数、增量 0、711 状态张量，配置为
  batch 4、72 epochs，两个启动器语法通过。
- 首次从完整 bundle 克隆时仅因非必要 83MB LFS 演示 GIF 下载 EOF 中断；失败目录改名保留在
  `PairMOT_terminaltransportcenter_0803_25_99_failed_lfs_0838` 供审计，随后以
  `GIT_LFS_SKIP_SMUDGE=1` 重建成功。状态 `PREPARED/NO_GPU`，不抢占当前 0803_21。

## 2026-08-04 09:19 CST：0803_21 epoch-8 完整评估

- transported semantic margin e8 cls HOTA/DetA/AssA `38.854/32.835/48.082`，det
  `46.716/42.206/53.560`。相对原始 decoder e8 `41.972/48.178` 为
  `-3.118/-1.462`，相对 Encoder e8 `45.269/50.193` 为 `-6.415/-3.477`；虽然相对
  e4 已恢复，但成熟前仍未形成结构增益。
- pair mAP/AP50 `0.191545/0.342944`，both-independent mAP/AP50
  `0.236078/0.406699`；375,537,014-byte checkpoint、5416 条检测、50 序列、28 CSV、
  108 个非空评测文件和 `async_done=1` 完整。
- 不用 e8 直接否决，99 当前 GPU1/2、PGID `1384944` 继续 e12；语义与 strong geometry 的
  组合仍冻结。`0803_25 center-only` 保持 PREPARED/NO_GPU，等待三节点成熟判断后交接。

## 2026-08-04 09:43 CST：0803_18 epoch-12 成熟停止

- geometry + semantic margins e12 cls HOTA/DetA/AssA `45.404/37.264/57.553`，det
  `51.784/46.413/59.898`。相对原始 decoder e12 `47.395/54.436` 为
  `-1.991/-2.652`，相对 Encoder e12 `49.680/56.541` 为 `-4.276/-4.757`；相对单独
  terminal geometry `0803_13` e12 `48.289/54.539` 也为 `-2.885/-2.755`。
- pair mAP/AP50 `0.2298/0.4115`，both-independent mAP/AP50 `0.2732/0.4675`；
  381,027,687-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件和
  `async_done=1` 完整。
- e4/e8/e12 三个完整节点形成成熟双负轨迹，不属于 e4/e8 早停。核验产物后精确 TERM PGID
  `387859`，成员 `23→0`；checkpoint 和全部评测产物保留，释放的两卡交给正交 shape-only
  候选，不再扩展 semantic-margin 组合。

## 2026-08-04 09:43 CST：0803_23 epoch-12 保持强联合增益

- transported full-tangent e12 cls HOTA/DetA/AssA `50.145/41.895/62.293`，det
  `56.375/49.620/66.403`。相对原始 decoder e12 `47.395/54.436` 为
  `+2.750/+1.939`，联合 `+4.689`；相对 Encoder e12 `49.680/56.541` 为
  `+0.465/-0.166`，只剩 det `0.166` 的同点缺口。
- pair mAP/AP50 `0.2745/0.4824`，both-independent mAP/AP50 `0.3205/0.5354`；
  381,087,476-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件和
  `async_done=1` 完整。
- e8 的同点双正收窄为 e12 的 cls 正、det 微负，但对原 decoder 的大幅双正仍保持，且检测
  只差 Encoder `0.166`。178 当前 GPU0、PGID `3151184` 不停训，继续 e16；GPU0 仅为当前
  选择，不改变 178 单卡、序号不固定的资源规则。

## 2026-08-04 09:49 CST：197 动态 GPU2/3 启动 0803_24

- `0803_18` 停止后对全机两次连续核验：GPU0/1 仍由外部 PID `8290/8291` 各占约 16.2 GiB，
  GPU2/3 与 GPU4/5 均为空。按动态选择规则使用连续为 `1 MiB/0%` 的 GPU2/3，不写死 197
  序号，也不触碰 GPU0/1 外部任务。
- 双卡真数据 smoke 四步 loss `12.9370/19.4581/19.5799/21.1770`，grad
  `103.4576/104.3428/97.7304/97.0236`；DN/encoder 全有限，364,503,015-byte checkpoint
  的 642 个浮点 tensor 全有限，iterative-cls/DN 语义通过，错误扫描为空。
- fresh formal 固定 clean HEAD `44395ea`；screen `712275.pm_0803_24_formal_197`、PGID
  `712277`。iter50 为 `1.8555 s/iter`、loss `21.4056`、grad `136.6876`，GPU2/3 各约
  19.2 GiB，总、DN、encoder proposal 均有限，无致命错误，五门槛通过。状态 `RUNNING`，
  先收 e4/e8/e12，不用 e4/e8 直接否决。

## 2026-08-04 10:02 CST：0803_13 epoch-36 保持同点双超

- terminal mean geometry e36 cls HOTA/DetA/AssA `53.874/44.868/66.433`，det
  `60.860/53.299/71.836`。相对原始 decoder e36 `52.985/60.410` 为
  `+0.889/+0.450`，联合 `+1.339`；相对 Encoder e36 `52.912/60.707` 为
  `+0.962/+0.153`，联合 `+1.115`，继续保持成熟同点双正。
- pair mAP/AP50 `0.3098/0.5315`，both-independent mAP/AP50 `0.3520/0.5725`；
  414,119,542-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件和
  `async_done=1` 完整。
- e36 较 e32 绝对 HOTA 又提高 `+0.232/+0.329`，但当前和 `114.734` 仍低于 Encoder 最终和
  加 1.5 的严格门槛 `>118.330`，且距最终 cls/det 门槛仍为 `0.563/1.533`。252 固定
  GPU0/1、PGID `419164` 继续 e40；GPU2/3 保持空闲，252 不承担新结构筛选。

## 2026-08-04 10:11 CST：0803_26 product-tangent 分块传输已准备

- `0803_23` 的单个 5D 投影允许中心平移能量通过同一内积旋转到 log-size/angle detail，e12
  的 det 同点优势收窄到 `-0.166`。`0803_26` 把 terminal product geometry 分成独立的 center
  2D 与 shape 3D tangent bundle：先沿既有平移方向投影中心 detail，再沿既有尺度/角度方向
  投影 shape detail，不允许两组坐标跨维混合。
- 结构只顺序调用已有 center/shape 投影各一次，零参数、交换等变、class-agnostic，无
  reweight、新 layer、attention 或 loss，DN prefix 完全保留。定向测试覆盖末层各调用一次、
  与显式分块组合一致、swap/DN/有限反向和互斥配置，真实 py310 环境通过。
- 99 正式与 smoke 配置加载、`deepcopy` 和远端 Bash 语法通过；完整父/候选模型均为
  `22,771,111` 参数、711 状态张量，增量 0。隔离 checkout
  `/data/users/wangying01/lth/PairMOT_terminaltransportproduct_0803_26_99` 固定 clean HEAD
  `89ec85a`。首次 checkout 仅因非必要 83 MB LFS GIF 缺失中断，失败工作树保留为
  `PairMOT_terminaltransportproduct_0803_26_99_failed_lfs_1008`，随后用
  `GIT_LFS_SKIP_SMUDGE=1` 重建成功。状态 `PREPARED/NO_GPU`，排在 `0803_25` 后且不占卡。

## 2026-08-04 10:33 CST：0803_21 epoch-12 成熟双负

- transported semantic margin e12 cls HOTA/DetA/AssA `44.179/36.075/56.950`，det
  `52.106/46.623/60.266`。相对原始 decoder e12 `47.395/54.436` 为
  `-3.216/-2.330`，相对 Encoder e12 `49.680/56.541` 为 `-5.501/-4.435`；相对 e8 虽继续
  恢复，但未形成结构增益。
- pair mAP/AP50 `0.2260/0.3924`，both-independent mAP/AP50 `0.2668/0.4433`；
  epoch12 checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件和
  `async_done=1` 完整。
- e4/e8/e12 三个完整节点均被原始 decoder 与 Encoder 双侧支配，已满足成熟淘汰证据，不是
  e4/e8 直接否决。精确 TERM PGID `1384944` 后成员 `23→0`；checkpoint 与全部评测产物保留。

## 2026-08-04 10:39 CST：99 动态双卡启动 0803_25

- 0803_21 停止后 GPU0 外部 PID `1439554` 仍占约 5.9 GiB 且未触碰；GPU1/2 连续两次为
  `10 MiB/0%`。启动器按实时空闲集合动态选择 GPU1/2，99 配置与代码没有固定序号。
- 四步真数据 DDP smoke loss `12.9442/19.6043/19.6451/21.2550`，grad
  `102.4960/169.0954/141.0412/133.9741`；DN/encoder 全有限，iter_4 checkpoint 的 642 个
  浮点 tensor 全有限，iterative-cls/DN 语义通过，错误扫描为空。
- 再次连续空闲检查后 fresh formal 固定 clean HEAD `345a59c`；screen
  `1442843.pm_0803_25_formal_99`、PGID `1442845`。iter50 为 `0.9843 s/iter`、loss
  `21.4116`、grad `114.8470`，总、DN、encoder proposal 全有限，GPU1/2 各约 19.2 GiB，
  致命错误为 0，五门槛通过。状态 `RUNNING/TO_E4+`，按慢收敛规则收集 e4/e8/e12。

## 2026-08-04 10:58 CST：0803_23 epoch-16 优势回落

- transported full-tangent e16 cls HOTA/DetA/AssA `49.627/41.278/62.277`，det
  `56.820/50.118/66.772`。相对原始 decoder e16 `50.036/56.933` 为
  `-0.409/-0.113`，相对 Encoder e16 `51.091/58.320` 为 `-1.464/-1.500`；e12 相对原
  decoder 的 `+2.750/+1.939` 已转为轻微双负。
- pair mAP/AP50 `0.2716/0.4735`，both-independent mAP/AP50 `0.3148/0.5216`；
  386,614,516-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件和
  `async_done=1` 完整。
- e4/e8/e12 连续强双正后，e16 是首次相对原 decoder 轻微双负；原 decoder 本身存在后期回升，
  且同一作业已在异步评测期间进入 e17，因此单个 e16 不作为成熟否决。178 当前 GPU0、PGID
  `3151184` 保持到 e20，检验 5D tangent 是暂时谷值还是优势消失；GPU0 仍只是动态选择。

## 2026-08-04 11:26 CST：0803_13 epoch-40 继续到 epoch-44

- terminal log-size + periodic-angle e40 cls HOTA/DetA/AssA `54.057/44.872/66.828`，det
  `61.250/53.718/72.186`。相对原始 decoder e40 `54.059/61.102` 为
  `-0.002/+0.148`；相对 Encoder e40 `53.797/61.063` 为 `+0.260/+0.187`，同点仍双超，
  但联合优势只有 `+0.447`。
- pair mAP/AP50 `0.3138/0.5373`，both-independent mAP/AP50 `0.3553/0.5768`；
  419,607,222-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件和
  `async_done=1` 完整。
- e36→e40 绝对 HOTA 继续提高 `+0.183/+0.390`，未形成平台；但绝对和 `115.307` 仍低于
  严格 `>118.330`，距最终 cls/det 门槛 `54.437/62.393` 尚差 `0.380/1.143`。252 固定
  GPU0/1、PGID `419164` 已自然进入 e41，继续到 e44；GPU2/3 保持空闲，不在最慢资源上并发
  新筛选结构。

## 2026-08-04 11:54 CST：0803_25 center-only epoch-4 早期负向

- center-only transported tangent e4 cls HOTA/DetA/AssA `31.262/25.476/41.269`，det
  `37.687/32.938/44.208`。相对原始 decoder e4 `34.306/38.590` 为
  `-3.044/-0.903`，相对 full-tangent e4 `36.342/44.739` 为 `-5.080/-7.052`；中心分量
  单独使用尚未解释 full-tangent 的早期强增益。
- pair mAP/AP50 `0.1330/0.2521`，both-independent mAP/AP50 `0.1742/0.3207`；
  369,968,758-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件及
  `async_done=1` 完整。
- e4 只作结构归因，不是成熟否决。99 当前动态 GPU1/2、PGID `1442845` 已进入 e5，继续
  e8/e12；GPU0 外部任务不动。product-tangent `0803_26` 保持 PREPARED，不因单个 e4 抢占。

## 2026-08-04 12:11 CST：0803_24 shape-only epoch-4 早期负向

- shape-only transported tangent e4 cls HOTA/DetA/AssA `31.487/25.126/42.524`，det
  `37.808/32.338/45.300`。相对原始 decoder e4 `34.306/38.590` 为
  `-2.819/-0.782`，相对 full-tangent e4 `36.342/44.739` 为 `-4.855/-6.931`；相对
  center-only e4 仅高 `+0.225/+0.121`，单独 shape 同样没有解释 full-tangent 早期增益。
- pair mAP/AP50 `0.1277/0.2485`，both-independent mAP/AP50 `0.1699/0.3198`；
  369,969,127-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件及
  `async_done=1` 完整。
- e4 仍只作分量归因。197 动态 GPU2/3、PGID `712277` 继续 e8/e12，GPU0/1 外部任务不动；
  center/shape 两项成熟结果到齐前不把早期联合效应直接外推为 product-tangent 成功。

## 2026-08-04 12:13 CST：0803_23 full-tangent epoch-20 明显回升

- full transported tangent e20 cls HOTA/DetA/AssA `51.119/42.702/63.455`，det
  `57.969/51.405/67.704`。相对原始 decoder e20 `50.843/58.033` 为
  `+0.276/-0.064`；相对 Encoder e20 `51.514/58.922` 为 `-0.395/-0.953`。e16→e20
  绝对 HOTA 回升 `+1.492/+1.149`，同点差距也显著收窄。
- pair mAP/AP50 `0.2864/0.4981`，both-independent mAP/AP50 `0.3305/0.5452`；
  392,145,588-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件和
  `async_done=1` 完整。
- e20 尚未恢复 e12 的大幅联合优势，也远未达到最终绝对门槛；但 e16 单点谷值已出现明确反弹，
  不能成熟否决。178 当前动态 GPU0、PGID `3151184` 继续 e24，检验恢复是否持续；不为
  product-tangent 提前释放单卡。

## 2026-08-04 12:55 CST：0803_13 terminal geometry epoch-44 继续上升但未达目标

- e44 cls HOTA/DetA/AssA `54.381/44.931/67.539`，det
  `61.716/54.021/72.880`。相对 e40 `54.057/61.250` 回升 `+0.324/+0.466`；相对原始
  decoder e44 `54.415/61.737` 为 `-0.034/-0.021`，当前只是基本持平，未形成目标所需优势。
- 绝对和为 `116.097`，距严格 `>118.330` 尚差 `2.233`；距最终 cls/det 门槛
  `54.437/62.393` 分别差 `0.056/0.677`。因此该点不登记为成功，也不因未过最终门槛直接
  否决仍在上升的成熟曲线。
- pair mAP/AP50 `0.3178/0.5406`，both-independent mAP/AP50 `0.3591/0.5797`；
  425,094,518-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件及
  `async_done=1` 完整。
- 252 固定 GPU0/1 的 PGID `419164` 已继续到 epoch45，后续检查 e48；GPU2/3 不用于本任务。

## 2026-08-04 13:08 CST：0803_25 center-only epoch-8 继续到 epoch-12

- center-only e8 cls HOTA/DetA/AssA `41.359/34.708/51.115`，det
  `46.931/42.204/54.111`。e4→e8 回升 `+10.097/+9.244`，确认正在收敛；但相对原始 decoder
  e8 `41.972/48.178` 为 `-0.613/-1.247`，相对 full-tangent e8 `46.283/53.755` 为
  `-4.924/-6.824`，中心分量单独使用仍未复现联合切空间的早期增益。
- pair mAP/AP50 `0.2052/0.3754`，both-independent mAP/AP50 `0.2492/0.4369`；
  375,530,550-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件和
  `async_done=1` 完整。
- e8 不作直接否决。99 当前动态 GPU1/2、PGID `1442845` 继续 e12，GPU0 外部任务不动；
  `0803_26` 保持 PREPARED/NO_GPU，等待 center/shape 的成熟归因再决定接替。

## 2026-08-04 13:30 CST：0803_23 full-tangent epoch-24 恢复持续

- full transported tangent e24 cls HOTA/DetA/AssA `52.012/43.460/64.440`，det
  `58.551/51.647/68.767`。e20→e24 继续回升 `+0.893/+0.582`；相对原始 decoder e24
  `51.709/58.781` 为 `+0.303/-0.230`，相对 Encoder e24 `51.714/59.519` 为
  `+0.298/-0.968`。分类侧已双超同点参考，det 侧仍是主要瓶颈。
- pair mAP/AP50 `0.2937/0.5083`，both-independent mAP/AP50 `0.3373/0.5536`；
  397,671,156-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件和
  `async_done=1` 完整。
- e16→e20→e24 连续恢复，不能在 e24 停止成熟强线。178 当前动态 GPU0、PGID `3151184`
  已进入 e25，继续到 e28 检查 det 差距是否进一步收窄；GPU1 外部任务不动。

## 2026-08-04 14:19 CST：0803_27 position-tangent + product geometry 已准备

- 现有终层几何投影主要改善收敛速度，成熟期增益逐渐收窄；而直接共享分类 residual 或引入
  语义 margin 的历史候选会损伤 DetA 或 cls AssA。`0803_27` 因此只在最后一层把两帧既有
  cross-attention evidence 的交换奇差分，正交投影到 reference-point MLP 已编码的 detached
  位置位移方向；保留运动对齐的分类细节，去除横向外观噪声。回归侧独立执行 `0803_26` 的
  center 2D 与 shape 3D product-tangent 投影，递归 query、辅助层输出与 DN prefix 不变。
- 该结构零新增参数、状态、层、attention 与 loss；投影能量不增、交换等变、class-agnostic，
  不使用 reweight。远端真实 py310 环境新增两项定向测试全部通过，覆盖交换等变、DN 清零、
  正交余量、有限梯度、position detach、终层单次调用和互斥配置。完整 157 项 decoder 套件仅有
  1 项既有顺序相关 midpoint 测试失败，未修改的 `0803_26` 基线复现同一失败；该用例在父/新
  工作树单独运行均通过，因此不是新候选回归。
- 99 正式/四步 smoke 配置深拷贝、两份 launcher Bash 语法和专用整模构建审计通过：
  `22,771,111` 参数、增量 0、711 状态张量。隔离工作树
  `/data/users/wangying01/lth/PairMOT_positiontangent_0803_27_99` 固定 clean HEAD `aea3157`，未创建 smoke/formal workdir，
  状态 `PREPARED/NO_GPU`。启动器要求显式传入当时两张空闲 GPU，不固定 99 序号；等待
  `0803_25` e12 和 shape-only 成熟证据后再与 `0803_26` 排序，不抢占当前训练。

## 2026-08-04 14:23 CST：0803_13 terminal geometry epoch-48 分类过线但检测回落

- e48 cls HOTA/DetA/AssA `54.533/45.002/67.846`，det
  `61.587/53.995/72.635`。相对 e44 为 `+0.152/-0.129`，相对原始 decoder e48
  `54.609/62.091` 为 `-0.076/-0.504`；分类首次严格超过最终 Encoder `54.437`，但检测仍低于
  `62.393` 达 `0.806`，同一 checkpoint 未满足双过线。
- 绝对和 `116.120`，距严格 `>118.330` 尚差 `2.210`。pair mAP/AP50
  `0.3183/0.5399`，both-independent mAP/AP50 `0.3590/0.5779`；430,579,894-byte checkpoint、
  5416 条检测、50 序列、28 CSV、108 个非空评测文件和 `async_done=1` 完整。
- 该线 e44→e48 的 det 已回落，但原始 decoder 的检测峰值出现在 e52–e56，且用户明确要求不能
  因 decoder 收敛较慢而过早否决。252 因此仍只固定 GPU0/1、PGID `419164` 继续到 e52；
  GPU2/3 不用于本任务，新结构仍不在最慢资源上筛选。

## 2026-08-04 14:26 CST：0803_24 shape-only epoch-8 接近原 decoder

- shape-only e8 cls HOTA/DetA/AssA `41.910/34.476/53.595`，det
  `47.783/43.107/54.921`。相对原始 decoder e8 `41.972/48.178` 为 `-0.062/-0.395`，较
  center-only e8 `41.359/46.931` 高 `+0.551/+0.852`；shape 分量单独使用比 center 分量稳定。
- 它仍比 full-tangent e8 `46.283/53.755` 低 `4.373/5.972`，说明 full-tangent 的早期大幅增益
  不是两个独立分量的简单外推。pair mAP/AP50 `0.2046/0.3732`，both-independent mAP/AP50
  `0.2476/0.4338`；epoch8 checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件和
  `async_done=1` 完整。
- e8 不直接否决。197 当前动态 GPU2/3、PGID `712277` 已继续 e9→e12，GPU0/1 外部任务不动；
  e12 完整后再决定 197 接替纯 product-tangent `0803_26`，避免在分量未成熟时提前外推。

## 2026-08-04 14:33 CST：0803_25 center-only e12 成熟停止，0803_27 接替 99

- center-only e12 cls HOTA/DetA/AssA `46.196/38.381/57.929`，det
  `51.938/46.335/60.487`。相对原始 decoder e12 `47.395/54.436` 为 `-1.199/-2.498`，且
  e4/e8/e12 三个完整节点持续双负；该判断不是用 e4/e8 直接否决。pair mAP/AP50
  `0.2338/0.4199`，both-independent mAP/AP50 `0.2798/0.4809`；381,025,142-byte checkpoint、
  5416 条检测、50 序列、28 CSV、108 个非空评测文件和 `async_done=1` 完整。
- 精确 TERM PGID `1442845` 后成员 `23→0`。99 三张物理卡两次检查均仅 10 MiB，第二次利用率
  `0/1/0%`；遵守 99 本任务最多使用两卡的边界，启动时动态选择 GPU0/1，并未在代码或配置中
  固定序号。
- `0803_27` 真数据四步 DDP smoke loss `12.9372/19.4915/19.6438/21.2056`，grad
  `102.8284/107.8695/112.6019/123.3048`；DN/Encoder proposal 全有限，364,503,798-byte
  checkpoint 的 642 个浮点 tensor 全有限，iterative-cls/DN 审计通过，致命错误扫描为 0。
- fresh formal 固定 clean HEAD `aea3157`，screen `1470663.pm_0803_27_formal_99`、PGID
  `1470665`。iter50 为 `1.0201 s/iter`、loss `21.4337`、grad `92.6051`，总、DN、Encoder
  proposal 均有限，GPU0/1 各约 19.2 GiB，致命错误为 0，五门槛通过。状态
  `RUNNING/TO_E4+`；继续收 e4/e8/e12，不用 e4/e8 直接否决。

## 2026-08-04 14:44 CST：0803_28 position-tangent + full transport 已准备

- e8 的 full transported tangent `46.283/53.755` 分别比 center-only 高 `4.924/6.824`、比
  shape-only 高 `4.373/5.972`，说明跨中心与形状坐标的 5D 耦合可能是早期增益来源，不能只把
  product 分块作为后继。`0803_28` 保留 full 5D transported geometry，只把 `0803_27` 已验证的
  position-tangent feature detail 加到终层分类输出；递归 query、辅助层、DN 和回归输入仍走父线。
- 结构仍零参数、交换等变、class-agnostic，无 reweight、新 layer、attention 或 loss。远端 py310
  三项 position-tangent 定向测试全部通过：feature/full transport 各在终层调用一次，center/shape
  分块调用为零，参数/state 精确等于父线，输出有限且互斥配置生效。
- 197 2xb4 正式/四步 smoke 配置深拷贝、两份 launcher Bash 语法与专用整模构建审计通过：
  `22,771,111` 参数、增量 0、711 状态张量。99 CPU 隔离验证工作树
  `/data/users/wangying01/lth/PairMOT_positiontransport_0803_28_99` 固定 clean HEAD `11d6b2f`；
  完整 bundle 两端 SHA-256 一致，197 隔离 checkout
  `/data/users/litianhao/PairMOT_positiontransport_0803_28_197` 固定 clean HEAD `1e2be85`，launcher
  语法复核通过且 smoke/formal workdir 均不存在。状态 `PREPARED/NO_GPU`；待 197 shape-only e12
  闭环后再启动，启动器只要求当时两张空闲卡，不固定 GPU 序号。

## 2026-08-04 14:49 CST：0803_23 full-tangent epoch-28 检测继续改善

- full transported tangent e28 cls HOTA/DetA/AssA `51.971/43.572/64.105`，det
  `58.914/51.980/69.156`。相对 e24 `52.012/58.551` 为 `-0.041/+0.363`，分类基本平台、检测
  继续改善；相对原始 decoder e28 `52.177/59.280` 为 `-0.206/-0.366`，差距较小但仍未形成优势。
- pair mAP/AP50 `0.2959/0.5116`，both-independent mAP/AP50 `0.3393/0.5561`；
  403,196,980-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件和
  `async_done=1` 完整。
- e28 未达最终门槛，但 det 尚未回落，不能把分类单点平台作为慢收敛 decoder 的否决依据。178
  当前动态 GPU0、PGID `3151184` 已自然进入 e29，继续到 e32；GPU1 新出现的外部显存占用不动，
  本任务仍只使用一张卡且不固定其序号。

## 2026-08-04 15:14 CST：0803_29 position-tangent + osculating-plane 已准备

- `0803_23` 的 5D full-tangent 在 e8 有强增益，但到 e28 接近原 decoder 后出现成熟一维瓶颈；
  center-only/shape-only 又证明简单分块会丢失跨坐标耦合。`0803_29` 因此保留 5D 耦合，把终层
  pair detail 投影到由“既有帧间运动”和“当前双帧共同终层修正”张成的至多二维正交切平面。
  第一方向延续物体运动，第二方向允许沿当前共同修正产生帧间速度差，避免把所有横向有效细节
  一并删除；分类仍使用 0803_27 的 position-tangent evidence。
- 两个基方向均 detached，Gram-Schmidt 与安全单位化保证投影能量不增。结构仅在最后一层调用
  一次，DN prefix、递归 query 与辅助输出不变；零新增参数/state/attention/layer/loss，交换等变、
  class-agnostic，不使用 reweight。
- 99 py310 两项定向测试通过，覆盖终层 feature/plane 各调用一次、父模型参数/state 精确相等、
  交换等变、DN 保留、能量上界和有限梯度；formal/smoke 配置深拷贝、两份 launcher Bash 语法、
  整模构建均通过，模型 `22,771,111` 参数、增量 0、711 状态张量。隔离仓库
  `/data/users/wangying01/lth/PairMOT_tangentplane_0803_29_99_retry1` 为 clean HEAD `4738c27`。
- 第一次 bundle clone 因无关 Git LFS 演示 GIF 下载 EOF 中断；使用 `GIT_LFS_SKIP_SMUDGE=1`
  在 `_retry1` 隔离路径恢复，未触碰活动 checkout。当前 `PREPARED/NO_GPU`，没有 smoke/formal
  workdir 或队列；99 仍只运行动态双卡 `0803_27`。

## 2026-08-04 15:49 CST：0803_13 terminal geometry epoch-52 仍未满足严格目标

- e52 cls HOTA/DetA/AssA `54.807/45.165/68.197`，det
  `61.986/54.139/73.360`。相对 e48 `+0.274/+0.399`，相对原 decoder e52
  `54.695/62.388` 为 `+0.112/-0.402`；cls 超最终 Encoder `54.437` 达 `+0.370`，det 仍低
  最终 Encoder `62.393` 达 `0.407`，同一 checkpoint 没有双过线。
- 绝对和 `116.793`，距严格 `>118.330` 尚差 `1.537`。pair mAP/AP50
  `0.3206/0.5408`，both-independent `0.3604/0.5771`；436,067,574-byte checkpoint、
  5416 条检测、50 序列、28 CSV、108 个非空评测文件和 `async_done=1` 完整。
- 原 decoder 的 det 在 e56 达到 `62.456` 峰值，且本线 e48→e52 两项均恢复，因此固定 252
  GPU0/1 的 PGID `419164` 已自然进入 e53，继续到 e56；GPU2/3 不用于本任务。

## 2026-08-04 15:52 CST：0803_27 position/product epoch-4 仅登记早期负信号

- e4 cls HOTA/DetA/AssA `30.423/26.178/37.810`，det
  `38.129/34.332/43.541`；相对原 decoder e4 `34.306/38.590` 为 `-3.883/-0.461`。
  position-tangent 分类细节在早期明显拖累 cls，但 e4 不作为 decoder 直接否决。
- pair mAP/AP50 `0.1333/0.2615`，both-independent `0.1777/0.3340`；369,970,934-byte
  checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件和 `async_done=1` 完整。
- 99 动态 GPU0/1、PGID `1470665` 已进入 e5，继续 e8/e12；GPU2 空闲，序号不是一般固定规则。

## 2026-08-04 16:04 CST：0803_23 full-tangent epoch-32 继续上升

- e32 cls HOTA/DetA/AssA `52.627/44.100/64.897`，det
  `59.424/52.308/69.943`。较 e28 `51.971/58.914` 为 `+0.656/+0.510`；相对原 decoder
  e32 `52.566/59.955` 为 `+0.061/-0.531`。分类已略超原 decoder，检测仍是瓶颈，但曲线没有回落。
- pair mAP/AP50 `0.2989/0.5139`，both-independent `0.3421/0.5580`；408,725,620-byte
  checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件和 `async_done=1` 完整。
- 178 当前动态 GPU0、PGID `3151184` 已进入 e33，继续 e36；GPU1 外部任务保持不动。该保留由
  e28→e32 双升的成熟趋势决定，不是因为忽略最终门槛。

## 2026-08-04 16:36 CST：0803_24 shape-only epoch-12 成熟停止

- e12 cls HOTA/DetA/AssA `47.512/39.186/60.080`，det
  `53.757/48.086/61.961`。相对原 decoder e12 `47.395/54.436` 为 `+0.117/-0.679`；分类
  刚转为小幅正向，但 det 在 e4/e8/e12 三个完整节点始终没有优势，也远低于 full-tangent 早期信号。
- pair mAP/AP50 `0.2459/0.4382`，both-independent `0.2925/0.4978`；381,036,199-byte
  checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件和 `async_done=1` 完整。
- 该分量归因已跨过 e4/e8 并完成 e12 成熟节点，故精确 TERM PGID `712277`，成员 `23→0`；
  GPU2/3 显存回到 1 MiB，GPU0/1 外部任务未动。资源转给保留 5D 耦合的 `0803_28`。

## 2026-08-04 16:42 CST：197 动态 GPU2/3 启动 0803_28

- shape-only 停止后 GPU2/3 连续三次为 `1 MiB/0%`，formal 前又连续两次通过；按 197 只限制
  两卡数的规则动态选择 2/3，不写死一般序号。远端 clean HEAD `1e2be85`、formal/smoke workdir
  fresh、两份 launcher Bash 语法和目标配置均再次核验。
- 四步真数据 DDP smoke loss `12.9435/19.4797/19.5753/21.1677`，grad
  `102.5861/118.0605/117.2151/111.1674`；DN/Encoder proposal 全有限，364,503,143-byte
  checkpoint 的 642 个浮点 tensor 全有限，错误扫描为 0，完成后 GPU2/3 释放。
- fresh formal screen `1016334.pm_0803_28_formal_197`、PGID `1016336`；iter50
  `1.8245 s/iter`、loss `21.3980`、grad `127.8108`，7 个成员、GPU2/3 各约 19.2 GiB，
  总、DN、Encoder proposal 均有限且无致命错误，五门槛通过。状态 `RUNNING/TO_E4+`；继续
  e4/e8/e12，不用 e4/e8 直接否决。

## 2026-08-04 19:09 CST：历史心跳审计、四线成熟节点与 0803_29 启动

- 252 心跳点名的历史 `0803_01 fresh` 和 `0801_09 e56 resume` 均无训练进程或 screen。
  前者 `last_checkpoint` 为 e12，e4/e8/e12 三组评测产物保留；后者为 e64，e60/e64 检测与
  TrackEval 产物保留。当前唯一合法 252 训练仍是固定 GPU0/1 的 `0803_13` PGID `419164`；
  GPU2/3 为 `1 MiB/0%`，未被本任务使用。
- `0803_13` e56 cls HOTA/DetA/AssA `54.980/45.191/68.675`，det
  `62.009/54.194/73.397`；cls 超最终 Encoder `+0.543`，det 仍低 `0.384`，总和
  `116.989` 距严格 `>118.330` 差 `1.341`。e60 回落到 `54.855/61.870`，DetA/AssA 为
  cls `45.293/68.240`、det `54.355/72.865`。e56/e60 checkpoint、检测、50 序列、28 CSV、
  108 文件与完整 TrackEval 均闭环；PGID 已到 e62，只保留 e64 平台确认。
- 178 `0803_23` e36 为 `52.856/60.111`，e40 为 `52.689/60.163`；e36→e40 cls
  `-0.167`、det `+0.052`，且 e40 相对原 decoder 为 `-1.370/-0.939`。两节点 AP 与完整
  TrackEval 齐全，当前动态 GPU0、PGID `3151184` 已到 e43，继续到 e44 成熟确认；GPU1 外部
  任务不动。
- 99 `0803_27` e8 `41.889/48.165`、e12 `46.017/52.755`；e12 相对原 decoder
  `-1.378/-1.681`，DetA/AssA 与 AP 未形成成熟反转。该结论来自完整 e4/e8/e12 三节点，故
  精确 TERM PGID `1470665`，成员 `23→0`，不是 e4/e8 早停。
- 动态空闲 GPU0/1 上，`0803_29` 四步真数据 smoke loss
  `12.9379/19.4607/19.5619/21.2173`，grad
  `102.7358/123.0885/106.2158/116.3757`；DN/Encoder 与 642 个浮点 checkpoint tensor
  全有限，错误扫描为 0。fresh formal clean HEAD `4738c27`，screen
  `1582834.pm_0803_29_formal_99`、PGID `1582836`；iter50 `0.9770 s/iter`、loss
  `21.3957`、grad `109.0853`，7 个进程、GPU0/1 各约 19.2 GiB，五门槛通过，登记
  `RUNNING/TO_E4+`。99 仍只占两卡且序号是本次动态选择。
- 197 `0803_28` e4 cls HOTA/DetA/AssA `31.244/25.433/41.202`，det
  `38.396/32.487/46.486`；相对原 decoder `-3.062/-0.194`，相对 `0803_27` e4
  `+0.821/+0.267`。pair mAP/AP50 `0.1307/0.2564`，checkpoint、28 CSV、108 文件和完整
  TrackEval 齐全。该点只登记 position 分类仍慢、full transport 较 product 略好；动态 GPU2/3
  的 PGID `1016336` 继续 e8/e12，GPU0/1 外部任务不动。

## 2026-08-04 19:20 CST：0803_30 geometry-only osculating-plane 已准备

- `0803_27/28` 的 position-tangent 分类在 e4 都明显拖慢 cls，而 178 `0803_23` 的成熟
  full-tangent 几何主要停在一维 detail 投影。下一单因素因此只把终层 normal-query box detail
  从既有 motion 一维切线改为“既有 motion + detached pair-common terminal correction”张成的
  至多二维正交切平面；分类、DN、辅助层和递归 reference 完全保持 `0803_23` 路径。
- 新开关与 position-tangent 组合模式分离，定向测试确认几何调用恰好一次、position feature
  调用为零、原三元 decoder 输出契约不变、交换等变、DN 保留、投影能量不增且梯度有限。
  两项远端 unittest 通过；formal/smoke 配置均完成深拷贝，整模为 `22,771,111` 参数、增量 0、
  711 个 state tensor，两份启动器 Bash 语法通过。
- 178 隔离仓库 `/data1/users/litianhao01/PairMOT_terminaltransportplane_0803_30` 固定 clean HEAD
  `c2069cd`。状态仅 `PREPARED/NO_GPU`，未创建 smoke/formal workdir；等待 `0803_23` e44 完整
  checkpoint、检测和 TrackEval 后再决定交接，不热更新当前训练仓库。启动器要求传入届时一张
  空闲 GPU，不固定 178 序号。

## 2026-08-04 19:52 CST：0803_23 epoch-44 恢复，不为后继提前停止

- e44 cls HOTA/DetA/AssA `53.672/44.220/67.433`，det
  `60.553/53.202/71.309`；相对 e40 `52.689/60.163` 为 `+0.983/+0.390`，否定了 e40
  小幅平台可直接外推的判断。相对原 decoder e44 `54.415/61.737` 仍低 `0.743/1.184`，相对
  最终 Encoder 低 `0.765/1.840`，绝对和 `114.225` 距严格 `>118.330` 差 `4.105`。
- pair mAP/AP50 `0.3052/0.5199`，both-independent `0.3463/0.5598`；425,288,244-byte
  checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件齐全。该节点未达最终目标，
  但 11 个完整节点的最新段仍在恢复，因此不能以“已有成熟点数量多”为由忽略慢收敛趋势。
- 178 当前动态 GPU0、PGID `3151184` 已自然进入 e45，继续 e48；GPU1 外部任务不动。
  `0803_30` 保持 clean PREPARED/NO_GPU，不创建 smoke/formal workdir、不抢占正在恢复的强线。

## 2026-08-04 20:14 CST：0803_13 epoch-64 成熟停线并释放 252

- e64 同一 checkpoint 的 cls HOTA/DetA/AssA 为 `54.930/45.198/68.617`，det 为
  `61.999/54.369/73.115`；cls 高于最终 Encoder `54.437` 达 `+0.493`，但 det 仍低于
  `62.393` 达 `0.394`，绝对和 `116.929` 距严格 `>118.330` 尚差 `1.401`，未双过线、未达目标。
- pair mAP/AP50 `0.3224/0.5369`，both-independent `0.3612/0.5717`；epoch64 checkpoint、
  5416 条检测、50 序列、28 CSV、108 个非空评测文件及 `async_done=1` 全部闭环。该点较 e56
  最佳 `54.980/62.009` 还低 `0.050/0.010`，e56→e60→e64 没有继续改善证据。
- 在 e64 异步 TrackEval 完成后，精确 TERM 训练 PGID `419164`，成员 `23→0`，TrackEval 残留
  为 0；252 GPU0/1 显存均回到 `1 MiB`，GPU2/3 仍为 `1 MiB`，没有被本任务占用。全部
  checkpoint、检测和 TrackEval 产物保留。该停线发生在 e64 成熟长轨迹，不是以 e4/e8 早停。
- 20:14 资源审计：99 `0803_29` 在 epoch4 iter900，动态 GPU0/1、7 个正式进程健康；197
  `0803_28` 在 epoch7 iter750，动态 GPU2/3、23 个正式进程健康，GPU0/1 外部任务不动；178
  `0803_23` 在 epoch46 iter750，动态 GPU0、9 个正式进程健康，GPU1 外部任务不动。继续优先
  收集 99 e4、197 e8 和 178 e48，不以 e4/e8 直接否决 decoder。

## 2026-08-04 20:28 CST：0803_30 在 252 固定 GPU0/1 正式运行，0803_29 e4 登记

- 252 释放后，为 geometry-only osculating-plane `0803_30` 新增独立 2x4 配置与 smoke/formal
  启动器；只把 178 已验证候选移植到 global batch 8，并固定 252 GPU0/1，不改变 decoder
  科学结构。隔离 checkout `/data/users/litianhao01/PairMOT_terminaltransportplane_0803_30_252`
  为 clean HEAD `c3fc5a1`，两项定向测试、配置深拷贝、启动器语法与完整构建通过：
  `22,771,111` 参数、增量 0、711 个 state tensor。
- 首次 smoke 在模型训练前发现继承的 197 数据/GMC/TrackEval 物理路径；失败目录保留并改名为
  `smoke_0803_30_iterative_cls_terminal_transport_plane_4iter_failed_data_path_20260804_2022`。
  只修复 252 物理路径后 fresh 重试，双卡四步 loss `12.9396/19.5077/19.6125/21.2174`、
  grad `102.6709/115.7236/99.8417/105.2715`，DN/Encoder 与 642 个浮点 checkpoint tensor
  全有限，364,502,774-byte `iter_4.pth` 落盘，致命错误为 0。
- fresh formal screen `798987.pm_0803_30_formal_252`、PGID `798989`；iter50
  `1.1484 s/iter`、loss `21.3741`、grad `109.9191`，7 个进程，GPU0/1 各约 19.2 GiB，
  total、DN、Encoder proposal 全有限且 fatal=0，五门槛通过。状态 `RUNNING/TO_E4+`；GPU2/3
  为 `1 MiB`，不用于本任务。
- 99 `0803_29` e4 cls HOTA/DetA/AssA `30.658/25.232/39.985`，det
  `38.402/32.550/46.208`；相对原 decoder e4 `34.306/38.590` 为 `-3.648/-0.188`，相对
  `0803_27` e4 为 `+0.235/+0.273`。pair mAP/AP50 `0.1306/0.2495`，checkpoint、5416 条检测、
  50 序列、28 CSV、108 个非空文件完整。切平面改善同类早期几何，但 position 分类仍慢；动态
  GPU0/1、PGID `1582836` 已进入 e5，继续 e8/e12，不以 e4 直接否决。

## 2026-08-04 21:14 CST：0803_23 e48 双升续跑，0803_28 e8 继续成熟窗口

- 178 `0803_23` e48 cls HOTA/DetA/AssA `53.944/44.633/67.242`，det
  `60.888/53.443/71.770`；较 e44 `53.672/60.553` 再升 `+0.272/+0.335`。相对原 decoder e48
  `54.609/62.091` 仍低 `0.665/1.203`，相对最终 Encoder 低 `0.493/1.505`；绝对和 `114.832`
  距严格 `>118.330` 尚差 `3.498`，未达目标。
- pair mAP/AP50 `0.3054/0.5221`、both-independent `0.3461/0.5618`，430,807,092-byte
  checkpoint、5416 条检测、50 序列、28 CSV、108 个非空文件及 `async_done=1` 完整。e44→e48
  自身双升且原 decoder 的检测峰值在 e52–e56，因此动态 GPU0、PGID `3151184` 继续到 e52，
  不把尚未过最终线误当成平台；GPU1 外部任务不动。
- 197 `0803_28` e8 cls HOTA/DetA/AssA `38.401/32.595/48.019`，det
  `44.520/41.869/48.825`；相对原 decoder e8 `41.972/48.178` 为 `-3.571/-3.658`，相对
  `0803_27` position/product e8 `41.889/48.165` 为 `-3.488/-3.645`，说明 position 分类与
  full transport 在中期没有形成互补。
- pair mAP/AP50 `0.1898/0.3386`、both-independent `0.2370/0.4073`，375,532,775-byte
  checkpoint、5416/50、28 CSV、108 文件与异步完成证据齐全。e8 只登记中期负信号；动态
  GPU2/3、PGID `1016336` 已进入 e9，继续 e12 后再作成熟判定，GPU0/1 外部任务不动。
