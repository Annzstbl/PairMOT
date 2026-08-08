# PairMOT 多服务器实验状态总表

更新时间：2026-08-09 00:40 CST。

本文档记录当前论文相关正式实验在各服务器上的分布和状态。状态由实际训练进程、共享
存储中的 checkpoint/日志及已有报告交叉确认。`smoke_*`、`tmp_*`、`profile_*` 和
`detcheck` 等短测不作为正式实验单列；同一实验的失败启动、重启和最终有效目录合并说明。

状态统一使用英文：`RUNNING`表示当前存在训练进程；`QUEUED`表示等待脚本仍存活且尚未
启动正式训练；`PREPARED`表示代码和静态验证已完成但尚未进入队列；`COMPLETED`表示完成目标
训练或评估；`STOPPED`表示主动取消、硬件中断或不再续跑；`NONE`表示该服务器当前无运行或
排队任务。时间均为CST；无法从日志或启动记录可靠确定时留空。

## 当前资源总览

资源边界为：252 固定 GPU0/1（2 卡、且为最慢资源）；99 总计 2 卡、178 总计 1 卡、197 总计 2 卡，后三者不固定 GPU 序号。每台机器同一时间不超过对应总卡数。

| 服务器 | 当前实验 | 当前进度 | 排队实验 | 工作目录根路径 |
| --- | --- | --- | --- | --- |
| 99 本机 | `0808_06 product-tangent delayed LR clock`（动态 GPU0/1） | RUNNING/E21I300/E20_COMPLETE/TO_E72；screen/main `2606264/2606266` | e20 `50.048/57.885`，总和领先 197 同点 `0.839`；继续 e24+ | `/data4/litianhao/PairMmot/workdir_99` |
| 197 | `0808_07 product-tangent staged delayed LR clock`（动态 GPU0/1） | RUNNING/E21I350+/E20_COMPLETE/TO_E72；screen/main `3583196/3583197` | e20 `50.650/56.444`；继续 e24/e28 | `/data4/litianhao/PairMmot/workdir_197` |
| 252 | `0808_08 product-tangent decoder/head local Adam clock`（固定 GPU0/1） | RUNNING/E1I600/TO_E72；screen/main `1642666/1642667` | formal 健康；e4/e8 仅诊断，继续成熟节点；GPU2/3 未使用 | `/data4/litianhao/PairMmot/workdir_252` |
| 178 | `0808_03 product-tangent decoder/head LR×4/3`（动态 GPU0） | RUNNING/E35/E32_COMPLETE/TO_E72；main `1346509` | `0809_01` 局部延迟 LR 已静态验证但不占卡；先闭环 e36 | `/data4/litianhao/PairMmot/workdir_178` |
| AutoDL | 无训练 | 所有实例关机 | 无 | `/root/autodl-tmp/work_dirs` |

## 2026-08-08 19:11 CST：252 e16 与 99 延迟-LR e4 闭环

- 252 `0808_04` e16 cls HOTA/DetA/AssA 为 `48.631/41.238/59.606`，det 为
  `56.237/50.630/64.554`，sum `104.868`；相对自身 e12 双升 `1.092/1.760`，但相对
  直接父线 e16 仍低 `2.589/1.133`，也低 178 同点 `0.884/0.594`。pair mAP/AP50
  `0.267205/0.462558`、both-independent `0.311959/0.514185`。
- 252 的 386,529,334-byte checkpoint SHA-256 为
  `d6bf6cc77c84b9118c460c8803b9ef0b7115c36db45407814328d0542a057603`；meta
  `16/16608`，642 个浮点张量全有限，iterative-cls 与 DN 状态已训练。5416/50、28 CSV、
  108 个非空文件、50 predictions 闭合，TrackEval 耗时 377.5 秒。main 已恢复 e17 iter300，
  只使用固定 GPU0/1；因 e12→e16 仍双升，继续 e20/e24+。
- 99 `0808_06` e4 cls HOTA/DetA/AssA `31.479/26.080/40.997`，det
  `38.507/33.709/44.986`，sum `69.986`；pair `0.132251/0.258609`，
  both-independent `0.175543/0.331471`。相对父线 e4 低 `3.795/5.342`，只登记早期
  负诊断，不以 e4 否决，继续 e8/e12 与首次 LR 切换。
- 99 e4 checkpoint 为 369,982,902 bytes，SHA-256
  `ec2df1f9a36ab1fa7028e5d03be65b6375c2753ed3b4c83a2731d1c080367786`；5416/50、
  28 CSV、108 文件、50 predictions 闭合，TrackEval 耗时 251.2 秒。正式线已到 e5 iter650，
  动态 GPU0/1 运行，GPU2 未触碰。

## 2026-08-08 19:24 CST：197 分阶段延迟-LR e4 闭环

- 197 `0808_07` e4 cls HOTA/DetA/AssA `31.568/25.863/41.490`，det
  `40.475/33.355/50.267`，sum `72.043`；pair mAP/AP50
  `0.136393/0.257404`，both-independent `0.177854/0.325145`。相对父线 e4 仍低
  `3.706/3.374`，但比 99 `0808_06` 同点高 `0.089/1.968`。仅作早期稳定性诊断，
  不能在 e4 否决，继续 e8/e12 与 e12 后的首次 LR 提升。
- 197 e4 checkpoint 为 369,976,295 bytes，SHA-256
  `72ed271d3467a66c5a0e93b1890b0b488df798c0cd9ca3bd06234cd5851a1b30`；5416/50、
  28 CSV、108 个非空文件、50 predictions 完整，TrackEval 283.5 秒自然结束。正式线已恢复
  e5 iter350，动态 GPU0/1 运行，GPU2–5 空闲。

## 2026-08-08 19:54 CST：178 `0808_03` e20 全量闭环并继续

- e20 cls HOTA/DetA/AssA `50.513/41.739/63.720`，det
  `57.931/50.697/68.493`，sum `108.444`；相对 e16 双升 `0.998/1.100`，相对直接
  product-tangent 父线 e20 `52.198/58.132` 只低 `1.685/0.201`，联合差距由 e16 的
  `2.244` 收窄到 `1.886`。因此保留到 e24+，不作成熟停线。
- pair mAP/AP50 `0.276800/0.471258`、both-independent
  `0.318240/0.516288`，均较 e16 上升。392,144,884-byte checkpoint SHA-256 为
  `a49d946cfa5e042c590c9ed6d2fd76d9f77cb6ffe11245df3baedf52b1bacbae`；meta
  `20/20760`，642 个浮点张量、iterative-cls 与 DN 状态审计通过。
- 5416/50、28 CSV、108 个非空文件、50 predictions 完整，TrackEval 250.2 秒自然结束；
  main 已恢复 e21 iter300，只使用 GPU0。GPU1 的约 10.3 GiB 外部任务未触碰，仍严格遵守
  178 单卡上限。

## 2026-08-08 20:36 CST：99/197 e8 与 252 e20 全量闭环

- 99 `0808_06` e8 cls HOTA/DetA/AssA `42.129/34.567/54.207`，det
  `46.937/42.547/53.729`，sum `89.066`；pair mAP/AP50
  `0.202483/0.373713`、both-independent `0.246844/0.436480`。相对直接父线 e8 低
  `4.544/6.985`。375,549,430-byte checkpoint 的 SHA-256 为
  `fc74e41074a472a99e2921e4ca830e0fe41db379f9333bdf38eda9efb1990755`，meta
  `8/8304`；model/EMA 各 642 个浮点张量全有限，5416/50、28 CSV、108 非空文件和
  50 predictions 完整，TrackEval 273.6 秒自然完成。训练已恢复至 e9 iter1000；e12
  首次升 LR 前不作成熟停线。
- 197 `0808_07` e8 cls `40.206/34.342/49.543`、det
  `46.565/43.035/52.302`，sum `86.771`；pair `0.198946/0.363087`，both-independent
  `0.244086/0.426887`。它低父线 `6.467/7.357`，也低 99 同点 `1.923/0.372`。
  375,541,479-byte checkpoint SHA-256
  `da4a2027cc81312007ac108e4f46eafcd01569ec06ad8f73de0f4f623080bf74`，meta
  `8/8304`，完整有限性、5416/50、28/108/50 与 328.8 秒 TrackEval 均闭环。e12/e24
  调度切换尚未发生，训练继续 e12/e16。
- 252 `0808_04` e20 cls `49.772/41.991/61.137`、det
  `57.365/50.966/66.730`，sum `107.137`。相对 e16 双升 `1.141/1.128`，但低父线 e20
  `2.426/0.767`、联合差距 `3.193`，也低 178 同点 `0.741/0.566`。pair
  `0.274364/0.471042`、both-independent `0.318469/0.520843` 均较 e16 上升。
  392,030,454-byte checkpoint SHA-256
  `e5e920cd3730b4fd8e1df84d7c247ff396d0a9a6ca4b3922b22b54d7d925680f`，meta
  `20/20760`，model/EMA 各 642 个浮点张量全有限；5416/50、28/108/50 与 387.1 秒
  TrackEval 完整。main 已恢复 e21 iter350，只使用固定 GPU0/1，GPU2/3 空闲。
- 四台现场核对为 99 e9i1000、197 e9i600、178 e23i1000、252 e21i350；全部正式日志
  有限且 GPU 与进程一致。178 的 decoder/head-only 加速仍是当前最强路线，优先收集 e24；
  另外三线继续其既定成熟窗口，不用 e8 单点释放资源。

## 2026-08-08 18:39 CST：178 `0808_03` e16 全量闭环并继续

- e16 cls HOTA/DetA/AssA `49.515/41.094/62.034`，det
  `56.831/50.640/65.907`，sum `106.346`。相对 e12 双升
  `1.517/1.668`；相对直接父线 e16 `51.220/57.370` 只低 `1.705/0.539`，联合低
  `2.244`。严格 e72 验收器 rc=2 是预期行为；该成熟点仍在恢复，不作停止判断。
- pair mAP/AP50 `0.273383/0.468078`、both-independent
  `0.315196/0.515004`，均较 e12 提高。386,621,172-byte checkpoint 的 SHA-256 为
  `ea0200ff9d5557d11e93bba39e458ae7c985865c300302a8a86daa29a6b8d31d`；
  iterative-cls/DN 已训练，642 个浮点张量全有限。
- 5416/50 检测、28 CSV、108 个非空评测文件、50 个非空 prediction 全部闭环；异步
  TrackEval 在 253.7 秒后自然结束。main `1346509` 已恢复 e17 iter350，GPU0 约
  21.7 GiB，GPU1 为空，严格遵守 178 单卡上限。继续收集 e20/e24+。

## 2026-08-08 18:15 CST：197 e16 完整审计、成熟停止与 `0808_07` 交接

- `0808_02` e16 cls HOTA/DetA/AssA `47.432/38.067/61.774`，det
  `57.103/49.726/67.955`，sum `104.535`；pair mAP/AP50
  `0.247341/0.430837`，both-independent `0.287731/0.477695`。同一
  386,531,943-byte checkpoint、5416/50 检测、28 CSV、108 个非空评测文件、50 个非空
  prediction 和异步完成日志闭环，严格验收器 rc=2。
- 相对直接 product-tangent 父线 e16 `51.220/57.370`，该全局高 LR 线仍低
  `3.788/0.267`，联合低 `4.055`；e12→e16 主要仅 det 回升，cls 父线差距反而扩大。
  在 e4/e8/e12/e16 四节点成熟证据后，PGID `2811864` 精确 TERM，成员 `23→0`，
  GPU0/1 连续两次为 `1 MiB/0%`；这是成熟换线，不是 e4/e8 否决。
- 后继 `0808_07` 的 clean detached HEAD 为 `adca76a`。真实 GPU0/1 DDP smoke 四步
  loss `12.9372/19.4996/19.6153/21.2276`、grad
  `102.8418/91.7395/85.7160/93.2128`；364,513,383-byte checkpoint 的
  iterative-cls/DN 与 642 个浮点张量检查通过。fresh formal screen/main
  `3583196/3583197` 达到 e1 iter50，LR/loss/grad
  `2.5488e-6/21.4029/112.8655`，双卡约 19.2 GiB 且满载，正式日志更新、全部损失有限、
  fatal=0，故登记 `RUNNING/TO_E72`。
- `0808_07` 只改变训练 LR 时序：e0–11 为 `1e-4`，e12–23 为
  `1.2037682266e-4`，e24–71 为 `1.4490579434e-4`；名义 LR 积分严格等于父线 e96。
  最终 product-tangent 推理结构、参数/state、数据、loss 均不变，class-agnostic、无
  reweight、无推理计算增量。

## 2026-08-08 14:04 CST：四条收敛压缩线完成 warmup 完整性审计

- 99 在 e2 iter1000 达到目标全局 LR `1.25e-4`，loss/grad
  `11.8907/33.7476`；197 在 e2 iter1000 达到 `1.3333e-4`，随后自然进入 e3 iter100，
  当前 loss/grad `11.1644/33.9302`。两台均只使用动态 GPU0/1，额外 GPU 未触碰。
- 178 在 e2 iter1000 达到全局 `1e-4`，配置日志已证明 decoder/bbox-head 参数组实际为
  `1.3333e-4`；当前 e3 iter150 loss/grad `11.4395/34.4983`，只使用 GPU0。
- 252 的压缩 warmup 在总 iter1500 后达到 `1.3333e-4`，跨平台后继续到 e2 iter900，
  loss/grad `11.3209/29.4490`；严格只用固定 GPU0/1，GPU2/3 保持 1 MiB。
- 四条 screen、进程、GPU 与正式日志一致，fatal 扫描均为 0，尚无 scheduled epoch
  checkpoint（interval=4），符合预期。e4 将保存 checkpoint 并触发 5416/50 检测与异步
  TrackEval；`bff42cb` 的严格验收器已以相同 SHA-256 放置在各机 `/tmp`，未热更新存活仓库。

## 2026-08-08 14:25 CST：前三线进入 e4；完整 Adam 时钟候选静态闭环

- 99/197/178 已分别进入 e4 iter150/500/500，252 到 e3 iter1000；四条正式日志持续
  更新且 fatal 扫描为 0。197/178 的 e4 checkpoint 将最先产生，之后按 checkpoint
  有限性、iterative-cls/DN 语义、5416/50 检测、28/108 TrackEval 顺序闭环；e4 仅诊断。
- `0808_05 complete Adam clock compression` 在 `0808_04` 的 LR、warmup、EMA、Liquid
  `96→72` 压缩上，新增且仅新增 Adam 指数记忆时钟压缩：`beta' = beta^(96/72)`，即
  `beta1=0.8689404461`、`beta2=0.9986668889`。由此 `beta'^72 = beta^96`，同时提高 LR
  已自然匹配 AdamW 的积分 weight decay；推理模型、数据、loss 和全局 batch 均不变。
- 本地提交 `30cbc71` 已在 99 新隔离 checkout
  `/data/users/wangying01/lth/PairMOT_0808_05_adamclock_99/ai4rs/ai4rs` 完成两配置 deepcopy、
  两 launcher 远端 `bash -n`、父/子完整构建和 state-shape 比对；均为 22,771,111 参数、
  711 states、增量 0。当前严格登记 `STATIC_VALIDATED/NO_SMOKE/NO_FORMAL/NO_GPU`；不热更新
  `0808_01`，也不在任一运行资源上抢占 smoke。

## 2026-08-08 14:57 CST：四线 epoch-4 全量闭环并继续成熟节点

- 99 `0808_01` e4 cls HOTA/DetA/AssA `31.249/25.744/40.540`，det
  `38.298/32.864/45.670`；pair mAP/AP50 `0.136265/0.263169`，both-independent
  `0.178522/0.330828`。369,972,662-byte checkpoint 的 SHA-256 为
  `621b7d482f310feb78e1b4a10b322a0101ea57b94a6ed6389758902005e42ea7`。
- 197 `0808_02` e4 cls `33.362/27.235/44.499`，det `40.192/35.690/46.596`；pair
  `0.149637/0.283232`，both-independent `0.192098/0.351060`。369,967,911-byte
  checkpoint SHA-256 为 `c661e7392d68a861c451d85520a2b4b2e4dd2f7a8b97f10e75598ecbeadc1a70`。
- 178 `0808_03` e4 cls `34.176/28.092/44.525`，det `38.801/35.123/44.211`；pair
  `0.151597/0.280968`，both-independent `0.199536/0.353928`。369,976,884-byte
  checkpoint SHA-256 为 `ad6a531ac45e51d3df2149f2dc8c78aaf8cec5b6a758cbbeb94adc3fac60e453`。
- 三点均由同一 checkpoint 的 5416 条检测、50 序列、28 CSV、108 个非空 TrackEval 文件和
  `async_done=1` 完整证明。相对直接 product-tangent 父线 e4 `35.274/43.849`，三线 HOTA
  均早期落后；但 197/178 的 det DetA 分别比父线 `34.333` 高 `1.357/0.790`，主要瓶颈是
  det AssA 早期下降，而非定位不足。该证据用于调度与归因，不作为 e4 否决。
- 252 `0808_04` e4 cls `34.359/28.251/43.986`，det `41.302/35.660/49.384`；pair
  `0.150897/0.278264`，both-independent `0.195096/0.349369`，是四条新线中当前最佳。
  369,967,606-byte checkpoint SHA-256 为
  `6cbb726f395c11f3beeb61b5b7a23817437a168e928a40a1210a95f6b60cf940`；642 个浮点张量全有限，
  iterative-cls/DN 已训练，5416/50、28/108、50 predictions 与 `async_done=1` 完整。相对父线
  HOTA 低 `0.915/2.547`，但 det DetA 高 `1.327`、AssA 低 `8.716`，一致时钟压缩明显缓解了
  单纯加 LR 的早期关联损伤。训练已进入 e5，固定 GPU0/1，GPU2/3 仍为 1 MiB；四条均继续
  e8/e12，不提前释放或替换。

## 2026-08-08 15:11 CST：关联保真的延迟 LR 时钟候选静态闭环

- e4 显示立即提高 LR 的路线先改善 det DetA、同时损伤 det AssA；因此新增第二后备
  `0808_06 delayed LR clock`。它前 12 epoch 完全保持父线 `1e-4`，从 e12 边界起仅把 LR
  乘 `1.4`，满足 `12×1 + 60×1.4 = 96`，使 e72 的名义 LR 积分等于父线 e96，同时保留
  早期关联形成阶段。推理模型、参数/state、数据、loss、EMA 与 Liquid 均不变，class-agnostic、
  无 reweight、无新增推理计算。
- 本地提交 `f13a48c` 的初版在远端构建时发现当前 MMEngine 不注册字符串形式
  `MultiStepLR`；任何训练或 workdir 创建前即失败。`8cddf56` 改为显式调度器类后，配置
  deepcopy、两 launcher `bash -n`、72-epoch LR 序列均通过：e0--e11 为 `1e-4`，e12--e72
  为 `1.4e-4`。父/子完整构建均为 22,771,111 参数、711 states、差异 0。
- 首次隔离 clone 因无关 LFS 演示 GIF 缺失对象失败，保留在
  `/data/users/wangying01/lth/PairMOT_0808_06_delayedlr_99_clone_failed_lfs_20260808_1507`；随后全程
  `GIT_LFS_SKIP_SMUDGE=1` 的新隔离 checkout
  `/data/users/wangying01/lth/PairMOT_0808_06_delayedlr_99/ai4rs` 为 clean detached
  `8cddf56`。smoke/formal 目录均不存在、无进程、未使用 GPU，严格状态为
  `STATIC_VALIDATED/NO_SMOKE/NO_FORMAL/NO_GPU`，不得登记 RUNNING。
- 15:11 四条正式线主进程、目标 GPU、正式日志均存活且 fatal=0；99/197/178/252 分别到
  e6 iter400、e7 iter50、e6 iter550、e5 iter1000。继续优先收集 e8，不因 e4 释放资源。

## 2026-08-08 16:21 CST：四线 epoch-8 全量闭环并继续 e12

- 99/197/178/252 的 e8 cls/det HOTA 分别为 `43.047/50.080`、`44.177/50.218`、
  `44.233/49.984`、`42.943/49.685`，绝对和依次为 `93.127/94.395/94.217/92.628`。
  相对直接 product-tangent 父线 e8 `46.673/53.922`，四线均未形成中期加速优势；197 为
  当前四线最佳总和，但只比 178 高 `0.178`。
- 四线 cls DetA/AssA 依次为 `35.180/55.317`、`37.007/54.797`、
  `37.236/54.933`、`36.233/53.447`；det 为 `44.431/58.864`、`45.748/57.122`、
  `45.638/56.774`、`45.089/56.616`。父线为 cls `39.891/56.595`、det
  `47.262/64.136`，主要共同缺口仍是 det AssA，同时伴随 DetA 损失。
- pair mAP/AP50 依次为 `0.212543/0.384172`、`0.222107/0.405285`、
  `0.227299/0.402691`、`0.222818/0.389035`；both-independent 为
  `0.253997/0.440963`、`0.267313/0.464481`、`0.276508/0.466215`、
  `0.265917/0.445627`，均低父线 `0.252434/0.444294` 与 `0.303859/0.513938`。
- 四个 checkpoint 字节数为 `375547254/375536359/375572596/375535222`，SHA-256
  依次为 `6a92311b…2f5da`、`72680692…25af8e`、`bd00c286…75fc5`、
  `d949fca2…d69a80`；`last_checkpoint`、iterative-cls/DN、5416/50、28 CSV、108 非空
  文件与 50 predictions 全部闭环，统一验收器因早期绝对阈值按预期返回 2。四条 formal
  已自然恢复并到 e10i100/e11i250/e10i200/e9i350，fatal=0；252 仍固定 GPU0/1。
- e8 只证明“从 epoch 1 立即压缩时钟”暂时破坏关联/AP，不构成 decoder 否决。四线继续到
  e12 成熟窗口；若成熟证据释放资源，`0808_06`（e12 后再升 LR）优先于继承立即压缩假设的
  `0808_05`。两者仍无 smoke/formal/GPU，不能登记 RUNNING。

## 2026-08-08 13:33 CST：开启 decoder e96→e72 收敛压缩主目标

- 新严格保底线要求同一个 epoch-72 checkpoint 的 cls/det HOTA 分别严格超过 Encoder
  `54.437/62.393`，且绝对和严格超过 `117.830`（相对 Encoder 增量和 `>1.0`）；冲刺线为
  达到或超过当前 decoder e96 的 `55.739/62.616`、绝对和 `118.355`。当前父线 e72
  `55.170/62.165`、和 `117.335`，说明主要缺口是 det `0.228` 与总和 `0.495`，不能以
  e4/e8 直接否决 decoder。
- 首批四条均保持最终 terminal-only product-tangent 推理模型、数据、loss、全局 batch 与
  e72 终点不变，参数/state 均为 `22,771,111/711`：99 `0808_01` 仅把全局 LR 从
  `1e-4` 提到 `1.25e-4`；197 `0808_02` 仅提到 `1.333333e-4`；178 `0808_03` 保持
  全局 LR，只令 decoder 与 bbox head 的 LR multiplier 为 `4/3`；252 `0808_04` 将全局
  LR 提到 `4/3`，并按 `72/96` 同步把 warmup `2000→1500` iter、EMA
  momentum `1e-4→1.333333e-4`/gamma `2000→1500`、Liquid anneal/hard-start
  `36→27` epoch。所有路线 class-agnostic、无 reweight、无新增推理参数或明显计算量。
- 隔离提交为 `81aaf00`；四台均从 SHA-256
  `c8ce37b65df32575435941bcf0537f6a4c29cb55bddadb8934c8826ecff44721` 的完整 bundle
  建立 clean detached checkout。99 首次克隆因无关 Git LFS 大文件 smudge EOF 保留为
  `_clone_failed_lfs` 审计目录；其后以跳过无关 LFS 资产的全新 clean clone 完成，不覆盖旧目录。
- 四台均通过远端 launcher `bash -n`、配置 deepcopy、父子完整模型构建与逐键 state-shape
  比对；目标与父线均为 `22,771,111` 参数、711 states。真实 smoke 各完成 4 iter，
  `iter_4.pth` 存在，iterative-cls residual/DN absolute 头已训练，642 个浮点 checkpoint
  张量全部有限，total/DN/Encoder proposal/grad 全有限。
- 动态复核后分配 99 GPU0/1、197 GPU0/1、178 GPU0；252 严格固定 GPU0/1，GPU2/3
  未使用。四条 formal 均已达到真实 iter50：99 loss/grad `21.3299/106.8418`，197
  `21.3255/107.4264`，178 iter50 已通过且当前 iter200 为 `19.3843/70.2974`，252
  iter100 `20.1859/112.8898`；screen/进程、GPU 驻留、正式日志更新、iter50 与有限值/fatal
  扫描五门槛一致，故严格登记 `RUNNING/TO_E72`。

## 2026-08-06 10:15 CST：252 e96 通过全部严格门槛，释放并行候选

- `0806_06 factorized product-tangent` e96 cls HOTA/DetA/AssA
  `55.739/46.578/68.819`，det `62.616/54.836/73.944`。相对 Encoder
  `54.437/62.393` 的增量为 `1.302/0.223`，绝对和 `118.355`、增量和 `1.525`，
  三个严格条件均成立；总和超过 `>118.330` 门槛 `0.025`。e96 相对 e92 HOTA
  再升 `0.205/0.186`，并成为全轨迹唯一最佳同点。
- pair mAP/AP50 `0.3247/0.5358`、both-independent `0.3633/0.5718`。496,152,054-byte
  checkpoint meta `96/99648`，model/EMA 711/712 states、各 642 个浮点张量全有限，
  optimizer 497 states、scheduler、message hub 与 iterative-cls/DN 语义完整。检测
  5416/50，TrackEval 28 CSV/108 非空文件/50 预测/`async_done=1`，366.1 秒自然完成。
  screen、训练与异步进程均自然退出，252 四卡均为 1 MiB/0%。
- 达标 decoder 为零新增参数/state 的 terminal-only product-tangent 几何输运，class-agnostic，
  无 reweight、额外 loss、attention 或 layer，只有常数级末层逐元素计算，符合模型设计约束。
- 99 `0806_07` 健康运行到 e4 iter350 后因目标已达成精确停止，PGID `2037143` 成员
  `7→0`；178 `0806_02` 健康运行到 e3 iter600 后精确停止，PGID `274434` 成员 `9→0`。
  两者均为 `GOAL_ACHIEVED_NOT_REJECTED`，不是以 e4/e8 否决；产物保留，外部作业未触碰。

## 2026-08-06 09:35 CST：178 完成 e12 成熟交接，0806_02 正式运行

- `0806_05 scale-orientation split product-tangent` e12 cls HOTA/DetA/AssA
  `44.366/34.045/60.856`，det `49.670/42.362/60.165`，绝对和 `94.036`。虽然相对 e8
  HOTA 继续上升 `3.298/4.669`，但相对直接 product-tangent 父线 e12 仍低
  `5.418/6.573`，父线的四项 DetA/AssA 也全部更高；pair AP/AP50
  `0.2110/0.3670`、both-independent `0.2509/0.4149` 同样落后。该结论来自完整
  e4/e8/e12 成熟窗口，而非早期节点否决。
- 381,098,996-byte checkpoint meta `12/12456`、model/EMA 711/712 states、642 个有限
  浮点张量、497 optimizer states、scheduler 与 iterative-cls/DN 检查全部通过；检测
  5416/50、TrackEval 28 CSV/108 非空文件和 `async_done=1` 齐全。PGID `175356` 精确
  TERM 后成员 `8→0`、screen 消失、GPU0 连续归零，GPU1 外部作业未触碰。
- `0806_02 log-SPD product-tangent` 随后在 clean detached `9c5018a` 上 fresh 启动。
  配置 deepcopy、launcher 语法、22,771,111 参数/711 states 完整构建、真实 1×8 四步
  smoke、364,507,508-byte checkpoint 有限性和 iterative-cls/DN 语义均通过。formal
  screen/main PGID `274432/274434` 的 iter50 lr/loss/grad
  `2.5488e-6/21.0199/112.7436`，DN、Encoder proposal、进程、GPU0 与 fatal=0 五门槛
  一致，登记 `RUNNING/E1/TO_E4+`。GPU1 仍由外部作业使用，未触碰；e4/e8 只作诊断。

## 2026-08-06 09:24 CST：e92 严格未过；99 成熟负线交接 0806_07

- 252 `0806_06` e92 cls/det HOTA `55.534/62.430`，DetA/AssA 为
  `46.365/68.575` 与 `54.766/73.626`，同点和 `117.964`。单项分别过线
  `1.097/0.037`，但严格总和仍差 `0.366`；相对 e84/e88 均双升，继续同一状态到 e96。
  490,678,390-byte checkpoint meta `92/95496`，model/EMA 711/712 states、各 642 浮点
  张量全有限、optimizer 497 states、scheduler 与 iterative-cls/DN 语义完整；5416/50、
  28 CSV、108 非空文件和异步完成齐全。固定 GPU0/1，GPU2/3 未用。
- 99 `0804_17` e24 cls/det `49.794/57.460`，DetA/AssA 为
  `41.179/62.318` 与 `50.101/68.276`。虽较 e20 继续双升且四项 AP 全升，但相对直接
  product-tangent e24 仍低 `2.684/1.311`，六个完整节点后成熟判负。397,519,734-byte
  checkpoint meta `24/24912`、711/712 states、642 浮点张量、497 optimizer states、
  5416/50、28/108 和异步完成均通过。PGID `1995834` 精确 TERM，成员 `23→0`、screen
  消失，GPU0/1 连续归零，GPU2 未用。
- 资源释放后，`0806_07 stratified product-tangent` 在 clean detached `ead7e1e` 重新通过
  config deepcopy、远端 `bash -n`、22,771,111 参数/711 states 完整构建。动态 GPU0/1
  四步 smoke 的 loss/grad、DN、Encoder proposal 全有限，364,503,990-byte checkpoint
  通过有限性与 iterative-cls/DN 语义检查；formal screen/PGID `2037141/2037143` 的
  iter50 loss/grad `21.4216/101.1440`，两 rank/GPU/正式日志/fatal 扫描五门槛通过，登记
  `RUNNING/E1/TO_E4+`。GPU2 空闲，e4/e8 不作直接否决。

## 2026-08-06 08:15 CST：178 e8 完整评估，继续到 e12

- `0806_05` e8 cls HOTA/DetA/AssA `41.068/32.461/54.947`，det
  `45.001/40.302/51.792`；较 e4 HOTA `+7.537/+9.072`，但相对直接 product-tangent
  e8 仍低 `5.605/8.921`。pair mAP/AP50 `0.1955/0.3456`、both-independent
  `0.2388/0.4024`，定位、关联和 AP 均弱于强父线。
- 375,572,724-byte checkpoint meta `8/8304`，model/EMA 711/712 states、各 642 浮点
  张量全有限，optimizer 497 states、scheduler、loss scaler 与 iterative-cls/DN 语义完整；
  5416/50、28 CSV、108 非空文件、50/50 非空预测、`async_done=1` 齐全，TrackEval
  耗时 236.3 秒。
- 训练已自然进入 e9 iter300，继续 e12，不以 e8 直接否决。任务仍只用动态 GPU0；GPU1
  外部作业约 30.4 GiB/99% 利用率，本任务未触碰。
- 252 `0806_06` 同期到 e92 iter200，loss/grad `7.9346/40.1549`，固定 GPU0/1 正常、
  GPU2/3 为 1 MiB；继续等待 e92 同 checkpoint 的完整 checkpoint、检测和 TrackEval。

## 2026-08-06 07:58 CST：99 e20 全量结果继续上升，保留到 e24

- `0804_17` e20 cls HOTA/DetA/AssA `48.948/40.416/61.552`，det
  `56.926/49.543/67.702`，较 e16 HOTA `+0.633/+1.342`，两侧 DetA、AssA 也全升；
  pair mAP/AP50 `0.2606/0.4598`、both-independent `0.3037/0.5110`，均继续改善。
- 相对直接 product-tangent 父线 e20 `52.198/58.132` 仍低 `3.250/1.206`；绝对和
  `105.874`，距严格 cls/det/和门槛 `5.489/5.467/12.456`，目标未达。
- 392,024,246-byte checkpoint meta `20/20760`，model/EMA 711/712 states、各 642 浮点
  张量全有限，optimizer 497 states、scheduler、loss scaler 与 iterative-cls/DN 语义完整；
  5416/50、28 CSV、108 非空文件、50/50 非空预测及 `async_done=1` 齐全，TrackEval
  耗时 275.4 秒。训练已自然进入 e21，继续 e24，不以 e20 未达标提前停止。

## 2026-08-06 07:39 CST：三条资源健康；0806_07 静态就绪

- 99 `0804_17` 到 e20 iter550，178 `0806_05` 到 e7 iter800，252 `0806_06` 到 e90
  iter200；loss、grad、DN 与 Encoder proposal 均有限。99 仍只用动态 GPU0/1，178 只用
  动态 GPU0 且未触碰 GPU1 新出现的外部作业，252 严格只用固定 GPU0/1、GPU2/3 仍为 1 MiB。
- 新候选 `0806_07 stratified product-tangent` 在零 reference-motion stratum 保留 frame detail，
  非退化样本与直接 product-tangent 完全相同；零参数、class-agnostic、无 reweight 或明显计算
  增量。99 clean detached `ead7e1e` 已通过定向单测、目标/smoke 配置 deepcopy、两个 launcher
  `bash -n` 与父子完整构建；模型均为 22,771,111 参数、711 states。
- 完整 bundle 双端 SHA-256 为
  `d68c34da75a3595f5edf07b44afab2cc52b3ed242105a4b0303003466c63242f`。首次 clone 的多余
  目录层已在确认 clean、无进程和无 workdir 后归档为 `_layout_failed`；显式设置隔离
  `PYTHONPATH` 后构建闭环。当前状态严格为 `STATIC_VALIDATED/NO_SMOKE/NO_FORMAL/NO_GPU`，
  不占用 99 正式训练资源，也不登记 RUNNING。

## 2026-08-06 07:16 CST：252 e88 严格未达标并安全续到 e96

- `0806_04` e88 cls/det HOTA `55.397/62.403`，DetA/AssA 为
  `46.268/68.372` 与 `54.759/73.540`，绝对和 `117.800`。单项分别过线
  `0.960/0.010`，但严格总和仍差 `0.530`；相对 e84 HOTA、DetA/AssA 与四项 AP 均回落，
  e84 仍是该段最佳，目标未完成。
- 485,230,326-byte e88 checkpoint meta `88/91344`，model/EMA 711/712 states、各 642
  浮点张量全有限、optimizer 497 states、scheduler 与 iterative-cls/DN 语义完整；5416/50、
  28 CSV、108 非空文件、50/50 非空预测、`async_done=1` 齐全，TrackEval 耗时 371.5 秒。
  `0806_04` 与异步进程自然结束，四卡归零后才开始后继。
- `0806_06` 新隔离 checkout clean detached `445cac8`；完整 bundle 双端 SHA-256
  `de41380f6dbc591c3d8f2007cf06844ab41fb1aa01114a39b447b05838df77f0`。两配置 deepcopy、
  远端 launcher `bash -n`、父/子 22,771,111 参数/711 states 完整构建、零状态差与 e88 严格加载
  均通过，未热更新 `0806_04` 仓库。
- 固定 GPU0/1 的真实四步 DDP smoke loss
  `12.9389/19.3870/19.5357/21.1102`、grad
  `102.9586/84.7106/85.6180/88.6991`，全部 DN/Encoder proposal 有限；364,503,158-byte
  checkpoint 的 642 浮点张量与 iterative-cls/DN 语义通过，结束后 GPU 归零。
- formal screen `1168832.pairmot_0806_06_e96` 精确恢复 e88/91344；e89 iter50 的
  lr/loss/grad `1e-4/8.1478/42.6524`，fatal=0，GPU0/1 各约 19.4 GiB，GPU2/3 均 1 MiB。
  五项门槛齐全，登记 `RUNNING/E89/TO_E92+`。同期 99 到 e19 iter250、178 到 e6 iter300，
  均遵守资源上限并继续成熟窗口。

## 2026-08-06 06:50 CST：99 e16 全量闭环；178/252 进入评估窗口

- 99 `0804_17` e16 cls/det HOTA 为 `48.315/55.584`，DetA/AssA 分别
  `39.942/60.785` 与 `48.982/65.232`，绝对和 `103.899`。相对 e12 HOTA
  `+2.054/+2.153`，DetA、AssA 与四项 AP 全升；pair mAP/AP50 为
  `0.2530/0.4489`，both-independent 为 `0.2972/0.5026`。相对直接 product-tangent
  父线 e16 的差距缩小到 `1.721/1.349`，故继续 e20/e24，不在仍有恢复趋势时早停。
- e16 距严格 cls/det/同点和门槛仍差 `6.122/6.809/14.431`。386,528,054-byte
  checkpoint meta `16/16608`，model/EMA 711/712 states、各 642 浮点张量全有限、optimizer
  497 states、scheduler 完整，iterative-cls/DN 训练语义通过。5416/50 检测、28 CSV、108
  非空文件、50/50 非空预测与 `metrics.json` 齐全，TrackEval 耗时 282.1 秒并自然退出。
- 178 `0806_05` e4 cls/det HOTA `33.531/35.929`，DetA/AssA 为
  `27.266/43.909` 与 `33.130/40.522`，绝对和 `69.460`；相对直接 product-tangent e4
  HOTA 低 `1.743/7.920`，DetA/AssA 也均低。pair mAP/AP50 `0.1435/0.2772`、
  both-independent `0.1888/0.3507`，均低于父线。这只是 e4 负诊断，不作否决。
- 369,976,948-byte checkpoint meta `4/4152`，model/EMA 711/712 states、各 642 浮点
  张量全有限、optimizer 497 states、scheduler 与 iterative-cls/DN 训练语义完整。5416/50、
  28 CSV、108 非空文件、50/50 非空预测和 `async_done=1` 齐全；TrackEval 耗时 221.2 秒。
  训练已恢复到 e5 iter250，仍仅使用动态 GPU0，继续 e8/e12；GPU1 未用。
- 252 `0806_04` 于 06:49 到 e88 iter600，固定 GPU0/1 各约 21.6 GiB，GPU2/3 仍为
  1 MiB；等待 e88 同一 checkpoint 的检测、TrackEval 与完整性审计后，再决定是否启用
  `0806_06` 纯时长兜底。

## 2026-08-06 05:51 CST：活体资源复核与 252 续时长兜底准备

- 252 `0806_04` 到 e85 iter900，loss/grad `7.7661/34.7581`，固定 GPU0/1 各约
  21.6 GiB，GPU2/3 均为 1 MiB；99 `0804_17` 到 e14 iter700，loss/grad
  `10.6477/44.6135`，动态 GPU0/1 各约 21.4 GiB，GPU2 空闲；178 `0806_05` 到
  e1 iter650，loss/grad `17.2523/193.5542`，动态 GPU0 约 31.4 GiB，GPU1 空闲。
  三线 screen、正式进程、GPU 与日志一致，total、DN、Encoder proposal 和 grad 均有限。
- 新增 `0806_06` e88→e96 纯时长 config、smoke config 与固定 252 GPU0/1 launcher；只改训练
  终点、评估终点和独立输出目录，静态 Python/Bash 检查通过。状态为
  `PREPARED/NOT_DEPLOYED/NO_REMOTE_WORKDIR/NO_GPU`，不热更新 `0806_04` 活跃仓库。
  只有 e88 全量同 checkpoint 仍未严格达标时，才在新隔离 checkout 完成 config deepcopy、
  完整构建、真实双卡 smoke、checkpoint 验收和 formal iter50 五门槛后登记 RUNNING。
- 05:59 已把 `445cac8` 相对 `d664a03` 的增量 bundle 暂存到 252，双端 SHA-256 均为
  `1cddd6dd997633ff2ae23814c34b6bb90f091f600f2dddf92e3fa7fb55f437b7`。目标 checkout 和
  formal workdir 均不存在，未 fetch、未构建、未运行 smoke，故这只是可核验传输准备，不改变
  `PREPARED/NOT_DEPLOYED/NO_GPU` 状态，也不触碰 `0806_04` 活跃 checkout。

## 2026-08-06 04:16 CST：252 duration-only 延长与 99 e8 诊断闭环

- 252 `0806_01` e80 同点 cls/det HOTA 为 `55.446/62.342`，DetA/AssA 分别
  `46.260/68.403` 与 `54.740/73.466`，绝对和 `117.788`。cls 门槛通过，但 det 低
  `0.051`、总和低 `0.542`，严格未完成。pair mAP/AP50 `0.3229/0.5361` 与
  both-independent `0.3624/0.5738` 相对 e76 仍上升，故沿同一优化轨迹延长而非改模型。
- e80 checkpoint 为 474,302,646 bytes、meta `80/83040`，model/EMA 711/712 states、各
  642 个浮点张量全有限，optimizer 497 states；5416/50 检测与 28 CSV、108 非空文件、
  50/50 非空预测、`async_done=1` 完整。新 `0806_04` 隔离 checkout `3e8dde1` 完成配置
  deepcopy、完整构建、远端 `bash -n`、真实双卡 smoke、checkpoint 兼容和 formal iter50
  五项门槛，screen/PGID `1087790/1087792` 到 e81 iter350；只用固定 GPU0/1。
- 99 `0804_17` e8 cls/det HOTA `41.392/47.685`，DetA/AssA 为
  `33.414/54.156` 与 `42.820/55.100`；pair mAP/AP50 `0.1933/0.3595`，
  both-independent `0.2361/0.4212`。375,534,838-byte checkpoint meta `8/8304`，
  711/712 states、各 642 浮点张量全有限、optimizer 497 states；5416/50、28/108/50 和
  `async_done=1` 均闭环。e8 仅作诊断，训练继续到 e12+。
- 178 `0806_03` 在 04:14 保存 e12 checkpoint，随后进入 1354-iter 正式检测；04:16 到
  300/1354。待 TrackEval、AP、DetA/AssA 和 checkpoint 完整性全部闭环后再作成熟判定。

## 2026-08-06 04:31 CST：178 Householder e12 全量闭环并释放 GPU

- e12 cls HOTA/DetA/AssA `45.162/36.909/57.747`，det
  `52.066/46.731/60.028`，绝对和 `97.228`。相对 e8 HOTA 回升
  `+2.566/+4.618`，说明按约束保留到成熟窗口是必要的；但相对直接 product-tangent e12
  仍低 `4.622/4.177`，三项严格门槛分别差 `9.275/10.327/21.102`，故成熟判定为负。
- pair mAP/AP50 `0.2286/0.4080`、both-independent `0.2721/0.4632`；
  381,000,052-byte checkpoint meta `12/12456`，model/EMA 711/712 states、各 642 个浮点
  张量全有限，optimizer 497 states。5416/50 检测、28 CSV、108 个非空文件、50/50 非空预测、
  `async_done=1` 齐全，TrackEval 耗时 248.4 秒。
- screen、正式成员与异步进程均自然结束，GPU0/1 均回到 `1 MiB/0%`。178 仍只允许总计
  1 卡；为了先完成最接近的归因，`0806_02` 保持静态就绪，不在 99 `0804_17` e12+ 前抢跑。

## 2026-08-06 04:38 CST：178 完成 0806_02 预部署 smoke，不启动 formal

- 178 两张物理卡连续两轮均为 `1 MiB/0%`，按总计 1 卡上限动态选择 GPU0 做短 smoke。
  clean detached `9c5018a` 的 formal/smoke 配置 deepcopy、两个 launcher `bash -n`、
  22,771,111 参数/711 states 父子完整构建和零增量检查均通过。
- 首次手工构建因未显式设置隔离 `PYTHONPATH` 而从旧 checkout 导入，立即报旧 head 不接受新配置键；
  没有创建 workdir。按 launcher 的真实环境固定 `PYTHONPATH` 后完整构建通过，确认是环境污染而非
  结构失败，并把该签名写入记录以避免 formal 前复现。
- 真 1×8 四步 smoke 的 loss `21.3720/20.6724/20.9473/21.2590`、grad
  `59.9638/67.2856/78.2630/78.3876`，DN/Encoder proposal 全有限；364,507,508-byte
  checkpoint 的 iterative-cls/DN 语义和 642 个浮点张量均通过。结束后 GPU0/1 回到 1 MiB，
  formal 目录不存在，故只登记 `SMOKE_VALIDATED/NO_FORMAL/GPU_FREE`。

## 2026-08-06 04:55 CST：主线未到 checkpoint；178 准备零 GPU 后继

- 252 `0806_04` 的 screen/PGID `1087790/1087792`、rank、正式日志与 GPU 交叉一致：04:44
  到 e82 iter900，loss/grad `7.9193/44.6804`，固定 GPU0/1 各约 19.4 GiB；GPU2/3 均为
  1 MiB。e84 checkpoint 尚未产生，继续原轨迹。
- 99 `0804_17` 的 screen/PGID `1995832/1995834`、rank、正式日志与 GPU 交叉一致：04:43
  到 e11 iter150，loss/grad `10.2649/40.1892`，动态 GPU0/1 各约 21.4 GiB；GPU2 仅
  10 MiB。当前仍只有 e4/e8 checkpoint，继续到 e12+，不作早停。
- 178 的非占卡 fallback `0806_05 scale-orientation split product-tangent` 在新 clean detached
  `bf3fb3a` 中完成配置 deepcopy、两个 launcher `bash -n`、定向不变量单测和完整父子构建：
  `22,771,111` 参数、711 states、增量 0。它保持 center transport，只在二维 log-size 平面投影
  scale detail，并逐帧保留 proposed periodic angle；parameter-free、class-agnostic、无 reweight、
  无明显计算量增长。未运行 smoke、未创建 formal、未占 GPU，严格等待 99 e12+ 与 252 e84
  后再决定 `0806_02/0806_05` 的正式先后。

## 2026-08-06 05:00 CST：178 完成 0806_05 smoke，不启动 formal

- 178 连续两轮全卡空闲后按总计 1 卡上限动态选择 GPU0。`0806_05` 真实 1×8 四步 smoke
  的 loss 为 `21.3692/20.6400/20.9566/21.2581`，grad 为
  `60.1735/60.6046/64.2854/67.0694`；DN/Encoder proposal 全有限，无致命错误。
- 364,507,636-byte `iter_4.pth` 通过 iterative-cls/DN 语义和 642 浮点张量有限性检查；
  screen/process 自然结束，GPU0/1 回到 1 MiB。formal 目录不存在，故只登记
  `SMOKE_VALIDATED/NO_FORMAL/GPU_FREE`。
- 99 `0804_17` 于 04:59 到 e12 iter50，252 `0806_04` 于 04:56 到 e83 iter450；两条正式
  训练的 total、DN、Encoder proposal 与 grad 均有限。先等 99 e12 全量和 252 e84，不抢跑
  `0806_02/0806_05` formal。

## 2026-08-06 05:43 CST：e84 严格总和未过，178 scale-orientation 接替

- 252 e84 同点 cls HOTA/DetA/AssA `55.474/46.328/68.477`，det
  `62.422/54.790/73.568`，绝对和 `117.896`。虽然 cls/det 分别过 Encoder
  `+1.037/+0.029`，总和仍低 `>118.330` 门槛 `0.434`，故目标未完成。pair mAP/AP50
  `0.3235/0.5364`、both-independent `0.3632/0.5740` 与 HOTA 相对 e80 均仍上升，固定
  GPU0/1 保持相同 duration-only 轨迹到 e88；05:42 到 e85 iter450，GPU2/3 未用。
- `epoch_84.pth` 为 479,753,014 bytes、meta `84/87192`，711 model/712 EMA states、
  642 个浮点张量全有限、optimizer 497 states；5416/50 检测、28 CSV、108 非空文件、
  50/50 非空预测、`async_done=1` 和 392.1 秒 TrackEval 全部闭环。
- 99 e12 同点 cls/det HOTA `46.261/53.431`，DetA/AssA 为
  `38.506/58.002` 与 `47.832/61.670`。相对 e8 回升 `+4.869/+5.746`，AP 也明显恢复；
  但相对直接 product-tangent e12 仍双低 `3.523/2.812`。381,033,974-byte checkpoint、
  711/712 states、642 有限张量、optimizer 497 states、5416/50/28/108/50 与异步完成均齐全。
  因仍在恢复而继续到 e16，当前 05:41 到 e14 iter200，动态 GPU0/1 正常、GPU2 未用。
- 由成熟机制证据选择较少偏离强父线的 `0806_05 scale-orientation split`，而不是更强耦合的
  `0806_02 log-SPD`。178 两轮全卡空闲、formal workdir 缺失、clean detached `bf3fb3a` 与
  launcher `bash -n` 再确认后，动态选择 GPU0 于 05:40 fresh 启动；GPU1 保持 1 MiB。
  screen/PGID `175354/175356`，iter50 为 `0.9508 s/iter`、loss/grad
  `21.0004/106.2984`，iter100 亦有限；进程、GPU、正式日志、iter50 与有限性五门槛齐全，
  登记 `RUNNING/E1/TO_E4+`。`0806_02` 保持 `SMOKE_VALIDATED/NO_FORMAL`。

## 2026-08-06 03:12 CST：178 安全恢复 0804_09 的成熟窗口

- 178 GPU0 连续多次为 `1 MiB/0%`，GPU1 外部任务约 `14.1 GiB/93%+`；仅选择动态空闲的
  GPU0。`0806_03` 从 197 `0804_09` 的 375,529,191-byte e8 checkpoint（meta `8/8304`）
  恢复到 e12，物理 batch 从 2×4 改为 1×8但全局 batch 仍为 8，无梯度累积。
- 隔离 clean detached checkout `36ddea5` 中，formal/smoke 配置 deepcopy、远端 `bash -n`、
  Householder 完整构建和源/目标 checkpoint 兼容均通过：`22,771,111` 参数、711 个同形 state、
  增量 0。真实 1×8 smoke 四步的 total、DN、Encoder proposal、grad 和 642 个 checkpoint
  浮点张量均有限。
- formal screen/PGID `112694/112696` 从 e8/8304 精确加载，e9 iter50 的 loss/grad
  `10.8587/39.1599`，GPU0 约 31.4 GiB，五项门槛通过后登记 `RUNNING/E9/TO_E12`。
  这是被硬件中断的既有候选成熟化，不构成对 e8 的否决，也不提前部署 `0806_02`。
- 03:12 同期：252 固定 GPU0/1 到 e79 iter200，GPU2/3 为 1 MiB；99 动态 GPU0/1 到
  e6 iter200，GPU2 为 10 MiB，两路均继续既定成熟节点。

## 2026-08-06 02:56 CST：99 的 0804_17 e4 仅作诊断并继续

- e4 cls HOTA/DetA/AssA `31.620/25.920/41.202`、det
  `38.330/33.740/44.778`；相对强父线 e4 HOTA `-1.229/+1.011`，其中 det
  DetA/AssA 为 `-1.208/+3.535`，说明早期 det 正值来自关联补偿覆盖，而非均衡改善。
- pair mAP/AP50 `0.1325/0.2594`、both-independent `0.1741/0.3271`，均低于强父线和
  `0804_16` e4。369,972,470-byte checkpoint meta `4/4152`，711 model/712 EMA states
  中各 642 个浮点张量有限，optimizer 497 states 完整；5416/50、28 CSV、108 非空文件、
  50/50 preds、`async_done=1` 和 254.2 秒 TrackEval 完整。
- 该节点不作成熟否决。PGID `1995834` 已到 e5 iter250，动态 GPU0/1 的 total、DN、Encoder
  proposal 与 grad 有限，GPU2 空闲；继续 e8/e12+。252 同时到 e78 iter400并继续 e80。

## 2026-08-06 02:35 CST：252 的 0806_01 e76 完整闭环，继续 e80

- e76 cls HOTA/DetA/AssA `55.233/45.956/68.400`、det
  `62.255/54.670/73.373`；相对 e72 HOTA 双升 `+0.063/+0.090`，但 det 仍低最终
  Encoder `0.138`，绝对和 `117.488` 距严格 `>118.330` 仍差 `0.842`，不得登记成功。
- pair mAP/AP50 `0.3187/0.5331`、both-independent `0.3577/0.5699`，四项相对 e72
  分别 `+0.0001/-0.0004/-0.0004/-0.0010`，检测侧基本平台。468,824,374-byte
  checkpoint meta `76/78888`，711 model/712 EMA states 中各 642 个浮点张量全有限，
  optimizer 497 states 完整。
- 5416/50 检测、28 CSV、108 非空评测文件、50/50 非空预测、`async_done=1` 和 398.6 秒
  TrackEval 全部闭环。PGID `1025568` 已自然到 e77 iter300，正式训练有限，固定 GPU0/1
  继续到 e80，GPU2/3 仍不用。
- 99 `0804_17` 到 e4 iter400，动态 GPU0/1 健康、GPU2 空闲；继续完成 e4 全评测并成熟到
  e8/e12+，不以 e4/e8 直接否决。

## 2026-08-06 02:08 CST：178 的 0804_16 完成 e12 成熟闭环

- e12 cls HOTA/DetA/AssA `46.594/36.616/61.916`、det
  `52.528/45.194/63.207`；相对强父线 e12 `48.289/54.539` 为
  `-1.695/-2.011`，绝对和 `99.122`，未通过 cls `>54.437`、det `>62.393` 或
  同点和 `>118.330` 任一严格门槛。
- e8→e12 HOTA 虽回升 `+4.195/+4.929`，双 DetA、双 AssA 与四项 AP 也回升，但 e12
  pair mAP/AP50 `0.2299/0.4025`、both-independent `0.2718/0.4529` 仍全面低于强父线。
  结论来自 e4/e8/e12 完整窗口，不是 e4/e8 直接否决。
- 381,077,556-byte checkpoint meta `12/12456`、642 个有限浮点张量、5416/50 检测、
  28 CSV、108 个非空评测文件、50/50 个非空预测、`async_done=1` 和 240.1 秒 TrackEval
  均闭环。确认无异步评测残留后精确 TERM PGID `4175891`，成员归零、screen 消失，
  GPU0 回到 `1 MiB/0%`；GPU1 外部作业未触碰。
- 02:07 活体复核：252 固定 GPU0/1 的 `0806_01` 到 e76 iter300，99 动态 GPU0/1 的
  `0804_17` 到 e2 iter850，两路正式 loss、DN、Encoder proposal 与 grad 均有限；252
  GPU2/3 和 99 GPU2 保持空闲。`0806_02` 不提前部署，仍等待 `0804_17` e12+ 成熟判定。

## 2026-08-06 01:38 CST：99 的 0804_17 通过动态五门槛

- 99 三卡连续三次低占用且没有训练进程后，动态选择当时空闲的 GPU0/1；这不把 99 变成
  固定卡号资源。`0804_17` 在 clean detached `a4914d0` 上完成真实双卡 4-iter smoke：
  total/DN/Encoder proposal/grad 全有限，364,503,990-byte checkpoint 的 iterative-cls/DN
  已训练，642 个浮点张量全有限，fatal=0。
- fresh formal screen/PGID `1995832/1995834` 到 e1 iter50，lr/loss/grad
  `2.5488e-6/21.3951/128.3636`，DN/Encoder proposal 全有限，GPU0/1 各约 19.2 GiB、
  GPU2 空闲；五门槛通过后登记 `RUNNING/E1/TO_E4+`。
- 这是独立空闲资源上的并行成熟化，不构成对 178 e8 的早停判断：178 `0804_16` 仍在
  GPU0 到 e12 iter300并继续完整 e12 检测/TrackEval，GPU1 外部作业未动。252 固定 GPU0/1
  到 e74 iter800，GPU2/3 未用。

## 2026-08-06 01:28 CST：0806_02 log-SPD product-tangent 静态就绪

- `0806_02` 用 `(log(wh), log(w/h)cos(2θ), log(w/h)sin(2θ))` 的 log-SPD 正交形状坐标，
  在不改中心切线的前提下做终层 product-tangent transport。结构零参数/state、class-agnostic、
  无 reweight、无明显计算量增长，严格排在 `0804_17` 之后作为 fallback。
- 99/178 隔离仓库均 clean detached `9c5018a`；双端配置 deepcopy、launcher `bash -n`、
  精确单测 `1/1` 与完整构建通过，模型 `22,771,111` 参数、`711` state tensors、增量 0。
  formal/smoke workdir 均不存在，未使用 GPU，登记
  `STATIC_VALIDATED/NOT_DEPLOYED/NO_GPU/FALLBACK_AFTER_0804_17`。
- 同时复核存活作业：252 固定 GPU0/1 的 `0806_01` 到 e74 iter300，GPU2/3 未用；178
  GPU0 的 `0804_16` 到 e11 iter700，GPU1 外部作业未动。正式日志的 total、DN、Encoder
  proposal loss 与 grad norm 均有限；当前没有 e76/e12 新 checkpoint，继续等待成熟节点。

## 2026-08-06 01:03 CST：252 的 0806_01 e72→e80 成熟续训启动

- e72 相对 e68 的双 HOTA、双 DetA/AssA 与四项 AP 全升，cls 已过线、det 只差 `0.228`，
  同点和离严格目标只差 `0.995`，因此在最慢且只承接成熟路线的 252 固定 GPU0/1 上做
  纯时长单因素延长到 e80，e76/e80 做完整双评测。
- clean detached `5cbf438` 的隔离 checkout 完成新旧 config deepcopy 精确比较、两 launcher
  `bash -n` 与完整构建；除 `max_epochs 72→80`、独立 workdir/TrackEval 目录外配置相同，
  22,771,111 参数、711 states、增量 0。e72 的 scheduler 已结束 2000-iter warmup，
  optimizer 497 states 均在 step74736，EMA 712 states 完整，续训不重启任何状态。
- 双卡真数据 smoke 4/4 步的 loss/grad、DN、Encoder proposal 全有限，364,503,158-byte
  checkpoint 的 711 model/712 EMA states 全有限。正式 PGID `1025568` 精确恢复
  `epoch72/iter74736`，e73 iter50 的 lr/loss/grad 为 `1e-4/7.8670/45.5998`，fatal=0；
  GPU0/1 各约 19.4 GiB，GPU2/3 保持 1 MiB/0%。五门槛通过后登记 RUNNING。

## 2026-08-06 00:50 CST：178 的 0804_16 e8 完整评估

- e8 cls HOTA/DetA/AssA `42.399/34.082/55.489`、det
  `47.599/41.760/56.323`；相对强父线 e8 HOTA `-2.603/-1.484`，AssA
  `+1.492/+2.969`，但 DetA `-5.055/-4.965`，呈明显检测覆盖到关联的交换。
- pair mAP/AP50 `0.2031/0.3667`、both-independent `0.2475/0.4243`，相对父线四项约
  `-0.0304/-0.0581/-0.0384/-0.0702`。375,558,516-byte checkpoint meta `8/8304`，
  iterative-cls/DN 已训练且 642 张量有限；5416/50、28 CSV、108 非空文件、50 preds、
  `async_done=1` 和 233.2 秒 TrackEval 完整。
- e8 只登记为机制诊断，不直接否决。PGID `4175891` 已恢复 e9 iter250，仍仅占 GPU0，
  所有关键训练分量有限并继续 e12+；GPU1 外部作业未动。0804_17 保持静态未部署。

## 2026-08-06 00:34 CST：252 的 0804_01 e72 最终闭环

- e72 cls HOTA/DetA/AssA `55.170/45.881/68.469`、det
  `62.165/54.560/73.313`；cls 超 Encoder `0.733`，det 仍低 `0.228`，同点和
  `117.335` 距严格 `>118.330` 仍差 `0.995`，因此不登记为目标成功。
- 相对 e68，HOTA `+0.317/+0.282`、DetA `+0.262/+0.181`、AssA
  `+0.256/+0.382`；pair mAP/AP50 `0.3186/0.5335`、both-independent
  `0.3581/0.5709`，四项也分别提升 `0.0028/0.0016/0.0031/0.0020`。
- 463,375,734-byte checkpoint meta `72/74736`，iterative-cls/DN 已训练，642 个浮点
  张量全有限；5416/50、28 CSV、108 非空文件、50 preds、`async_done=1` 和 361.6 秒
  TrackEval 完整。训练组已自然退出，固定 GPU0/1 释放，GPU2/3 始终未碰。
- 结果支持下一个已静态闭环的单因素 `0804_17 quotient-anisotropy product-tangent`，但仍须
  等待 178 的 0804_16 e12+ 成熟交接。178 e8 checkpoint 已审计为 meta `8/8304`、
  642 张量有限，当前完成检测/TrackEval 后继续 e12+；不以 e8 淘汰。

## 2026-08-06 00:10 CST：0804_17 的 99 静态端口就绪

- clean detached `a4914d0` 的 99 隔离 checkout 已补齐 `2xb4` formal/smoke 配置与启动器；
  两配置远端加载和 `deepcopy`、两 launcher `bash -n`、精确单测 `1/1` 及父/候选完整构建
  全部通过。父/候选均为 `22,771,111` 参数、711 states、增量 0，两个目标 workdir 均不存在。
- 初次 clone 因无关 README 的缺失 LFS GIF 失败，现场保留为 `..._failed_lfs_20260806_0005`；
  `GIT_LFS_SKIP_SMUDGE=1` 重建后仓库 clean。未执行 smoke、未创建正式目录、未占 GPU，状态仍为
  `STATIC_VALIDATED/NOT_DEPLOYED/NO_GPU`，只在 178 的 0804_16 e12+ 成熟交接后进入五门槛。
- 00:08 现场为 252 固定 GPU0/1 e72 iter450、178 GPU0 e7 iter450；99 三卡空闲，178 GPU1
  外部作业与 252 GPU2/3 均未触碰。

## 2026-08-05 23:58 CST：99 的 0804_15 e12 成熟停止

- e12 cls HOTA/DetA/AssA `44.473/36.631/56.799`、det
  `52.042/46.624/60.223`，同点和 `96.515`；相对强父线 `0803_13` e12 HOTA
  `-3.816/-2.497`，距最终 Encoder 与严格总和门槛仍低 `9.964/10.351/21.815`。
- pair mAP/AP50 `0.2229/0.4053`、both-independent `0.2644/0.4612`，相对父线四项均负；
  381,022,134-byte checkpoint meta `12/12456`，iterative-cls/DN 已训练且 642 张量有限，
  5416/50、28 CSV、108 非空文件、50 preds、`async_done=1` 和 277.8 秒 TrackEval 完整。
- 该结论来自 e4/e8/e12 三个完整节点，不是早停。核验 PGID `1958429` 的 23 个成员与异步
  评测后精确 TERM，训练组、screen、评测进程归零；GPU0/1 回到 `10 MiB/0%`，GPU2 未动。
  0804_17 保持静态未部署，等待 178 的 0804_16 e12+ 合法交接；252 已到 e71 iter700。

## 2026-08-05 23:36 CST：178 的 0804_16 e4 同点双正

- `0804_16 quotient-anisotropy shape consensus` e4 cls HOTA/DetA/AssA
  `34.257/28.361/43.972`、det `37.860/34.602/43.208`；相对强父线 `0803_13` e4，
  HOTA `+1.408/+0.541`，cls DetA/AssA `+1.679/+0.051`，det `-0.346/+1.965`。
- pair mAP/AP50 `0.1505/0.2873`、both-independent `0.1963/0.3615`，相对父线四项
  `+0.0106/+0.0260/+0.0082/+0.0224`。369,970,548-byte checkpoint meta `4/4152`，
  iterative-cls/DN 已训练，642 个浮点张量全有限；5416/50、28 CSV、108 非空文件、
  50 preds、`async_done=1` 和 231.8 秒 TrackEval 完整。
- e4 只作早期正向机制证据，不直接接受或淘汰。PGID `4175891` 已恢复到 e5 iter300，
  仍仅占 GPU0 并继续 e8/e12+；GPU1 外部作业未动。99 已进入 e12 iter400，252 固定
  GPU0/1 到 e70 iter800，0804_17 保持静态未部署。

## 2026-08-05 23:10 CST：252 e68 与 0804_17 静态候选

- 252 e68 cls HOTA/DetA/AssA `54.853/45.619/68.213`，det
  `61.883/54.379/72.931`；cls 过线 `0.416`，det 仍差 `0.510`，同点和 `116.736` 距严格
  总和仍差 `1.594`。相对 e64 HOTA `+0.082/+0.247`，det DetA/AssA 与四项 AP 全升，固定
  GPU0/1 继续最终 e72，23:09 到 e69 iter500，GPU2/3 未用。
- pair mAP/AP50 `0.3158/0.5319`、both-independent `0.3550/0.5689`；457,893,430-byte
  checkpoint meta、iterative-cls/DN、642 张量有限性、5416/50、28 CSV、108 非空文件、
  50 preds、`async_done=1` 与 464.5 秒 TrackEval 完整。只读诊断 PGID `854175` 已精确清理，
  正式训练 PGID `823929` 不受影响。
- `0804_17 quotient-anisotropy product-tangent` 仅替换 0804_01 的 shape tangent：中心切线不变，
  在物理各向异性商空间运输 detail，并保留逐帧 log-area。结构零参数/state、class-agnostic、无
  reweight。178 隔离 clean detached `bd3f6fc` 已通过定向测试、两配置 deepcopy、两 launcher
  语法、父/新完整构建与 fresh 目录检查；父/新均 `22,771,111` 参数、711 states、增量 0。
  当前为 `STATIC_VALIDATED/NOT_DEPLOYED/NO_GPU`，不抢占仅用 GPU0 到 e4 iter400 的 0804_16，
  GPU1 外部作业未动。

## 2026-08-05 22:41 CST：99 的 0804_15 e8 完整评估

- e8 cls HOTA/DetA/AssA `40.836/33.809/52.106`，det `46.935/42.481/53.812`；相对强父线
  `0803_13` HOTA `-4.166/-2.148`、cls DetA/AssA `-5.328/-1.891`、det
  `-4.244/+0.458`。商空间形状共识仍主要损伤检测覆盖，仅保留 det AssA 小幅正差。
- pair mAP/AP50 `0.1916/0.3533`、both-independent `0.2391/0.4247`，相对父线四项低
  `0.041879/0.071503/0.046766/0.069786`。375,529,398-byte checkpoint、训练语义、642 张量、
  5416/50、28 CSV、108 非空文件、50 preds、`async_done=1` 与 320.5 秒 TrackEval 完整。
- e8 不作直接否决；screen/PGID `1958427/1958429` 已续入 e9，动态 GPU0/1 各约 21.4 GiB，
  继续 e12+，GPU2 未用。22:41 的并行现场为 252 固定 GPU0/1 到 e68 iter300、178 新线仅用
  GPU0 到 e2 iter450；178 GPU1 外部作业与 252 GPU2/3 未触碰。

## 2026-08-05 22:17 CST：178 成熟交接到 0804_16

- `0804_14` e12 cls HOTA/DetA/AssA `45.127/37.940/56.257`，det
  `52.260/46.927/60.484`；相对强父线 HOTA `-3.162/-2.279`，DetA/AssA 四项也全部为负。
  pair mAP/AP50 `0.2315/0.4131`、both-independent `0.2749/0.4677`，相对父线四项分别低
  `0.030039/0.051663/0.034810/0.055163`。
- 381,083,316-byte checkpoint、训练语义、642 张量、5416/50、28 CSV、108 非空文件、
  50 preds、`async_done=1` 和 254.7 秒 TrackEval 完整。e4/e8/e12 形成成熟负证据；精确 TERM
  PGID `4074056` 后 9 个成员归零、screen 消失，GPU0 连续空闲，GPU1 外部作业未动。
- clean `7a2ed53` 的 `0804_16` 在 GPU0 完成四步真数据 smoke，四步 loss/grad、DN、Encoder
  有限；364,506,868-byte checkpoint 已训练且 642 张量有限，fatal=0。fresh formal
  screen/PGID `4175889/4175891` 到 iter50：`0.9817 s/iter`、loss/grad
  `21.0161/111.8280`，GPU0 约 31.4 GiB；五门槛全部通过后登记 `RUNNING/TO_E4+`。
- 22:17 并行进度为 99 e8 iter350、252 e67 iter300；99 GPU2、252 GPU2/3 均未使用。

## 2026-08-05 21:40 CST：252 e64 与 99 e4 同 checkpoint 闭环

- 252 `0804_01` e64 cls HOTA/DetA/AssA `54.771/45.792/67.555`，det
  `61.636/54.314/72.466`；cls 过最终 Encoder `0.334`，det 仍低 `0.757`，同点和
  `116.407` 距严格门槛尚差 `1.923`。相对 e60 HOTA 双升 `+0.058/+0.096`，固定 GPU0/1
  继续 e68+，GPU2/3 未用。
- e64 pair mAP/AP50 `0.3143/0.5293`、both-independent `0.3542/0.5678`；452,412,790-byte
  checkpoint、训练语义、642 张量、5416/50、28 CSV、108 非空文件、50 preds、
  `async_done=1` 和 421.0 秒 TrackEval 完整。21:37 已到 e65 iter350。
- 99 `0804_15` e4 cls HOTA/DetA/AssA `31.348/25.622/40.778`，det
  `38.373/33.669/45.166`；相对强父线 HOTA `-1.501/+1.054`，det 的 AssA 增益伴随 DetA
  回落。pair mAP/AP50 `0.1322/0.2501`、both-independent `0.1744/0.3206`，四项均低于父线。
- 369,969,398-byte checkpoint、训练语义、642 张量、5416/50、28 CSV、108 非空文件、
  50 preds、`async_done=1` 和 252.6 秒 TrackEval 完整。e4 不直接否决，GPU0/1 已续到 e6
  iter150，继续 e8/e12+，GPU2 未用。178 同期仅用 GPU0 进入 e12 iter50；e12 checkpoint
  尚未生成。`0804_16` 仍为 `STATIC_VALIDATED/NOT_DEPLOYED/NO_GPU`。

## 2026-08-05 20:57 CST：178 的 0804_14 e8 同点闭环

- e8 cls HOTA/DetA/AssA `41.641/33.189/54.583`，det
  `47.227/40.661/57.446`；相对强父线 HOTA `-3.361/-1.856`、DetA
  `-5.948/-6.064`、AssA `+0.586/+4.092`，属于明确的检测覆盖向关联搬运。
- pair mAP/AP50 `0.197085/0.356198`、both-independent `0.240597/0.414809`，相对父线四项
  分别低 `0.036394/0.068605/0.045269/0.079677`。375,563,252-byte checkpoint、训练语义、
  642 张量、5416/50、28 CSV、108 非空文件、50 preds、`async_done=1` 和 242.9 秒 TrackEval
  完整。
- e8 不直接否决；动态 GPU0、PGID `4074056` 已续到 e9 iter250，继续 e12+，不越过 178
  单卡授权。20:56 并行进度为 99 e3 iter1000、252 e63 iter600；99 GPU2 与 252 GPU2/3 未用。

## 2026-08-05 20:36 CST：0804_16 双端静态验证完成

- `0804_16 terminal quotient-anisotropy shape consensus` 以
  `log-area + (log-aspect·cos2θ, log-aspect·sin2θ)` 表示终层物理形状，在旋转框轴交换商空间共享
  两帧形状增量；近正方形时角度方向自然退化。中心、分类、DN、loss、attention、层数和递归
  reference 不变，零参数/state、class-agnostic、无 reweight，只有终层常数逐元素开销。
- 99/178 隔离 checkout 均 clean detached `7a2ed53`；两端 `2/2` 定向测试、formal/smoke config
  deepcopy、launcher 语法、fresh workdir 与父/新完整构建全部通过。父/新模型均为
  `22,771,111` 参数和 711 state tensors，增量 0；未修改三个存活训练仓库。
- 状态严格为 `STATIC_VALIDATED/NOT_DEPLOYED/NO_GPU`，不占卡。只有成熟交接后才能重新检查动态
  卡并依次通过真 smoke、checkpoint 和 formal iter50 五门槛。20:35 三线进度为 252 e62
  iter600、178 e8 iter750、99 e2 iter850。

## 2026-08-05 20:11 CST：252 e60 同点闭环

- e60 cls HOTA/DetA/AssA `54.713/45.662/67.571`，det `61.540/54.262/72.360`；cls 过最终
  Encoder `0.276`，det 仍低 `0.853`，同点和 `116.253` 距严格门槛尚差 `2.077`。
- 较 e56 HOTA `+0.139/+0.224`，cls DetA/AssA `+0.164/-0.034`，det
  `+0.116/+0.376`；成熟曲线尚未平台，固定 GPU0/1 继续 e64+，GPU2/3 未用。
- pair mAP/AP50 `0.313658/0.528289`、both-independent `0.354370/0.567988`；
  446,932,022-byte checkpoint、训练语义、642 张量、5416/50、28 CSV、108 非空文件、50 preds、
  `async_done=1` 和 412.6 秒 TrackEval 完整。20:10 已到 e61 iter350。

## 2026-08-05 20:06 CST：99 成熟交接到 0804_15

- `0804_13` e12 cls HOTA/DetA/AssA `45.423/37.244/57.850`，det
  `52.430/46.839/60.743`，相对强父线 HOTA `-2.866/-2.109`。e4/e8/e12 三点形成成熟负证据；
  pair mAP/AP50 `0.2286/0.4179`、both-independent `0.2726/0.4724` 也全部低于父线。
- 381,030,262-byte checkpoint、iterative-cls/DN、642 张量、5416/50、28 CSV、108 非空文件、
  50 preds、`async_done=1` 和 277.3 秒 TrackEval 完整。精确 TERM PGID `1891973` 后成员
  `23→0`，screen 消失，GPU0/1 连续空闲，GPU2 不动。
- clean `491e329` 的 `0804_15` 在动态 GPU0/1 完成 4-iter DDP smoke；四步 loss/grad、DN、
  Encoder 有限，364,505,910-byte checkpoint 已训练且 642 张量有限，fatal=0。fresh formal
  screen/PGID `1958427/1958429` 到 iter50：`0.9888 s/iter`、loss/grad
  `21.4132/114.8830`，GPU0/1 各约 19.2 GiB，五门槛通过，登记 `RUNNING/TO_E4+`。
- 252 e60 checkpoint 已通过健康检查、同点评测中；178 e6 iter1000，仍只用 GPU0。

## 2026-08-05 19:40 CST：178 的 0804_14 e4 完整闭环

- e4 cls HOTA/DetA/AssA `33.145/26.651/44.064`，det `37.230/33.238/43.278`；相对强父线
  HOTA `+0.296/-0.089`，cls DetA/AssA `-0.031/+0.143`，det `-1.710/+2.035`。最近半球
  边界恢复了 fold 丢失的分类信号，但检测侧仍以 DetA 换 AssA，尚未形成同点双正。
- pair mAP/AP50 `0.1430/0.2685`、both-independent `0.1871/0.3400`；checkpoint 语义与
  642 张量有限性通过，5416/50、28 CSV、108 非空文件、50 preds、`async_done=1` 和 219.6 秒
  TrackEval 完整。e4 不直接否决，动态 GPU0、PGID `4074056` 已续到 e5 iter350，GPU1 外部
  任务不动。
- 19:40 并行进度为 99 e12 iter450、252 固定 GPU0/1 e60 iter200；99 GPU2、252 GPU2/3
  均未触碰。`0804_15` 继续严格保持静态验证、未部署、无 GPU。

## 2026-08-05 18:58 CST：0804_15 quotient log-shape 静态后继就绪

- 新后继仅在终层对旋转框 `(w,h,theta) ~ (h,w,theta+pi/2)` 的等价主轴表示做 quotient
  对齐：根据 detached reference 的最短主轴 lift 交换必要的 width/height log-size tangent，
  对称求均值后映回各帧；中心与成熟 periodic-angle 以外的所有路径不变。结构零参数/state、
  swap-equivariant、class-agnostic、无 reweight 或新增层，额外开销为常数 elementwise 操作。
- 99/178 隔离 checkout 均为 clean detached `491e329`；两端各 `2/2` 定向测试、两份 config
  deepcopy、两份 launcher `bash -n` 与父/新整模构建通过，参数/state 均为
  `22,771,111/711`、增量 0。
- 四个目标 smoke/formal workdir 均不存在，状态严格为
  `STATIC_VALIDATED/NOT_DEPLOYED/NO_GPU`。等待 99 e12 与 178 0804_14 成熟决策，不热更新或抢占
  存活训练；真正部署时仍需真实 smoke、checkpoint 与 formal iter50 五门槛。
- 18:58 三线为 252 e58 iter100、99 e10 iter100、178 e3 iter450，资源分配与授权上限一致。

## 2026-08-05 18:45 CST：252 e56 与 99 e8 同 checkpoint 闭环

- 252 `0804_01` e56 cls HOTA/DetA/AssA `54.574/45.498/67.605`，det
  `61.316/54.146/71.984`；cls 过最终 Encoder `0.137`，det 仍低 `1.077`，同点和
  `115.890` 距严格门槛仍差 `2.440`。相对 e52 HOTA `+0.260/+0.338` 且 DetA/AssA 全升，
  固定 GPU0/1 继续 e60+；GPU2/3 未用。
- e56 pair mAP/AP50 `0.313371/0.528904`、both-independent `0.353368/0.567596`；
  441,451,318-byte checkpoint 的 iterative-cls/DN 与 642 张量检查通过，5416/50、28 CSV、
  108 非空文件、50 preds、`async_done=1` 和 382.3 秒 TrackEval 完整。
- 99 `0804_13` e8 cls HOTA/DetA/AssA `41.261/34.161/52.333`，det
  `47.265/42.563/54.363`；相对强父线 HOTA `-3.741/-1.818`，定位/覆盖损伤仍是主因，但不以
  e8 直接否决，动态 GPU0/1 继续 e12+，GPU2 不动。
- e8 pair mAP/AP50 `0.199485/0.367706`、both-independent `0.244864/0.430787`；
  375,530,998-byte checkpoint、iterative-cls/DN、642 张量、5416/50、28/108、50 preds、
  `async_done=1` 和 280.1 秒 TrackEval 完整。18:45 进度为 252 e57 iter450、99 e9 iter350、
  178 `0804_14` e2 iter600，三线关键数值有限且资源边界未变。

## 2026-08-05 18:22 CST：178 e12 成熟交接到 0804_14

- `0804_12` e12 cls HOTA/DetA/AssA `47.019/37.990/60.804`，det
  `52.807/46.629/61.984`；相对强父线 e12 HOTA `-1.270/-1.732`，DetA
  `-3.445/-3.162`，AssA `+2.639/+0.174`。pair mAP/AP50 `0.2436/0.4206`、
  both-independent `0.2865/0.4717`，四项也均低于父线。e4/e8/e12 完整窗口表明中心球面中点
  以定位换关联，故成熟停止，不属于早期否决。
- 381,071,860-byte checkpoint 的 iterative-cls/DN 与 642 张量检查通过；5416/50、50 检测文件、
  28 CSV、108 非空文件、50 preds、`async_done=1` 和 241.2 秒 TrackEval 完整。精确 TERM
  PGID `3968124` 后成员 `9→0`，只释放 GPU0，GPU1 外部任务未动。
- `0804_14` 的真实 GPU0 smoke loss/grad 全有限，364,506,676-byte checkpoint 已训练且 642 张量
  有限，错误扫描 0。fresh formal screen/PGID `4074054/4074056` 到 iter50：`0.9456 s/iter`、
  loss/grad `21.0279/103.0120`，total/DN/Encoder proposal 有限；五门槛全部通过后登记
  `RUNNING/TO_E4+`。继续 e4/e8/e12+，不以早期 checkpoint 直接否决。
- 18:22 并行状态：252 固定 GPU0/1 到 e55 iter300，GPU2/3 未用；99 经正确入口到 e8 iter350，
  动态 GPU0/1 与日志一致，GPU2 未触碰；197 仍为 `STOPPED/HOST_CPU_THROTTLED`。

## 2026-08-05 17:36 CST：178 的 0804_14 动态 checkout 预置完成

- 新隔离 checkout
  `/data1/users/litianhao01/PairMOT_hemisphereboundarycenterlogshape_0804_14_178` 为 clean detached
  HEAD `6666085`，不修改存活 `0804_12` 仓库；0804_14 的 smoke/formal workdir 均不存在，
  状态严格为 `PREPARED/NO_GPU`。
- formal/smoke config deepcopy、两份 launcher `bash -n`、定向 unittest 与父/新完整构建均通过；
  参数/state `22,771,111/711`、增量 0、smoke 4 iter。首次构建检查误加载旧 editable 基仓库，
  固定隔离 `PYTHONPATH` 后通过；178 无 pytest，改用同一测试文件的 unittest 入口，不是模型失败。
- 活跃 `0804_12` 到 e11 iter500 且仅用 GPU0；必须等 e12 完整成熟证据、精确停止与连续空闲
  检查后，才允许 0804_14 真实 smoke/formal。197 17:31 仍仅 `132-147 MHz`，继续禁用。

## 2026-08-05 17:28 CST：99 0804_13 e4 同 checkpoint 闭环

- `0804_13 hemisphere-fold center + mature log-shape consensus` e4 cls HOTA/DetA/AssA
  `30.896/25.987/39.169`、det `37.806/33.666/43.476`；相对强父线 `0803_13` e4 HOTA
  `-1.953/+0.487`，分类明显受损而 det 仅微增，且弱于 178 `0804_12` 的 e4 双侧表现。
- pair mAP/AP50 `0.137848/0.264067`、both-independent `0.180727/0.333931`；
  369,969,782-byte checkpoint 的 iterative-cls/DN 与 642 浮点张量检查通过。5416/50、
  28 CSV、108 非空文件、50 preds、`async_done=1` 和 258.3 秒 TrackEval 完整。
- e4 只作诊断，不直接否决 decoder；动态 GPU0/1 继续 e8/e12+，17:28 到 e5 iter300，
  GPU2 外部任务未动。同期 252 e53 iter1000、178 e10 iter1000 均有限；`0804_14` 继续
  `STATIC_VALIDATED/NOT_DEPLOYED/NO_GPU`。

## 2026-08-05 17:15 CST：252 e52 与 178 e8 同 checkpoint 完整闭环

- 252 `0804_01` e52 cls HOTA/DetA/AssA `54.314/45.363/67.175`、det
  `60.978/53.997/71.336`，同点和 `115.292`，距严格 cls/det/总和门槛仍差
  `0.123/1.415/3.038`。相对 e48 HOTA 双升 `+0.146/+0.369`，故固定 GPU0/1 继续 e56+；
  17:15 到 e53 iter350，GPU2/3 保持未用。
- e52 pair mAP/AP50 `0.313803/0.530176`、both-independent `0.354068/0.569205`；
  435,968,374-byte checkpoint 的 iterative-cls/DN 与 642 浮点张量检查通过。5416/50、
  28 CSV、108 非空文件、50 preds、`async_done=1` 和 394.2 秒 TrackEval 完整。
- 178 `0804_12` e8 cls HOTA/DetA/AssA `43.401/32.729/60.312`、det
  `48.520/39.772/62.131`，相对强父线 e8 `-1.601/-0.563`；e4 早期双正未保持。pair
  mAP/AP50 `0.200536/0.356655`、both-independent `0.240331/0.408671`。375,558,452-byte
  checkpoint、iterative-cls/DN、642 张量、5416/50、28 CSV、108 非空文件、50 preds、
  `async_done=1` 和 233.3 秒 TrackEval 均闭环；不以 e8 直接否决，单 GPU0 继续 e12+，
  17:15 到 e10 iter150。
- 99 `0804_13` 同期到 e4 iter850，动态 GPU0/1 与 formal 日志一致，关键损失有限；GPU2
  外部任务未触碰。`0804_14` 仍为双端口 `STATIC_VALIDATED/NOT_DEPLOYED/NO_GPU`，不抢占
  当前训练资源。

## 2026-08-05 16:26 CST：0804_14 球面半空间最近投影静态就绪

- 新单因素 `0804_14 hemisphere-boundary center + mature log-shape consensus` 只处理中心
  detail 的运动反向样本：已经可行的 detail 恒等；反向 detail 去掉负纵向分量并把横向分量
  归一到原范数，从而落在球面闭半空间的最近边界；严格反平行使用确定性二维垂直方向。它避免
  `0804_13` 折返的等幅过冲，不是类别感知、reweight、gate 或 scale 扫描。
- terminal-only、零参数/state、交换等变、范数保持，DN/分类/loss/attention/层数/reference
  全不变，额外开销仅二维投影与归一化。99/178 隔离静态 checkout 分别为 clean HEAD
  `66e38e8/6666085`；定向测试、config deepcopy、launcher 语法和完整父/新构建均通过，
  参数/state `22,771,111/711`、增量 0、smoke 4 iter。
- 两端 smoke/formal workdir 均不存在，严格登记 `STATIC_VALIDATED/NOT_DEPLOYED/NO_GPU`；不抢占
  当前 99/178 训练。16:26 审计为 99 e1 iter900、178 e7 iter800、252 e51 iter250，均有限；
  先闭环 178 e8 与 252 e52，再选择合法动态端口。

## 2026-08-05 16:13 CST：99 正确端口恢复与 0804_13 五门槛启动

- 99 的正确 SSH 别名走端口 `2367`；此前端口 22 超时被错误解释为控制不可达。恢复后核实
  `0804_11` 的唯一训练 PGID `1791967` 占用动态 GPU0/1，GPU2 是外部任务；精确 TERM 后
  PGID 成员归零、screen 消失，GPU0/1 连续两轮为 `12 MiB/0%`，GPU2 未动。
- `0804_13` 在隔离 v2 checkout、clean detached HEAD `84ad131` 上完成配置 deepcopy、launcher
  语法、定向测试与父/新完整构建，参数/state `22,771,111/711`、增量 0。首次 clone 因缺失
  LFS 对象只留下失败构建证据，未用于训练；v2 通过 skip-smudge 安全部署。
- 真实 GPU0/1 四步 DDP smoke 的 total、DN、Encoder、grad 有限；364,505,654-byte checkpoint
  中 iterative-cls/DN 已训练且 642 个浮点张量全有限。fresh formal screen/PGID
  `1891971/1891973` 于 16:10:57 启动，iter50 `1.0030 s/iter`、loss/grad
  `21.4104/121.9437`，两 rank/GPU/log/有限项和致命扫描五门槛均通过，故登记
  `RUNNING/TO_E4+`。继续 e4/e8/e12+，不早停。
- 16:13 并行审计中，178 `0804_12` 仅 GPU0 到 e6 iter1000，252 `0804_01` 固定 GPU0/1
  到 e50 iter650；均有限且无致命错误。继续分别等待 e8 和 e52 的 checkpoint、检测与
  TrackEval 同点闭环，252 GPU2/3 仍未使用。

## 2026-08-05 15:53 CST：178 e4、252 e48 与 99 e16 完整闭环

- 178 `0804_12` e4 cls HOTA/DetA/AssA `34.909/27.213/47.152`、det
  `42.639/32.932/57.756`；相对强父线 `0803_13` e4 HOTA `+2.060/+5.320`，cls
  DetA/AssA `+0.531/+3.231`，det `-2.016/+16.513`。pair mAP/AP50
  `0.1459/0.2774`、both-independent `0.1905/0.3501`，相对父线四项均正。checkpoint
  369,970,612 bytes，iterative-cls residual 最大值 `0.0550922`，DN 与 642 张量有限；
  5416/50、28 CSV、108 非空文件、50 preds、`async_done=1` 和 224.3 秒 TrackEval 完整。
  e4 只作早期正归因，PGID `3968124` 已到 e5 iter750，仅用动态 GPU0，继续 e8/e12+。
- 252 `0804_01` e48 cls HOTA/DetA/AssA `54.168/45.451/66.613`、det
  `60.609/53.693/70.819`，和 `114.777`，严格三门槛仍差 `0.269/1.784/3.553`。相对
  e44 HOTA `+0.284/+0.415`，两侧 DetA/AssA 与 pair/both AP 四项均升；430,483,510-byte
  checkpoint、DN/iterative-cls、642 张量、5416/50、28 CSV、108 文件、50 preds、
  `async_done=1` 和 383.8 秒 TrackEval 全部通过。PGID `823929` 固定 GPU0/1 到 e49
  iter650，继续 e52；GPU2/3 不动。
- 99 `0804_11` e16 cls HOTA/DetA/AssA `46.103/37.228/59.905`、det
  `53.390/47.668/61.908`；自身 e12→e16 回升 `+2.003/+2.376`，但相对强父线 e16
  `50.415/57.456` 仍低 `4.312/4.066`，四项 AP 也低约
  `0.04396/0.07773/0.04755/0.07743`。386,506,998-byte checkpoint、DN/iterative-cls、
  642 张量、5416/50、28 CSV、108 文件、50 preds、`async_done=1` 与 285.9 秒 TrackEval
  完整。共享日志到 e17 iter250；控制机直连仍超时，不伪报停止，待 SSH 恢复后精确 TERM。
- 197 15:56 短命令仍需约 24–32 秒，load `16.95`，12 核抽样仅 `131–157 MHz`；GPU0–3
  空闲而 GPU4/5 有外部任务。保持 `STOPPED/HOST_CPU_THROTTLED`，不恢复旧线、不部署新线，
  不把“显存空闲”误写成主机可用。

## 2026-08-05 15:04 CST：运行态复核与 0804_13 双端口静态闭环

- 252 screen/PGID `823928/823929`、7 个成员和固定 GPU0/1 驻留一致，正式日志到 e47
  iter450，GPU2/3 为 `1 MiB/0%`；e48 checkpoint 尚未落盘，不对半成品作结论。
- 178 screen/PGID `3968121/3968124`、9 个成员和动态 GPU0 驻留一致，正式日志到 e3
  iter350，total、DN、Encoder、grad 有限。GPU1 当前为 `1 MiB/0%`，但 178 的授权是总计
  1 卡，故不并发启动任何后继；继续等待 e4/e8/e12+ 完整节点。
- 99 共享正式日志到 15:03 的 e14 iter850，说明训练仍活跃；控制机直连和新增的 252→99
  探针均连接超时。保持 `RUNNING/CONTROL_UNREACHABLE`，链路恢复后精确 TERM PGID
  `1791967` 并连续检查两张动态卡释放。
- `0804_13` 的 178 等价 `1x8` 配置、smoke 配置和两份 launcher 已加入 commit `4bf8964`；
  静态 checkout clean HEAD `4bf8964`，单测、`bash -n`、配置 deepcopy 与父/新完整构建通过，
  参数/state `22,771,111/711`、增量 0，目标 workdir 均不存在。99/178 两条部署路由都仍是
  `STATIC_VALIDATED/NOT_DEPLOYED/NO_GPU`，不得在真实 smoke、checkpoint 和 formal iter50
  五项动态门槛前写作 RUNNING。

## 2026-08-05 14:55 CST：0804_13 静态后继就绪但未部署

- 新单因素 `0804_13 hemisphere-fold center + mature log-shape consensus` 只在中心 detail
  与运动方向内积为负时反射纵向分量；内积非负和全部横向 detail 保持原样，完整范数不变。
  它直接针对 `0804_11` 的 DetA/AssA/AP 同步损伤，同时避免 rank-one 删除横向能量和每样本
  强制旋转。零参数/state、swap-equivariant、class-agnostic、无 reweight、新层、attention
  或 loss，额外开销只有二维内积与条件反射。
- commit `84ad131` 在 178 全新隔离静态 checkout
  `/data1/users/litianhao01/PairMOT_hemispherefoldcenterlogshape_0804_13_static178` clean；定向测试
  `1/1 OK`，配置 deepcopy、launcher 语法和父/新完整构建通过，参数/state
  `22,771,111/711`、增量 0，99 目标 smoke/formal workdir 均不存在。
- 状态严格为 `STATIC_VALIDATED/NOT_DEPLOYED/NO_GPU`。99 SSH 未恢复，因此未创建 99 checkout，
  未做真 DDP smoke/checkpoint/formal；只有 `0804_11` 精确停止、动态双卡连续空闲并完成五项
  门槛后才能登记 RUNNING。178 `0804_12` 同期正常到 e2 iter800，活跃仓库未热更新。

## 2026-08-05 14:43 CST：99 e12 闭环与控制链路异常

- `0804_11` e12 cls HOTA/DetA/AssA `44.100/36.004/56.845`、det
  `51.014/46.188/58.474`；相对成熟 `0803_13` e12 HOTA `-4.189/-3.525`，cls
  DetA/AssA `-5.431/-1.320`，det `-3.603/-3.336`。pair mAP/AP50
  `0.219747/0.394447`、both-independent `0.262616/0.449127`，四项相对父线也全部下降。
  自身 e8→e12 HOTA 仍回升 `+4.690/+4.971`，但 e4/e8/e12 完整窗口和强父线支配已支持
  成熟停止，不属于 e4/e8 早停。
- 381,012,982-byte checkpoint meta `12/12456`，residual 最大绝对值 `0.0877105`，
  iterative-cls/DN 与 642 个浮点张量通过；5416/50、28 CSV、108 非空文件、50 preds、
  `async_done=1` 与 279.2 秒 TrackEval 完整。
- 共享正式日志在 14:36 到 e13 iter300，但控制机和 178 到 99 的 SSH 均超时，当前无法安全
  核验/TERM PGID `1791967`。不伪报停止，也不通过共享盘热注入控制；状态暂记
  `RUNNING/CONTROL_UNREACHABLE`，链路恢复后立即精确停止并连续复核动态卡释放。
- 197 14:42 可连接但短命令约 26 秒，CPU 仍约 `129–140 MHz`、load 约 14；GPU0–3 空闲、
  GPU4 外部任务存在。继续视为 `STOPPED/HOST_CPU_THROTTLED`，不恢复或启动实验。

## 2026-08-05 14:32 CST：178 成熟接替与 252 e44 完整闭环

- 178 `0804_10 covariant-Frenet product-tangent` e12 cls HOTA/DetA/AssA
  `46.788/38.193/59.523`，det `53.443/47.724/61.963`；相对直接 product-tangent e12
  HOTA `-2.996/-2.800`，DetA/AssA 和 AP 也全部退化。该实验已完成 e4/e8/e12 三个完整
  checkpoint、检测和 TrackEval 节点，因此 TERM PGID `3856480`、成员 `9→0` 属于成熟停止，
  不是以 e4/e8 早停。e12 的 381,103,476-byte checkpoint、642 个有限浮点张量、
  5416/50、28 CSV、108 非空文件、50 preds、`async_done=1` 与 241.2 秒 TrackEval 齐全。
- GPU0 连续两轮为 `1 MiB/0%` 后，`0804_12 spherical-midpoint center + mature log-shape
  consensus` 在 clean HEAD `d85e837` 完成真实单卡四步 smoke：loss
  `21.3696/20.6405/20.9079/21.2148`、grad 全有限，364,506,164-byte `iter_4.pth`
  的 iterative-cls/DN 和 642 个浮点张量通过。fresh formal screen/PGID
  `3968121/3968124`、9 成员，iter50 `0.9509 s/iter`、loss/grad
  `21.0208/137.1233`，total、DN、Encoder 与 grad 有限、fatal 0。只用动态 GPU0；GPU1
  外部任务未动。五项动态门槛全部通过后登记 `RUNNING/TO_E4+`。
- 252 `0804_01` e44 cls HOTA/DetA/AssA `53.884/45.205/66.371`、det
  `60.194/53.522/70.058`，同点和 `114.078`；严格三门槛仍差
  `0.553/2.199/4.252`。相对 e40 HOTA 双升 `+0.434/+0.545`；pair mAP/AP50
  `0.312939/0.531436`、both-independent `0.354629/0.573133`。424,999,478-byte
  checkpoint、642 个有限浮点张量、5416/50、28 CSV、108 非空文件、50 preds、
  `async_done=1` 与 392.3 秒 TrackEval 完整。PGID `823929` 已进 e45，固定 GPU0/1
  继续 e48；GPU2/3 不动。

## 2026-08-05 13:25 CST：99 center-tangent e8 完整闭环

- `0804_11` e8 cls HOTA/DetA/AssA `39.410/32.941/49.304`，det
  `46.043/41.679/52.688`；相对成熟父线 `0803_13` e8 HOTA `-5.592/-3.040`，
  cls DetA/AssA `-6.196/-4.693`，det `-5.046/-0.666`。相对自身 e4 HOTA 则回升
  `+8.977/+8.393`，故只记录中期定位/覆盖损伤，不以 e8 直接否决，继续 e12。
- pair mAP/AP50 `0.188469/0.346839`、both-independent `0.235004/0.416619`；
  375,519,798-byte checkpoint meta `8/8304`，12 个 residual 最大绝对值 `0.0704144`，
  PairDN 与 642 个浮点张量审计通过。5416/50、28 CSV、108 个非空文件、50 preds、
  `async_done=1` 完整，TrackEval 275.8 秒。PGID `1791967` 已恢复 e9；GPU2 外部任务不动。
- 13:24 资源现场为：252 固定 GPU0/1 到 e42 iter900、GPU2/3 不用；178 仅本任务 GPU0
  到 e10 iter500、GPU1 外部任务不动；99 GPU0/1 到 e9 iter350；197 13:08 仍连接超时。
  `0804_12` 保持 PREPARED/NO_GPU，未越过动态五门槛登记 RUNNING。

## 2026-08-05 13:06 CST：178 covariant-Frenet e8 完整闭环

- `0804_10` e8 cls HOTA/DetA/AssA `41.791/35.146/52.091`，det
  `48.125/44.414/54.070`；相对直接 product-tangent e8 HOTA `-4.882/-5.797`，
  cls DetA/AssA `-4.745/-4.504`，det `-2.848/-10.066`。相对自身 e4 HOTA 则回升
  `+7.602/+11.513`，所以只登记中期负差，不以 e8 直接否决，继续 e12。
- pair mAP/AP50 `0.2068/0.3707`、both-independent `0.2538/0.4346`；375,572,468-byte
  checkpoint meta `8/8304`，12 个 residual 最大绝对值 `0.0757188`，642 个浮点张量
  全有限；5416/50、28 CSV、108 个非空文件、50 preds、`async_done=1` 完整，TrackEval
  约 234 秒。PGID `3856480` 已恢复 e9，仅用动态 GPU0；GPU1 外部任务不动。

## 2026-08-05 12:55 CST：252 e40 完整闭环；178 e8 checkpoint 就绪

- 252 `0804_01` e40 cls HOTA/DetA/AssA `53.450/45.021/65.541`，det
  `59.649/53.232/69.133`，和 `113.099`；距严格三门槛仍差
  `0.987/2.744/5.231`。相对 e36 HOTA `+0.124/+0.356`；pair mAP/AP50
  `0.3087/0.5282`、both-independent `0.3508/0.5711`，四项也小幅提高，故 PGID
  `823929` 已恢复 e41并继续 e44，固定 GPU0/1，GPU2/3 不动。
- 419,513,398-byte e40 checkpoint meta `40/41520`，iterative-cls/DN 已训练且有限，
  642 个浮点张量全有限；5416/50、28 CSV、108 个非空文件、50 preds、`async_done=1`
  完整，TrackEval payload→metrics 约 356 秒。
- 178 `0804_10` e8 checkpoint 为 375,572,468 bytes，meta `8/8304`，12 个 residual
  最大绝对值 `0.0757188`；iterative-cls/DN 和 642 个浮点张量审计通过。e8 检测仍在进行，
  不使用半成品 HOTA，也不以 e8 直接否决。

## 2026-08-05 12:27 CST：0804_12 补齐 178 单卡静态路径

- `0804_12` 已新增 178 物理 `1x8`、全局 batch 8 的正式配置、四步 smoke 配置及两份安全
  launcher，与 99 `2x4` 候选保持同模型和训练协议。隔离 checkout
  `/data1/users/litianhao01/PairMOT_sphericalmidpointcenterlogshape_0804_12_178` 为 clean
  HEAD `d85e837`；球面中点定向单测 `1/1 OK`，配置 deepcopy、完整父/新构建和 launcher
  语法通过：`22,771,111` 参数、711 states、增量 0、smoke 4 iter。
- 当前仍为 `PREPARED/NO_GPU`，未创建动态 workdir、未做 smoke/checkpoint/formal iter50，
  也未热更新存活 `0804_10` 仓库。待 `0804_10` 完成 e8/e12 成熟窗口并释放动态单卡后，
  才按五项门槛接替。
- 12:27 资源复核：252 固定 GPU0/1 到 e40 iter400，GPU2/3 不用；178 仅本线 GPU0 到
  e7 iter600，GPU1 外部任务不动；正式 total/DN/Encoder/grad 有限。

## 2026-08-05 12:15 CST：99 e4 闭环与下一候选静态准备

- 99 `0804_11` e4 cls HOTA/DetA/AssA `30.433/24.962/39.419`，det
  `37.650/32.515/44.764`；相对成熟父线 `0803_13` e4 HOTA `-2.416/+0.331`，cls
  DetA/AssA `-1.720/-4.502`，det `-2.433/+3.521`，表现为 det 的 DetA→AssA 交换并伴随
  cls 损伤。pair mAP/AP50 `0.1336/0.2565`、both-independent `0.1748/0.3227`，四项均低于
  父线。369,965,878-byte checkpoint meta `4/4152`，12 个 residual 最大绝对值
  `0.0673615`，642 个浮点张量全有限；5416/50、28 CSV、108 个非空文件、50 preds、
  `async_done=1` 和 242.9 秒 TrackEval 完整。e4 不作直接否决，PGID `1791967` 已恢复 e5，
  继续 e8/e12 及成熟节点。
- `0804_12 spherical-midpoint center + mature log-shape consensus` 已在 99 隔离 checkout
  `/data/users/wangying01/lth/PairMOT_sphericalmidpointcenterlogshape_0804_12_99` 固定 clean
  HEAD `4f6d563`。只把 center 反对称 detail 方向改为学习方向和运动方向的最短符号球面中点，
  精确保留原 detail 范数；零参数/state、交换等变、class-agnostic、无 reweight，不增加层、attention 或 loss。
  定向单测、配置 deepcopy、完整构建与 launcher 语法已通过，父/新均 `22,771,111` 参数、
  711 states。当前严格为 `PREPARED/NO_GPU`，未做真实 DDP smoke、checkpoint 与 formal iter50，
  不登记 RUNNING，等待 99 现行成熟线释放双卡后再按五项门槛接替。
- 12:15 现场复核：252 固定 GPU0/1 到 e39 iter800、GPU2/3 空闲；178 仅本线 GPU0 到
  e6 iter800、GPU1 外部任务不动；99 仅本线 GPU0/1 到 e5 iter500、GPU2 外部任务不动。
  三线正式 total/DN/Encoder/grad 有限，未触碰外部任务或热更新存活仓库。

## 2026-08-05 11:55 CST：178 e4 闭环与 197 主机保护停线

- 178 `0804_10` e4 cls HOTA/DetA/AssA `34.189/27.630/45.087`，det
  `36.612/33.119/41.820`；相对直接 product-tangent e4 HOTA `-1.085/-7.237`，其中 det
  AssA 低 `16.280`。pair mAP/AP50 `0.1436/0.2761`、both-independent
  `0.1884/0.3473`，四项也低于父线。369,974,324-byte checkpoint meta `4/4152`，12 个
  residual 最大绝对值 `0.0634495`，642 个浮点张量全有限；5416/50、28 CSV、108 个非空文件、
  50 个预测与 `async_done=1` 完整。该点不作 e4 早停，PGID `3856480` 已恢复 e5，继续 e8/e12。
- 197 原 PGID `2390925` 在 e12 step12418 后超过 39 分钟无更新且 GPU0/1 近空闲；全机 80 核
  实测仅约 `118–167 MHz`，独立 PyTorch import 超时。TERM 后成员归零、GPU0/1 为
  `1 MiB/0%`；从可信 e8 的隔离恢复也未达到 formal iter50，故停止恢复 PGID `4191760` 与
  screen `4191757`。保留 e8、step12418 scalars 与全量 e4/e8 产物，等待主机恢复；GPU4/5
  外部任务与整机 governor 均未触碰。

## 2026-08-05 11:27 CST：252 product-tangent e36 完整闭环

- e36 同一 checkpoint 的 cls HOTA/DetA/AssA `53.326/44.846/65.452`，det
  `59.293/52.962/68.706`，和 `112.619`；距严格 `54.437/62.393/118.330` 仍差
  `1.111/3.100/5.711`。相对 e32 为 `+0.017/-0.027`，但 pair mAP/AP50 升到
  `0.3057/0.5242`、both-independent 升到 `0.3479/0.5678`，故 PGID `823929` 已恢复 e37，
  固定 GPU0/1 继续 e40+，GPU2/3 不动。
- 414,028,918-byte checkpoint meta `36/37368`，12 个 residual 最大绝对值 `0.1192054`，
  642 个浮点张量全有限；5416/50 检测记录、28 CSV、108 个非空评估文件、50 个非空预测、
  `async_done=1` 完整，TrackEval 415.6 秒。
- 同步进度：178 `0804_10` 到 e4 iter350，99 `0804_11` 到 e2 iter800；197 `0804_09`
  GPU0/1 主进程仍活跃但共享盘 I/O 拥塞，未稳定产出 e12 checkpoint，保持原进程等待完整节点。

## 2026-08-05 10:58 CST：99 启动 center-tangent + log-shape consensus

- `0804_11` 保留成熟 terminal log-size/周期角共识，只在最后 normal query 增加 2D center
  product-tangent 投影；分类、DN、loss、attention、层数、递归 reference 与辅助输出不变。
  结构为单因素、零参数/state、交换等变、class-agnostic、无 reweight。隔离 checkout
  `/data/users/wangying01/lth/PairMOT_terminalcenterlogshape_0804_11_99` 固定 clean HEAD
  `eec6fc9`；定向测试、配置 deepcopy、launcher 语法及完整父/新构建通过：
  `22,771,111` 参数、增量 0、711 states。
- 动态 GPU0/1 连续两次空闲后，真数据 DDP smoke loss
  `12.9358/19.4522/19.5966/21.1587`、grad
  `105.6080/162.7252/154.4735/142.0009`；total/DN/Encoder 全有限，364,505,078-byte
  checkpoint 的 iterative-cls/DN 更新与 642 个浮点张量全有限。fresh formal screen
  `1791965.formal_0804_11_99`、PGID `1791967`，iter50 `0.9719 s/iter`、loss/grad
  `21.4245/134.0124`，7 个成员，GPU0/1 各约 19.2 GiB，fatal 0。五门槛齐全，登记
  `RUNNING/TO_E12+`；GPU2 外部任务不动，e4/e8 不作直接否决。
- 同步进度：252 固定 GPU0/1 已到 e36，197 Householder 已到 e12，178 covariant-Frenet 已到
  e2；继续优先闭环 252 e36 和 197 e12，再收集 178/99 e4/e8/e12 与成熟节点。

## 2026-08-05 10:38 CST：178/99 e12 成熟闭环与 0804_10 动态接替

- 178 `0804_07 axis-Frenet` e12 cls HOTA/DetA/AssA `48.628/40.912/59.925`，det
  `54.069/48.455/62.596`；相对直接 product-tangent e12 HOTA `-1.156/-2.174`，pair
  mAP/AP50 `0.2617/0.4600`、both-independent `0.3092/0.5180`。381,099,572-byte
  checkpoint meta `12/12456`，642 个浮点张量全有限；5416/50/28/108、50 preds、
  `async_done=1` 和 254.8 秒 TrackEval 完整。e4/e8/e12 成熟反证后 TERM PGID
  `3752856`，成员 `9→0`，GPU0 释放，产物保留。
- 99 `0804_08 shared-metric` e12 cls HOTA/DetA/AssA `44.241/36.331/56.477`，det
  `52.755/47.148/61.146`；相对直接 product-tangent e12 HOTA `-5.543/-3.488`，pair
  mAP/AP50 `0.2256/0.4012`、both-independent `0.2676/0.4553`。381,050,294-byte
  checkpoint meta `12/12456`，642 个浮点张量全有限；5416/50/28/108、50 preds、
  `async_done=1` 和 287.7 秒 TrackEval 完整。TERM PGID `1751255` 后成员 `23→0`，
  GPU0/1 已释放，GPU2 外部任务未动。
- 178 GPU0 连续释放核验后，`0804_10` 真实单卡 smoke 四步 loss
  `21.3690/20.7041/20.9412/21.2466`、grad `60.4018/65.3105/88.5642/89.0179`；
  364,507,636-byte `iter_4.pth` 的 iterative-cls/DN 与 642 个浮点张量全有限。fresh
  formal screen `3856478.formal_0804_10_178`、PGID `3856480`，iter50
  `0.9524 s/iter`、loss/grad `21.0102/93.6656`，9 个成员，GPU0 约 31.4 GiB，正式
  total/DN/Encoder 全有限且 fatal 0。连同配置 deepcopy 与完整构建，五项动态门槛齐全，
  登记 `RUNNING/TO_E12+`；GPU1 外部约 5.9 GiB 负载不触碰。
- 10:40 四机复核：252 固定 GPU0/1 到 e35且 GPU2/3 空闲；178 仅本线 GPU0；197 仅本线
  GPU0/1且 GPU5 外部任务不动；99 GPU0/1 已释放且 GPU2 外部任务不动。授权卡数未扩大。

## 2026-08-05 10:03 CST：252 e32、197 e8 与下一轻量候选

- 252 固定 GPU0/1 的 `0804_01` e32 cls HOTA/DetA/AssA
  `53.309/44.791/65.462`，det `59.320/52.822/68.981`，同 checkpoint 和 `112.629`，
  距严格三门槛 `54.437/62.393/118.330` 仍差 `1.128/3.073/5.701`。较 e28 HOTA 双升
  `+0.668/+0.334`，pair mAP/AP50 升至 `0.3037/0.5206`、both-independent 升至
  `0.3472/0.5663`，所以 PGID `823929` 已恢复 e33并继续 e36+，GPU2/3 保持未用。
  408,537,142-byte checkpoint meta `32/33216`，642 个浮点张量全有限；5416/50/28/108、
  50 个非空预测、`async_done=1` 和 393.7 秒 TrackEval 完整。
- 197 动态 GPU0/1 的 `0804_09` e8 cls HOTA/DetA/AssA
  `42.596/34.975/54.658`，det `47.448/43.295/53.888`；相对直接 product-tangent e8
  HOTA `-4.077/-6.474`。pair mAP/AP50 `0.2045/0.3731`、both-independent
  `0.2503/0.4358`。375,529,191-byte checkpoint meta `8/8304`，642 个浮点张量全有限；
  5416/50/28/108、50 个非空预测、`async_done=1` 和 321.3 秒 TrackEval 完整。按 decoder
  慢收敛规则不以 e8 停止，PGID `2390925` 已恢复 e9并继续 e12。
- 下一候选 `0804_10 covariant-Frenet product-tangent` 在 178 隔离 checkout
  `/data1/users/litianhao01/PairMOT_terminalcovariantfrenet_0804_10` 的 clean HEAD
  `15213f6` 完成静态准备：只在 detached 半程方向定义的共同 Frenet 坐标中修正反对称 center
  inconsistency，shape、分类、DN、loss、层数与 attention 不变；零转角严格退化为父结构，
  交换等变、class-agnostic、无 reweight、参数/state 增量均为 0。2 项定向测试、配置
  deepcopy、完整构建（`22,771,111` 参数、711 states）和两份 launcher 语法均通过。
  当前仅为 `PREPARED/NO_GPU`；待 178 e12 闭环后仍须真实单卡 smoke、checkpoint 与 formal
  iter50 五项动态门槛，绝不提前登记 `RUNNING`。
- 10:02 资源审计：252 只用固定 GPU0/1；178 只用动态 GPU0；99 只用动态 GPU0/1且不动
  GPU2 外部任务；197 只用动态 GPU0/1。四条正式训练进程、GPU、日志和 workdir 一致，
  total/DN/Encoder/grad 均有限且 fatal 0，未热更新任何存活训练仓库。

## 2026-08-05 09:25 CST：178/99 epoch 8 完整闭环

- 178 `0804_07 axis-Frenet` e8 cls HOTA/DetA/AssA `44.638/36.575/57.023`，
  det `51.044/44.642/60.973`；相对直接 product-tangent e8 为 `-2.035/-2.878`，
  同点和 `95.682`，严格总和仍差 `22.648`。pair mAP/AP50 `0.2308/0.4073`、
  both-independent `0.2765/0.4691`。375,568,692-byte checkpoint 的 residual/642 浮点
  张量全有限，5416/50/28/108、50 preds 与 `async_done=1` 完整，TrackEval 242.4 秒。
  相对原 decoder e8 仍双正，因此不作 e8 淘汰，继续 e12。
- 99 `0804_08 shared-metric` e8 cls HOTA/DetA/AssA `41.011/33.708/53.262`，
  det `46.607/42.601/52.615`；相对直接 product-tangent e8 为 `-5.662/-7.315`，
  pair mAP/AP50 `0.2001/0.3607`、both-independent `0.2432/0.4210`，定位、关联与 AP
  同时下降。375,547,190-byte checkpoint meta `8/8304`，residual/642 浮点张量全有限，
  5416/50/28/108、50 preds 与 `async_done=1` 完整，TrackEval 261.9 秒。仍继续 e12成熟窗。
- 09:25 进度：252 固定 GPU0/1 e31 iter400，178 本任务 GPU0 e9，99 本任务动态
  GPU0/1 e9 iter350，197 本任务动态 GPU0/1 e6 iter1000；GPU2/3（252）、GPU1（178）
  保持空闲，99 GPU2 外部任务不动。四线损失、DN、Encoder proposal 与 grad 均有限。

## 2026-08-05 08:51 CST：197 Householder e4 与四机进度闭环

- `0804_09` e4 cls HOTA/DetA/AssA `31.605/25.498/41.950`，det
  `37.701/31.744/45.689`。相对原 decoder e4 为 `-2.701/-0.889`、相对 Encoder
  为 `-4.604/-1.052`，相对直接 product-tangent 为 `-3.669/-6.148`；后者的 det
  DetA/AssA 差值为 `-2.589/-12.411`，早期问题仍集中在关联与检测 AP，而非仅范数丢失。
- pair mAP/AP50 `0.1351/0.2542`、both-independent `0.1753/0.3184`；
  369,968,615-byte checkpoint 的 12 个 iterative-cls residual 张量已训练且有限，
  642 个浮点张量全有限。5416/50/28/108、50 个预测文件与 `async_done=1` 全部闭环，
  TrackEval 用时 273.9 秒。该结果只作 e4 归因，不提前停止；继续 e8/e12。
- 08:50 四机进度为：252 固定 GPU0/1 e30 iter150；178 本任务 GPU0 e8 iter650；99
  本任务动态 GPU0/1 e7 iter650；197 本任务动态 GPU0/1 e5 iter350。所有正式日志的
  total/DN/Encoder/grad 均有限，未分配 GPU 和外部任务均未改变。

## 2026-08-05 08:34 CST：252 product-tangent e28 完整闭环

- `0804_01` e28 cls HOTA/DetA/AssA `52.641/44.468/64.147`，det
  `58.986/52.513/68.606`，同点和 `111.627`；严格三门槛仍差
  `1.796/3.407/6.703`，尚未达标。相对 e24，双 HOTA `+0.163/+0.215`、DetA
  `+0.386/+0.437`，AssA `-0.330/-0.071`。
- pair mAP/AP50 `0.3001/0.5156`、both-independent `0.3442/0.5622`，均继续高于 e24。
  403,042,998-byte checkpoint、iterative-cls/DN、642 个有限浮点张量、5416/50/28/108、
  50 个预测文件及 `async_done=1` 全部闭环，异步评测 391.2 秒。检测/AP 仍增长，故固定
  GPU0/1 继续 e32+；GPU2/3 不用于本任务。

## 2026-08-05 08:11 CST：99 shared-metric e4 闭环

- `0804_08` e4 cls HOTA/DetA/AssA `31.368/26.157/40.232`，det
  `38.646/33.488/45.640`；相对原 decoder e4 为 `-2.938/+0.056`、相对 Encoder e4
  为 `-4.841/-0.107`，相对直接 product-tangent e4 为 `-3.906/-5.203`。det AssA
  相对父结构低 `12.460`，共同 metric 的早期主要问题是关联崩落而非单纯定位误差。
- pair mAP/AP50 `0.1409/0.2654`、both-independent `0.1843/0.3349`；
  369,971,894-byte checkpoint 的 12 个 iterative-cls residual 张量与全部 642 个浮点张量
  有限且已训练，5416/50/28/108 与 `async_done=1` 全部闭环，异步评测 234.4 秒。
  e4 只作归因、不早停；PGID `1751255` 已恢复到 e5 iter250，继续 e8/e12。
- 08:10 四机进度为：252 固定 GPU0/1 e28 iter500；178 本任务 GPU0 e6 iter50；99
  本任务动态 GPU0/1 e5 iter250；197 本任务动态 GPU0/1 e3 iter100。所有外部任务及未分配
  GPU 均保持不动。

## 2026-08-05 08:01 CST：178 axis-Frenet e4 闭环与四机资源审计

- `0804_07` e4 cls HOTA/DetA/AssA `35.434/28.624/45.484`，det
  `44.417/35.116/57.988`；相对原 decoder e4 为 `+1.128/+5.827`、相对 Encoder e4
  为 `-0.775/+5.664`。相对直接 product-tangent e4，cls/det HOTA 为
  `+0.160/+0.568`，det DetA `+0.783` 而 AssA `-0.112`，显示早期定位回补但尚无成熟优势。
- pair mAP/AP50 `0.1584/0.2908`、both-independent `0.2036/0.3605`；
  369,976,116-byte checkpoint、iterative-cls/DN、642 个有限浮点张量、5416/50/28/108
  与 `async_done=1` 全部闭环，异步评测 221.4 秒。e4 仅作归因，不提前停止；PGID
  `3752856` 已恢复到 e5 iter400，继续 e8/e12。
- 08:00 审计确认：252 固定 GPU0/1 到 e27 iter1000，GPU2/3 不用；178 仅本任务 GPU0，
  GPU1 外部任务不动；99 本任务动态 GPU0/1 到 e4 iter1000，GPU2 外部任务不动；197
  本任务动态 GPU0/1 到 e2 iter450。四条正式线的进程、GPU、日志和有限损失一致。

## 2026-08-05 07:40 CST：197 Frenet 成熟收口并由 0804_09 接替

- `0804_06` e12 cls HOTA/DetA/AssA `44.810/36.342/57.539`，det
  `51.879/45.809/61.000`；pair mAP/AP50 `0.2243/0.3983`、both-independent
  `0.2672/0.4546`。相对原 decoder e12 为 `-2.585/-2.557`，相对 Encoder 为
  `-4.870/-4.662`。381,030,887-byte checkpoint、iterative-cls/DN、642 个浮点张量、
  5416/50/28/108 与 326.4 秒异步完成全部闭环。
- e4/e8/e12 三个节点持续双负后精确 TERM PGID `482699`，成员 `23→0`，六张 GPU 两轮
  均为 `1 MiB/0%`。该判定来自成熟窗口，不是 e4/e8 直接否决。
- 新 `0804_09` 只把 product-tangent 的 center/shape rank-one detail 投影替换为正交、
  范数保持的 Householder 平行传输；零参数/状态增量、交换等变、class-agnostic，无
  prediction reweight、新层、attention 或 loss。独立 clean HEAD `84fa6cc` 的定向测试、
  配置 deepcopy、完整父/候选构建和 launcher 语法通过：`22,771,111` 参数、711 states。
- 动态 GPU0/1 真实 DDP smoke loss `12.9412/19.4368/19.6046/21.1094`、grad
  `102.5677/89.2892/81.2947/90.7800`；364,505,255-byte checkpoint 的 iterative-cls/DN
  与 642 张量全有限。fresh formal screen/PGID `2390923/2390925`，iter50
  `0.9251 s/iter`、loss/grad `21.4022/108.1453`，7 成员、GPU0/1 各约 19.2 GiB、
  total/DN/Encoder 有限、fatal 0，五项门槛通过，登记 `RUNNING/TO_E12+`。
- 07:38 资源复审：252 只用固定 GPU0/1、e26 iter950；178 只用动态 GPU0、e4 iter800，
  GPU1 外部任务不动；99 只用动态 GPU0/1、e3 iter650，GPU2 外部任务不动；197 只用
  动态 GPU0/1。所有本任务卡数均在边界内。

## 2026-08-05 07:08 CST：四节点闭环与 0804_07/08 动态接替

- 252 `0804_01` e24 为 cls/det `52.478/58.771`、同点和 `111.249`，严格三门槛仍差
  `1.959/3.622/7.081`。相对原 decoder e24 `+0.769/-0.010`，相对 Encoder
  `+0.764/-0.748`；e20→e24 仍双升 `+0.280/+0.639`，AP 亦继续增长。397,549,046-byte
  checkpoint、5416/50/28/108 与 399.5 秒异步完成完整；固定 GPU0/1 继续 e28+。
- 178 `0804_04 body-frame` e12 `47.220/53.882`，相对原 decoder
  `-0.175/-0.554`；99 `0804_05 SE(2)` e12 `45.531/51.727`，相对原 decoder
  `-1.864/-2.709`。两者均在 e4/e8/e12 完整闭环后精确停止，不是 e4/e8 早停。
- 178 `0804_07 axis-Frenet` 单卡 smoke/checkpoint/formal iter50 五门槛通过，fresh
  screen/PGID `3752854/3752856`，GPU0 运行、GPU1 未动；99 `0804_08 shared-metric`
  双卡五门槛通过，fresh screen/PGID `1751253/1751255`，动态 GPU0/1 运行，GPU2 外部任务
  不触碰。两项均为零参数、非 class-aware、无 reweight 的轻量单因素 decoder。
- 197 `0804_06 Frenet` e8 `40.641/47.268`，375,534,503-byte checkpoint、
  5416/50/28/108 与 322.4 秒异步完成完整；e8 只作中期归因，继续到 e12。

## 2026-08-05 06:00 CST：0804_08 静态准备与四机复审

- 99 新隔离 checkout
  `/data/users/wangying01/lth/PairMOT_terminalsharedmetric_0804_08_99` 固定提交
  `f9b923b`，活跃 `0804_05` 仓库仍 clean。新结构统一双帧 center update、参考 chord 与
  重构所用的 geometric-mean pair-size metric；零参数/状态增量、class-agnostic、无 reweight，
  等尺寸严格退化为 `0804_01`。2 项定向测试和完整父/候选构建通过，参数均为
  `22,771,111`、711 states；当前只登记 `PREPARED/NO_GPU`，未创建 smoke/formal workdir。
- 05:58 四机复审确认：252 仅 GPU0/1 到 e22 iter250；178 仅 GPU0 到 e11 iter600；
  99 动态 GPU0/1 到 e10 iter400；197 动态 GPU2/3 到 e8 iter450。screen、PGID、GPU、
  正式日志持续一致，fatal 扫描均为 0；已有 checkpoint 仍分别截至 e20/e8/e8/e4，故继续
  等待 e24/e12/e12/e8 的 checkpoint、检测/AP 与异步 TrackEval 闭环。

## 2026-08-05 05:41 CST：252 product-tangent e20 与 99 SE(2) e8 闭环

- 252 `0804_01` e20 cls HOTA/DetA/AssA `52.198/43.979/63.843`，det
  `58.132/51.652/67.820`，同 checkpoint 和 `110.330`；严格门槛仍差
  `2.239/4.261/8.000`。相对原 decoder e20 `+1.355/+0.099`，相对 Encoder
  `+0.684/-0.790`；e16→e20 双升 `+0.978/+0.762`，四项 AP 也小幅上升。
  392,051,702-byte checkpoint、5416/50、28 CSV、108 文件和 393.1 秒异步完成标记
  完整。固定 GPU0/1、PGID `823929` 已进入 e21并继续 e24+，GPU2/3 未用。
- 99 `0804_05` e8 cls HOTA/DetA/AssA `41.230/33.612/53.634`，det
  `46.977/42.481/54.032`。相对 body-frame e8，det DetA `+0.793` 但 AssA
  `-2.168`，双 HOTA `-0.417/-0.473`；pair mAP/AP50 `0.1972/0.3621`、
  both-independent `0.2415/0.4251`。375,526,710-byte checkpoint、5416/50、
  28 CSV、108 文件和 266.9 秒异步完成标记完整。动态 GPU0/1 继续 e12，不以 e8 早停。

## 2026-08-05 05:23 CST：178 body-frame e8 与 197 Frenet e4 闭环

- 178 `0804_04` e8 cls HOTA/DetA/AssA `41.647/33.670/54.231`，det
  `47.450/41.688/56.200`；相对原 decoder e8 `-0.325/-0.728`、相对 Encoder
  `-3.622/-2.743`。pair mAP/AP50 `0.2004/0.3618`、both-independent
  `0.2436/0.4184`，375,567,924-byte checkpoint、5416/50、28 CSV、108 文件和
  253.3 秒异步完成标记完整。继续 e12，不以 e8 早停。
- 197 `0804_06` e4 cls HOTA/DetA/AssA `31.656/26.584/39.932`，det
  `36.905/32.714/42.549`；相对 SE(2) e4 的 DetA 略高，但 AssA 低
  `2.152/4.330`。pair mAP/AP50 `0.1384/0.2648`、both-independent
  `0.1825/0.3378`，369,968,743-byte checkpoint、5416/50、28 CSV、108 文件和
  305.2 秒异步完成标记完整。PGID `482699` 已进入 e5并继续 e8/e12，不以 e4 早停。
- 252 固定 GPU0/1 已进 e20，GPU2/3 仍为本任务空闲；99 动态 GPU0/1 已进 e8。
  四台资源的卡数均未扩大，178 的 `0804_07` 仍严格为 `PREPARED/NO_GPU`。

## 2026-08-05 04:40 CST：0804_07 axis-Frenet 静态准备完成

- `0804_07` 仅在强轴向 product-tangent 的中心块中，把共享 chord projector 改成由
  参考 π 周期转角确定的前/后 endpoint tangent；shape tangent 与其余模型不变，零转角
  严格退化到 `0804_01`。该选择直接针对 body-frame/SE(2) e4 的 DetA/AP 损失，同时
  保留 `0804_01` e16 联合 `+1.621` 的轴向 metric。
- 结构零参数、class-agnostic、无 reweight、无新层/attention/loss。交换等变、DN 保留、
  有限梯度、终层唯一调用与零转角退化测试通过；正式/烟测配置 deepcopy、launcher
  `bash -n` 与父/候选完整构建通过，均为 `22,771,111` 参数、711 states、增量 0。
- clean detached HEAD `fe7e9fe` 位于独立
  `/data1/users/litianhao01/PairMOT_terminalaxisfrenet_0804_07`；活跃
  `0804_04` checkout 未热更新。当前无 smoke/formal workdir、无进程，严格状态
  `PREPARED/NO_GPU`；178 继续当前 e8/e12 窗口，释放后才进入真实 smoke 和五门槛。

## 2026-08-05 04:23 CST：99 SE(2) Lie-twist e4 完整评估

- e4 cls HOTA/DetA/AssA `31.979/25.609/42.084`，det
  `37.928/31.544/46.879`；相对原 decoder 为 `-2.327/-0.662`，相对 Encoder
  为 `-4.230/-0.825`。pair mAP/AP50 `0.1290/0.2500`、both-independent
  `0.1702/0.3194`，相对 body-frame e4 双 HOTA `-2.138/-1.311`。
- 369,967,414-byte checkpoint、5416/50、28 CSV、108 个非空评测文件与 248.4 秒
  async 完成标记完整。DetA/AP 下降且没有 AssA 补偿，提示角耦合 finite-motion 的
  中心修正在早期偏强；但只登记为 e4 结构归因，不作为直接停止理由。
- 动态 GPU0/1、PGID `1715384` 已进入 e5并继续 e8/e12，正式损失、DN、Encoder 与
  grad 有限。GPU 序号不构成固定分配。

## 2026-08-05 04:15 CST：252 product-tangent e16 完整闭环

- e16 cls HOTA/DetA/AssA `51.220/43.809/61.567`，det
  `57.370/51.202/66.638`，同 checkpoint 和 `108.590`；严格绝对门槛仍差
  `3.217/5.023/9.740`，当前不能登记成功。
- 相对原 decoder e16 为 `+1.184/+0.437`、联合 `+1.621`；相对 Encoder 为
  `+0.129/-0.950`，相对 full-tangent 为 `+1.593/+0.550`。e12→e16 双 HOTA
  `+1.436/+1.127`，pair mAP/AP50 `0.2914/0.5039`、both-independent
  `0.3379/0.5551` 也四项上升。
- 386,552,438-byte checkpoint、5416/50、28 CSV、108 个非空评测文件均完整。固定
  GPU0/1 的 PGID `823929` 已进入 e17并继续 e20/e24+；该决定基于成熟上升轨迹，
  不以 e16 单点未过最终绝对线停止，GPU2/3 仍未用于本目标。

## 2026-08-05 04:12 CST：178 e4 双 HOTA 到齐；252 e16 TrackEval 继续

- 178 `0804_04` e4 cls HOTA/DetA/AssA `34.117/27.146/45.226`，det
  `39.239/34.041/46.965`；相对原 decoder e4 为 `-0.189/+0.649`，相对 Encoder
  为 `-2.092/+0.486`。pair mAP/AP50 `0.1398/0.2697`、both-independent
  `0.1870/0.3463`，5416/50 与 TrackEval 完整；这是早期 det 轻正信号，PGID
  `3652382` 已进入 e5，继续 e8/e12 和成熟节点。
- 252 `0804_01` e16 pair mAP/AP50 `0.2914/0.5039`、both-independent
  `0.3379/0.5551`；异步 TrackEval PID `833515` 存活且文件持续生成，PGID
  `823929` 已进入 e17。GPU0/1 各约 21.6 GiB，GPU2/3 均为 1 MiB；待同 checkpoint
  cls/det summary 后严格核验 `54.437/62.393/118.330` 三门槛。
- 99 `0804_05` 已到 e4 iter900；197 `0804_06` 已进入 e2 iter50。两线进程组、GPU、
  正式日志均健康且损失/DN/Encoder/grad 有限，资源使用未超过 2/2 卡。

## 2026-08-05 04:02 CST：252 e16 checkpoint 与 178 e4 AP 到齐

- 252 `0804_01` e16 训练完成，386,552,438-byte checkpoint 落盘并进入 validation；PGID
  `823929`、固定 GPU0/1 与正式恢复日志一致，total/DN/Encoder/grad 有限、fatal 为 0，GPU2/3 未用。
- 178 `0804_04` e4 完成 validation，pair mAP/AP50 `0.1398/0.2697`、both-independent
  `0.1870/0.3463`，5416/50 完整；`val_track_eval/val_track_0001` 已启动，HOTA 尚未闭环，
  当前不做结论且继续保留 e8/e12 窗口。99 `0804_05` 已进入 e4 iter50并保持 finite。

## 2026-08-05 03:54 CST：197 完成成熟交接并启动 Frenet 正式轨迹

- `0804_03` e12 cls HOTA/DetA/AssA `44.836/36.679/57.472`，det
  `51.366/46.167/59.307`；相对原 decoder e12 为 `-2.559/-3.070`。pair mAP/AP50
  `0.2332/0.4164`、both-independent `0.2674/0.4572`，381,026,151-byte checkpoint、
  5416/50、28 CSV、108 个非空评测文件与 `async_done=1` 完整。e4/e8/e12 成熟窗口持续
  双负后停止 PGID `2540932`；复核无残留成员，GPU2/3 连续两轮 `1 MiB/0%`，不是 e4/e8 早停。
- clean HEAD `2c45640` 的 `0804_06` 在动态 GPU2/3 完成真实 DDP smoke，四步 loss
  `12.9369/19.5816/19.7056/21.2506`、grad `103.5813/99.5912/91.1271/93.4536`；
  DN/Encoder 与 642 个 checkpoint 浮点 tensor 全有限，364,504,615-byte checkpoint 完整。
- fresh formal screen `482698.pm_0804_06_formal_197`、PGID `482699`；iter50
  `1.0850 s/iter`、loss/grad `21.3925/110.7751`，7 个成员、动态 GPU2/3 各约 19.2 GiB，
  总/DN/Encoder proposal 有限且 fatal 为 0。五门槛通过，状态 `RUNNING/TO_E4+`；GPU5 外部任务未动。

## 2026-08-05 03:10 CST：0804_06 Frenet product tangent 静态就绪

- 新候选仅把 body-frame center detail 的共用弦 projector 换成 constant-turn 圆弧的前/后端点切线；
  shape、分类、DN、层、attention 和 loss 不变。它交换等变，零转角退化为 body-frame，零参数、
  class-agnostic、无 reweight，终层只增加三角函数和点积。
- 197 隔离仓库 clean HEAD `2c45640`；两项定向测试、正式/烟测配置 deepcopy、launcher 语法及
  完整父/候选构建通过，参数/state `22,771,111/711`、增量 0。状态仅为 PREPARED/NO_GPU：
  未创建 smoke/formal workdir，不改动活跃 0804_03 仓库；等待 e12 全产物闭环和动态两卡释放。

## 2026-08-05 03:04 CST：0804_05 在 99 完成五门槛并进入正式轨迹

- `0804_05` 保持 0804_01 的 shape tangent，只把终层中心位移改写为由已运输角增量驱动的
  SE(2) midpoint Lie twist，再沿参考轨迹投影并匹配回缩。它零参数/状态增量、class-agnostic、
  无 reweight，不改变分类、DN、layer、attention 或 loss；只增加终层逐元素 `sinc`/三角运算。
- 99 隔离仓库 clean HEAD `f2c60a9`；三项定向测试、正式/烟测配置 deepcopy、两份 launcher
  语法和完整父/候选构建通过，参数/state `22,771,111/711`、增量 0。动态 GPU0/1 两轮空闲后，
  四步 DDP smoke loss `12.9369/19.4810/19.5814/21.1211`、grad
  `103.5530/98.0642/155.6795/142.9127`，DN/Encoder 与 642 个 checkpoint 浮点 tensor
  全有限。
- fresh formal screen `1715383.pm_0804_05_formal_99`、PGID `1715384`；iter50
  `0.9695 s/iter`、loss/grad `21.3647/108.6043`，双 rank 与动态 GPU0/1 各约 19.2 GiB，
  正式日志/目录更新且无 fatal。五门槛通过，状态 `RUNNING/TO_E4+`；GPU0/1 只是本次动态选择。

## 2026-08-05 02:53 CST：成熟迁移、两项后继启动与两项中期/成熟判定

- 178 `0804_01` e12 完整闭环为 cls/det `49.784/56.243`，DetA/AssA
  `41.865/61.515` 与 `50.021/65.603`；相对原 decoder `+2.389/+1.807`，相对 Encoder
  `+0.104/-0.298`。pair mAP/AP50 `0.2758/0.4786`、both-independent
  `0.3214/0.5310`，checkpoint、5416/50/28/108 与 async 标志完整。未严格达标但成熟优势足以
  延长，原 PGID `3555710` 在完整 e12 后停止。
- 252 从该 e12 checkpoint 在固定 GPU0/1 恢复：隔离 HEAD `f356593`、screen
  `823928.pm_0804_01_resume252`、PGID `823929`，恢复日志为 epoch12/iter12456；formal iter50
  `1.2182 s/iter`、loss/grad `9.9167/39.2891`，五门槛通过并继续 e16/e20/e24+。GPU2/3
  保持 1 MiB，严格成功仍要求同一 checkpoint 同时 `cls>54.437`、`det>62.393`、和 `>118.330`。
- 释放后的 178 动态 GPU0 对 `0804_04` 完成四步真实 smoke 与 642 tensor checkpoint 审计；
  fresh screen `3652381.pm_0804_04_formal_178`、PGID `3652382`，formal iter50
  `0.9726 s/iter`、loss/grad `21.0216/101.0078`，五门槛通过并继续 e4/e8/e12；GPU1 外部任务不动。
- 99 `0804_02` angle-only e12 `45.595/53.257`，相对原 decoder `-1.800/-1.179`；e4/e8/e12
  完整窗口均无净优势后停止 PGID `1673454`，属于成熟收口而非 e4/e8 早停。
- 197 `0804_03` log-size-only e8 `41.299/46.767`，相对原 decoder `-0.673/-1.411`；完整
  checkpoint/AP/TrackEval 闭环，但只登记中期负信号，PGID `2540932` 继续 e12，GPU5 外部任务不动。

## 2026-08-05 01:46 CST：252 成熟接力端口就绪但未启动

- 252 隔离仓库 clean HEAD `f356593` 的 `0804_01` 2x4 端口与 178 1x8 科学模型完全相等；
  配置 deepcopy、两份 launcher 语法与完整构建通过，参数/state 为 `22,771,111/711`，全局
  batch 仍为 8。
- 固定 GPU0/1 两轮空闲后，四步 DDP smoke 的 loss/grad、DN、Encoder proposal、iterative-cls
  checkpoint 语义和 642 个浮点 tensor 均有限，错误扫描为 0。状态只为
  `PREPARED/WAIT_E12`：无 formal workdir/进程，等待 178 e12 证据后才决定是否接力。

## 2026-08-05 01:36 CST：实时资源与历史任务复核

- 178/99/197 三条正式线分别在 epoch10 iter50、epoch9 iter700、epoch6 iter750 健康运行；
  GPU 使用仍为动态 GPU0、GPU0/1、GPU2/3，正式损失、DN、Encoder proposal 与 grad 有限，
  无 fatal。外部 GPU1、GPU2、GPU5 未动。
- 252 GPU0/1/2/3 均为 `1 MiB/0%`。历史 `0803_01 fresh` 与 `0801_09 e56 resume`
  均无进程或 screen；前者 last checkpoint 为 e12，后者为 e64，点名的 checkpoint 与
  TrackEval 目录均保留。252 继续固定只用 GPU0/1，当前不抢占用于弱候选筛选。

## 2026-08-05 01:31 CST：178/99 e8 完整评估与 0804_04 静态准备

- 178 `0804_01` e8 cls/det HOTA `46.673/53.922`，DetA/AssA 为
  `39.891/56.595` 与 `47.262/64.136`；相对原 decoder `+4.701/+5.744`、相对 Encoder
  `+1.404/+3.729`、相对 full-tangent `+0.390/+0.167`。pair mAP/AP50
  `0.2524/0.4443`、both-independent `0.3039/0.5139`。375,558,772-byte checkpoint、
  5416/50/28/108 与 async 标志完整；动态 GPU0、PGID `3555710` 已到 e9并继续 e12，GPU1
  外部任务不动。
- 99 `0804_02` e8 cls/det HOTA `41.415/48.105`，DetA/AssA 为
  `34.045/53.460` 与 `43.261/55.420`；相对原 decoder `-0.557/-0.073`、相对 Encoder
  `-3.854/-2.088`。pair mAP/AP50 `0.2031/0.3660`、both-independent `0.2485/0.4294`；
  375,528,630-byte checkpoint、5416/50/28/108 与 async 标志完整。动态 GPU0/1、PGID
  `1673454` 已到 e9并继续 e12；GPU2 外部任务不动，不以 e8 早停。
- `0804_04` 仅把 product-tangent 的中心块改为双帧中间朝向定义的物体局部坐标，shape、分类、
  DN 与计算深度不变；零参数、class-agnostic、无 reweight。178 独立 clean HEAD `e7ef507` 的
  来源核验、2 项定向测试、配置 deepcopy、launcher 语法和零状态增量整模构建通过：
  `22,771,111` 参数、增量 0、711 states。状态仅为 PREPARED/NO_GPU，尚未 smoke/formal。

## 2026-08-05 01:14 CST：197 log-size-only e4 完整评估

- `0804_03` e4 cls/det HOTA `31.938/38.765`，DetA/AssA 为
  `26.971/40.785` 与 `34.129/45.148`；相对原 decoder e4 `-2.368/+0.175`，相对 terminal
  mean geometry `-0.911/+1.446`，相对 Encoder `-4.271/+0.012`。pair mAP/AP50
  `0.1437/0.2727`、both-independent `0.1873/0.3435`。尺度单因素当前是 cls 慢、det 极小正增益，
  只作早期归因。
- 369,969,511-byte checkpoint、5416/50、28 CSV、108 文件与 async 标志完整。动态 GPU2/3、
  PGID `2540932` 已继续 e5→e8/e12，total/DN/Encoder/grad 有限；GPU5 外部任务未动，不以
  e4 早停。

## 2026-08-05 00:49 CST：252 transport-plane e12 成熟停止

- `0803_30` e12 cls/det HOTA `45.089/51.741`，DetA/AssA 为
  `37.490/56.626` 与 `47.123/58.773`；相对原 decoder e12 `-2.306/-2.695`，相对 terminal
  mean geometry `-3.200/-2.798`，相对 Encoder `-4.591/-4.800`。pair mAP/AP50
  `0.2287/0.4149`、both-independent `0.2721/0.4709`。
- 381,043,830-byte checkpoint、5416/50、28 CSV、108 文件与 async 标志完整。e4/e8/e12
  三个完整节点均双负后精确停止 PGID `798989`，成员 `23→0`；固定 GPU0/1 回落至 1 MiB，
  GPU2/3 未触碰。252 是最慢资源，暂不用于新的弱结构筛选。

## 2026-08-05 00:19 CST：99 periodic-angle-only e4 完整评估

- `0804_02` e4 cls/det HOTA `33.265/38.716`，DetA/AssA 为
  `27.315/42.945` 与 `34.172/45.208`；相对原 decoder e4 `-1.041/+0.126`，相对 terminal
  mean geometry `+0.416/+1.397`，相对 Encoder `-2.944/-0.037`。pair mAP/AP50
  `0.1440/0.2746`、both-independent `0.1874/0.3461`，当前是 cls 慢、det 微正的早期信号。
- 369,965,814-byte checkpoint、5416/50、28 CSV、108 文件与 async 标志完整。动态 GPU0/1、
  PGID `1673454` 已到 e5并继续 e8/e12；GPU2 外部任务未动，不以 e4 早停。

## 2026-08-05 00:07 CST：178 product-tangent e4 完整评估

- `0804_01` e4 cls/det HOTA `35.274/43.849`，DetA/AssA 为
  `28.529/45.125` 与 `34.333/58.100`；相对原 decoder e4 `+0.968/+5.259`，相对 terminal
  mean geometry `+2.425/+6.530`，但相对 full-tangent e4 `-1.068/-0.890`。分块后 det AssA
  较 full-tangent 高 `3.388`、DetA 低 `3.776`，记录为关联稳定/定位召回的早期交换，不作否决。
- pair mAP/AP50 `0.1593/0.2965`、both-independent `0.2069/0.3704`；369,973,108-byte
  checkpoint、5416/50、28 CSV、108 文件与 async 标志完整。动态 GPU0、PGID `3555710`
  已到 e5，继续 e8/e12；GPU1 外部任务未动。

## 2026-08-04 23:29 CST：197 启动 log-size-only，252 e8 完整评估

- 197 `0803_28` e12 cls/det `43.953/50.679`，DetA/AssA 分别为
  `36.592/55.255` 与 `46.150/57.353`；相对原 decoder e12 为 `-3.442/-3.757`，相对
  position/product 为 `-2.064/-2.076`。pair mAP/AP50 `0.2224/0.4035`、both-independent
  `0.2675/0.4616`，381,032,167-byte checkpoint、5416/50/28/108 与异步完成标志完整。
  e4/e8/e12 成熟窗口持续双负后精确停止 PGID `1016336`，成员 `23→0`。
- `0804_03` 仅共享终层宽高的 log-domain 乘法增量，角度、中心、分类、DN、辅助输出与递归
  reference 不变；零参数、交换等变、class-agnostic，无 reweight 或计算堆叠。clean HEAD
  `c73e19a` 的定向单测、配置深拷贝、launcher 语法和零状态增量整模构建通过。真实 GPU2/3
  smoke 四步 loss `12.9355/19.3070/19.4537/21.0277`、grad
  `106.2667/108.1143/96.9112/103.3604`，DN/Encoder 与 642 个浮点 tensor 全有限；formal
  screen `2540930.pm_0804_03_formal_197`、PGID `2540932`，iter50
  `1.0330 s/iter`、loss/grad `21.4643/103.1342`，五门槛通过。
- 252 `0803_30` e8 cls/det `40.934/47.531`，DetA/AssA 为
  `33.759/52.305` 与 `42.486/55.014`；相对原 decoder e8 `-1.038/-0.647`，相对 terminal
  mean geometry `0803_13` e8 `-4.068/-1.552`。pair mAP/AP50 `0.1955/0.3595`、
  both-independent `0.2385/0.4191`，375,538,934-byte checkpoint、5416/50/28/108 与 async
  标志完整；固定 GPU0/1 的 PGID `798989` 已进入 e9，继续 e12，不以 e8 早停。

## 2026-08-04 23:05 CST：99 position-plane 收口并启动 periodic-angle-only

- `0803_29` e12 完整结果为 cls/det `45.384/51.334`，DetA/AssA 为 cls
  `37.106/57.832`、det `46.020/59.456`；相对原 decoder e12 为 `-2.011/-3.102`。
  pair mAP/AP50 `0.2213/0.4006`，checkpoint、5416/50、28/108 和 async 标志完整；e4/e8/e12
  成熟三节点均无优势，TERM PGID `1582836` 后成员 `23→0`。
- `0804_02` 是只作用于最终 π 周期角的零参数共识，中心/尺度/分类/DN 保持原路径；clean HEAD
  `2e3fe7e` 的定向测试、deepcopy、launcher 语法和零增量整模构建通过。首次 clone 的 LFS GIF
  缺失保留为失败审计，`GIT_LFS_SKIP_SMUDGE=1` 的 `_retry1` 为有效隔离 checkout。
- 动态 GPU0/1 两轮空闲检查后，真实 smoke 四组 loss/grad、DN/Encoder 与 642 个 checkpoint
  tensor 全有限。fresh formal screen `1673453.pm_0804_02_formal_99`、PGID `1673454`，iter50
  `0.9654 s/iter`、loss/grad `21.3937/115.9512`，五门槛通过，状态 `RUNNING/TO_E4+`。

## 2026-08-04 22:44 CST：178 full-tangent 收口并启动 factorized product-tangent

- `0803_23` e52 完整结果为 cls/det `54.197/60.991`，DetA/AssA 为 cls
  `44.752/67.727`、det `53.567/71.845`；较 e48 仅 `+0.253/+0.103`，相对原 decoder e52
  为 `-0.498/-1.397`。pair mAP/AP50 `0.3087/0.5262`，checkpoint、5416/50、28/108 和
  async 标志完整。结合 e44/e48/e52 成熟平台，TERM PGID `3151184` 后成员 `9→0`。
- `0804_01` 将 5D terminal detail 分解为独立 center 2D 与 shape 3D tangent 投影，零参数、
  class-agnostic、无 reweight 或额外计算层。clean HEAD `e2f5e7d` 的定向测试、deepcopy、
  launcher 语法及父/候选整模构建通过，均为 `22,771,111` 参数、711 states。
- 动态 GPU0 两次为 `1 MiB/0%` 后完成真实四步 smoke：四组 loss/grad、DN/Encoder 与 642 个
  checkpoint 浮点 tensor 全有限。fresh formal screen `3555709.pm_0804_01_formal_178`、PGID
  `3555710`，iter50 `0.9700 s/iter`、loss/grad `20.9975/132.7340`，五门槛通过；GPU1 外部
  任务不动，状态 `RUNNING/TO_E4+`。

## 2026-08-04 21:58 CST：0803_29 e8 与 0803_30 e4 完整闭环

- 99 `0803_29` e8 cls/det HOTA `40.865/46.308`，DetA/AssA 为 cls
  `34.345/50.731`、det `41.571/53.385`。较自身 e4 回升 `+10.207/+7.906`，但相对原 decoder
  e8 仍为 `-1.107/-1.870`；pair mAP/AP50 `0.1975/0.3638`。checkpoint、5416/50、28/108
  和异步完成标志完整，动态 GPU0/1 的 PGID `1582836` 已继续 e9→e12。
- 252 `0803_30` e4 cls/det HOTA `31.119/37.046`，DetA/AssA 为 cls
  `24.852/41.221`、det `30.516/45.751`；相对原 decoder e4 为 `-3.187/-1.544`，pair
  mAP/AP50 `0.1322/0.2509`。369,970,998-byte checkpoint、5416/50、28/108 与异步完成标志
  完整；固定 GPU0/1 的 PGID `798989` 已到 e5，GPU2/3 保持空闲。
- 两个节点均只用于诊断慢收敛与分解结构，不作 e4/e8 直接否决；99 继续 e12，252 继续 e8/e12。

## 2026-08-04 16:42 CST：shape-only 成熟停止，0803_28 接替 197

- `0803_24` e12 完整结果为 `47.512/53.757`，相对原 decoder e12
  `+0.117/-0.679`；pair mAP/AP50 `0.2459/0.4382`、both-independent
  `0.2925/0.4978`。它在 e4/e8/e12 三个完整节点均未取得 det 优势，因此完成分量归因后精确
  TERM PGID `712277`，成员 `23→0`；该决定不是 e4/e8 直接否决。
- GPU2/3 连续三次和 formal 前两次均为 `1 MiB/0%`，因此按动态规则再次选择 2/3；GPU0/1
  外部任务保持不动。`0803_28` 四步 smoke loss
  `12.9435/19.4797/19.5753/21.1677`，grad
  `102.5861/118.0605/117.2151/111.1674`，DN/Encoder 与 642 个浮点 checkpoint tensor 全有限。
- fresh formal clean HEAD `1e2be85`，screen `1016334.pm_0803_28_formal_197`、PGID
  `1016336`；iter50 `1.8245 s/iter`、loss `21.3980`、grad `127.8108`，7 个成员、GPU2/3
  各约 19.2 GiB，五门槛通过。状态 `RUNNING/TO_E4+`，继续 e4/e8/e12，不在早期节点否决。

## 2026-08-04 19:09 CST：e56/e60、e36/e40、e12/e4 与 0803_29 交接

- 252 历史 `0803_01 fresh` 与 `0801_09 e56 resume` 现场确认均无进程或 screen；最后断点
  分别为 e12/e64，成熟检测与 TrackEval 产物保留。当前 252 只有固定 GPU0/1 的 `0803_13`。
  该线 e56 `54.980/62.009` 为当前最好点，绝对和 `116.989` 距严格目标差 `1.341`；e60
  回落为 `54.855/61.870`。PGID `419164` 已到 e62，保留 e64 确认后再释放；GPU2/3 不用。
- 178 `0803_23` e36/e40 为 `52.856/60.111`、`52.689/60.163`，e40 只保留 det
  `+0.052` 的小幅增长且相对原 decoder 仍双负。动态 GPU0 的 PGID `3151184` 已到 e43，
  继续到 e44 完整评测，GPU1 外部任务不动。
- 99 `0803_27` e12 `46.017/52.755`，在 e4/e8/e12 成熟轨迹持续未超过原 decoder，精确
  TERM PGID `1470665` 后成员 `23→0`。动态 GPU0/1 上 `0803_29` 四步 smoke 的四组
  loss/grad、DN/Encoder 和 642 个 checkpoint tensor 全有限；formal clean HEAD `4738c27`、
  screen `1582834`、PGID `1582836`，iter50 `0.9770 s/iter`、loss/grad
  `21.3957/109.0853`，7 个进程、每卡约 19.2 GiB、致命错误 0，五门槛通过，状态
  `RUNNING/TO_E4+`。
- 197 `0803_28` e4 `31.244/38.396`，相对原 decoder `-3.062/-0.194`，但相对
  `0803_27` 同点 `+0.821/+0.267`。完整 checkpoint、检测、28 CSV、108 文件和 TrackEval
  闭环；e4 只作早期信号，动态 GPU2/3 的 PGID `1016336` 继续 e8/e12，GPU0/1 外部任务未动。

## 2026-08-04 19:20 CST：178 后继 0803_30 静态完备

- 新候选只把 `0803_23` 终层 5D box detail 的一维 motion-tangent 投影放宽到由既有 motion 与
  detached pair-common terminal correction 张成的至多二维正交切平面；分类路径完全不启用
  position-tangent evidence。它仍为零参数/状态、终层一次、交换等变、class-agnostic，无
  reweight、新 layer、attention 或 loss。
- 178 隔离 checkout `/data1/users/litianhao01/PairMOT_terminaltransportplane_0803_30` 为 clean
  HEAD `c2069cd`。两项语义/几何 unittest、配置深拷贝、两份 launcher Bash 语法和完整模型
  构建均通过：`22,771,111` 参数、增量 0、711 states。
- 状态 `PREPARED/NO_GPU`：未创建 smoke/formal workdir，不抢占 `0803_23` 的当前单卡。
  e44 完整闭环并释放后，仍须重新核验动态空闲卡、执行真实四步 smoke 和 formal iter50 五门槛。

## 2026-08-04 19:52 CST：178 0803_23 e44 继续恢复

- e44 cls/det HOTA `53.672/60.553`，DetA/AssA 为 cls `44.220/67.433`、det
  `53.202/71.309`。相对 e40 双升 `+0.983/+0.390`；相对原 decoder e44 仍为
  `-0.743/-1.184`，绝对和 `114.225` 距严格目标差 `4.105`。
- pair mAP/AP50 `0.3052/0.5199`、both-independent `0.3463/0.5598`；checkpoint、5416
  条检测、50 序列、28 CSV、108 文件和完整 TrackEval 闭环。e44 的明显恢复不支持停止强线，
  动态 GPU0 的 PGID `3151184` 继续 e48，GPU1 外部任务不动。
- `0803_30` 继续 `PREPARED/NO_GPU`。虽然已通过全部静态门槛，但不为启动后继而无视当前 decoder
  的晚收敛恢复；待 e48 再判断一维 full-tangent 是否真正平台。

## 2026-08-04 16:05 CST：e52/e32/e4 三节点闭环

- 252 固定 GPU0/1 的 `0803_13` e52 为 `54.807/61.986`；cls 超最终 Encoder `+0.370`，det
  仍低 `0.407`，绝对和 `116.793` 距严格 `>118.330` 仍差 `1.537`。相对原 decoder e52
  为 `+0.112/-0.402`。checkpoint、50 序列、28 CSV、108 文件和 `async_done=1` 完整；原
  decoder det 峰值仍在 e56，PGID `419164` 已进入 e53，继续 e56。
- 178 动态 GPU0 的 `0803_23` e32 为 `52.627/59.424`，较 e28 `+0.656/+0.510`，相对原
  decoder e32 为 `+0.061/-0.531`。det 仍落后但两项继续上升，完整评测后 PGID `3151184`
  已进入 e33，继续 e36；GPU1 外部任务不动。
- 99 动态 GPU0/1 的 `0803_27` e4 为 `30.423/38.129`，相对原 decoder e4
  `-3.883/-0.461`。该点仅作早期结构信号，不作 decoder 否决；完整评测后 PGID `1470665`
  已进入 e5，继续 e8/e12，99 仍只使用两卡且不固定一般序号。

## 2026-08-04 15:14 CST：0803_29 终层切平面运输已静态准备

- 为缓解 `0803_23` full-tangent 成熟期的一维运输瓶颈，`0803_29` 将终层 5D box detail 投影到
  “已建立帧间运动 + 当前双帧共同终层修正”张成的至多二维正交切平面；分类侧沿用 position-tangent
  evidence。该结构零参数、零状态、终层一次、交换等变、DN prefix 原样保留，不含 class-aware、
  reweight、新 attention/layer/loss。
- 99 CPU 隔离仓库 `/data/users/wangying01/lth/PairMOT_tangentplane_0803_29_99_retry1` 固定 clean
  HEAD `4738c27`。两项远端 py310 定向测试通过，覆盖终层调用、参数/state 等价、交换等变、DN
  保留、投影能量不增和有限梯度；正式/smoke 配置深拷贝、launcher Bash 语法及整模构建通过，
  模型 `22,771,111` 参数、增量 0、711 状态张量。
- 状态仅为 `PREPARED/NO_GPU`：未创建 smoke/formal workdir、未排队、不占卡。99 仍只运行
  `0803_27` 的动态双卡；`0803_29` 等成熟结果和资源释放后再排序。

## 2026-08-04 14:55 CST：四线重排与新候选接替

- 252 固定 GPU0/1 的 `0803_13` e48 为 `54.533/61.587`：cls 严格超过最终 Encoder
  `0.096`，det 仍低 `0.806`，总和 `116.120` 距严格 `>118.330` 仍差 `2.210`。完整
  checkpoint、检测、TrackEval 与 async 标记已核验；因原 decoder 的 det 峰值在 e52–e56，
  PGID `419164` 继续 e52，GPU2/3 不用于本任务。
- 178 动态单卡 `0803_23` e28 为 `51.971/58.914`，相对 e24 `-0.041/+0.363`；det 仍改善，
  因此 PGID `3151184` 继续 e32，不用分类平台单点否决慢收敛 decoder。
- 99 `0803_25` e12 `46.196/51.938` 在 e4/e8/e12 三个完整节点持续双负后精确停止，成员
  `23→0`。动态空闲 GPU0/1 完成 `0803_27` 真数据 smoke；fresh formal clean HEAD
  `aea3157`、PGID `1470665` 的 iter50 loss/grad `21.4337/92.6051`，DN、Encoder proposal、
  GPU 与错误扫描五门槛全部通过，状态 `RUNNING/TO_E4+`。99 只限制两卡数，不固定序号。
- 197 `0803_24` e8 `41.910/47.783` 后继续 e12，不以 e8 直接否决。零参数 `0803_28`
  position-tangent classification + full 5D transport 已迁移到
  `/data/users/litianhao/PairMOT_positiontransport_0803_28_197` clean HEAD `1e2be85`，两份 launcher
  语法通过且未创建 smoke/formal workdir，状态 `PREPARED/NO_GPU`；待 e12 闭环后再动态选两张
  空闲卡接替，不抢占当前 GPU2/3 或 GPU0/1 外部任务。

## 2026-08-03 23:06 CST：资源边界纠正

- 用户明确 252 只可使用 GPU0/1。原 `0803_12` PGID `4189798` 在 GPU2/3 的 23 个成员已全部退出，四卡显存均回落到 1 MiB；中断发生在 epoch 7，最近可恢复正式断点为 `epoch_4.pth`。
- `0803_12` 启动器默认卡号已改为 GPU0/1，并增加显式 `PAIRMOT_RESUME=1` 恢复路径；恢复后须重新通过进程、显存、正式日志、iter50 和有限损失五门槛。
- `0803_14` PGID `77558` 的 7 个成员此前已停止，GPU0/1 释放且无正式 epoch checkpoint；smoke 与 iter50 证据保留。澄清后迁移到 99 空闲 GPU1/2，不与 252 的固定双卡任务并行。
- 23:07 在固定 GPU0/1 启动恢复会话，PGID `123974`；23:08 正式日志到 epoch5 iter50，GPU0/1 各约 19.4 GiB，GPU2/3 各 1 MiB，loss `11.9163`、grad norm `31.7400`，DN 与 encoder proposal loss 均有限，错误扫描为空，状态恢复为 `RUNNING/TO_E12`。

## 2026-08-03 23:12 CST：0803_13 epoch 4 与资源实测

- 0803_13 e4 全量 TrackEval 完成：cls HOTA/DetA/AssA `32.849/26.682/43.921`，det `37.319/34.948/41.243`；相对 Encoder e4 HOTA `-3.360/-1.434`。检测诊断为 pair mAP/AP50 `0.1399/0.2613`、both-independent `0.1881/0.3391`。按 decoder 晚收敛约束继续 e8/e12。
- 四机实测：252 只占固定 GPU0/1；178 当前占 1 张 GPU0；197 当前占 2 张 GPU4/5；99 GPU0 为外部进程而 GPU1/2 空闲。只有 252 固定序号，故 99 GPU1/2 可形成合法双卡窗口。

## 2026-08-03 23:15 CST：非 252 资源只限制卡数

- 用户再次澄清：只有 252 指定 GPU0/1；99/178/197 分别只限制总计 2/1/2 卡，不限定序号。此前关于 99 GPU2 未授权及必须等待 GPU0 的判断撤销。
- 当前分配继续合法：178 使用 GPU0 一张、197 使用 GPU4/5 两张。99 的 GPU1/2 均空闲，可迁入 `0803_14`；迁移后仍需在 99 重新通过真实 DDP smoke 和 formal iter50 五门槛。

## 2026-08-03 23:22 CST：99 GPU1/2 接管 0803_14

- 99 隔离仓库 `/data/users/wangying01/lth/PairMOT_terminalarea_0803_14_99` 固定提交 `ed7823d`。正式与 smoke 配置完成 99 数据/GMC/TrackEval 路径迁移并通过加载、深拷贝、Bash 语法和零参数构建检查。
- GPU1/2 真数据 smoke 完成：loss `12.9438/19.3917/19.5249/21.1545`，grad norm `106.5528/91.3914/80.1813/83.8108`，642 个浮点 checkpoint 张量全有限。
- formal PGID `1327092` 在 epoch1 iter50 为 `0.9957 s/iter`、loss `21.4082`、grad norm `149.8043`；GPU1/2 各约 19.2 GiB，错误扫描为空，五门槛通过。GPU0 的外部进程保持不动。

## 2026-08-03 23:28 CST：178 后续候选 0803_16 已设计

- `0803_16 terminal normalized-center` 仅在最终输出共享 reference-local 中心增量，保留早中层递归 reference 与尺寸、角度、分类、DN 的逐帧自由度；零参数、非 class-aware、无 reweight 或额外计算层。
- 178 formal/smoke 配置与零增量检查已加入隔离提交 `c05cd21`；仓库 `/data1/users/litianhao01/PairMOT_terminalcenter_0803_16` 的终层单次调用测试和完整构建通过，模型 `22,771,111` 参数、增量 0、711 状态张量。排在 `0803_15 terminal angle-only` 之后，不占用 `0803_13` 的一张 GPU。

## 2026-08-03 23:40 CST：178 后继启动器就绪

- `0803_15/16` 的单卡 smoke/formal 启动器已加入并通过 Bash 语法检查；均含安全 Conda 激活、fresh 目录、防错误扫描以及 checkpoint 分类语义和全浮点有限性检查。
- 当前没有启动任何后继 workdir；GPU0 继续只运行 `0803_13`。资源释放后先执行 `0803_15` smoke，五门槛通过才允许 formal。
- `/data1/users/litianhao01/PairMOT_terminalangle_0803_15` 与 `/data1/users/litianhao01/PairMOT_terminalcenter_0803_16` 均已在无活动进程时快进到 clean 提交 `e9f56dc`，远端四个启动器再次通过 Bash 语法检查。

## 2026-08-03 23:52 CST：178 第三后继 0803_17 已准备

- `0803_17 terminal semantic margins` 保留每帧分类残差的 class mean，只在最终层平均双帧
  centered class margins；DN、早中层分类递推和 box 路径不变。结构为零参数、类别置换等变，
  无 class-aware、reweight、额外 attention/layer。
- 隔离仓库 `/data1/users/litianhao01/PairMOT_terminalmargin_0803_17` 固定 `e245127`；远端来源
  核验、3 项定向测试、整模构建和两个启动器语法检查通过，模型 `22,771,111` 参数、增量 0、
  711 状态张量。状态 `PREPARED`，排在 `0803_15/16` 后且未占 GPU。
- 23:50 四机仍各只运行既有合法分配；252 固定 GPU0/1，其余三机只遵守 2/1/2 卡总量，
  未抢占任何外部进程。

## 2026-08-04 00:27 CST：178 0803_13 epoch 8

- cls HOTA/DetA/AssA `45.002/39.137/53.997`，det `49.083/46.725/53.354`；相对原始
  `0801_09` 同点 HOTA 双升 `+3.030/+0.905`，相对 Encoder 同点仍低 `0.267/1.110`。
- pair mAP/AP50 `0.233479/0.424803`、both-independent `0.285866/0.494486`；checkpoint、
  50 序列、28 CSV、108 非空文件与 TrackEval `async_done=1` 完整。
- 该双正交增益足以保留到 e12 和更晚节点；178 不释放，`0803_15/16/17` 继续 PREPARED。

## 2026-08-04 00:42 CST：197/252 几何调度对照 epoch 8

- 197 `0803_11` e8 cls/det HOTA `40.377/45.730`，相对原始 decoder 同点
  `-1.595/-2.448`；DetA/AssA 与四项 AP 同向落后。
- 252 `0803_12` e8 cls/det HOTA `40.430/46.542`，相对原始 decoder同点
  `-1.542/-1.636`；四项 AP 同样全部落后。两项均有 checkpoint、50/28/108 与 TrackEval
  完整证据。
- 不按 e8 直接停止：197 继续使用当前两卡到 e12，252 继续固定 GPU0/1 到 e12；不抢占外部
  进程，也不新增晚两层或渐进几何派生。

## 2026-08-04 00:45 CST：99 0803_14 epoch 4

- cls HOTA/DetA/AssA `30.813/24.668/40.889`，det `36.985/31.609/44.441`；相对 Encoder
  同点 `-5.396/-1.768`，相对 0803_13 同点 `-2.036/-0.334`。
- pair mAP/AP50 `0.129617/0.246465`、both-independent `0.170361/0.313179`，50/28/108
  与 TrackEval 完整。只登记早期差距，99 GPU1/2 继续 e8/e12，不影响 GPU0 外部进程。

## 2026-08-04 00:55 CST：99/197 语义后继已准备

- 99 隔离仓库 `/data/users/wangying01/lth/PairMOT_terminalmargin_0803_17_99` 固定 `ac02fc2`；
  `0803_17` 2xb4 配置、smoke、零增量整模比较与启动器语法通过，模型 22,771,111 参数、
  711 状态张量。等待 `0803_14` e12 释放 GPU1/2，当前不占卡。
- 197 隔离仓库 `/data/users/litianhao/PairMOT_terminalgeommargin_0803_18_197` 固定 `ac02fc2`；
  组合 terminal log-size/周期角和 centered semantic margins，整模仍零参数/零状态增量。
  formal/smoke 与启动器检查通过，等待 `0803_11` e12 释放当前两卡。
- 上述准备不改变资源边界：99/197 各最多 2 卡且不固定序号，252 仍只固定 GPU0/1，178
  仍只用 1 卡；未抢占外部进程。

## 2026-08-04 01:45 CST：178 0803_13 epoch 12

- cls HOTA/DetA/AssA `48.289/41.435/58.165`，det `54.539/49.791/61.810`；相对原始
  `0801_09` 同点 `47.395/54.436` 仍双升 `+0.894/+0.103`，但联合优势由 e8 的 `+3.935`
  收窄至 `+0.997`。相对 Encoder 同点仍低 `1.391/2.002`。
- pair mAP/AP50 `0.261539/0.464763`、both-independent `0.309710/0.522863`；
  381,093,940-byte checkpoint、50 序列、28 CSV、108 非空文件及 `async_done=1` 完整。
- terminal-only 几何仍保持对原始 decoder 的双正收益，但优势正在衰减；按 decoder 晚收敛约束
  保持 PGID `3062903` 到 e16，检验是否会像全层几何分支一样反转，不提前释放 178 单卡。

## 2026-08-04 01:55 CST：0803_19 terminal full-tangent 已准备

- 新候选只在最终 normal-query 输出统一三种自然几何坐标：中心在各帧 reference-local 坐标中、
  尺寸在 log-ratio 空间中、角度在 π 周期切空间中取双帧平均；早中层递归 reference、分类与 DN
  保持逐帧独立。结构零参数、class-agnostic、无 reweight、新 attention/layer/loss。
- 178 隔离仓库 `/data1/users/litianhao01/PairMOT_terminalfulltangent_0803_19` 固定 `dc0e958`；
  终层三投影各调用一次的定向测试、两个启动器 Bash 语法和零状态增量整模比较通过：模型
  22,771,111 参数、增量 0、711 状态张量。当前 `PREPARED`，未创建 workdir、未占 GPU。

## 2026-08-04 01:57 CST：99 0803_14 epoch 8

- cls HOTA/DetA/AssA `41.384/34.365/52.257`，det `47.315/42.578/54.312`；相对原始
  decoder 同点 `41.972/48.178` 为 `-0.588/-0.863`，相对 0803_13 同点低 `3.618/1.768`。
- pair mAP/AP50 `0.206968/0.367917`、both-independent `0.251401/0.427578`；
  375,534,774-byte checkpoint、50 序列、28 CSV、108 非空文件和 `async_done=1` 完整。
- 终层只共享面积未复现完整 log-size 的收益；仍按慢收敛约束保持 PGID `1327092` 到 e12，
  GPU0 外部任务不动。e12 成熟后若仍弱，释放当前两卡给已准备的 `0803_17`。

## 2026-08-04 02:05 CST：252 0803_12 epoch 12 成熟停止

- cls HOTA/DetA/AssA `45.677/36.931/59.103`，det `52.131/46.532/60.474`；相对原始
  decoder 同点 `47.395/54.436` 为 `-1.718/-2.305`，相对 Encoder 同点低 `4.003/4.410`。
- pair mAP/AP50 `0.227110/0.412989`、both-independent `0.269190/0.466195`；
  381,003,318-byte checkpoint、50 序列、28 CSV、108 非空文件和 `async_done=1` 完整。
- e4/e8/e12 三个完整节点构成持续双负成熟轨迹。核验后精确 TERM PGID `123974`，成员
  `23→0`；252 GPU0/1/2/3 均回落到 `1 MiB/0%`。252 作为最慢资源保持空闲，只等待后续
  已在快速通道证明的成熟候选，不用于新结构筛选。

## 2026-08-04 02:10 CST：197 0803_20 组合后备已准备

- `0803_20` 把 0803_19 的终层 reference-local 中心、log-ratio 尺寸、π 周期角几何一致化与
  0803_17 的 centered semantic margins 组合；每帧 class mean、早中层 reference 与 DN 独立。
  零参数、类别置换等变、无 class-aware/reweight、新 attention/layer/loss。
- 197 隔离仓库 `/data/users/litianhao/PairMOT_terminalfulltangentmargin_0803_20_197` 固定
  `f179249`；formal/smoke 启动器语法与整模零增量比较通过：22,771,111 参数、711 状态张量。
  当前 `PREPARED`，排在 0803_18 后，不建 workdir、不占 GPU。

## 2026-08-04 02:50 CST：197 0803_11 e12 停止

- e12 cls/det HOTA `45.409/50.665`，相对原始 decoder 同点 `-1.986/-3.771`，相对 Encoder
  同点 `-4.271/-5.876`；pair mAP/AP50 `0.225726/0.405185`，both-independent
  `0.270304/0.461162`。
- checkpoint、50 序列、28 CSV、108 个非空文件与异步完成证据齐全；e4/e8/e12 成熟持续双负后
  精确停止 PGID `53708`，成员 `23→0`，GPU4/5 释放，断点保留。

## 2026-08-04 02:58 CST：197 0803_18 formal 运行

- 首个空闲检查包装器因 CSV 解析错误在创建 workdir 前退出且未占 GPU；修复后连续两次确认
  GPU4/5 空闲。四步真数据 smoke loss/grad、642 个浮点 checkpoint 张量、iterative-cls/DN
  语义和错误扫描全部通过。
- fresh formal 固定 clean HEAD `ac02fc2`，screen `387856.pm_0803_18_formal_197`、PGID
  `387859`；iter50 `1.7440 s/iter`、loss `21.3900`、grad `107.1625`，7 个进程，GPU4/5
  各约 19.2 GiB，五门槛通过。状态 `RUNNING`，收集 e4/e8/e12 及后期节点。

## 2026-08-04 03:00 CST：178 0803_13 e16

- e16 cls/det HOTA `50.415/57.456`，相对原始 decoder 同点仍为 `+0.379/+0.523`，相对 Encoder
  同点为 `-0.676/-0.864`。pair mAP/AP50 `0.275158/0.486134`、both-independent
  `0.320648/0.537430`，checkpoint 与 50/28/108 评估产物完整。
- 双正优势虽收窄但未反转，保持 PGID `3062903` 到 e20；不为排队候选提前释放 178 单卡。

## 2026-08-04 03:05 CST：99 0803_14 e12 停止

- e12 cls/det HOTA `46.987/52.992`，相对原始 decoder 同点 `-0.408/-1.444`，相对 Encoder
  同点 `-2.693/-3.549`；pair mAP/AP50 `0.241566/0.426861`、both-independent
  `0.286042/0.482380`，checkpoint 与 50/28/108 评估产物完整。
- e4/e8/e12 均未形成双正后精确停止 PGID `1327092`，成员 `23→0`；GPU1/2 释放，GPU0
  外部任务未受影响。

## 2026-08-04 03:09 CST：99 0803_17 formal 运行

- GPU1/2 连续两次空闲检查后，四步真数据 smoke 的 loss/grad、364,502,518-byte checkpoint、
  642 个浮点张量、iterative-cls/DN 语义与错误扫描全部通过。
- fresh formal 固定 clean HEAD `ac02fc2`，screen `1357907.pm_0803_17_formal_99`、PGID
  `1357909`；iter50 `0.9994 s/iter`、loss `21.3978`、grad `110.7768`，7 个进程，GPU1/2
  各约 19.2 GiB，五门槛通过。状态 `RUNNING`，收集 e4/e8/e12 及后期节点。

## 2026-08-04 03:16 CST：99 后继 0803_21 已预留

- 新结构只保留终层 centered class-margin detail 在前序累计 class-ranking 双帧差方向上的投影；
  transport detach，保留每帧 class mean、pair mean 和 DN absolute 路径。零参数、类别置换/帧交换
  等变，无 class-aware、reweight 或额外层。
- 99 formal/smoke 配置、构建审计和安全启动器已完成本地语法检查。隔离 checkout
  `/data/users/wangying01/lth/PairMOT_terminaltransport_0803_21_99` 固定 clean `a7b37ef`；既有
  py310 无 pytest，未修改环境并改用标准 unittest，3 项定向测试通过。配置加载/深拷贝、远端
  launcher 语法和整模比较通过：22,771,111 参数、零增量、711 tensors。
- 状态 `PREPARED/NO_GPU`；未创建 smoke/formal workdir，不抢占当前 0803_17。

## 2026-08-04 03:26 CST：197 后继 0803_22 已预留

- 0803_22 保持 0803_18 的终层 log-size/周期角，只把完全平均的 centered margins 替换为沿前序
  class-ranking 方向投影的 transported margins；零参数、类别置换/帧交换等变，无 class-aware、
  reweight 或额外层。
- 197 formal/smoke 配置、零状态构建审计与安全启动器已完成本地语法检查。隔离 checkout
  `/data/users/litianhao/PairMOT_terminalgeomtransport_0803_22_197` 固定 clean `41c08d8`；3 项
  transported-margin 测试、配置加载/深拷贝、远端 launcher 语法和整模比较全部通过：
  22,771,111 参数、零增量、711 tensors。状态 `PREPARED/NO_GPU`；未建 workdir，不抢占
  当前 0803_18。

## 2026-08-04 03:42 CST：178 后继 0803_23 已预留

- 0803_23 在 center-local、log-size、π 周期角联合切空间中保留 pair-common 末层更新，并只让
  frame detail 沿早中层 reference 已形成的相对变换继续，抑制正交几何抖动；transport 显式
  detach，DN 不变。零参数、class-agnostic、frame-swap 等变，无 class-aware/reweight/新层。
- 178 隔离 checkout `/data1/users/litianhao01/PairMOT_terminaltransporttangent_0803_23` 固定 clean
  `d6af6d32`；两项定向测试、正式/smoke 配置整模构建、零状态比较和 launcher 语法通过：
  22,771,111 参数、零增量、711 tensors。状态 `PREPARED/NO_GPU`；等待 0803_13 e20 成熟决策，
  不抢占当前唯一一张 PairMOT GPU。

## 2026-08-04 04:13 CST：178 0803_13 epoch 20

- cls/det HOTA `51.791/58.526`，相对原始 decoder 同点 `+0.948/+0.493`，联合 `+1.441`；相对
  Encoder 同点为 `+0.277/-0.396`。e20 的父线双正优势较 e16 回升。
- pair mAP/AP50 `0.288615/0.506941`、both-independent `0.333302/0.555375`；392,138,804-byte
  checkpoint、5416 条检测、50 序列、28 CSV、108 非空文件与异步完成标记完整。
- PGID `3062903`、9 个训练成员和 GPU0 占用正常，已进入 e21。依据 e8/e12/e16/e20 四个完整
  双正节点继续到 e24；0803_15/16/19/23 均保持 PREPARED/NO_GPU，不抢占当前强轨迹。

## 2026-08-04 04:25 CST：99 0803_17 epoch 4

- cls HOTA/DetA/AssA `32.203/26.308/42.066`，det `37.822/32.233/45.135`；相对原始 decoder
  e4 HOTA 为 `-2.103/-0.768`，相对 Encoder e4 为 `-4.006/-0.931`。
- pair mAP/AP50 `0.140801/0.262022`、both-independent `0.183663/0.330727`；369,970,486-byte
  checkpoint、50 序列、28 CSV、108 非空评估文件和异步完成标记完整。
- 原等待器漏用双卡验证目录的零填充 `epoch_03`，已按实际权威目录核验并纠正后续监控；训练本身
  正常进入 e5。该点不作 e4 否决，继续 e8/e12 后再做成熟判定。

## 2026-08-04 04:53 CST：252 0803_13 成熟迁移端口通过

- 端口只把 178 1x8 的物理放置改成 252 固定 GPU0/1 的 2x4；有效 model、optimizer、scheduler、
  train loop、hooks 严格相等，全局 batch 仍为 8。首次检查发现额外显式 False 开关后先修正，
  未在不一致状态占用 GPU。
- 隔离 checkout clean `bec1a1c`，整模 22,771,111 参数、零增量、711 tensors，launcher 语法
  通过。GPU0/1 两次空闲后真数据 DDP smoke 四步 loss/grad 全有限，364,501,750-byte checkpoint
  的 642 个浮点张量和 iterative-cls/DN 语义检查通过，退出后两卡均 `1 MiB/0%`。
- 状态 `PREPARED/NO_FORMAL_GPU`。只有 178 e24 完整评估继续双正，才精确停止原 PGID 后从
  epoch24 在 252 单点恢复；禁止两个训练进程并发写共享 workdir。

## 99 本机

代码路径：`/data/users/wangying01/lth/PairMOT/ai4rs`。正式训练通常使用GPU 0、1；`0723_01`
按用户指令例外使用GPU 2、3，不设置温度watchdog或自动暂停限制。

| Status | 实验 | 开始时间 | 结束时间 | 类型/主要改动 | 进度或说明 |
| --- | --- | --- | --- | --- | --- |
| STOPPED | `0731_28_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_terminalcentermotionfactorizedevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 22:48 | 2026-08-01 03:47 | 在 `0731_21` 上把反对称 box detail 限定为旋转框中心 `x/y`；`w/h/angle` 保持父模型几何，classification common 不变 | e16 cls HOTA/DetA/AssA `48.845/39.751/62.295`，det `57.176/48.934/69.124`；相对 Encoder 同点 HOTA `-2.246/-1.144`、DetA `-2.853/-2.780`。pair mAP/AP50 `0.2635/0.4778`，相对父轨迹 `-0.0204/-0.0145`；both-independent `0.3017/0.5115`，相对父轨迹 `-0.0206/-0.0145`。e12/e16 连续双 HOTA 下降且 e16 HOTA、DetA、AP 全面恶化；epoch 16 checkpoint、检测 metrics、TrackEval 与 108 个评估文件核验后精确停止，GPU0/1 已释放 |
| STOPPED | `0731_24_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_terminalconfidentcommonfactorizedevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 18:49 | 2026-07-31 22:35 | `0731_21` 因子结构的 common 路由乘 detached 双帧分类置信度；不新增参数 | e12 `48.271/56.179`，相对 Encoder 同点 `-1.409/-0.362`；DetA `-1.834/-1.524`，四项 AP 下降 `0.014159/0.007145/0.014924/0.007281`。checkpoint、metrics 与 54 个 TrackEval 原始文件验证后停止 |
| COMPLETED | `0727_12_paper_base_liquid_encoder_p5temporal_crossscalebudget` | 2026-07-27 20:18 | 2026-07-28 18:20 | 严格继承`0727_01`的Base+Liquid、P5 temporal MHA及common/detail Dual-Evidence；用每层`[common mean, abs(detail) mean]`生成三尺度token，并结合三尺度均值上下文预测逐通道common/detail尺度预算。预算在P3/P4/P5维softmax后乘3，每个分支/通道总预算严格为3，仅重分配尺度贡献，不改变平均残差强度；描述侧停止梯度，输出层零初始化，无额外loss或高分辨率卷积 | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 60为 `54.217/61.875`，同点 pair mAP/AP50 `0.316913/0.534141`，both-independent `0.353674/0.563208`。未超过 Encoder 最终 `54.437/62.393`，不进入decoder主线；进程已退出 |
| COMPLETED | `0723_05_pairdn_paircoherent_le180_cpdse_local` | 2026-07-24 03:56 | 2026-07-25 03:56 | `0723_01` + consistency-preserving DSE local：保留`x.mean`主路径，以归一化通道离散度生成零初始化、最大绝对值0.5的逐像素SE-logit残差；8个新增参数，无额外loss | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 68为`53.536/61.619`，同点pair mAP/AP50为`0.3148/0.5320`；相对Paper Base为`+0.222/-0.363`，不满足双提升。PairMOT训练及评测进程已退出，99不排新任务；不对其他用户当前GPU占用做清理或调度 |
| COMPLETED | `0723_01_paper_liquid_independent_diffproduct_pairdn_paircoherent_le180` | 2026-07-23 03:21 | 2026-07-24 03:25 | accuracy-fixed `0718_01` + PairDN重构：pair两侧共享相对噪声，正负比2:1，正样本加难，负样本按旋转IoU筛选；DN padding隔离；保持`width_longer=True,start_angle=0`并使用等价框一致、固定角度权重的L1 | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 64为`53.955/62.032`，同epoch pair mAP/AP50为`0.3114/0.5268`；相对Paper Base双提升`+0.641/+0.050` |
| COMPLETED | `0721_05_liquid_dse_diffproduct_accuracyfix` | 2026-07-21 23:57 | 2026-07-22 21:25 | accuracy-fixed `0718_01` + DSE；以identity初始化的grouped `1x1`融合Conv3D响应的channel mean/RMS，增强稀疏目标证据；启动时仍使用修复前的Negative-DN噪声 | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `54.635/61.895`，同epoch pair mAP/AP50为`0.3254/0.5438`；相对同协议父配置`0721_02`双提升`+0.308/+0.236` |
| COMPLETED | `0720_03_liquid_diffproduct_qc_dualmoment_accuracyfix_20260721` | 2026-07-21 03:03 | 2026-07-21 23:54 | `0718_01` + 同时守恒SE总门控量和响应相关一阶矩；加入20260721训练准确性修复 | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `54.650/61.547`，相对旧Paper Base为`+1.336/-0.435`，不满足双提升目标 |
| STOPPED | `0720_03_liquid_diffproduct_qc_dualmoment`原run | 2026-07-21 00:19 | 2026-07-21 03:02 | 与上项模型配置相同，但未包含20260721训练准确性修复 | 主进程运行约2小时43分、到epoch 9后主动停止；旧checkpoint保留，不resume |
| COMPLETED | `0719_06_paper_base_longtail_reweight` | 2026-07-20 05:47 | 2026-07-21 00:16 | Paper Base仅增加已验证的long-tail正样本分类权重 | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `54.087/62.336`，是当前首个相对Paper Base双HOTA提升结果 |
| COMPLETED | `0718_05_paper_liquid_adaptiveanchor_pcdp` | 2026-07-19 07:08 | 2026-07-20 05:50 | Paper Liquid；adaptive-anchor + PCDP局部细节保护 | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `54.142/61.317` |
| COMPLETED | `0718_03_anchorcompetitive_adaptiveanchor` | 2026-07-18 10:19 | 2026-07-19 07:13 | anchor-competitive + adaptive-anchor | epoch 72、18/18 TrackEval |
| COMPLETED | `0716_02_paper_base_r18_coco_full_1200x900_bf16_orderedpairs_reboot_fresh` | 2026-07-16 17:10 | 2026-07-17 12:20 | 当前Paper Base | 多个启动/重启目录中仅`reboot_fresh`为有效正式run；epoch 72、18/18 TrackEval |
| COMPLETED | `0704_03_rerun_liquid_unique_allgt_init2_keep_lr` |  |  | 早期Liquid rerun | epoch 72，TrackEval完成 |
| STOPPED | `0705_02_pyramidlocal` |  |  | Encoder pyramid-local | 最高保留epoch 68；已有35个TrackEval目录，结果已进入Encoder报告 |
| COMPLETED | `0708_01_tristate_decoder` |  |  | tri-state decoder | epoch 72 |
| COMPLETED | `0708_03_tristate_decoder_zeroinit` |  |  | tri-state decoder zero-init | epoch 72 |
| COMPLETED | `0708_04_tristate_decoder_sepffn_zeroinit` |  |  | tri-state decoder + separate FFN | epoch 72 |
| COMPLETED | `0709_01_liquid8` |  |  | 8-group plain Liquid | epoch 72 |
| COMPLETED | `0709_05_liquid8_laf_patternbias` |  |  | pattern-bias LAF | epoch 72 |
| COMPLETED | `0710_01_liquid8_groupmod` |  |  | coverage-aware GroupMod | epoch 72 |
| COMPLETED | `0711_01_liquid8_laf_wide_groupmod` |  |  | Wide LAF + GroupMod | epoch 72；历史half-data Liquid最佳核心之一 |
| COMPLETED | `0712_01_wide_groupmod_outputres` |  |  | Wide LAF + GroupMod + output residual | epoch 72 |
| COMPLETED | `0713_04_longtail_residual_adapter_2gpu_fresh` |  |  | long-tail residual adapter | 初始run停在epoch 4；fresh双卡run完成epoch 72 |
| STOPPED | `0713_05_pairaware_laf_wide`初始run |  |  | pair-aware LAF早期实现 | 仅完成smoke/val验证，没有形成正式训练结果 |
| COMPLETED | `0714_01_pairaware_laf_wide` |  |  | 关闭可视化、保留TrackEval的pair-aware LAF正式run | epoch 72；另一个no-trackeval启动版本未作为结果使用 |
| COMPLETED | `0715_01_half_bf16_encoder_findfalse` |  |  | half-data BF16与`find_unused=False`基线 | epoch 72 |
| COMPLETED | `0715_02_wide_groupmod_pairtransport` |  |  | Wide LAF + GroupMod + PairTransport | epoch 72 |
| STOPPED | `0715_03_wide_groupmod_pairbandcontext` |  |  | pair-band context候选 | 未形成有效checkpoint，重启后队列未恢复 |
| COMPLETED | `0715_05_liquid8_final_pairtransport_paironly_coco365_full_bf16` |  |  | 旧全量Liquid pair-only路径 | epoch 72 |
| COMPLETED | `0715_07_full_baseline_elliptical_spectral_zeroshot` |  |  | proposal zero-shot | 完成检测、跟踪和TrackEval，不重新训练 |
| COMPLETED | `0715_08_full_classaware_elliptical_spectral_rank30_zeroshot` |  |  | 早期class-aware zero-shot分析 | 完成评估；仅作为分析，不作为最终模型 |
| COMPLETED | `0716_01_full_sizeaware_elliptical_spectral_rank30_zeroshot` |  |  | size-aware elliptical + spectral zero-shot | 完成评估 |
| STOPPED | `0716_03_paper_base_plus_liquid` |  |  | 最初Paper Liquid本机run | GPU 2硬件掉卡，epoch 1附近取消，不作为结果 |
| STOPPED | `0717_01_settransport`本机run |  |  | Set-Transport Liquid | 因GPU 2/3掉卡风险在epoch 2主动停止，正式run迁移AutoDL |
| COMPLETED | `0717_02_originalhard` |  |  | 原始逐group hard sampler对照 | epoch 72、18/18 TrackEval |

## 197 服务器

登录：`litianhao@10.106.14.197`；当前正式代码路径：`/data/users/litianhao/PairMOT_sync_3cb888d/ai4rs`。
当前正式训练使用GPU 4、5。

| Status | 实验 | 开始时间 | 结束时间 | 类型/主要改动 | 进度或说明 |
| --- | --- | --- | --- | --- | --- |
| RUNNING | `0801_04_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_symmetricposition_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-08-01 11:25 |  | 仅把共享 decoder self-attention 的 pair-position 输入改为两帧均值的交换对称表示；保留两帧独立 deformable cross-attention、原有有序 frame-feature fusion、Encoder、proposal、PairDN、head、loss 和训练协议 | 不新增参数、层、attention、分支、loss 或矩阵乘法；参数量与父配置同为 `22,758,775`。110 项单测、配置深拷贝和完整构建通过；真实双卡 4-iter smoke 总/DN/encoder loss 与 grad norm 有限，checkpoint 中 24 组独立 attention 张量最大差异 `0.00078709`。formal fresh 于 11:25 启动，iter 50 为 `1.7400 s/iter`、loss `21.5687`、grad norm `115.9326`，两卡约 19.2 GiB，无数值或分布式异常；e4 仅作结构信号，e8/e12 判断持续性 |
| STOPPED | `0801_01_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_terminalcoupleddiagonalfactorizedevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-08-01 01:07 | 2026-08-01 06:46 | 针对 `0731_27` detail gate 长期约为 common gate 三倍的实测失衡，把两条末层路线的独立 gate 合并为一个共用逐通道 gate；仅 256 参数，无新增 decoder 层、attention、分支、loss 或矩阵乘法 | e12 cls HOTA/DetA/AssA `47.158/38.483/59.975`，det `55.516/47.987/66.488`；相对 Encoder 同点 HOTA `-2.522/-1.025`、DetA `-2.877/-2.358`，仅 det AssA `+0.909`。pair mAP/AP50 `0.2468/0.4538`，both-independent `0.2834/0.4872`；其中 pair mAP 与 both AP50 相对已确认父值下降 `0.02637/0.02455`。e8/e12 连续显示 DetA→AssA 搬运且 cls 进一步恶化；epoch 12 checkpoint、检测 metrics、TrackEval metrics、50 序列 txt 与 108 个评估文件完整后精确停止，GPU4/5 已释放 |
| STOPPED | `0731_27_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_terminaldiagonalfactorizedevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 22:16 | 2026-08-01 00:42 | 保留独立 attention 与末层 common/detail 语义，把两个 `256×256` 稠密门简化为两个逐通道向量 | e8 cls/det HOTA `43.344/49.456`，相对 Encoder 同点 `-1.925/-0.737`；cls DetA/AssA `-2.641/-1.160`，det `-3.760/+3.246`；pair mAP/AP50 下降 `0.024681/0.022977`，both-independent 下降 `0.027357/0.024470`。e4 的强早期增益没有保持；epoch 8 checkpoint、检测、TrackEval 与 54 个原始文件完整后精确停止，GPU4/5 已释放 |
| STOPPED | `0731_25_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_terminalconfidentdetailfactorizedevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 18:49 | 2026-07-31 22:11 | 仅以 detached 双帧分类置信度约束 detail 修正，无新增参数 | e8 `43.629/50.129`，相对 Encoder 同点 `-1.640/-0.064`；DetA 与四项 AP 系统性下降，checkpoint、metrics 和 54 个 TrackEval 原始文件验证后停止 |
| STOPPED | `0728_01_paper_base_liquid_encoder_p5temporal_dualevidence_decoder0708_03` | 2026-07-28 09:30 | 2026-07-29 01:29 | 严格以`0727_01`为父配置，冻结Base、Liquid、P5 temporal、Dual-Evidence encoder、proposal、PairDN和loss；仅加入`0708_03`的`pointer/query_prev/query_curr` tri-state decoder，并启用零初始化frame-pointer循环耦合，不使用separate FFN | 完整评估到epoch 48，共12个评测点；最后也是已评测最佳为 `52.587/60.682`，pair mAP/AP50 `0.305359/0.528296`。训练在epoch 52 iter 250收到外部SIGTERM，与随后“全部停止”调度一致，并非模型异常；不resume，已被当前轻量terminal方向取代 |
| STOPPED | `0727_11_paper_base_liquid_encoder_p5temporal_momentcompetitive`的197队列 | 2026-07-27 12:15 | 2026-07-27 12:59 | 继承当前最强的`0727_01` P5 MHA与Dual-Evidence双残差；用每通道`RMS-mean(abs(x))`稀疏性矩补充common/detail全局描述，并以两路softmax共享预算替代彼此独立的sigmoid门 | 队列本身运行正常，但用户明确改在AutoDL立即训练；197未运行smoke、未创建正式workdir，队列在AutoDL正式训练通过iter 50验收后关闭。同一实验ID及科学配置迁移至AutoDL，不构成新实验 |
| STOPPED | `0727_10_paper_base_liquid_encoder_p5temporal_detailredistribute` | 2026-07-27 08:10 | 2026-07-27 12:14 | 严格继承`0727_09`，将空间门描述停止梯度并把detail空间调制归一化为均值1 | `0727_09` epoch 16已相对Base+Liquid双降`-0.075/-0.399`，det DetA/AssA也同时下降`0.375/0.353`，表明问题并非仅是空间门全局能量漂移；在smoke和正式训练前取消，两个目标workdir均未创建，由不使用空间门的`0727_11`替代 |
| COMPLETED | `0727_09_paper_base_liquid_encoder_p5temporal_detailspatial` | 2026-07-27 05:31 | 2026-07-28 04:02 | 严格继承`0727_01`的P5 MHA和Dual-Evidence；common共享残差完全不变，仅用局部common/detail幅值生成单通道空间可靠性，对signed-detail残差做逐位置调制。空间卷积零初始化并使用`2*sigmoid`，因此初始函数与父配置逐元素相同 | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `54.106/62.321`，同epoch pair mAP/AP50为`0.3122/0.5273`。相对`0727_01`为`-0.331/-0.072`，未明显超过，故按计划进入Decoder阶段 |
| STOPPED | `0727_07_paper_base_liquid_encoder_p5temporal_spatialbranchtrust` | 2026-07-27 03:05 | 2026-07-27 05:24 | 完整继承`0727_02`的双分支空间门，并增加无参数branch-energy trust region | `0727_02` epoch 12已相对Base+Liquid双降，证明其主要问题是common空间门压低检测覆盖，而不是后期残差能量失控；队列在执行smoke或正式训练前取消，未产生正式workdir |
| STOPPED | `0727_02_paper_base_liquid_encoder_p5temporal_spatialevidence` | 2026-07-27 01:39 | 2026-07-27 05:24 | 固定`0723_01` Liquid和`0705_01` P5 MHA；post-FPN使用common/detail双残差，并以2通道局部空间证据门控P3/P4/P5两条更新 | 主动停在epoch 12后。epoch 12为`47.411/54.593`，相对同epoch Base+Liquid双降`-0.676/-1.042`；det DetA/AssA分别下降`1.052/0.829`，pair mAP下降`0.0108`。与epoch 8的`-0.035/+0.465`结合，说明同时空间门控common/detail会持续削弱检测覆盖，后继改为只门控detail |
| COMPLETED | `0726_01_pairdn_paircoherent_le180_dse_cpdse_sparsereserve` | 2026-07-26 01:57 | 2026-07-27 01:36 | `0725_01` + Sparse-Reserve CP-DSE：以现有归一化dispersion map的空间RMS减均值衡量类别无关的稀疏小目标证据，对pair两帧取共享group reserve；仅衰减负向CP-DSE残差，正向修正和pair共享性质不变；无新参数、阈值或loss | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 68为`54.230/61.634`，同epoch pair mAP/AP50为`0.3138/0.5301`。相对Paper Base为`+0.916/-0.348`，相对父配置`0725_01`为`-0.896/-0.364`，不再派生Sparse-Reserve |
| COMPLETED | `0725_01_pairdn_paircoherent_le180_dse_cpdse_pairglobal` | 2026-07-25 02:45 | 2026-07-26 01:41 | `0723_01` + DSE + pair-global CP-DSE：DSE以mean/RMS增强局部检测证据，CP-DSE以pair共享group残差补充关联一致性；两条路径分别针对`0723_03`的det AssA下降和`0723_06`的det DetA下降，无额外loss | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为`55.126/61.998`，同epoch pair mAP/AP50为`0.3231/0.5449`；相对Paper Base双提升`+1.812/+0.016`，det DetA/AssA为`-0.138/+0.545`；相对DSE父配置同epoch为`+0.328/+0.245` |
| COMPLETED | `0723_06_pairdn_paircoherent_le180_cpdse_pairglobal` | 2026-07-24 03:58 | 2026-07-25 01:53 | `0723_01` + consistency-preserving DSE pair-global：保留`x.mean`主路径，将归一化离散度汇聚为pair共享group标量后生成有界SE-logit残差；8个新增参数，无额外loss | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为`54.124/61.914`，同点pair mAP/AP50为`0.3124/0.5241`；相对Paper Base为`+0.810/-0.068`，det AssA提高`0.645`但DetA下降`0.432` |
| COMPLETED | `0723_03_pairdn_paircoherent_le180_dse` | 2026-07-23 06:19 | 2026-07-24 03:41 | `0723_01` + DSE；保留两帧共享DN相对噪声和全部新PairDN/L1设置，只用identity初始化的16参数mean/RMS evidence mixer增强fusion证据 | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 68为`55.036/61.745`，同epoch pair mAP/AP50为`0.3239/0.5402`；相对Paper Base为`+1.722/-0.237`，不能视为双提升 |
| COMPLETED | `0722_01_liquid_independent_diffproduct_dnnegativefix` | 2026-07-22 10:34 | 2026-07-23 05:54 | `0721_02`的PairDN生成器修复版：将Negative DN从可趋近零的乘积噪声改为独立符号乘`uniform(1,2)`，并将attention mask修正为同一contrastive group内正负块互相可见；因此不是单独的box-noise消融 | 完成epoch 72和计划TrackEval；最终指标见Paper Mainline报告 |
| COMPLETED | `0721_02_liquid_independent_diffproduct_accuracyfix` | 2026-07-21 14:59 | 2026-07-22 10:31 | 修复训练几何/GMC及PairDN negative目标后的`0718_01`严格复跑；尚未包含Negative-DN外环噪声修复 | 完成epoch 72和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `54.327/61.659`，HOTA和`115.986`；同epoch pair mAP/AP50为`0.3161/0.5340` |
| COMPLETED | `0720_02_liquid_diffproduct_qc_responsemass` | 2026-07-20 18:19 | 2026-07-21 13:52 | `0718_01` + response-weighted fusion quality conservation | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `54.463/61.437`，HOTA和`115.900`，但det低于Base `0.545` |
| COMPLETED | `0719_04_paper_liquid_widelaf_groupmod` | 2026-07-19 20:41 | 2026-07-20 16:31 | Paper协议严格复现`0711_01`核心：独立8-group + Wide LAF + GroupMod，无pair router/transport和set约束 | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `53.932/60.908`，HOTA和`114.840` |
| COMPLETED | `0718_06_cpas_settransport` | 2026-07-18 23:37 | 2026-07-19 20:45 | CPAS + Set-Transport + difference/product | epoch 72、18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `54.877/61.274` |
| COMPLETED | `0718_01_independent_diffproduct` | 2026-07-18 02:05 | 2026-07-18 21:48 | 独立group + difference/product pair coupling | epoch 72、18/18 TrackEval |
| STOPPED | `0703_liquid_*bothvis*` |  |  | 早期Liquid/both-visible路径 | 停在epoch 48 |
| STOPPED | `0704_02_liquid_*typed_pairtopk*` |  |  | typed proposal早期路径 | 停在epoch 64 |
| STOPPED | `0704_03_liquid_*unique_pairdn_allgt*` |  |  | 早期Liquid all-GT | 停在epoch 36 |
| COMPLETED | `0708_02_tristate_decoder_sepffn` |  |  | decoder separate FFN | epoch 72 |
| COMPLETED | `0709_03_liquid8_laf_overlap` |  |  | overlap-aware LAF | epoch 72 |
| COMPLETED | `0710_02_liquid8_laf_outputres` |  |  | LAF output residual | epoch 72 |
| COMPLETED | `0711_02_liquid8_laf_wide_bandattn` |  |  | Wide LAF + band attention | epoch 72 |
| STOPPED | `0713_05_pairaware_laf_wide`系列 |  |  | pair-aware LAF远程尝试 | 多次启动异常；最高有效checkpoint epoch 12，不再续跑 |
| STOPPED | `0714_02_amp_fastgdloss/fp32transformer`系列 |  |  | AMP边界探索 | fast-GDLoss无正式checkpoint；FP32-transformer版本停在epoch 4 |
| COMPLETED | `0714_03_half_hybrid_amp_fixed` |  |  | 修复后的hybrid AMP对照 | epoch 72 |
| COMPLETED | `0715_04_wide_groupmod_pairchangegate` |  |  | pair-change gated coupling | epoch 72 |
| STOPPED | `0716_03_paper_base_plus_liquid` |  |  | 初始Paper Liquid | epoch 20后因route/group坍塌主动停止 |
| COMPLETED | `0716_04_groupsetunique` |  |  | 跨group集合唯一hard约束 | epoch 72、18/18 TrackEval |
| STOPPED | `0717_03_hardsoftcontext` |  |  | hard采样 + soft context | epoch 12 checkpoint，约epoch 15主动停止，不resume |

资源测试（2026-07-23 13:40-13:48）：在用户私有目录
`/dev/shm/litianhao/pairmot_hsmot_cache`构建了带锁、容量预检、文件指纹校验和原子切换的
HSMOT JPEG缓存。缓存复制train/test共41,514张JPEG、占用约11 GiB，标注和GMC仍指向持久盘；
构建后`/dev/shm`剩余约239 GiB、系统available memory约452 GiB。GPU3对`0723_03 DSE`
进行单卡physical batch 8、8 workers、120 iter并发测试：默认allocator在首次反向OOM；
设置`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`后完成测试，去除前20 iter后的均值为
`2.5767 s/iter`，其中`data_time=0.1823 s`，MMEngine峰值memory约21.1 GiB，设备采样峰值约
23.8 GiB。同期GPU4/5正式训练均值未下降（相对紧邻运行前窗口约快1%，属于正常波动），仅
出现少量`data_time`尖峰。结论：tmpfs缓存本身容量和隔离方式安全，也未测得对GPU4/5总吞吐
的负面影响；但DSE单卡batch 8显存余量不足，不建议作为长期正式配置，正式单卡应使用
batch 4 + accumulation 2或先进一步降低峰值显存。

CSPR-DSE一epoch候选试验（2026-07-23 14:01-14:45）：在`0723_03`的PairDN+DSE上增加
CSPR，分别用低分辨率共享Conv3D预览增强route descriptor、用mean/RMS混合增强fusion
evidence，两条路径正交且不增加额外loss。GPU3使用tmpfs、physical batch 8、8 workers及
`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`完成完整epoch；checkpoint元数据确认
`epoch=1, iter=1038, seed=3407`。跳过前100 iter后平均`2.4174 s/iter`，其中
`data_time=0.0494 s`、计算约`2.3679 s`，MMEngine记录峰值约21.2 GiB，设备采样峰值约
24.17 GB。GPU4/5同期由`0.8975`变为`0.9094 s/iter`，约慢1.33%。当前数据读取只占约2%，
增加`num_workers`不会解决单卡相对双卡慢的问题；差距主要来自3090单卡承担双倍样本及
极限显存下的计算/工作区效率。该epoch证明组合结构和梯度路径可训练，不构成最终性能证据；
是否正式长训应结合178上独立CSPR的中期结果决定。

## 252 服务器

登录：`litianhao01@10.106.15.252`；代码路径：
`/data/users/litianhao01/PairMmot/ai4rs`。当前正式训练使用GPU 0、1。

| Status | 实验 | 开始时间 | 结束时间 | 类型/主要改动 | 进度或说明 |
| --- | --- | --- | --- | --- | --- |
| STOPPED | `0730_10_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_symmetricpair_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 18:41；2026-08-01 10:59 resume | 2026-08-01 12:31 | 两帧共享 decoder deformable cross-attention，并对 frame-evidence fusion 与 pair-position fusion 显式平均正反帧序；零新增参数，不改 Encoder、proposal、PairDN、head、loss 或训练调度 | e8 cls HOTA/DetA/AssA `42.890/33.328/58.017`，det `48.228/41.431/57.793`；相对 Encoder 同点 HOTA `-2.379/-1.965`、DetA `-4.335/-5.630`。pair mAP/AP50 `0.2034/0.3969`，both-independent `0.2390/0.4301`，检测 AP 同样系统性下降。`epoch_8.pth`、检测 metrics、50 序列、TrackEval `async_done=1` 与 108 个文件完整后精确停止；GPU0/1 已释放。e4 增益未持续，完整对称化不再继续到 e12 |
| STOPPED | `0801_03_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_terminaldiagonalcentermotiondetailonly_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-08-01 05:56 | 2026-08-01 08:54 | 严格继承 `0801_02` 的 Encoder 分类路径、独立 attention、最终中心 `x/y` detail 与全部训练协议；唯一变化是把 `256×256` 稠密 gate 改为 256 维逐通道 gate | e8 cls HOTA/DetA/AssA `44.183/35.231/58.370`，det `50.011/44.289/58.441`；相对 Encoder 同点 HOTA `-1.086/-0.182`、DetA `-2.432/-2.772`，AssA `+1.069/+3.296`。pair mAP/AP50 `0.210025/0.406177`，both-independent `0.245219/0.438068`；相对 Encoder 分别下降约 `0.0277/0.0247` 与 `0.0310/0.0279`。epoch 8 checkpoint、检测 metrics、TrackEval `async_done=1`、50 序列与 108 个评估文件完整；结构审计显示 6 组 attention 最大差异 `0.059180`、唯一 gate 最大值 `0.337337`，排除未学习。双 HOTA、双 DetA 与四项 AP 同向下降，仅 AssA 上升，按系统性退化规则停止；screen/worker 已退出，GPU0/1 均为 `0%/1 MiB` |
| STOPPED | `0801_02_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_terminalcentermotiondetailonly_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-08-01 02:44 | 2026-08-01 05:38 | 以 Encoder/0731_28 为对照，完全取消 classification common 修正，只保留最终 box head 的严格反对称中心 x/y detail；宽高角、辅助输出和 recurrent references 保持父路径 | e8 cls HOTA/DetA/AssA `44.632/35.582/58.513`，det `49.687/43.373/59.043`；相对 Encoder 同点 HOTA `-0.637/-0.506`、DetA `-2.081/-3.688`，pair mAP/AP50 下降 `0.0219/0.0130`，both-independent 下降 `0.0243/0.0142`。HOTA、DetA、AP 系统性下降；完整 e8 checkpoint、检测 metrics、TrackEval 与 108 个评估文件核验后精确停止，GPU0/1 已释放 |
| STOPPED | `0731_29_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_terminaldiagonalcentermotionfactorizedevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 23:15 | 2026-08-01 02:12 | `0731_27` 与 `0731_28` 的轻量几何交叉：common/detail 均使用逐通道门控，反对称 box detail 仅修正中心 `x/y`，`w/h/angle` 保持父几何；无新增层、attention、分支或 loss | e8 cls/det HOTA `44.148/49.376`，相对 Encoder 同点 `-1.121/-0.817`；cls/det DetA 为 `-2.028/-3.357`，pair mAP/AP50 下降 `0.028467/0.019227`，both-independent 下降 `0.030575/0.019963`。完整 checkpoint、检测 metrics、TrackEval metrics、50 个序列 txt 与 108 个评估文件核验后精确停止，GPU0/1 已释放 |
| STOPPED | `0731_26_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_terminalconfidentbothfactorizedevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 18:49 | 2026-07-31 23:12 | 同时用 detached 双帧分类置信度约束 common/detail 修正；不新增参数 | e12 `48.766/55.694`，相对 Encoder 同点 `-0.914/-0.847`；cls/det DetA 为 `40.056/47.910`，pair mAP/AP50 为 `0.262434/0.479349`。epoch 12 checkpoint、检测 metrics、TrackEval metrics 与 54 个原始结果文件完整验证后精确停止；与 `0731_24/25` 一起否定全部 confidence 放置，不再派生 |
| STOPPED | `0727_04_paper_base_liquid_encoder_p5temporal_detailenergy` | 2026-07-28 00:50 | 2026-07-29 01:51 | 固定`0723_01` Liquid与P5 MHA，保持`0726_03` common/detail结构；仅对两帧反向的signed-detail残差施加逐通道、逐样本的原始pair-detail RMS上限，防止时序修正能量超过输入帧差。约束使用detached统计、无参数、无loss且仍严格保持pair均值与帧交换等变 | 完整评估到epoch 56，共14个评测点；最后也是已评测最佳为 `53.796/61.711`，pair mAP/AP50 `0.309336/0.528966`。训练在epoch 59 iter 150收到外部SIGTERM，与“全部停止”调度一致；不resume，未达到Encoder目标 |
| COMPLETED | `0726_03_paper_base_liquid_encoder_p5temporal_commondetail_pairdn_paircoherent_le180` | 2026-07-26 20:35 | 2026-07-28 00:49 | `0726_02`的严格encoder后继：保留`0723_01` Base + Liquid、P5双向全局MHA和全部训练协议；将原三尺度方向性pyramid-local替换为pair-common/detail分解。共享`[mean, abs(detail)]`描述控制奇函数local detail变换，两帧施加等大反向残差，严格保持pair均值且交换帧时输出等变 | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为`54.654/62.240`，同epoch pair mAP/AP50为`0.3239/0.5380`。相对Base+Liquid双提升`+0.699/+0.208`，相对Paper Base双提升`+1.340/+0.258`；det DetA/AssA相对Base+Liquid也同时提高`+0.039/+0.447` |
| COMPLETED | `0725_03_pairdn_paircoherent_le180_dse_cpdse_dettangent_fast` | 2026-07-25 11:05 | 2026-07-26 17:44 | `0725_01` + Detection-Tangent CP-DSE：按已有Conv3D second moment和DSE gate的group敏感度形成pair共享检测重要性向量，只移除CP-DSE残差沿该向量的一阶检测响应分量；保留其余关联修正，无新增参数/loss | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `54.229/61.808`，同epoch pair mAP/AP50为`0.3196/0.5340`；相对Paper Base为`+0.915/-0.174`，det DetA/AssA分别为`-0.074/-0.122`；相对未投影父配置`0725_01`同epoch为`-0.897/-0.190`，不再沿该方向派生 |
| STOPPED | `0725_03`两个速度审计启动 | 2026-07-25 10:54 | 2026-07-25 11:03 | 与上项模型语义相同；依次消除重复`[B,C,G,H,W]`二阶矩归约，并将完整gate-map sigmoid导数改为group-pooled logit导数 | 两个启动均只到epoch 1 iter 100、无正式checkpoint/TrackEval，不resume且不进入结果；保留目录仅用于速度审计，分别以`orderedpairs_fresh`和`optimized_fresh`结尾 |
| COMPLETED | `0723_07_pairdn_paircoherent_le180_pecg` | 2026-07-24 06:42 | 2026-07-25 10:44 | `0723_01` + Pair Evidence Consensus Gate：仅在route覆盖和Conv3D证据均一致的位置收缩两帧SE gate差异；修正项两侧反号，严格保持pair平均gate；8参数，无额外loss | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 52为`53.693/61.151`，同点pair mAP/AP50为`0.3096/0.5262`；相对Paper Base为`+0.379/-0.831`，det DetA/AssA分别下降`1.227/0.079`，不再派生该方向 |
| COMPLETED | `0723_02_paper_liquid_independent_diffproduct_pairdn_independent_le180` | 2026-07-23 03:30 | 2026-07-24 06:12 | `0723_01`严格DN噪声消融：保持模型、数据、初始化、正负比、难度筛选、attention mask和loss不变，仅将pair两帧共享的相对box噪声改为两帧独立采样 | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为`53.637/61.679`，同epoch pair mAP/AP50为`0.3155/0.5308`；相对Paper Base为`+0.323/-0.303`，不满足双提升 |
| COMPLETED | `0721_04_liquid_bsac_diffproduct_accuracyfix` | 2026-07-22 00:52 | 2026-07-23 01:50 | accuracy-fixed `0718_01` + BSAC；在Liquid采样后、共享Conv3D前加入24参数的band-slot条件校准；启动时仍使用修复前的Negative-DN噪声 | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `54.530/61.528`，同epoch pair mAP/AP50为`0.3211/0.5353`；相对同协议父配置`0721_02`为`+0.203/-0.131` |
| COMPLETED | `0720_01_liquid_diffproduct_qc_gatemass` | 2026-07-20 18:19 | 2026-07-21 19:25 | `0718_01` + gate-mass tangent fusion quality conservation | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `54.103/61.581`，相对Paper Base为`+0.789/-0.401`，不满足双提升目标 |
| COMPLETED | `0719_03_pairconsensus_relaxedset_nopacde` | 2026-07-19 16:45 | 2026-07-20 18:04 | 0719 pair-consensus严格消融，仅移除PACDE | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `54.209/61.504`，HOTA和`115.713` |
| COMPLETED | `0718_04_adaptiveanchor_sase` | 2026-07-18 13:12 | 2026-07-19 16:15 | adaptive-anchor + SASE小目标增强 | epoch 72、18/18 TrackEval |
| COMPLETED | `o2_rtdetr_*coco_pretrain_3dse_reduction2` |  |  | 单帧RT-DETR检测器 | epoch 72；后续用于single-det + BoT-SORT检测跟踪对照 |
| COMPLETED | `0704_01_*resume_from_epoch40_to72` |  |  | 历史half-data baseline resume | 从epoch 40续训到72；报告baseline采用其高指标 |
| COMPLETED | `0708_01_tristate_decoder` |  |  | tri-state decoder远程run | epoch 72 |
| COMPLETED | `0709_02_liquid8_liquidawarefusion` |  |  | 初版Liquid-aware fusion | epoch 72 |
| COMPLETED | `0709_04_liquid8_laf_wide_overlap` |  |  | Wide LAF + overlap context | epoch 72；历史有效Liquid核心之一 |
| COMPLETED | `0710_03_liquid8_sampler_bandattn` |  |  | sampler band attention | epoch 72 |
| COMPLETED | `0711_03_wide_groupmod_bandattn` |  |  | Wide LAF + GroupMod + band attention | epoch 72 |
| COMPLETED | `0713_01_longtail_reweight` |  |  | long-tail reweight | epoch 72 |
| COMPLETED | `0713_02_finecls_margin` |  |  | fine-class margin | epoch 72 |
| COMPLETED | `0713_03_longtail_proto_gate` |  |  | long-tail prototype gate | epoch 72 |
| COMPLETED | `0714_01_0704_resume_coco365_full_unique_allgt` |  |  | 旧全量数据baseline | epoch 72；同名AMP启动版本未形成正式checkpoint |
| COMPLETED | `0715_06_liquid8_pairbandcontext_paironly_coco365_full_bf16` |  |  | pair-band context全量Liquid | epoch 72 |
| COMPLETED | `0716_05_groupsetunique_encoder` |  |  | Paper Liquid group-set unique + temporal/pyramid Encoder | epoch 72、18/18 TrackEval |
| NONE | - |  |  | - | 当前没有等待中的下一实验 |

## 178 服务器

登录：`litianhao01@10.106.15.178`；代码路径：
`/data1/users/litianhao01/PairMOT/ai4rs`。PairMOT仅使用GPU 0；GPU 1不纳入调度。

| Status | 实验 | 开始时间 | 结束时间 | 类型/主要改动 | 进度或说明 |
| --- | --- | --- | --- | --- | --- |
| RUNNING | `0801_05_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_symmetricfeature_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` | 2026-08-01 11:57 |  | 只把每层独立双帧 cross-attention 输出在进入共享 recurrent query 前取均值，并经原 `cross_fusion` 一次；保留独立 cross-attention、原有有序 pair-position、Encoder、proposal、PairDN、head、loss 与训练协议 | 零新增参数和矩阵乘法，参数量与 Encoder 均为 `22,758,775`。e4 cls HOTA/DetA/AssA `34.947/25.960/49.970`，det `38.300/30.836/48.870`；相对 Encoder 同点 HOTA `-1.262/-0.453`，DetA `-1.108/-1.618`，cls/det AssA `-2.124/+1.404`。pair mAP/AP50 `0.1399/0.2951`，both-independent `0.1685/0.3240`。checkpoint、检测、50 序列、TrackEval `async_done=1` 与 108 个评估文件完整；13:11 已继续至 epoch 5 iter 500，loss `10.1890`、grad norm `48.9170`，无异常；按放宽规则继续 e8 |
| STOPPED | `0731_21_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_terminalorthogonalfactorizedevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` | 2026-07-31 15:13；2026-08-01 00:52 resume | 2026-08-01 03:18 | 独立双帧 attention + 末层分类 common 与严格反对称 box detail 分解；零初始化 | e40 cls/det HOTA `53.655/60.379`，相对 Encoder 同点 `-0.142/-0.684`；e32/e36/e40 连续未双超。e40 检测、TrackEval、50 序列与 108 文件已归档至 `val_track_0010_resume_epoch40` 并校验；精确停止后 GPU0 释放，不再 resume |
| STOPPED | `0731_16_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_terminalcommonevidencebypass_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` | 2026-07-31 12:42 | 2026-07-31 15:03 | 继承`0727_01`及其共享 decoder 路径，只在最后一层最终双帧预测头前注入 swap-invariant、零起点且有界的共同 cross-attention 证据；不改 recurrent query、所有 auxiliary output 及任何供后续层消费的 reference | e8 `43.972/49.378`，相对Encoder同点 `-1.297/-0.815`；两次完整评估后停止并由 `0731_21` 的common/detail正交分解替代。无训练故障，不resume |
| STOPPED | `0727_08_paper_base_liquid_encoder_p5temporal_dualbranchtrust` | 2026-07-27 20:51 | 2026-07-27 21:37 | 完整保留`0727_01`的P5 MHA与common/detail双残差，不使用空间门；仅分别限制shared-common和signed-detail更新RMS不超过其对应输入证据RMS，未超限更新不变 | 前序完成18/18评测且GPU连续空闲后，严格队列完成真实数据smoke并fresh启动；按用户指令主动停在epoch 3 iter 750。未生成正式epoch checkpoint，不resume、不进入论文结果；训练、launcher及队列进程均已退出，GPU 0释放，GPU 1上的其他任务未受影响 |
| STOPPED | `0727_06_paper_base_liquid_encoder_p5temporal_sharedscalar` | 2026-07-27 02:43 | 2026-07-27 03:47 | 保留P5 MHA和signed-detail，但以两帧共享的逐位置标量增益替代channel-mixing common残差 | 只进入等待队列，未创建smoke、正式目录或训练进程。`0727_01` epoch 12的det AssA已恢复到相对Base `+0.006`，同时det DetA `+1.449`，证明原common分支可实现无关联代价的检测增益；移除其通道表达能力的设计依据消失，因此队列撤销并由`0727_08`替代 |
| STOPPED | `0727_03_paper_base_liquid_encoder_p5temporal_scalesplit` | 2026-07-27 00:29 | 2026-07-27 02:40 | 固定`0723_01` Liquid和P5 MHA；P3只启用common共享检测增强，P4/P5启用common+signed-detail | 只进入等待队列，未创建smoke、正式目录或训练进程。epoch 8诊断表明实际矛盾是common残差带来det DetA `+2.710`但det AssA `-2.918`；移除P3 detail没有针对该问题，因此主动撤销队列并由`0727_06`替代 |
| COMPLETED | `0727_01_paper_base_liquid_encoder_p5temporal_dualevidence` | 2026-07-27 00:17 | 2026-07-27 20:48 | 固定`0723_01` Liquid、PairDN、proposal、decoder和loss，保留`0705_01` P5双向MHA；post-FPN将common共享检测证据同向注入两帧，将signed detail关联证据反向注入两帧，零门控且严格帧交换等变 | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `54.437/62.393`，同epoch pair mAP/AP50为`0.3201/0.5371`。相对Base+Liquid双提升`+0.482/+0.361`；相对Paper Base双提升`+1.123/+0.411`，det DetA/AssA也分别提高`+0.526/+0.311` |
| PREPARED | `0727_05_paper_base_liquid_encoder_p5temporal_spatialdetail` |  |  | 回退到`0726_03`的detail-only、pair均值守恒结构；以逐层空间均值归一化的common/detail局部能量生成`2→1`空间可靠性门，只调制signed-detail，不改共享单帧特征。门初值严格为1、范围`[0,2]`，总计57参数，无额外loss | 本地和178均通过20项单测及逐文件哈希，包含相同父权重与非零gamma下逐元素严格等价测试；完整模型参数为22,677,421。仅完成准备，未启动、未占GPU；作为detail-only保守回退候选，当前优先执行更直接针对DetA/AssA矛盾的`0727_06` |
| COMPLETED | `0725_02_pairdn_paircoherent_le180_dse_cpdse_centered_protocolfix` | 2026-07-25 08:30 | 2026-07-26 11:57 | `0725_01`的DSE + pair-global CP-DSE组合，但将8个group的CP-DSE logit残差去均值；单卡physical batch 4 + accumulation 2，并将warmup从2000改为4000 micro-iter、EMA改为`interval=2,gamma=4000`，保持按optimizer update计的原协议时间尺度 | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 68为cls HOTA `53.730`、det HOTA `61.864`，同epoch pair mAP/AP50为`0.3100/0.5239`；相对Paper Base为`+0.416/-0.118`，相对Base+Liquid仍未双侧超过 |
| STOPPED | `0725_02_pairdn_paircoherent_le180_dse_cpdse_centered`初始accumulation run | 2026-07-25 06:57 | 2026-07-25 08:27 | 模型与上项相同，physical batch 4 + accumulation 2，但仍沿用2000 micro-iter warmup和每micro-iter计数的EMA | 到epoch 4后主动停止，不resume；MMEngine源码核实ParamSchedulerHook和EMAHook均按micro-iteration推进，导致warmup和EMA时间尺度相对原bs8协议缩短一半。该run及其1/18 TrackEval只作审计，不进入结果表 |
| COMPLETED | `0723_08_pairdn_paircoherent_le180_scpd` | 2026-07-24 08:53 | 2026-07-25 06:38 | `0723_01` + Spectral-Coordinate Pair Dispersion：将只读group色散经soft coverage投影到8个物理谱段，在谱段坐标求pair共识后投影回各帧自身route；跨group去均值、零初始化、有界8参数，无额外loss | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为`54.465/61.213`，同点pair mAP/AP50为`0.3152/0.5334`；相对Paper Base为`+1.151/-0.769`，det DetA/AssA为`53.712/72.176`，失败原因主要是关联质量下降 |
| COMPLETED | `0723_04_pairdn_paircoherent_le180_cspr` | 2026-07-23 12:31 | 2026-07-24 08:50 | `0723_01` + CSPR；保留两帧共享DN相对噪声，以detached shared-Conv3D在24x32低分辨率生成谱段预览统计，替换简单global route descriptor | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为`54.523/61.738`，同epoch pair mAP/AP50为`0.3162/0.5376`；相对Paper Base为`+1.209/-0.244`，不满足双提升 |
| COMPLETED | `0721_03_liquid_bsr_diffproduct_accuracyfix_dnnegativefix` | 2026-07-22 10:07 | 2026-07-23 06:36 | 配置继承`0721_02`并以12x16 block recurrent descriptor替换global route statistics；正式启动源码同时包含修复后的Negative-DN外环采样和contrastive-group attention mask，实际语义更接近`0722_01 + BSR` | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `53.107/61.118`，同epoch pair mAP/AP50为`0.3082/0.5220`，结果不支持用BSR替换global descriptor |
| COMPLETED | `0721_01_liquid_diffproduct_qc_responsemass_1xb8_shm_accuracyfix_20260721` | 2026-07-21 03:03 | 2026-07-22 02:43 | `0718_01` + response-weighted fusion quality conservation；单卡batch 8；JPEG使用安全tmpfs缓存；加入20260721训练准确性修复 | 完成epoch 72和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `54.006/60.934`，HOTA和`114.940`；同epoch pair mAP/AP50为`0.3148/0.5297` |
| STOPPED | `0721_01_liquid_diffproduct_qc_responsemass_1xb8_shm`原run | 2026-07-21 01:23 | 2026-07-21 03:02 | 与上项模型配置相同，但未包含20260721训练准确性修复 | 主进程运行约1小时39分、到epoch 4后主动停止；旧checkpoint保留，不resume |
| COMPLETED | `0719_05_paper_base_rerun` | 2026-07-20 00:20 | 2026-07-21 01:02 | 无Liquid的Paper Base复跑；seed 3407、LR `1e-4`、BF16、全量1200x900；单卡batch 8保持global batch 8 | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `52.417/61.265`，HOTA和比主Base低`1.614` |
| COMPLETED | `0719_02_pairconsensus_reliability` | 2026-07-19 03:02 | 2026-07-20 00:28 | reliability-weighted pair-consensus + PACDE，单卡batch 8 | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 64为cls/det HOTA `55.190/60.689` |
| COMPLETED | `ctracker_hsmot_r50_3dse_rotated_1200x900_bs4_acc2` |  |  | CTracker HSMOT适配 | 完成100 epochs及最终推理/TrackEval；checkpoint已归档到共享存储 |

队列日志：
`/data4/litianhao/PairMmot/workdir_178/queue_0721_01_after_0719_05.log`、
`/data4/litianhao/PairMmot/workdir_178/queue_0719_05_after_0719_02.log`、
`/data4/litianhao/PairMmot/workdir_178/queue_0721_03_after_0721_01.log`、
`/data4/litianhao/PairMmot/workdir_178/queue_0723_04_after_0721_03.log`。

## AutoDL 临时服务器

AutoDL实例地址会变化，初始化、数据布局、训练、finalizer和关机流程见
`pairmot-autodl`技能及`/data/users/wangying01/lth/PairMOT/autodl`。以下只记录正式实验，
不将实例迁移和环境smoke作为实验。

| Status | 实验 | 开始时间 | 结束时间 | 类型/主要改动 | 进度或说明 |
| --- | --- | --- | --- | --- | --- |
| COMPLETED | `0727_11_paper_base_liquid_encoder_p5temporal_momentcompetitive` | 2026-07-27 12:57 | 2026-07-28 06:37 | 固定`0723_01` Base+Liquid、P5 MHA和全部论文协议；保留Dual-Evidence双残差，以逐通道`RMS-mean(abs(x))`感知稀疏目标证据，并用common/detail两路softmax共享门控预算。单卡RTX 5090 physical batch 8保持global batch、LR、EMA和72-epoch语义 | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 68为cls/det HOTA `54.853/62.050`，同epoch pair mAP/AP50为`0.3190/0.5384`，HOTA和`116.903`。相对Base+Liquid为`+0.898/+0.018`，相对`0727_01`为`+0.416/-0.343`且HOTA和仅高`0.073`，不能视为明显胜出。finalizer已归档epoch 68、epoch 72、日志和完整TrackEval，状态为`publish_skipped_no_deploy_key artifacts_preserved best_epoch=68`；权威结果已取回`autodl/results/0727_11` |
| COMPLETED | `0726_02_paper_base_liquid_encoder_p5temporal_pyramidlocal_pairdn_paircoherent_le180` | 2026-07-26 15:37 | 2026-07-27 09:23 | `0723_01` Base + Liquid + `0705_01` Encoder：在FPN前P5加入全局双向pair temporal MHA，在FPN后P3/P4/P5加入逐层pyramid-local adapter；两支均以零初始化gamma残差接入，不改Liquid、proposal、decoder、PairDN或loss | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `54.742/61.631`，同点pair mAP/AP50为`0.3223/0.5429`。相对直接父配置`0723_01`最佳点为`+0.787/-0.401`，不满足双侧超过Base+Liquid；相对Paper Base为`+1.428/-0.351`。finalizer已将epoch 72、日志、完整TrackEval归档到`/autodl-fs/data/PairMOT_results/0726_02`；因无deploy key跳过GitHub发布，权威结果已取回`autodl/results/0726_02` |
| COMPLETED | `0719_01_pairconsensus_relaxedset_pacde` | 2026-07-19 02:22 | 2026-07-19 22:39 | pair共享route + relaxed Set-Transport + pair-aligned fusion + PACDE | epoch 72、18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `53.883/61.411`；结果已从共享盘取回 |
| COMPLETED | `0717_01_paper_base_plus_liquid_settransport` |  |  | Set-Transport Liquid | epoch 72、18/18 TrackEval；结果归档于`/root/autodl-fs/PairMOT_results/0717_01` |
| COMPLETED | `0718_02_paper_base_plus_liquid_anchorcompetitive` |  |  | anchor-competitive Liquid | epoch 72、18/18 TrackEval；finalizer已归档到`/autodl-fs/data/PairMOT_results/0718_02` |

2026-07-27 11:18在无卡模式完成数据盘整理。删除前逐项验证`0717_01`、`0718_02`、
`0719_01`和`0726_02`共享归档中的finalizer状态及全部`SHA256SUMS`；随后清理四个完整
workdir、过时smoke、epoch1中断目录、MMCV临时编译树和空launcher日志。数据盘由
`49GB/98%`降至`11GB/21%`，释放约`38GB`；保留完整HSMOT、13,713个真实GMC JSON、
AutoDL元数据和空`work_dirs`。审计位于
`/root/autodl-fs/PairMOT_maintenance/cleanup_20260727_111815`，相对路径哈希复核全部通过。
可复用流程已保存为`~/.codex/skills/pairmot-autodl-data-cleanup`。

## 维护规则

1. 新实验编号在所有服务器之间全局递增；当前最后分配编号为`0801_04`，下一编号为`0801_05`。
2. 新任务启动、停止、完成或迁移后，应同步更新本表和
   `docs/20260706_multi_server_experiment_plan.md`。
3. `RUNNING`和`QUEUED`状态以实际进程为准，不能仅依据workdir存在或旧screen名称判断。
4. 论文结果表只采用完整评估结果；中止训练和smoke/profile目录不得混入正式指标。
5. 当前Encoder目标严格固定`0723_01` Base+Liquid：唯一最佳epoch 64的cls HOTA为
   `53.955`、det HOTA为`62.032`。Liquid、PairDN、proposal、decoder、loss及论文数据协议
   均保持不变，只允许修改Encoder；成功条件是在同一个唯一最佳epoch上同时超过两个门槛。
   正式实验使用252/197双卡和178单卡，不使用研究代码重跑Base。
6. 正式结果使用
   `projects/multispec_pair_rotated_rtdetr/tools/summarize_pair_experiment.py WORK_DIR`
   复核。工具默认要求18个完整TrackEval目录、按payload真实step映射epoch、拒绝HOTA和并列，
   并从同一step的`val_det/metrics.json`读取AP；阶段诊断必须显式使用`--allow-partial`。
7. 用户于2026-07-27明确重新启用99进行两卡Encoder探索；当前仅使用GPU 0、1运行
   `0727_12`，GPU 2、3保持空闲。不得添加温度watchdog，也不得未经新指令在GPU 2、3并行
   启动第二个任务。
8. AutoDL不纳入自主排队或自动启动；只有用户明确介入并提供运行资源时才训练。其余时间
   只读取已有日志、checkpoint和评测产物进行监控，不因其他服务器任务结束而唤醒实例。
9. 197的Encoder阶段在`0727_09`结束后执行条件收口：若其未同时明显超过`0727_01`两侧
   HOTA，则转入Decoder阶段。首个Decoder实验`0728_01`固定`0727_01`的全部上游模块，只
   复用历史最佳`0708_03` tri-state zero-init decoder；不得混入新的Encoder、proposal或
   loss改动。
10. `0727_11`是AutoDL上的最后一个Encoder实验。整个Encoder阶段只等待已运行的99
    `0727_12`和252 `0727_04`完成，不再追加新Encoder结构；两项收尾后按统一选点规则确定
    最终Encoder，并继续已启动的Decoder主线。

2026-07-23 22:56阶段趋势（仅用于调度，不作为最终结果）：四个任务共同已有的epoch 36中，
`0723_01`共享DN噪声为`51.673/59.845`，`0723_02`独立DN噪声为`50.912/59.227`，
`0723_03`共享噪声+DSE为`53.553/60.704`，`0723_04`共享噪声+CSPR为
`52.506/59.513`（均为cls/det HOTA）。当前证据不支持独立pair噪声；DSE同时领先两个HOTA，
CSPR主要提高cls DetA但det AssA低于`0723_01`。较晚点中`0723_03`在epoch 52达到
`54.581/61.335`，相对`0723_01`同epoch为`+0.869/-0.227`，提示DSE后期的剩余问题是关联
质量而非分类覆盖。epoch 56继续得到`54.705/61.465`，相对`0723_01`同epoch
`53.922/61.721`为`+0.783/-0.256`，该分化具有连续性。所有判断必须等各自18/18评估后
重新计算，当前不据此提前启动新实验。

2026-07-24 04:13补充：`0723_02`截至epoch 64为`53.571/61.726`，相对Paper Base为
`+0.257/-0.256`，同点pair mAP/AP50为`0.3125/0.5267`；`0723_04`截至epoch 52为
`54.369/60.721`，相对Base为`+1.055/-1.261`。两者继续证明扩大帧间随机性或route描述
能力容易损害det关联。为利用252即将释放的双卡，新增`0723_07 PECG`：保持`0723_01`
共享PairDN和原Liquid结构，仅在谱段覆盖与Conv3D证据均一致的位置，对两帧SE gate做
保持pair均值的收缩。53项测试、完整模型构建及本机双卡4-iteration smoke已通过；252队列
在04:13正确报告`epoch72=no, eval=16/18`，没有提前占用GPU。

2026-07-23 23:29补充：`0723_02`独立pair噪声在epoch 52为`52.852/61.211`，相对
`0723_01`同epoch`53.712/61.562`双降`-0.860/-0.351`；其中cls/det AssA分别下降
`2.147/0.522`。连续评测证据已经足以停止继续探索独立pair噪声，后续PairDN保留两帧共享
相对扰动。

2026-07-24 00:22补充：`0723_03`在epoch 60为`54.699/61.695`，相对共享PairDN同epoch
`53.951/61.907`为`+0.748/-0.212`。DSE的HOTA和达到`116.394`，与`0719_06`
的`116.423`接近，但det仍未达标；类别上主要是car和tricycle AssA下降，而truck和bus
DetA明显提高。`0723_04`在epoch 40为`53.303/59.914`，相对共享PairDN为
`+0.810/-0.734`，且det AssA下降`1.500`。下一候选优先保留DSE的谱段离散度证据，但必须
用有界残差调制替代无约束mean/RMS替换；CSPR不作为主方向。

2026-07-24 00:54补充：`0723_01`在epoch 64达到`53.955/62.032`，已相对Paper Base唯一
最佳epoch 68的`53.314/61.982`双提升`+0.641/+0.050`，但尚未超过`0719_06`
`54.087/62.336`，也尚未完成18/18评测。`0723_02`在epoch 56仍相对共享噪声双降
`-0.590/-0.322`，其中cls AssA下降`1.321`，进一步确认独立pair噪声无效。

2026-07-24 01:26补充：`0723_03`epoch 64为`54.685/61.824`，相对共享PairDN同epoch
继续保持`+0.730/-0.208`，说明DSE增益与det代价在后期均稳定。`0723_04`epoch 44为
`53.728/60.314`，相对共享PairDN为`+0.269/-0.769`，其中det AssA下降`1.421`；CSPR
中后期未扭转关联损失。

2026-07-24 02:43补充：`0723_01`epoch 68为`53.847/62.060`，当前按HOTA和选取的阶段
唯一最佳仍是epoch 64的`53.955/62.032`。`0723_03`epoch 68达到`55.036/61.745`，
相对共享PairDN同epoch为`+1.189/-0.315`；其cls DetA增加`1.912`，但det AssA下降
`0.743`。DSE已有足够分类裕量，下一结构应以有界残差主动换回关联质量。`0723_02`
epoch 60为`53.610/61.561`，相对共享PairDN仍双降`-0.341/-0.346`；`0723_04`
epoch 48为`54.063/60.831`，相对共享PairDN为`+0.592/-0.349`。

2026-07-24 04:27补充：`0723_04`新增epoch 56结果`54.421/60.887`，相对Paper Base为
`+1.107/-1.095`。固定epoch 56对比直接父配置`0723_01`，cls/det HOTA为
`+0.499/-0.834`；det DetA仅下降`0.111`，det AssA下降`1.737`，IDSW由`1844`增加到
`2379`。CSPR的主要代价因此确定为帧间身份排序不稳定，而非检测覆盖不足；不再沿其route
preview方向扩展。正在排队的PECG保持原route，只收缩高一致性位置的SE gate差异，与该诊断
直接对应。

2026-07-24 05:32补充：CP-DSE首个epoch 4评测已完成。local `0723_05`为
`37.778/44.770`，pair-global `0723_06`为`37.388/44.660`；相对`0723_01`同epoch分别为
`+1.077/-0.240`与`+0.687/-0.350`。local的det AssA提高`0.502`、IDSW减少75，但det
DetA下降`0.698`，说明局部色散早期有利于关联但抑制检测覆盖；pair-global同时损失det
DetA/AssA。两组8参数残差仍小于`0.005`，首点不足以判定最终性能，继续训练。`0723_02`
epoch 68为`53.609/61.680`，阶段唯一最佳仍是epoch 64的`53.571/61.726`。

2026-07-24 06:05补充：`0723_04`epoch 60为`54.231/61.263`，相对`0723_01`同epoch为
`+0.280/-0.644`。det DetA仅下降`0.074`，det AssA下降`1.290`，IDSW增加513；CSPR的
关联损失连续存在，不再作为后续结构基础，但保留运行至完整18/18用于正式报告。

2026-07-24 06:47补充：`0723_02`完成72 epochs和18/18 TrackEval，唯一最佳epoch 72为
`53.637/61.679`，相对Paper Base为`+0.323/-0.303`；相对共享噪声父配置同epoch为
`+0.043/-0.333`。其det AssA提高`0.102`，但det DetA下降`0.552`且FP增加2748，证明独立
pair噪声主要损害检测精度。`0723_07 PECG`随后完成远端精确smoke并于06:42双卡fresh启动，
正式iter 200及全部五项启动门槛通过。同期`0723_06`epoch 8达到`44.661/50.551`，相对父
配置同epoch双提升`+0.349/+0.714`，det DetA/AssA提高`0.441/1.150`，是当前最积极趋势。

## 2026-07-30 18:25 CST decoder 调度覆盖

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 197 GPU 4,5 | `0730_09_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_motiontrust_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 18:23 /  | 严格继承 `0727_01`，新增检测置信度门控、位移包络约束的反对称 motion-trust decoder。配置深拷贝、47 项单测和双卡真实数据 4-iter smoke 通过；18:24 到 epoch 1 iter 50，`0.9085 s/iter`、loss `21.3915`、grad norm `120.0892`，正式启动五项门槛全部通过。 |
| STOPPED | 197 GPU 4,5 | `0730_08 ... decoder_competitiveevidence ... fresh` | 2026-07-30 / 2026-07-30 17:58 | epoch 4 cls/det HOTA `36.277/43.729`；相对 `0727_01` 同点 `+0.068/+4.976`，但增益集中于 det AssA `+9.441`，pair mAP `-0.0041`，完成 checkpoint、检测及 2/2 TrackEval 后按门槛停止。 |
| STOPPED | 252 GPU 0,1 | `0730_07 ... decoder_commonmotion_sharedevidence ... fresh` | 2026-07-30 / 2026-07-30 17:58 | epoch 4 cls/det HOTA `36.926/42.702`；相对 `0727_01` 同点 `+0.717/+3.949`，但 cls/det DetA 均约 `-1.0`，pair mAP/AP50 和 both-independent AP50 未形成一致提升，完成 checkpoint、检测及 2/2 TrackEval 后停止。 |

当前可调度资源为 99 双卡、197 双卡、252 双卡、178 单卡；AutoDL 全部关机。197 新实验使用干净副本 `/data/users/litianhao/PairMOT_sync_3cb888d`，历史目录不作代码源。`0730_09` 代码提交为 `f231b01`；99 canonical 与 197 已精确同步，252/178 在状态文档提交后同步到同一提交。空闲卡只用于模型结构实验，不进行 class-specific/long-tail reweight 或 residual-scale 参数扫描。

## 2026-07-30 18:46 CST decoder 资源与启动状态

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 197 GPU 4,5 | `0730_09_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_motiontrust_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 18:23 /  | 197 两卡已正式纳入当前 decoder 资源池并由本实验占用。18:45 到 epoch 1 iter 900，约 `1.5280 s/iter`、loss `12.1506`、grad norm `21.8685`；总 loss、DN loss 与 encoder loss 均有限，无 Traceback、OOM、NaN、NCCL 或 unused-parameter 错误。GPU 4/5 各约 `19.2 GiB`。首个结构判断点为 epoch 4。 |
| RUNNING | 252 GPU 0,1 | `0730_10_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_symmetricpair_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 18:41 /  | 严格继承 `0727_01`，decoder 两帧共享 deformable cross-attention 权重，并对 cross-frame feature fusion 与 pair-position fusion 显式平均正反两种帧序；不增加参数，不修改 encoder、proposal、PairDN、head、loss 或训练协议。52 项 decoder 单测、配置深拷贝、launcher shell 审计和双卡真实数据 4-iter smoke 通过；预训练权重中 24 组 attention 参数逐项相等，smoke checkpoint 中 4 个 fusion 矩阵的帧交换误差最大为 `8.45e-09`。正式训练五项启动门槛通过，18:45 到 epoch 1 iter 200，约 `1.0837 s/iter`、loss `18.9054`、grad norm `56.8993`，GPU 0/1 各约 `19.2 GiB`，无异常。首个结构判断点为 epoch 4。 |

当前资源池确认为 99 双卡、197 双卡、252 双卡、178 单卡；其中 197 GPU 4/5 与 252 GPU 0/1 正在运行结构实验，99 GPU 0/1 与 178 GPU 0 保留给 epoch 4 诊断后有明确模型依据的下一候选，不以参数扫描填卡。代码提交 `5341c32` 已通过带 prerequisite 的 Git bundle 精确同步到 99、197、252 和 178；各服务器既有未跟踪文件未覆盖或删除。

## 2026-07-30 20:29 CST decoder epoch 4 门控覆盖

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 197 GPU 4,5 | `0730_09_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_motiontrust_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 18:23 /  | epoch 4 checkpoint、检测、结构审计及 2/2 TrackEval 完成并通过固定门槛。cls HOTA/DetA/AssA `37.878/27.722/55.486`，det `44.102/34.288/58.241`；相对 `0727_01` 同点 HOTA `+1.669/+5.349`、DetA `+0.654/+1.834`。pair mAP/AP50 `0.16257/0.31578`，both-independent mAP/AP50 `0.18872/0.34231`，均高于父配置。三层 motion-trust adapter 最大绝对权重 `0.14190/0.13821/0.12903`，均有限且非零。继续训练到 epoch 8；20:27 到 epoch 5 iter 350，训练健康。 |
| STOPPED | 252 GPU 0,1 | `0730_10_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_symmetricpair_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 18:41 / 2026-07-30 20:12 | epoch 4 完成 checkpoint、检测和 2/2 TrackEval。cls HOTA/DetA/AssA `36.750/26.756/54.632`，det `42.604/33.160/55.890`；pair mAP/AP50 `0.1496/0.3108`，both-independent mAP/AP50 `0.1786/0.3400`。虽 det HOTA/DetA 与 AP50 提高，但 pair mAP 相对父配置下降 `0.00765`，未通过 `0.003` 保护线，因此停止并释放 GPU。共享 attention 结构保持，fusion 半矩阵 `1.788e-7` 差异属于 FP32 漂移，不是前向交换等变失效。 |

当前可调度资源仍为 99 GPU 0/1、197 GPU 4/5、252 GPU 0/1、178 GPU 0，共 7 张；197 GPU 4/5 正由 `0730_09` 占用，99 GPU 0/1、252 GPU 0/1 与 178 GPU 0 空闲。99 GPU 2 和 197 其他 GPU 不属于本实验调度范围。AutoDL 全部关机。四台服务器的目标代码提交均为 `fbbf137`，已有未跟踪文件继续保留。

## 2026-07-30 20:52 CST decoder 并行启动

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 197 GPU 4,5 | `0730_09_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_motiontrust_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 18:23 /  | epoch 4 已通过固定门槛并继续到 epoch 8。20:52 到 epoch 6 iter 250，约 `1.5907 s/iter`、loss `11.0446`、grad norm `40.5960`；总、DN、encoder loss 均有限，GPU 4/5 正常占用。 |
| RUNNING | 252 GPU 0,1 | `0730_11_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedrouting_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 20:49 /  | 严格继承 `0727_01`，只在每层 decoder 两帧 deformable cross-attention 之间共享 `sampling_offsets` 与 `attention_weights`；`value_proj` 和 `output_proj` 保持独立，避免 `0730_10` 全共享导致的 pair mAP 损失。56 项 decoder 单测、配置深拷贝、权重/GMC/路径审计及双卡真实数据 4-iter smoke 通过；smoke checkpoint 的 12 组 routing 最大误差为 `0`，12 组独立 projection 最大差异为 `7.7571e-4`。正式 fresh 训练五项启动门槛通过，20:52 到 epoch 1 iter 150，约 `1.1223 s/iter`、loss `19.6946`、grad norm `122.9360`，总、DN、encoder loss 均有限，无 Traceback、OOM、NaN、NCCL、DDP reduction 或 unused-parameter 错误。首个判断点为 epoch 4。 |

`0730_11` 代码提交为 `9049422`，通过带 prerequisite 的 Git bundle 精确同步至 99、197、252 和 178；各服务器原有未跟踪文件均保留。当前形成两路正式并行：197 的 motion-trust 与 252 的 shared-routing。178 GPU 0 仍空闲，等待另一项有独立模型依据的 decoder 结构；99 的双卡额度已恢复，但实时 GPU 0 和 GPU 2 被其他用户的 UNet 任务占用，仅 GPU 1 空闲，未将其错误计作可立即使用的双卡 DDP 资源。

## 2026-07-30 21:13 CST 第三路结构实验启动

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 178 GPU 0 | `0730_12_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_motiontrust_sharedevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` | 2026-07-30 21:10 /  | 组合 `0730_09` 的 detection-confident bounded motion-trust 与 `0730_06` 的 swap-invariant shared-evidence。两者分别作用于反对称框更新和共享 query，保持 `0727_01` 的 encoder、proposal、PairDN、head、loss、数据与训练协议不变。57 项 decoder 单测、配置深拷贝、路径/权重/GMC/launcher 审计通过。单卡 physical batch 8 的真实数据 4-iter smoke 中总、DN、encoder loss 与 grad norm 均有限；checkpoint 三层 motion-trust 最大绝对权重为 `3.7246e-4/3.6952e-4/3.6620e-4`，三层 shared-evidence 为 `3.9646e-4/3.9722e-4/3.9787e-4`，均有限且非零。正式 fresh 训练五项启动门槛通过，21:12 到 epoch 1 iter 100，约 `0.8798 s/iter`、loss `20.1321`、grad norm `90.7448`；GPU 0 设备占用约 31.4 GiB，无 Traceback、OOM、NaN、DDP reduction 或 unused-parameter 错误。首判 epoch 4。 |

当前形成三路正式并行：197 `0730_09`、252 `0730_11`、178 `0730_12`。99 没有漏排：
虽然双卡额度已恢复，但实时 GPU 0/2 被其他用户的 UNet 任务各占约 21 GiB，仅 GPU 1
空闲；不得终止或挤占外部任务，也不得将单张空闲卡误报为双卡 DDP 资源。四台服务器代码已
同步到 `f1ffbd7`，其中 `c43635c` 引入组合结构，`f1ffbd7` 使可信本地 smoke checkpoint
兼容 PyTorch 2.6+ 的显式非 weights-only 加载；各服务器既有未跟踪文件均保留。

## 2026-07-30 22:35 CST shared-routing 门控与 shared-attention 接替

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 197 GPU 4,5 | `0730_09_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_motiontrust_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 18:23 /  | epoch 8 checkpoint 已产生，训练到 epoch 9 iter 1000；总、DN、encoder loss 和梯度有限。等待 epoch 8 完整检测、2/2 TrackEval 与结构审计后按固定同点门槛决定继续或停止。 |
| RUNNING | 178 GPU 0 | `0730_12_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_motiontrust_sharedevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` | 2026-07-30 21:10 /  | epoch 4 checkpoint 已产生，训练到 epoch 6 iter 100；等待 epoch 4 完整检测、2/2 TrackEval 与两组 adapter 结构审计。 |
| STOPPED | 252 GPU 0,1 | `0730_11_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedrouting_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 20:49 / 2026-07-30 22:19 | epoch 4 完整 artifacts 后停止。cls HOTA/DetA/AssA `36.504/27.255/51.884`，det `42.163/33.695/53.992`，pair mAP/AP50 `0.149753/0.305401`，both-independent mAP/AP50 `0.178556/0.335452`。pair mAP 相对父配置下降 `0.007499`，未通过保护线；结构审计通过。 |
| RUNNING | 252 GPU 0,1 | `0730_13_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 22:32 /  | 只共享每层两帧 cross-attention 的 `attention_weights`，保留独立 `sampling_offsets/value_proj/output_proj`。61 项 decoder 单测、配置/launcher 审计和双卡真实数据 4-iter DDP smoke 通过；smoke checkpoint 的 6 组 attention 误差为零，18 组应独立参数已分化。正式训练五项门槛通过，22:35 到 epoch 1 iter 100，约 `1.1160 s/iter`、loss `20.5979`、grad norm `103.9917`，GPU 0/1 各约 `19.2 GiB`，无异常。首判 epoch 4。 |

99 在 22:31 的实时采样中三张 GPU 均为约 `10 MiB/0%`，此前“仅 GPU 1 空闲”的判断已失效。
当前不复制已有实验或用参数扫描填卡；99 两卡仅分配给与 197/178/252 不重复且经过完整
配置、单测和真数据 smoke 的新 decoder 结构。99 canonical 与 252 已位于 `091af97`；
197/178 的当前进程保持启动时代码，完成门控后再快进同步。

## 2026-07-30 22:46 CST 第四路结构实验启动

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 99 GPU 0,1 | `0730_14_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_motiontrust_sharedattention_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 22:45 /  | 组合 `0730_09` 的 detection-confident bounded motion-trust 与 `0730_13` 的 frame-localized shared attention。前者作用于有界反对称框修正，后者只共享 attention 权重并保留逐帧 sampling offsets/value/output projection；构成主效应与交互项，不改 encoder、proposal、PairDN、head、loss、数据或训练协议。62 项 decoder 单测、正式/短测配置深拷贝、launcher 审计与双卡真实数据 4-iter DDP smoke 通过。smoke checkpoint 的三层 motion adapter 均非零，6 组 attention 误差为零，18 组独立参数最大差异 `7.9339e-4`。正式训练五项门槛通过，22:46 到 epoch 1 iter 50，约 `0.9575 s/iter`、loss `21.4104`、grad norm `100.1361`，GPU 0/1 各约 `19.2 GiB`，无异常。首判 epoch 4。 |

至此四台服务器均有独立 decoder 结构实验：197 `0730_09`、178 `0730_12`、
252 `0730_13`、99 `0730_14`。当前 canonical 代码提交为 `b829704`；完成本状态提交后
以 Git bundle 快进同步到 252/197/178，保留各服务器未跟踪文件且不重启已有训练。

## 2026-07-30 23:00 CST epoch 门控与 197 接替实验

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| STOPPED | 197 GPU 4,5 | `0730_09_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_motiontrust_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 18:23 / 2026-07-30 22:52 | epoch 8 完整 artifacts 与结构审计后停止。cls HOTA/DetA/AssA `45.498/36.788/58.984`，det `51.160/45.018/60.159`，pair mAP/AP50 `0.230058/0.430536`，both-independent mAP/AP50 `0.265862/0.464984`。相对父配置 HOTA 小幅上升，但 cls/det DetA 下降 `0.875/2.043`、pair mAP 下降 `0.007676`，为 AssA 搬运而非共同改善。精确终止后 GPU 4/5 均为 `1 MiB/0%`。 |
| RUNNING | 178 GPU 0 | `0730_12_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_motiontrust_sharedevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` | 2026-07-30 21:10 /  | epoch 4 cls HOTA/DetA/AssA `40.559/29.273/59.330`，det `46.017/34.057/64.397`，pair mAP/AP50 `0.187248/0.342315`，both-independent mAP/AP50 `0.213887/0.366343`；HOTA、DetA、pair mAP 与 both AP50 均高于父配置。motion-trust 与 shared-evidence 三层权重均有限非零，结构门控通过，继续到 epoch 8。 |
| RUNNING | 197 GPU 4,5 | `0730_15_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedevidence_sharedattention_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 22:58 /  | epoch 4 checkpoint、检测、完整 TrackEval 与结构审计通过。cls HOTA/DetA/AssA `36.732/27.680/51.849`，det `41.818/33.239/53.766`；pair mAP/AP50 `0.157715/0.320595`，both-independent mAP/AP50 `0.186071/0.345886`。相对父配置 cls/det HOTA `+0.523/+3.065`、DetA `+0.612/+0.785`，全部固定保护线通过，继续到 epoch 8；但同点低于 `0730_13`，shared-evidence 暂未显示正交增益。 |

当前四路正式训练为 99 `0730_14`、197 `0730_15`、252 `0730_13`、178 `0730_12`。
canonical 代码提交为 `0782826`；本次状态提交后四机统一快进，保留所有既有 artifacts 和
未跟踪目录。

## 2026-07-30 23:45 CST 178 门控失败与结构接替

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| STOPPED | 178 GPU 0 | `0730_12_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_motiontrust_sharedevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` | 2026-07-30 21:10 / 2026-07-30 23:30 | epoch 8 checkpoint、检测、2/2 TrackEval 与结构审计完整。cls HOTA/DetA/AssA `44.387/33.262/61.885`，det `49.824/40.708/63.485`，pair mAP/AP50 `0.2144/0.3809`，both-independent mAP/AP50 `0.2457/0.4098`。相对父配置 cls/det HOTA `-0.882/-0.369`、DetA `-4.401/-6.353`，pair mAP `-0.02333`、both AP50 `-0.05615`；AssA 增益不足以抵消检测损失，按固定保护线停止并释放 GPU。 |
| RUNNING | 178 GPU 0 | `0730_16_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_antisymmetricdetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` | 2026-07-30 23:48 /  | 接替结构只在逐帧 cls/reg head 前注入由真实双帧 cross-attention 证据生成的有界 `-detail/+detail`，保持 recurrent shared query、下一层路径和两帧特征中点严格等于父模型；不直接改框或共享 query。69 项 decoder 单测、配置深拷贝、launcher 审计和代码差异检查通过；真实数据 4-iter smoke 的总、DN、encoder loss 和 grad norm 均有限，checkpoint 三层 adapter 均非零且结构检查通过。正式训练五项门槛通过，23:49 到 epoch 1 iter 50，约 `0.9458 s/iter`、loss `21.1606`、grad norm `107.4478`，GPU 0 约 `31.4 GiB`，无异常。首判 epoch 4。 |

实时资源状态已恢复四路正式训练：252 `0730_13`、99 `0730_14`、197 `0730_15`、
178 `0730_16`。四台服务器结构基线为 `1112a56`，状态记录已同步至 `770eae7`；
各服务器既有未跟踪文件均保留，活动训练未因同步重启；AutoDL 继续保持全部关机。

## 2026-07-31 00:02 CST 252 epoch-4 门控

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 252 GPU 0,1 | `0730_13_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 22:32 /  | epoch 4 checkpoint、检测、完整 TrackEval 与结构审计通过。cls HOTA/DetA/AssA `37.559/27.119/55.846`，det `43.257/33.530/56.895`；pair mAP/AP50 `0.1547/0.3182`，both-independent mAP/AP50 `0.1839/0.3467`。相对父配置 cls/det HOTA `+1.350/+4.504`、DetA `+0.051/+1.076`；pair mAP 仅下降约 `0.00255`，未超过 `0.003` 保护线，both AP50 提高约 `0.0235`。6 组共享 attention 权重误差为零，18 组独立参数最大差异 `0.03846`。全部固定门槛通过，继续到 epoch 8。 |

99 `0730_14` 的 epoch 4 完整 TrackEval 与结构审计也已通过：cls
HOTA/DetA/AssA `37.075/27.355/53.989`，det `42.159/32.966/55.263`，
pair mAP/AP50 `0.1625/0.3105`，both-independent mAP/AP50 `0.1887/0.3369`；
全部保护线通过，继续到 epoch 8。但其 cls/det HOTA 同点低于 `0730_13`
`0.484/1.098`，motion-trust 组合暂未形成正交增益。197 `0730_15` 的 epoch 4
完整门控随后也通过并继续到 epoch 8；178 `0730_16` 正常训练。

## 2026-07-31 00:24 CST 197 epoch-4 门控

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 197 GPU 4,5 | `0730_15_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedevidence_sharedattention_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 22:58 /  | epoch 4 checkpoint、检测、完整 TrackEval 与联合结构审计已完成。cls HOTA/DetA/AssA `36.732/27.680/51.849`，det `41.818/33.239/53.766`；pair mAP/AP50 `0.157715/0.320595`，both-independent mAP/AP50 `0.186071/0.345886`。相对父配置 HOTA、DetA 与 AP 保护线全部通过，继续到 epoch 8。三层 shared-evidence adapter 均非零，6 组共享 attention 误差为零，18 组独立参数最大差异 `0.036013`。同点 cls/det HOTA 低于 `0730_13` `0.827/1.439`，因此 shared-attention 仍是当前主要早期增益来源。 |

当前四路并行保持：252 `0730_13`、99 `0730_14`、197 `0730_15` 均继续到
epoch 8；178 `0730_16` 首判 epoch 4。178 的 warmup grad norm 曾短时升至约
`543`，但到 epoch 2 iter 350/400 已回落至 `28.2/25.3`，loss 同步下降且无
NaN/OOM/Traceback，确认不是持续性训练异常。

## 2026-07-31 01:13 CST 178 接替与四路并行恢复

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| STOPPED | 178 GPU 0 | `0730_16_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_antisymmetricdetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` | 2026-07-30 23:48 / 2026-07-31 00:59 | epoch 4 cls HOTA/DetA/AssA `36.684/27.590/52.398`，det `39.221/31.788/49.436`；pair mAP/AP50 `0.1700/0.3110`，both-independent mAP/AP50 `0.1992/0.3428`。HOTA/AP 提高，但 det DetA 相对父配置下降 `0.666`，超过 `0.5` 门槛 `0.166`，因此精确停止。 |
| RUNNING | 178 GPU 0 | `0731_01_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_antisymmetricdetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` | 2026-07-31 01:11 /  | epoch 4 checkpoint、检测、完整 TrackEval 与两项结构审计均完成。cls HOTA/DetA/AssA `37.590/28.607/52.920`，det `40.313/33.923/49.759`；相对 `0727_01` 同点 HOTA `+1.381/+1.560`、DetA `+1.539/+1.469`。pair mAP `0.173430`、both-independent AP50 `0.356102`，分别提高 `0.016177/0.032953`；全门槛通过，继续到 epoch 8。 |

四路结构实验保持并行：99 `0731_02` 已到 epoch 3，252 `0731_03` 与 197
`0731_04` 均接近 epoch 3，178 `0731_01` 已通过 epoch 4 全门槛并进入 epoch 5。
四台代码和记录统一到 `7e184b1`，同步未重启任何在途训练。

## 2026-07-31 03:08 CST 99 epoch-4 全门槛

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 99 GPU 0,1 | `0731_02_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_envelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 01:44 /  | epoch 4 checkpoint、检测、完整 TrackEval 与结构审计均完成。cls HOTA/DetA/AssA `37.859/28.112/54.688`，det `42.873/34.173/55.071`；相对 `0727_01` 同点 HOTA `+1.650/+4.120`、DetA `+1.044/+1.719`。pair mAP `0.167896`、both-independent AP50 `0.349436`，分别提高 `0.010643/0.026287`。三层 enveloped-detail 权重均有限非零，全门槛通过，继续到 epoch 8。 |

当前仍为四路正式结构实验并行：99 `0731_02` 与 178 `0731_01` 已通过 epoch 4
全门槛并继续到 epoch 8；252 `0731_03`、197 `0731_04` 正等待 epoch 4 完整评估。

## 2026-07-31 03:12 CST 197 epoch-4 全门槛

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 197 GPU 4,5 | `0731_04_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_orthogonalevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 01:52 /  | epoch 4 checkpoint、检测、完整 TrackEval 与结构审计均完成。cls HOTA/DetA/AssA `36.831/27.861/52.246`，det `43.581/34.573/56.147`；相对 `0727_01` 同点 HOTA `+0.622/+4.828`、DetA `+0.793/+2.119`。pair mAP `0.161207`、both-independent AP50 `0.346363`，分别提高 `0.003954/0.023214`。两类三层门控均有限非零，全门槛通过，继续到 epoch 8。 |

99、197、178 三路已通过 epoch 4 全门槛并继续到 epoch 8；252 `0731_03`
仍在等待完整 epoch 4 评估，四台训练进程均保持运行。

## 2026-07-31 03:16 CST 252 epoch-4 门控与接替

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| STOPPED | 252 GPU 0,1 | `0731_03_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_commonevidencebypass_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 01:44 / 2026-07-31 03:15 | epoch 4 checkpoint、检测、完整 TrackEval 与结构审计均完成。cls HOTA/DetA/AssA `36.564/27.324/52.415`，det `43.279/34.694/55.415`；HOTA/DetA 均提高，但 pair mAP `0.153565` 相对父配置下降 `0.003688`，超过固定 `0.003` 保护线 `0.000688`。三层门控均有限非零，按门槛精确停止，GPU 0/1 已释放。 |
| PREPARED | 252 GPU 0,1 | `0731_05_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_envelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` |  /  | 接替结构组合 shared-attention 与受真实帧差包络约束的 head-only 细节，不含 common-evidence bypass、loss 调权、类别 reweight 或 residual-scale；等待单测、配置审计和真实数据 smoke。 |

当前 99、197、178 三路正式训练继续；252 已按门槛停止失败候选并进入结构接替验证。

## 2026-07-31 03:25 CST 252 接替实验启动

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 252 GPU 0,1 | `0731_05_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_envelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 03:23 /  | 79 项 decoder 单测、配置深拷贝、完整模型构建、双卡真实数据 4-iter smoke 与组合 checkpoint 审计通过。正式训练 03:25 到 epoch 1 iter 50：`1.1589 s/iter`、loss `21.4509`、grad norm `104.0930`，总/DN/encoder loss 有限；GPU0/1 各约 `19.2 GiB`、100% 利用，无 Traceback/OOM/NaN/NCCL。五项启动门槛通过，首判 epoch 4。 |

四路正式结构实验再次并行：252 `0731_05` 首判 epoch 4；99 `0731_02`、
197 `0731_04`、178 `0731_01` 继续各自的 epoch 8 持续性检查。

## 2026-07-31 03:40 CST 178 Epoch-8 Gate 与结构接替

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| STOPPED | 178 GPU 0 | `0731_01_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_antisymmetricdetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` | 2026-07-31 01:11 / 2026-07-31 03:37 | epoch 8 完整 artifacts 与结构审计完成。cls HOTA/DetA/AssA `45.152/37.611/57.666`，det `50.817/46.745/57.206`；相对父配置 cls HOTA `-0.117`、det HOTA `+0.624`，pair mAP `+0.008558`、both-independent AP50 `+0.013936`。因 cls HOTA 唯一低于父配置，按固定门槛精确停止。 |
| PREPARED | 178 GPU 0 | `0731_06_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_regressionenvelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` |  /  | 保留 shared-attention 分类共享路径，将零起点、受观测帧差包络约束的 swap-odd 细节只注入框回归/reference 更新；待单测、完整模型构建与真数据 4-iter smoke 后启动。 |

当前 99、197、252 三路正式训练不受影响；178 GPU 0 已释放并用于上述结构性接替，
不是 residual-scale、loss 权重或类别 reweight 调参。

## 2026-07-31 03:47 CST 178 接替实验正式启动

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 178 GPU 0 | `0731_06_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_regressionenvelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` | 2026-07-31 03:45 /  | 80 项单测、完整模型构建、单卡真数据 4-iter smoke 与组合结构审计通过；smoke 三层门控均获非零更新。03:46 到 epoch 1 iter 50：`0.9610 s/iter`、loss `20.9556`、grad norm `111.7257`，总/DN/encoder loss 有限，无训练异常，五项启动门槛通过。 |

四路正式结构实验恢复并行：99 `0731_02`、197 `0731_04` 继续 epoch 8，
252 `0731_05` 和 178 `0731_06` 首判 epoch 4。四台代码统一到 `62f1028`，
同步没有重启任何在途训练。

## 2026-07-31 04:35 CST epoch-8 淘汰与四机并行恢复

- 99 `0731_02 enveloped-detail` 的 epoch 8 完整结果为：cls HOTA/DetA/AssA
  `45.128/37.117/57.200`，det `50.223/44.663/58.189`，pair mAP/AP50
  `0.236938/0.434026`，both-independent mAP/AP50 `0.275284/0.473473`。
  相对 `0727_01` 同点，cls HOTA `-0.141`、cls DetA `-0.546`、det DetA
  `-2.398`，仅 det AssA `+3.044`；HOTA/DetA 门槛失败，完成全部 artifacts 后停止。
- 197 `0731_04 orthogonal-evidence` 的 epoch 8 完整结果为：cls HOTA/DetA/AssA
  `44.760/36.328/57.530`，det `49.732/44.183/57.992`，pair mAP/AP50
  `0.226459/0.424496`，both-independent mAP/AP50 `0.263142/0.463058`。
  相对父配置 cls/det HOTA 为 `-0.509/-0.461`，cls/det DetA 为
  `-1.335/-2.878`，pair mAP `-0.011275`；共同证据旁路进一步把检测覆盖搬运到
  AssA，故淘汰。评估完成后旧 launcher 曾继续到 epoch 9 iter 400 左右，04:22
  审计发现后精确停止；epoch 9 未形成新判断点，也不用于任何结论。
- 两台暂时空闲不是资源不足，而是两个候选刚完成固定 epoch-8 淘汰。接替结构采用同一
  `classification_enveloped_detail_decoder` 开关：框回归和迭代 reference 严格保持
  共享 decoder 路径，仅分类状态接收零起点、swap-odd、逐元素受观测帧差包络约束的细节。
  99 `0731_07` 不共享 attention，用于测量分类专用细节的主效应；197 `0731_08`
  叠加 shared-attention，用于检验能否保留其 det 增益并补回 cls。它们与 252 的
  full-path `0731_05`、178 的 regression-only `0731_06` 构成可归因的结构路径拆分，
  不含 scale、loss 权重或类别重权扫描。
- 82 项 decoder 单测全部通过；两个正式配置均通过深拷贝和完整模型构建。99/197
  双卡真数据 4-iter smoke 均生成 `iter_4.pth`，总、DN、encoder loss 和 grad norm
  有限。99 三层门控最大权重为 `0.000304/0.000270/0.000287`；197 的 6 组共享
  attention 误差为零、18 组独立参数最大差异 `0.000794`，三层门控为
  `0.000283/0.000295/0.000290`。
- 正式 fresh 训练均基于提交 `7dee533`：99 GPU 0/1 的 `0731_07` 于 04:34 到
  iter 50，`0.9703 s/iter`、loss `21.4241`、grad norm `112.0013`；197 仅使用
  GPU 4/5 的 `0731_08` 已越过 iter 100，iter 100 为 `0.8708 s/iter`、loss
  `20.5811`、grad norm `114.1662`。两项均无 Traceback、OOM、NaN、NCCL 或
  unused-parameter 错误。99、197、252、178 四台现均为 RUNNING，首判 epoch 4。

## 2026-07-31 05:44 CST 197 epoch-4 淘汰

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| STOPPED | 197 GPU 4,5 | `0731_08_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_classificationenvelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 04:33 / 2026-07-31 05:44 | epoch 4 checkpoint、检测、完整 TrackEval、原始 CSV 和结构检查全部完成。cls HOTA/DetA/AssA `36.312/27.192/52.399`，det `42.336/33.389/54.669`；pair mAP/AP50 `0.153285/0.312909`，both-independent mAP/AP50 `0.182831/0.343821`。双HOTA与DetA通过，但pair mAP相对父配置下降 `0.003968`，超过固定保护线，故精确停止。共享attention误差为零、独立参数及三层分类门控均已学习。 |

197 GPU 4/5 已释放。因99 `0731_07` 是同一分类专用细节的无 shared-attention
主效应对照，先等待其完整epoch 4结果，再决定197接替结构，避免在因果证据缺失时
启动新的参数或模块拼接实验。

## 2026-07-31 06:02 CST HOTA 优先门槛与四机并行恢复

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 252 GPU 0,1 | `0731_05_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_envelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 03:23 /  | epoch 4 双 HOTA 通过，继续到 epoch 8。 |
| RUNNING | 178 GPU 0 | `0731_06_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_regressionenvelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` | 2026-07-31 03:45 /  | epoch 4 双 HOTA 通过，继续到 epoch 8。 |
| RUNNING | 197 GPU 4,5 | `0731_08_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_classificationenvelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 04:33；06:00 resumed /  | 将 AP 降为诊断项后，从原 epoch 4 checkpoint 原位恢复；已确认 epoch 5 iter 100，GPU 4/5 正常。 |
| RUNNING | 99 GPU 0,1 | `0731_09_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_regressionenvelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 06:02 /  | 178 `0731_06` 的 2xb4 复现；84 项单测、模型构建、真实数据 smoke 与结构 checkpoint 检查通过，正式 fresh 启动。 |
| STOPPED | 99 GPU 0,1 | `0731_07_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_classificationenvelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 04:33 / 2026-07-31 05:50 | epoch 4 cls HOTA/DetA `35.533/26.341` 均低于父配置，pair mAP 下降 `0.006227`；即使 det HOTA 提高，仍不满足双 HOTA 目标，完成全部 artifacts 后停止。 |

后续筛选以同 epoch cls HOTA 与 det HOTA 为一级标准，DetA/AssA解释来源，AP仅诊断
明显检测崩塌。最终 decoder 只有同时超过 encoder `0727_01` 的
cls/det HOTA `54.437/62.393`，才进入论文性能递进主线。

## 2026-07-31 06:08 CST 178 epoch-8 结果与接替

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| STOPPED | 178 GPU 0 | `0731_06_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_regressionenvelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` | 2026-07-31 03:45 / 2026-07-31 06:08 | epoch 8 cls/det HOTA `44.398/48.552`，相对父配置同点下降 `0.871/1.641`；pair mAP `0.221531`、both-independent AP50 `0.430058`。完整 TrackEval、原始 CSV、检测 metrics 与结构审计齐全；双 HOTA 主门槛失败后停止。异步评估期间误入 epoch 9，其迭代不参与结论。 |
| RUNNING | 178 GPU 0 | `0731_11_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_midpointregressionenvelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` | 2026-07-31 06:11 /  | 分类保持共享；帧细节在 5D box-logit residual 空间严格反对称，新增修正的 pair midpoint 为零。配置构建、代码测试、真数据 smoke 与结构 checkpoint 检查通过；06:12 达到 iter 50，`0.9405 s/iter`、loss `20.9608`，无异常。 |

## 2026-07-31 06:17 CST 252 epoch-8 阶段结果

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 252 GPU 0,1 | `0731_05_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_envelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 03:23 /  | epoch 8 cls HOTA/DetA/AssA `45.341/37.690/56.836`，det `51.589/45.176/60.817`；相对父配置双 HOTA `+0.072/+1.396`。pair mAP `0.2380`、both-independent AP50 `0.4738`。完整证据与结构审计通过，按 HOTA 主门槛保留；det 增益主要由 AssA 驱动、DetA 低 `1.885`，后续需持续观察。 |

## 2026-07-31 06:35 CST 四路进度与 terminal-only 候选

- 四路训练在代码同步前后均连续运行：99 `0731_09` 到 epoch 2 iter 900，197
  `0731_08` 到 epoch 7 iter 200，252 `0731_05` 到 epoch 10 iter 250，178
  `0731_11` 到 epoch 2 iter 450；总、DN、encoder loss 与 grad norm 均有限。
- 下一决策点依次为 197 epoch 8、99/178 epoch 4、252 epoch 12。197 若双 HOTA
  失败，优先使用已完成配置准备的 `0731_10 midpoint-regression` 双卡复现。
- 新增 `terminal_enveloped_detail_decoder`：前两层 decoder、辅助输出及 iterative
  references 保持共享父路径，只在最后一层的逐帧分类和回归输出前注入零起点、
  swap-odd、受真实帧差包络约束的细节。该结构直接隔离全路径 detail 的逐层
  reference 递归污染，不是 scale、loss 权重或类别重权扫描。
- 86 项 decoder 单测、配置深拷贝、完整模型构建及 detector-level 初始化后的零门控
  均已通过。此候选尚未分配实验编号、正式配置或 GPU，仅在现有评估提供支持且资源
  释放后进入 smoke。四台代码已无重启地快进到提交 `764ff7d`，未跟踪文件均保留。

## 2026-07-31 07:25 CST 197 淘汰与 midpoint 双卡接替

- 197 `0731_08 shared-attention + classification-only enveloped-detail` 的 epoch 8
  checkpoint、检测 metrics、完整 TrackEval、原始 CSV 和结构审计均已完成。
  cls HOTA/DetA/AssA 为 `43.801/35.436/56.745`，det 为
  `49.318/43.792/57.563`；相对 `0727_01` 同点的双 HOTA 分别下降
  `1.468/0.875`，cls/det DetA 分别下降 `2.227/3.269`。pair mAP
  `0.213045`、both-independent AP50 `0.436989` 也明显低于父配置。
- checkpoint 中 6 组共享 attention 最大误差为零，18 组独立参数最大差异
  `0.067752`，三层分类细节门最大权重
  `0.075650/0.106004/0.114755`。结构已充分学习，失败原因是分类专用细节使
  检测覆盖下降，而不是模块未生效。07:14 精确停止，GPU 4/5 释放。
- 接替实验 `0731_10` 将帧细节放在 5D box-logit residual 空间形成严格
  `-detail/+detail` 修正，使新增回归细节不移动 pair midpoint；分类路径继续共享。
  正式配置深拷贝、完整模型构建、launcher 语法和目标资源均通过复核。
- 197 双卡真实数据 4-iter smoke 的四步总、DN、encoder loss 与 grad norm
  均有限；6 组 attention 误差为零、18 组独立参数最大差异 `0.000788`，
  三层 midpoint 细节门最大权重 `0.000394/0.000391/0.000394`。07:18 fresh
  启动正式训练，07:20 达到 epoch 1 iter 100：`0.8707 s/iter`、loss
  `20.5465`、grad norm `101.3330`，GPU 4/5 正常，无训练异常。

## 2026-07-31 07:25 CST 99/178 epoch-4 双 HOTA 通过

- 99 `0731_09 regression-only` 的 epoch 4 cls HOTA/DetA/AssA 为
  `37.813/27.802/55.047`，det 为 `44.030/33.349/59.176`；相对父配置
  HOTA 提高 `1.604/5.277`、DetA 提高 `0.734/0.895`。pair mAP
  `0.170013`、both-independent AP50 `0.353940`，完整检测、TrackEval、原始
  CSV 和结构检查均通过，保留到 epoch 8。
- 178 `0731_11 midpoint-regression` 的 epoch 4 cls HOTA/DetA/AssA 为
  `38.668/30.574/53.235`，det 为 `43.586/38.232/51.254`；相对父配置
  HOTA 提高 `2.459/4.833`，cls DetA/AssA 提高 `3.506/1.141`，
  det DetA/AssA 提高 `5.778/3.788`。pair mAP `0.184185`、
  both-independent AP50 `0.391559`，结构审计通过，保留到 epoch 8。
- 99 回归分支与 178 midpoint 分支均在 HOTA、DetA、AssA 和 AP 上同步提高；
  当前不做参数扫描，继续用 epoch 8 判断早期覆盖增益能否保持。

## 2026-07-31 07:45 CST 0731_05 epoch-12 决策

- 252 `0731_05 shared-attention + full-path enveloped-detail` 的 epoch 12
  checkpoint、检测 metrics、完整 TrackEval、原始 CSV 与结构审计均已完成。
  cls HOTA/DetA/AssA 为 `50.171/42.372/61.238`，det 为
  `56.430/49.540/66.445`。
- 相对 `0727_01` 同点 `49.680/56.541`，cls HOTA 提高 `0.491`，det HOTA
  低 `0.111`，因此该点不构成严格双 HOTA 通过。分解上 cls DetA/AssA 为
  `+1.012/-0.572`，det DetA/AssA 为 `-0.805/+0.866`。
- det DetA 差距已由 epoch 8 的 `-1.885` 收窄至 `-0.805`，且 pair mAP
  `0.276140`、both-independent AP50 `0.534695` 分别高于父配置同点
  `0.273170/0.511752`。AP 只作覆盖诊断，不用于改写 HOTA 结论。
- 鉴于 det HOTA 仅低 `0.111`、检测覆盖差距正在收窄，且这是当前最成熟候选，
  保留到 epoch 16 作最后一次中期确认；若仍不能双超越父 encoder，则完成全量
  评估后停止并释放 252 给 terminal-only 结构。

## 2026-07-31 07:58 CST 0731_12 terminal-only 静态就绪

- 已为 252 预留 `0731_12 shared-attention + terminal-only enveloped-detail`
  的正式 2xb4 配置、4-iter 真数据 smoke 配置、正式 launcher 与 smoke launcher。
  它只在最终 decoder 输出前注入帧细节；前两层、辅助输出与 iterative references
  均保持 shared-attention 父路径。
- 两份配置均通过 `copy.deepcopy`；正式配置完整构建为
  `MultispecPairRotatedRTDETR`，且仅启用 `shared_attention_decoder=True` 与
  `terminal_enveloped_detail_decoder=True`。调用 detector `init_weights()` 后唯一
  terminal gate 的参数最大绝对值为零。
- 两份 launcher 均通过 `bash -n`，目标正式与 smoke workdir 在 252 上均确认不存在。
  当前状态仅为 `PREPARED`：未运行 smoke、未创建目录、未占 GPU，也未进入队列。
  只有 `0731_05` epoch 16 或其他完整 HOTA 证据触发接替时，才执行真实 DDP smoke。

## 2026-07-31 09:20 CST epoch-16 决策与 terminal 结构接替

- 252 `0731_05 full-path enveloped-detail` 的 epoch 16 完整结果为 cls
  HOTA/DetA/AssA `51.007/42.964/62.130`，det
  `57.940/50.762/68.403`。相对 encoder 同点 HOTA 为
  `-0.084/-0.380`；虽然 pair mAP `0.2853` 与 both-independent AP50
  `0.5439` 高于父配置，但一级双 HOTA 门槛失败。09:07 精确终止目标进程组，
  epoch 16 及此前全部 checkpoint/评测产物保留，GPU 0/1 确认释放。
- 99 `0731_09 regression-only` 的 epoch 8 cls/det HOTA 为
  `44.043/49.271`，相对 encoder 同点 `-1.226/-0.922`，完成结构审计后停止。
  99 已于 08:46 接替运行 `0731_12 terminal-only`；正式 iter 50 总、DN、
  encoder loss 与 grad norm 均有限，09:20 已进入 epoch 2。
- 197 `0731_10 midpoint-regression` 的 epoch 4 cls/det HOTA 为
  `38.794/44.142`，相对父配置 `+2.585/+5.389`，双 DetA/AssA 也全部提高，
  因而保留到 epoch 8；09:06 到 epoch 7 iter 650。
- 178 `0731_11 midpoint-regression` 的 epoch 8 cls/det HOTA 为
  `45.173/51.263`，相对父配置 `-0.096/+1.070`。cls 只窄幅落后、det 增益明确，
  尚不足以淘汰；保留到 epoch 12，09:05 到 epoch 11 iter 350。
- 新增 `0731_13 terminal-midpoint enveloped-detail`：分类只在最终层接收
  bounded swap-odd 帧细节；回归也只在最终层接收帧细节，并把新增 5D box-logit
  residual 显式构造成严格 `-detail/+detail`，使 pair midpoint 不发生漂移。
  前两层、辅助输出和递归 reference 与 shared-attention 父路径逐元素一致。
- `0731_13` 的针对性单测、配置深拷贝、完整模型构建、launcher 语法、252 真数据
  双卡 4-iter smoke 与 checkpoint 结构验收均通过。smoke 中 6 组 attention
  最大误差为零，18 组独立参数最大差异 `0.000794`，唯一 terminal gate 最大权重
  `0.000393`。09:18 fresh 启动正式训练，09:19 达到 iter 50：
  loss `22.1957`、grad norm `115.9534`，GPU 0/1 各约 `19.2 GiB`，无
  Traceback/OOM/NaN/NCCL。代码提交为 `d9c97e0`。

## 2026-07-31 09:50 CST midpoint 决策与 HOTA 优先恢复

- 197 `0731_10 midpoint-regression` epoch 8 的 cls/det HOTA 为
  `45.254/50.846`，相对 encoder 同点 `-0.015/+0.653`；178 同结构
  `0731_11` epoch 12 为 `49.478/56.451`，相对同点 `-0.202/-0.090`。
  两项 checkpoint、检测、完整 TrackEval 与原始 CSV 均已落盘，09:47 精确停止，
  GPU 释放后未残留目标训练进程。
- 按用户最新的 cls/det HOTA 主指标规则，重新恢复此前因 DetA/mAP 次要保护线
  提前停止、但 HOTA 当时双过的两个结构方向。197 从 `0730_09 motion-trust`
  epoch 8 恢复；该点相对父配置 HOTA `+0.229/+0.967`，09:49 已到 epoch 9
  iter 50，继续到 epoch 12。178 从 `0730_16 antisymmetric-detail` epoch 4
  恢复；该点相对父配置 HOTA `+0.475/+0.468`，09:50 已到 epoch 5 iter 50，
  继续到 epoch 8。
- 两个恢复均加载原 optimizer、EMA、epoch 与 iter，不是 fresh 重跑。178 当前
  PyTorch 2.6 对旧可信 MMEngine checkpoint 的 weights-only 默认值不兼容，只在
  恢复脚本中设置 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`，未修改模型、loss 或配置。
  两项正式 iter 50 的总/DN/encoder loss 与 grad norm 均有限，无
  Traceback/OOM/NaN；197 仍严格只使用 GPU 4/5。

## 2026-07-31 10:04 CST 99 terminal-only epoch-4 结果

- `0731_12` epoch 4 cls HOTA/DetA/AssA 为 `37.799/28.644/53.288`，
  det 为 `43.483/34.398/56.345`；相对 encoder 同点 HOTA
  `+1.590/+4.730`，且 cls/det 的 DetA、AssA 四项全部提高。
- pair mAP `0.165773`、both-independent AP50 `0.355169`，未显示检测覆盖崩塌。
  结构检查确认 6 组 shared attention 严格相同、18 组独立参数已分化、唯一
  terminal gate 有限非零。checkpoint、检测、TrackEval 与原始 CSV 均完整，
  因而继续到 epoch 8；10:03 已进入 epoch 5。

## 2026-07-31 10:49 CST 252 terminal-midpoint epoch-4 结果

- `0731_13` epoch 4 cls HOTA/DetA/AssA 为 `36.148/27.020/51.933`，
  det 为 `43.247/33.958/56.389`；相对 encoder 同点分别为
  `-0.061/-0.048/-0.161` 与 `+4.494/+1.504/+8.923`。因此当前不记为
  双 HOTA 提升，但 cls 差距极窄，det 的检测与关联组成量均明显提高。
- pair mAP `0.153978`，相对父配置约 `-0.003275`；both-independent AP50
  `0.335487`，相对约 `+0.012338`。按当前规则 AP 仅作诊断，未显示与 det HOTA
  同向的检测崩塌。
- checkpoint、检测 metrics、完整 TrackEval 和原始 CSV 均已落盘。结构检查确认
  6 组 shared attention 最大误差为零、18 组独立参数最大差异 `0.037019`，
  唯一 terminal-midpoint gate 最大权重 `0.081010`，排除模块未学习。
- 同 epoch 它低于 99 `0731_12 terminal-only` `1.651/0.236`，说明新增 midpoint
  约束尚未形成早期正交增益；但其 cls 仅低父配置 `0.061`，且该结构的核心假设是避免
  中期递归污染，因此允许继续到 epoch 8 做一次稳定性判定。epoch 8 若仍非双 HOTA
  通过，则停止，不再延长。

## 2026-07-31 11:12 CST 178/197 恢复实验收口

- 197 `0730_09 motion-trust` 恢复后的 epoch 12 完整结果为：cls
  HOTA/DetA/AssA `48.877/40.354/61.309`，det
  `55.819/48.907/65.923`。相对 encoder 同点 HOTA `-0.803/-0.722`，
  DetA `-1.006/-1.438`，仅 det AssA `+0.344`。pair mAP/AP50
  `0.263196/0.479269`，both-independent mAP/AP50
  `0.300922/0.513250`。三层 motion-trust 权重最大值
  `0.453352/0.294524/0.256910`，结构已充分学习但双 HOTA 失败，已停止并释放
  GPU 4/5。
- 178 `0730_16 antisymmetric-detail` 恢复后的 epoch 8 完整结果为：cls
  HOTA/DetA/AssA `44.591/37.157/56.398`，det
  `49.685/46.466/54.883`。相对 encoder 同点 HOTA `-0.678/-0.508`，
  DetA `-0.506/-0.595`，AssA `-0.903/-0.262`。pair mAP/AP50
  `0.241645/0.435966`，both-independent mAP/AP50
  `0.279185/0.474553`。三层结构权重最大值
  `0.035530/0.036084/0.034844`，同样是方向失败而非结构未学习，已停止并释放
  GPU 0。
- 两项恢复评估复用了旧 `val_track_0001` 目录，已分别复制保留为
  `val_track_epoch12_resume_20260731` 与
  `val_track_epoch08_resume_20260731`。共享存储元数据时钟与登录节点约有 25 分钟
  偏移，结果通过 checkpoint、检测 metrics、TrackEval metrics 与原始 CSV 的
  写入顺序和内容交叉确认。99 `0731_12` 的 epoch 8 训练已完成并等待完整评估；
  252 `0731_13` 在 epoch 6 稳定运行。178/197 暂时保留，待 99 同点结果后部署
  下一项有因果依据的结构实验。

## 2026-07-31 11:33 CST 99 epoch-8 决策与新结构准备

- 99 `0731_12 terminal-only` epoch 8 的完整结果为 cls
  HOTA/DetA/AssA `43.178/35.246/55.355`，det
  `50.010/43.700/59.276`。相对 encoder 同点 HOTA `-2.091/-0.183`，
  DetA `-2.417/-3.361`，cls AssA `-1.946`，仅 det AssA `+4.131`。
  pair mAP/AP50 `0.217536/0.410164`，both-independent AP50 `0.448156`，
  也未显示检测覆盖保护。完整 checkpoint、检测、TrackEval、原始 CSV 与结构审计
  后停止，GPU 0/1 释放。
- checkpoint 中 6 组 shared attention 误差为零、18 组独立参数最大差异
  `0.063149`、唯一 terminal gate 最大权重 `0.117809`，排除模块未学习。
  结论是：去除递归污染仍不够，最终分类分数专门化本身会改变匹配与筛选并损伤
  DetA；不能继续用缩小 residual scale 掩盖该结构问题。
- 新增两项因果对照。`0731_14` 只在最终框输出使用 bounded frame detail，分类
  hidden state 在所有层严格等于 shared parent；`0731_15` 进一步把最终新增框
  residual 构造成严格反对称，使 pair midpoint 不变。两项均不改 encoder、
  proposal、PairDN、head、loss 或训练协议。
- 89 项 decoder 单测、两份正式和两份 smoke 配置深拷贝、两个完整模型构建及四个
  launcher 语法检查通过。下一步在 99/197 各执行真实双卡 4-iter smoke；只有
  checkpoint 结构、有限 loss 与正式 iter 50 五项门槛全部通过才记为 `RUNNING`。

## 2026-07-31 11:45 CST terminal regression 两项正式启动

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 197 GPU 4,5 | `0731_15_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_terminalmidpointregressionenvelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 11:43 /  | 4-iter 真实数据 DDP smoke 的总、DN、encoder loss 与 grad norm 全部有限；checkpoint 中 6 组 attention 严格共享、18 组独立参数最大差异 `7.79018e-4`，terminal midpoint gate 最大值 `3.92011e-4`。正式 fresh 训练于 11:45 到 epoch 1 iter 50，`0.8932 s/iter`、loss `22.1811`、grad norm `124.8561`，GPU 4/5 各约 `19.2 GiB`，无 Traceback/OOM/NaN/NCCL/DDP 异常。 |
| RUNNING | 99 GPU 0,1 | `0731_14_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_terminalregressionenvelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 11:43 /  | 4-iter 真实数据 DDP smoke 的总、DN、encoder loss 与 grad norm 全部有限；checkpoint 中 6 组 attention 严格共享、18 组独立参数最大差异 `7.85518e-4`，terminal gate 最大值 `3.92929e-4`。正式 fresh 训练于 11:45 到 epoch 1 iter 50，`0.9528 s/iter`、loss `22.1461`、grad norm `126.9356`，GPU 0/1 各约 `19.2 GiB`，无 Traceback/OOM/NaN/NCCL/DDP 异常；GPU 2 上外部任务未触碰。 |

- 两项实验均基于提交 `1575d96`，99、197、252、178 已验证为同一 commit；
  252 的既有 `0731_13` 进程保持运行且未重启。首个统一科学决策点为 epoch 4，
  主判据仍是相对 `0727_01` 同 epoch 的 cls/det HOTA，DetA/AssA 用于归因，
  mAP 仅作检测崩塌诊断。

## 2026-07-31 12:59 CST 0731_14/15 epoch-4 双提升

- 两项实验的 epoch 4 checkpoint、检测 metrics、完整 TrackEval、54 个原始
  TrackEval 输出及结构检查均已形成。固定父配置 `0727_01` 同点为：cls
  HOTA/DetA/AssA `36.209/27.068/52.094`，det
  `38.753/32.454/47.466`；pair mAP/AP50 `0.157253/0.296134`，
  both-independent mAP/AP50 `0.184465/0.323149`。
- 99 `0731_14 terminal regression-only` 的 cls 为
  `37.209/27.864/53.684`，det 为 `43.646/34.679/56.188`。相对父配置
  cls/det HOTA `+1.000/+4.893`，DetA `+0.796/+2.225`，AssA
  `+1.590/+8.722`。pair mAP/AP50 `0.163485/0.323237`，
  both-independent mAP/AP50 `0.192363/0.351268`，均无检测退化。
  checkpoint 中 6 组 shared attention 误差为零、18 组独立参数最大差异
  `0.031446`、terminal gate 最大权重 `0.070805`。
- 197 `0731_15 terminal midpoint-regression` 的 cls 为
  `38.153/27.923/55.246`，det 为 `44.506/34.409/59.052`。相对父配置
  cls/det HOTA `+1.944/+5.753`，DetA `+0.855/+1.955`，AssA
  `+3.152/+11.586`。pair mAP/AP50 `0.161120/0.318966`，
  both-independent mAP/AP50 `0.189558/0.346678`，也均提高。
  checkpoint 中 shared attention 误差为零、独立参数最大差异 `0.029741`、
  terminal midpoint gate 最大权重 `0.068100`。
- 两者均通过 HOTA 主门槛并继续到 epoch 8。197 的早期 HOTA 更高，但更依赖
  AssA；99 的 det DetA 更高、AssA 搬运较少，因此 epoch 8 重点检查哪条路径能
  避免历史上的 DetA→AssA 中期退化。

## 2026-07-31 14:38 CST epoch-8 淘汰与严格正交结构接替

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| STOPPED | 252 GPU 0,1 | `0731_03_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_commonevidencebypass_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 01:44 / 2026-07-31 14:08 前 | epoch 8 cls/det HOTA `44.798/50.415`，相对 encoder 同点 `-0.471/+0.222`；cls DetA/AssA `-1.030/-0.209`，det `-2.602/+3.680`。完整产物和结构审计齐全后停止。 |
| STOPPED | 99 GPU 0,1 | `0731_14_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_terminalregressionenvelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 11:43 / 2026-07-31 14:08 前 | epoch 8 cls/det HOTA `44.321/49.059`，相对父配置 `-0.948/-1.134`；DetA 分别下降 `1.901/3.480`。模块已学习，停止并保留 epoch 4/8 全部产物。 |
| STOPPED | 197 GPU 4,5 | `0731_15_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_terminalmidpointregressionenvelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 11:43 / 2026-07-31 14:08 前 | epoch 8 cls/det HOTA `43.918/49.071`，相对父配置 `-1.351/-1.122`；DetA 分别下降 `2.925/4.146`。与`0731_14`形成独立重复，排除偶然波动。 |
| STOPPED_INVALID | 252 GPU 0,1 | `0731_17_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_terminalfactorizedevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 14:05 / 2026-07-31 14:14 前 | 正式 epoch 1 iter 300 前后审计发现 boxes 仍以`common_output`为基底，不满足严格正交设计；首个正式 epoch 前停止，不进入实验结果。 |
| RUNNING | 252 GPU 0,1 | `0731_18_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_terminalorthogonalfactorizedevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 14:15 /  | 分类只接收共同证据，boxes 只接收严格反对称 detail；共同证据不改变任一 box reference，detail midpoint 为零。98 项单测、完整构建、真实 2 卡 smoke 和正式五项门槛通过；14:38 到 epoch 2 iter 50。 |
| RUNNING | 99 GPU 0,1 | `0731_19_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_terminalclassificationcommonevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 14:30 /  | 不共享 decoder attention，只有末层 classification 接收共同证据；boxes、aux 与 recurrent references 严格保持父模型。100 项单测、完整构建、真实 2 卡 smoke 和正式五项门槛通过；14:38 到 epoch 1 iter 400。 |
| RUNNING | 197 GPU 4,5 | `0731_20_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_terminalclassificationcommonevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 14:33 /  | 与`0731_19`唯一核心结构差异是共享 decoder attention；同样只改变末层 classification。100 项单测、完整构建、真实 2 卡 smoke 和正式五项门槛通过；14:38 到 epoch 1 iter 250。 |

- 178 `0731_16` epoch 4 为 cls/det HOTA `37.750/43.723`，相对父配置
  `+1.541/+4.970`，但 DetA `-0.565/-0.841`、AssA `+5.091/+14.900`，
  显示明显 DetA→AssA 搬运；14:38 已进入 epoch 8，等待完整结果后再决定是否停止。
- 当前四台服务器 tracked HEAD 均为干净的 `ac9d629`。178 运行时由
  `d78500d` 启动，252 运行时由 `ad99b0d` 启动；仓库后续快进不改变已加载进程。

## 2026-07-31 15:03 CST 0731_16 epoch-8 淘汰

- 178 `0731_16 terminal common-evidence bypass` 的 epoch 8 checkpoint、检测
  metrics、完整 TrackEval、54 个原始输出文件和结构审计均已形成。cls
  HOTA/DetA/AssA 为 `43.972/32.412/62.419`，det 为
  `49.378/39.704/63.985`。
- 相对 `0727_01` 同点，cls HOTA/DetA/AssA 为
  `-1.297/-5.251/+5.118`，det 为 `-0.815/-7.357/+8.840`。epoch 4 的
  早期双 HOTA 增益完全没有保持，且两路都出现强烈的 DetA→AssA 搬运。
- pair mAP/AP50 为 `0.205329/0.366688`，相对父配置下降
  `0.032405/0.064222`；both-independent mAP/AP50 为
  `0.236540/0.395712`，下降 `0.039642/0.070239`。检测诊断与 DetA 结论一致。
- checkpoint 中唯一 terminal-common gate 最大权重为 `0.032061`，epoch/iter
  为 `8/8304`，排除结构未学习。15:03 精确关闭唯一 screen，训练进程退出，
  GPU0 为 `0%/1 MiB`；epoch 4/8 全部产物保留，GPU1 未触碰。

## 2026-07-31 15:14 CST 0731_21 正式启动

- 当前三项严格隔离实验构成了 `shared attention × antisymmetric box detail` 的
  2×2 结构设计，但缺少“独立 attention + 分类共同证据 + 反对称 box detail”。
  `0731_21` 补齐该单元：分类只接收末层共同证据，boxes 只接收严格反对称
  5D detail residual；auxiliary output 与 recurrent references 保持父路径，
  两帧 attention-weight 投影保持独立。
- 原实现人为要求 factorized evidence 必须开启 shared attention。审计确认其
  common/antisymmetric 代数、box midpoint 守恒和父路径隔离并不依赖参数共享后，
  提交 `a77e135` 移除该不必要绑定并增加独立 attention 专项测试。完整 100 项
  decoder 单测、正式/smoke 配置深拷贝、完整模型构建、launcher 语法及目标环境
  数据/GMC/预训练权重检查全部通过。
- 178 GPU0 的真实 4-iter smoke 最终 loss/grad norm 为
  `19.8328/67.2125`，总、DN、encoder loss 有限。checkpoint 中 6 组独立
  attention-weight 最大差异 `7.67466e-4`，common/detail terminal gate 最大值
  分别为 `3.97146e-4/3.92367e-4`；结构已更新且 midpoint 单测通过。
- 15:13 fresh 正式启动，15:14 到 epoch 1 iter 50：学习率
  `2.5488e-6`、time `0.9401 s/iter`、loss `20.9349`、grad norm
  `90.7400`；GPU0 约 31.4 GiB，总、DN、encoder loss 有限，无
  Traceback/OOM/NaN/NCCL。GPU1 未触碰，首判 epoch 4。

## 2026-07-31 15:50 CST 0731_18/19 epoch-4 决策

- 252 `0731_18 shared-attention + terminal orthogonal factorized evidence`
  的 epoch 4 checkpoint、检测 metrics、完整 TrackEval、54 个原始结果文件和
  checkpoint 结构审计均已完成。cls HOTA/DetA/AssA 为
  `37.315/26.954/55.487`，det 为 `41.743/31.826/56.276`；相对
  `0727_01` 同点，cls/det HOTA 提高 `1.106/2.990`，但 DetA 分别变化
  `-0.114/-0.628`、AssA 提高 `3.393/8.810`。pair mAP/AP50 为
  `0.158907/0.304586`，both-independent mAP/AP50 为
  `0.185215/0.329253`，四项检测诊断均高于父配置。
- 该 checkpoint 中 6 组 shared attention 最大误差为零、18 组独立参数最大差异
  `0.033962`；terminal common/detail gate 最大权重分别为
  `0.031387/0.090002`。结构确已学习，且严格正交检查通过。它按双 HOTA 主门槛
  继续到 epoch 8；中期重点检查当前轻度 DetA→AssA 搬运是否扩大。
- 99 `0731_19 independent-attention + terminal classification common
  evidence` 的 epoch 4 同样完成全部产物与结构审计。cls
  HOTA/DetA/AssA 为 `36.810/27.099/53.921`，det 为
  `43.194/33.533/56.890`；相对父配置双 HOTA 提高 `0.601/4.441`，
  cls/det DetA 也提高 `0.031/1.079`。pair mAP/AP50 为
  `0.156129/0.309147`，both-independent mAP/AP50 为
  `0.183573/0.334282`；mAP 的小幅波动不改变 HOTA 主判断，AP50 与 DetA
  未显示检测覆盖崩塌。
- `0731_19` 唯一 terminal common gate 最大权重为 `0.034268`，排除模块未学习。
  该实验继续到 epoch 8。现阶段它比 `0731_18` 更少依赖 AssA 搬运，是两个已完成
  epoch-4 单元中更干净的候选；最终成功标准仍为同一 checkpoint 的 cls/det HOTA
  同时超过 `54.437/62.393`。
- 197 `0731_20` 于 15:50 到 epoch 3 iter 1000，178 `0731_21` 到
  epoch 3 iter 450；两项均继续等待各自 epoch 4 完整评估。四项训练总、DN、
  encoder loss 有限，无 Traceback/OOM/NaN/NCCL。

## 2026-07-31 16:38 CST 0731_20/21 epoch-4 决策

- 197 `0731_20 shared attention + classification common only` 的 epoch 4
  checkpoint、检测 metrics、完整 TrackEval、54 个原始结果文件和结构审计均已
  完成。cls HOTA/DetA/AssA 为 `37.345/28.037/52.923`，det 为
  `43.627/34.822/56.234`；相对 `0727_01` 同点，cls/det HOTA 提高
  `1.136/4.874`，DetA 提高 `0.969/2.368`，AssA 提高
  `0.829/8.768`。pair mAP/AP50 为 `0.156468/0.318016`，
  both-independent mAP/AP50 为 `0.185581/0.346145`。
- `0731_20` 的唯一 terminal common gate 最大权重为 `0.033425`；6 组
  shared attention 最大误差为零，18 组其余逐帧参数最大差异 `0.027972`。
  结构已学习且满足预期，继续到 epoch 8。
- 178 `0731_21 independent attention + classification common +
  antisymmetric box detail` 的 epoch 4 cls HOTA/DetA/AssA 为
  `38.031/29.960/52.258`，det 为 `41.072/36.731/47.129`；相对父配置
  双 HOTA 提高 `1.822/2.319`，DetA 提高 `2.892/4.277`。pair mAP/AP50
  `0.174049/0.350164`、both-independent mAP/AP50
  `0.205622/0.379277`，四项检测诊断分别提高
  `0.016797/0.054029/0.021157/0.056128`，是当前四个单元中检测覆盖改善最强、
  最少依赖 AssA 搬运的候选。
- `0731_21` 的 6 组独立 attention-weight 最大差异为 `0.036236`，
  common/detail gate 最大权重为 `0.033932/0.054266`；严格正交结构审计通过。
  它继续到 epoch 8。
- epoch-4 的 `shared attention × antisymmetric box detail` 2×2 对照显示明显
  交互：不加 box detail 时，共享 attention 使 `0731_20` 相对 `0731_19`
  cls/det HOTA 提高 `0.535/0.433`，DetA 提高 `0.938/1.289`；而加入 detail
  后，共享 attention 使 `0731_18` 相对 `0731_21` 的 cls DetA 下降 `3.006`、
  det DetA 下降 `4.905`。独立 attention 下加入 detail 可显著提高 DetA，
  但 det AssA 降低 `9.761`，因而 det HOTA 相对 classification-only 单元下降
  `2.122`。所以不能把 shared attention 与 antisymmetric detail 当作可直接相加
  的独立增益；四项均保留到 epoch 8 检验中期稳定性。

## 2026-07-31 17:39 CST epoch-8 分流与梯度隔离实验启动

- 99 `0731_19 independent attention + classification common only` 的 epoch 8
  cls/det HOTA 为 `43.480/50.152`，相对 `0727_01` 同点
  `-1.789/-0.041`；cls/det DetA 分别下降 `2.883/3.568`。252
  `0731_18 shared attention + factorized evidence` 为 `43.681/51.660`，
  相对父配置 `-1.588/+1.467`，cls/det DetA 下降 `3.302/2.752`。
  两项均在 checkpoint、检测、54 个 TrackEval 原始文件和结构审计齐全后停止，
  保留全部 epoch 4/8 产物。
- 178 `0731_21 independent attention + factorized evidence` 的 epoch 8
  cls HOTA/DetA/AssA 为 `46.642/38.691/59.237`，det 为
  `52.107/46.230/60.830`；相对 `0727_01` 同点双 HOTA
  `+1.373/+1.914`。pair mAP/AP50 提高 `0.010728/0.035152`，
  both-independent 提高 `0.010185/0.033446`。54 个 TrackEval 原始文件完整；
  独立 attention 最大差异 `0.056624`，common/detail gate
  `0.036273/0.072394`。该候选通过 epoch-8 门槛并继续训练。
- 代码审计确认旧 terminal common/detail 模式虽然对 residual/detail 本体停止梯度，
  gate 的归一化两帧 evidence 仍向共享 decoder 表征反向传播。提交 `2c8b13b`
  新增结构开关，只切断这条附加梯度路径而保持 gate 可学习；103 项 decoder 单测、
  4 份配置深拷贝、2 个完整模型构建和双机真实 4-iter DDP smoke 均通过。
- 99 `0731_22 detached classification-common` 与 252
  `0731_23 detached orthogonal factorized` 于 17:37 fresh 启动。17:39 两者均到
  epoch 1 iter 50，GPU/正式日志/有限总损失、DN loss、encoder loss 与 grad norm
  五项门槛通过；smoke gate 分别更新到 `3.935e-4`，以及
  `3.925e-4/3.949e-4`。两项首判均为 epoch 4。

## 2026-07-31 18:33 CST 0731_20 收口与 0731_22/23 等价性纠正

- 197 `0731_20` epoch 8 完整结果为：cls HOTA/DetA/AssA
  `43.670/34.495/58.446`，det `49.863/43.705/58.916`。相对
  `0727_01` 同点 HOTA `-1.599/-0.330`、DetA `-3.168/-3.356`；
  AP 诊断也全面下降。54 个 TrackEval 原始文件、检测 metrics、checkpoint 和结构
  审计齐全；common gate `0.052896`，shared attention 误差为零。18:31 精确停止，
  GPU 4,5 已释放。
- 复核提交前后的真实代码确认，归一化 common/detail gate evidence 原本就由
  `_normalized_shared_evidence()` 以 `.detach()` 返回。`2c8b13b` 新增的
  `terminal_detach_gate_evidence` 对该张量再次 detach，不改变任何前向或反向语义。
  因而 17:39 记录的“旧模式仍存在 gate evidence 梯度回传”结论无效，现撤销。
- 99 `0731_22` 与 252 `0731_23` 分别与 `0731_19/21` 数学等价，不构成有效消融，
  已于首个 checkpoint 前在 18:32 精确停止。两台 GPU 均释放，日志保留但不进入
  正式科学结果表。当前唯一继续训练的是 178 `0731_21`；其 epoch 8 双 HOTA 增益
  仍是有效结果，但机制应归因于独立双帧 attention 与末层 common/detail 正交分解，
  不能归因于新增的梯度隔离开关。

## 2026-07-31 18:55 CST object-reliable terminal factorization 三路启动

- 提交 `737210b` 在 `0731_21` 的末层正交 common/detail 结构上增加
  `terminal_factorized_confidence`。可靠度取两条父分类路径最大 object confidence 的
  detached 几何均值，只控制 common 修正、detail 修正或两者；不引入类别权重、loss
  重加权、阈值、可学习 scale，也不修改 encoder、proposal、PairDN 或训练协议。
- 102 项 decoder 单测、3 份正式配置深拷贝与完整构建、3 份 smoke 配置深拷贝、
  6 份 launcher 语法检查均通过。99/197/252 的目标双卡真实数据 4-iter smoke 均产生
  有限总损失、DN loss、encoder loss 和 grad norm，且独立 attention 与 terminal gate
  均获得非零更新。
- 99 `0731_24 object-confident common`、197 `0731_25 object-confident detail`、
  252 `0731_26 object-confident common+detail` 均于 18:49 fresh 启动。18:52 分别达到
  epoch 1 iter `150/100/150`，screen、worker、目标 GPU、正式日志、有限损失与无致命
  错误五项门槛全部通过。197 进程实际 cwd 为干净的
  `/data/users/litianhao/PairMOT_sync_3cb888d/ai4rs`、HEAD `737210b`；旧
  `/data/users/litianhao/PairMOT` 不是本次训练目录。
- 178 `0731_21` 的 epoch 12 完整结果为 cls HOTA/DetA/AssA
  `49.159/40.613/61.861`，det `56.341/49.482/66.346`；相对 `0727_01` 同点
  HOTA `-0.521/-0.200`、DetA `-0.747/-0.863`、AssA `+0.051/+0.767`。
  这表明 epoch 8 的双增益已出现轻度 DetA→AssA 回落，但差距尚小且 AP50 仍提高，
  因此保留到 epoch 16 做持续性判定；三条 confidence 实验的首判点均为 epoch 4。

## 2026-07-31 19:52 CST 0731_21 epoch-16 与早期评估规则修订

- epoch 16 的完整 checkpoint、检测 metrics、TrackEval 和 54 个原始文件均已落盘。
  cls HOTA/DetA/AssA 为 `50.273/41.786/62.674`，det 为
  `58.085/51.049/68.372`。相对 `0727_01` 同点，双 HOTA
  `-0.818/-0.235`、双 DetA `-0.818/-0.665`，cls AssA `-0.554`，det AssA
  `+0.402`。pair/both mAP 分别低 `0.000938/0.001007`，AP50 分别高
  `0.000355/0.001311`，属于小幅但连续的 HOTA/DetA 回落。
- 用户指出前期 eval 可能过严。规则据此修订：epoch 4 只作结构学习和灾难性退化检查；
  epoch 8/12 用于观察 DetA/AssA 收敛顺序；单点小幅双降不淘汰。只有连续节点出现
  系统性退化，或 HOTA、DetA、AP 同时明显恶化时才早停。有竞争力的简单结构至少观察到
  epoch 16/20。`0731_21` 当前已出现两个连续小幅双降点，但没有 AP 全面恶化，且用户要求
  放宽早期判定，因此不在 epoch 16 停止，继续到 epoch 20。
- 模型复杂度和效率作为硬约束：后继不得堆叠 decoder 层、额外 attention、高分辨率分支
  或额外 loss；须保持可解释 common/detail 语义，并在同卡同温条件下实测速度。当前
  `0731_24/25/26` 为无参数 query 级可靠度路由，不属于复杂模型堆叠。

## 2026-07-31 20:35 CST 三路 object-confidence epoch-4 结果

- 99 `0731_24 confident-common` 的 cls HOTA/DetA/AssA 为
  `36.689/26.771/53.796`，det 为 `42.562/32.498/56.889`；相对
  `0727_01` 同点 HOTA `+0.480/+3.809`，但 DetA 仅 `-0.297/+0.044`，
  主要是 AssA 提升。pair/both mAP 略降，AP50 分别提高 `0.011296/0.008962`。
- 252 `0731_26 confident-common+detail` 的 cls HOTA/DetA/AssA 为
  `36.958/26.877/54.801`，det 为 `41.792/32.810/54.722`；相对父配置
  HOTA `+0.749/+3.039`、DetA `-0.191/+0.356`。其双 HOTA 通过 epoch-4
  结构学习检查，但同样主要依赖 AssA，pair/both mAP 略降而 AP50 提高。
- 197 `0731_25 confident-detail` 的 cls HOTA/DetA/AssA 为
  `35.824/25.781/53.847`，det 为 `41.591/30.920/57.109`；相对父配置
  HOTA `-0.385/+2.838`、DetA `-1.287/-1.534`，pair/both mAP 也分别下降
  `0.007218/0.008269`，是三路中最弱的一项。
- epoch 4 只作结构是否学习和灾难性退化检查，不能据此否定简单结构。因此三路均继续：
  `0731_24/26` 至少观察 epoch 8/12，`0731_25` 先观察 epoch 8；只有连续节点出现
  HOTA、DetA 与 AP 的系统性恶化才停止。三种路由均不增加可学习参数、decoder 深度、
  attention 或 loss；若形成候选主线，必须补做同卡同温训练与推理速度比较。

## 2026-07-31 21:02 CST 0731_21 epoch-20 恢复至近同点

- epoch 20 checkpoint、`val_det/epoch_19`、第五次完整 TrackEval 与 54 个原始结果文件
  均已落盘。cls HOTA/DetA/AssA 为 `51.475/43.145/63.273`，det 为
  `58.956/51.645/69.703`。
- 相对 `0727_01` 同点，cls/det HOTA 为 `-0.039/+0.034`，cls/det DetA 为
  `+0.154/-0.554`，AssA 为 `-0.476/+0.974`。严格双 HOTA 门槛仍未通过，但已经从
  epoch 16 的 `-0.818/-0.235` 明显恢复到近似持平。
- pair mAP/AP50 分别提高 `0.003474/0.007421`，both-independent mAP/AP50 分别
  提高 `0.004285/0.008951`。因此该恢复不是整体检测 AP 崩塌造成的虚假关联增益。
- 该结构简单、无额外 decoder 深度或 attention，并且 e20 出现恢复趋势；训练已自然进入
  epoch 21，故不在 e20 停止，继续观察 epoch 24。若 e24 双 HOTA 转为同点正增益，继续
  验证持续性；若再次出现明确双降，再结合 e8-e24 全轨迹决定是否释放 178。

## 2026-07-31 21:47 CST confidence 路由 epoch-8 与复杂度审计

- 99 `0731_24 confident-common` e8 的 cls HOTA/DetA/AssA 为
  `44.103/35.015/59.108`，det 为 `50.795/44.820/59.804`；相对
  `0727_01` 同点 HOTA `-1.166/+0.602`、DetA `-2.648/-2.241`、AssA
  `+1.807/+4.659`。pair mAP/AP50 下降 `0.017332/0.017646`，both mAP/AP50
  下降 `0.022292/0.021760`。
- 252 `0731_26 confident-common+detail` e8 的 cls HOTA/DetA/AssA 为
  `44.283/35.635/57.949`，det 为 `50.390/43.986/59.457`；相对父配置
  HOTA `-0.986/+0.197`、DetA `-2.028/-3.075`、AssA `+0.648/+4.312`。
  pair mAP/AP50 下降 `0.009266/0.001364`，both mAP/AP50 下降
  `0.012395/0.004377`。
- 与不加 confidence 的 `0731_21` e8 `46.642/52.107` 相比，`0731_24` 双 HOTA
  低 `2.539/1.312`，`0731_26` 低 `2.359/1.717`。两项独立结果共同说明双边分类
  confidence 会过度衰减末层检测修正，把 DetA/AP 搬运为 AssA，而不是改善 factorization。
- 按放宽后的判定规则，两项都继续到 e12 做最后一次持续性确认；但不再创建新的 confidence
  组合。若 e12 未发生实质恢复，则停止并回到无 confidence 的 `0731_21` 结构轨迹。
- checkpoint 状态审计确认 `0731_21` 相对 Encoder 仅新增两个 `256×256` 无偏置线性门，
  共 `131,072` 个参数；状态量由 `22,784,060` 增至 `22,915,132`，增幅约 `0.575%`。
  24/25/26 相对 21 不新增参数，但末层需额外计算两次父分类分支得到 confidence，不能称为
  零计算开销。最终候选仍须在同卡同温条件下实测训练和推理速度。

## 2026-07-31 22:19 CST decoder 资源与轻量结构更新

- 178 `0731_21` epoch 24 为 cls/det HOTA `52.141/59.381`，相对 Encoder 同点
  `+0.427/-0.138`；四项 AP 均明显提升，继续到 epoch 28。
- 197 `0731_25` epoch 8 为 `43.629/50.129`，DetA/AP 系统性下降；产物验证完整后已停止。
  confidence 三路均弱于原始 `0731_21`，不再创建衍生组合。
- 197 GPU 4,5 已切换到 `0731_27 terminal diagonal factorized evidence`。它把
  `0731_21` 的两个稠密 256×256 门简化为两个逐通道向量，只新增 512 参数，不增加 decoder
  深度、attention、分支或 loss。103 项单测、完整构建、双卡真实 smoke 与 checkpoint 检查
  通过；正式训练已到 epoch 1 iter 50，五项启动门槛通过。
- 当前并行任务：99 `0731_24` 与 252 `0731_26` 观察 epoch 12 后收口，178 `0731_21`
  观察 epoch 28，197 `0731_27` 首看 epoch 4。最终仍以同一 checkpoint 的 cls/det HOTA
  同时超过 `54.437/62.393` 为成功标准，并补做同卡同温效率验证。

## 2026-07-31 22:35 CST 0731_24 epoch-12 收口

- 99 `0731_24 confident-common` e12 的 cls HOTA/DetA/AssA 为
  `48.271/39.526/61.431`，det 为 `56.179/48.821/66.894`。相对 Encoder 同点，
  cls/det HOTA `-1.409/-0.362`，DetA `-1.834/-1.524`，AssA `-0.379/+1.315`。
- pair mAP/AP50 相对同点下降 `0.014159/0.007145`，both-independent mAP/AP50
  下降 `0.014924/0.007281`。因此 e8 的 DetA/AP 损失没有在 e12 恢复，属于连续系统性退化。
- epoch 12 checkpoint、检测 metrics、TrackEval metrics 与 54 个原始评估文件全部验证完整。
  22:35 精确终止唯一目标进程组，screen 与 worker 均退出，99 GPU 0,1 已释放。
- confidence-common、confidence-detail 和 confidence-common+detail 已形成一致否定证据；不再
  扩展 confidence 或 residual-scale 组合。99 后续只接收作用机制独立、结构轻量且不明显增加
  计算量的 decoder 实验。

## 2026-07-31 22:49 CST 0731_28 中心运动约束启动

- `0731_21` e24 的 cls HOTA 已相对同点提高 `0.427`，det AssA 提高 `0.814`，但 det
  DetA 下降 `0.735`、det HOTA 仍低 `0.138`。据此建立结构假设：相邻帧反对称 detail
  应主要表达中心位移，不应同时扰动短时内较稳定的宽、高和旋转角。
- 提交 `8a24666` 增加 `0731_28`：classification common 完全不变，box detail 只作用于
  旋转框 `x/y`，`w/h/angle` 严格保留父模型输出；仍保持零初始化、帧交换等变和精确 pair
  midpoint。相对 `0731_21` 不新增参数、decoder 层、attention、分支、loss 或实质 FLOPs。
- 104 项 decoder 单测通过，覆盖零起点、中心两维非零修正、后三维严格不变、pair midpoint、
  分类正交和 detail gate 梯度。正式/短测配置深拷贝、完整模型构建、launcher 语法与双卡真实
  4-iter smoke 均通过；checkpoint 中 6 组独立 attention 已分化，两个 gate 均有非零更新。
- 99 GPU 0,1 于 22:48 fresh 启动；22:49 到 epoch 1 iter 50，GPU 显存约
  `19.2 GB/rank`，MMEngine 约 `11.17 GB/rank`，总/DN/encoder loss 与梯度有限，
  无 Traceback/OOM/NaN/NCCL。首个 HOTA 结构检查点为 epoch 4。

## 2026-08-01 02:16 CST decoder 运行状态更新

- 252 `0731_29 diagonal + center-motion` e8 的 cls/det HOTA 为
  `44.148/49.376`，相对 Encoder 同点下降 `1.121/0.817`；cls/det DetA 分别下降
  `2.028/3.357`。pair mAP/AP50 下降 `0.028467/0.019227`，
  both-independent mAP/AP50 下降 `0.030575/0.019963`。e8 checkpoint、检测 metrics、
  TrackEval metrics、50 个序列 txt 与 108 个评估文件验证完整后已停止，252 GPU0/1 释放。
- 178 `0731_21` 原位恢复后的 e36 cls/det HOTA 为 `52.699/60.048`，相对 Encoder
  同点 `-0.213/-0.659`；cls/det DetA 为 `-0.262/-0.866`。但 pair mAP/AP50
  提高 `0.004177/0.006375`，both-independent mAP/AP50 提高
  `0.005167/0.008009`，不属于 HOTA、DetA 与 AP 全部同向恶化，继续到 e40。
- 恢复训练的 TrackEval 内存计数复位导致 e36 写入旧 `val_track_0001`。异步评估结束后，
  e36 已完整复制为 `val_track_0009_resume_epoch36`；e8 原始产物也已预先复制为
  `val_track_0002_pre_resume_epoch8`，防止下一次 e40 评估覆盖。两份归档均验证 50 个序列
  txt、108 个文件及 metrics SHA256 一致。
- `HSMOTPairAPMetric` 已修复为从已有 `val_track_XXXX*` 目录恢复最大计数，包含带恢复后缀的
  归档目录；目标测试 5/5 通过。该修复只影响以后 resume 的评估命名，不改变当前训练状态。
- 当前正式运行：99 `0731_28` 已完成 e12 检测评估并等待异步 TrackEval，178 `0731_21`
  等待 e40，197 `0801_01` 已通过 e4 gate 并继续到 e8；252 当前空闲。后续只推进结构简单、机制可解释的轻量 decoder，
  不增加 decoder 深度、额外 attention、高分辨率分支或辅助 loss；论文候选同卡同温吞吐下降
  原则上不得超过 5%。

## 2026-08-01 02:28 CST 0801_01 epoch-4 轻量结构 Gate

- 197 `0801_01 coupled diagonal factorized evidence` e4 cls/det HOTA 为 `36.757/42.605`，
  相对 Encoder 同点 `+0.548/+3.852`。cls DetA/AssA 变化 `-0.079/+1.900`，det 为
  `+0.934/+7.816`，不是依靠单一路径牺牲检测覆盖获得的表面提升。
- pair mAP/AP50 为 `0.154920/0.312850`，相对父轨迹 `-0.002332/+0.016715`；
  both-independent mAP/AP50 为 `0.184995/0.341277`，变化 `+0.000530/+0.018128`。
  e4 checkpoint、检测 metrics、TrackEval metrics 与 50 个序列结果完整。
- 该结构仅新增 256 个逐通道参数，不增加 decoder 层、attention、分支、loss 或矩阵乘法，
  符合复杂度硬约束。e4 只作结构 gate，不据此宣布胜出；继续到 e8/e12 检查双 HOTA 持续性。

## 2026-08-01 02:38 CST 0731_28 e12 与 0801_02 最小剥离

- 99 `0731_28` e12 为 `49.186/56.248`，相对 Encoder 同点 `-0.494/-0.293`；e8 的
  双 HOTA 增益没有保持。因 AP50 与 det AssA 仍提升，按放宽规则继续到 e16，而非单点停止。
- 新候选 `0801_02` 完全保持 Encoder 分类路径，只在最终 box head 对中心 x/y 加入严格反对称、
  midpoint-preserving detail。它只有一个 `256×256` gate（65,536 参数，约 0.29%），不新增
  decoder 层、attention、分支或 loss；106 项单测和完整构建通过，待 252 真实 smoke。

## 2026-08-01 02:47 CST 0801_02 正式启动

- 首次 smoke 在数据加载前发现父配置误指向 99 数据根目录，未执行训练迭代；失败目录完整保留为
  `smoke_0801_02_terminal_center_motion_detail_only_4iter_failed_bad_data_root`。配置改为继承 252
  已验证父链后，模型结构与参数数目不变。
- 修正后双卡真实 4-iter smoke 的总/DN/encoder loss 与 grad norm 全部有限；checkpoint 仅含一个
  detail gate，最大绝对值 `0.0003952`，6 组双帧 attention 最大分化 `0.0007661`，结构检查通过。
- 252 GPU0/1 于 02:44 fresh 启动。正式 iter 50 为 `1.1947 s/iter`、loss `22.3855`、
  grad norm `144.2432`，显存约 `19.2 GiB/rank`，无 Traceback/OOM/NaN/NCCL；五项门槛通过，
  e4 只作结构 gate，e8/e12 判断持续性。

## 2026-08-01 02:58 CST 四路实时复核与启动保护

- 四路训练进程均健康：99 `0731_28` 到 e14、197 `0801_01` 到 e6、252 `0801_02`
  到 e1、178 `0731_21` 已进入 e40；指定 GPU 显存占用合理，总/DN/encoder loss 与 grad norm
  有限，当前时段未发现 Traceback、OOM、NaN、NCCL 或 DDP 异常，也没有尚未登记的新完整评估。
- 运维复核发现 `0801_02` 的正式与 smoke launcher 在 `nounset` 生效时直接加载 Conda。
  已按安全激活顺序补充 `set +u` / `set -u`，两份脚本均通过 `bash -n`；提交为 `d806611`，
  GitHub 与 99/197/252/178 tracked HEAD 已同步。该提交只修复未来启动路径，不重启、不热替换，
  不改变当前四个训练进程的科学代码或轨迹。

## 2026-08-01 03:19 CST 0731_21 e40 收口

- 178 `0731_21 terminal orthogonal factorization` e40 cls/det HOTA 为
  `53.655/60.379`，相对 Encoder 同点 `53.797/61.063` 为 `-0.142/-0.684`。
  cls DetA/AssA 变化 `+0.388/-1.043`，det 为 `-0.572/-0.902`；分类检测覆盖基本受保护，
  但关联和 det 路径仍未恢复。
- pair mAP/AP50 为 `0.3148/0.5390`，相对父轨迹提高 `0.0092/0.0127`；
  both-independent mAP/AP50 为 `0.3536/0.5704`，提高 `0.0107/0.0146`。
  AP 全升说明模型没有整体检测崩溃，但不能替代 cls/det HOTA 的主目标。
- e40 checkpoint、检测结果、TrackEval metrics、50 个序列 txt 与 108 个评估文件完整；旧进程写入的
  `val_track_0002` 已复制归档为 `val_track_0010_resume_epoch40`，两边文件数、序列数与 metrics
  SHA256 一致。结合 e32/e36/e40 连续双 HOTA 未超过父轨迹，03:18 精确终止训练进程组，
  178 GPU0 已释放；该较复杂分支不再 resume，也不进入论文 decoder 主线。
- 178 暂不立即启动新结构。先等待仅 256 参数的 `0801_01` e8 与分类不改动的 `0801_02` e4，
  再选择单一、可解释、低开销的下一步，避免为占满 GPU 运行低信息实验。

## 2026-08-01 03:47 CST 0731_28 e16 收口

- 99 `0731_28 terminal center-motion factorization` e16 cls HOTA/DetA/AssA 为
  `48.845/39.751/62.295`，det 为 `57.176/48.934/69.124`；相对 Encoder 同点
  `51.091/42.604/63.228` 与 `58.320/51.714/67.970`，双 HOTA 分别下降
  `2.246/1.144`，双 DetA 分别下降 `2.853/2.780`。det AssA 的 `+1.154` 不能抵消检测覆盖损失。
- pair mAP/AP50 为 `0.2635/0.4778`，相对父轨迹 `0.2839/0.4923` 下降
  `0.0204/0.0145`；both-independent 为 `0.3017/0.5115`，相对父轨迹
  `0.3223/0.5260` 下降 `0.0206/0.0145`。结合 e12 的双 HOTA 落后，构成连续且系统性的
  HOTA、DetA 与 AP 同向恶化，否定稠密 common gate 与中心运动 detail 的绑定结构。
- epoch 16 checkpoint、检测 metrics、TrackEval、50 序列结果与 108 个评估文件核验完整后，
  精确终止 PGID `580205`；screen 与全部 worker 已退出，99 GPU0/1 为 `0%/10 MiB`。
  99 暂时空闲，等待 `0801_01` e8 与 `0801_02` e4，避免在证据到达前增加新复杂度。

## 2026-08-01 04:14 CST 0801_02 epoch-4 结构 Gate

- 252 `0801_02 terminal center-motion detail-only` e4 cls HOTA/DetA/AssA 为
  `36.757/26.915/54.146`，det 为 `42.974/33.391/56.520`；相对 Encoder 同点
  `36.209/27.068/52.094` 与 `38.753/32.454/47.466`，双 HOTA 分别提高
  `0.548/4.221`。cls DetA 只下降 `0.153`，det DetA 提高 `0.937`，不是用明显检测覆盖损失
  换取表面关联增益。
- pair mAP/AP50 为 `0.1555/0.3083`，相对父轨迹变化约 `-0.0018/+0.0122`；
  both-independent 为 `0.1828/0.3358`，变化约 `-0.0017/+0.0127`。AP50 与双 HOTA
  同向改善，mAP 仅轻微波动，支持“分类保持 Encoder、只给最终中心 x/y 注入反对称运动 detail”
  这一最小结构假设。
- epoch 4 checkpoint、检测 metrics、TrackEval metrics 与 50 个序列结果完整，训练已正常进入
  epoch 5；无 Traceback/OOM/NaN/NCCL。e4 仍只作结构 gate，继续到 e8/e12 检查增益持续性。
  该模型只增加 `65,536` 参数（约 `0.29%`），无新 decoder 层、attention、分支或 loss；
  只有形成持续 HOTA 候选后才做同卡同温速度测试，吞吐下降上限为 `5%`。

## 2026-08-01 04:36 CST 0801_01 epoch-8 持续性复核

- 197 `0801_01 coupled diagonal factorized evidence` e8 cls HOTA/DetA/AssA 为
  `43.546/34.571/58.371`，det 为 `50.488/44.120/59.866`；相对 Encoder 同点
  `45.269/37.663/57.301` 与 `50.193/47.061/55.145`，cls/det HOTA 变化
  `-1.723/+0.295`，DetA 变化 `-3.092/-2.941`，AssA 变化 `+1.070/+4.721`。
  e4 的双 HOTA 强信号没有保持，det 的微弱正增益主要来自 AssA 搬运。
- pair mAP/AP50 为 `0.2118/0.4015`，相对父轨迹 `0.2377/0.4309` 下降
  `0.0259/0.0294`；both-independent 为 `0.2454/0.4323`，相对
  `0.2762/0.4660` 下降 `0.0308/0.0337`。epoch 8 checkpoint、检测 metrics、
  TrackEval metrics、50 序列 txt 与 108 个评估文件核验完整，无运行异常。
- 该点表现为 cls HOTA、双 DetA 与四项 AP 系统性下降，但 det HOTA 尚有 `+0.295`，不满足
  “双 HOTA、DetA、AP 全部同向恶化”的强停止条件。按放宽后的早期判定继续到 e12 作最后一次
  持续性复核；当前标记为弱候选、倾向否定，不创建共享门衍生实验，也不提前做速度测试。

## 2026-08-01 08:54 CST 0801_03 epoch-8 收口

- 252 `0801_03 terminal diagonal center-motion detail-only` e8 cls HOTA/DetA/AssA 为
  `44.183/35.231/58.370`，det 为 `50.011/44.289/58.441`；相对 Encoder 同点
  `45.269/37.663/57.301` 与 `50.193/47.061/55.145`，双 HOTA 下降
  `1.086/0.182`、双 DetA 下降 `2.432/2.772`，只有 AssA 提高 `1.069/3.296`。
- pair mAP/AP50 为 `0.210025/0.406177`，both-independent 为
  `0.245219/0.438068`；相对 Encoder 约下降 `0.0277/0.0247` 与
  `0.0310/0.0279`。逐通道化没有避免稠密 `0801_02` 的中期检测覆盖退化。
- epoch 8 checkpoint、`val_det/epoch_07/metrics.json`、TrackEval `async_done=1`、
  50 个序列 txt 与 108 个评估文件完整。结构审计通过：6 组独立 attention 最大差异
  `0.059180`，唯一 256 维 detail gate 最大绝对值 `0.337337`；结论是结构充分学习但机制失败。
- 双 HOTA、双 DetA 和四项 AP 同向下降，达到放宽规则下的系统性停止条件。08:54 精确终止
  PGID `1232033`；screen 与全部训练 worker 退出，252 GPU0/1 连续采样均为空闲。
  该 terminal center-motion detail-only 路线不再 resume，也不派生稠密门、额外 attention 或参数扫描。
## 2026-08-01 09:11 CST：178 恢复 0731_01 至 epoch 12

- `0731_01 shared-attention + antisymmetric detail` 从原 `epoch_8.pth` 恢复，使用 178 GPU0、物理 batch 8；不改变结构、损失或训练协议。该候选仅增加 `122,592` 参数（`+0.539%`），同机相近协议训练速度下降约 `2.3%`。
- 首次恢复遇到 PyTorch 2.6 对旧 MMEngine checkpoint 的 `weights_only` 加载拒绝，未进入训练且 GPU 自动释放；专用 launcher 对受信任本地断点显式设置 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` 后恢复成功。
- 09:10 正式到达 epoch 9 iter 50：`0.9572 s/iter`、loss `10.2507`、grad norm `46.6676`，总/DN/encoder losses 均有限，GPU0 约 `31.4 GiB`，无 Traceback/OOM/NaN/NCCL/DDP 异常。状态为 `RUNNING`，只续至 epoch 12 完整评估。

## 2026-08-01 09:21 CST：252 恢复 0731_05 至 epoch 20

- 该候选 e8 曾相对 Encoder 双超 `+0.072/+1.396`，e12/e16 仅出现窄幅 HOTA 差距且 AP 未系统性退化；按当前持久性规则从原 `epoch_16.pth` 补证一个评估点，而非新建复杂模型。
- 参数增量为 `122,592`（`+0.539%`），沿用 252 GPU0/1、physical `2xb4` 和原优化轨迹。09:20 到 epoch 17 iter 50，`1.1663 s/iter`、loss `10.4376`、grad norm `61.8605`，总/DN/encoder losses 均有限，两卡各约 `19.4 GiB`，无训练异常。

## 2026-08-01 10:23 CST：178 0731_01 epoch-12 收口

- epoch 12 cls HOTA/DetA/AssA 为 `48.465/40.393/60.978`，det 为
  `55.436/49.657/63.797`；相对 Encoder 同点 HOTA `49.680/56.541` 下降
  `1.215/1.105`，双 DetA 与双 AssA 也均下降。
- pair mAP/AP50 为 `0.271857/0.477335`，both-independent 为
  `0.310303/0.513735`。checkpoint、检测 metrics、50 序列、TrackEval
  `async_done=1` 与 108 个评估文件均已核验。
- e8 的单侧 det/AP 优势没有保持为中期双 HOTA 增益。10:22 精确终止 PGID
  `2522015`，目标进程全部退出且 178 GPU0 释放；不继续 e16，不从该结构派生参数扫描或复杂模块。

## 2026-08-01 10:52 CST：252 0731_05 epoch-20 收口

- epoch 20 cls HOTA/DetA/AssA 为 `51.640/43.389/63.028`，det 为
  `58.491/51.360/68.934`。相对 Encoder 同点，cls 为 `+0.126/+0.398/-0.721`，
  det 为 `-0.431/-0.839/+0.205`；仍是两条输出在检测覆盖与关联之间的相反取舍。
- pair mAP/AP50 为 `0.289663/0.516488`，both-independent 为
  `0.329802/0.551333`。AP50 分别提高约 `0.0139/0.0160`，但不能替代 det HOTA
  的失败。checkpoint、检测 metrics、50 序列、TrackEval `async_done=1` 与 108 个文件完整。
- 10:51 精确终止 PGID `1387291`，252 GPU0/1 均已释放。该轨迹不继续到 e24，
  也不派生 scale、loss 权重、类别重权或更深 decoder。

## 2026-08-01 11:59 CST：178 启动 0801_05 symmetric-feature

- `0801_05` 与 197 的 `0801_04 symmetric-position` 构成正交单因素对照：前者只消除
  frame-feature fusion 的帧序偏置，后者只消除 pair-position 的帧序偏置；两者均保留
  独立 prev/curr deformable cross-attention。`0801_05` 把两帧输出均值重复后送入现有
  `cross_fusion` 一次，因此不新增参数、层、attention、分支、loss 或矩阵乘法。
- 提交 `9bda2ed` 已精确同步四机和 GitHub。99 上 113 项 decoder 单测、formal/smoke
  配置深拷贝、完整父/新模型构建、状态结构与同参数量 `22,758,775` 检查通过。
- 178 GPU0 真实 4-iter smoke 的最终 loss/grad norm 为 `20.1570/185.1551`，总、DN、
  encoder proposal loss 均有限；checkpoint 中 24 组 prev/curr attention 独立，最大差异
  `0.00076836`，6 个 fusion 张量有限。formal fresh 于 11:57 启动，11:58 iter 50 为
  `0.9385 s/iter`、loss `21.1807`、grad norm `98.8892`，五项门槛通过且 GPU0 约
  `31.4 GiB`。首判 e4，仅以 e8/e12 同点 Cls/Det HOTA 决定持续性。

## 2026-08-01 12:32 CST：252 0730_10 epoch-8 收口

- 完整 symmetric-pair 在 e8 得到 cls HOTA/DetA/AssA `42.890/33.328/58.017`，det
  `48.228/41.431/57.793`。相对 Encoder e8 的 `45.269/50.193`，双 HOTA 分别下降
  `2.379/1.965`；cls/det DetA 分别下降 `4.335/5.630`，尽管 AssA 提高，仍是明显的
  DetA→AssA 搬运，未保持 e4 的双 HOTA 信号。
- pair mAP/AP50 为 `0.2034/0.3969`，both-independent 为 `0.2390/0.4301`；已知相对
  Encoder pair mAP 与 both-independent AP50 分别下降约 `0.0343/0.0359`。HOTA、DetA
  与 AP 同向系统性退化，满足放宽规则下的精确淘汰条件。
- `epoch_8.pth`、`val_det/epoch_07/metrics.json`、50 个检测/跟踪序列、TrackEval
  `async_done=1`、28 个 CSV 与总计 108 个评估文件均已核验。12:31 精确终止 PGID
  `1466786`，252 GPU0/1 均为 `0% / 1 MiB`。完整对称化不继续到 e12；197 的
  position-only 与 178 的 feature-only 因保留独立 cross-attention，继续作为正交归因实验。

## 2026-08-01 13:12 CST：178 0801_05 epoch-4 结构信号

- feature-only symmetry 的 e4 cls HOTA/DetA/AssA 为 `34.947/25.960/49.970`，det 为
  `38.300/30.836/48.870`。相对 Encoder e4，双 HOTA 为 `-1.262/-0.453`，双 DetA 为
  `-1.108/-1.618`；cls AssA 下降 `2.124`，det AssA 提高 `1.404`。det 路径呈现轻度
  DetA→AssA 搬运，暂不支持“去除 feature fusion 帧序”这一假设。
- pair mAP/AP50 为 `0.1399/0.2951`，both-independent 为 `0.1685/0.3240`；mAP 相对
  Encoder 同点约下降 `0.016–0.017`，但 AP50 基本持平。`epoch_4.pth`、检测结果、50 个
  序列、TrackEval `async_done=1` 与总计 108 个评估文件均已核验。
- e4 只作结构信号，不作提前淘汰。13:11 训练已健康进入 epoch 5 iter 500，loss
  `10.1890`、grad norm `48.9170`，GPU0 约 `31.5 GiB`，无 Traceback/OOM/NaN/NCCL。
  继续到 e8 检验持续性，不追加 gate、scale、attention 或其他复杂结构。
## 2026-08-01 13:59 CST：decoder 三路状态

- 197 GPU4/5：`0801_04 symmetric-position` 继续训练到 e8；e4 cls/det HOTA
  `35.531/41.704`，相对 Encoder `-0.678/+2.951`。
- 178 GPU0：`0801_05 symmetric-feature` 继续训练到 e8；e4 cls/det HOTA
  `34.947/38.300`，相对 Encoder `-1.262/-0.453`。
- 252 GPU0/1：`0801_06 symmetric-position + residual-preserving fusion` 已通过
  双卡真实 smoke 与正式 iter-50 五项门槛，状态为 `RUNNING`。模型参数不增加；同机
  2xb4 日志对照仅慢约 `0.25%`，符合效率约束。
- 三路均以 cls/det HOTA 为主判据，DetA/AssA 用于解释，AP 用于系统性退化诊断；
  不追加参数扫描、类别重加权、scale sweep、额外层、attention 或 loss。
## 2026-08-01 14:16 CST：178 释放

- 178 `0801_05 symmetric-feature` e8 完整结果为 cls/det HOTA
  `43.178/49.175`，相对 Encoder 同点 `-2.091/-1.018`；双 DetA、双 AssA和两项 mAP
  也同时下降。完整产物核验后已停止，不再继续 e12。
- 178 GPU0 已释放。当前有效训练保留为 197 `0801_04 symmetric-position` 与
  252 `0801_06 symmetric-position + residual-preserving fusion`；等待它们的 e8/e4
  证据后再决定是否利用 178/99 启动下一项，避免无证据并行消耗。

## 2026-08-03 01:20 CST：严格合并增益目标下的资源状态

- 当前成功门槛为同点 cls/det HOTA 均超过 `54.437/62.393`，且绝对增量和严格大于
  `1.5`（总 HOTA 严格大于 `118.330`）。`0801_09` e56 总增益仅 `0.279`，需继续探索。
- 252 GPU0/1：`0803_01` 零参数 pair-shared objectness iterative decoder 已通过真实双卡
  smoke 与正式 iter-50 门槛，formal 已到 e1 iter 600，状态 `RUNNING`。GPU2/3 保留空闲。
- 197 GPU4/5：正在执行 `0801_09` 的 2x4 portability smoke；因该机 Python/model 首次初始化
  很慢，尚未完成 4 iter，故还没有从 epoch 56 启动正式续训。GPU0–3 当前空闲但不纳入本任务。
- 178 GPU0 被外部作业占用；99 SSH 仍不可用。所有启动均使用提交 `bd1c329`，未在存活训练中
  热更新代码。

## 2026-08-03 01:38 CST：252 四卡双主线

- 252 GPU0/1 继续运行 `0803_01` fresh；GPU2/3 已从 `0801_09` e56 恢复原优化轨迹。
  两项均通过正式 iter-50 有限性门槛，后者恢复点为 e57 iter 50、`1.3730 s/iter`。
- 197 的同配置 4-iter smoke 数值正常但慢至约 `80 s/iter`，因此不在该机启动正式续训；
  资源策略从“等待 197”改为 252 四卡并行，避免晚期节点被机器异常速度阻塞。

## 2026-08-03 02:12 CST：178 启动 0803_02 单卡主线

- 178 GPU0 已释放外部占用；在不突破该机单卡授权的前提下启动 `0803_02 pair-shared shape
  refinement`。单卡 batch 8 与 252 的全局 batch 8 一致，验证 loader 维持单进程设置。
- commit `d3dca93` 已在无目标训练进程时安全 fast-forward 到 178；配置深拷贝、完整模型构建、
  真实数据 smoke、checkpoint 更新审计均通过。正式 PGID `2857661` 的 iter 50 为
  `0.9497 s/iter`、loss `21.0192`、grad norm `104.8574`，五项运行门槛通过，状态
  为 `RUNNING`。
- 当前并行布局为：252 GPU0/1 跑 `0803_01` fresh，252 GPU2/3 跑 `0801_09` e56 续训，
  178 GPU0 跑 `0803_02` fresh。197 GPU4/5 因异常慢暂不部署，99 SSH 仍不可达。

## 2026-08-03 02:36 CST：0803_01 e4 完整评测

- 252 GPU0/1 的 `0803_01` e4 checkpoint、检测、50 序列、TrackEval metrics、28 个 CSV 和
  54 个原始评估文件均完整。cls/det HOTA 为 `30.075/36.992`，相对 Encoder e4 分别
  `-6.134/-1.761`；双 DetA、双 AssA 与四项 AP 也全部下降。
- 该结果只登记为早期系统性负向证据，不按 e4 直接停止。训练已恢复到 e5，继续保留 e8/e12
  延迟收敛窗口；252 GPU2/3 的 `0801_09` e56 续训仍正常，02:35 位于 e59 iter 750。
- 99 已通过正确的 SSH 端口恢复可达，但 GPU0/1 被外部计算占用，不抢占；资源状态从
  `UNREACHABLE` 修正为 `REACHABLE/EXTERNALLY_OCCUPIED`。

## 2026-08-03 02:49 CST：0803_03 后备候选已验证但未排队

- 本地 commit `9d90733` 新增仅共享普通 query angle residual 的 `0803_03`；中心、宽高、
  分类与 DN 均不共享。252 临时隔离 clone 的 3 项定向测试和完整模型构建通过，完整模型
  `22,771,111` 参数、参数/state 零增量。
- 临时 clone 与 bundle 已在验证后清理；活动中的 252/178 仓库均未热更新。该候选当前仅为
  `PREPARED`，不属于排队任务；部署要等待 `0803_02` 完整节点与授权 GPU 释放。

## 2026-08-03 03:20 CST：252 0801_09 e60 完整评估

- e60 cls/det HOTA 为 `54.489/62.422`，绝对合并增益仅 `0.081`，总 HOTA `116.911`，
  未达到 `>118.330`。相对 e56，cls HOTA/DetA/AssA 为 `-0.164/-0.391/+0.360`，det 为
  `-0.034/-0.029/+0.010`；pair 与 both-independent 的 mAP/AP50 也均轻微回落。
- checkpoint、检测、50 序列、TrackEval `async_done=1`、54 个 eval 文件与总计 108 个 raw
  文件完整。训练已进入 e61，保留到 e64 作成熟平台确认；当前不释放 GPU2/3。

## 2026-08-03 03:28 CST：178 0803_02 e4 完整评估

- e4 cls HOTA/DetA/AssA 为 `33.322/27.476/42.883`，det 为
  `37.485/33.886/42.930`。相对父线 e4，cls HOTA/DetA/AssA 为
  `-0.984/-0.169/-1.724`，det 为 `-1.105/+0.247/-2.992`；完整 shape 共享目前主要伤害
  关联，det 覆盖没有同步退化。
- pair mAP/AP50 为 `0.146883/0.272219`，both-independent 为
  `0.189566/0.338169`。checkpoint、检测、50 序列、TrackEval `async_done=1`、54 个 eval
  文件与总计 108 个 raw 文件完整。训练已进入 e5，继续 e8/e12，不按 e4 释放 GPU0。

## 2026-08-03 04:08 CST：252 0803_01 e8 完整评估

- e8 cls HOTA/DetA/AssA 为 `39.208/32.700/49.369`，det 为
  `46.359/41.539/53.625`。相对 `0801_09` 父线 e8，cls HOTA/DetA/AssA 为
  `-2.764/-2.481/-3.763`，det 为 `-1.819/-3.195/-0.293`；相对 Encoder e8 的
  cls/det HOTA 为 `-6.061/-3.834`。
- pair mAP/AP50 为 `0.196815/0.351435`，both-independent 为
  `0.235883/0.408144`；四项比 e4 明显恢复，但仍全部低于父线 e8。checkpoint、检测、
  50 序列、TrackEval `async_done=1`、54 个 eval 文件与总计 108 个 raw 文件完整。
- 训练已进入 e9，并按 decoder 晚收敛约束继续 e12；不以 e8 直接停止。当前证据只用于否定
  继续派生 objectness 硬共享变体，下一结构优先级仍由正交几何线和成熟节点共同决定。

## 2026-08-03 04:43 CST：178 0803_02 e8 完整评估

- e8 cls HOTA/DetA/AssA 为 `42.652/35.433/54.403`，det 为
  `47.723/42.688/55.569`。相对 `0801_09` 父线 e8，cls HOTA 提高 `0.680`，det HOTA
  降低 `0.455`；双 AssA 提高 `1.271/1.651`，但 det DetA 降低 `2.046`。
- pair mAP/AP50 `0.218642/0.387668`、both-independent `0.262145/0.443668`，四项均高于
  父线 e8。checkpoint、检测、50 序列、TrackEval `async_done=1`、54 个 eval 文件与总计
  108 个 raw 文件完整；训练已进入 e9 并继续 e12，178 GPU0 不释放。
- 完整 shape 共享在 e8 已恢复检测 AP 并改善关联，但 `w/h` 共享仍损伤 det 覆盖；因此保留
  `0803_03` angle-only 为资源释放后的下一候选，不做 gate、scale、class-aware 或 reweight 派生。

## 2026-08-03 04:58 CST：252 0801_09 e64 平台确认并停止

- e64 同一 checkpoint 的 cls HOTA/DetA/AssA 为 `54.326/44.326/68.761`，det 为
  `62.572/54.840/73.891`。cls 低于严格 Encoder 基线 `0.111`；总和 `116.898`，相对基线
  合并增益仅 `0.068`，远低于要求的 `>1.5`。
- pair mAP/AP50 为 `0.313163/0.528518`，both-independent 为
  `0.352795/0.564884`，均较 e60 回落。epoch-64 checkpoint、50 序列、5416 条记录、
  TrackEval `async_done=1`、54 个 eval 文件和总计 108 个 raw 文件完整。
- e56/e60/e64 已充分证明纯续训进入平台。04:57 精确终止 PGID `3292233`，保留全部产物；
  GPU2/3 连续检查为空闲，下一步在独立 checkout 部署 `0803_03`，避免热更新 GPU0/1 正在使用的
  `0803_01` 活跃仓库。

## 2026-08-03 05:08 CST：252 0803_03 angle-only 正式启动

- 为避免热更新 GPU0/1 的活跃 `0803_01` 仓库，在
  `/data/users/litianhao01/PairMmot_angle_0803_03` 建立隔离 checkout，并以 Git bundle
  fast-forward 到提交 `0989cfd`；关键 launcher/config SHA-256 与本地一致，原活跃仓库保持
  `bd1c329`。目标 formal/smoke 配置 deepcopy 和完整父/新模型构建通过：二者均为
  `22,771,111` 参数、`711` 个 state tensor，增量为零。
- GPU2/3 连续空闲后，真实双卡 4-iter smoke 完成。4 个迭代的 loss 为
  `12.9387/19.2688/19.3129/20.2840`，grad norm 为
  `102.8563/109.7766/100.7381/105.8495`；总、DN、encoder loss 有限，无 Traceback、OOM、
  NaN、NCCL、DDP reduction 或 unused-parameter 错误，364 MB `iter_4.pth` 与 checkpoint
  语义检查通过。
- formal fresh 于 05:06 启动，PGID `3460950`。05:07 iter 50 为 `1.3920 s/iter`、loss
  `21.3894`、grad norm `130.3343`，双卡各约 19.2 GiB；进程、GPU、正式日志、正常训练迭代、
  有限总/DN/encoder loss 五项门槛全部通过。首批完整节点为 e4/e8/e12，不按 e4/e8 直接否决。

## 2026-08-03 05:32 CST：252 0803_01 e12 完整评估并停止

- e12 cls HOTA/DetA/AssA 为 `43.962/35.877/57.193`，det 为
  `52.094/46.748/60.144`；相对 `0801_09` 父线 e12 的 cls/det HOTA 仍低
  `3.433/2.342`，相对 Encoder e12 低 `5.718/4.447`。本实验从 e8 到 e12 的 HOTA 已恢复
  `4.754/5.735`，故停止依据是成熟节点的结构性负向结果，不是 e4/e8 早停。
- e12 pair mAP/AP50 `0.222812/0.393028`，both-independent `0.265827/0.449310`；checkpoint、
  5416 条记录、50 序列、TrackEval `async_done=1`、28 个 CSV 与 108 个评估文件完整。
  05:31 精确停止 PGID `3268273`，全部成员退出，252 GPU0/1 释放。GPU2/3 的 `0803_03`
  继续健康训练，05:32 位于 e2；不在 angle-only 首个完整节点前盲目占用释放资源。

## 2026-08-03 05:57 CST：178 0803_02 e12 完整评估并停止

- e12 cls HOTA/DetA/AssA 为 `46.101/38.593/57.475`，det 为
  `50.453/46.095/57.200`；相对 `0801_09` 父线 e12 的 cls/det HOTA 仍低
  `1.294/3.983`。e8→e12 已恢复 `3.449/2.730`，因此该结论来自 e4/e8/e12 完整轨迹，
  不是按早期节点停止。
- pair mAP/AP50 为 `0.233785/0.420219`，both-independent 为
  `0.277695/0.473812`；checkpoint、5416 条记录、50 序列、TrackEval `async_done=1`、
  28 个 CSV 与 108 个评估文件完整。05:56 精确停止 PGID `2857661`，9 个成员全部退出，
  178 GPU0 为 `1 MiB/0%`。不派生完整 shape 的 gate/scale 版本，继续等待 `0803_03`
  angle-only 的 e4/e8/e12 局部几何证据。

## 2026-08-03 06:27 CST：178 0803_04 周期角度共识正式启动

- `0803_04` 将普通 query 的两帧角度更新解释为 π 周期切空间增量，以圆周中点共享更新，再分别
  相对原 reference 编码；x/y/w/h、分类与 DN 不变。该结构零参数、class-agnostic、无 reweight、
  新 loss/attention/layer，和 `0803_03` 的 raw-logit residual 均值构成坐标语义对照。
- 178 上 3 项定向测试和完整 126 项 decoder 回归通过；配置深拷贝与父/新完整构建证明均为
  `22,771,111` 参数、711 个 state tensor。首次 smoke 因 4 iter 小于 logger interval 50 而
  缺 loss/grad 日志，只保留为不充分短测；retry1 逐 iter 记录的 loss
  `21.3730/20.6351/20.9249/21.2029`、grad norm
  `61.5507/73.7742/81.8450/75.9738`，DN/encoder loss 与 checkpoint 检查全部通过。
- formal fresh 06:25 启动，PGID `2893156`。iter 50 为 `0.9421 s/iter`、loss `21.0028`、
  grad norm `102.4835`；9 个成员、GPU0 约 31.4 GiB、正式日志与所有 loss group 均正常，五项
  门槛通过。首批完整节点仍为 e4/e8/e12，禁止把 e4/e8 当作直接否决点。

## 2026-08-03 06:46 CST：252 0803_03 e4 完整评估

- e4 cls HOTA/DetA/AssA `31.076/24.972/41.528`，det
  `37.040/31.623/44.375`；相对 `0801_09` 父线 e4 的 HOTA 约低 `3.230/1.550`，
  DetA 低 `2.673/2.016`，AssA 低 `3.079/1.547`。早期覆盖和关联同时退化。
- pair mAP/AP50 `0.127673/0.238560`，both-independent `0.170334/0.309608`，相对父线
  四项约低 `0.016450/0.038933/0.019225/0.040032`。checkpoint、5416 条记录、50 序列、
  TrackEval `async_done=1`、28 个 CSV 与 108 个评估文件完整。
- 训练已进入 e5，按用户约束继续 e8/e12，不以 e4 停止；该结果只支持用 `0803_04` 区分
  raw-logit 坐标失真与角度共享本身，不支持参数扫描或复杂化。

## 2026-08-03 06:59 CST：252 0803_05 normalized-center 正式启动

- `0803_05` 只改变普通 query 的中心 refinement 坐标：解码候选中心后，以各自 reference 的
  `w/h` 表示局部归一化增量，求两帧共同校正并映回各自 reference；尺寸、角度、分类、DN、loss、
  attention 和 decoder 层数不变。结构为零参数、class-agnostic、无 reweight，使用隔离 checkout，
  未热更新 GPU2/3 活跃的 `0803_03` 仓库。
- 3 项定向测试与 129 项 decoder 回归通过；完整父/新模型均为 `22,771,111` 参数、711 个 state
  tensor。双卡真数据 smoke 四步 loss/grad 均有限，364,473,270-byte checkpoint 与 DN 隔离语义
  检查通过，结束后 GPU0/1 回收。
- formal fresh 于 06:57 启动，PGID `3549855`；iter50 为 `1.1465 s/iter`、loss `21.3848`、
  grad norm `102.6548`，两卡各约 19.2 GiB，正式进程、有限总/DN/Encoder loss、资源占用与错误
  扫描五项门槛全部通过。继续收集 e4/e8/e12，不按 e4/e8 单点停止。

## 2026-08-03 07:10 CST：178 0803_06 frame-evidence classification 已准备

- 对 `0801_09` 的分类作用路径审计表明，两帧 iterative classification residual head 读取同一
  融合后 state，而每层已经存在的 prev/curr cross-attention evidence 没有进入对应分类 head。
  `0803_06` 将既有帧证据分别送入对应帧分类 residual；共享 recurrent query、框回归、reference、
  DN、loss 与训练协议不变。
- 新路径不增加参数、attention、loss、层数、class-aware 信号或 reweight。178 隔离 checkout
  `/data1/users/litianhao01/PairMOT_framecls_0803_06` 为干净提交 `fd790e9`；2 项定向测试、完整
  131 项 decoder 回归、配置深拷贝、launcher 语法和完整构建均通过，构建结果为
  `22,771,111` 参数、参数增量 `0`、711 个 state tensor。
- 当前仅为 `PREPARED`：没有等待进程、未运行真数据 smoke、未建立正式 workdir、未占用 GPU。
  178 GPU0 的 `0803_04` 保持原仓库原进程运行；待其完整节点与资源决策后，再按真实 smoke、
  checkpoint 语义检查和 formal iter-50 五门槛顺序部署。

## 2026-08-03 07:46 CST：178 0803_04 e4 完整评估

- e4 cls HOTA/DetA/AssA 为 `36.024/29.194/46.643`，det 为
  `43.788/34.870/57.251`。相对 `0803_03` raw-logit angle e4，双 HOTA
  `+4.948/+6.748`、双 DetA `+4.222/+3.247`、双 AssA `+5.115/+12.876`；π 周期
  切空间显著修复 raw-logit 共享的坐标问题。
- 相对父线 `0801_09` e4，HOTA 为 `+1.718/+5.198`、DetA 为 `+1.549/+1.231`、AssA
  为 `+2.036/+11.329`。相对 Encoder e4，HOTA 为 `-0.185/+5.035`：det 强正向，但 cls
  仍有早期关联缺口，故继续到 e8/e12，不把 e4 当作停止或最终通过节点。
- pair mAP/AP50 `0.1634/0.3034`，both-independent `0.2103/0.3774`；`epoch_4.pth`
  369,971,828 bytes，5416 条记录、50 序列、TrackEval `async_done=1`、28 个 CSV 与 108 个
  评估文件完整。07:43 训练已进入 e5；178 不释放，`0803_06` 继续 `PREPARED`、未排队。

## 2026-08-03 07:54 CST：178 0803_07 组合候选已准备

- `0803_07` 正交组合两个零参数机制：分类 head 使用各帧已有 cross-attention evidence；普通 query
  的 angle residual 使用 π 周期切空间圆周中点。共享 recurrent query、x/y/w/h、DN、loss、
  attention 数量与 decoder 深度不变；无 class-aware 路由或 score reweight。
- 组合不变量测试证明，在相同权重/输入下，shared hidden state 与全部 periodic references 和
  `0803_04` 逐元素一致，只有返回给分类 head 的 prev/curr evidence 分离。目标环境 1 项定向测试、
  完整 132 项 decoder 回归、配置深拷贝、launcher 语法和完整构建全部通过：`22,771,111`
  参数、参数增量 `0`、711 个 state tensor。
- 隔离 checkout 为 `/data1/users/litianhao01/PairMOT_framecls_0803_06`，提交 `ee36e33`；活动训练
  仓库仍为 `9fb501a`，未热更新且 PGID `2893156` 的 9 个成员存活。该候选仅 `PREPARED`，没有
  smoke、formal workdir、等待进程或 GPU 占用；部署顺序等待 `0803_04` e8/e12 成熟证据。

## 2026-08-03 08:22 CST：252 0803_03 raw-angle e8 完整评估

- e8 cls HOTA/DetA/AssA 为 `40.644/33.308/52.868`，det 为
  `47.265/43.143/53.748`。相对自身 e4，双 HOTA 恢复 `+9.568/+10.225`、双 DetA
  恢复 `+8.336/+11.520`、双 AssA 恢复 `+11.340/+9.373`，直接证明不能用 e4
  否决 decoder；训练继续 e12。
- 相对同机制父线 `0801_09` e8，cls/det HOTA 仍低 `1.328/0.913`，DetA 低
  `1.873/1.591`，AssA 低 `0.264/0.170`。raw-logit angle 共识在 e8 已接近父线，
  但尚未形成任何超过严格 Encoder 的证据。
- pair mAP/AP50 为 `0.1899/0.3504`，both-independent 为 `0.2340/0.4138`。
  `epoch_8.pth` 为 375,537,846 bytes；5416 条记录、50 序列、TrackEval
  `async_done=1`、28 个 CSV 与 108 个非空评估文件完整。训练已无缝进入 e9，PGID
  `3460950` 保持运行。

## 2026-08-03 08:28 CST：252 0803_05 normalized-center e4 完整评估

- e4 cls HOTA/DetA/AssA 为 `31.737/25.308/42.669`，det 为
  `37.202/32.484/43.663`。相对 raw-angle `0803_03` e4，双 HOTA 仅
  `+0.661/+0.162`；cls DetA/AssA 与 det DetA 略增，但 det AssA 下降 `0.712`，
  尚无中心局部坐标共识的明确早期优势。
- 相对 `0801_09` 父线 e4，cls/det HOTA 仍低 `2.569/1.388`。pair mAP/AP50 为
  `0.1266/0.2417`，both-independent 为 `0.1684/0.3118`；5416 条记录、50 序列、
  TrackEval `async_done=1`、28 个 CSV 与 108 个非空文件完整。
- 训练已进入 e5，PGID `3549855` 保持运行。该结果只登记为 e4 归因，不作为停止理由；
  继续收集 e8/e12，避免把 decoder 慢收敛误判为结构失败。

## 2026-08-03 09:03 CST：178 0803_04 periodic-angle e8 完整评估

- e8 cls HOTA/DetA/AssA 为 `45.587/38.410/56.277`，det 为
  `52.915/46.571/62.716`。相对 raw-angle `0803_03` e8，双 HOTA
  `+4.943/+5.650`、双 DetA `+5.102/+3.428`、双 AssA `+3.409/+8.968`；
  周期切空间优势从 e4 延续到 e8，不是单个早期点的偶然波动。
- 相对 `0801_09` 父线 e8，双 HOTA 为 `+3.615/+4.737`；相对 Encoder e8，双 HOTA
  为 `+0.318/+2.722`，合并提升 `+3.040`。这是当前 decoder 首次在同一 checkpoint、同一
  训练点双侧超过 Encoder；但严格目标仍以最终 `54.437/62.393` 为准，所以只记为强正向机制证据，
  不宣告达标，继续 e12 与成熟训练。
- pair mAP/AP50 为 `0.2423/0.4242`，both-independent 为 `0.2917/0.4922`；
  `epoch_8.pth` 为 375,562,996 bytes，5416 条记录、50 序列、TrackEval `async_done=1`、
  28 CSV 与 108 个非空文件完整。PGID `2893156` 的 9 个成员存活，09:02 已到 e9 450/1038。
- `0803_07 frame-evidence + periodic-angle` 因而提升为下一优先候选。已新增 252 双卡协议的配置、
  4-iter smoke 和安全 launcher，当前只完成本地语法检查；GPU2/3 仍由 `0803_03` 占用，未 smoke、
  未建 formal workdir、未排队，也未热更新活动仓库。

## 2026-08-03 09:10 CST：252 0803_07 双卡候选完成目标环境静态验证

- 在新建隔离 checkout `/data/users/litianhao01/PairMmot_framecls_periodic_0803_07` 导入提交
  `f3752db`；活动 `0803_03` 仓库仍为 `0989cfd`，没有 fetch、checkout 或工作区写入。
- 安全激活目标 py310 并把隔离 repo 放到 `PYTHONPATH` 首位后，配置深拷贝与父/新完整模型构建
  通过：两者均为 `22,771,111` 参数、`711` 个 state tensor，参数与 state 增量严格为零；
  252 formal/smoke 配置分别保持 2×b4 和 4-iter 语义。
- 完整 decoder 回归 `132 passed`，另有 2 个 subtests 通过；formal/smoke launcher 的 `bash -n`
  均通过。当前状态仍为 `PREPARED`：没有真数据 smoke、formal workdir、等待进程或 GPU 占用；
  待 `0803_03` e12 完整收口并释放 GPU2/3 后再执行 smoke 与 formal 五门槛。

## 2026-08-03 09:58 CST：252 0803_03 e12 收口并停止

- e12 cls HOTA/DetA/AssA 为 `43.687/35.906/55.761`，det 为
  `51.887/46.564/59.976`。相对自身 e8，双 HOTA 继续恢复 `+3.043/+4.622`，说明
  e4/e8 等待是必要的；但相对父线 e12 仍低 `3.708/2.549`，相对 Encoder e12 仍低
  `5.993/4.654`，完整 e4/e8/e12 轨迹已形成成熟系统性负向证据。
- pair mAP/AP50 为 `0.2209/0.4002`，both-independent 为 `0.2618/0.4528`；
  `epoch_12.pth` 为 381,042,678 bytes，5416 条记录、50 序列、TrackEval `async_done=1`、
  28 CSV 与 108 个非空文件完整。
- 09:58 精确终止 PGID `3460950`，23 个成员全部退出，GPU2/3 回到 `1 MiB/0%`；所有产物保留。
  资源转给 e8 已强正向的周期角度与帧证据联合候选 `0803_07`。

## 2026-08-03 09:55 CST：252 0803_05 e8 完整评估

- e8 cls HOTA/DetA/AssA 为 `39.525/32.693/50.508`，det 为
  `45.114/40.486/51.873`；相对自身 e4，双 HOTA 恢复 `+7.788/+7.912`，继续证明
  decoder 的晚收敛，但相对 raw-angle e8 低 `1.119/2.151`，相对父线 e8 低
  `2.447/3.064`。
- pair mAP/AP50 为 `0.1849/0.3383`，both-independent 为 `0.2299/0.4011`；
  375,526,262-byte checkpoint、5416 条记录、50 序列、TrackEval `async_done=1`、28 CSV
  与 108 个非空文件完整。训练继续 e12，不按 e8 停止。

## 2026-08-03 10:05 CST：252 0803_07 正式启动

- 真数据双卡 4-iter smoke 的 loss 为 `12.9405/19.2554/19.2316/20.1093`，grad norm 为
  `103.2723/94.9056/82.0348/79.9432`；总、DN、Encoder loss 有限，364,473,334-byte
  checkpoint 写入，iterative classification residual 与 DN absolute 语义 checker 通过，GPU 回收。
- formal fresh PGID `3694870`；iter 50 为 `1.2858 s/iter`、loss `21.4228`、grad norm
  `119.3820`，总/DN/Encoder loss 有限。进程、GPU2/3 各约 19.2 GiB、正式日志、真实迭代、
  有限 loss 与错误扫描五项门槛通过，状态为 `RUNNING`；继续收集 e4/e8/e12 与成熟节点。

## 2026-08-03 10:17 CST：178 0803_04 periodic-angle e12 完整评估

- e12 cls HOTA/DetA/AssA 为 `47.913/39.775/60.546`，det 为
  `55.257/49.050/64.762`；相对 e8，双 HOTA 再升 `+2.326/+2.342`、双 DetA
  `+1.365/+2.479`、双 AssA `+4.269/+2.046`，没有出现 e8 后回落或平台。
- 相对父线 `0801_09` e12，双 HOTA 仍为 `+0.518/+0.821`；相对 Encoder e12 则低
  `1.767/1.284`。周期角度机制在 e8/e12 都显著优于父线，但尚未达到严格最终阈值，继续更成熟
  checkpoint，不能在 e12 停止。
- pair mAP/AP50 为 `0.2577/0.4395`，both-independent 为 `0.3023/0.4959`；
  `epoch_12.pth` 为 381,090,228 bytes，5416 条记录、50 序列、TrackEval `async_done=1`、
  28 CSV 与 108 个非空文件完整。PGID `2893156` 保持运行。

## 2026-08-03 11:22 CST：252 0803_05 e12 完整评估并释放

- e12 cls HOTA/DetA/AssA `43.161/36.061/53.834`，det
  `49.396/44.754/56.154`；e8→e12 双 HOTA `+3.636/+4.282`，但相对父线同点仍为
  `-4.234/-5.040`，相对 Encoder 同点为 `-6.519/-7.145`。e4/e8/e12 完整轨迹支持成熟负向结论。
- pair mAP/AP50 `0.2142/0.3842`，both-independent `0.2592/0.4419`；checkpoint、
  5416 条检测、50 序列、28 CSV 与 108 个非空文件完整。
- 精确 PGID `3549855` 经配置路径预检后终止，23 个成员全部退出；GPU0/1 为 1 MiB/0%，
  GPU2/3 上 `0803_07` 继续运行。GPU0/1 转入 `0803_06 frame-evidence-cls` 双卡准备。

## 2026-08-03 11:31 CST：252 0803_06 启动与 178 0803_04 e16

- `0803_06` 隔离 checkout 固定提交 `0585d9a`；132 项回归、2 个 subtests、父/新完整构建通过，
  两者均为 `22,771,111` 参数和 711 state tensors，增量为零。真数据 smoke 四步 loss/grad 有限，
  checkpoint 与分类/DN 语义检查通过。
- `0803_06` formal fresh PGID `3765372`；iter50 为 `1.2946 s/iter`、loss `21.4356`、
  grad norm `109.5840`，7 个进程成员、GPU0/1 各约 19.2 GiB、正式日志、真实迭代、有限 loss 与
  错误扫描五门槛通过，状态为 `RUNNING`。
- `0803_04` e16 cls HOTA/DetA/AssA `48.474/40.377/60.513`，det
  `55.272/49.385/64.188`；e12→e16 双 HOTA `+0.561/+0.015`。pair mAP/AP50
  `0.2628/0.4523`，both-independent `0.3057/0.5042`；checkpoint 与 50/28/108 产物完整。
  PGID `2893156` 继续 e20，不能用单个趋平区间否定 decoder 的延迟收敛。

## 2026-08-03 11:45 CST：252 0803_07 e4 完整评估

- e4 cls HOTA/DetA/AssA `32.535/24.868/45.326`，det
  `38.723/31.579/49.105`；相对 `0803_04` periodic-angle e4，双 HOTA
  `-3.489/-5.065`，且 DetA/AssA 同时下降。pair mAP/AP50 `0.1220/0.2350`，
  both-independent `0.1618/0.3021`，也显著低于单因素。
- checkpoint、5416 条检测、50 序列、28 CSV 与 108 个非空文件完整。PGID `3694870`
  继续 e8/e12，禁止按 e4 早停；该点只用于指导下一候选保留共享分类 midpoint、仅注入
  swap-odd 帧细节，不直接替换公共语义状态。

## 2026-08-03 11:52 CST：178 0803_08 静态准备

- `0803_08` 以 shared decoder state 为两帧分类输入的精确 midpoint，只加入既有 frame evidence
  差值的一半作为 `-/+` swap-odd detail；同时保留 `0803_04` 的 periodic-angle refinement。
  回归、reference、DN、loss、attention 和 decoder 深度不变，零参数且无类别路由/reweight。
- 隔离 checkout 固定提交 `8dd19d8`；133 项完整 decoder 测试通过，父/新模型均为
  `22,771,111` 参数、711 state tensors，增量为零；配置深拷贝与 launcher 语法通过。
- 状态为 `PREPARED`。没有真数据 smoke、formal workdir、队列或 GPU 占用；待 `0803_04` e20
  与 `0803_06/07` 成熟节点后按因果证据决定部署，不热更新活动仓库。

## 2026-08-03 12:02 CST：178 0803_09 静态准备

- `0803_09` 只对 normal query 的宽高使用 reference-local log 比例增量共识，并对角度使用 π 周期
  midpoint；中心、分类、DN 与全部主路径不变。它分离了 `0803_02 full shape` 中尺寸不变量与
  已证伪 raw-angle 坐标的混杂，零参数且无类别路由/reweight。
- 隔离 checkout 固定提交 `35e18f1`；135 项完整测试通过，父/新模型均为 `22,771,111` 参数、
  711 state tensors，增量为零；配置深拷贝和 launcher 语法通过。状态为 `PREPARED`，没有真数据
  smoke、formal workdir、队列或 GPU 占用。

## 2026-08-03 12:46 CST：178 0803_04 e20 完整评估

- e20 cls HOTA/DetA/AssA `49.446/41.202/61.747`，det
  `55.397/49.953/63.587`；e16→e20 双 HOTA `+0.972/+0.125`。cls 的 DetA/AssA 同时提高，
  det 的 DetA 提高而 AssA 小幅下降，表现为分类继续延迟收敛、检测接近平台但未全面退化。
- pair mAP/AP50 `0.2706/0.4658`，both-independent `0.3132/0.5154`，四项相对 e16
  均继续提高。checkpoint 392,152,244 bytes，50/28/108 产物完整。
- PGID `2893156` 已进入 e21，继续 e24；严格最终阈值尚差 `4.991/6.996`，状态仍为 `RUNNING`。

## 2026-08-03 13:04 CST：252 0803_06 e4 完整评估

- e4 cls HOTA/DetA/AssA `30.698/24.130/41.666`，det
  `38.350/30.533/49.530`；相对 periodic-angle 单因素 `0803_04` e4，双 HOTA
  `-5.326/-5.438`，相对 Encoder e4 为 `-5.511/-0.403`。直接以帧特异 cross-attention
  evidence 替换共享分类状态在早期同时伤害覆盖与关联。
- pair mAP/AP50 `0.1200/0.2291`，both-independent `0.1621/0.3011`。369,968,182-byte
  checkpoint、5416 条检测、50 序列、28 CSV 与 108 个非空评估文件完整；异步评测已明确结束。
- PGID `3765372` 的 23 个进程成员仍存活并已进入 e5。遵循 decoder 可能慢收敛的约束，继续
  e8/e12，不以 e4 停止；该节点只把后续候选优先级转向保留 shared midpoint 的 `0803_08`，
  不支持直接 frame-evidence 路由。

## 2026-08-03 13:22 CST：252 0803_07 e8 完整评估

- e8 cls HOTA/DetA/AssA `41.380/32.175/56.046`，det
  `47.515/39.879/58.497`；相对 periodic-angle 单因素同点双 HOTA `-4.207/-5.400`，
  相对 Encoder 同点 `-3.889/-2.678`。e4→e8 虽回升 `+8.845/+8.792`，但没有追回直接
  frame-evidence 分类路由造成的结构差距。
- pair mAP/AP50 `0.1887/0.3403`，both-independent `0.2297/0.3976`；相对 periodic-angle
  同点分别低 `0.0536/0.0839` 和 `0.0620/0.0946`。375,531,382-byte checkpoint、
  127426 条检测、50 序列、28 CSV 与 108 个非空评测文件完整。
- PGID `3694870` 的 23 个成员已进入 e9。继续到 e12 以完成慢收敛审计；e8 结果强化
  `0803_08` 必须保留 shared classification midpoint 的设计依据，不再准备 direct 路由变体。

## 2026-08-03 14:05 CST：178 0803_04 e24 平台确认并释放

- e24 cls HOTA/DetA/AssA `50.133/41.785/62.533`，det
  `55.346/50.198/63.103`；相对 e20 双 HOTA `+0.687/-0.051`。cls DetA/AssA 仍同时增长，
  det DetA `+0.245` 但 AssA `-0.484`，确认分类延迟收敛仍在、检测 HOTA 已平台。
- pair mAP/AP50 `0.2754/0.4742`，both-independent `0.3181/0.5229`，相对 e20 四项仍升
  `0.0048/0.0084` 和 `0.0049/0.0075`。397,682,100-byte checkpoint、5416 条检测、
  50 序列、28 CSV 与 108 个非空评测文件完整。
- 严格最终阈值仍差 `4.304/7.047`。完整节点核验后精确终止 PGID `2893156`，9 个成员全部退出，
  178 GPU0 回到 `1 MiB/0%`；e24 断点保留可恢复。下一步优先用 `0803_09` 的 reference-local
  log-size tangent 补充已验证的 periodic-angle，直接检验检测几何平台。

## 2026-08-03 14:12 CST：178 0803_09 正式运行

- 隔离 checkout 固定 `35e18f1c`。真数据 4-step smoke loss
  `21.3700/20.6566/20.9046/21.1935`、grad
  `117.3254/104.8011/100.5948/101.3451`，全部有限；364,505,012-byte checkpoint
  完整，语义检查确认 iterative classification residual 与 DN absolute heads 已训练。
- fresh formal PGID `2971994`；真实 iter50 为 `0.9750 s/iter`、loss `21.0017`、grad
  `109.5454`。9 个进程成员存活，GPU0 驻留约 31.4 GiB，无 OOM/Traceback/NCCL/non-finite，
  provenance 与目标 workdir 正确，formal 五门槛通过。
- 状态为 `RUNNING`。该结构零参数、class-agnostic、无 reweight，只在 normal query 上组合
  reference-local log-size tangent 与 π-periodic angle tangent；下一完整判定点为 e4/e8/e12，
  不按早期节点单独否决。

## 2026-08-03 14:29 CST：252 0803_06 e8 完整评估

- e8 cls HOTA/DetA/AssA `40.922/32.859/53.835`，det
  `46.854/41.568/54.556`；e4→e8 双 HOTA `+10.224/+8.504`，但相对 periodic-angle
  单因素同点仍低 `4.665/6.061`，相对 Encoder 同点低 `4.347/3.339`。
- pair mAP/AP50 `0.1906/0.3551`，both-independent `0.2334/0.4148`；相对 periodic-angle
  同点分别低 `0.0517/0.0691` 和 `0.0583/0.0774`。375,533,238-byte checkpoint、
  128933 条检测、50 序列、28 CSV 与 108 个非空文件完整。
- PGID `3765372` 的 23 个成员已进入 e9。单因素与联合分支在 e8 都显著负于 periodic-angle，
  direct frame-evidence 路由的结构性负向结论一致；仍继续 e12 完成慢收敛审计。

## 2026-08-03 15:01 CST：252 0803_07 e12 成熟负向并释放

- e12 cls HOTA/DetA/AssA `43.504/34.301/58.469`，det
  `51.168/44.130/61.359`；e8→e12 双 HOTA `+2.124/+3.653`，但相对 periodic-angle
  单因素同点仍低 `4.409/4.089`，相对 Encoder 同点低 `6.176/5.373`。
- pair mAP/AP50 `0.2034/0.3683`，both-independent `0.2430/0.4213`；相对 periodic-angle
  同点分别低 `0.0543/0.0712` 和 `0.0593/0.0746`。381,031,670-byte checkpoint、
  143610 条检测、50 序列、28 CSV 与 108 个非空文件完整。
- e4/e8/e12 三点证明 direct frame-evidence 分类路由为成熟负向。精确终止 PGID `3694870`，
  23 个成员全部退出，GPU2/3 回到 `1 MiB/0%`；e12 断点保留。空闲双卡转给精确保留
  shared classification midpoint 的 `0803_08`。

## 2026-08-03 15:10 CST：252 0803_08 正式运行

- 新隔离 checkout `/data/users/litianhao01/PairMmot_framedetail_0803_08` 固定 `08356f9`；活动
  `0803_06` 仓库未更新。135 项回归与 2 个 subtest 通过；目标导入路径校验后，全模型为
  `22,771,111` 参数、增量 0、711 state tensors。
- 真数据 DDP smoke loss `12.9402/19.2481/19.2302/20.1285`，grad
  `103.2465/94.2024/82.7476/82.7848`，全部有限；364,473,462-byte checkpoint 与
  iterative-cls/DN 语义检查通过。
- fresh formal PGID `3940521`；真实 iter50 `1.2866 s/iter`、loss `21.4134`、grad
  `109.5162`，7 个成员存活，GPU2/3 各约 19.2 GiB，错误扫描、进程组、资源、provenance 与
  真实迭代五门槛通过。继续 e4/e8/e12，不按早期节点单独否决。

## 2026-08-03 15:25 CST：178 0803_09 e4 完整评估

- e4 cls HOTA/DetA/AssA `36.930/29.828/47.715`，det
  `44.486/36.346/56.708`；相对 periodic-angle 单因素同点双 HOTA `+0.906/+0.698`，
  合并 `+1.604`，双 DetA `+0.634/+1.476`。相对 Encoder e4 双 HOTA `+0.721/+5.733`。
- pair mAP/AP50 `0.1743/0.3157`，both-independent `0.2193/0.3845`；相对 periodic-angle
  同点分别提高 `0.0109/0.0123` 和 `0.0090/0.0071`。369,973,748-byte checkpoint、
  114290 条检测、50 序列、28 CSV 与 108 个非空文件完整。
- log-size tangent 在 HOTA、DetA 与四项 AP 上形成一致正向几何证据，但严格最终阈值仍差
  `17.507/17.907`。PGID `2971994` 的 9 个成员已进入 e5，继续 e8/e12，不以 e4 宣告达标。

## 2026-08-03 15:57 CST：252 0803_06 e12 延迟追赶

- e12 cls HOTA/DetA/AssA `45.752/36.629/59.854`，det
  `52.950/46.363/62.645`；e8→e12 双 HOTA `+4.830/+6.096`。相对 periodic-angle 同点
  缺口从 e8 的 `4.665/6.061` 缩到 `2.161/2.307`，相对 Encoder e12仍低 `3.928/3.591`。
- pair mAP/AP50 `0.2238/0.4040`，both-independent `0.2689/0.4633`；相对 e8 四项增长
  `0.0332/0.0489` 和 `0.0355/0.0485`。381,033,910-byte checkpoint、147478 条检测、
  50 序列、28 CSV 与 108 个非空文件完整。
- 单因素比联合分支 e12 双 HOTA 高 `2.248/1.782`，四项 AP 高
  `0.0204/0.0357/0.0259/0.0420`。PGID `3765372` 已进入 e13；由于成熟缺口显著收窄，
  继续 e16，不把 e12 当作直接否决点。

## 2026-08-03 16:22 CST：178 0803_10 零参数尺寸候选已准备

- 隔离 checkout `/data1/users/litianhao01/PairMOT_logarea_0803_10_repo` 固定 `8998142`；
  共享对数面积但保留逐帧纵横比变化，并叠加周期角度共识。该结构不增加参数、类别感知、重加权、
  attention 或 decoder 深度。
- 137 项 decoder 回归、脚本语法和目标模型完整构建均通过；父/新模型都是 `22,771,111` 参数、
  711 state tensors。当前为 `PREPARED/WAITING_GPU`，未修改或抢占运行中的 178 仓库/进程。

## 2026-08-03 16:41 CST：178 0803_09 e8 完整评估

- e8 cls/det HOTA `46.170/53.539`，相对 periodic-angle 同点 `+0.583/+0.624`，相对
  Encoder e8 `+0.901/+3.346`；DetA/AssA 为 `38.770/57.150` 与 `47.234/63.398`。
- pair mAP/AP50 `0.2470/0.4398`，both-independent `0.2926/0.4984`，四项相对
  periodic-angle 均保持正增益。375,564,404-byte checkpoint、153665 条检测、50 序列、
  28 CSV 与 108 个非空文件完整，异步评估正常完成。
- PGID `2971994` 已进入 e9，继续 e12；严格最终阈值尚差 `8.267/8.854`，目标未达成。

## 2026-08-03 16:53 CST：252 0803_08 e4 完整评估

- e4 cls/det HOTA `32.065/39.067`，相对 periodic-angle 同点 `-3.959/-4.721`，相对
  Encoder e4 `-4.144/+0.314`；pair mAP/AP50 `0.1332/0.2541`，both-independent
  `0.1721/0.3160`，四项 AP 相对 periodic-angle 均为负。
- 369,968,886-byte checkpoint、95613 条检测、50 序列、28 CSV 与 108 个非空文件完整。
  PGID `3940521` 保持运行并继续 e8/e12，不以 e4 直接否决；e8 监控已建立。

## 2026-08-03 17:00 CST：178 0803_11 已准备

- 隔离 checkout `/data1/users/litianhao01/PairMOT_lategeom_0803_11_repo` 固定 `4c58c57`；仅在
  decoder 最后两层施加 log-size 与周期角度共识，第一层保持逐帧独立。结构零参数、非类别感知、
  无重加权且不增加 attention/深度。
- 定向层数测试、138 项完整回归、目标配置深拷贝、launcher 语法和完整模型构建全部通过；
  父/新模型均为 `22,771,111` 参数、711 state tensors。状态 `PREPARED/WAITING_GPU`。

## 2026-08-03 17:05 CST：252 0803_10 双卡候选已落盘

- 隔离 checkout `/data/users/litianhao01/PairMmot_logarea_0803_10_252` 固定 `1f0147c`；252
  formal/smoke 配置、脚本语法和目标全模型构建通过，参数增量为零。
- 当前 GPU0/1 与 GPU2/3 分别由 `0803_06/08` 占用，未抢占、未启动。任一成熟负向实验释放双卡后，
  可直接按真数据 4-iter smoke、checkpoint 语义检查、formal iter50 五门槛部署。

## 2026-08-03 17:06 CST：197 暂时不可达

- `0803_10` 主增量已完整落盘，隔离 clone 已创建但仍为旧提交 `2024222`；clone 后 SSH 主动断开，
  三次有间隔复核依次为 connection refused、connection refused、`No route to host`。
- 未启动训练、未占 GPU、未修改 197 活动主仓库。保留可恢复现场并暂停重连；252/178 实验继续。

## 2026-08-03 17:24 CST：252 0803_06 e16 晚收敛继续

- e16 cls/det HOTA `47.584/55.930`，相对 e12 `+1.832/+2.980`，相对 periodic-angle e16
  `-0.890/+0.658`；DetA/AssA 为 `37.952/62.296` 与 `47.734/67.966`。
- pair mAP/AP50 `0.2363/0.4165`，both-independent `0.2795/0.4724`，四项相对 e12 继续
  增长。386,533,238-byte checkpoint、153698 条检测、50 序列、28 CSV 与 108 文件完整。
- PGID `3765372` 已进入 e17，继续 e20；追赶速度虽减慢但检测反超、分类缺口继续收窄，尚无成熟
  停止依据。严格最终阈值仍差 `6.853/6.463`。

## 2026-08-03 17:33 CST：252 GPU0/1 从 0803_06 接替到 0803_10

- 权威同点复核显示 `0803_06` e16 被原始 `0801_09` decoder 严格支配 `2.452/1.003`，且
  AP 同向落后；依据 e4/e8/e12/e16 完整轨迹精确停止 PGID `3765372`，23 成员退出，GPU0/1
  释放，e16 完整产物保留。
- `0803_10` 首次 smoke 在训练前因 GMC cache 路径大小写错误退出；保留失败日志后修正，retry
  四步 loss/grad 全有限，364,501,942-byte checkpoint 与语义检查通过。
- fresh formal PGID `4053545`；iter50 `1.2074 s/iter`、loss `21.3965`、grad `114.6571`，
  GPU0/1 各约 19.2 GiB，五门槛通过。状态 `RUNNING`，e4/e8/e12 均设置完整性监控口径。

## 2026-08-03 17:49 CST：197 SSH 恢复但 GPU 查询仍不可用

- SSH 可返回主机时间；隔离仓库仍为 HEAD `2024222`，两份 0803_10 bundle 完整保留。
- `nvidia-smi` 在显式 5 秒边界内无响应并返回超时码 `124`，故未 fetch、未构建、未占卡；等待
  GPU 子系统恢复后再执行受控续接，当前不影响 178/252 的三条正式实验。

## 2026-08-03 18:00 CST：178 0803_09 epoch 12

- 完整 TrackEval 为 cls/det HOTA `49.206/56.275`，DetA `41.546/49.874`，AssA
  `60.406/66.109`；相对原始 `0801_09` 同点联合提高 `3.650`，相对 periodic-angle 联合提高
  `2.311`。相对 Encoder 同点仍差 `0.474/0.266`，但不构成晚收敛否决依据。
- pair mAP/AP50 `0.2687/0.4703`、both-independent `0.3145/0.5247`；checkpoint 与
  50/28/108 产物完整，异步评估 401.5 秒正常完成。训练已进入 e13，继续长轨迹。

## 2026-08-03 18:35 CST：252 0803_08 epoch 8

- cls/det HOTA `40.688/47.811`，DetA `33.752/42.903`，AssA `51.391/55.164`；相对
  Encoder 同点低 `4.581/2.382`，相对 periodic-angle 低 `4.899/5.104`。
- pair mAP/AP50 `0.1995/0.3624`、both-independent `0.2402/0.4175`；checkpoint 与
  50/28/108 产物完整。e4→e8 正常增长但未恢复结构差距，仍按约束继续 e12 后再判定。

## 2026-08-03 18:45 CST：252 0803_11 静态候补

- 隔离 checkout `/data/users/litianhao01/PairMmot_lategeom_0803_11_252` 固定提交 `e09efb9`；
  双卡 formal/smoke 配置、路径和端口静态校验通过。
- 登录环境残留 `PYTHONPATH` 的首次错误导入已被隔离根检查捕获；固定到新 checkout 后，模型构建
  22,771,111 参数、零参数增量、711 state tensors，后两层几何投影单测通过。尚未占 GPU。

## 2026-08-03 19:06 CST：252 0803_10 epoch 4

- cls/det HOTA `32.399/39.251`，DetA `26.881/34.057`，AssA `41.726/46.221`；相对
  Encoder 同点 `-3.810/+0.498`，相对 periodic-angle `-3.625/-4.537`。
- pair mAP/AP50 `0.1351/0.2571`、both-independent `0.1799/0.3309`；checkpoint 与
  50/28/108 产物完整。继续 e8/e12，避免对 decoder 晚收敛作 e4 否决。

## 2026-08-03 19:22 CST：178 0803_09 epoch 16

- cls/det HOTA `50.732/57.218`，DetA `42.845/50.913`，AssA `62.101/66.744`；相对
  Encoder 同点 `-0.359/-1.102`，相对原始 decoder 同点 `+0.696/+0.285`，相对
  periodic-angle 同点 `+2.258/+1.946`。
- pair mAP/AP50 `0.2789/0.4860`、both-independent `0.3251/0.5393`；checkpoint 与
  50/28/108 产物完整。对强父线联合优势收窄至 `0.981`，但保持双侧领先，继续 e20/e24。

## 2026-08-03 19:36 CST：197 0803_11 接管 GPU4/5

- GPU 查询恢复后，隔离提交 `38ae0d4` 的 4-iter DDP smoke 数值、364,473,447-byte checkpoint、
  错误扫描与语义检查通过；GPU4/5 随后释放为 1 MiB。
- fresh formal PGID `53708`；iter50 `1.4287 s/iter`、loss `21.3917`、grad `120.7769`，
  7 个进程、双卡各 19,226 MiB，provenance 与 workdir 精确一致。状态 `RUNNING`，等待 e4。

## 2026-08-03 20:28 CST：252 GPU2/3 从 0803_08 接替到 0803_12

- 0803_08 e12 cls/det HOTA `44.177/52.763`，相对 Encoder 同点 `-5.503/-3.778`、相对
  periodic-angle `-3.736/-2.494`；AP 与三节点差距同向负面。完整 checkpoint 与 50/28/108
  产物确认后，精确停止 PGID `3940521`，23 个成员退出，GPU2/3 释放。
- 0803_12 采用首层自由、中层 log-area+周期角、末层 log-size+周期角的渐进零参数投影；单测、
  零参数全构建、DDP smoke 与 checkpoint 语义检查通过。
- fresh formal PGID `4189798`；iter50 `1.2941 s/iter`、loss `21.3858`、grad `113.3648`，
  7 个进程、GPU2/3 各 19,192 MiB，provenance/workdir/错误扫描通过。状态 `RUNNING`。

## 2026-08-03 20:32 CST：178 0803_09 epoch 20

- e20 cls/det HOTA `49.781/57.217`，相对 e16 `-0.951/-0.001`，相对原始 decoder 同点
  `-1.062/-0.816`；e12 的双侧优势到 e16 收窄并在 e20 反转。
- checkpoint、50 序列、28 CSV、108 个非空文件与 det/track 指标完整；保留训练至 e24 作
  连续成熟趋势确认，不按单个中期节点直接否决。

## 2026-08-03 20:35 CST：252 0803_10 epoch 8

- e8 cls/det HOTA `41.567/48.238`，相对 Encoder 同点 `-3.702/-1.955`、相对 periodic-angle
  `-4.020/-4.677`；checkpoint、50 序列、28 CSV、108 个非空文件完整。
- 继续到 e12 作成熟判定；252 GPU0/1 formal PGID `4053545` 保持运行。

## 2026-08-03 20:44 CST：178 后继 0803_13 已准备

- terminal-only 尺度/周期角共识只作用于最终输出，不改变前三层 reference 传播；零参数、无额外
  attention/layer、class-agnostic、无 reweight。
- 隔离提交 `1b7f904` 的目标单测及全模型构建通过：22,771,111 参数、零增量、711 state
  tensors。等待 0803_09 e24 成熟确认并释放 GPU0 后执行真数据 smoke。

## 2026-08-03 21:48 CST：178 从 0803_09 切换至 0803_13 smoke

- 0803_09 e24 cls/det HOTA `50.256/57.489`，相对原始 decoder 同点 `-1.453/-1.292`；
  checkpoint、50 序列、28 CSV、108 个非空产物完整。四个中后期节点确认后精确停止 PGID
  `2971994`，成员 `9→0`，GPU0 释放且 e24 checkpoint 保留。
- 隔离提交 `1b7f904` 的 0803_13 四步真数据 smoke 已启动，PGID `3061443`；等待完整
  loss/grad/checkpoint/错误扫描后再决定 formal。

## 2026-08-03 21:52 CST：178 0803_13 formal 运行

- smoke 四步 loss/grad 全有限，364,505,716-byte checkpoint、711 个 state tensor、错误扫描
  与语义检查通过。
- fresh formal PGID `3062903`；iter50 `0.9711 s/iter`、loss `20.9881`、grad `102.8326`，
  9 个进程，GPU0/提交/config/workdir/错误扫描五门槛通过。状态 `RUNNING`。

## 2026-08-03 22:06 CST：252 GPU0/1 从 0803_10 切换到 0803_14

- 0803_10 e12 cls/det HOTA `44.971/52.008`，相对 Encoder 同点 `-4.709/-4.533`；e12
  checkpoint 与 50/28/108 产物完整后精确停止 PGID `4053545`，成员 `23→0`，GPU0/1 释放。
- 0803_14 只在最终输出共享面积/周期角，保留逐帧纵横比；目标单测和零参数全模型构建通过，
  四步 DDP smoke PGID `75663` 已启动。

## 2026-08-03 22:11 CST：252 0803_14 formal 运行

- smoke 四步 loss/grad、364,502,454-byte checkpoint、711 state tensors 与错误扫描通过。
- fresh formal PGID `77558`；iter50 `1.2555 s/iter`、loss `21.3640`、grad `102.0121`，
  7 个进程、GPU0/1 各 19,192 MiB，provenance/config/workdir 一致。状态 `RUNNING`。

## 2026-08-03 22:13 CST：252 0803_12 epoch 4

- e4 cls/det HOTA `32.057/38.097`，相对 Encoder 同点 `-4.152/-0.656`；checkpoint、50
  序列、28 CSV、108 个非空文件完整。
- 保持 formal PGID `4189798` 继续 e8/e12，不按 e4 直接停止。

## 2026-08-03 22:29 CST：197 0803_11 epoch 4

- e4 cls/det HOTA `31.540/38.185`，相对 Encoder 同点 `-4.669/-0.568`；checkpoint、50
  序列、28 CSV、108 个非空文件完整，TrackEval 用时 711.3 秒。
- 保持 formal PGID `53708` 继续 e8/e12，不按 e4 直接停止。

## 2026-08-03 22:31 CST：178 后继 0803_15 已准备

- terminal-angle 只在最终输出共享周期角，其他 box 分量与所有 recurrent reference 逐帧独立；
  目标单测及零参数全模型构建通过。
- 隔离提交 `d181a98` 等待 0803_13 成熟判定，不额外占用 GPU。

## 2026-08-04 05:19 CST：197 0803_18 epoch 4

- e4 cls HOTA/DetA/AssA `30.440/24.784/40.480`，det
  `38.288/33.547/44.847`；相对原始 decoder 同点 `-3.866/-0.302`，相对 Encoder 同点
  `-5.769/-0.465`。
- pair mAP/AP50 `0.125500/0.238364`、both-independent
  `0.169556/0.314817`；checkpoint、5416 条检测、50 序列、28 CSV、108 文件及异步完成
  标志完整。PGID `387859` 继续 e8/e12，不按 e4 直接停止。

## 2026-08-04 05:41 CST：成熟轨迹迁入 252，178 启动 0803_23

- 0803_13 e24 cls/det `52.841/59.322`，相对原始 decoder 同点 `+1.132/+0.541`、联合
  `+1.673`；相对 Encoder 同点 `+1.127/-0.197`。完整 checkpoint、AP、50/28/108 产物
  核验后，178 PGID `3062903` 成员 `9→0`。
- 由于两机 UID 不同且共享文件系统不支持 ACL，252 不写旧 workdir；以旧 `epoch_24.pth`
  为只读源，在 252 自有目录续跑。e25 iter50 loss/grad `9.0711/53.3604`，DN/encoder 有限，
  PGID `419164` 只占固定 GPU0/1。
- 99 0803_17 e8 `39.478/46.483`，checkpoint 与完整 TrackEval 齐全，继续 e12。
- 178 当前 GPU0 完成 0803_23 四步 smoke 与 formal iter50；正式 `0.9593 s/iter`、loss/grad
  `21.0341/100.6837`，PGID `3144617` 五门槛通过。该 GPU 序号仅为当前选择，178 仍只受
  单卡总量约束。

## 2026-08-04 05:55 CST：178 0803_23 数值修复后 fresh 重启

- 首次 formal 在 e1 iter350 出现候选特有的非有限匹配代价保护；极小 reference 定向构造确认
  旧尺度 `exp` 会产生有限前向但 NaN 反向。旧 PGID `3144617` 于 iter650 精确停止，成员
  `9→0`，无 epoch checkpoint，不纳入性能结果。
- log-domain 先 clamp 再 exp 的等价修复提交在 178 为 `e2b399b2`；3 项定向测试、零参数整模
  构建、启动器语法和真实 smoke 通过。修复版 smoke 四步总 loss/grad 有限，642 个 checkpoint
  浮点 tensor 有限且无同类警告。
- `_finite_fresh` formal PGID `3151184` 在 iter50 为 `0.9841 s/iter`、loss/grad
  `21.0123/105.5777`，DN/encoder 有限、同类警告为 0，五门槛通过。当前只占 178 一张 GPU0；
  GPU0 仍只是当前选择而非固定授权。

## 2026-08-04 06:05 CST：178 0803_23 越过原故障点

- 修复版正式训练已到 epoch1 iter700，覆盖旧实现首次告警 iter350 的两倍区间；非有限匹配代价
  告警 `0`、致命错误 `0`。该点 loss/grad 为 `16.8861/229.6270`，总、DN、encoder proposal
  各分量均有限。
- 因而 log-domain 尺度解码修复已获得真实训练覆盖证据；保持 PGID `3151184` 继续 e4/e8/e12，
  不改动运行仓库，也不把尚未产生的性能评测提前记为成功。

## 2026-08-04 06:20 CST：178 保守后继 0803_24 已准备

- 0803_24 只对终层 normal queries 的 log-width/log-height/周期角切向量做传输：保留 pair-common
  更新，并把 frame detail 投影到前序 reference 已建立的相对尺寸/角度变换；中心 residual 原样逐帧
  保留。它针对 0803_13 e24 的 det 小幅落后，同时避免 0803_23 的中心传输与跨维投影。
- 该结构零参数、帧交换等变、class-agnostic，无 reweight、新层、attention、loss 或额外主矩阵乘法。
  4 项定向测试、配置深拷贝、两份启动器语法和整模构建通过；模型仍为 `22,771,111` 参数、
  增量 0、711 tensors。178 隔离仓库为
  `/data1/users/litianhao01/PairMOT_terminaltransportshape_0803_24`，clean HEAD `d470f96e`。
- 当前状态仅 `PREPARED/NO_GPU`，未创建 smoke/formal workdir；继续让 0803_23 独占 178 的一张卡，
  后续是否启动由现有完整 e12/e28 证据决定。

## 2026-08-04 06:53 CST：99 0803_17 e12 停止并切换 0803_21

- 0803_17 e12 cls HOTA/DetA/AssA `45.597/38.179/56.731`，det
  `52.020/47.161/59.329`；相对原始 decoder e12 `47.395/54.436` 为
  `-1.798/-2.416`，相对 Encoder e12 `49.680/56.541` 为 `-4.083/-4.521`。
  pair mAP/AP50 `0.238097/0.417747`，both-independent `0.282968/0.475183`。
- 381,044,662-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件和
  `async_done=1` 完整。e4/e8/e12 三节点持续双负后停止 PGID `1357909`；现有成员为 0，GPU1/2
  回落到 `10 MiB/0%`，GPU0 外部任务不变。这是成熟三节点判定，不是 e4/e8 早停。
- 0803_21 随后在 GPU1/2 完成四步 DDP smoke：loss
  `12.9371/19.5029/19.6174/21.1605`，grad
  `102.8686/100.6335/94.7345/94.9512`；DN/encoder 有限，364,502,646-byte checkpoint
  的 642 个浮点 tensor 全有限，iterative-cls/DN 语义通过。
- 两次空闲检查后 fresh formal screen `1384942.pm_0803_21_formal_99`、PGID `1384944` 启动；
  iter50 `0.9814 s/iter`、loss `21.3915`、grad `104.8633`，总、DN、encoder proposal
  全有限，GPU1/2 各约 19.2 GiB，无致命错误，五门槛通过。状态 `RUNNING`，继续 e4/e8/e12。

## 2026-08-04 09:43 CST：197 0803_18 e12 停止，0803_24 接替

- 0803_18 e12 cls/det HOTA `45.404/51.784`，相对原 decoder e12
  `-1.991/-2.652`、相对 Encoder `-4.276/-4.757`；pair mAP/AP50
  `0.2298/0.4115`、both-independent `0.2732/0.4675`。checkpoint、5416 条检测、50 序列、
  28 CSV、108 文件和异步完成证据完整。
- e4/e8/e12 成熟三节点双负后精确停止 PGID `387859`，成员 `23→0`。连续两次核验后动态选择
  空闲 GPU2/3；GPU0/1 外部 PID `8290/8291` 未动，符合 197 仅限制两卡总量、不固定序号的规则。
- 0803_24 双卡 smoke 四步与 642 个浮点 checkpoint tensor 全有限；fresh formal clean HEAD
  `44395ea`，screen `712275.pm_0803_24_formal_197`、PGID `712277`。iter50
  `1.8555 s/iter`、loss/grad `21.4056/136.6876`，GPU2/3 各约 19.2 GiB，五门槛通过，
  状态 `RUNNING/TO_E4+`。

## 2026-08-04 09:43 CST：178 0803_23 e12 继续强主线

- e12 cls/det HOTA `50.145/56.375`，相对原 decoder 同点 `+2.750/+1.939`、联合
  `+4.689`，相对 Encoder 同点 `+0.465/-0.166`。pair mAP/AP50 `0.2745/0.4824`、
  both-independent `0.3205/0.5354`；checkpoint、50/28/108 和异步完成证据完整。
- e12 只剩 det `0.166` 的 Encoder 同点缺口，PGID `3151184` 继续 e16，不为后继提前释放
  178 单卡；当前 GPU0 只是动态选择，不是固定序号授权。

## 2026-08-04 10:02 CST：252 0803_13 e36 继续成熟长轨迹

- e36 cls/det HOTA `53.874/60.860`，相对原 decoder e36 `+0.889/+0.450`、联合
  `+1.339`，相对 Encoder e36 `52.912/60.707` 为 `+0.962/+0.153`。pair mAP/AP50
  `0.3098/0.5315`、both-independent `0.3520/0.5725`；checkpoint、5416 条检测、50 序列、
  28 CSV、108 文件和异步完成证据完整。
- 绝对和 `114.734` 仍低于严格 `>118.330`，同点 Encoder 联合优势 `1.115` 也未过 `1.5`。
  固定 GPU0/1 的 PGID `419164` 继续 e40；GPU2/3 空闲，资源规则不变。

## 2026-08-04 10:11 CST：99 0803_26 product-tangent 已准备

- 新候选把 0803_23 的单一 5D tangent 投影分解为独立 center 2D 与 shape 3D 投影，保留完整
  几何传输但禁止中心/形状跨维内积干扰。它零参数、交换等变、class-agnostic，无 reweight、
  新层、attention 或 loss，DN prefix 保持不变。
- 远端定向单测、两配置加载/深拷贝、两启动器 Bash 语法和整模审计通过：父/候选均为
  `22,771,111` 参数、711 tensors，增量 0。99 隔离 checkout clean HEAD `89ec85a`。
- 首次 LFS smudge 失败工作树改名保留，随后 `GIT_LFS_SKIP_SMUDGE=1` 重建成功。状态
  `PREPARED/NO_GPU`，排在 `0803_25` 后，不改变当前 99 两卡占用或卡号规则。

## 2026-08-04 10:33 CST：99 0803_21 e12 成熟交接判据成立

- e12 cls/det HOTA `44.179/52.106`，对应 DetA/AssA 为
  `36.075/56.950` 与 `46.623/60.266`；相对原 decoder e12 `-3.216/-2.330`，相对 Encoder
  e12 `-5.501/-4.435`。pair mAP/AP50 `0.2260/0.3924`，both-independent
  `0.2668/0.4433`。
- epoch12 checkpoint、5416 条检测、50 序列、28 CSV、108 个非空文件及
  `async_done=1` 齐全。e4/e8/e12 连续双负，形成成熟停止依据，不属于 e4/e8 早停。
- 精确 TERM PGID `1384944` 后成员 `23→0`；checkpoint 与评测产物保留。GPU0 外部任务未动，
  GPU1/2 连续两次空闲后交给 0803_25。99 只限制总计两卡，不固定 GPU 序号。

## 2026-08-04 10:39 CST：99 0803_25 fresh formal 运行

- 动态 GPU1/2 上四步 DDP smoke loss `12.9442/19.6043/19.6451/21.2550`，grad
  `102.4960/169.0954/141.0412/133.9741`；DN/encoder、642 个 checkpoint 浮点 tensor、
  iterative-cls/DN 语义与错误扫描全部通过。
- 再次连续核验后 fresh formal screen `1442843.pm_0803_25_formal_99`、PGID `1442845`；
  iter50 `0.9843 s/iter`、loss/grad `21.4116/114.8470`，GPU1/2 各约 19.2 GiB，总、DN、
  encoder proposal 全有限，五门槛通过。状态 `RUNNING/TO_E4+`。
- GPU1/2 是本次实时选择，不是固定授权；GPU0 外部 PID `1439554` 未触碰。`0803_26` 保持
  PREPARED/NO_GPU，等待 0803_25 的成熟节点交接。

## 2026-08-04 10:58 CST：178 0803_23 e16 继续到 e20

- e16 cls/det HOTA `49.627/56.820`，DetA/AssA 分别为 `41.278/62.277` 与
  `50.118/66.772`；相对原 decoder e16 `-0.409/-0.113`、相对 Encoder
  `-1.464/-1.500`。pair mAP/AP50 `0.2716/0.4735`，both-independent
  `0.3148/0.5216`。
- 386,614,516-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空评测文件与
  `async_done=1` 完整。e16 较 e12 为 cls `-0.518`、det `+0.445`，HOTA 结构由强双正收敛为
  相对原 decoder 轻微双负。
- 因 e4/e8/e12 连续强双正、原 decoder 有已知后期回升，单个 e16 不足以成熟否决。当前 GPU0、
  PGID `3151184` 不停训，继续 e20；178 仍只用一张动态选择的卡。

## 2026-08-04 11:26 CST：252 0803_13 e40 保持成熟同点双正

- e40 cls/det HOTA `54.057/61.250`，DetA/AssA 分别为 `44.872/66.828` 与
  `53.718/72.186`；相对原 decoder e40 `-0.002/+0.148`，相对 Encoder e40
  `53.797/61.063` 为 `+0.260/+0.187`。pair mAP/AP50 `0.3138/0.5373`，
  both-independent `0.3553/0.5768`。
- 419,607,222-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空文件及
  `async_done=1` 齐全。e36→e40 绝对 HOTA 继续 `+0.183/+0.390`，因此不按 e40 停止。
- 绝对和 `115.307` 距严格 `>118.330` 仍差 `3.023`，两项绝对门槛也仍差 `0.380/1.143`。
  固定 GPU0/1 的 PGID `419164` 已进入 e41，继续到 e44；GPU2/3 保持空闲，252 资源规则不变。

## 2026-08-04 11:54 CST：99 0803_25 e4 继续慢收敛观察

- center-only e4 cls/det HOTA `31.262/37.687`，DetA/AssA 分别为
  `25.476/41.269` 与 `32.938/44.208`；相对原 decoder e4 `-3.044/-0.903`，相对
  full-tangent e4 `-5.080/-7.052`。pair mAP/AP50 `0.1330/0.2521`，both-independent
  `0.1742/0.3207`。
- 369,968,758-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空文件和
  `async_done=1` 完整。该早期负向不直接淘汰 decoder；动态 GPU1/2、PGID `1442845` 继续
  e8/e12，GPU0 外部任务保持不动。
- `0803_26` product-tangent 仍为 PREPARED/NO_GPU，不根据单个 e4 提前切换；99 只限制两卡
  总量，不固定序号的规则不变。

## 2026-08-04 12:11 CST：197 0803_24 e4 完整归因

- shape-only e4 cls/det HOTA `31.487/37.808`，DetA/AssA 分别为
  `25.126/42.524` 与 `32.338/45.300`；相对原 decoder e4 `-2.819/-0.782`，相对
  full-tangent e4 `-4.855/-6.931`。pair mAP/AP50 `0.1277/0.2485`，both-independent
  `0.1699/0.3198`。
- 369,969,127-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空文件和
  `async_done=1` 完整。shape-only 只比 center-only 同点高 `+0.225/+0.121`，两项分量早期
  都未复现 full-tangent 的联合增益。
- e4 不直接否决。动态 GPU2/3、PGID `712277` 继续 e8/e12；GPU0/1 外部任务保持不动，197
  仍仅限制两卡总量而不固定序号。

## 2026-08-04 12:13 CST：178 0803_23 e20 回升后继续 e24

- e20 cls/det HOTA `51.119/57.969`，DetA/AssA 分别为 `42.702/63.455` 与
  `51.405/67.704`；相对原 decoder e20 `+0.276/-0.064`，相对 Encoder e20
  `51.514/58.922` 为 `-0.395/-0.953`。e16→e20 为 `+1.492/+1.149`，确认 e16 不是可直接
  外推的稳定下行点。
- pair mAP/AP50 `0.2864/0.4981`、both-independent `0.3305/0.5452`；392,145,588-byte
  checkpoint、5416 条检测、50 序列、28 CSV、108 个非空文件与 `async_done=1` 完整。
- 当前动态 GPU0、PGID `3151184` 保持到 e24，观察同点差距是否继续收窄；178 仍只占一张卡，
  序号不固定，暂不为后继提前停训。

## 2026-08-04 12:55 CST：252 0803_13 e44 继续到 e48

- e44 cls/det HOTA `54.381/61.716`，DetA/AssA 分别为 `44.931/67.539` 与
  `54.021/72.880`；较 e40 双升 `+0.324/+0.466`，但相对原 decoder e44
  `54.415/61.737` 仍为 `-0.034/-0.021`。
- 绝对和 `116.097` 距严格 `>118.330` 尚差 `2.233`，且相对最终 cls/det 门槛仍差
  `0.056/0.677`，故不登记为目标完成。pair mAP/AP50 `0.3178/0.5406`、
  both-independent `0.3591/0.5797`。
- 425,094,518-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空文件和
  `async_done=1` 完整。PGID `419164` 在固定 GPU0/1 上继续 e48；GPU2/3 未被本任务使用。

## 2026-08-04 13:08 CST：99 0803_25 e8 完整后继续 e12

- center-only e8 cls/det HOTA `41.359/46.931`，DetA/AssA 分别为
  `34.708/51.115` 与 `42.204/54.111`；较 e4 回升 `+10.097/+9.244`，但相对原 decoder
  e8 为 `-0.613/-1.247`，相对 full-tangent e8 为 `-4.924/-6.824`。
- pair mAP/AP50 `0.2052/0.3754`、both-independent `0.2492/0.4369`；
  375,530,550-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空文件与
  `async_done=1` 完整。
- e8 不是直接否决点。动态 GPU1/2、PGID `1442845` 继续 e12，GPU0 外部任务保持不动；
  99 仍只限制两卡总量、不固定序号，`0803_26` 暂不抢占。

## 2026-08-04 13:30 CST：178 0803_23 e24 恢复持续并继续 e28

- e24 cls/det HOTA `52.012/58.551`，DetA/AssA 分别为 `43.460/64.440` 与
  `51.647/68.767`；较 e20 回升 `+0.893/+0.582`，相对原 decoder e24
  `+0.303/-0.230`，相对 Encoder e24 `+0.298/-0.968`。
- pair mAP/AP50 `0.2937/0.5083`、both-independent `0.3373/0.5536`；
  397,671,156-byte checkpoint、5416 条检测、50 序列、28 CSV、108 个非空文件与
  `async_done=1` 完整。
- e16→e20→e24 连续恢复。动态 GPU0、PGID `3151184` 已进入 e25 并继续 e28；GPU1 外部
  任务保持不动，178 仍只占一张卡且不固定序号。

## 2026-08-04 20:14 CST：252 e64 完整闭环后成熟释放

- 252 固定 GPU0/1 的 `0803_13` e64 cls/det HOTA 为 `54.930/61.999`，对应 DetA/AssA
  为 `45.198/68.617` 与 `54.369/73.115`。cls 超最终 Encoder `+0.493`，det 仍低 `0.394`；
  绝对和 `116.929` 距严格 `>118.330` 尚差 `1.401`，未达同 checkpoint 双过线目标。
- pair mAP/AP50 `0.3224/0.5369`、both-independent `0.3612/0.5717`，checkpoint、5416 条检测、
  50 序列、28 CSV、108 个非空文件和异步完成证据齐全。e64 还略低于本线 e56 最佳
  `54.980/62.009`，在 e56/e60/e64 成熟平台确认后精确 TERM PGID `419164`，成员 `23→0`。
- 252 GPU0/1 已回到 `1 MiB`，GPU2/3 仍未被本任务使用；全部结果保留。99 `0803_29`
  在 e4 尾段，197 `0803_28` 在 e7，178 `0803_23` 在 e46，进程、正式日志与各自动态资源
  均健康；下一批闭环依次为 99 e4、197 e8、178 e48。

## 2026-08-04 20:28 CST：252 接续 0803_30，99 收齐 0803_29 e4

- `0803_30` 的 252 2x4 隔离 checkout 为 clean HEAD `c3fc5a1`。定向测试、配置深拷贝、整模
  构建与启动器语法通过，参数 `22,771,111`、增量 0、711 states。首次 smoke 只暴露 197→252
  物理数据路径遗漏，失败目录保留归档；修复物理路径后 fresh 四步 smoke、DN/Encoder、
  364,502,774-byte checkpoint 的 642 个浮点 tensor 与错误扫描全部通过。
- 252 固定 GPU0/1 的 fresh formal screen `798987.pm_0803_30_formal_252`、PGID `798989`；
  iter50 `1.1484 s/iter`、loss/grad `21.3741/109.9191`，7 个成员、GPU0/1 各约 19.2 GiB，
  五门槛通过，登记 `RUNNING/TO_E4+`；GPU2/3 保持空闲。
- 99 `0803_29` e4 cls/det HOTA `30.658/38.402`，DetA/AssA 为
  `25.232/39.985` 与 `32.550/46.208`；相对原 decoder e4 `-3.648/-0.188`，相对 0803_27 e4
  `+0.235/+0.273`。pair mAP/AP50 `0.1306/0.2495`，checkpoint、5416/50、28 CSV、108 文件
  完整；动态 GPU0/1 的 PGID `1582836` 继续 e8/e12，不以 e4 早停。

## 2026-08-04 21:14 CST：178 e48 与 197 e8 完整评估

- 178 `0803_23` e48 cls/det HOTA `53.944/60.888`，DetA/AssA 分别为
  `44.633/67.242` 与 `53.443/71.770`；较 e44 双升 `+0.272/+0.335`，但相对原 decoder e48
  仍低 `0.665/1.203`。绝对和 `114.832` 距严格目标差 `3.498`。pair mAP/AP50
  `0.3054/0.5221`、both-independent `0.3461/0.5618`，checkpoint、5416/50/28/108 和异步
  完成证据齐全。自身曲线未平台，PGID `3151184` 继续 e52。
- 197 `0803_28` e8 cls/det HOTA `38.401/44.520`，DetA/AssA 为
  `32.595/48.019` 与 `41.869/48.825`；相对原 decoder e8 `-3.571/-3.658`，相对 position/product
  e8 `-3.488/-3.645`。pair mAP/AP50 `0.1898/0.3386`、both-independent `0.2370/0.4073`，
  checkpoint、5416/50/28/108 和异步完成证据完整；PGID `1016336` 继续 e12，不以 e8 早停。

## 2026-08-08 17:25 CST：e12 成熟审计与 99 资源交接

- 197 `0808_02` e12 完整 `47.147/54.337`，DetA/AssA 为 cls
  `37.813/62.011`、det `48.809/62.587`；pair `0.241215/0.428460`、both-independent
  `0.282467/0.478276`。det 对父线缺口较 e8 收窄，main PGID `2811864` 继续动态 GPU0/1，
  当前 e16 iter150，formal 数值有限、fatal=0。
- 178 `0808_03` e12 完整 `47.998/55.163`，DetA/AssA 为 cls
  `40.190/59.892`、det `49.761/63.220`；pair `0.264565/0.457265`、both-independent
  `0.308771/0.508690`，为当前最强成熟点。main PGID `1346509` 继续动态 GPU0，当前
  e14 iter600；GPU1 未使用。
- 99 `0808_01` e12 完整 `45.078/53.335`，DetA/AssA 为 cls
  `37.002/57.264`、det `47.145/62.637`；pair `0.231580/0.408462`、both-independent
  `0.273380/0.462161`。e4/e8/e12 三节点均明显弱于 197/178 后精确 TERM PGID
  `2578723`，成员 `23→0`、screen 消失、GPU0/1 归零；产物保留，状态
  `STOPPED/E12/MATURE_STRICT_FAIL`，不是 e4/e8 早停。
- 252 `0808_04` e12 完整 `47.539/54.477`，DetA/AssA 为 cls
  `40.233/58.344`、det `49.370/62.125`；pair `0.255269/0.445445`、both-independent
  `0.301642/0.501309`。381,031,798-byte checkpoint SHA-256 `5b9f67a6…e4a0`，642 个
  浮点 tensor 全有限且 iterative-cls/DN 已训练；5416/50、28 CSV、108 非空文件、50
  predictions 与 `async_done=1` 完整。e8→e12 总和恢复 `9.388` 为四线最大，main
  `1579745` 继续固定 GPU0/1 到 e16/e72，GPU2/3 为 1 MiB。

## 2026-08-08 17:43 CST：99 启动 0808_06 delayed LR clock

- 释放后的动态 GPU0/1 完成 `0808_06` 四步真实 DDP smoke，四个 loss/grad、DN、Encoder
  proposal 全有限；364,512,886-byte checkpoint 的 642 个浮点张量、iterative-cls/DN
  语义与错误扫描通过，GPU 随后归零。
- 三次 formal 启动在训练前失败，均只有 144-byte `launch.log`、无 checkpoint 或 GPU 占用；
  根因最终定位为 test GMC 路径大小写错误 `PairMmot`，正确目录为 `PairMOT` 且含 5416 文件。
  三个失败目录均保留归档。`169f5f2` 增加 30 次共享目录重试，`f701e73` 修正路径；远端隔离
  checkout clean、launcher 语法与两个 GMC 缓存重新通过，未热更新任何存活训练仓库。
- fresh workdir
  `/data4/litianhao/PairMmot/workdir_99/0808_06_final_product_tangent_delayedlrclock_72e_2xb4_fresh`
  的 screen/main 为 `2606264/2606266`；e1 iter50 LR/loss/grad
  `2.5488e-6/21.4017/134.7102`，双 rank、GPU0/1 各约 19.2 GiB、正式日志与有限
  total/DN/Encoder、fatal=0 五门槛通过。登记 `RUNNING/TO_E72`；GPU2 保持 10 MiB。

## 2026-08-08 17:55 CST：0808_07 staged delayed LR 静态就绪

- 新后备只把 e12 的 `1.4×` 单次跳变分解为 e12/e24 两次
  `g=1.2037682266×`；实际训练 LR 区间为 12 轮 `1.0×`、12 轮 `g×`、48 轮
  `g²=1.4490579434×`，解析与实测积分均为 96 个父线 epoch。最终模型、参数、数据、loss、
  EMA、Liquid 与推理计算均不变，非 class-aware、无 reweight。
- 提交 `adca76a` 已部署到 197 新隔离 clean checkout。formal/smoke config deepcopy、两个
  launcher 语法、72轮 LR 序列、父/候选完整构建通过：均为 `22,771,111` 参数、711 states。
  两个目标 workdir 不存在、无 `0808_07` 训练进程、未占 GPU，状态仅
  `STATIC_VALIDATED/NO_SMOKE/NO_FORMAL`。
- 当前 `0808_02` main `2811864` 仍独占动态 GPU0/1，17:55 到 e16 iter1000，total、DN、
  Encoder proposal 与 grad 全有限。必须先完成 e16 checkpoint/检测/TrackEval，再据成熟轨迹
  决定保留原线或释放双卡给 `0808_07` 的真实 smoke；不热更新存活仓库。

## 2026-08-08 21:11 CST：178 `0808_03` e24 全量闭环并继续

- e24 cls HOTA/DetA/AssA 为 `51.215/42.178/64.593`，det 为
  `58.367/50.939/69.202`，sum `109.582`；相对自身 e20 双升 `0.702/0.436`，相对直接
  product-tangent 父线 e24 `52.478/58.771` 的差距为 `1.263/0.404`，联合差距从
  e20 的 `1.886` 收窄到 `1.667`，继续保持当前最强成熟趋势。
- pair mAP/AP50 `0.279480/0.475666`、both-independent
  `0.320369/0.520240`，四项均较 e20 提升。397,671,732-byte checkpoint SHA-256
  `525b3dd347dfc7da31e7175c76ed86d93069d9bfa16f0702c8fbf5f3b04f8725`，meta
  `24/24912`；model/EMA 各 642 个浮点张量全有限，iterative-cls/DN 审计通过。
- 5416/50 检测、28 CSV、108 非空 TrackEval 文件、50 非空 predictions 完整；TrackEval
  255.9 秒自然结束。main `1346509` 已恢复到 e25 iter500，只使用动态 GPU0；GPU1 有外部
  占用且从未被本目标使用。99/197 分别进入 e12 iter100/50，252 到 e23 iter300；全部资源边界合规。

## 2026-08-08 22:03 CST：99/197 e12 与 252 e24 全量闭环

- 197 `0808_07` e12 cls HOTA/DetA/AssA `47.190/38.691/59.991`，det
  `53.110/47.655/61.287`；pair mAP/AP50 `0.2433/0.4239`，both-independent
  `0.2885/0.4807`。99 `0808_06` e12 为 cls `45.402/37.226/58.322`、det
  `51.604/46.850/58.907`；pair `0.2314/0.4109`、both-independent
  `0.2738/0.4660`。197 同点全指标领先，但两条线的 LR 改变都发生在 e12 之后，因此继续
  e16，不以切换前 checkpoint 否决。
- 99/197 checkpoint 大小为 381,051,446/381,036,583 bytes，SHA-256 为
  `ce1fea065968be18e0f4c743e424a12e4cb0a8ee33f24b18c2d9846e4c15548c` /
  `f7b6251075201140fc6cddf1c85486cbbbc70c52c28084cae2383b93703c6304`；meta 均
  `12/12456`，model/EMA 各 642 个浮点张量全有限，iterative-cls/DN 已训练。两边均完成
  5416/50、28 CSV、108 非空文件与 50 predictions；TrackEval 286.2/341.4 秒自然结束。
  当前 99 e14 iter500、197 e14 iter800，LR 分别为 `1.4e-4/1.2038e-4`。
- 252 `0808_04` e24 cls `49.951/42.031/61.401`，det
  `58.453/51.154/69.169`，sum `108.404`；pair `0.2771/0.4761`、both-independent
  `0.3207/0.5251`。相对父线 e24 低 `2.527/0.318`，相对 178 同点 det 高 `0.086`，但
  cls 低 `1.264`、总和低 `1.178`；主要差异是 cls AssA 低 `3.192`，故只保留成熟对照。
- 252 e24 checkpoint 397,531,510 bytes，SHA-256
  `92b69f5ea0ff702378804d4b007877aadbd9bd0cbed6a98e2f0e254f0935abbf`，meta
  `24/24912`；model/EMA 全有限并完成 5416/50、28/108/50 闭环，TrackEval 378.4 秒。
  main 恢复 e25 iter350，固定 GPU0/1；GPU2/3 未使用。178 到 e28 iter450，只用 GPU0。

## 2026-08-08 22:29 CST：178 `0808_03` e28 首次双超直接父线

- e28 cls HOTA/DetA/AssA `52.850/43.936/65.433`，det
  `59.702/51.790/71.166`，sum `112.552`；相对自身 e24 双升 `1.635/1.335`，DetA 与
  AssA 四项全部提升。相对直接 product-tangent 父线 e28 `52.641/58.986` 双超
  `0.209/0.716`，总和领先 `0.925`；虽然 DetA 略低，cls/det AssA 分别高
  `1.286/2.560`，局部学习率加速已形成明确关联优势。
- pair mAP/AP50 `0.2992/0.5105`，both-independent `0.3406/0.5543`，均较 e24 上升。
  403,198,580-byte checkpoint SHA-256
  `614243d4a87efc01983fc7dcf9e22742f081aa451cff095ec4ad94a60d04447c`，meta
  `28/29064`；model/EMA 各 642 个浮点张量全有限，iterative-cls/DN 已训练。
- 5416/50、28 CSV、108 非空文件、50 predictions 完整；TrackEval 252.9 秒自然结束。
  main `1346509` 已恢复 e29 iter350，只用 GPU0。同期 197 e16 iter500、99 e15 iter950、
  252 e26 iter750；资源边界无违规。

## 2026-08-08 22:59 CST：99/197 延迟 LR e16 全量闭环

- 99 `0808_06` e16 cls HOTA/DetA/AssA `48.335/40.549/60.188`，det
  `55.763/49.857/64.564`，sum `104.098`；pair mAP/AP50 `0.2560/0.4553`，
  both-independent `0.2994/0.5084`。相对 e12 双升 `2.933/4.159`，一次 `1.4×` 跳变
  后恢复明确。
- 197 `0808_07` e16 cls `48.551/40.821/59.843`，det
  `55.024/49.892/62.710`，sum `103.575`；pair `0.2574/0.4558`，both-independent
  `0.3023/0.5088`。相对 99 cls 高 `0.216`，det 低 `0.739`、总和低 `0.523`；AP 接近，
  主要差异来自 99 det AssA 高 `1.854`。
- 99/197 checkpoint 大小为 386,561,078/386,529,319 bytes，SHA-256 为
  `8b6e9deec38bc3da98de3219fce9c1c6bf5d7f8ad486d288b433b19640490c64` /
  `8e54260575338aef8108e1a17ad6739cd8f44d88c6a5a3c26bbf8d89c001423e`，meta 均
  `16/16608`；model/EMA 全有限并完成 5416/50、28/108/50 闭环。TrackEval 281.5/337.3
  秒自然结束。99 恢复 e17 iter300，197 到 e18 iter50；两条继续 e20，不作首次响应早停。

## 2026-08-08 23:41 CST：252 e28 成熟换线并启动 `0808_08`

- `0808_04` e28 cls HOTA/DetA/AssA `50.677/42.170/62.973`，det
  `59.213/51.537/70.457`，sum `109.890`；pair mAP/AP50 `0.2821/0.4844`，
  both-independent `0.3242/0.5298`。相对 178 同点 cls/det 低 `2.173/0.489`，DetA、AssA
  和 AP 全部更低，故七个完整节点后成熟释放，不是 e4/e8 早停。
- 403,030,198-byte e28 checkpoint SHA-256
  `f07ef31c7e1541df223933bbe7697d46ff20868db80ceb9e7f947cb292dcbd52`，meta
  `28/29064`；model/EMA 全有限并完成 5416/50、28/108/50 闭环，TrackEval 389.8 秒。
  PGID `1579745` 于 e29 iter550 精确 TERM，成员 `7→0`；GPU0/1 连续两次 `1 MiB/0%`，
  GPU2/3 未使用，全部正式产物保留，状态 `STOPPED/E29I550/E28_COMPLETE/MATURE_DOMINATED`。
- `0808_08` 相对 178 强线只新增 decoder/head 参数组 Adam betas 的 `96→72` 时钟压缩，
  全局 backbone/encoder LR、betas、warmup、EMA、Liquid、模型与 loss 不变。isolated clean
  HEAD `b41a936` 通过配置 deepcopy、launcher 语法、父/子整模构建和优化器参数组审计；均为
  22,771,111 参数、711 states、增量 0。
- 固定 GPU0/1 DDP smoke 四步 loss `12.9389/19.4649/19.5867/21.1698`，grad/DN/encoder
  proposal 全有限；iter4 checkpoint 364,506,742 bytes，SHA-256
  `9acaf973dc0e7fcc8c4724c2a8a1fbd82a1e7a02455d9afa77a085a6061b1ede`，642 个浮点张量
  全有限且 iterative-cls/DN 已训练。fresh formal screen/main `1642666/1642667` 达 e1
  iter50，LR/loss/grad `2.5488e-6/21.4287/120.2927`，双卡约 19.2 GiB 满载、正式日志更新、
  fatal=0，五项门槛通过后登记 `RUNNING/TO_E72`。现场 99 e19i650、197 e20i850、178
  e33i100，全部在合法卡数内。

## 2026-08-08 23:45 CST：178 `0808_03` e32 完整闭环

- e32 cls HOTA/DetA/AssA `51.872/43.247/64.217`，det
  `59.521/51.622/70.992`，sum `111.393`；相对自身 e28 为 `-0.978/-0.181`，四个
  DetA/AssA 分量及 pair/both AP 均回撤。pair mAP/AP50 `0.2943/0.4987`，
  both-independent `0.3343/0.5405`。
- 相对直接父线 e32 `53.309/59.320`，cls 低 `1.437`、det 高 `0.201`；det AssA 高
  `2.011`、DetA 低 `1.200`，局部加速仍保留关联优势但定位/AP 未同步。单个 e32 回撤不覆盖
  e28 的同点双超成熟证据，故继续 e36，而不是以一节点停止。
- checkpoint 408,728,116 bytes，SHA-256
  `38aaa4f56fce98a9300c6c8926d6c7983ce1495274b92be671ae8e3db84fdd3f`，meta
  `32/33216`；model/EMA 711/712 keys、642 个浮点张量全有限，iterative-cls/DN 已训练。
  5416/50、28 CSV、108 非空文件、50 predictions 闭环，TrackEval 268.7 秒。main
  `1346509` 已恢复 e33 iter350，只用 GPU0。

## 2026-08-08 23:56 CST：197 `0808_07` e20 全量闭环

- e20 cls HOTA/DetA/AssA `50.650/42.356/62.730`，det
  `56.444/50.569/65.114`，sum `107.094`；相对 e16 双升 `2.099/1.420`，四个 DetA/AssA
  分量全部提升。相对直接父线 e20 仍低 `1.548/1.688`，相对 178 同点 cls 高 `0.137`、det
  低 `1.487`，当前不是主候选。
- pair mAP/AP50 `0.2692/0.4706`、both-independent `0.3142/0.5221`，均较 e16 提升。
  checkpoint 392,028,519 bytes，SHA-256
  `bd3397e5e0b4f4ca1d9edbc6e6a3ad6d9efd7188e92b075aa690e161a7c2f1e2`，meta
  `20/20760`；model/EMA 711/712 keys、642 个浮点张量全有限，iterative-cls/DN 已训练。
- 5416/50、28 CSV、108 非空文件、50 predictions 完整；TrackEval 330.0 秒自然结束。
  main 已恢复 e21 iter350、动态 GPU0/1、fatal=0。由于第二阶段 LR 在 e24 后才生效，继续
  e24 基准及 e28 响应，不在 e20 提前否决完整策略。

## 2026-08-09 00:17 CST：99 `0808_06` e20 全量闭环

- e20 cls HOTA/DetA/AssA `50.048/41.914/62.323`，det
  `57.885/51.156/67.890`，sum `107.933`；相对 e16 双升 `1.713/2.122`，四个 DetA/AssA
  分量全部提高。相对 197 e20 cls 低 `0.602`、det 高 `1.441`、总和高 `0.839`，det AssA
  高 `2.776` 是主要优势。
- pair mAP/AP50 `0.2677/0.4738`、both-independent `0.3112/0.5241`，与 197 同点基本
  同档；相对 178 e20 仅低 `0.465/0.046`，故继续 e24+，不成熟停线。197 仍需跨过第二阶段
  LR 边界，也继续 e24/e28。
- checkpoint 392,071,606 bytes，SHA-256
  `8bdc0a5b92bd5695be3c18c9ea8d60a47db75690418b70bf6b482f54df3d8103`，meta
  `20/20760`；model/EMA 711/712 keys、642 个浮点张量全有限，iterative-cls/DN 已训练。
  5416/50、28 CSV、108 非空文件、50 predictions 完整，TrackEval 288.4 秒。main 已恢复
  e21 iter300，动态 GPU0/1、fatal=0。
## 2026-08-09 00:40 CST：178 下一局部延迟 LR 候选静态就绪

- `0809_01` 在 e1–e12 保持最终 product-tangent 父线优化轨迹，e12 后只把 decoder/head
  参数组 LR 提升为 `1.4×`；这使局部名义 LR 积分在 e72 等于父线 e96，同时不改变模型、
  state、loss、数据或推理开销。策略为 class-agnostic、无 reweight。
- isolated checkout 固定 clean detached `eb2c70a`。tagged scheduler 单测 `2/2 OK`，两配置
  deepcopy、两 launcher `bash -n`、父/候选完整构建通过；二者均为 22,771,111 参数、711
  states。实际 498 个 optimizer groups 中只有 178 个 decoder/head 组在模拟 e12 后由
  `1e-4` 变为 `1.4e-4`，其余组保持各自原 LR。
- 状态严格为 `STATIC_VALIDATED/NO_GPU`：smoke/formal workdir 均未创建，GPU 未占用；当前
  `0808_03` 继续独占动态 GPU0 到 e36，GPU1 空闲。只有成熟证据支持精确交接后，才依次执行
  真实 smoke、checkpoint 有限性和 formal iter50 五门槛。
