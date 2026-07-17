# PairMOT Paper Mainline Experiment Report

更新时间：2026-07-17 CST

## 1. 实验目标

本报告记录论文主线的正式可复现实验。目标是在完全相同的数据、初始化、训练精度、随机种子和评测协议下，逐步验证 Liquid、Encoder、Decoder 的增益，并将 Proposal affinity 作为无需重新训练的 zero-shot 推理模块单独验证。

主线顺序：

1. `Base`；
2. `Base + Liquid`；
3. `Base + Liquid + Encoder`；
4. 第 3 步 checkpoint 加入 zero-shot Proposal；
5. `Base + Liquid + Encoder + Decoder`，fresh train；
6. 第 5 步 checkpoint 加入 zero-shot Proposal，形成完整模型。

第 4 步不进入第 5 步训练。否则 Proposal 将不再是 zero-shot 模块，无法保持论文中的方法定义。

## 2. 固定实验协议

| 项目 | 正式设置 |
| --- | --- |
| 消融骨干 | RT-DETR R18vd，3-layer decoder |
| 初始化 | 官方 COCO-only R18 checkpoint，经统一 PairMOT adapter 适配 |
| 数据 | HSMOT full train 75 sequences / 8297 unique adjacent pairs；test 50 sequences / 5416 pairs |
| 输入 | 原始 `1200x900`，`keep_ratio=True`，不做非等比拉伸 |
| Pair | train/test 均枚举唯一时间正序 `t-1 -> t`、gap=1；两帧共享 resize/flip/rotate |
| GMC | 真实 sparse-LK + RANSAC cache，train 8297 个唯一相邻变换；缺失即报错，禁止 identity fallback |
| Batch | 2 GPUs x 4 images/pairs，global batch 8 |
| 训练 | fresh 72 epochs，AdamW，base LR `1e-4`，LinearLR warmup 2000 iter |
| 精度 | BF16 through shared encoder，encoder 输出后转 FP32；decoder/head/matcher/loss 为 FP32 |
| DDP | `find_unused_parameters=False` |
| 随机性 | global seed 3407，pair sampler seed 3407，`deterministic=False` |
| 验证 | 每 4 epochs，18 个完整 validation + TrackEval 点 |
| 可视化 | 仅关闭绘图；detection evaluation 和 TrackEval 全部保留 |
| 停止规则 | 禁止 early stopping；所有训练实验完整运行 72 epochs |

所有需要训练的模块实验必须从同一个 COCO-only adapted checkpoint fresh train，禁止从前一阶段 checkpoint resume。除被消融模块及其新参数学习率外，不允许改变训练协议。

## 3. 模型定义

`Base` 固定使用 `0704_01` 的最终 PairMOT 结构：track-union all-GT、dual classification、no presence head、unique pair top-k、PairDN，以及 pair proposal supervision。Baseline 不包含 Liquid、temporal/pyramid encoder、elliptical-spectral Proposal、tristate decoder 或 long-tail 专用模块。

COCO 80类分类头不加载；HSMOT 8类 prev/curr 分类头重新初始化。RGB stem 权重原样重排至 Conv3D spectral kernel。单帧 cross-attention 同时初始化 prev/curr 分支；水平框前四维复制到旋转框 head，angle 维保持目标初始化。Pair-only 层使用模型定义的确定性初始化。

## 4. 实验矩阵

| 阶段 | 训练 | 唯一模型变化 | 作用 |
| --- | --- | --- | --- |
| A Base | 是 | 无 | 论文统一基准 |
| B +Liquid | 是 | 最终 Liquid sampler/fusion | 验证谱段动态建模 |
| C +Encoder | 是 | 最终 temporal/pyramid encoder | 验证跨帧多尺度表征 |
| C-P +Proposal | 否 | size-aware elliptical motion + spectral affinity | 验证 zero-shot proposal affinity |
| D +Decoder | 是 | 最终 pair decoder | 验证解码阶段交互 |
| D-P Full | 否 | 在 D checkpoint 上启用 Proposal | 完整 R18 方法 |

为增强因果归因，主线完成后至少补充 `Base + Encoder` 和 `Base + Decoder` 两项独立消融。Long-tail 作为解决 cls HOTA 的正交分支，先报告 `Base + Long-tail`，验证稳定后再加入完整模型，不混入主链基础结论。

## 5. 指标与选择规则

每个训练实验只按 `cls_HOTA + det_HOTA` 在全部 18 个评测点中选取唯一最佳 epoch。表格必须分别展示 `cls_HOTA` 与 `det_HOTA`，禁止合并成一个指标展示或跨 epoch 拼接。

同时报告该 epoch 的 cls/det MOTA、IDF1、pair mAP 和 pair AP50。AP 不参与
checkpoint 选择，也不允许另选 AP 最优 epoch；所有论文指标必须来自
`cls_HOTA + det_HOTA` 所确定的同一个唯一 epoch。最终论文还需记录参数量、FLOPs、
训练显存、训练吞吐和推理速度。

## 6. 当前进展

| ID | 实验 | 服务器/GPU | 配置 | 状态 |
| --- | --- | --- | --- | --- |
| `0716_02` | Paper Base R18 COCO full 1200x900 BF16 | 99 / GPU 0,1 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_coco_full_1200x900_bf16_reboot_fresh_99.py` | 已完成72 epochs和18/18异步TrackEval；唯一最佳为epoch 68 |
| `0716_04` | Paper Base + Liquid group-set-unique R18 COCO full 1200x900 BF16 | 197 / GPU 0,3 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_plus_liquid_groupsetunique_coco_full_1200x900_bf16_197.py` | 正常训练；已验收到epoch 1 iter 150 |
| `0716_05` | Paper Base + Liquid group-set-unique + Encoder R18 COCO full 1200x900 BF16 | 252 / GPU 0,1 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_plus_liquid_groupsetunique_encoder_coco_full_1200x900_bf16_252.py` | 30项单测和100 iter DDP测试通过；正式训练已验收到epoch 1 iter 50 |
| `0717_01` | Parallel Liquid Set-Transport structural candidate | 99 / GPU 2,3 | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_paper_base_plus_liquid_settransport_coco_full_1200x900_bf16_99.py` | 23项单测和100 iter DDP测试通过；正式运行因GPU 2/3掉卡风险在epoch 2 iter 250主动取消，不作为结果 |

工作目录：

`/data4/litianhao/PairMmot/workdir_99/0716_02_paper_base_r18_coco_full_1200x900_bf16_orderedpairs_restart`

当前正式运行：

`/data4/litianhao/PairMmot/workdir_99/0716_02_paper_base_r18_coco_full_1200x900_bf16_orderedpairs_reboot_fresh`

`/data4/litianhao/PairMmot/workdir_99/0716_03_paper_base_plus_liquid_r18_coco_full_1200x900_bf16_orderedpairs`

当前正式 Base + Liquid 运行：

`/data4/litianhao/PairMmot/workdir_197/0716_04_paper_base_plus_liquid_groupsetunique_r18_coco_full_1200x900_bf16_orderedpairs_fresh`

当前正式 Base + Liquid + Encoder 运行：

`/data4/litianhao/PairMmot/workdir_252/0716_05_paper_base_plus_liquid_groupsetunique_encoder_r18_coco_full_1200x900_bf16_orderedpairs_fresh`

## 7. 结果表

Tracking 表在实验完成后填写：

| 实验 | unique epoch | cls HOTA | det HOTA | cls MOTA | cls IDF1 | det MOTA | det IDF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A Base | 68 | 53.314 | 61.982 | 44.690 | 62.218 | 60.599 | 72.303 |
| B Base + Liquid | - | - | - | - | - | - | - |
| C Base + Liquid + Encoder | - | - | - | - | - | - | - |

AP 诊断结果，epoch 必须与 Tracking 表一致：

| 实验 | AP epoch | pair mAP | pair AP50 |
| --- | ---: | ---: | ---: |
| A Base | 68 | 0.3149 | 0.5225 |
| B Base + Liquid | - | - | - |
| C Base + Liquid + Encoder | - | - | - |

## 8. 可复现性记录

- 代码基准 commit：`3c8af7419d9a79b6251268936805c9666bd2ab99`；正式启动时工作树含未提交研究改动，MMEngine 会在 workdir 保存完整 resolved config，启动日志和代码 diff hash需一并保留。
- COCO R18 source SHA256：`3ba8b5c909c9a1c4f21e96d0a7251ab1a485093955ca327d0061fef8d33c66f0`。
- 权重适配 manifest：`/data4/litianhao/PairMmot/pretrained_weights/rtdetr_coco_pair_family_manifest.json`。
- 正式实验不得 resume，不得使用旧 COCO+Objects365 adapted checkpoint，不得回退到历史有效高度 800 的输入协议。
- 已取消 strict run log：`/data4/litianhao/PairMmot/workdir_99/0716_02_paper_base_r18_coco_full_1200x900_bf16/20260716_154409/20260716_154409.log`。
- 已取消 strict run 的 tracked diff patch SHA256：`3e28bf0a6eef22ad1dec5a72a363ae4d2b7a3d5d1ef6d7ea3d843c2f92e2b26e`。
- 已取消 strict run 的代码文件 manifest SHA256：`e38bb254283bcdbbbc202daca37c1416e31d1e3bb5b19961fcd022d5d1dcf38e`，包含1033个源码/配置文件的逐文件哈希。
- 已取消 strict run 的代码快照：`source_snapshot_20260716_1544.tar.gz`，SHA256
  `b9ced3dde98a8703bd354ad17bc659a16c37b0de7e3627b356ae4f755079b6e5`；保存在正式 workdir，可恢复本次运行使用的完整源码。

### 8.1 启动前验收

静态检查确认：train/test 为 8372/5416 samples；实际样本输入为
`(pair=2, channel=8, height=900, width=1200)`；模型为 R18、3-layer decoder，
可训练参数 21,907,028；adapted checkpoint 加载无 unexpected key。

双卡 smoke 使用正式 batch、BF16边界和
`find_unused_parameters=False` 完成4次前向/反向。峰值显存约 9166 MB/GPU，均值约
1.1813 s/iter；总 loss、decoder loss、PairDN loss 和 encoder proposal loss 均为有限值，
未出现 OOM、NaN、unused parameter 或 dtype 错误。

### 8.2 正式启动

`0716_02` 于 2026-07-16 15:44 CST 在 99 的 GPU 0/1 fresh 启动。首个正式日志点为
epoch 1 iter 50：`time=0.9545 s/iter`、`memory=9190 MB`（日志统计）、
`loss=33.0213`、`grad_norm=95.9832`；同时采样的设备显存约 15.5 GB/GPU。
日志中未发现 traceback、OOM、NaN 或 unused parameter。按训练迭代 ETA 约20小时，另需
计入每4 epoch的18次完整 validation 与 TrackEval。

该首次运行启用了 strict deterministic。根据用户指令，运行在 epoch 1 内主动终止，日志和
源码快照保留在无后缀历史 workdir，不作为论文结果。模型或数值本身没有报错。

### 8.3 Non-deterministic fresh restart

正式论文 baseline 改为固定 seed 3407、`deterministic=False`，并显式清除
`CUBLAS_WORKSPACE_CONFIG`。其余模型、数据、初始化、优化器、BF16边界、DDP、GMC和
评测设置完全不变。新运行使用 `_nondet_restart` workdir，且不加载首次运行 checkpoint。

新运行于 2026-07-16 15:57 CST 在 GPU 0/1 启动，resolved runtime 明确记录
`deterministic: False`。epoch 1 iter 50 为 `time=0.8223 s/iter`、日志显存 9146 MB、
`loss=32.9911`、`grad_norm=93.5086`；设备显存约15.3 GB/GPU，无 OOM、NaN、unused
parameter 或 traceback。其正式 run log 为
`0716_02..._nondet_restart/20260716_155757/20260716_155757.log`。

Non-deterministic restart 的复现材料：tracked patch SHA256
`0910b70069d12dabf2fb7ac9d933078403712358a406dd8bda8fc255e43c67d8`；逐文件
manifest SHA256 `180a79c733243bba52afbefaae59b86d7a799ba934e1408e0245bcec66807d42`；
完整源码快照 `source_snapshot_20260716_nondet_restart.tar.gz` SHA256
`69b51831de30ef5d9d262a63a0f6b93bd8618b4cbfce8449c34ebc04cef3ea8b`。

该运行仍沿用了历史训练采样：每个 anchor 在 `t-1` 和 `t+1` 中随机选择后再按时间排序。
虽然模型输入没有反向时间，但一个 epoch 会重复部分相邻 pair 并遗漏另一些 pair。根据用户
确认，该运行在 epoch 1 内主动终止，不作为论文结果。

### 8.4 Unique ordered-pair fresh restart

最终论文协议取消 anchor 双向随机选择：训练和测试均使用 `frame_intervals=(1,)`、
`random_interval_range=None`，直接枚举每个唯一 `t-1 -> t` pair。full train 因而严格为
8297个 pair，每个 epoch 各出现一次，不重复、不遗漏，并与 validation/test 时间方向一致。
新运行使用 `_orderedpairs_restart` workdir，保持 `deterministic=False` 且完全 fresh train。

最终运行于 2026-07-16 16:08 CST 在 99 的 GPU 0/1 启动。启动前实际构建数据集并确认
8297个 `(sequence, prev, curr)` key 全部唯一、所有 `curr-prev=1`，首个 transformed sample
为 `(2, 8, 900, 1200)`。runtime resolved config 明确记录 `deterministic=False`、
`find_unused_parameters=False` 和 `resume=False`。

epoch 1 iter 50 为 `time=0.8185 s/iter`、日志显存 `9166 MB`、`loss=34.0257`、
`grad_norm=97.2520`；iter 100 为 `0.8031 s/iter`。截至 iter 150，主损失、PairDN损失和
encoder proposal损失均为有限值，未出现 traceback、OOM、NaN、unused parameter 或 DDP
reduction错误。正式日志为
`0716_02..._orderedpairs_restart/20260716_160843/20260716_160843.log`；启动阶段训练 ETA
约17小时，另需考虑每4 epoch执行的完整 validation 与 TrackEval。

本次最终运行的 tracked patch SHA256 为
`e90cc658c62467f85db16c2cb0be237a1d2f98556b75d17533a45ee84757b564`；1284个
Python/YAML/shell文件的逐文件 manifest SHA256 为
`563afa719fb880e1644b843344f716a7d63827084b8d660517fc13f7d5103d3e`；完整源码快照
`source_snapshot_20260716_orderedpairs_restart.tar.gz` SHA256 为
`bfcbe18aadb4958664cdf4425ac0b229fe625065509da8f3809f19ea79dd9a19`，均保存在最终
workdir根目录。

### 8.5 Base + Liquid fresh run

`0716_03` 只在 `0716_02` 论文 Base 上增加已由 `0715_05` 全量实验验证有效的最终
Liquid：8个循环三谱段组、独立单帧 sampler、wide overlap-aware LAF、group modulation、
pair sampler router和pair transport；两个有向关系模块均只使用有序 `[x,y]`。COCO-only
初始化、1200x900输入、8297个唯一 `t-1 -> t` pair、BF16边界、优化器、72-epoch schedule、
GMC和完整评测协议均与 Base 相同。所有 Liquid 新参数使用 `lr_mult=1.0`。

实验于 2026-07-16 16:16 CST 在本机 GPU 2/3 fresh 启动。epoch 1 iter 50 为
`time=0.8857 s/iter`、日志显存 `10691 MB`、`loss=36.0175`、`grad_norm=169.5151`；
主损失、PairDN和encoder proposal loss均为有限值，未出现 traceback、OOM、NaN、unused
parameter或DDP reduction错误。正式日志为
`0716_03.../20260716_161637/20260716_161637.log`。

启动代码的 tracked patch SHA256 为
`fd0b4b2a7e98b25cf8179a674def5c127eadadadd7f537cbafbb34c069e1cfdc`；1286个
Python/YAML/shell文件的manifest SHA256为
`0780b1fadef05f3b6e58038f9c97effc564a6a63f3e8d3ef266f7979541b382b`；完整源码快照
`source_snapshot_20260716_base_plus_liquid.tar.gz` SHA256为
`bbafd5bc39ef5515a194c0ac606f91656bcb6af8e71285a28ba46a21bfda65c7`。

该次启动最后写入 epoch 1 iter 1000，随后 GPU 2 从驱动层掉卡，`nvidia-smi` 对设备
`0000:B1:00.0` 返回 `Unknown Error`。2026-07-16 16:41 CST 按用户指令终止并清理 GPU 2/3
对应的完整进程组。checkpoint间隔为4 epochs，因此没有正式 checkpoint；本次运行作废，
后续不得 resume，须在健康GPU上以相同配置fresh train。同期 GPU 0/1 的 `0716_02` 进程组
保持存活且训练日志持续推进，未发现NCCL、CUDA、NaN或loss异常。

随后为允许用户手动重启本机，`0716_02` 于 2026-07-16 16:44 CST 在 epoch 3 iter 600
主动停止。停止前 `time=0.7560 s/iter`、`loss=17.3561`、`grad_norm=45.4604`，没有异常。
checkpoint从epoch 4开始每4 epochs保存，因此本次没有checkpoint；重启后不得resume，须以
相同配置fresh train。

### 8.6 Base reboot-fresh run

服务器重启后四张GPU均恢复正常，`0716_02` 于 2026-07-16 17:10 CST 在 GPU 0/1以新
workdir完全fresh启动。新配置与最终Base的model、dataset、optimizer、scheduler、hooks、
randomness逐项相同，仅覆盖workdir和TrackEval输出目录；仍为8297个唯一有序相邻pair、
COCO-only初始化、1200x900、BF16、`find_unused_parameters=False`和`resume=False`。

epoch 1 iter 50 为 `time=0.8307 s/iter`、日志显存 `9166 MB`、`loss=34.0826`、
`grad_norm=121.4044`；主损失、PairDN和encoder proposal loss均为有限值，未出现CUDA、
NCCL、NaN、OOM、unused parameter或DDP reduction错误。正式日志为
`0716_02..._reboot_fresh/20260716_171004/20260716_171004.log`。

启动代码的tracked patch SHA256为
`4c3250f4323f61c5b3997ce70d04d0c8096e8863c6dda7368f5c7d1ec0ec2bb0`；源码manifest
SHA256为`0a9215dc3f3b0f1b6c2bc41d602760f9032ffd56f5f599e44f0065d714a505af`；完整源码快照
`source_snapshot_20260716_base_reboot_fresh.tar.gz` SHA256为
`645e304993e99e01f8626807656e044c0b3ed82408cd10c1c8ff200023e48188`。

该运行已完成全部72 epochs、18次validation和18/18异步TrackEval，未发现异步评测失败
或残留进程。严格按`cls_HOTA + det_HOTA`在18个点中选择，唯一最佳为epoch 68：
`cls_HOTA=53.314`、`det_HOTA=61.982`、`cls_MOTA=44.690`、
`cls_IDF1=62.218`、`det_MOTA=60.599`、`det_IDF1=72.303`，HOTA sum为
`115.296`。同一epoch 68的检测指标为`pair_mAP=0.3149`、`pair_AP50=0.5225`。
epoch 72的HOTA sum为`114.990`，因此仅作为末轮稳定性参考，不作为论文checkpoint。

### 8.7 Base + Liquid fresh run on 197

本机失败尝试不resume。同步当前代码后，`0716_03` 于 2026-07-16 17:15 CST 在197的
GPU 0/3使用新workdir完全fresh启动。197专用配置仅覆盖远端data、GMC、workdir和
TrackEval路径；model、optimizer、scheduler、hooks、randomness、COCO-only初始化、
1200x900输入、8297个唯一有序pair、BF16边界及72-epoch评测协议均与本机正式配置一致。

epoch 1 iter 50 为 `time=1.0818 s/iter`、日志显存 `10692 MB`、`loss=36.0201`、
`grad_norm=194.2549`；主损失、PairDN和encoder proposal loss均有限，未出现CUDA、NCCL、
NaN、OOM、unused parameter或DDP reduction错误。正式日志为
`0716_03..._fresh/20260716_171544/20260716_171544.log`。

远端启动代码的tracked patch SHA256为
`2cf34504aa6af3679111cd4b6562cbf597348b90f15e651257b5e42518831ed2`；源码manifest
SHA256为`7dd1c5f3cb15d109045e17668586d8c06fb91b5f0e1ec796575907a964b54e20`；完整源码快照
`source_snapshot_20260716_base_plus_liquid_197.tar.gz` SHA256为
`b79da8e6f0ef78ada288d02f11c6bbf3a1b75755c49bd8b6500d6faba5fa3b51`。

该运行在epoch 21 iter 50主动停止且不resume。soft sampler的argmax预览已经出现
`432/431`等相同无序三波段集合，说明原实现只保证group内部不重复，不能阻止跨group
坍塌，因而不再作为最终论文Liquid运行。

### 8.8 Base + Liquid group-set-unique fresh run on 197

`0716_04`保持`0716_03`的模型、数据、初始化、优化器、BF16边界和评测协议，只在hard
train/eval增加跨group的无序波段集合唯一分配。8个group从`C(8,3)=56`个候选集合中
选择互不相同的集合，集合内部保留最高分排列；采用GPU上的regret-first greedy和
straight-through反向，不增加loss。soft fusion仍保留所有连续谱段权重，不做互斥屏蔽。

本地及197均通过20项sampler/stem测试。实验于2026-07-16 23:22 CST在197 GPU 0/3以
独立workdir完全fresh启动。epoch 1 iter 50为`time=0.9771 s/iter`、日志显存
`10692 MB`、`loss=35.9972`、`grad_norm=169.3626`，主损失、PairDN和encoder proposal
loss均有限。监控为`hard=False, unique_sets=8.00, max_set_repeat=1.00`，确认hard预览的
8组集合全部唯一。正式日志位于
`/data4/litianhao/PairMmot/workdir_197/0716_04_paper_base_plus_liquid_groupsetunique_r18_coco_full_1200x900_bf16_orderedpairs_fresh/launch.log`。

### 8.9 Base + Liquid group-set-unique + Encoder on 252

`0716_05`严格继承`0716_04`的最终Liquid和论文协议，只加入历史encoder最佳
`0705_01 p5temporal_pyramidlocal`：shared AIFI后的P5 global pair temporal adapter，以及
CCFF/FPN后的P3/P4/P5 pyramid-local pair adapter。两条残差gate均为零初始化；adapter参数
使用`lr_mult=2.0`，gate gamma使用`lr_mult=20.0, decay_mult=0.0`，其余参数学习率不变。

252上的20项Liquid/stem测试和10项temporal adapter测试全部通过。正式启动前使用GPU 0/1
完成100 iter双卡DDP测试：`find_unused_parameters=False`无报错，框架统计显存
`11387 MB/rank`，两个gate和attention/local block均收到梯度，全部loss有限。测试进程组已
完整停止并释放显存，未写入正式目录。

正式实验于2026-07-16 23:36 CST在252 GPU 0/1 fresh启动，workdir为
`/data4/litianhao/PairMmot/workdir_252/0716_05_paper_base_plus_liquid_groupsetunique_encoder_r18_coco_full_1200x900_bf16_orderedpairs_fresh`。
epoch 1 iter 50为`time=1.5238 s/iter`、`memory=11387 MB`、`loss=35.8960`、
`grad_norm=178.3931`；Liquid hard预览为`unique_sets=8.00, max_set_repeat=1.00`，未出现
CUDA、NCCL、OOM、NaN、unused parameter或DDP错误。

### 8.10 Parallel Liquid Set-Transport candidate

`0717_01`是相对`0716_04`的单变量结构探索。它不修改最终Liquid的pair router、wide LAF、
group modulation或pair transport，而是在soft sampler中增加无参数的集合容量传输：将
三slot概率映射到56个无序三波段集合及其6种排列，利用48个slack token和16次log-Sinkhorn
得到容量受限的连续group-set分配，再还原到原波段概率接口。结构强度在前12 epochs从0
增加到1，hard阶段的ST梯度也通过该投影，不增加辅助loss。

23项单测及GPU 2/3上的100 iter双卡DDP测试通过。正式运行于2026-07-17 00:15 CST fresh
启动，epoch 1 iter 50为`time=0.9347 s/iter`、`memory=10695 MB`、`loss=36.0532`、
`grad_norm=177.8087`，未出现异常。workdir为
`/data4/litianhao/PairMmot/workdir_99/0717_01_paper_base_plus_liquid_settransport_r18_coco_full_1200x900_bf16_orderedpairs_fresh`。

该实验目前作为并行Liquid候选，不改变A/B/C主链定义。完成后与`0716_04`分别按唯一最佳
`cls_HOTA + det_HOTA`比较，只有在两个HOTA方向和稳定性均有依据时才考虑替换主线B。

本机正式训练于2026-07-17在epoch 2 iter 250主动取消。取消原因是GPU 2/3存在历史掉卡
风险，而非模型异常；进程组、worker和screen均已完整清理，GPU显存释放。此次不完整运行
不进入结果表，Set-Transport实现和已通过的单元/DDP验证保留，后续在稳定服务器上应fresh
重跑。

同一科学实验保留`0717_01`编号，于2026-07-17 05:40 CST迁移到双卡AutoDL实例并从
COCO适配权重fresh重跑，不使用99上的任何checkpoint。AutoDL路径覆盖配置只修改HSMOT、
GMC、预训练和workdir位置；模型、每卡batch 4、全局batch 8、`lr=1e-4`、BF16边界、
`find_unused_parameters=False`、72 epochs和每4 epoch完整评测均与99原运行一致。workdir为
`/root/autodl-tmp/work_dirs/0717_01_paper_base_plus_liquid_settransport_r18_coco_full_1200x900_bf16_orderedpairs_autodl_fresh`。

epoch 1 iter 50为`time=0.9282 s/iter`、`memory=10703 MB/rank`、`loss=35.7344`、
`grad_norm=193.7061`，与99原运行的`0.9347 s/iter`相符。两卡`nvidia-smi`占用约
18.8 GB/GPU；Set-Transport监控为`strength=0.004`、`unique_sets=8.00`、
`max_set_repeat=1.00`、`set_max_load=0.250`，无CUDA、NCCL、OOM、NaN或unused parameter
错误。当前ETA约19小时，早期99运行仍不计入最终结果。
