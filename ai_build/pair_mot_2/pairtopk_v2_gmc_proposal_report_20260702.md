# PairTopK v2 GMC Proposal 实验报告

更新时间：2026-07-02 09:46 CST

## 目标

本阶段目标是改进 pair-wise detection 的 two-stage proposal 生成方式，并验证其在 HSMOT half gap=1 训练上的效果。核心要求：

- 两帧分别生成单帧 encoder proposals。
- 在 proposal level 计算跨帧 affinity。
- 基于 affinity 构造候选 proposal pairs。
- 基于 pair quality 选择 top-k pair proposals，而不是只按 match score。
- 对 pair proposals 加 encoder-level pair supervision。
- 训练和评测只考虑两帧都存在的 GT，不处理 birth/death。
- GMC 必须缓存复用，避免多实验重复计算。

## 主要代码改动

| 文件 | 作用 |
| --- | --- |
| `mmrotate/datasets/hsmot_pair.py` | 为 pair dataset 增加 `gmc_cache_dir` / `allow_missing_gmc`，在样本 metainfo 中写入 prev->curr GMC 矩阵。 |
| `mmrotate/datasets/transforms/loading_hsmot_pair.py` | `PackHSMOTPairInputs` 传递 `gmc_matrix`，并在存在增强 homography 时同步变换 GMC。 |
| `projects/multispec_pair_rotated_rtdetr/tools/build_hsmot_gmc_cache.py` | 构建 Bot-SORT 风格稀疏 LK GMC cache，按序列和帧对保存 JSON。 |
| `projects/multispec_pair_rotated_rtdetr/multispec_pair_rotated_rtdetr/multispec_pair_rotated_rtdetr.py` | 新增 `query_init='pair_topk_v2'`，实现 per-frame proposal、GMC-aware affinity、pair query fusion、pair quality ranking、encoder pair outputs。 |
| `projects/multispec_pair_rotated_rtdetr/multispec_pair_rotated_rtdetr/pair_rotated_rtdetr_head.py` | 对 pair_topk_v2 输出增加 encoder-level pair supervision。 |
| `projects/multispec_pair_rotated_rtdetr/configs/*pairtopk_v2*.py` | no-DN / PairDN v2 实验配置。 |

## GMC Cache

| split | cache 目录 | 状态 |
| --- | --- | --- |
| train half gap=1 | `/data/users/litianhao01/PairMmot/workdir/aux/gmc_cache/hsmot_train_gap1` | 已构建，约 3839 个帧对。 |
| test gap=1 | `/data/users/litianhao01/PairMmot/workdir/aux/gmc_cache/hsmot_test_gap1` | 已构建，约 5416 个帧对。 |

## 当前实验

| 实验 | 配置 | work_dir | 状态 |
| --- | --- | --- | --- |
| pairtopk_v2 no-DN | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2.py` | `/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2` | 服务器重启后曾从 epoch12 恢复；epoch16 val 已完成。2026-07-02 10:30 检查时进程已停止，最新 checkpoint 为 epoch16；按 2 卡限制暂不与 unique 并发恢复。 |
| pairtopk_v2 PairDN | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_pairdn.py` | `/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_pairdn` | 队列中，no-DN 结束后运行。 |
| pairtopk_v2 unique trial no-DN | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique.py` | `/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique` | epoch4 val 已完成。2026-07-02 10:31 按 2 卡限制从 `epoch_4.pth` 恢复，仅使用 GPU0/1，已进入 epoch5 train。 |
| pairtopk_v2 unique trial PairDN | `o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique_pairdn.py` | `/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique_pairdn` | 已准备，未启动。 |

## 服务器重启与恢复

2026-07-02 03:29:30 CST 服务器发生异常重启，所有训练进程中断。重启后检查到 4 张 GPU 均空闲。

恢复情况：

- `pairtopk_v2 no-DN` 成功从 `/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2/epoch_12.pth` 恢复，日志显示 `resumed epoch: 12, iter: 5808`，并已进入 `Epoch(train) [13]`。
- `baseline dualcls/nopres PairDN` 已保存到 `epoch_20.pth`，但直接 `--resume` 失败。原因是当前源码中新增的 `pair_quality_predictor` 进入了非 v2 配置模型，导致 optimizer 参数组数量与 checkpoint 不一致。已修复为仅在 `query_init='pair_topk_v2'` 时创建该模块，避免后续 baseline 类配置恢复失败。
- 由于 GPU0/1 空闲，已启动 `pairtopk_v2 unique trial no-DN`。该 trial 从适配预训练权重启动，日志已进入 `Epoch(train) [1]`。

2026-07-02 10:30 CST 再次检查时，4 张 GPU 均空闲，原 v2 与 unique trial 均无训练进程；日志最后分别停在原 v2 `Epoch(train) [18][250/484]`、unique `Epoch(train) [5][150/484]`，没有正常完成或 early-stop 记录。根据用户要求，后续调度改为最多只使用 2 张卡：不再同时跑原 v2 与 unique/PairDN。

已优先恢复更有判别价值的 `pairtopk_v2 unique trial no-DN`，启动日志为：

`/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique/launch_resume_20260702_1031.log`

恢复验证结果：

- 仅使用 GPU0/1；GPU2/3 保持空闲。
- 自动从 `/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique/epoch_4.pth` 恢复。
- 日志显示 `resumed epoch: 4, iter: 1936`。
- 已进入 `Epoch(train) [5]`，启动健康。

## 指标记录

| 实验 | epoch | pair_AP50 | independent_AP50 | pair_mAP50_95 | independent_mAP50_95 | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline dualcls/nopres no-DN | 4 | 0.4118 | 0.4378 | 0.2046 | 0.2312 | 历史对照，manual val epoch4。 |
| baseline dualcls/nopres no-DN | 28 | 0.4581 | 0.4770 | 0.2469 | 0.2717 | 历史对照，manual val epoch28。 |
| baseline dualcls/nopres PairDN | 4 | 0.4083 | 0.4360 | 0.2079 | 0.2349 | 历史对照，当前 PairDN 方向首个 val。 |
| baseline dualcls/nopres PairDN | 8 | 0.4204 | 0.4439 | 0.2208 | 0.2462 | 当前 baseline PairDN 第二个 val。 |
| baseline dualcls/nopres PairDN | 12 | 0.4313 | 0.4533 | 0.2288 | 0.2539 | 当前 baseline PairDN 第三个 val，仍在提升。 |
| baseline dualcls/nopres PairDN | 16 | 0.4562 | 0.4772 | 0.2429 | 0.2695 | 当前 baseline PairDN 第四个 val，提升明显，暂不应手动停止。 |
| baseline dualcls/nopres PairDN | 20 | 0.4450 | 0.4644 | 0.2428 | 0.2679 | 较 epoch16 回落，可能开始波动或平台期。 |
| pairtopk_v2 no-DN | 4 | 0.3772 | 0.3990 | 0.1870 | 0.2130 | 首个 val，低于 baseline epoch4。 |
| pairtopk_v2 no-DN | 8 | 0.3968 | 0.4170 | 0.2059 | 0.2318 | 较 epoch4 明显提升，但仍低于 baseline PairDN epoch8。 |
| pairtopk_v2 no-DN | 12 | 0.4241 | 0.4446 | 0.2191 | 0.2453 | 较 epoch8 继续明显提升，暂不应手动停止。 |
| pairtopk_v2 no-DN | 16 | 0.4356 | 0.4557 | 0.2217 | 0.2489 | 较 epoch12 提升 0.0115，刚超过 early-stop `min_delta=0.01`，仍应继续观察。 |
| pairtopk_v2 unique trial no-DN | 4 | 0.4164 | 0.4422 | 0.2050 | 0.2335 | 首个 val 明显高于原始 v2 epoch4，接近 baseline no-DN epoch4，应继续到 epoch8。 |

## 初步分析

pairtopk_v2 no-DN 不是完全失败，AP 已经正常输出，且训练 loss 与 encoder proposal loss 都在下降。epoch4 到 epoch8，`pair_AP50` 从 0.3772 提升到 0.3968，`independent_AP50` 从 0.3990 提升到 0.4170。

但它仍然弱于当前 baseline PairDN：epoch16 时 `pair_AP50=0.4356`，低于 baseline PairDN epoch16 的 0.4562。需要注意的是，这里 no-DN v2 与 PairDN baseline 并不完全公平；v2 从 epoch12 到 epoch16 仍有 `+0.0115 pair_AP50` 的增长，刚好超过 early-stop `min_delta=0.01`，所以当前不能判定为失败方案，也不应在此时强行停止。

当前最可疑的问题不是训练崩溃，而是 proposal pair top-k 的候选覆盖率或排序策略偏保守：

- `learned_quality_weight=0.15` 依赖新初始化的 `pair_quality_predictor`，早期可能引入排序噪声。
- 当前 top-k 允许同一个 prev/curr proposal 被多个 pair 重复占用，可能降低 query 对不同目标的覆盖。
- `affinity_thr=0.25` 和 `max_center_dist=0.18` 对 GMC 误差较敏感，可能过滤掉正确候选。

已准备的 trial 方向：

- 开启 `unique_pair_selection=True`，优先保留一对一 proposal pairs。
- 将 `learned_quality_weight` 置为 0，避免未训练 quality head 干扰早期 ranking。
- 将 `affinity_thr` 从 0.25 降到 0.15，提高候选召回。
- 保持 proposal quality 为主导：`proposal_quality_weight=0.85`，`affinity_rank_weight=0.15`。

2026-07-02 09:54 CST，`pairtopk_v2 unique trial no-DN` 完成 epoch4 val：`pair_AP50=0.4164`，`independent_AP50=0.4422`，`pair_mAP50_95=0.2050`，`independent_mAP50_95=0.2335`。相对原始 v2 no-DN epoch4 的 `0.3772/0.3990` 有明显提升，并且已经接近历史 baseline no-DN epoch4 的 `pair_AP50=0.4118`。这说明当前的试探改动方向有效，至少缓解了早期 proposal pair top-k 排序和重复占用问题。下一步应继续观察 epoch8，确认提升是否能够延续，而不是只改善早期 warmup。

## 待完成

- 当前只保留一个 2-GPU 作业：继续观察 `pairtopk_v2 unique trial no-DN` 到 epoch8 val。
- unique epoch8 完成后，再决定是否恢复原始 `pairtopk_v2 no-DN` 从 epoch16 继续，或直接切换到 unique PairDN / v2 PairDN；所有后续实验串行运行，最多使用 2 张 GPU。
- no-DN v2 / unique 队列结束后运行 PairDN v2，比较 DN 对该 proposal 方案的影响。
- 训练结束后选择 best checkpoint 做 tracking 横向对比。
- 汇总最终 AP、曲线和 tracking 结果，补充最终结论。
