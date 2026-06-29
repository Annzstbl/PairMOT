# Pair RT-DETR 过拟合实验进展报告

更新日期：2026-06-26  
工程：`/data/users/litianhao01/PairMmot/ai4rs/projects/multispec_pair_rotated_rtdetr`

## 1. 目标与验收口径

本阶段目标是在相同的数据、预训练和训练预算下，依次验证：

1. 单帧 RT-DETR 能否在固定小数据集上过拟合；
2. 双帧同帧（prev/curr 为同一图像）能否过拟合；
3. 双帧序贯（真实相邻帧）能否过拟合。

验收采用 `run_hsmot_*_overfit_acceptance.py`。核心要求为：训练损失足够低、所有 GT 均有唯一高分同类预测匹配、匹配预测的 IoU 满足阈值、presence 判断正确。双帧同帧还要求 prev/curr 两侧均正确。当前脚本默认的损失阈值较严格（`final_loss_sum <= 2.0`）；因此报告同时记录覆盖率、重复匹配和 IoU，以区分“未学会”与“验收阈值未达”。

## 2. 公平性控制

- 单帧、双帧均以同一预训练符号 `O2_R18_HSMOT_3DSE_R2_E72` 为起点：
  `/data/users/litianhao01/PairMmot/workdir/o2_rtdetr_r18vd_2xb4_72e_hsmot_coco_pretrain_3dse_reduction2/epoch_72.pth`。
- 双帧模型通过 `tools/load_pair_pretrain.py` 由该单帧权重初始化：固定统计为 `copied=512`、`cross_attn_expanded=24`、`reg_branches_curr_copied=24`、`dropped_dn=1`。双帧新增分支之外的共享参数均来自同一预训练。
- 同帧实验使用真实 HSMOT 序列 `data30-8` 的连续 10 帧构造，得到 9 对、990 个 GT pair；`same_frame=True` 时 prev/curr 图像与 GT 均为同一帧的深拷贝，已检查数据集实现，非伪同帧。
- 单帧实验使用真实 HSMOT 序列 `data52-1` 的 10 帧，GT 总数 1317。
- 常用训练资源为两张 GPU、每卡 batch size 4。所有试验均从相同的适配预训练权重重新开始，未混用中间 checkpoint。

## 3. 单帧：已验收通过

### 发现与修复

初始验收出现平均 IoU 约 `0.01`，但诊断脚本显示每个 GT 均存在同类预测，最佳 IoU 约 `0.94-0.95`。根因是验收端按最高分贪心，未先处理几何对应，导致同一高分重复预测遮蔽了正确预测；不是模型无法过拟合。

已将单帧验收匹配改为：在相同类别、高分候选中，按 IoU 优先做一对一贪心匹配。与此同时，将过拟合配置 `max_per_img` 从 50 提升到 300，避免 130+ 个目标被推理 Top-K 截断。

### 最终结果

| 指标 | 结果 |
| --- | ---: |
| `final_loss_sum` | 1.762232 |
| GT instances | 1317 |
| matched queries | 1317 |
| match ratio | 1.0000 |
| mean IoU | 0.947755 |

结论：**单帧模型在固定真实数据子集上已稳定过拟合并通过验收。**

## 4. 双帧同帧：已完成的尝试

### 4.1 初始 learned-query 基线

`query_init='learned'` 训练 1000 iter 后：

| 指标 | 结果 |
| --- | ---: |
| final loss sum | 10.6304 |
| matched pairs | 400 / 990 |
| match ratio | 0.4040 |
| mean IoU prev/curr | 0.6607 / 0.6647 |
| presence accuracy | 0.9900 |

结论：定位在部分 query 上成立，但固定 learned query 无法充分覆盖密集的 110 个对象/帧场景。

### 4.2 dual-topk、验收和旋转框修复

主要改动：

- 使用 `query_init='dual_topk'`，由 encoder proposal 初始化 pair query；
- DDP 设置 `find_unused_parameters=True`，因为 dual-topk 不使用 learned query/ref embedding；
- 修复双帧 metric 对 dict/object field 的兼容和一对一 IoU 优先匹配；
- 旋转框预测统一执行 `RotatedBoxes.regularize_boxes`；
- checkpoint hook 显式设置 `by_epoch=False`，确保保存 `iter_*.pth`；
- `max_per_img` 提升为 300。

3000 iter 后结果：`909/990`，match ratio `0.9182`，mean IoU prev/curr `0.7571/0.6605`。这证明双帧分支可学习，但 curr 分支显著弱于 prev，未达标。

### 4.3 对齐 prev/curr 的 Top-K 索引

原实现将 prev 的 top-k query 与 curr 独立 top-k reference 强行配对，一个 pair query 的两侧可能对应不同物体。现已改为：以 prev top-k 索引同时 gather curr proposal/reference，使同一个 pair query 对应同一 encoder hypothesis。

结果（3000 iter）：

| 指标 | 结果 |
| --- | ---: |
| matched pairs | 932 / 990 |
| match ratio | 0.9414 |
| mean IoU prev/curr | 0.7615 / 0.7624 |
| presence accuracy | 0.9925 |
| acceptance loss sum | 4.5464 |

结论：该修复消除了 prev/curr 的系统性不对称，属于必要的正确性修复；但仍有少量缺失和重复匹配，损失未达到阈值。

### 4.4 Pair fusion 的单帧等价初始化

Pair decoder 原来的随机 `cross_fusion([out_prev, out_curr])` 会覆盖两个 cross-attention 已带残差的输出。对同帧输入，这会在初始化时破坏单帧预训练表示。

已将 `cross_fusion` 和 `pair_pos_fusion` 初始化为 `0.5 * prev + 0.5 * curr` 的 identity-average。该改动仅影响新加的双帧融合层，保留单帧预训练行为。

这是当前最好的 Varifocal 基线：

| checkpoint | matched pairs | match ratio | mean IoU prev/curr | loss sum |
| --- | ---: | ---: | ---: | ---: |
| iter 1000 | 943 / 990 | 0.9525 | 0.7394 / 0.7511 | - |
| iter 2000 | 960 / 990 | 0.9697 | 0.7680 / 0.7549 | 3.5866 |
| iter 3000 | 954 / 990 | 0.9636 | 0.6982 / 0.7449 | 3.6604 |

结论：同帧双帧模型已经接近覆盖验收，但未达到 loss 和完全唯一匹配要求；最佳 checkpoint 是 `iter_2000.pth`，后续训练并未继续改善。

## 5. 诊断结论

对 fusion-average iter 2000 的逐 GT 诊断：

- 不考虑置信度时，贪心唯一匹配覆盖 `990/990`；其中约 `95.66%` 的最佳几何候选 IoU >= 0.5。
- 加入 score >= 0.35 后，匹配为 `961/990`，其余 29 个主要是候选间竞争导致的唯一匹配失败，而非完全没有候选。
- 同类最高分预测的平均 IoU 仅约 `0.0565`，IoU >= 0.5 的比例约 `6.97%`。
- 几何最佳候选的平均分约 `0.4869`，只有约 `38.28%` 的分数 >= 0.5。

因此当前主要矛盾不是“没有框”，而是**分类置信度与几何质量没有对齐**：高 IoU 框存在，但高分常落在错误几何 query 上。该判断解释了为何增大 query 数、仅改变 assigner 或延长训练没有稳定收益。

## 6. 已验证但未改善的对照

| 对照 | 目的 | 1000 iter 结果/结论 |
| --- | --- | --- |
| GT-noised reference | 估计 GT 附近 reference 上限 | 随机参考使结果不稳定，不能作为可靠验收上限，停止 |
| `clip_grad=1.0` | 抑制梯度异常 | 训练停滞，停止 |
| encoder auxiliary loss | 直接监督 encoder proposal | `929/990`，match ratio `0.9384`，低于 fusion-average，已撤回默认实现 |
| 几何-only Hungarian cost | 排除分类 assignment 影响 | `851/990`，match ratio `0.8596`，明显退化 |
| `num_queries=600` | 提高候选容量 | `913/990`，match ratio `0.9222`，退化 |
| 3000/4500 iter LR decay | 让后期细化 | `905/990`，match ratio `0.9141`，退化 |
| rotated-IoU Varifocal target | 使分类质量更贴近旋转框 | `918/990`，match ratio `0.9273`，退化 |

这些实验均以相同适配预训练和同一 990-pair 同帧数据运行，说明当前瓶颈不是显而易见的训练时长、query 容量或角度 IoU 定义。

## 7. 当前运行中的 FocalLoss 对照

配置：`o2_pair_rtdetr_r18vd_overfit_sameframe_focalcls.py`。仅把分类损失改为标准 sigmoid FocalLoss，使正样本目标为 1，而不是 Varifocal 的 IoU 质量目标。

| 时刻 | matched pairs | match ratio | mean IoU prev/curr | 结论 |
| --- | ---: | ---: | ---: | --- |
| iter 1000 | 979 / 990 | 0.9889 | 0.6643 / 0.6688 | 分数覆盖大幅改善，定位下降 |
| iter 3000 | 985 / 990 | 0.9949 | 0.6232 / 0.6180 | 重复匹配仅 5，但定位继续下降 |

结论：FocalLoss 直接验证了上述诊断。正标签分类可以解决置信度/唯一匹配问题，却使定位质量显著低于 Varifocal 基线，故不能作为最终方案。

## 8. 当前代码状态与可保留修复

建议保留的正确性/工程修复：

- `max_per_img=300`；
- 单帧和双帧验收的一对一 IoU 优先匹配；
- 双帧 `dual_topk` 与 prev/curr 对齐 Top-K gather；
- 旋转框 canonicalization；
- iter checkpoint 保存配置；
- pair fusion 的 identity-average 初始化；
- 单帧/双帧预测诊断脚本。

不应默认保留的实验性改动：encoder auxiliary loss、geometry-only assignment、600 query、LR decay、rbox-VFL、纯 FocalLoss；这些应继续作为独立配置对照，不能污染公平基线。

## 9. 下一步建议

双帧同帧尚未通过，序贯实验不应提前作为最终比较结论。下一步应以 fusion-average + hbox Varifocal 的最佳基线为起点，做最小、可隔离的混合分类损失实验：保留原 Varifocal 的 IoU 质量监督，同时增加小权重的正标签 BCE/Focal auxiliary loss。目标是提升高 IoU 候选的排序和唯一性，同时不放弃定位质量。

建议执行顺序：

1. 完成并记录当前 FocalLoss 的 acceptance 结果，作为“排序上限、定位下限”对照；
2. 新增独立配置，仅启用小权重（例如 0.1、0.25）的正标签分类辅助项；默认模型行为保持不变；
3. 优先跑至 1000 iter；只有当 coverage 和 IoU 同时超过 fusion-average 1000 结果时，继续至 2000；
4. 同帧通过后，冻结模型和预训练策略，用完全相同的训练预算启动真实序贯双帧过拟合；
5. 每次运行记录 GPU 占用、配置 diff、预训练适配统计、数据子集和 iter checkpoint 指标，保证可复现比较。

## 10. 关键路径

- 单帧验收数据：`/data/users/litianhao01/PairMmot/tmp/hsmot_single_overfit_accept`
- 同帧 fusion-average 最佳结果：`/data/users/litianhao01/PairMmot/tmp/hsmot_pair_sameframe_fusionavg_accept/work_dir`
- 当前 FocalLoss 结果：`/data/users/litianhao01/PairMmot/tmp/hsmot_pair_sameframe_focalcls_accept/work_dir`
- 续接记忆：`/data/users/litianhao01/.codex/memories/extensions/ad_hoc/notes/2026-06-26-*.md`

## 11. 2026-06-26：验收指标改造

原验收将聚合 loss、固定 score 阈值下的一对一匹配和 IoU 混合作为硬条件。该设计不适合作为跨单帧/双帧的主结论：loss 的数值受层数、损失权重和归一化方式影响，固定 score 阈值也无法评价排序质量。

现在的验收定义如下：

| 场景 | 硬指标 | 默认宽松阈值 | 说明 |
| --- | --- | ---: | --- |
| 单帧 | rotated `AP50` | 0.90 | 标准检测覆盖与分数排序 |
| 单帧 | rotated `mAP50:95` | 0.40 | 小目标场景下的定位质量下限 |
| 双帧独立检测 | prev/curr AP 的宏平均 `independent_AP50` | 0.90 | 检测能力，不考察关联 |
| 双帧独立检测 | `independent_mAP50:95` | 0.40 | 双侧旋转框定位质量 |
| 双帧关联 | `pair_AP50` | 0.80 | 同一 query 正确关联 prev/curr 的能力 |
| 双帧关联 | `pair_mAP50:95` | 0.30 | 关联后的严格定位质量 |

双帧 pair AP 对 GT pair 与预测 pair 按类别、全局 score 排序做标准一对一 AP 匹配。双侧可见目标使用 `min(IoU_prev, IoU_curr)`；仅单侧可见目标要求预测 presence 模式与 GT 一致，并仅计算有效侧 IoU。独立 AP 将 prev、curr 输出分别作为检测结果，以 `class_score * presence_score` 排序并取两侧 AP 宏平均。`association_gap_AP50 = independent_AP50 - pair_AP50` 用于区分检测问题和关联问题。

已用两类回归验证实现：

- 已通过的单帧 checkpoint：`AP50=1.000`、`AP75=1.000`、`mAP50:95=0.923`，新验收通过。
- 同帧 FocalLoss iter 3000：`independent_AP50=0.7615`、`independent_mAP50:95=0.3450`、`pair_AP50=0.7265`、`pair_mAP50:95=0.3106`，未通过。其 `association_gap_AP50=0.0350`，表明该实验的主失败点是检测/定位，不是关联。
- 同帧 fusion-average iter 2000：`independent_AP50=0.9585`、`independent_mAP50:95=0.5230`、`pair_AP50=0.9510`、`pair_mAP50:95=0.4789`，通过，是当前最佳 checkpoint。iter 3000 的 `pair_AP50=0.9517` 略高，但 `pair_mAP50:95=0.3873`、`independent_mAP50:95=0.4491` 均明显更低，因此不选用。

实现位置：

- `multispec_pair_rotated_rtdetr/overfit_ap.py`：共享 AP、独立 AP 和 pair AP；
- `tools/run_hsmot_single_overfit_acceptance.py`：单帧标准 AP 验收；
- `tools/run_hsmot_pair_overfit_acceptance.py`：双帧独立 AP + pair AP 验收；
- `multispec_pair_rotated_rtdetr/pair_overfit_metric.py`：训练期 validation 同步记录上述 AP，原匹配统计仅作为诊断。

## 12. 同帧双帧历史实验的统一 AP 重评

以下 checkpoint 均在各自保留的真实同帧数据子集上，使用第 11 节的同一 AP 实现重新评估。`AP50` 更偏向覆盖和排序，`mAP50:95` 更能反映小目标的旋转框定位质量，因此以 pair mAP 为主排序。

| 实验 | checkpoint | independent AP50 | independent mAP50:95 | pair AP50 | pair mAP50:95 | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| fusion-average | iter 2000 | 0.9585 | **0.5230** | 0.9510 | **0.4789** | 当前最佳综合结果 |
| q600 | iter 1000 | 0.9449 | 0.4866 | 0.9233 | 0.4274 | 不如基线，query 增量无收益 |
| encoder auxiliary loss | iter 1000 | 0.8829 | 0.4576 | 0.8662 | 0.4273 | 定位尚可，覆盖明显不足 |
| aligned-topk | iter 3000 | 0.9453 | 0.4798 | 0.9301 | 0.4240 | 修复双侧对齐有效，但不及 fusion-average |
| LR decay | iter 1000 | 0.9428 | 0.4857 | 0.9238 | 0.4229 | 未优于基线 |
| fusion-average | iter 3000 | **0.9627** | 0.4491 | **0.9517** | 0.3873 | AP50 略高，但后期定位退化 |
| dual-topk（早期） | iter 3000 | 0.8964 | 0.4591 | 0.8475 | 0.3722 | prev/curr 不平衡、关联差距较大 |
| FocalLoss 分类 | iter 3000 | 0.7615 | 0.3450 | 0.7265 | 0.3106 | 分数覆盖改善，但定位退化 |
| rbox-VFL | iter 1000 | 0.9016 | 0.3552 | 0.8560 | 0.2949 | rotated-IoU 分类目标退化 |
| geometry-only assignment | iter 1000 | 0.8922 | 0.3816 | 0.8404 | 0.2873 | 去掉分类匹配代价退化 |

没有纳入排名的实验：GT-noised reference 与 clip-grad=1.0 没有可用训练 checkpoint；最早的同帧 workdir 被多次复用且仅保留 `epoch_1/iter_20`，不具备明确的训练预算和配置归属，不能作为公平对照。

结论：已经做过多项结构、query、assignment、分类目标和优化策略对照。真正优于其他已评估方案的仍是 **fusion-average + 原 hbox Varifocal**，并且应在 iter 2000 停止，而不是继续到 3000。后续实验应以它为唯一同帧基线，不再重复已证实退化的 q600、geometry-only assignment、rbox-VFL、纯 FocalLoss 或当前 LR decay。

## 13. 双帧序贯过拟合验收：通过

在同帧验收通过后，冻结模型结构和公平性策略，仅将数据切换到真实相邻帧：

- 真实序列：HSMOT `data30-8`，frames 9..18，构成 9 个 `frame_interval=1` 的相邻帧对；
- 数据状态：8 个 mixed pair、1 个 disappear pair；
- 训练：2 x RTX 3090，per-GPU batch size 4，3000 iter 预算、1000 iter 验证；
- 初始化：与单帧/同帧相同的 `O2_R18_HSMOT_3DSE_R2_E72`，适配统计 `copied=512 / cross_attn_expanded=24 / reg_branches_curr_copied=24 / dropped_dn=1`；
- 选择：依据同帧结论，在 iter 2000 保存后停止，不继续训练到可能退化的 iter 3000。

训练期首次 validation 暴露 AP metric 在 DDP evaluator 输入为 dict 时的字段访问错误。已将共享 AP 序列化器改为同时支持 dict 与对象；首次任务的 `iter_1000` 未落盘，随后从完全相同的预训练和数据重新运行。该问题仅影响训练期指标收集，未改变模型、数据或训练目标。

最终 checkpoint：

`/data/users/litianhao01/PairMmot/tmp/hsmot_pair_sequential_fusionavg_accept/work_dir_retry/iter_2000.pth`

最终独立验收结果：

| 指标 | 结果 | 阈值 | 状态 |
| --- | ---: | ---: | --- |
| independent AP50 | 0.9431 | 0.90 | 通过 |
| independent mAP50:95 | 0.4786 | 0.40 | 通过 |
| pair AP50 | 0.8884 | 0.80 | 通过 |
| pair mAP50:95 | 0.3654 | 0.30 | 通过 |
| presence accuracy | 0.9947 | 诊断项 | 正常 |

分支诊断：prev `mAP50:95=0.5487`，curr `mAP50:95=0.4086`，符合相邻帧当前侧定位更难的预期；`association_gap_AP50=0.0547`，较 iter 1000 的约 0.132 显著改善。结论：**双帧序贯过拟合在既定宽松 AP 验收下已经通过。**

## 14. 9→10 可视化的低分 pedestrian 诊断

对 `val_vis/iter_002000/0004_pair-overfit-real_9_10.jpg` 的检查表明，图中大部分绿色 pedestrian GT 没有对应橙色框的直接原因是可视化使用固定 `score_thr=0.35`，而不是该帧没有预测框或坐标映射错误。

- 该真实 pair 为 frame 9→10，106 个 GT；
- 不施加分数阈值时，106/106 GT 都存在唯一同类几何匹配，平均有效双侧 IoU 为 0.669；
- 阈值匹配数：`0.10: 106`、`0.20: 104`、`0.25: 89`、`0.30: 68`、`0.35: 40`、`0.40: 21`；
- 其中 pedestrian 有 54 个 GT、60 个预测，最高分类分数仅 0.355：`>=0.35` 只有 1 个、`>=0.25` 有 40 个、`>=0.10` 有 57 个；
- 对比 car 为 29 个 GT、`>=0.35` 有 28 个预测。因此问题集中于该 pair 的 pedestrian 置信度校准，不是全类别或 presence 分支失败。

可视化 hook 的阈值匹配还按“每个 GT 的最高类别分数”分配 query，并不以 IoU 优先，因此对密集小目标会比 AP 的全局排序匹配更保守。结论：这不是数据/坐标 bug，也不是完全漏检；但它揭示了序贯模型在该困难 pair 的 pedestrian score calibration 弱。当前 AP 验收仍通过，因为 AP 评估所有分数并衡量整体排序；若部署要求固定阈值下每帧稳定召回，应额外加入 per-pair、per-class `Recall@score` 下限，或单独训练/校准分类分数，不能仅依赖现有 AP 验收。

## 15. 可视化更新

原有可视化将预测按 GT 顺序分配给最高类别分数 query，容易将“低分但几何正确”显示为没有检测。现已改为同时生成三种视图：

| 文件 | 橙/黄/青预测含义 | 用途 |
| --- | --- | --- |
| `*.jpg` | 橙色：所有 `score>=0.35` 且该侧 presence>=0.5 的预测 | 固定部署阈值下的真实输出 |
| `*_low_score.jpg` | 黄色：所有 `score>=0.10` 且该侧 presence>=0.5 的预测 | 检查低置信度目标是否已有候选框 |
| `*_iou_diag.jpg` | 青色：同类、一对一、双侧 `min IoU>=0.5` 的 GT-IoU 优先匹配 | 检查几何覆盖和关联，不代表部署输出 |

绿色框始终为 GT。图底部 `match` 仍是旧的 `score>=0.35` 诊断统计，不能与 IoU 诊断图中的青色框数量直接比较。

已用序贯 accepted checkpoint 重新生成以下关键帧对的三种视图：

`/data/users/litianhao01/PairMmot/tmp/hsmot_pair_sequential_fusionavg_accept/work_dir_retry/val_vis/iter_002000/`

- `0000_pair-overfit-real_1_2*`
- `0001_pair-overfit-real_3_4*`
- `0002_pair-overfit-real_5_6*`
- `0003_pair-overfit-real_7_8*`
- `0004_pair-overfit-real_9_10*`

## 16. 增大时序间隔的序贯过拟合：通过

为进入下一阶段的难度验证，在不改变模型、优化器、训练预算、预训练权重或 AP 验收阈值的前提下，将序贯 pair 的源帧间隔从 1 增加到 3 和 5。为确保 `gap=3` 与 `gap=5` 彼此严格可比，两组均固定使用 HSMOT `data36-9`、起始源帧 43，并各抽取 10 个源帧（构成 9 个 mixed pair）。mini 数据集重新编号为 1..10，仅源帧 ID 间隔不同。

| 组别 | 源帧 ID | GT pair 数 | GPU | 初始化 |
| --- | --- | ---: | --- | --- |
| gap=3 | 43, 46, 49, 52, 55, 58, 61, 64, 67, 70 | 854 | 0,1 | 相同单帧预训练适配权重 |
| gap=5 | 43, 48, 53, 58, 63, 68, 73, 78, 83, 88 | 919 | 2,3 | 相同单帧预训练适配权重 |

训练均为 2 x RTX 3090、per-GPU batch size 4、3000 iter 预算、1000 iter 验证。数据构造工具新增 `--source-frame-interval`、`--source-seq`、`--source-start-frame`，并将实际选择的源帧 ID 写入 manifest；验收脚本已透传这些参数。

iter 1000 的首次结果：

| 组别 | independent AP50 | independent mAP50:95 | pair AP50 | pair mAP50:95 | 阶段结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| gap=3 | 0.9067 | 0.4404 | 0.8273 | 0.3295 | 已通过全部阈值 |
| gap=5 | 0.7826 | 0.3874 | 0.6215 | 0.2592 | 未通过，继续至 iter 2000 |

`gap=3` 说明现有双帧序贯模型能够在三帧源间隔的真实运动中记忆训练子集。`gap=5` 的当前侧检测明显变难（prev AP50 0.9039，curr AP50 0.6613），并带来较高的关联差距 `association_gap_AP50=0.1611`；这更符合运动幅度和遮挡增加的预期，尚不能据此判定模型或数据实现存在错误。

iter 2000 的独立 checkpoint 验收结果：

| 组别 | independent AP50 | independent mAP50:95 | pair AP50 | pair mAP50:95 | association gap AP50 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| gap=3 | 0.9239 | 0.5167 | 0.8912 | 0.4197 | 0.0327 | 通过 |
| gap=5 | 0.9134 | 0.5291 | 0.8578 | 0.4252 | 0.0556 | 通过 |

checkpoint：

- gap=3：`/data/users/litianhao01/PairMmot/tmp/hsmot_pair_gap3_fusionavg_accept/work_dir/iter_2000.pth`
- gap=5：`/data/users/litianhao01/PairMmot/tmp/hsmot_pair_gap5_fusionavg_accept/work_dir/iter_2000.pth`

两个 checkpoint 都通过独立检测 `AP50>=0.90`、`mAP50:95>=0.40` 和关联 `AP50>=0.80`、`mAP50:95>=0.30`。`gap=5` 在 iter 1000 未通过，但到 iter 2000 四项指标全部越过阈值，说明该设置只是比 `gap=3` 收敛慢，并非结构或数据管线缺陷。训练在保存 iter 2000 后主动停止，未继续到 3000，以遵循已有的最佳停止点结论并释放 GPU 资源。

阶段结论：**增加到原始源帧间隔 3 和 5 的双帧序贯真实数据，都已完成过拟合验收。** 后续可进一步扩展到更长间隔、更多序列或不重叠验证集，以研究泛化边界；这些不应混入当前“能否过拟合”的验收结论。

## 17. PairDN 实现与序贯过拟合验收：通过

新增 PairDN：以 pair GT 的 track-id union 为一个 DN 单位，生成共享的带标签噪声 query、prev/curr 两套带噪旋转 reference、DN-group attention mask，以及分类、双侧 presence、双侧 box/IoU 的直接 DN 监督。缺失侧不计算 box/IoU，但仍学习该侧 presence=0。原无 DN 配置保持不变。

首轮使用 `dn_loss_weight=1.0` 在与序贯基线相同的真实 `data30-8` frames 9..18、2 x RTX 3090、per-GPU bs=4、2000 iter 条件下未通过：independent AP50=0.7896、pair AP50=0.6975，但两项 mAP 已越过阈值。DN loss 数值稳定且持续下降，因此不是数据、mask 或双侧 target 的实现错误。

原因分析：PairDN 同时为每个 decoder layer 增加分类、双 presence、双 box、双 IoU 七项损失；在仅 9 个 pair 的 overfit 小集上，将它们与 matching loss 同权相加，辅助任务的梯度规模过大，减慢了 matching query 的 AP50 拟合。这是 loss 标定问题，而不是 PairDN 结构无法工作。

将 DN 总权重设为 `0.2` 后，从完全相同预训练重新训练，最终 checkpoint：

`/data/users/litianhao01/PairMmot/tmp/hsmot_pair_sequential_pairdn_w02_accept/work_dir/iter_2000.pth`

| 指标 | PairDN w=0.2 | 阈值 | 状态 |
| --- | ---: | ---: | --- |
| independent AP50 | 0.9589 | 0.90 | 通过 |
| independent mAP50:95 | 0.5097 | 0.40 | 通过 |
| pair AP50 | 0.9208 | 0.80 | 通过 |
| pair mAP50:95 | 0.4104 | 0.30 | 通过 |
| association gap AP50 | 0.0381 | 诊断项 | 正常 |

结论：**PairDN 已实现并通过真实序贯 overfit 验收。** 全量训练应从 `dn_loss_weight=0.2` 起步；不应使用未标定的 1.0 总权重。全量阶段需额外记录 matching 与 DN 分支损失、gap 分桶 AP，验证该权重在数据规模增大后是否仍合适。

## 18. `half.txt` 正式训练前审计（进行中）

### 已修正的问题

1. 原 `HSMOTPairDataset` 只能构造固定的正向 `frame_interval`，不能实现每个 epoch 对每个锚帧随机选取 `[-5,+5]` 的时间邻居。现扩展为：训练使用 `random_interval_range=(1,5)`，每个有效锚帧恰好产生一个样本；候选从同序列的 `-5..-1,+1..+5` 可用帧均匀随机选择，之后统一为 past-to-future 输入顺序。随机性由 `sample_seed + epoch + sequence + frame_id` 决定，可复现；训练 DataLoader 关闭 `persistent_workers`，并由 `PairDatasetEpochHook` 在每 epoch 重建伙伴映射。
2. 正式 pair 配置此前不存在。新增 `configs/hsmot_pair.py` 与 `configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn.py`：R18、dual-topk、PairDN `dn_loss_weight=0.2`、2 GPU x batch 4、72 epoch、每 6 epoch 验证。
3. 单帧预训练不能直接用于 pair decoder。已生成适配 checkpoint：`pretrained_weights/o2_r18_hsmot_3dse_r2_e72_pair_adapted/pair_adapted_pretrain.pth`。映射复制了全部 24 个双分支 cross-attention 参数和 24 个 current 回归分支参数；presence、pair fusion、learned refs/query 及 PairDN label embedding 为结构新增参数，保留随机初始化。
4. 发现先前 PairDN 的两份 DN copy 都只是小幅噪声，未形成真正的正/负 contrastive DN。现已改为正样本在半框内扰动、负样本在 `[1,2)` 个半框距离带扰动。此前 w=0.2 的过拟合结论针对旧实现，需对该修正版本重新做一次相同的序贯 overfit 验收后才可作为正式训练依据。
5. 原验证可视化会保存全部样本的三种图。正式配置限制为每次验证前 24 个样本，并保存 deploy、low-score、IoU-diagnostic 三视图；`TrainingCurveHook` 已适配独立 AP/关联 AP 曲线。

### 数据与验证规模核验

- `train_half.txt` 包含 30 个序列；可用训练锚帧共 **3869**。epoch 1 核验为 3869 个 pair 且 3869 个唯一 `(seq, anchor_frame)`，满足“每张训练图片一次”的约束。一次抽样中 gap 1..5 的样本数为 778/803/779/743/766。
- 固定测试验证集采用 gap 1、3、5 的并集，共 **15948** 对：5416 / 5316 / 5216。验证日志输出总 `independent_AP50`、`independent_mAP50_95`、`pair_AP50`、`pair_mAP50_95`、`association_gap_AP50`，并输出每个 gap 的同名分桶指标。
- 单 GPU 短 smoke：1 个训练 batch 的前向、反向和所有 PairDN loss 正常，显存约 5.25 GB；2 个验证样本完整产出 AP 日志、6 张三视图和曲线图。输出位于 `/tmp/pair_half_train_val_smoke/20260626_230749/`。

### 下一步

1. 使用修正后的 PairDN 正负噪声，在此前相同真实序贯小集、相同预训练和 `dn_loss_weight=0.2` 下重新完成 2000 iter AP 验收。
2. 验收通过后，在空闲的两张 3090 上启动正式 `half.txt` 72-epoch 训练；每 6 epoch 记录完整 AP、分 gap AP、checkpoint、24 个代表性样本三视图和曲线图。

### PairDN 正负噪声修正后的 gap=1 重验收：通过

在与第 17 节相同的真实序贯数据（`data30-8`，源帧 9 起，10 帧）、相同的 2 x RTX 3090、每卡 batch size 4、2000 iter、相同单帧预训练和 `dn_loss_weight=0.2` 条件下，仅替换为真正的正/负 DN 噪声后重新验收：

| independent AP50 | independent mAP50:95 | pair AP50 | pair mAP50:95 | association gap |
| ---: | ---: | ---: | ---: | ---: |
| 0.9644 | 0.5700 | 0.9313 | 0.4702 | 0.0331 |

四项阈值均通过。checkpoint：`/data/users/litianhao01/PairMmot/tmp/hsmot_pair_sequential_pairdn_contrastive_accept/work_dir/iter_2000.pth`。

正式验证范围按最新决定收窄为 **gap=1 序贯帧**：共 5416 个测试 pair；不再将 gap=3/5 纳入当前验收。正式工作目录固定为 `/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn`。

## 19. `half.txt` 正式训练已启动

- 启动时间：2026-06-26 23:54（Asia/Shanghai）。
- GPUs 0,1；2 x RTX 3090、per-GPU batch size 4、全局 batch size 8。
- 训练：72 epoch，随机有效间隔 `[-5,+5]`；每 epoch 3869 个锚帧、约 484 iter。
- 验证：仅 gap=1 序贯帧、5416 个 pair；每 6 epoch，输出独立 AP、Pair AP、association gap、24 个样本三视图和 checkpoint。
- 初始运行正常：epoch 1 的 150/484 iter 时 `iter_time≈0.768s`、`data_time≈0.026s`、每卡显存约 10.6 GB、无 NaN/异常梯度。训练本体 ETA 约 7.5h；包含 12 次验证的总预估约 12-15h。
- 策略：保持主线不改；在第 2 或第 3 次验证（epoch 12 或 18）后，以 AP、association gap、matching/DN loss 比例和可视化失败模式决定一个最小变量的并行对照，使用 GPUs 2,3。

### 2026-06-27 监督记录：epoch 6/12 验证

后台 30 分钟监督脚本已按时写入 `<work_dir>/supervision.log`，但 Codex 对话窗口本身不会自动唤醒发消息；人工监督需要在窗口保持运行或收到用户消息后继续检查。主训练进程持续正常，epoch 6 与 epoch 12 验证后均恢复训练。

| 验证点 | independent AP50 | independent mAP50:95 | pair AP50 | pair mAP50:95 | association gap AP50 | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| epoch 6 | 0.2155 | 0.0855 | 0.1288 | 0.0386 | 0.0867 | 早期指标，链路正常 |
| epoch 12 | 0.2118 | 0.0881 | 0.1430 | 0.0556 | 0.0688 | Pair AP 小幅提升，独立 AP50 基本未提升 |

epoch 12 验证 forward 在 `01:36:35` 到达 `[2700/2708]`，最终 AP 行在 `01:53:35` 写出，最后阶段耗时约 17 分钟。进程期间 CPU/GPU 均活跃，随后进入 epoch 13，因此不是死锁；但正式验证 metric 明显偏慢。代码检查显示 `HSMOTPairAPMetric` 在 `process()` 中对 GT/query 做嵌套匹配并反复调用旋转 IoU，`compute_metrics()` 又按类别和 IoU 阈值做全量排序 AP，且 `independent_ap_metrics()` 会额外重复一次 pair AP 计算。后续若验证频率或数据规模继续增加，应优化该 metric；当前不影响 checkpoint 安全。

基于 epoch 12 结果，检测独立 AP50 未随训练明显提升，而当前主训练使用随机 gap 1..5、验收只看 gap=1。为判断随机较大间隔是否拖慢 gap=1 早期收敛，新增最小变量并行对照配置：

`projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train.py`

该配置仅将训练 `random_interval_range` 改为 `(1, 1)`，其余模型、PairDN、预训练、优化器、验证指标和可视化全部沿用主配置。计划使用 GPUs 2,3 启动，主线 GPUs 0,1 不改。

### 2026-06-27 01:55 并行对照已启动

- 对照：gap1-only train ablation。
- 配置：`projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train.py`。
- Workdir：`/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train`。
- GPU：2,3；launcher PID `2541334`，train ranks `2541391/2541392`。
- 初始状态：同一 pair-adapted pretrain 加载正常；expected missing keys 与主跑一致。epoch 1 iter 50 已输出，loss=27.4683，显存约 16GB/卡，无 NaN 或启动异常。
- 对照后台监控：PID `2541860`，同样每 1800 秒写入 `<gap1_workdir>/supervision.log`。

### 2026-06-27 12:12 监督记录：主线继续提升，gap1-only 不占优

当前两组训练进程均正常运行，无 OOM/NaN/退出。主线使用 GPUs 0,1，已完成 epoch 66 验证并继续训练；gap1-only 对照使用 GPUs 2,3，已完成 epoch 54 验证并继续训练。

主线正式训练（随机 gap 1..5 训练，gap=1 验证）最近验证：

| 验证点 | independent AP50 | independent mAP50:95 | pair AP50 | pair mAP50:95 | association gap AP50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| epoch 54 | 0.3055 | 0.1282 | 0.2101 | 0.0759 | 0.0954 |
| epoch 60 | 0.3209 | 0.1391 | 0.2243 | 0.0854 | 0.0966 |
| epoch 66 | 0.3322 | 0.1459 | 0.2316 | 0.0902 | 0.1005 |

gap1-only train ablation 最近验证：

| 验证点 | independent AP50 | independent mAP50:95 | pair AP50 | pair mAP50:95 | association gap AP50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| epoch 42 | 0.2666 | 0.1059 | 0.1535 | 0.0464 | 0.1131 |
| epoch 48 | 0.2730 | 0.1104 | 0.1532 | 0.0484 | 0.1198 |
| epoch 54 | 0.2728 | 0.1125 | 0.1485 | 0.0484 | 0.1243 |

结论：gap1-only 对照没有带来更好的 gap=1 验证表现。它早期 independent AP50 较高，但中后期基本平台化，pair AP50 在 epoch 54 仍仅 0.1485；主线随机 gap 1..5 在 epoch 48 后持续超过对照，并在 epoch 66 达到 pair AP50 0.2316。当前不应将主线改为 gap1-only；随机 gap 1..5 训练更可能提供有用的时序扰动和泛化。

指标计算仍是主要开销：验证 forward 结束到最终 AP 行写出约 20 分钟量级，但进程保持 CPU/GPU 活跃并会恢复训练，因此不是死锁。后续若需要加快迭代，应优先优化 `HSMOTPairAPMetric` 的重复 pair AP 计算和 GT-query 旋转 IoU 嵌套匹配。

### 2026-06-27 13:42 监督记录：训练中断与主线续训

13:33 检查时 4 张 GPU 均已空闲，原主线与 gap1-only 的 launcher/rank/monitor 进程全部消失。原始日志没有 OOM、Traceback、RuntimeError 或正常完成记录；主线最后落盘 checkpoint 为 `epoch_66.pth`，之后日志停在 epoch 67 train `[200/484]` 附近；gap1-only 最后落盘 checkpoint 为 `epoch_54.pth`，之后日志停在 epoch 56 train `[350/484]` 附近。因此判断为外部/上层进程中断，而不是模型代码报错或自然完成。

尝试对主线用原配置 `--resume` 恢复时，`EMAHook` 在 checkpoint 加载阶段报 `KeyError: decoder.layers.0.cross_attn.sampling_offsets.weight`；禁用 EMAHook 后继续 `--resume` 又因 optimizer 参数组数量不一致失败。为完成主线最终验证，改用最小补救方案：从原主线 `epoch_66.pth` 作为 `load_from`，禁用 EMAHook，训练 6 个 epoch 并在第 6 个 epoch 做一次 gap=1 正式验证。该方案不恢复 optimizer/EMA 状态，但保留 epoch 66 模型权重、相同数据、相同 PairDN、相同验证指标；原 epoch 67 日志显示 LR 仍为 `1e-4`，因此续训的优化器重置风险可接受，并已在结论中单独标注。

新增临时续训配置：

`projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_cont_epoch66_to72.py`

续训 workdir：

`/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_cont_epoch66_to72`

第一次启动后发现继承了原 2000-iter warmup，LR 从 `2.5e-6` 起步，不适合作为 6-epoch continuation；该短跑在约 100 iter 后已终止并废弃。随后将 continuation 配置的 `param_scheduler=[]`，使用恒定 `1e-4` 重新启动到独立 workdir：

`/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_cont_epoch66_to72_lr1e4`

有效启动状态：2026-06-27 13:45 已成功加载原主线 `epoch_66.pth`，第 1 epoch `[50/484]` 与 `[100/484]` 日志均为 `base_lr=1.0000e-04`，GPUs 0,1 正常训练。gap1-only 对照已证明不优于主线，暂不优先恢复；GPU 2,3 保持空闲备用。

### 2026-06-27 14:52 监督记录：主线 lr1e-4 continuation 完成

主线 continuation 已完成 6 个 epoch 续训并完成 gap=1 正式验证，进程正常退出，4 张 GPU 均已空闲。

- 有效 workdir：`/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_cont_epoch66_to72_lr1e4`
- checkpoint：`epoch_6.pth`，约 253 MB，保存时间 2026-06-27 14:23。
- curves：`20260627_134503/vis_data/curves/`，`TrainingCurveHook` 已保存 overview、learning_rate、total_loss、grad_norm、loss_components、validation_map、loss_vs_val_epoch。
- val_vis：`val_vis/` 已生成。

该 run 是从原主线 `epoch_66.pth` 加载模型权重后的 continuation，不是严格 `--resume`：optimizer、EMA、scheduler 没有从原 run 连续恢复。因此它只能作为“权重续训补救结果”，不能等同于无中断训练的 epoch 72。

| 验证点 | independent AP50 | independent mAP50:95 | pair AP50 | pair mAP50:95 | association gap AP50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原主线 epoch 66 | 0.3322 | 0.1459 | 0.2316 | 0.0902 | 0.1005 |
| continuation epoch 6 | 0.2959 | 0.1339 | 0.2175 | 0.0925 | 0.0784 |

结论：continuation 没有带来 AP50 提升。独立 AP50 从 0.3322 降到 0.2959，pair AP50 从 0.2316 降到 0.2175；pair mAP50:95 从 0.0902 小幅到 0.0925，但幅度很小，且存在 optimizer/EMA 重置影响。不能将该续训视为主线已继续改善。当前最可信的正式主线最高点仍是原 epoch 66：independent AP50 0.3322、pair AP50 0.2316、pair mAP50:95 0.0902。

评估耗时继续偏长：验证 forward 在 14:30:15 到达 `[2700/2708]`，最终指标行在 14:48:30 写出，最后聚合耗时约 18 分钟。期间日志静默，但两个 rank CPU 接近满载，GPU1 有占用；最终正常写出指标，说明不是死锁。后续正式训练若需要更高频验证，应优先优化 `HSMOTPairAPMetric` 的 pair AP/independent AP 聚合或至少加入 metric 阶段进度日志。

gap1-only 对照未恢复：它在 epoch 54 指标显著低于主线 epoch 54/60/66，且严格 resume 会遇到同类 EMA/optimizer 状态问题。除非后续需要专门补齐 ablation 的训练曲线，否则不建议优先占用 GPU 恢复该对照。

### 2026-06-27 14:57 监督记录：启动低 LR continuation 诊断

由于 `1e-4` continuation 从原 `epoch_66.pth` 出发后 AP50 下降，下一步做最小变量诊断：保留相同权重、数据、PairDN、验证指标、6 epoch 续训、无 EMAHook、无 scheduler，仅将重置 optimizer 的基础 LR 从 `1e-4` 降到 `2e-5`。目的不是替代严格 resume，而是验证前一轮退化是否与 optimizer 重置后的步长偏大有关。

新增配置：

`projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_cont_epoch66_to72_lr2e5.py`

启动信息：

- 启动时间：2026-06-27 14:55。
- GPU：0,1；GPU 2,3 保持空闲备用。
- launcher PID：`16556`；torch launcher/ranks：`16565`、`16625/16626`。
- workdir：`/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_cont_epoch66_to72_lr2e5`
- 外部日志：`/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_cont_epoch66_to72_lr2e5.launch.log`
- 已确认加载原主线 `epoch_66.pth`。
- 已确认第 1 epoch `[50/484]` 日志为 `base_lr=2.0000e-05 lr=2.0000e-05`，显存约 16 GB/卡，iter time 约 0.77s，无启动错误。

预期：约 38-45 分钟完成 6 epoch 训练，然后进入 gap=1 全量验证；最终 AP 聚合可能仍需额外约 18-20 分钟。该实验的关键验收是与原 epoch66 和 `1e-4` continuation 的 AP 对比，而不是训练 loss 单独下降。

### 2026-06-27 17:13 监督记录：低 LR continuation 完成、resume 修复并恢复两条训练线

低 LR continuation 已完成 6 个 epoch 训练和 gap=1 验证，正常退出，曲线与可视化已写出：

- workdir：`/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_cont_epoch66_to72_lr2e5`
- checkpoint：`epoch_6.pth`，保存时间 2026-06-27 15:33。
- curves：`20260627_145600/vis_data/curves/`
- val_vis：`val_vis/`

| 验证点 | independent AP50 | independent mAP50:95 | pair AP50 | pair mAP50:95 | association gap AP50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原主线 epoch 66 | 0.3322 | 0.1459 | 0.2316 | 0.0902 | 0.1005 |
| continuation lr1e-4 epoch 6 | 0.2959 | 0.1339 | 0.2175 | 0.0925 | 0.0784 |
| continuation lr2e-5 epoch 6 | 0.3449 | 0.1562 | 0.2410 | 0.0998 | 0.1040 |

结论：低 LR continuation 明显优于 `1e-4` continuation，并小幅超过原主线 epoch 66。说明前一轮 AP50 退化主要与 optimizer 重置后的步长偏大有关；在无法严格恢复 optimizer/EMA 的情况下，`2e-5` 是更合理的权重续训补救 LR。不过该 run 仍不是严格 resume，不能替代完整无中断训练结果。

resume 失败根因已修复。此前 `--resume` 会在正式配置含有 `load_from=pair_adapted_pretrain.pth` 时触发 MMEngine 的“从 `load_from` 恢复”路径，因此实际恢复的是预训练 checkpoint，而不是 workdir 的 `last_checkpoint`；这会导致 EMAHook 处理不完整预训练权重时报 `KeyError`，禁用 EMA 后又因预训练 checkpoint optimizer param group 为 330、当前模型为 405 而报 optimizer group mismatch。已修改 `tools/train.py`：只有命令行显式 `--resume` 时设置 `cfg.resume=True` 并清空 `cfg.load_from=None`，使其按帮助文本从 `work_dir/last_checkpoint` 自动恢复；同时不再把配置内的 `resume=True` 无条件覆盖成 False。

修复验证：

- `python -m py_compile tools/train.py` 通过。
- 主线 strict resume 日志：`resume_fixed_20260627_1716.log`
  - `Auto resumed from .../o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn/epoch_66.pth`
  - `resumed epoch: 66, iter: 31944`
  - 已进入 `Epoch(train) [67]`，LR 为 `1.0000e-04`，无 EMA/optimizer 报错。
- gap1-only strict resume 日志：`resume_fixed_20260627_1717.log`
  - `Auto resumed from .../o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train/epoch_54.pth`
  - `resumed epoch: 54, iter: 26136`
  - 启动阶段无 resume 报错。

当前运行状态：

- 主线正式 strict resume：GPUs 0,1，launcher PID `22779`，torch launcher PID `22782`，从 epoch 66 继续到 max epoch 72；预计约 40 分钟训练后进入最终验证，验证聚合仍可能额外耗时约 20 分钟。
- gap1-only 对照 strict resume：GPUs 2,3，launcher PID `23227`，torch launcher PID `23230`，从 epoch 54 继续到 max epoch 72；用于补齐 gap1-only 对照曲线和最终指标。

下一步验收重点：等待主线 strict resume 的 epoch 72 验证结果，优先与原 epoch 66、lr2e-5 continuation 做 AP 对比；同时观察 gap1-only 是否在 epoch 60/66/72 继续落后主线。如果主线 strict resume 不如 lr2e-5 continuation，需要判断 EMA 权重、scheduler 状态和 LR 对最终 AP 的影响，再考虑在空闲 GPU 上做更低 LR 或 EMA-free 的最小变量探索。

### 2026-06-27 19:00 监督记录：strict resume 结果、指标优化与下一轮探索

本轮先修复并扩展 `tools/train.py` 的 resume 语义。此前只修正了 `--resume` 自动恢复时清空 `load_from` 的问题；现在进一步支持 `--resume /path/to/checkpoint.pth`，用于把指定 checkpoint 严格恢复到新的 work_dir。语义如下：

- `--resume`：从当前 `work_dir/latest_checkpoint` 自动恢复，且清空配置里的预训练 `load_from`。
- `--resume /path/to/epoch_x.pth`：设置 `resume=True + load_from=指定 checkpoint`，恢复模型、optimizer、scheduler、EMA 等训练状态。

语法检查已通过：

- `python -m py_compile tools/train.py projects/multispec_pair_rotated_rtdetr/multispec_pair_rotated_rtdetr/overfit_ap.py projects/multispec_pair_rotated_rtdetr/multispec_pair_rotated_rtdetr/pair_overfit_metric.py`

正式验证指标侧做了最小工程优化：

- `HSMOTPairAPMetric` 默认 `max_dets=100`，采用 COCO 风格每图 top-100 检测进入 AP，避免全量 query 参与 AP 聚合。
- `independent_ap_metrics` 复用已计算的 `pair_AP50`，不再为了 `association_gap_AP50` 重复计算 pair AP。
- 当 `report_gaps=(1,)` 且验证集本身全是 gap=1 时，gap1 指标直接复用 full 指标，避免同一批样本重复 AP 计算。

这改变了历史 all-det AP 的口径，后续正式 full-val 对比应使用当前 maxDets=100 口径；历史 all-det 数值只能作为趋势参考。

主线与 gap1-only checkpoint 已用 1GPU 重新完成 full half gap=1 验证。验证 forward 约 12 分钟，最后 16 个样本加 AP 聚合约 4.5 分钟，已可接受但仍偏慢。结果如下：

| 验证点 | independent AP50 | independent mAP50:95 | pair AP50 | pair mAP50:95 | association gap AP50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 主线 strict resume epoch 72, maxDets=100 | 0.3326 | 0.1512 | 0.2323 | 0.0947 | 0.1003 |
| gap1-only strict resume epoch 60, maxDets=100 | 0.2565 | 0.1071 | 0.1380 | 0.0465 | 0.1185 |

结论：

- 主线随机间隔训练 `(1,5)` 在 gap=1 验证上明显优于 gap1-only 训练；当前不建议把正式训练改成 gap1-only。
- 主线 strict resume epoch72 与原 epoch66/低 LR continuation 同量级，pair AP50 约 0.23、independent AP50 约 0.33，说明恢复后的训练没有崩坏；但由于当前 maxDets=100 口径与早期 all-det 口径不同，不能直接按小数点后三位比较。
- gap1-only 的检测和关联都明显较弱，且 `association_gap_AP50` 更大，说明它不是当前主线的更优采样策略。

已启动新探索实验：主线 epoch72 严格恢复并延长到 epoch84。

- 启动时间：2026-06-27 18:58。
- GPU：0,2；GPU3 被其他用户 `tianyuyang01` 占用，GPU1 空闲备用。
- launcher PID：`48380`；torch launcher PID：`48383`。
- workdir：`/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_strictresume_epoch72_to84`
- 外部日志：`/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_strictresume_epoch72_to84.log`
- 启动命令使用 `--resume /data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn/epoch_72.pth` 和 `--cfg-options train_cfg.max_epochs=84 default_hooks.checkpoint.max_keep_ckpts=4`。
- 已确认日志出现 `resumed epoch: 72, iter: 34848`。
- 已确认进入 `Epoch(train) [73][50/484]`，`lr=1e-4`，loss 约 `20.9859`，grad_norm 约 `21.8075`，无启动错误。

下一步：持续监督 epoch78 验证指标。如果 epoch78 AP 高于 epoch72，则继续到 epoch84；如果 epoch78 明显下降，则优先考虑停止该 run，并在空闲卡上做低 LR 严格/半严格续训对照。

### 2026-06-27 19:12 监督记录：并行低 LR load-from 对照启动

strict resume 主线继续稳定运行，已到 `Epoch(train) [74]`，loss 和 grad_norm 与 epoch72 附近同量级，未见异常。

为提前判断“继续训练退化是否主要来自 LR/optimizer 状态”，在 GPU1,3 上启动低 LR load-from 对照。该 run 不是严格 resume：从主线 `epoch_72.pth` 加载模型权重，但 optimizer、scheduler、EMA 状态重置；固定 LR 为 `2e-5`，训练 12 epoch，验证点对应 load-from 后第 6/12 epoch。

先启动过一次错误口径：

- workdir：`.../o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_loadfrom_epoch72_to84_lr2e5`
- 问题：继承了 base config 的 2000-iter `LinearLR` warmup，首个日志实际 `lr=5.0975e-07`，不是目标 `2e-5`。
- 处理：已停止该 run，不纳入实验结论。

已新增修正配置：

`projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_loadfrom_epoch72_to84_lr2e5.py`

修正点：

- `load_from=/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn/epoch_72.pth`
- `optim_wrapper.optimizer.lr = 2e-5`
- `param_scheduler = []`
- `max_epochs = 12`
- workdir：`/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_loadfrom_epoch72_to84_lr2e5_nosched`

启动信息：

- 启动时间：2026-06-27 19:10。
- GPU：1,3。
- launcher PID：`54904`；torch launcher PID：`54907`。
- 外部日志：`/data/users/litianhao01/PairMmot/workdir/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_loadfrom_epoch72_to84_lr2e5_nosched.log`
- 已确认加载 `epoch_72.pth`。
- 已确认首个训练日志 `Epoch(train) [1][50/484]` 为 `base_lr=2.0000e-05 lr=2.0000e-05`，loss `20.2417`，grad_norm `23.5565`。

后续比较口径：

- strict resume epoch78/84：真实延续训练状态，LR=1e-4。
- load-from lr2e-5 epoch6/12：权重续训补救策略，optimizer/EMA 重置，不作为严格公平主结论，但可判断低 LR 是否能避免继续训练退化。
