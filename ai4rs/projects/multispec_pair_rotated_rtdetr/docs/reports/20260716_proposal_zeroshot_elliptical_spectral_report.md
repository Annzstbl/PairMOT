# Proposal Zero-shot Report: Elliptical Motion + Spectral Similarity

更新时间：2026-07-16 CST

## 1. 任务定位

本报告独立记录 pair proposal affinity 的 zero-shot 改进。它不改变训练模型，不增加可训练
参数，也不属于 full-data baseline 后续 `+long-tail`、`+liquid`、`+encoder`、`+decoder`
模块消融主线。

统一对照为 full-data baseline：

| 项目 | 内容 |
|---|---|
| 实验 | `0714_01_0704_resume_coco365_full_unique_allgt` |
| checkpoint | `/data4/litianhao/PairMmot/workdir_252/0714_01_0704_resume_coco365_full_unique_allgt/epoch_72.pth` |
| cls HOTA | 52.374 |
| det HOTA | 60.318 |
| pair mAP | 0.2928 |
| pair AP50 | 0.5062 |

所有 zero-shot 实验加载同一个 checkpoint。现有 `pre_topk=900`、
`candidate_topk=1800`、`max_center_dist=0.18`、单侧可见候选保留、unique pair
selection 和最终 300 query top-k 均未改变，只调整 pair proposal 候选之间的 affinity。

## 2. 第一版组合：`0715_07`

`0715_07_full_baseline_elliptical_spectral_zeroshot` 包含两项改动：

- `elliptical motion`：先用真实 GMC 将前帧候选中心映射到后帧，再把残余位移投影到前帧
  旋转框的长轴和短轴。长轴方向允许更大的标准差，短轴方向施加更强惩罚；长短轴尺度由
  proposal 的像素长宽比生成，方形目标自然退化为接近各向同性。
- `spectral similarity`：不构建全分辨率 spectral-angle map，也不对 900 x 900 box pair
  分别做 ROIAlign。每个候选框只在中心及正负宽轴、正负高轴共 5 个内部点进行一次
  batched `grid_sample`，恢复这部分 8-band 原始值后平均、归一化，再用矩阵乘得到候选间
  光谱余弦相似度。
- affinity 权重保持总和为 1：encoder query similarity `0.15`、elliptical geometry
  `0.50`、classification prior `0.25`、spectral similarity `0.10`。因此原
  `affinity_thr=0.15` 的数值尺度基本保持不变。

### 2.1 Tracking 结果

| experiment | cls HOTA | det HOTA | cls MOTA | cls IDF1 | det MOTA | det IDF1 |
|---|---:|---:|---:|---:|---:|---:|
| full baseline | 52.374 | 60.318 | 44.159 | 62.126 | 57.407 | 70.957 |
| `0715_07` elliptical + spectral | 52.780 | 60.244 | 44.635 | 62.662 | 57.531 | 70.806 |
| delta | +0.406 | -0.074 | +0.476 | +0.536 | +0.124 | -0.151 |

用于唯一方案选择的 `cls_HOTA + det_HOTA` 从 `112.692` 变为 `113.024`，提高
`0.332`。各项指标仍独立展示，不将 cls 和 det 指标合并为一个性能指标。

### 2.2 AP 结果

| experiment | pair mAP | pair AP50 | both mAP | both AP50 |
|---|---:|---:|---:|---:|
| full baseline | 0.2928 | 0.5062 | 0.3011 | 0.5209 |
| `0715_07` elliptical + spectral | 0.2952 | 0.5105 | 0.3034 | 0.5251 |
| delta | +0.0024 | +0.0043 | +0.0023 | +0.0042 |

### 2.3 分类 HOTA 变化

| class | baseline | `0715_07` | delta |
|---|---:|---:|---:|
| awning-bike | 45.040 | 45.361 | +0.321 |
| bike | 41.266 | 42.698 | +1.432 |
| bus | 71.302 | 71.635 | +0.333 |
| car | 80.004 | 79.765 | -0.239 |
| pedestrian | 42.192 | 42.076 | -0.116 |
| tricycle | 37.831 | 38.080 | +0.249 |
| truck | 39.610 | 40.812 | +1.202 |
| van | 61.745 | 61.816 | +0.071 |

### 2.4 第一版结论与效率

固定张量微基准使用 batch 4、每帧 900 proposals、8 x 800 x 1200 输入：旧 affinity
每个样本 `0.703 ms`，新 affinity `1.240 ms`；两帧光谱描述子整个 batch 为
`0.689 ms`，估算每 batch 总增量 `2.838 ms`。真实测试约 `0.20 s/iter`，新增部分约
占 `1.4%`，测试显存稳定为 `4168 MiB`。

第一版组合提高了 cls HOTA、方案选择分数和 AP，主要增益来自 `bike`、`truck`；但
det HOTA 和 det IDF1 轻微下降，说明统一 ellipse 和绝对光谱约束对部分真实 pair 偏紧。
组合结果无法区分 motion 和 spectrum 的独立贡献，因此继续进行独立 zero-shot 消融。

结果路径：

`/data4/litianhao/PairMmot/workdir_99/0715_07_full_baseline_elliptical_spectral_zeroshot`

## 3. 独立消融与深入优化

| zero-shot variant | cls HOTA | det HOTA | pair mAP | 结论 |
|---|---:|---:|---:|---|
| full baseline | 52.374 | 60.318 | 0.2928 | 对照 |
| elliptical motion only | 52.769 | 60.377 | 0.2951 | HOTA 与 AP 的主要正贡献 |
| raw spectral only | 52.286 | 60.193 | 0.2932 | 原始绝对光谱余弦是负贡献 |
| relative spectral conservative | 52.636 | 60.439 | 0.2944 | 相对残差优于绝对余弦，但 ellipse 过度减弱 |
| relative spectral enhanced | 52.780 | 60.691 | 0.2948 | 提高 geometry/rank 权重明显改善 det HOTA |
| class-aware rank 0.25 | 52.892 | 60.736 | 0.2954 | cls 达到 `+0.5`，det 尚未达到 |
| `0715_08` class-aware rank 0.30 | 52.921 | 60.876 | 0.2953 | 类别门控诊断上界，不作为最终通用方案 |
| size-aware area `1.0e-3` | 52.683 | 60.903 | 0.2935 | 回退范围过大，削弱 ellipse 收益 |
| `0716_01` size-aware area `3.5e-4` | 52.886 | 60.942 | 0.2947 | 无类别特判，两项 HOTA 均超过 baseline `0.5` |

独立消融说明：elliptical motion 本身能够同时提高 cls HOTA 和 det HOTA；raw spectral
absolute cosine 会退化。类别门控能够验证 motion 与 spectrum 的适用条件，但不能作为
最终通用方法。最终将适用条件改写为仅依赖 proposal 几何面积的尺度门控。

## 4. GT 运动统计

GT motion 统计使用 test 集 50 个序列、真实 GMC 和 180387 个相邻帧同轨迹目标。表中
`P(long>short)` 表示 GMC 补偿后，沿目标长轴的绝对位移大于短轴位移的比例：

| class | pairs | P(long>short) | long motion p90 (px) | short motion p90 (px) |
|---|---:|---:|---:|---:|
| car | 67243 | 0.658 | 10.76 | 1.21 |
| bike | 9326 | 0.654 | 21.60 | 1.82 |
| pedestrian | 85390 | 0.565 | 4.10 | 3.42 |
| van | 6114 | 0.660 | 10.79 | 1.21 |
| truck | 2183 | 0.750 | 46.61 | 1.78 |
| bus | 1602 | 0.801 | 20.33 | 1.60 |
| tricycle | 2489 | 0.614 | 6.44 | 1.40 |
| awning-bike | 6040 | 0.654 | 16.23 | 1.92 |

类别统计只用于分析，不用于模型分支。方向性较弱的目标通常也是旋转角不稳定的小目标，
因此最终方案改用归一化面积判断 motion 可靠性。test GT 面积中位数为 `3.46e-4`，60 分位
为 `1.02e-3`；zero-shot 对比表明中位数附近的 `3.5e-4` 阈值优于 `1.0e-3`。

## 5. 类别门控诊断：`0715_08`

`0715_08` 曾按类别选择 isotropic fallback 和 spectral correction，用于确认不同线索的
适用范围。它取得 `cls_HOTA=52.921`、`det_HOTA=60.876`，但类别特判不满足最终方法的
通用性要求，因此只保留为诊断上界，不再作为正式方案。

## 6. 最终通用方案：`0716_01`

最终方案不读取类别来选择 motion 或 spectrum，只使用归一化 box 面积：

- 前帧 proposal 面积 `w*h <= 3.5e-4` 时，旋转角和长短轴方向视为不可靠，motion 回退到
  baseline isotropic geometry；大于阈值时使用 elliptical motion；
- pair 尺寸定义为两侧归一化面积的几何平均
  `sqrt((w_p*h_p)*(w_c*h_c))`；仅当 pair area `<= 3.5e-4` 时启用 relative spectral；
- 光谱描述子仍使用 5 点 median pooling、raw spectrum 与 log-chromaticity；
- relative spectral 仍只在中心距离合法且两侧预测类别一致的候选集合内做 row/column
  中心化。这里的类别一致性是 baseline 的通用 matching constraint，不按类别改变算法；
- proposal quality rank weight 为 `0.70`，pair affinity rank weight 为 `0.30`；原有
  pre-topk、candidate-topk、hard center gate、单侧候选保留和 unique selection 不变。

### 6.1 Tracking 结果

正式结果使用同一个 full baseline `epoch_72.pth`，无训练、无新增参数：

| experiment | cls HOTA | det HOTA | cls MOTA | cls IDF1 | det MOTA | det IDF1 |
|---|---:|---:|---:|---:|---:|---:|
| full baseline | 52.374 | 60.318 | 44.159 | 62.126 | 57.407 | 70.957 |
| `0716_01` | 52.886 | 60.942 | 44.573 | 62.921 | 57.992 | 72.056 |
| delta | +0.512 | +0.624 | +0.414 | +0.795 | +0.585 | +1.099 |

用于唯一方案选择的 `cls_HOTA + det_HOTA` 从 `112.692` 提高到 `113.828`，变化
`+1.136`。cls HOTA 与 det HOTA 均实现超过 `0.5` 的 zero-shot 提升；相对类别门控
`0715_08`，cls HOTA 低 `0.035`，det HOTA 高 `0.066`。

### 6.2 AP 结果

| experiment | pair mAP | pair AP50 |
|---|---:|---:|
| full baseline | 0.2928 | 0.5062 |
| `0716_01` | 0.2947 | 0.5094 |
| delta | +0.0019 | +0.0032 |

### 6.3 分类 HOTA 变化

| class | baseline | `0716_01` | delta |
|---|---:|---:|---:|
| awning-bike | 45.040 | 45.604 | +0.564 |
| bike | 41.266 | 42.600 | +1.334 |
| bus | 71.302 | 71.605 | +0.303 |
| car | 80.004 | 80.543 | +0.539 |
| pedestrian | 42.192 | 42.971 | +0.779 |
| tricycle | 37.831 | 37.636 | -0.195 |
| truck | 39.610 | 40.071 | +0.461 |
| van | 61.745 | 62.060 | +0.315 |

七个类别提升，tricycle 下降 `0.195`。该结果不再由类别白名单塑造；小目标统一回退后，
pedestrian 提升 `0.779`，同时保留了 bike、truck 等方向性目标的主要收益。

### 6.4 计算效率

最终结构固定张量微基准中，batch 4 的两帧 mixed spectral descriptor 为 `0.976 ms`，
每样本 relative affinity 为 `1.739 ms`，descriptor 加 4 个样本 affinity 总计
`7.932 ms/batch`。相对旧 affinity 路径约增加 `5.1 ms/batch`，占真实
`0.18-0.20 s/iter` 的约 `2.5%`；测试显存仍为 `4168 MiB`。

面积门控只增加逐 proposal 和 pair 的 GPU 张量比较，不引入 CPU 分支。实测测试显存仍为
`4168 MiB`，单 batch 约 `0.18-0.20 s`，与上一版相同。

## 7. 配置与结果路径

正式配置：

`projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_coco365_full_sizeaware_elliptical_spectral_rank30_zeroshot_99.py`

正式结果：

`/data4/litianhao/PairMmot/workdir_99/0716_01_full_sizeaware_elliptical_spectral_rank30_zeroshot`

## 8. 结论

最终 `0716_01` 在不训练、不增加模型参数、不按类别改变处理方法、不改变 top-k 候选机制
的前提下，同时提升 cls HOTA `+0.512` 和 det HOTA `+0.624`。有效设计是用目标尺度表达
旋转方向的可靠性：小目标使用 isotropic motion，并用相对光谱补充判别；较大目标使用
elliptical motion。该方案作为后续 proposal affinity 研究的独立 baseline，不计入
full-data 模块逐项训练消融链。
