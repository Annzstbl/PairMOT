# 20260709 Decoder Experiment HOTA Report

## 1. 对比对象

本报告汇总 decoder 系列实验，重点看 HOTA。AP 不作为主要判断依据。

Baseline 使用 252 上补齐的高指标 `0704_01 resume`，而不是早期手工记录的低指标。

| 实验 | 角色 | 服务器/路径 |
|---|---|---|
| `0704_01 resume` | 非 tri-state decoder baseline，`unique + PairDN + all-GT` | `workdir_252/0704_01...resume_from_epoch40_to72` |
| `0708_01` 252 | tri-state decoder 原始实验 | `workdir_252/0708_01...tristate_decoder` |
| `0708_01` 99 rerun | tri-state decoder 原始实验本机重跑 | `workdir_99/0708_01...tristate_decoder` |
| `0708_02` 197 | tri-state decoder + separated FFN | `workdir_197/0708_02...tristate_decoder_sepffn` |
| `0708_03` 99 | tri-state decoder + recurrent coupling zero-init | `workdir_99/0708_03...tristate_decoder_zeroinit` |
| `0708_04` 99 | `0708_03` + separated FFN | `workdir_99/0708_04...tristate_decoder_sepffn_zeroinit` |

选择规则：

- 不合并不同 epoch 的单项最优。
- 每个实验统一按 `cls_HOTA + det_HOTA` 选择唯一最佳 epoch。
- 重点比较 `cls_HOTA`、`det_HOTA` 和两者之和；MOTA/IDF1 只用于解释。

## 2. 实验内容与结构改动

所有 decoder 实验都继承 `0704_01` 的 proposal、matching、PairDN、loss、all-GT 监督和验证设置。也就是说，这组实验只讨论 pair decoder 内部结构变化，不把训练目标或 proposal 机制混进结论。

结构关系：

```text
0704_01 resume
  -> 0708_01 tri-state decoder
      -> 0708_02 tri-state decoder + separated FFN
      -> 0708_03 tri-state decoder + zero-init recurrent coupling
          -> 0708_04 tri-state decoder + zero-init recurrent coupling + separated FFN
```

| 实验 | 相对来源 | 具体改动 | 设计意图 | 不变项 |
|---|---|---|---|---|
| `0704_01 resume` | `0704_01` 原实验从 epoch 40 续训到 72 | 无 decoder 新结构；作为 `unique + PairDN + all-GT` 高指标 baseline | 给 decoder 系列提供同训练长度、同监督目标的强 baseline | proposal、matching、PairDN、loss、all-GT |
| `0708_01` | `0704_01 resume` | 开启 `tristate_decoder=True`，把 pair decoder 从单一 shared-query 状态改成 `pointer / query_prev / query_curr` 三状态 | 让 pair decoder 显式区分关联指针和前后帧目标查询，减少一个 query 同时承担两帧表达的冲突 | proposal、matching、PairDN、loss、all-GT |
| `0708_02` | `0708_01` | 开启 `tristate_separate_ffn=True`，在 frame-specific cross-attention 后将 prev/curr FFN 解耦 | 给前后帧查询各自独立的后处理容量，避免 prev/curr 在 FFN 中过度共享 | tri-state 主结构、proposal、matching、PairDN、loss、all-GT |
| `0708_03` | `0708_01` | 开启 `tristate_zero_init_coupling=True`，将新增的 pointer-to-frame 和 frame-to-pointer recurrent coupling 分支零初始化 | 保留 tri-state recurrent coupling 的可学习能力，同时避免训练初期被随机初始化的新耦合分支扰动 | tri-state 主结构、proposal、matching、PairDN、loss、all-GT |
| `0708_04` | `0708_03` + `0708_02` | 同时开启 `tristate_zero_init_coupling=True` 和 `tristate_separate_ffn=True` | 结合 0708_03 的稳定起点和 0708_02 的 prev/curr 独立容量，测试二者是否互补 | proposal、matching、PairDN、loss、all-GT |

两份 `0708_01` 的含义：

- `0708_01` 252 是早先在 252 上完成的原始 tri-state decoder 实验。
- `0708_01` 99 rerun 是本机重新跑完的同配置实验，用于观察裸 tri-state 的重复运行稳定性。
- 两者结构相同，但训练运行不同，因此报告中分开列出，不合并指标。

## 3. 唯一最佳点

| 实验 | 最佳 epoch | cls HOTA | cls MOTA | cls IDF1 | det HOTA | det MOTA | det IDF1 | cls HOTA + det HOTA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `0704_01 resume` | 67 | 45.523 | 34.750 | 52.845 | 58.120 | 51.956 | 66.997 | 103.643 |
| `0708_01` 252 | 71 | 45.236 | 34.483 | 52.236 | 58.217 | 52.597 | 67.343 | 103.453 |
| `0708_01` 99 rerun | 59 | 46.044 | 34.181 | 53.680 | 58.052 | 52.686 | 67.302 | 104.096 |
| `0708_02` 197 | 71 | 46.275 | 34.657 | 53.829 | 58.236 | 52.175 | 67.216 | 104.511 |
| `0708_03` 99 | 71 | 46.271 | 34.953 | 53.747 | 58.360 | 52.888 | 67.482 | 104.631 |
| `0708_04` 99 | 71 | 45.787 | 35.208 | 53.443 | 58.373 | 53.493 | 67.613 | 104.160 |

相对 `0704_01 resume`：

| 实验 | delta cls HOTA | delta det HOTA | delta sum | delta cls IDF1 | delta det IDF1 |
|---|---:|---:|---:|---:|---:|
| `0708_01` 252 | -0.287 | +0.097 | -0.190 | -0.609 | +0.346 |
| `0708_01` 99 rerun | +0.521 | -0.068 | +0.453 | +0.835 | +0.305 |
| `0708_02` 197 | +0.752 | +0.116 | +0.868 | +0.984 | +0.219 |
| `0708_03` 99 | +0.748 | +0.240 | +0.988 | +0.902 | +0.485 |
| `0708_04` 99 | +0.264 | +0.253 | +0.517 | +0.598 | +0.616 |

按 `cls_HOTA + det_HOTA` 排序：

| rank | 实验 | epoch | sum |
|---:|---|---:|---:|
| 1 | `0708_03` 99 | 71 | 104.631 |
| 2 | `0708_02` 197 | 71 | 104.511 |
| 3 | `0708_04` 99 | 71 | 104.160 |
| 4 | `0708_01` 99 rerun | 59 | 104.096 |
| 5 | `0704_01 resume` | 67 | 103.643 |
| 6 | `0708_01` 252 | 71 | 103.453 |

## 4. HOTA 结论

### 4.1 最佳 decoder 版本

`0708_03` 是当前 decoder 系列的最佳点。相对 `0704_01 resume`：

- `cls_HOTA +0.748`
- `det_HOTA +0.240`
- 综合分 `+0.988`
- `cls_IDF1 +0.902`
- `det_IDF1 +0.485`

这说明 zero-init recurrent coupling 是有效改动，而且收益不只来自 det 侧；它同时改善 cls-side 和 det-side HOTA。

### 4.2 `0708_02` 很接近，但略低于 `0708_03`

`0708_02` 是 197 上的 separated FFN 版本，没有 zero-init recurrent coupling。它达到：

- `cls_HOTA=46.275`，全表最高，略高于 `0708_03` 的 `46.271`
- `det_HOTA=58.236`，低于 `0708_03` 的 `58.360`
- 综合分 `104.511`，比 `0708_03` 低 `0.120`

因此如果只看 cls HOTA，`0708_02` 非常强；但按统一规则 `cls_HOTA + det_HOTA`，`0708_03` 仍是更稳的代表点。

### 4.3 `0708_04` 的 det 侧最好，但 cls 侧掉太多

`0708_04` 在 det 侧很强：

- `det_HOTA=58.373`，全表最高
- `det_MOTA=53.493`，全表最高
- `det_IDF1=67.613`，全表最高

但它的 `cls_HOTA=45.787`，比 `0708_03` 低 `0.484`，也比 `0708_02` 低 `0.488`。由于本阶段更关心整体 HOTA，`0708_04` 不适合作为主线替代 `0708_03`。它更像是 det-side 诊断方向，而不是综合最优 decoder。

### 4.4 原始 tri-state decoder 有波动

`0708_01` 有两份结果：

- 252：sum `103.453`，低于 baseline `-0.190`
- 99 rerun：sum `104.096`，高于 baseline `+0.453`

这说明原始 tri-state decoder 不是稳定强收益。它可能有正向信号，但对随机性、机器或训练过程较敏感；后续不建议把裸 `0708_01` 当作可靠主线。真正稳定的提升来自后续结构约束，尤其是 `0708_03`。

## 5. 训练曲线观察

末段 HOTA 趋势：

| 实验 | epoch 59 sum | epoch 63 sum | epoch 67 sum | epoch 71 sum | 最佳点 |
|---|---:|---:|---:|---:|---|
| `0708_01` 99 rerun | 104.096 | 103.876 | 103.871 | 103.756 | epoch 59 |
| `0708_02` 197 | 103.681 | 104.164 | 104.389 | 104.511 | epoch 71 |
| `0708_03` 99 | 103.739 | 104.073 | 104.516 | 104.631 | epoch 71 |
| `0708_04` 99 | 103.914 | 103.760 | 104.090 | 104.160 | epoch 71 |

观察：

- `0708_02` 和 `0708_03` 后期仍在上升，最终 epoch 71 是最佳点。
- `0708_03` 从 epoch 63 到 71 持续提升，说明 zero-init recurrent coupling 没有造成后期退化。
- `0708_01` 99 rerun 在 epoch 59 达峰，后期略回落，进一步说明裸 tri-state 不够稳定。
- `0708_04` 后期也提升，但 cls 侧上限不足，综合分没有追上 `0708_03`。

## 6. 最终建议

1. decoder 系列主线推荐 `0708_03 epoch_71`，对应训练保存 checkpoint `epoch_72.pth`。
2. `0708_02` 值得保留为 cls-side 强对照；它和 `0708_03` 的差距只有 `0.120`，但 det HOTA 稍弱。
3. `0708_04` 不建议作为主线继续投入；它证明 separated FFN 能强化 det 侧，但会压低 cls HOTA。
4. 裸 `0708_01` 不应作为结论点；两次运行一正一负，稳定性不足。
5. 后续 decoder 改动应优先沿 `0708_03` 做小步实验；若想利用 `0708_04` 的 det-side 优势，需要设计机制避免 cls-side HOTA 损失。
