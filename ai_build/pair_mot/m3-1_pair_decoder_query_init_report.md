# M3-1 Pair Decoder Query 初始化修正报告

> **文档性质**：在 [m3_pair_decoder_report.md](./m3_pair_decoder_report.md)（M3j Pair Decoder 模块）之后，**修正 M5 接入阶段与 M3 设计不一致的问题**：过拟合验收应使用 **与图像内容无关的 learnable Pair token + 双 learnable reference**，Encoder Top-K 初始化推迟到独立后续里程碑。  
> 本报告同时引入 **`query_init` 三模式开关**，用于结构验证、调试定位与最终集成。

| 项 | 内容 |
|----|------|
| 里程碑 | M3-1 — learnable query/reference 命名统一 + `query_init` 路由 |
| 前置文档 | [m3_pair_decoder_report.md](./m3_pair_decoder_report.md)、[m4_pair_head_report.md](./m4_pair_head_report.md) |
| 日期 | 2026-06-17 |
| 仓库 | `/data/users/litianhao01/PairMmot/ai4rs` |
| 原则 | **修正 M5 默认路径，不改动 Pair Decoder 层内 self/cross-attn 结构** |

---

## 1. 问题与动机

M3j 已在 `PairRotatedRTDETRTransformerDecoder` 内实现 learnable placeholder（`nn.Embedding`），但 M5 接入 `MultispecPairRotatedRTDETR` 时 **默认走 `_topk_pair_queries`（Encoder Top-K）**，与 M3 过拟合验证目标不一致：

| 维度 | M3j 设计意图 | M5 接入（修正前） |
|------|-------------|------------------|
| Query content | learnable `query_embedding` | prev memory Top-K gather |
| Reference | learnable 双 embedding | Top-K proposals + reg offset |
| 过拟合用途 | 验证 Decoder 结构 / 梯度 / Head+Matcher | 混入 encoder 内容初始化，难以隔离问题 |
| DN | 不使用 | 已关闭（一致） |
| Encoder Top-K | 留待后续里程碑 | 已提前接入（不一致） |

**修正目标**：过拟合默认使用 **纯 learnable** 初始化；Top-K 仅通过 `query_init='dual_topk'` 显式开启；调试模式 `gt_noised` 用于区分「Query 搜索能力」与「Head/Matcher/Loss」问题。

---

## 2. 代码改动

### 2.1 Decoder 内 embedding 命名（`pair_rotated_rtdetr_layers.py`）

与 DINO / 用户建议命名对齐：

```python
self.query_embedding = nn.Embedding(num_queries, embed_dims)
self.ref_prev_embedding = nn.Embedding(num_queries, 5)
self.ref_curr_embedding = nn.Embedding(num_queries, 5)
```

| 旧名（M3j） | 新名（M3-1） |
|------------|-------------|
| `query_content` | `query_embedding` |
| `reference_prev_embed` | `ref_prev_embedding` |
| `reference_curr_embed` | `ref_curr_embedding` |

Reference embedding 仍存 **inverse_sigmoid 空间**，前向时 `sigmoid()` 与 O2 reference refinement 一致。  
当 `forward(..., query=None, reference_prev=None, reference_curr=None)` 时，Decoder 自动 expand 到 batch。

### 2.2 Detector 级 `query_init` 开关（`multispec_pair_rotated_rtdetr.py`）

新增构造参数：

```python
query_init: Literal['learned', 'gt_noised', 'dual_topk'] = 'learned'
gt_ref_noise_scale: float = 0.02  # 仅 gt_noised 使用
```

`forward_transformer`（`pair_mode=True`）经 `_init_pair_decoder_queries` 路由：

| `query_init` | Query | Reference | 用途 |
|--------------|-------|-----------|------|
| `'learned'`（**默认**） | Decoder `query_embedding` | Decoder 双 `ref_*_embedding` | **过拟合验收**：验证 Pair Decoder + Head + Matcher |
| `'gt_noised'` | 仍用 learnable query | GT pair 归一化 bbox + 高斯噪声；无效侧回退 learnable ref | **调试**：定位 Query 初始化 / cross-attn 搜索 |
| `'dual_topk'` | prev memory Top-K gather | prev/curr 各自 Top-K proposal + reg | **后续集成**：Encoder 内容初始化（原 M5 默认行为） |

**过拟合 config 约定**（`o2_pair_rtdetr_r18vd_overfit.py`）：

```python
model.update(
    type='MultispecPairRotatedRTDETR',
    pair_mode=True,
    query_init='learned',
    num_queries=50,
    dn_cfg=None,
    ...
)
```

### 2.3 `gt_noised` 调试逻辑

1. 从 `batch_data_samples[i].pair_gt_instances` 读取双帧 rbox；
2. 按 `[img_w, img_h, img_w, img_h, angle_factor]` 归一化；
3. 加 `N(0, gt_ref_noise_scale)` 噪声并 clamp 到 `(1e-4, 1-1e-4)`；
4. 写入前 `min(num_gt, num_queries)` 个 slot；`valid_prev/valid_curr=False` 的侧回退 learnable ref；
5. 超出 GT 数的 query slot 保持 learnable ref（背景 query）。

**诊断解读**（与用户需求一致）：

- **learned 不过拟合，gt_noised 能过拟合** → Pair Head / Matcher 大概率正确；问题在 Query 初始化或 cross-attention 搜索。
- **gt_noised 仍不过拟合** → 优先排查 valid mask、Pair Hungarian、坐标归一化、角度编码、reference refinement、loss 实现。

---

## 3. 过拟合验收配置对照

| 项 | 过拟合（M5 验收） | 全量训练（后续） |
|----|------------------|-----------------|
| Pair tokens | learnable | `dual_topk` 或 hybrid |
| Dual references | learnable | `dual_topk` |
| Hungarian | Pair-level | Pair-level |
| DN | **关闭** | 待 `PairCdnQueryGenerator` |
| Encoder Top-K | **关闭**（`query_init='learned'`） | `query_init='dual_topk'` |

---

## 4. 修改文件清单

| 路径 | 说明 |
|------|------|
| `pair_rotated_rtdetr_layers.py` | embedding 重命名 |
| `multispec_pair_rotated_rtdetr.py` | `query_init` / `_gt_noised_pair_queries` / `_init_pair_decoder_queries` |
| `configs/o2_pair_rtdetr_r18vd_overfit.py` | `query_init='learned'` |
| `tests/test_projects/test_pair_rotated_rtdetr_decoder.py` | embedding 命名与 learnable 梯度测试 |

---

## 5. 测试结果

```bash
cd ai4rs && python -m pytest tests/test_projects/test_pair_rotated_rtdetr_decoder.py -v
```

**11 passed**（含新增 `test_learned_embedding_names`、`test_learnable_query_gradients`）。

---

## 6. 后续里程碑（未在本报告范围）

1. **`query_init='dual_topk'` 正式验收**：Encoder Top-K + Pair Decoder 联合回归（独立里程碑，非 M3-1）。
2. **Pair DN query**：`PairCdnQueryGenerator` + self-attn mask。
3. **全量 HSMOT pair 训练 config**。

---

## 7. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-06-17 | M3-1：learnable 默认过拟合路径 + `query_init` 三模式 |
