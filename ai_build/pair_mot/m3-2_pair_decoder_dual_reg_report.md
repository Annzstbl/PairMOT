# M3-2 Pair Decoder 双帧独立回归与梯度修正报告

> **文档性质**：在 [m3-1_pair_decoder_query_init_report.md](./m3-1_pair_decoder_query_init_report.md) 之后，**修正 Pair Decoder 层内双 reference 更新、self-attn 位置编码与 Head/Detector 接入**，使 prev/curr 框可独立运动，reference 梯度语义与 `RotatedDinoTransformerDecoder` 一致。  
> 本里程碑 **仅改 Pair 侧实现**，不修改 `RotatedRTDETRTransformerDecoder` / `rotated_dino_layers.py` 等原始单帧代码。

| 项 | 内容 |
|----|------|
| 里程碑 | M3-2 — 双回归分支 + 有序 pos 融合 + DINO 式 reference 梯度 |
| 前置文档 | [m3-1_pair_decoder_query_init_report.md](./m3-1_pair_decoder_query_init_report.md)、[m4_pair_head_report.md](./m4_pair_head_report.md) |
| 日期 | 2026-06-17 |
| 仓库 | `/data/users/litianhao01/PairMmot/ai4rs` |
| 原则 | **Pair Decoder / Head / Detector 适配；原始 O2-RTDETR 文件只读复用** |

---

## 1. 问题与动机

M3j / M3-1 交付的 Pair Decoder 在 M5 接入与过拟合调试中暴露三处与 Pair MOT 设计不一致：

| 问题 | M3-1 状态 | 影响 |
|------|-----------|------|
| 单 `reg_branches` 同时更新 prev/curr | `tmp` 共享 | 两帧框被迫同步偏移，无法独立跟踪运动 |
| self-attn 位置编码简单平均 | `0.5×(pos_prev+pos_curr)` | 丢失 prev→curr 时间顺序 |
| reference 更新与输出混用同一 tensor | 先 append 再 `.detach()` 或 `inverse_sigmoid` 提前 detach | 与 DINO 中间层输出 / 下一层 reference 语义不一致 |

**修正目标**：prev/curr **独立回归**；self-attn pos **有序融合**；**当前层预测**（供 Head loss）与 **下一层 reference**（迭代状态）分离，格式对齐 `rotated_dino_layers.py` §78–121。

---

## 2. Pair Decoder 改动（`pair_rotated_rtdetr_layers.py`）

### 2.1 双回归分支 API

`forward` 参数由单一 `reg_branches` 改为：

```python
reg_branches_prev: nn.ModuleList
reg_branches_curr: nn.ModuleList
```

每层：

```python
layer_output = self.norm(query)
tmp_prev = reg_branches_prev[lid](layer_output)
tmp_curr = reg_branches_curr[lid](layer_output)
```

### 2.2 有序 self-attn 位置编码

新增 `pair_pos_fusion = Linear(2C, C)`，替换均值：

```python
query_pos = self.pair_pos_fusion(
    torch.cat([query_pos_prev, query_pos_curr], dim=-1))
```

cross-attn 仍各自使用 `query_pos_prev` / `query_pos_curr`。

### 2.3 Reference 更新（对齐 DINO）

参考 `RotatedDinoTransformerDecoder`：

```python
new_reference_prev = tmp_prev + inverse_sigmoid(reference_prev, eps=1e-3)
new_reference_prev = new_reference_prev.sigmoid()
new_reference_curr = tmp_curr + inverse_sigmoid(reference_curr, eps=1e-3)
new_reference_curr = new_reference_curr.sigmoid()
reference_prev = new_reference_prev.detach()
reference_curr = new_reference_curr.detach()

hidden_states.append(layer_output)
references_prev.append(new_reference_prev)
references_curr.append(new_reference_curr)
```

| Tensor | 梯度 | 用途 |
|--------|------|------|
| `new_reference_*`（append 到列表） | 保留（经 `tmp` + `inverse_sigmoid(ref)`） | Head bbox loss / 当前层预测 |
| `reference_*`（下一层输入） | `.detach()` 整段 | 阻断跨层 reference 梯度链 |

**不应**在 `inverse_sigmoid(reference)` 输入处提前 `.detach()`（旧 O2-RTDETR unact 写法）；整段 sigmoid 后再 detach，与 DINO 一致。

### 2.4 Learnable reference 初始化

`ref_prev_embedding` / `ref_curr_embedding`：**相同初始值、独立 `nn.Embedding` 参数**（取消 M3j 中 curr +0.05 偏移）。

---

## 3. Head / Detector 适配

### 3.1 `PairRotatedRTDETRHead`

- 继承的 `reg_branches` → **prev 侧** decoder 回归
- 新增 `reg_branches_curr`（`deepcopy(reg_branches)`），`init_weights` 同步初始化末层

Head `forward` 仍直接消费 decoder 输出的 `references_prev/curr` 列表，无需二次 reg。

### 3.2 `MultispecPairRotatedRTDETR`

- `_pair_decoder_reg_branches()`：切片 `reg_branches` / `reg_branches_curr` 传入 Decoder
- `forward_decoder_pair`：`reg_branches_prev=` / `reg_branches_curr=`
- `_topk_pair_queries`：curr 侧 proposal 回归改用 `reg_branches_curr[num_layers]`

### 3.3 预训练适配（`load_pair_pretrain.py`）

单帧 O2 ckpt 加载时，额外复制：

`bbox_head.reg_branches.*` → `bbox_head.reg_branches_curr.*`

---

## 4. 修改文件清单

| 路径 | 说明 |
|------|------|
| `pair_rotated_rtdetr_layers.py` | 双 reg、pair_pos_fusion、DINO 式 reference 更新 |
| `pair_rotated_rtdetr_head.py` | `reg_branches_curr` |
| `multispec_pair_rotated_rtdetr.py` | `_pair_decoder_reg_branches`、双分支调用、Top-K curr reg |
| `tools/load_pair_pretrain.py` | curr reg 权重复制 |
| `tests/test_projects/test_pair_rotated_rtdetr_decoder.py` | M3-2 梯度 / 分化 / shape 测试 |
| `tests/test_projects/test_pair_rotated_rtdetr_head.py` | 双分支 decoder 调用适配 |

---

## 5. 测试结果

```bash
cd ai4rs && python -m pytest \
  tests/test_projects/test_pair_rotated_rtdetr_decoder.py \
  tests/test_projects/test_pair_rotated_rtdetr_head.py -v
```

| 测试项 | 验收点 |
|--------|--------|
| `test_dual_reg_branches_receive_gradients` | prev/curr 回归分支均有非零梯度 |
| `test_gradients_reach_both_memories` | 两帧 memory 均有梯度 |
| `test_same_init_refs_diverge_with_different_memories` | 相同初始 ref、不同 memory → 最终 prev/curr 框可分化 |
| `test_stacked_reference_shape` | `stack(refs)`: `[num_layers, B, N, 5]` |
| `test_no_nan_or_inf` / `test_amp_fp16_forward` | 数值稳定、FP16 前向 |

**合计：25 passed**（decoder 14 + head 11，2026-06-17，py310）。

---

## 6. 与 M3-1 的衔接

| 项 | M3-1 | M3-2 |
|----|------|------|
| Query 初始化 | `query_init` 三模式 | 不变 |
| Decoder reg 接口 | 单 `reg_branches` | `reg_branches_prev/curr` |
| self-attn pos | 平均 | `pair_pos_fusion` |
| reference 梯度 | 单 tmp + 混用 detach | 双 tmp + DINO 式 append/detach |
| Head reg | 仅 `reg_branches` | + `reg_branches_curr` |

过拟合 config 仍为 `query_init='learned'`、`dn_cfg=None`；M3-2 不改变 M3-1 过拟合路径选择，仅修正 Decoder 内部行为与接入。

---

## 7. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-06-17 | M3-2：双帧独立回归、有序 pos 融合、DINO 式 reference 梯度 + Head/Detector 适配 |
