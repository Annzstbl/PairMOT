# 20260704 实验结果分析报告

## 1. 实验对象

本报告检查两个 20260703/20260704 延续实验：

1. `0703_baseline_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_typed_pairtopk_v1_pairdn`
   - workdir: `/data/users/litianhao01/PairMmot/workdir/0703_baseline_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_typed_pairtopk_v1_pairdn`
   - 主要改动：typed proposal generation，`300 survival + 30 curr-only + 30 prev-only`。
   - 训练设置：`max_epochs=72`，val interval=4，early stop 监控 `pair/pair_mAP50_95`，patience=4。
   - 实际状态：跑到 epoch 40 后 early stop。日志显示 best score 约 0.232，连续 4 次记录未超过 `min_delta=0.001` 后停止。

2. `0703_liquid_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique_pairdn`
   - workdir: `/data4/litianhao/PairMmot/workdir_197/0703_liquid_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_bothvis_dualcls_nopres_pairtopk_v2_unique_pairdn`
   - 主要改动：在 0702 unique baseline 上启用 Liquid Spectral Sampling Conv3D stem。
   - liquid 设置：`tau=2.0`，`init_logit=2.0`，`head_weight_std=1e-3`，low-res gradient correction，`tau` 在 36 epoch 内降到 0.5，36 epoch 后 hard=True。
   - 训练设置：`max_epochs=48`，early stop 监控 `pair/pair_mAP50_95`，patience=4。
   - 实际状态：跑满 epoch 48，最后一次 val_det 为 epoch 47。

两个实验的 val 都保存了 `val_det/epoch_**`，并触发了异步 track/eval。track 参数均为 `nb0.6_tr0.2_iou0.25_birthiou0.5_age30`。

## 2. 核心结果

### 2.1 最佳 AP checkpoint 对比

| 实验 | 最佳 val_det epoch | pair mAP50:95 | pair AP50 | both AP50 | new AP50 | disappear AP50 | independent AP50 | association gap AP50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| typed_pairtopk_v1 | 39 | 0.2328 | 0.4109 | 0.4212 | 0.0011 | 0.0006 | 0.4309 | 0.0200 |
| liquid_unique | 47 | 0.2369 | 0.4206 | 0.4356 | 0.0000 | 0.0000 | 0.4404 | 0.0198 |

结论：

- liquid_unique 的 pair mAP50:95 比 typed_pairtopk_v1 高 0.0041，AP50 高 0.0097，both AP50 高 0.0144。
- typed_pairtopk_v1 虽然加入了 curr-only/prev-only proposal，但 new/disappear AP 仍然接近 0。它确实输出了极少量可匹配 single-visible 结果，但目前没有形成有效能力。
- liquid_unique 没有 typed single-only 机制，因此 new/disappear AP 为 0 是预期行为。它的优势主要来自 survival/both 目标。

### 2.2 最佳 AP checkpoint 的 tracking 对比

| 实验 | epoch | cls HOTA | cls MOTA | cls IDF1 | det HOTA | det MOTA | det IDF1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| typed_pairtopk_v1 | 39 | 45.114 | 33.297 | 52.398 | 57.502 | 51.290 | 66.148 |
| liquid_unique | 47 | 45.900 | 34.183 | 53.837 | 57.114 | 50.420 | 66.039 |

结论：

- liquid_unique 在分类相关 tracking 上更好：cls HOTA +0.786，cls MOTA +0.886，cls IDF1 +1.439。
- typed_pairtopk_v1 在 det tracking 上略强：det HOTA +0.388，det MOTA +0.870，det IDF1 +0.109。
- 这说明 liquid 的检测/关联质量提升更偏向类别一致性和 IDF1；typed 当前没有把 single-visible 设计转化为 new/disappear 召回优势。

### 2.3 tracking 最优点

typed_pairtopk_v1:

- 最高 cls HOTA 出现在 epoch 35：45.207。
- 最高 det HOTA 也出现在 epoch 35：57.543。
- 最佳 pair mAP 出现在 epoch 39：0.2328。
- pair mAP 与 tracking 的最优点不完全一致，但差距很小。epoch 35 到 39 的 pair mAP 从 0.2305 到 0.2328，cls HOTA 从 45.207 到 45.114，det HOTA 从 57.543 到 57.502。

liquid_unique:

- pair mAP、cls HOTA、det HOTA 都在最后的 epoch 47 最好或接近最好。
- 当前曲线仍在缓慢上升，48 epoch 结束不是因为 early stop，而是到达 max_epochs。

## 3. typed_pairtopk_v1 分析

typed_pairtopk_v1 的目标是把 proposal 拆成 survival、curr-only、prev-only 三类，缓解旧 unique topk 只偏向 survival 的问题。但从结果看：

- all pair AP 没有超过 liquid/unique，最佳 pair mAP50:95 为 0.2328。
- new/disappear AP 几乎为 0，即使到 epoch 39，new AP50 只有 0.0011，disappear AP50 只有 0.0006。
- tracking 结果相对稳定，但主要提升来自 survival/both 部分，而不是 single-visible 部分。

当前现象说明：typed proposal 的结构已经接入训练和 val 流程，但 single-only 分支还没有学到可用检测能力。可能原因包括：

1. single-visible 样本在全量 pair AP 中占比低，训练梯度被 survival 主导。
2. only 侧监督需要非 visible 侧 cls 贴近 0，这比 survival 匹配更难稳定。
3. proposal 阶段的 only 候选来自未被 survival 使用的单帧 top-M，质量受 survival 抢占和单帧 score 排序影响。
4. 当前 AP/track 仍主要依赖两帧都存在的目标，因此 single-only 的错误很难通过总 mAP 直接暴露，但 new/disappear 子指标已经暴露了问题。

建议下一步不要直接扩大 typed 训练规模，而是先做针对性诊断：

- 对 val_det 中的 curr-only/prev-only 输出数量、score 分布、IoU 命中率做单独统计。
- 分开看 proposal loss 与 decoder loss 中 only GT 的正样本数。
- 可视化 only GT 附近是否有 query，判断是 proposal 阶段没召回，还是 decoder 分类/回归没学起来。

## 4. liquid_unique 分析

这次 liquid 实验解决了之前“谱段顺序几乎不变”的问题。监控日志统计：

- LiquidSampler log 数量：464。
- `changed_ratio` 平均值：0.4702，最小 0.3403，最大 0.5903。
- `max_prob` 平均值：0.6381，训练后期可到 1.0。
- `entropy` 平均值：1.0178，后期 hard=True 后部分记录 entropy 降到 0。
- 监控到的 dominant pattern 基本不重复，例如首条为 `061 / 123 / 137 / 355 / 646 / 530`，末条为 `212 / 125 / 220 / 367 / 456 / 117`。

这说明修改后的初始化和退火策略有效：采样器不再被固定窗口 `012 / 123 / 234 / 345 / 456 / 567` 锁死，确实学到了非局部谱段组合。

性能上，liquid_unique 在 survival/both AP 上优于 typed_pairtopk_v1，也优于 typed 的 cls tracking。由于 liquid 实验运行在另一台服务器，训练速度不能直接横向比较；日志中的平均 train iter time 为 0.842s，typed 主服务器日志为 1.012s，但这包含硬件、IO 和环境差异，不应作为 liquid 更快的证据。

## 5. AP 与 tracking 的关系

这两个实验中，pair mAP 与 tracking 的关系比 0702 baseline vs 0628 fixed 时更一致：

- typed 内部：pair mAP 从 epoch 3 到 39 持续提升，tracking 也总体提升；但最高 tracking 在 epoch 35，最佳 pair mAP 在 epoch 39。
- liquid 内部：pair mAP 和 tracking 基本同步提升，最佳点都在最后。
- 跨实验比较：liquid 的 pair AP 更高，同时 cls HOTA/IDF1 更高；det HOTA/MOTA 则略低。

因此当前结论不是“pair AP 不能用”，而是：

- survival/both AP 与 tracking 仍有较强相关性。
- all pair AP 中 new/disappear 子集目前太弱，无法说明 typed single-only 是否有效。
- checkpoint selection 可以继续用 `pair/pair_mAP50_95`，但报告和曲线必须同时看 cls/det HOTA、MOTA、IDF1，尤其是 IDF1。

## 6. 当前结论

1. typed_pairtopk_v1 已完整跑通训练、val_det、异步 track/eval 和 early stop，但核心目标 single-visible 还没有成功。new/disappear AP 接近 0，不能证明 typed proposal 当前有效。
2. liquid_unique 跨服务器实验有效。它不仅 AP 更高，而且 Liquid Spectral Sampling 的 pattern 明显发生变化，说明采样器真的在学习非固定谱段组合。
3. 两个实验中，liquid_unique 是当前更强的模型候选：best pair mAP50:95=0.2369，cls HOTA=45.900，cls IDF1=53.837。
4. typed_pairtopk_v1 的价值目前主要是工程路径打通，不是性能提升。后续应先定位 only 分支为何没有召回，再决定是否继续投入训练资源。

## 7. 后续建议

优先级建议：

1. 继续 liquid_unique 方向：把 48 epoch 作为当前有效结果，后续可考虑 72 epoch 或基于 best checkpoint 小学习率续训，观察是否还能提升。
2. typed_pairtopk_v1 不建议直接大规模继续训练。先做 only 分支专项分析：正样本数、proposal 命中率、decoder 输出分数、可视化。
3. 报告/曲线中继续保留四组 AP：all、both、new、disappear。typed 这类实验必须重点看 new/disappear，否则很容易被 both survival 指标掩盖。
4. checkpoint selection 可继续以 `pair/pair_mAP50_95` 为主，但每次 val 后的 track/eval 结果必须保留，用于发现 AP 与 MOT 指标偏离。

## 8. 0704 新增跨服务器实验配置

基于今天的分析，已经新增三组可提交到 git 的实验配置。其中两组是给另一台服务器运行的 liquid 实验，均继承当前已跑通的 liquid unique 配置：

```text
projects/multispec_pair_rotated_rtdetr/configs/
```

### 8.1 非 liquid 对照：unique + all GT

配置文件：

```text
o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt.py
```

目的：

- 不使用 typed proposal。
- 继续使用 0702 baseline 的 non-typed `pair_topk_v2` + `unique_pair_selection=True`。
- 将 `train_both_visible_only=False`，使 decoder loss、encoder/proposal loss 和 PairDN direct loss 都在 track-union all GT 上监督。
- 对 single-visible GT：可见侧 dual-cls 输出真实类别，不可见侧 dual-cls 输出 background，即 sigmoid 类别全 0；不可见侧 box loss weight 为 0。

关键设置：

```text
query_init = pair_topk_v2
num_queries = 300
decoder.num_queries = 300
test_cfg.max_per_img = 300
train_both_visible_only = False
max_epochs = 72
early stop = pair/pair_mAP50_95, patience=4
```

### 8.2 跨服务器 liquid：typed proposal

配置文件：

```text
o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_typed_pairtopk_v1_pairdn_liquid.py
```

目的：

- 保留 Liquid Spectral Sampling Conv3D stem。
- proposal 改为 typed proposal：`300 survival + 30 curr-only + 30 prev-only`。
- 关闭 both-visible 过滤，typed 三类 GT 分别监督。

关键设置：

```text
query_init = typed_pair_topk_v1
num_queries = 360
decoder.num_queries = 360
test_cfg.max_per_img = 360
train_both_visible_only = False
num_survival_queries = 300
num_curr_only_queries = 30
num_prev_only_queries = 30
max_epochs = 72
work_dir = /data4/litianhao/PairMmot/workdir_197/0704_02_liquid_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_typed_pairtopk_v1_pairdn
```

### 8.3 跨服务器 liquid：unique + all GT

配置文件：

```text
o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid.py
```

目的：

- 保留 Liquid Spectral Sampling Conv3D stem。
- 不使用 typed proposal，继续使用 non-typed unique proposal。
- 关闭 both-visible 过滤，让 proposal/decoder/DN 全部使用 all GT 监督。

关键设置：

```text
query_init = pair_topk_v2
num_queries = 300
decoder.num_queries = 300
test_cfg.max_per_img = 300
train_both_visible_only = False
unique_pair_selection = True
max_epochs = 72
work_dir = /data4/litianhao/PairMmot/workdir_197/0704_03_liquid_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt
```

### 8.4 跨服务器运行前置条件

另一台服务器 clone 代码后，需要保证以下文件/目录存在：

```text
/data4/litianhao/PairMmot/data/hsmot/train
/data4/litianhao/PairMmot/data/hsmot/test
/data4/litianhao/PairMmot/pretrained_weights/o2_r18_hsmot_3dse_r2_e72_pair_dualcls_pairdn_adapted/pair_dualcls_pairdn_adapted_pretrain.pth
/data4/litianhao/PairMmot/workdir/aux/gmc_cache/hsmot_train_gap1
/data4/litianhao/PairMmot/workdir/aux/gmc_cache/hsmot_test_gap1
/data4/litianhao/PairMmot/TrackEval
```

两个 liquid 配置通过 `_pairmot_root = Path(__file__).resolve().parents[4]` 推导 `/data4/litianhao/PairMmot`，所以只要 repo 放在：

```text
/data4/litianhao/PairMmot/ai4rs
```

data、pretrain、gmc cache 和 TrackEval 的相对路径就会自动匹配。

### 8.5 建议启动命令

typed + liquid：

```bash
cd /data4/litianhao/PairMmot/ai4rs
source /data/users/litianhao01/anaconda3/etc/profile.d/conda.sh
conda activate py310
CUDA_VISIBLE_DEVICES=0,1 PORT=29721 bash tools/dist_train.sh \
  projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_typed_pairtopk_v1_pairdn_liquid.py \
  2 \
  --work-dir /data4/litianhao/PairMmot/workdir_197/0704_02_liquid_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_typed_pairtopk_v1_pairdn
```

unique + all GT + liquid：

```bash
cd /data4/litianhao/PairMmot/ai4rs
source /data/users/litianhao01/anaconda3/etc/profile.d/conda.sh
conda activate py310
CUDA_VISIBLE_DEVICES=0,1 PORT=29722 bash tools/dist_train.sh \
  projects/multispec_pair_rotated_rtdetr/configs/o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt_liquid.py \
  2 \
  --work-dir /data4/litianhao/PairMmot/workdir_197/0704_03_liquid_o2_pair_rtdetr_r18vd_2xb4_72e_hsmot_half_pairdn_gap1train_dualcls_nopres_pairtopk_v2_unique_pairdn_allgt
```

### 8.6 已完成验证

本机已完成配置解析验证：

```text
typed liquid:
query_init = typed_pair_topk_v1
num_queries = 360 / 360
max_per_img = 360
train_both_visible_only = False
liquid_sampler = True

unique all-GT liquid:
query_init = pair_topk_v2
num_queries = 300 / 300
max_per_img = 300
train_both_visible_only = False
liquid_sampler = True
```

同时补充了 head 单测，确认 dual-cls/no-presence 下 all-GT single-visible 监督满足：

- 不可见侧 cls target 为 background。
- 不可见侧 bbox weight 为 0。
- PairDN target 保留 all GT，并按 `valid_prev/valid_curr` 设置每侧监督。

测试命令：

```bash
python -m pytest tests/test_projects/test_pair_rotated_rtdetr_head.py -q
```

结果：

```text
14 passed
```
