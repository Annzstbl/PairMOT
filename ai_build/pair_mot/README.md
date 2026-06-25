# Pair MOT — 工程文档目录

本目录存放 Pair MOT / O2-RTDETR 相关的设计与审计文档。

## 文档列表

文档按时间顺序阅读：

| 文件 | 性质 | 说明 |
|------|------|------|
| [o2_rtdetr_audit_report.md](./o2_rtdetr_audit_report.md) | **只读审计（M1 前）** | 未改任何代码；O2-RTDETR 调用链、tensor shape、Top-K、Hungarian/DN、Pair 扩展 **规划** |
| [m1_pair_data_pipeline_report.md](./m1_pair_data_pipeline_report.md) | **M1 改动报告** | 审计之后实施的 `HSMOTPairDataset`、pair transforms、测试与真实数据校验 |
| [m2_pair_model_report.md](./m2_pair_model_report.md) | **M2 改动报告** | M1 之后实施的 `MultispecPairRotatedRTDETR`、Pair 预处理器、等价性测试 |
| [m3_pair_decoder_report.md](./m3_pair_decoder_report.md) | **M3j 改动报告** | M2 之后实施的 `PairRotatedRTDETRTransformerDecoder`、双 cross-attn DecoderLayer、单元测试 |
| [m3-1_pair_decoder_query_init_report.md](./m3-1_pair_decoder_query_init_report.md) | **M3-1 修正报告** | learnable query/ref 命名统一、`query_init` 三模式、过拟合默认路径修正 |
| [m3-2_pair_decoder_dual_reg_report.md](./m3-2_pair_decoder_dual_reg_report.md) | **M3-2 修正报告** | 双回归分支、有序 pos 融合、DINO 式 reference 梯度、Head/Detector 适配 |
| [m4_pair_head_report.md](./m4_pair_head_report.md) | **M4 改动报告** | M3j 之后实施的 `PairRotatedRTDETRHead`、`PairHungarianAssigner`、Pair Match Costs、单元测试 |
| [m4-stage_report_current_implementation.md](./m4-stage_report_current_implementation.md) | **阶段性汇总（推荐）** | M1–M5 当前实现逻辑、M3-1/M3-2、双模式对比、限制与后续 |

## 相关代码路径（ai4rs）

- 模型：`projects/rotated_rtdetr/`、`projects/rotated_dino/`、`projects/multispec_rotated_rtdetr/`、`projects/multispec_pair_rotated_rtdetr/`
- HSMOT 配置：`projects/multispec_rotated_rtdetr/configs/o2_rtdetr_r50vd_2xb4_72e_hsmot_coco_pretrain.py`
- Pair 数据管线：`mmrotate/datasets/hsmot_pair.py`、`mmrotate/datasets/pair_gt.py`
- Pair 可视化：`tools/visualize_hsmot_pair_real.py`

## 环境

审计基于 **conda py310**，mmdet 安装于：

`/data/users/litianhao01/anaconda3/envs/py310/lib/python3.10/site-packages/mmdet`
