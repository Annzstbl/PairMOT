# PairMOT 多服务器实验状态总表

更新时间：2026-07-22 23:55 CST。

本文档记录当前论文相关正式实验在各服务器上的分布和状态。状态由实际训练进程、共享
存储中的 checkpoint/日志及已有报告交叉确认。`smoke_*`、`tmp_*`、`profile_*` 和
`detcheck` 等短测不作为正式实验单列；同一实验的失败启动、重启和最终有效目录合并说明。

状态统一使用英文：`RUNNING`表示当前存在训练进程；`QUEUED`表示等待脚本仍存活且尚未
启动正式训练；`COMPLETED`表示完成目标训练或评估；`STOPPED`表示主动取消、硬件中断或不再续跑；
`NONE`表示该服务器当前无运行或排队任务。时间均为CST；无法从日志或启动记录可靠确定时留空。

## 当前资源总览

| 服务器 | 当前实验 | 当前进度 | 排队实验 | 工作目录根路径 |
| --- | --- | --- | --- | --- |
| 99 本机 | `0721_05 DSE-Liquid` | epoch 43 iter 800；TrackEval 10/18 | 无 | `/data4/litianhao/PairMmot/workdir_99` |
| 197 | `0722_01 corrected Negative-DN 0718_01 rerun` | epoch 50 iter 350；TrackEval 12/18 | 无 | `/data4/litianhao/PairMmot/workdir_197` |
| 252 | `0721_04 BSAC-Liquid` | epoch 34 iter 300；TrackEval 8/18 | 无 | `/data4/litianhao/PairMmot/workdir_252` |
| 178 | `0721_03 BSR + corrected Negative DN` | epoch 9 iter 350；TrackEval 2/18 | 无 | `/data4/litianhao/PairMmot/workdir_178` |
| AutoDL | 无卡模式，无训练 | `0719_01`已完成72 epochs和18/18 TrackEval | 无 | `/autodl-fs/data/PairMOT_results/0719_01` |

## 99 本机

代码路径：`/data/users/wangying01/lth/PairMOT/ai4rs`。正式训练通常使用GPU 0、1；GPU
2、3有掉卡历史，不再并行安排正式任务。

| Status | 实验 | 开始时间 | 结束时间 | 类型/主要改动 | 进度或说明 |
| --- | --- | --- | --- | --- | --- |
| RUNNING | `0721_05_liquid_dse_diffproduct_accuracyfix` | 2026-07-21 23:57 |  | accuracy-fixed `0718_01` + DSE；以identity初始化的grouped `1x1`融合Conv3D响应的channel mean/RMS，增强稀疏目标证据；启动时仍使用修复前的Negative-DN噪声 | epoch 43 iter 800，10/18 TrackEval；阶段最佳epoch 40为cls/det HOTA `53.398/60.479`；约`0.89 s/iter`，loss和梯度有限 |
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

登录：`litianhao@10.106.14.197`；代码路径：`/data/users/litianhao/PairMOT/ai4rs`。
当前正式训练使用GPU 4、5。

| Status | 实验 | 开始时间 | 结束时间 | 类型/主要改动 | 进度或说明 |
| --- | --- | --- | --- | --- | --- |
| RUNNING | `0722_01_liquid_independent_diffproduct_dnnegativefix` | 2026-07-22 10:34 |  | `0721_02`的PairDN生成器修复版：将Negative DN从可趋近零的乘积噪声改为独立符号乘`uniform(1,2)`，并将attention mask修正为同一contrastive group内正负块互相可见；因此不是单独的box-noise消融 | epoch 50 iter 350，12/18 TrackEval；阶段最佳epoch 48为cls/det HOTA `53.775/60.531`，同epoch pair mAP/AP50为`0.3079/0.5229`；相对`0721_02`同epoch为`+0.157/-0.224`；训练loss和梯度有限，ETA约5小时31分；screen：`queue_0722_01_197` |
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

## 252 服务器

登录：`litianhao01@10.106.15.252`；代码路径：
`/data/users/litianhao01/PairMmot/ai4rs`。当前正式训练使用GPU 0、1。

| Status | 实验 | 开始时间 | 结束时间 | 类型/主要改动 | 进度或说明 |
| --- | --- | --- | --- | --- | --- |
| RUNNING | `0721_04_liquid_bsac_diffproduct_accuracyfix` | 2026-07-22 00:52 |  | accuracy-fixed `0718_01` + BSAC；在Liquid采样后、共享Conv3D前加入24参数的band-slot条件校准；启动时仍使用修复前的Negative-DN噪声 | epoch 34 iter 300，8/18 TrackEval；阶段最佳epoch 32为cls/det HOTA `51.916/59.067`；约`1.05 s/iter`，loss和梯度有限；screen：`train_0721_04` |
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
| RUNNING | `0721_03_liquid_bsr_diffproduct_accuracyfix_dnnegativefix` | 2026-07-22 10:07 |  | BSR-Liquid：以12x16 block recurrent descriptor替换全局route statistics；同时采用修复后的Negative-DN外环采样和contrastive-group attention mask | 补齐漏同步的BSR实现后v2 smoke通过4/4真实迭代并fresh启动；epoch 9 iter 350，2/18 TrackEval；阶段最佳epoch 8为cls/det HOTA `43.793/49.771`；约`0.82 s/iter`，loss/梯度有限；screen：`queue_0721_03_178` |
| COMPLETED | `0721_01_liquid_diffproduct_qc_responsemass_1xb8_shm_accuracyfix_20260721` | 2026-07-21 03:03 | 2026-07-22 02:43 | `0718_01` + response-weighted fusion quality conservation；单卡batch 8；JPEG使用安全tmpfs缓存；加入20260721训练准确性修复 | 完成epoch 72和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `54.006/60.934`，HOTA和`114.940`；同epoch pair mAP/AP50为`0.3148/0.5297` |
| STOPPED | `0721_01_liquid_diffproduct_qc_responsemass_1xb8_shm`原run | 2026-07-21 01:23 | 2026-07-21 03:02 | 与上项模型配置相同，但未包含20260721训练准确性修复 | 主进程运行约1小时39分、到epoch 4后主动停止；旧checkpoint保留，不resume |
| COMPLETED | `0719_05_paper_base_rerun` | 2026-07-20 00:20 | 2026-07-21 01:02 | 无Liquid的Paper Base复跑；seed 3407、LR `1e-4`、BF16、全量1200x900；单卡batch 8保持global batch 8 | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `52.417/61.265`，HOTA和比主Base低`1.614` |
| COMPLETED | `0719_02_pairconsensus_reliability` | 2026-07-19 03:02 | 2026-07-20 00:28 | reliability-weighted pair-consensus + PACDE，单卡batch 8 | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 64为cls/det HOTA `55.190/60.689` |
| COMPLETED | `ctracker_hsmot_r50_3dse_rotated_1200x900_bs4_acc2` |  |  | CTracker HSMOT适配 | 完成100 epochs及最终推理/TrackEval；checkpoint已归档到共享存储 |

队列日志：
`/data4/litianhao/PairMmot/workdir_178/queue_0721_01_after_0719_05.log`、
`/data4/litianhao/PairMmot/workdir_178/queue_0719_05_after_0719_02.log`、
`/data4/litianhao/PairMmot/workdir_178/queue_0721_03_after_0721_01.log`。

## AutoDL 临时服务器

AutoDL实例地址会变化，初始化、数据布局、训练、finalizer和关机流程见
`pairmot-autodl`技能及`/data/users/wangying01/lth/PairMOT/autodl`。以下只记录正式实验，
不将实例迁移和环境smoke作为实验。

| Status | 实验 | 开始时间 | 结束时间 | 类型/主要改动 | 进度或说明 |
| --- | --- | --- | --- | --- | --- |
| COMPLETED | `0719_01_pairconsensus_relaxedset_pacde` | 2026-07-19 02:22 | 2026-07-19 22:39 | pair共享route + relaxed Set-Transport + pair-aligned fusion + PACDE | epoch 72、18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `53.883/61.411`；结果已从共享盘取回 |
| COMPLETED | `0717_01_paper_base_plus_liquid_settransport` |  |  | Set-Transport Liquid | epoch 72、18/18 TrackEval；结果归档于`/root/autodl-fs/PairMOT_results/0717_01` |
| COMPLETED | `0718_02_paper_base_plus_liquid_anchorcompetitive` |  |  | anchor-competitive Liquid | epoch 72、18/18 TrackEval；finalizer已归档到`/autodl-fs/data/PairMOT_results/0718_02` |
| NONE | - |  |  | - | 当前为无卡模式，无持续占用GPU的进程，也没有AutoDL等待任务 |

## 维护规则

1. 新实验编号在所有服务器之间全局递增；当前最后分配编号为`0722_01`，下一编号为`0722_02`。`0721_06 CSPR`当前仅为已实现候选，不在正式训练队列中。
2. 新任务启动、停止、完成或迁移后，应同步更新本表和
   `docs/20260706_multi_server_experiment_plan.md`。
3. `RUNNING`和`QUEUED`状态以实际进程为准，不能仅依据workdir存在或旧screen名称判断。
4. 论文结果表只采用完整评估结果；中止训练和smoke/profile目录不得混入正式指标。
