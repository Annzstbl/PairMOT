# PairMOT 多服务器实验状态总表

更新时间：2026-07-30 23:45 CST。

本文档记录当前论文相关正式实验在各服务器上的分布和状态。状态由实际训练进程、共享
存储中的 checkpoint/日志及已有报告交叉确认。`smoke_*`、`tmp_*`、`profile_*` 和
`detcheck` 等短测不作为正式实验单列；同一实验的失败启动、重启和最终有效目录合并说明。

状态统一使用英文：`RUNNING`表示当前存在训练进程；`QUEUED`表示等待脚本仍存活且尚未
启动正式训练；`PREPARED`表示代码和静态验证已完成但尚未进入队列；`COMPLETED`表示完成目标
训练或评估；`STOPPED`表示主动取消、硬件中断或不再续跑；`NONE`表示该服务器当前无运行或
排队任务。时间均为CST；无法从日志或启动记录可靠确定时留空。

## 当前资源总览

| 服务器 | 当前实验 | 当前进度 | 排队实验 | 工作目录根路径 |
| --- | --- | --- | --- | --- |
| 99 本机 | `0730_14 motion-trust + shared-attention decoder` | 4-iter DDP smoke 与联合结构检查通过；正式 fresh 训练 epoch 1 iter 50 | 无 | `/data4/litianhao/PairMmot/workdir_99` |
| 197 | `0730_15 shared-evidence + shared-attention decoder` | 4-iter DDP smoke 与联合结构检查通过；正式 fresh 训练 epoch 1 iter 50 | 无 | `/data4/litianhao/PairMmot/workdir_197` |
| 252 | `0730_13 shared-attention decoder` | 4-iter DDP smoke 和结构检查通过；正式 fresh 训练 epoch 1 iter 100 | 无 | `/data4/litianhao/PairMmot/workdir_252` |
| 178 | `0730_16 antisymmetric frame-detail decoder` | `0730_12` epoch 8 保护线失败后已停止；`0730_16` 完成代码、69 项单测和静态审计，正在同步及 smoke | 无 | `/data4/litianhao/PairMmot/workdir_178` |
| AutoDL | 无训练 | 所有实例关机 | 无 | `/root/autodl-tmp/work_dirs` |

## 99 本机

代码路径：`/data/users/wangying01/lth/PairMOT/ai4rs`。正式训练通常使用GPU 0、1；`0723_01`
按用户指令例外使用GPU 2、3，不设置温度watchdog或自动暂停限制。

| Status | 实验 | 开始时间 | 结束时间 | 类型/主要改动 | 进度或说明 |
| --- | --- | --- | --- | --- | --- |
| RUNNING | `0727_12_paper_base_liquid_encoder_p5temporal_crossscalebudget` | 2026-07-27 20:18 |  | 严格继承`0727_01`的Base+Liquid、P5 temporal MHA及common/detail Dual-Evidence；用每层`[common mean, abs(detail) mean]`生成三尺度token，并结合三尺度均值上下文预测逐通道common/detail尺度预算。预算在P3/P4/P5维softmax后乘3，每个分支/通道总预算严格为3，仅重分配尺度贡献，不改变平均残差强度；描述侧停止梯度，输出层零初始化，无额外loss或高分辨率卷积 | 新增37,696参数，完整模型`22,796,471`，相对父配置`+0.166%`。32项功能/梯度/帧交换等变测试、配置深拷贝、完整构建和精确2卡真实数据4-iter DDP smoke全部通过；正式GPU 0、1 fresh训练已到epoch 7，约`0.972 s/iter`、MMEngine峰值约11.25 GB/rank，总/DN/encoder loss及grad norm有限；尚未产生首个正式评测点 |
| COMPLETED | `0723_05_pairdn_paircoherent_le180_cpdse_local` | 2026-07-24 03:56 | 2026-07-25 03:56 | `0723_01` + consistency-preserving DSE local：保留`x.mean`主路径，以归一化通道离散度生成零初始化、最大绝对值0.5的逐像素SE-logit残差；8个新增参数，无额外loss | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 68为`53.536/61.619`，同点pair mAP/AP50为`0.3148/0.5320`；相对Paper Base为`+0.222/-0.363`，不满足双提升。PairMOT训练及评测进程已退出，99不排新任务；不对其他用户当前GPU占用做清理或调度 |
| COMPLETED | `0723_01_paper_liquid_independent_diffproduct_pairdn_paircoherent_le180` | 2026-07-23 03:21 | 2026-07-24 03:25 | accuracy-fixed `0718_01` + PairDN重构：pair两侧共享相对噪声，正负比2:1，正样本加难，负样本按旋转IoU筛选；DN padding隔离；保持`width_longer=True,start_angle=0`并使用等价框一致、固定角度权重的L1 | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 64为`53.955/62.032`，同epoch pair mAP/AP50为`0.3114/0.5268`；相对Paper Base双提升`+0.641/+0.050` |
| COMPLETED | `0721_05_liquid_dse_diffproduct_accuracyfix` | 2026-07-21 23:57 | 2026-07-22 21:25 | accuracy-fixed `0718_01` + DSE；以identity初始化的grouped `1x1`融合Conv3D响应的channel mean/RMS，增强稀疏目标证据；启动时仍使用修复前的Negative-DN噪声 | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `54.635/61.895`，同epoch pair mAP/AP50为`0.3254/0.5438`；相对同协议父配置`0721_02`双提升`+0.308/+0.236` |
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
| RUNNING | `0728_01_paper_base_liquid_encoder_p5temporal_dualevidence_decoder0708_03` | 2026-07-28 09:30 |  | 严格以`0727_01`为父配置，冻结Base、Liquid、P5 temporal、Dual-Evidence encoder、proposal、PairDN和loss；仅加入`0708_03`的`pointer/query_prev/query_curr` tri-state decoder，并启用零初始化frame-pointer循环耦合，不使用separate FFN | 条件队列在`0727_09`完成后正确进入smoke，但旧tri-state结构保留的不可达`cross_fusion`和末层无消费者pointer更新与`find_unused_parameters=False`冲突，首次smoke于04:04退出。已保持预测逻辑不变，冻结这些结构性不可训练参数；18项单测及修复后双卡4-iter真数据smoke通过。09:30 fresh启动，09:31确认epoch 1 iter 50为`0.9245 s/iter`、`grad_norm=86.8207`、MMEngine显存约11.27 GB/rank，GPU设备占用约19.3 GB/rank；总/DN/encoder loss有限，无DDP、NaN、OOM或NCCL错误 |
| STOPPED | `0727_11_paper_base_liquid_encoder_p5temporal_momentcompetitive`的197队列 | 2026-07-27 12:15 | 2026-07-27 12:59 | 继承当前最强的`0727_01` P5 MHA与Dual-Evidence双残差；用每通道`RMS-mean(abs(x))`稀疏性矩补充common/detail全局描述，并以两路softmax共享预算替代彼此独立的sigmoid门 | 队列本身运行正常，但用户明确改在AutoDL立即训练；197未运行smoke、未创建正式workdir，队列在AutoDL正式训练通过iter 50验收后关闭。同一实验ID及科学配置迁移至AutoDL，不构成新实验 |
| STOPPED | `0727_10_paper_base_liquid_encoder_p5temporal_detailredistribute` | 2026-07-27 08:10 | 2026-07-27 12:14 | 严格继承`0727_09`，将空间门描述停止梯度并把detail空间调制归一化为均值1 | `0727_09` epoch 16已相对Base+Liquid双降`-0.075/-0.399`，det DetA/AssA也同时下降`0.375/0.353`，表明问题并非仅是空间门全局能量漂移；在smoke和正式训练前取消，两个目标workdir均未创建，由不使用空间门的`0727_11`替代 |
| COMPLETED | `0727_09_paper_base_liquid_encoder_p5temporal_detailspatial` | 2026-07-27 05:31 | 2026-07-28 04:02 | 严格继承`0727_01`的P5 MHA和Dual-Evidence；common共享残差完全不变，仅用局部common/detail幅值生成单通道空间可靠性，对signed-detail残差做逐位置调制。空间卷积零初始化并使用`2*sigmoid`，因此初始函数与父配置逐元素相同 | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `54.106/62.321`，同epoch pair mAP/AP50为`0.3122/0.5273`。相对`0727_01`为`-0.331/-0.072`，未明显超过，故按计划进入Decoder阶段 |
| STOPPED | `0727_07_paper_base_liquid_encoder_p5temporal_spatialbranchtrust` | 2026-07-27 03:05 | 2026-07-27 05:24 | 完整继承`0727_02`的双分支空间门，并增加无参数branch-energy trust region | `0727_02` epoch 12已相对Base+Liquid双降，证明其主要问题是common空间门压低检测覆盖，而不是后期残差能量失控；队列在执行smoke或正式训练前取消，未产生正式workdir |
| STOPPED | `0727_02_paper_base_liquid_encoder_p5temporal_spatialevidence` | 2026-07-27 01:39 | 2026-07-27 05:24 | 固定`0723_01` Liquid和`0705_01` P5 MHA；post-FPN使用common/detail双残差，并以2通道局部空间证据门控P3/P4/P5两条更新 | 主动停在epoch 12后。epoch 12为`47.411/54.593`，相对同epoch Base+Liquid双降`-0.676/-1.042`；det DetA/AssA分别下降`1.052/0.829`，pair mAP下降`0.0108`。与epoch 8的`-0.035/+0.465`结合，说明同时空间门控common/detail会持续削弱检测覆盖，后继改为只门控detail |
| COMPLETED | `0726_01_pairdn_paircoherent_le180_dse_cpdse_sparsereserve` | 2026-07-26 01:57 | 2026-07-27 01:36 | `0725_01` + Sparse-Reserve CP-DSE：以现有归一化dispersion map的空间RMS减均值衡量类别无关的稀疏小目标证据，对pair两帧取共享group reserve；仅衰减负向CP-DSE残差，正向修正和pair共享性质不变；无新参数、阈值或loss | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 68为`54.230/61.634`，同epoch pair mAP/AP50为`0.3138/0.5301`。相对Paper Base为`+0.916/-0.348`，相对父配置`0725_01`为`-0.896/-0.364`，不再派生Sparse-Reserve |
| COMPLETED | `0725_01_pairdn_paircoherent_le180_dse_cpdse_pairglobal` | 2026-07-25 02:45 | 2026-07-26 01:41 | `0723_01` + DSE + pair-global CP-DSE：DSE以mean/RMS增强局部检测证据，CP-DSE以pair共享group残差补充关联一致性；两条路径分别针对`0723_03`的det AssA下降和`0723_06`的det DetA下降，无额外loss | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为`55.126/61.998`，同epoch pair mAP/AP50为`0.3231/0.5449`；相对Paper Base双提升`+1.812/+0.016`，det DetA/AssA为`-0.138/+0.545`；相对DSE父配置同epoch为`+0.328/+0.245` |
| COMPLETED | `0723_06_pairdn_paircoherent_le180_cpdse_pairglobal` | 2026-07-24 03:58 | 2026-07-25 01:53 | `0723_01` + consistency-preserving DSE pair-global：保留`x.mean`主路径，将归一化离散度汇聚为pair共享group标量后生成有界SE-logit残差；8个新增参数，无额外loss | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为`54.124/61.914`，同点pair mAP/AP50为`0.3124/0.5241`；相对Paper Base为`+0.810/-0.068`，det AssA提高`0.645`但DetA下降`0.432` |
| COMPLETED | `0723_03_pairdn_paircoherent_le180_dse` | 2026-07-23 06:19 | 2026-07-24 03:41 | `0723_01` + DSE；保留两帧共享DN相对噪声和全部新PairDN/L1设置，只用identity初始化的16参数mean/RMS evidence mixer增强fusion证据 | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 68为`55.036/61.745`，同epoch pair mAP/AP50为`0.3239/0.5402`；相对Paper Base为`+1.722/-0.237`，不能视为双提升 |
| COMPLETED | `0722_01_liquid_independent_diffproduct_dnnegativefix` | 2026-07-22 10:34 | 2026-07-23 05:54 | `0721_02`的PairDN生成器修复版：将Negative DN从可趋近零的乘积噪声改为独立符号乘`uniform(1,2)`，并将attention mask修正为同一contrastive group内正负块互相可见；因此不是单独的box-noise消融 | 完成epoch 72和计划TrackEval；最终指标见Paper Mainline报告 |
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

资源测试（2026-07-23 13:40-13:48）：在用户私有目录
`/dev/shm/litianhao/pairmot_hsmot_cache`构建了带锁、容量预检、文件指纹校验和原子切换的
HSMOT JPEG缓存。缓存复制train/test共41,514张JPEG、占用约11 GiB，标注和GMC仍指向持久盘；
构建后`/dev/shm`剩余约239 GiB、系统available memory约452 GiB。GPU3对`0723_03 DSE`
进行单卡physical batch 8、8 workers、120 iter并发测试：默认allocator在首次反向OOM；
设置`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`后完成测试，去除前20 iter后的均值为
`2.5767 s/iter`，其中`data_time=0.1823 s`，MMEngine峰值memory约21.1 GiB，设备采样峰值约
23.8 GiB。同期GPU4/5正式训练均值未下降（相对紧邻运行前窗口约快1%，属于正常波动），仅
出现少量`data_time`尖峰。结论：tmpfs缓存本身容量和隔离方式安全，也未测得对GPU4/5总吞吐
的负面影响；但DSE单卡batch 8显存余量不足，不建议作为长期正式配置，正式单卡应使用
batch 4 + accumulation 2或先进一步降低峰值显存。

CSPR-DSE一epoch候选试验（2026-07-23 14:01-14:45）：在`0723_03`的PairDN+DSE上增加
CSPR，分别用低分辨率共享Conv3D预览增强route descriptor、用mean/RMS混合增强fusion
evidence，两条路径正交且不增加额外loss。GPU3使用tmpfs、physical batch 8、8 workers及
`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`完成完整epoch；checkpoint元数据确认
`epoch=1, iter=1038, seed=3407`。跳过前100 iter后平均`2.4174 s/iter`，其中
`data_time=0.0494 s`、计算约`2.3679 s`，MMEngine记录峰值约21.2 GiB，设备采样峰值约
24.17 GB。GPU4/5同期由`0.8975`变为`0.9094 s/iter`，约慢1.33%。当前数据读取只占约2%，
增加`num_workers`不会解决单卡相对双卡慢的问题；差距主要来自3090单卡承担双倍样本及
极限显存下的计算/工作区效率。该epoch证明组合结构和梯度路径可训练，不构成最终性能证据；
是否正式长训应结合178上独立CSPR的中期结果决定。

## 252 服务器

登录：`litianhao01@10.106.15.252`；代码路径：
`/data/users/litianhao01/PairMmot/ai4rs`。当前正式训练使用GPU 0、1。

| Status | 实验 | 开始时间 | 结束时间 | 类型/主要改动 | 进度或说明 |
| --- | --- | --- | --- | --- | --- |
| RUNNING | `0727_04_paper_base_liquid_encoder_p5temporal_detailenergy` | 2026-07-28 00:50 |  | 固定`0723_01` Liquid与P5 MHA，保持`0726_03` common/detail结构；仅对两帧反向的signed-detail残差施加逐通道、逐样本的原始pair-detail RMS上限，防止时序修正能量超过输入帧差。约束使用detached统计、无参数、无loss且仍严格保持pair均值与帧交换等变 | 严格队列在父实验18/18完成后通过真实数据smoke并fresh启动；当前epoch 4 iter 350，约`1.29 s/iter`、峰值约11.0 GB/rank，总/DN/encoder loss与梯度有限 |
| COMPLETED | `0726_03_paper_base_liquid_encoder_p5temporal_commondetail_pairdn_paircoherent_le180` | 2026-07-26 20:35 | 2026-07-28 00:49 | `0726_02`的严格encoder后继：保留`0723_01` Base + Liquid、P5双向全局MHA和全部训练协议；将原三尺度方向性pyramid-local替换为pair-common/detail分解。共享`[mean, abs(detail)]`描述控制奇函数local detail变换，两帧施加等大反向残差，严格保持pair均值且交换帧时输出等变 | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为`54.654/62.240`，同epoch pair mAP/AP50为`0.3239/0.5380`。相对Base+Liquid双提升`+0.699/+0.208`，相对Paper Base双提升`+1.340/+0.258`；det DetA/AssA相对Base+Liquid也同时提高`+0.039/+0.447` |
| COMPLETED | `0725_03_pairdn_paircoherent_le180_dse_cpdse_dettangent_fast` | 2026-07-25 11:05 | 2026-07-26 17:44 | `0725_01` + Detection-Tangent CP-DSE：按已有Conv3D second moment和DSE gate的group敏感度形成pair共享检测重要性向量，只移除CP-DSE残差沿该向量的一阶检测响应分量；保留其余关联修正，无新增参数/loss | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `54.229/61.808`，同epoch pair mAP/AP50为`0.3196/0.5340`；相对Paper Base为`+0.915/-0.174`，det DetA/AssA分别为`-0.074/-0.122`；相对未投影父配置`0725_01`同epoch为`-0.897/-0.190`，不再沿该方向派生 |
| STOPPED | `0725_03`两个速度审计启动 | 2026-07-25 10:54 | 2026-07-25 11:03 | 与上项模型语义相同；依次消除重复`[B,C,G,H,W]`二阶矩归约，并将完整gate-map sigmoid导数改为group-pooled logit导数 | 两个启动均只到epoch 1 iter 100、无正式checkpoint/TrackEval，不resume且不进入结果；保留目录仅用于速度审计，分别以`orderedpairs_fresh`和`optimized_fresh`结尾 |
| COMPLETED | `0723_07_pairdn_paircoherent_le180_pecg` | 2026-07-24 06:42 | 2026-07-25 10:44 | `0723_01` + Pair Evidence Consensus Gate：仅在route覆盖和Conv3D证据均一致的位置收缩两帧SE gate差异；修正项两侧反号，严格保持pair平均gate；8参数，无额外loss | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 52为`53.693/61.151`，同点pair mAP/AP50为`0.3096/0.5262`；相对Paper Base为`+0.379/-0.831`，det DetA/AssA分别下降`1.227/0.079`，不再派生该方向 |
| COMPLETED | `0723_02_paper_liquid_independent_diffproduct_pairdn_independent_le180` | 2026-07-23 03:30 | 2026-07-24 06:12 | `0723_01`严格DN噪声消融：保持模型、数据、初始化、正负比、难度筛选、attention mask和loss不变，仅将pair两帧共享的相对box噪声改为两帧独立采样 | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为`53.637/61.679`，同epoch pair mAP/AP50为`0.3155/0.5308`；相对Paper Base为`+0.323/-0.303`，不满足双提升 |
| COMPLETED | `0721_04_liquid_bsac_diffproduct_accuracyfix` | 2026-07-22 00:52 | 2026-07-23 01:50 | accuracy-fixed `0718_01` + BSAC；在Liquid采样后、共享Conv3D前加入24参数的band-slot条件校准；启动时仍使用修复前的Negative-DN噪声 | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `54.530/61.528`，同epoch pair mAP/AP50为`0.3211/0.5353`；相对同协议父配置`0721_02`为`+0.203/-0.131` |
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
| STOPPED | `0727_08_paper_base_liquid_encoder_p5temporal_dualbranchtrust` | 2026-07-27 20:51 | 2026-07-27 21:37 | 完整保留`0727_01`的P5 MHA与common/detail双残差，不使用空间门；仅分别限制shared-common和signed-detail更新RMS不超过其对应输入证据RMS，未超限更新不变 | 前序完成18/18评测且GPU连续空闲后，严格队列完成真实数据smoke并fresh启动；按用户指令主动停在epoch 3 iter 750。未生成正式epoch checkpoint，不resume、不进入论文结果；训练、launcher及队列进程均已退出，GPU 0释放，GPU 1上的其他任务未受影响 |
| STOPPED | `0727_06_paper_base_liquid_encoder_p5temporal_sharedscalar` | 2026-07-27 02:43 | 2026-07-27 03:47 | 保留P5 MHA和signed-detail，但以两帧共享的逐位置标量增益替代channel-mixing common残差 | 只进入等待队列，未创建smoke、正式目录或训练进程。`0727_01` epoch 12的det AssA已恢复到相对Base `+0.006`，同时det DetA `+1.449`，证明原common分支可实现无关联代价的检测增益；移除其通道表达能力的设计依据消失，因此队列撤销并由`0727_08`替代 |
| STOPPED | `0727_03_paper_base_liquid_encoder_p5temporal_scalesplit` | 2026-07-27 00:29 | 2026-07-27 02:40 | 固定`0723_01` Liquid和P5 MHA；P3只启用common共享检测增强，P4/P5启用common+signed-detail | 只进入等待队列，未创建smoke、正式目录或训练进程。epoch 8诊断表明实际矛盾是common残差带来det DetA `+2.710`但det AssA `-2.918`；移除P3 detail没有针对该问题，因此主动撤销队列并由`0727_06`替代 |
| COMPLETED | `0727_01_paper_base_liquid_encoder_p5temporal_dualevidence` | 2026-07-27 00:17 | 2026-07-27 20:48 | 固定`0723_01` Liquid、PairDN、proposal、decoder和loss，保留`0705_01` P5双向MHA；post-FPN将common共享检测证据同向注入两帧，将signed detail关联证据反向注入两帧，零门控且严格帧交换等变 | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `54.437/62.393`，同epoch pair mAP/AP50为`0.3201/0.5371`。相对Base+Liquid双提升`+0.482/+0.361`；相对Paper Base双提升`+1.123/+0.411`，det DetA/AssA也分别提高`+0.526/+0.311` |
| PREPARED | `0727_05_paper_base_liquid_encoder_p5temporal_spatialdetail` |  |  | 回退到`0726_03`的detail-only、pair均值守恒结构；以逐层空间均值归一化的common/detail局部能量生成`2→1`空间可靠性门，只调制signed-detail，不改共享单帧特征。门初值严格为1、范围`[0,2]`，总计57参数，无额外loss | 本地和178均通过20项单测及逐文件哈希，包含相同父权重与非零gamma下逐元素严格等价测试；完整模型参数为22,677,421。仅完成准备，未启动、未占GPU；作为detail-only保守回退候选，当前优先执行更直接针对DetA/AssA矛盾的`0727_06` |
| COMPLETED | `0725_02_pairdn_paircoherent_le180_dse_cpdse_centered_protocolfix` | 2026-07-25 08:30 | 2026-07-26 11:57 | `0725_01`的DSE + pair-global CP-DSE组合，但将8个group的CP-DSE logit残差去均值；单卡physical batch 4 + accumulation 2，并将warmup从2000改为4000 micro-iter、EMA改为`interval=2,gamma=4000`，保持按optimizer update计的原协议时间尺度 | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 68为cls HOTA `53.730`、det HOTA `61.864`，同epoch pair mAP/AP50为`0.3100/0.5239`；相对Paper Base为`+0.416/-0.118`，相对Base+Liquid仍未双侧超过 |
| STOPPED | `0725_02_pairdn_paircoherent_le180_dse_cpdse_centered`初始accumulation run | 2026-07-25 06:57 | 2026-07-25 08:27 | 模型与上项相同，physical batch 4 + accumulation 2，但仍沿用2000 micro-iter warmup和每micro-iter计数的EMA | 到epoch 4后主动停止，不resume；MMEngine源码核实ParamSchedulerHook和EMAHook均按micro-iteration推进，导致warmup和EMA时间尺度相对原bs8协议缩短一半。该run及其1/18 TrackEval只作审计，不进入结果表 |
| COMPLETED | `0723_08_pairdn_paircoherent_le180_scpd` | 2026-07-24 08:53 | 2026-07-25 06:38 | `0723_01` + Spectral-Coordinate Pair Dispersion：将只读group色散经soft coverage投影到8个物理谱段，在谱段坐标求pair共识后投影回各帧自身route；跨group去均值、零初始化、有界8参数，无额外loss | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为`54.465/61.213`，同点pair mAP/AP50为`0.3152/0.5334`；相对Paper Base为`+1.151/-0.769`，det DetA/AssA为`53.712/72.176`，失败原因主要是关联质量下降 |
| COMPLETED | `0723_04_pairdn_paircoherent_le180_cspr` | 2026-07-23 12:31 | 2026-07-24 08:50 | `0723_01` + CSPR；保留两帧共享DN相对噪声，以detached shared-Conv3D在24x32低分辨率生成谱段预览统计，替换简单global route descriptor | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为`54.523/61.738`，同epoch pair mAP/AP50为`0.3162/0.5376`；相对Paper Base为`+1.209/-0.244`，不满足双提升 |
| COMPLETED | `0721_03_liquid_bsr_diffproduct_accuracyfix_dnnegativefix` | 2026-07-22 10:07 | 2026-07-23 06:36 | 配置继承`0721_02`并以12x16 block recurrent descriptor替换global route statistics；正式启动源码同时包含修复后的Negative-DN外环采样和contrastive-group attention mask，实际语义更接近`0722_01 + BSR` | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `53.107/61.118`，同epoch pair mAP/AP50为`0.3082/0.5220`，结果不支持用BSR替换global descriptor |
| COMPLETED | `0721_01_liquid_diffproduct_qc_responsemass_1xb8_shm_accuracyfix_20260721` | 2026-07-21 03:03 | 2026-07-22 02:43 | `0718_01` + response-weighted fusion quality conservation；单卡batch 8；JPEG使用安全tmpfs缓存；加入20260721训练准确性修复 | 完成epoch 72和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `54.006/60.934`，HOTA和`114.940`；同epoch pair mAP/AP50为`0.3148/0.5297` |
| STOPPED | `0721_01_liquid_diffproduct_qc_responsemass_1xb8_shm`原run | 2026-07-21 01:23 | 2026-07-21 03:02 | 与上项模型配置相同，但未包含20260721训练准确性修复 | 主进程运行约1小时39分、到epoch 4后主动停止；旧checkpoint保留，不resume |
| COMPLETED | `0719_05_paper_base_rerun` | 2026-07-20 00:20 | 2026-07-21 01:02 | 无Liquid的Paper Base复跑；seed 3407、LR `1e-4`、BF16、全量1200x900；单卡batch 8保持global batch 8 | 已完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `52.417/61.265`，HOTA和比主Base低`1.614` |
| COMPLETED | `0719_02_pairconsensus_reliability` | 2026-07-19 03:02 | 2026-07-20 00:28 | reliability-weighted pair-consensus + PACDE，单卡batch 8 | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 64为cls/det HOTA `55.190/60.689` |
| COMPLETED | `ctracker_hsmot_r50_3dse_rotated_1200x900_bs4_acc2` |  |  | CTracker HSMOT适配 | 完成100 epochs及最终推理/TrackEval；checkpoint已归档到共享存储 |

队列日志：
`/data4/litianhao/PairMmot/workdir_178/queue_0721_01_after_0719_05.log`、
`/data4/litianhao/PairMmot/workdir_178/queue_0719_05_after_0719_02.log`、
`/data4/litianhao/PairMmot/workdir_178/queue_0721_03_after_0721_01.log`、
`/data4/litianhao/PairMmot/workdir_178/queue_0723_04_after_0721_03.log`。

## AutoDL 临时服务器

AutoDL实例地址会变化，初始化、数据布局、训练、finalizer和关机流程见
`pairmot-autodl`技能及`/data/users/wangying01/lth/PairMOT/autodl`。以下只记录正式实验，
不将实例迁移和环境smoke作为实验。

| Status | 实验 | 开始时间 | 结束时间 | 类型/主要改动 | 进度或说明 |
| --- | --- | --- | --- | --- | --- |
| COMPLETED | `0727_11_paper_base_liquid_encoder_p5temporal_momentcompetitive` | 2026-07-27 12:57 | 2026-07-28 06:37 | 固定`0723_01` Base+Liquid、P5 MHA和全部论文协议；保留Dual-Evidence双残差，以逐通道`RMS-mean(abs(x))`感知稀疏目标证据，并用common/detail两路softmax共享门控预算。单卡RTX 5090 physical batch 8保持global batch、LR、EMA和72-epoch语义 | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 68为cls/det HOTA `54.853/62.050`，同epoch pair mAP/AP50为`0.3190/0.5384`，HOTA和`116.903`。相对Base+Liquid为`+0.898/+0.018`，相对`0727_01`为`+0.416/-0.343`且HOTA和仅高`0.073`，不能视为明显胜出。finalizer已归档epoch 68、epoch 72、日志和完整TrackEval，状态为`publish_skipped_no_deploy_key artifacts_preserved best_epoch=68`；权威结果已取回`autodl/results/0727_11` |
| COMPLETED | `0726_02_paper_base_liquid_encoder_p5temporal_pyramidlocal_pairdn_paircoherent_le180` | 2026-07-26 15:37 | 2026-07-27 09:23 | `0723_01` Base + Liquid + `0705_01` Encoder：在FPN前P5加入全局双向pair temporal MHA，在FPN后P3/P4/P5加入逐层pyramid-local adapter；两支均以零初始化gamma残差接入，不改Liquid、proposal、decoder、PairDN或loss | 完成72 epochs和18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `54.742/61.631`，同点pair mAP/AP50为`0.3223/0.5429`。相对直接父配置`0723_01`最佳点为`+0.787/-0.401`，不满足双侧超过Base+Liquid；相对Paper Base为`+1.428/-0.351`。finalizer已将epoch 72、日志、完整TrackEval归档到`/autodl-fs/data/PairMOT_results/0726_02`；因无deploy key跳过GitHub发布，权威结果已取回`autodl/results/0726_02` |
| COMPLETED | `0719_01_pairconsensus_relaxedset_pacde` | 2026-07-19 02:22 | 2026-07-19 22:39 | pair共享route + relaxed Set-Transport + pair-aligned fusion + PACDE | epoch 72、18/18 TrackEval；唯一最佳epoch 72为cls/det HOTA `53.883/61.411`；结果已从共享盘取回 |
| COMPLETED | `0717_01_paper_base_plus_liquid_settransport` |  |  | Set-Transport Liquid | epoch 72、18/18 TrackEval；结果归档于`/root/autodl-fs/PairMOT_results/0717_01` |
| COMPLETED | `0718_02_paper_base_plus_liquid_anchorcompetitive` |  |  | anchor-competitive Liquid | epoch 72、18/18 TrackEval；finalizer已归档到`/autodl-fs/data/PairMOT_results/0718_02` |

2026-07-27 11:18在无卡模式完成数据盘整理。删除前逐项验证`0717_01`、`0718_02`、
`0719_01`和`0726_02`共享归档中的finalizer状态及全部`SHA256SUMS`；随后清理四个完整
workdir、过时smoke、epoch1中断目录、MMCV临时编译树和空launcher日志。数据盘由
`49GB/98%`降至`11GB/21%`，释放约`38GB`；保留完整HSMOT、13,713个真实GMC JSON、
AutoDL元数据和空`work_dirs`。审计位于
`/root/autodl-fs/PairMOT_maintenance/cleanup_20260727_111815`，相对路径哈希复核全部通过。
可复用流程已保存为`~/.codex/skills/pairmot-autodl-data-cleanup`。

## 维护规则

1. 新实验编号在所有服务器之间全局递增；当前最后分配编号为`0728_01`，下一编号为`0728_02`。
2. 新任务启动、停止、完成或迁移后，应同步更新本表和
   `docs/20260706_multi_server_experiment_plan.md`。
3. `RUNNING`和`QUEUED`状态以实际进程为准，不能仅依据workdir存在或旧screen名称判断。
4. 论文结果表只采用完整评估结果；中止训练和smoke/profile目录不得混入正式指标。
5. 当前Encoder目标严格固定`0723_01` Base+Liquid：唯一最佳epoch 64的cls HOTA为
   `53.955`、det HOTA为`62.032`。Liquid、PairDN、proposal、decoder、loss及论文数据协议
   均保持不变，只允许修改Encoder；成功条件是在同一个唯一最佳epoch上同时超过两个门槛。
   正式实验使用252/197双卡和178单卡，不使用研究代码重跑Base。
6. 正式结果使用
   `projects/multispec_pair_rotated_rtdetr/tools/summarize_pair_experiment.py WORK_DIR`
   复核。工具默认要求18个完整TrackEval目录、按payload真实step映射epoch、拒绝HOTA和并列，
   并从同一step的`val_det/metrics.json`读取AP；阶段诊断必须显式使用`--allow-partial`。
7. 用户于2026-07-27明确重新启用99进行两卡Encoder探索；当前仅使用GPU 0、1运行
   `0727_12`，GPU 2、3保持空闲。不得添加温度watchdog，也不得未经新指令在GPU 2、3并行
   启动第二个任务。
8. AutoDL不纳入自主排队或自动启动；只有用户明确介入并提供运行资源时才训练。其余时间
   只读取已有日志、checkpoint和评测产物进行监控，不因其他服务器任务结束而唤醒实例。
9. 197的Encoder阶段在`0727_09`结束后执行条件收口：若其未同时明显超过`0727_01`两侧
   HOTA，则转入Decoder阶段。首个Decoder实验`0728_01`固定`0727_01`的全部上游模块，只
   复用历史最佳`0708_03` tri-state zero-init decoder；不得混入新的Encoder、proposal或
   loss改动。
10. `0727_11`是AutoDL上的最后一个Encoder实验。整个Encoder阶段只等待已运行的99
    `0727_12`和252 `0727_04`完成，不再追加新Encoder结构；两项收尾后按统一选点规则确定
    最终Encoder，并继续已启动的Decoder主线。

2026-07-23 22:56阶段趋势（仅用于调度，不作为最终结果）：四个任务共同已有的epoch 36中，
`0723_01`共享DN噪声为`51.673/59.845`，`0723_02`独立DN噪声为`50.912/59.227`，
`0723_03`共享噪声+DSE为`53.553/60.704`，`0723_04`共享噪声+CSPR为
`52.506/59.513`（均为cls/det HOTA）。当前证据不支持独立pair噪声；DSE同时领先两个HOTA，
CSPR主要提高cls DetA但det AssA低于`0723_01`。较晚点中`0723_03`在epoch 52达到
`54.581/61.335`，相对`0723_01`同epoch为`+0.869/-0.227`，提示DSE后期的剩余问题是关联
质量而非分类覆盖。epoch 56继续得到`54.705/61.465`，相对`0723_01`同epoch
`53.922/61.721`为`+0.783/-0.256`，该分化具有连续性。所有判断必须等各自18/18评估后
重新计算，当前不据此提前启动新实验。

2026-07-24 04:13补充：`0723_02`截至epoch 64为`53.571/61.726`，相对Paper Base为
`+0.257/-0.256`，同点pair mAP/AP50为`0.3125/0.5267`；`0723_04`截至epoch 52为
`54.369/60.721`，相对Base为`+1.055/-1.261`。两者继续证明扩大帧间随机性或route描述
能力容易损害det关联。为利用252即将释放的双卡，新增`0723_07 PECG`：保持`0723_01`
共享PairDN和原Liquid结构，仅在谱段覆盖与Conv3D证据均一致的位置，对两帧SE gate做
保持pair均值的收缩。53项测试、完整模型构建及本机双卡4-iteration smoke已通过；252队列
在04:13正确报告`epoch72=no, eval=16/18`，没有提前占用GPU。

2026-07-23 23:29补充：`0723_02`独立pair噪声在epoch 52为`52.852/61.211`，相对
`0723_01`同epoch`53.712/61.562`双降`-0.860/-0.351`；其中cls/det AssA分别下降
`2.147/0.522`。连续评测证据已经足以停止继续探索独立pair噪声，后续PairDN保留两帧共享
相对扰动。

2026-07-24 00:22补充：`0723_03`在epoch 60为`54.699/61.695`，相对共享PairDN同epoch
`53.951/61.907`为`+0.748/-0.212`。DSE的HOTA和达到`116.394`，与`0719_06`
的`116.423`接近，但det仍未达标；类别上主要是car和tricycle AssA下降，而truck和bus
DetA明显提高。`0723_04`在epoch 40为`53.303/59.914`，相对共享PairDN为
`+0.810/-0.734`，且det AssA下降`1.500`。下一候选优先保留DSE的谱段离散度证据，但必须
用有界残差调制替代无约束mean/RMS替换；CSPR不作为主方向。

2026-07-24 00:54补充：`0723_01`在epoch 64达到`53.955/62.032`，已相对Paper Base唯一
最佳epoch 68的`53.314/61.982`双提升`+0.641/+0.050`，但尚未超过`0719_06`
`54.087/62.336`，也尚未完成18/18评测。`0723_02`在epoch 56仍相对共享噪声双降
`-0.590/-0.322`，其中cls AssA下降`1.321`，进一步确认独立pair噪声无效。

2026-07-24 01:26补充：`0723_03`epoch 64为`54.685/61.824`，相对共享PairDN同epoch
继续保持`+0.730/-0.208`，说明DSE增益与det代价在后期均稳定。`0723_04`epoch 44为
`53.728/60.314`，相对共享PairDN为`+0.269/-0.769`，其中det AssA下降`1.421`；CSPR
中后期未扭转关联损失。

2026-07-24 02:43补充：`0723_01`epoch 68为`53.847/62.060`，当前按HOTA和选取的阶段
唯一最佳仍是epoch 64的`53.955/62.032`。`0723_03`epoch 68达到`55.036/61.745`，
相对共享PairDN同epoch为`+1.189/-0.315`；其cls DetA增加`1.912`，但det AssA下降
`0.743`。DSE已有足够分类裕量，下一结构应以有界残差主动换回关联质量。`0723_02`
epoch 60为`53.610/61.561`，相对共享PairDN仍双降`-0.341/-0.346`；`0723_04`
epoch 48为`54.063/60.831`，相对共享PairDN为`+0.592/-0.349`。

2026-07-24 04:27补充：`0723_04`新增epoch 56结果`54.421/60.887`，相对Paper Base为
`+1.107/-1.095`。固定epoch 56对比直接父配置`0723_01`，cls/det HOTA为
`+0.499/-0.834`；det DetA仅下降`0.111`，det AssA下降`1.737`，IDSW由`1844`增加到
`2379`。CSPR的主要代价因此确定为帧间身份排序不稳定，而非检测覆盖不足；不再沿其route
preview方向扩展。正在排队的PECG保持原route，只收缩高一致性位置的SE gate差异，与该诊断
直接对应。

2026-07-24 05:32补充：CP-DSE首个epoch 4评测已完成。local `0723_05`为
`37.778/44.770`，pair-global `0723_06`为`37.388/44.660`；相对`0723_01`同epoch分别为
`+1.077/-0.240`与`+0.687/-0.350`。local的det AssA提高`0.502`、IDSW减少75，但det
DetA下降`0.698`，说明局部色散早期有利于关联但抑制检测覆盖；pair-global同时损失det
DetA/AssA。两组8参数残差仍小于`0.005`，首点不足以判定最终性能，继续训练。`0723_02`
epoch 68为`53.609/61.680`，阶段唯一最佳仍是epoch 64的`53.571/61.726`。

2026-07-24 06:05补充：`0723_04`epoch 60为`54.231/61.263`，相对`0723_01`同epoch为
`+0.280/-0.644`。det DetA仅下降`0.074`，det AssA下降`1.290`，IDSW增加513；CSPR的
关联损失连续存在，不再作为后续结构基础，但保留运行至完整18/18用于正式报告。

2026-07-24 06:47补充：`0723_02`完成72 epochs和18/18 TrackEval，唯一最佳epoch 72为
`53.637/61.679`，相对Paper Base为`+0.323/-0.303`；相对共享噪声父配置同epoch为
`+0.043/-0.333`。其det AssA提高`0.102`，但det DetA下降`0.552`且FP增加2748，证明独立
pair噪声主要损害检测精度。`0723_07 PECG`随后完成远端精确smoke并于06:42双卡fresh启动，
正式iter 200及全部五项启动门槛通过。同期`0723_06`epoch 8达到`44.661/50.551`，相对父
配置同epoch双提升`+0.349/+0.714`，det DetA/AssA提高`0.441/1.150`，是当前最积极趋势。

## 2026-07-30 18:25 CST decoder 调度覆盖

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 197 GPU 4,5 | `0730_09_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_motiontrust_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 18:23 /  | 严格继承 `0727_01`，新增检测置信度门控、位移包络约束的反对称 motion-trust decoder。配置深拷贝、47 项单测和双卡真实数据 4-iter smoke 通过；18:24 到 epoch 1 iter 50，`0.9085 s/iter`、loss `21.3915`、grad norm `120.0892`，正式启动五项门槛全部通过。 |
| STOPPED | 197 GPU 4,5 | `0730_08 ... decoder_competitiveevidence ... fresh` | 2026-07-30 / 2026-07-30 17:58 | epoch 4 cls/det HOTA `36.277/43.729`；相对 `0727_01` 同点 `+0.068/+4.976`，但增益集中于 det AssA `+9.441`，pair mAP `-0.0041`，完成 checkpoint、检测及 2/2 TrackEval 后按门槛停止。 |
| STOPPED | 252 GPU 0,1 | `0730_07 ... decoder_commonmotion_sharedevidence ... fresh` | 2026-07-30 / 2026-07-30 17:58 | epoch 4 cls/det HOTA `36.926/42.702`；相对 `0727_01` 同点 `+0.717/+3.949`，但 cls/det DetA 均约 `-1.0`，pair mAP/AP50 和 both-independent AP50 未形成一致提升，完成 checkpoint、检测及 2/2 TrackEval 后停止。 |

当前可调度资源为 99 双卡、197 双卡、252 双卡、178 单卡；AutoDL 全部关机。197 新实验使用干净副本 `/data/users/litianhao/PairMOT_sync_3cb888d`，历史目录不作代码源。`0730_09` 代码提交为 `f231b01`；99 canonical 与 197 已精确同步，252/178 在状态文档提交后同步到同一提交。空闲卡只用于模型结构实验，不进行 class-specific/long-tail reweight 或 residual-scale 参数扫描。

## 2026-07-30 18:46 CST decoder 资源与启动状态

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 197 GPU 4,5 | `0730_09_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_motiontrust_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 18:23 /  | 197 两卡已正式纳入当前 decoder 资源池并由本实验占用。18:45 到 epoch 1 iter 900，约 `1.5280 s/iter`、loss `12.1506`、grad norm `21.8685`；总 loss、DN loss 与 encoder loss 均有限，无 Traceback、OOM、NaN、NCCL 或 unused-parameter 错误。GPU 4/5 各约 `19.2 GiB`。首个结构判断点为 epoch 4。 |
| RUNNING | 252 GPU 0,1 | `0730_10_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_symmetricpair_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 18:41 /  | 严格继承 `0727_01`，decoder 两帧共享 deformable cross-attention 权重，并对 cross-frame feature fusion 与 pair-position fusion 显式平均正反两种帧序；不增加参数，不修改 encoder、proposal、PairDN、head、loss 或训练协议。52 项 decoder 单测、配置深拷贝、launcher shell 审计和双卡真实数据 4-iter smoke 通过；预训练权重中 24 组 attention 参数逐项相等，smoke checkpoint 中 4 个 fusion 矩阵的帧交换误差最大为 `8.45e-09`。正式训练五项启动门槛通过，18:45 到 epoch 1 iter 200，约 `1.0837 s/iter`、loss `18.9054`、grad norm `56.8993`，GPU 0/1 各约 `19.2 GiB`，无异常。首个结构判断点为 epoch 4。 |

当前资源池确认为 99 双卡、197 双卡、252 双卡、178 单卡；其中 197 GPU 4/5 与 252 GPU 0/1 正在运行结构实验，99 GPU 0/1 与 178 GPU 0 保留给 epoch 4 诊断后有明确模型依据的下一候选，不以参数扫描填卡。代码提交 `5341c32` 已通过带 prerequisite 的 Git bundle 精确同步到 99、197、252 和 178；各服务器既有未跟踪文件未覆盖或删除。

## 2026-07-30 20:29 CST decoder epoch 4 门控覆盖

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 197 GPU 4,5 | `0730_09_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_motiontrust_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 18:23 /  | epoch 4 checkpoint、检测、结构审计及 2/2 TrackEval 完成并通过固定门槛。cls HOTA/DetA/AssA `37.878/27.722/55.486`，det `44.102/34.288/58.241`；相对 `0727_01` 同点 HOTA `+1.669/+5.349`、DetA `+0.654/+1.834`。pair mAP/AP50 `0.16257/0.31578`，both-independent mAP/AP50 `0.18872/0.34231`，均高于父配置。三层 motion-trust adapter 最大绝对权重 `0.14190/0.13821/0.12903`，均有限且非零。继续训练到 epoch 8；20:27 到 epoch 5 iter 350，训练健康。 |
| STOPPED | 252 GPU 0,1 | `0730_10_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_symmetricpair_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 18:41 / 2026-07-30 20:12 | epoch 4 完成 checkpoint、检测和 2/2 TrackEval。cls HOTA/DetA/AssA `36.750/26.756/54.632`，det `42.604/33.160/55.890`；pair mAP/AP50 `0.1496/0.3108`，both-independent mAP/AP50 `0.1786/0.3400`。虽 det HOTA/DetA 与 AP50 提高，但 pair mAP 相对父配置下降 `0.00765`，未通过 `0.003` 保护线，因此停止并释放 GPU。共享 attention 结构保持，fusion 半矩阵 `1.788e-7` 差异属于 FP32 漂移，不是前向交换等变失效。 |

当前可调度资源仍为 99 GPU 0/1、197 GPU 4/5、252 GPU 0/1、178 GPU 0，共 7 张；197 GPU 4/5 正由 `0730_09` 占用，99 GPU 0/1、252 GPU 0/1 与 178 GPU 0 空闲。99 GPU 2 和 197 其他 GPU 不属于本实验调度范围。AutoDL 全部关机。四台服务器的目标代码提交均为 `fbbf137`，已有未跟踪文件继续保留。

## 2026-07-30 20:52 CST decoder 并行启动

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 197 GPU 4,5 | `0730_09_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_motiontrust_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 18:23 /  | epoch 4 已通过固定门槛并继续到 epoch 8。20:52 到 epoch 6 iter 250，约 `1.5907 s/iter`、loss `11.0446`、grad norm `40.5960`；总、DN、encoder loss 均有限，GPU 4/5 正常占用。 |
| RUNNING | 252 GPU 0,1 | `0730_11_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedrouting_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 20:49 /  | 严格继承 `0727_01`，只在每层 decoder 两帧 deformable cross-attention 之间共享 `sampling_offsets` 与 `attention_weights`；`value_proj` 和 `output_proj` 保持独立，避免 `0730_10` 全共享导致的 pair mAP 损失。56 项 decoder 单测、配置深拷贝、权重/GMC/路径审计及双卡真实数据 4-iter smoke 通过；smoke checkpoint 的 12 组 routing 最大误差为 `0`，12 组独立 projection 最大差异为 `7.7571e-4`。正式 fresh 训练五项启动门槛通过，20:52 到 epoch 1 iter 150，约 `1.1223 s/iter`、loss `19.6946`、grad norm `122.9360`，总、DN、encoder loss 均有限，无 Traceback、OOM、NaN、NCCL、DDP reduction 或 unused-parameter 错误。首个判断点为 epoch 4。 |

`0730_11` 代码提交为 `9049422`，通过带 prerequisite 的 Git bundle 精确同步至 99、197、252 和 178；各服务器原有未跟踪文件均保留。当前形成两路正式并行：197 的 motion-trust 与 252 的 shared-routing。178 GPU 0 仍空闲，等待另一项有独立模型依据的 decoder 结构；99 的双卡额度已恢复，但实时 GPU 0 和 GPU 2 被其他用户的 UNet 任务占用，仅 GPU 1 空闲，未将其错误计作可立即使用的双卡 DDP 资源。

## 2026-07-30 21:13 CST 第三路结构实验启动

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 178 GPU 0 | `0730_12_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_motiontrust_sharedevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` | 2026-07-30 21:10 /  | 组合 `0730_09` 的 detection-confident bounded motion-trust 与 `0730_06` 的 swap-invariant shared-evidence。两者分别作用于反对称框更新和共享 query，保持 `0727_01` 的 encoder、proposal、PairDN、head、loss、数据与训练协议不变。57 项 decoder 单测、配置深拷贝、路径/权重/GMC/launcher 审计通过。单卡 physical batch 8 的真实数据 4-iter smoke 中总、DN、encoder loss 与 grad norm 均有限；checkpoint 三层 motion-trust 最大绝对权重为 `3.7246e-4/3.6952e-4/3.6620e-4`，三层 shared-evidence 为 `3.9646e-4/3.9722e-4/3.9787e-4`，均有限且非零。正式 fresh 训练五项启动门槛通过，21:12 到 epoch 1 iter 100，约 `0.8798 s/iter`、loss `20.1321`、grad norm `90.7448`；GPU 0 设备占用约 31.4 GiB，无 Traceback、OOM、NaN、DDP reduction 或 unused-parameter 错误。首判 epoch 4。 |

当前形成三路正式并行：197 `0730_09`、252 `0730_11`、178 `0730_12`。99 没有漏排：
虽然双卡额度已恢复，但实时 GPU 0/2 被其他用户的 UNet 任务各占约 21 GiB，仅 GPU 1
空闲；不得终止或挤占外部任务，也不得将单张空闲卡误报为双卡 DDP 资源。四台服务器代码已
同步到 `f1ffbd7`，其中 `c43635c` 引入组合结构，`f1ffbd7` 使可信本地 smoke checkpoint
兼容 PyTorch 2.6+ 的显式非 weights-only 加载；各服务器既有未跟踪文件均保留。

## 2026-07-30 22:35 CST shared-routing 门控与 shared-attention 接替

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 197 GPU 4,5 | `0730_09_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_motiontrust_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 18:23 /  | epoch 8 checkpoint 已产生，训练到 epoch 9 iter 1000；总、DN、encoder loss 和梯度有限。等待 epoch 8 完整检测、2/2 TrackEval 与结构审计后按固定同点门槛决定继续或停止。 |
| RUNNING | 178 GPU 0 | `0730_12_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_motiontrust_sharedevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` | 2026-07-30 21:10 /  | epoch 4 checkpoint 已产生，训练到 epoch 6 iter 100；等待 epoch 4 完整检测、2/2 TrackEval 与两组 adapter 结构审计。 |
| STOPPED | 252 GPU 0,1 | `0730_11_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedrouting_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 20:49 / 2026-07-30 22:19 | epoch 4 完整 artifacts 后停止。cls HOTA/DetA/AssA `36.504/27.255/51.884`，det `42.163/33.695/53.992`，pair mAP/AP50 `0.149753/0.305401`，both-independent mAP/AP50 `0.178556/0.335452`。pair mAP 相对父配置下降 `0.007499`，未通过保护线；结构审计通过。 |
| RUNNING | 252 GPU 0,1 | `0730_13_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 22:32 /  | 只共享每层两帧 cross-attention 的 `attention_weights`，保留独立 `sampling_offsets/value_proj/output_proj`。61 项 decoder 单测、配置/launcher 审计和双卡真实数据 4-iter DDP smoke 通过；smoke checkpoint 的 6 组 attention 误差为零，18 组应独立参数已分化。正式训练五项门槛通过，22:35 到 epoch 1 iter 100，约 `1.1160 s/iter`、loss `20.5979`、grad norm `103.9917`，GPU 0/1 各约 `19.2 GiB`，无异常。首判 epoch 4。 |

99 在 22:31 的实时采样中三张 GPU 均为约 `10 MiB/0%`，此前“仅 GPU 1 空闲”的判断已失效。
当前不复制已有实验或用参数扫描填卡；99 两卡仅分配给与 197/178/252 不重复且经过完整
配置、单测和真数据 smoke 的新 decoder 结构。99 canonical 与 252 已位于 `091af97`；
197/178 的当前进程保持启动时代码，完成门控后再快进同步。

## 2026-07-30 22:46 CST 第四路结构实验启动

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 99 GPU 0,1 | `0730_14_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_motiontrust_sharedattention_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 22:45 /  | 组合 `0730_09` 的 detection-confident bounded motion-trust 与 `0730_13` 的 frame-localized shared attention。前者作用于有界反对称框修正，后者只共享 attention 权重并保留逐帧 sampling offsets/value/output projection；构成主效应与交互项，不改 encoder、proposal、PairDN、head、loss、数据或训练协议。62 项 decoder 单测、正式/短测配置深拷贝、launcher 审计与双卡真实数据 4-iter DDP smoke 通过。smoke checkpoint 的三层 motion adapter 均非零，6 组 attention 误差为零，18 组独立参数最大差异 `7.9339e-4`。正式训练五项门槛通过，22:46 到 epoch 1 iter 50，约 `0.9575 s/iter`、loss `21.4104`、grad norm `100.1361`，GPU 0/1 各约 `19.2 GiB`，无异常。首判 epoch 4。 |

至此四台服务器均有独立 decoder 结构实验：197 `0730_09`、178 `0730_12`、
252 `0730_13`、99 `0730_14`。当前 canonical 代码提交为 `b829704`；完成本状态提交后
以 Git bundle 快进同步到 252/197/178，保留各服务器未跟踪文件且不重启已有训练。

## 2026-07-30 23:00 CST epoch 门控与 197 接替实验

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| STOPPED | 197 GPU 4,5 | `0730_09_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_motiontrust_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 18:23 / 2026-07-30 22:52 | epoch 8 完整 artifacts 与结构审计后停止。cls HOTA/DetA/AssA `45.498/36.788/58.984`，det `51.160/45.018/60.159`，pair mAP/AP50 `0.230058/0.430536`，both-independent mAP/AP50 `0.265862/0.464984`。相对父配置 HOTA 小幅上升，但 cls/det DetA 下降 `0.875/2.043`、pair mAP 下降 `0.007676`，为 AssA 搬运而非共同改善。精确终止后 GPU 4/5 均为 `1 MiB/0%`。 |
| RUNNING | 178 GPU 0 | `0730_12_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_motiontrust_sharedevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` | 2026-07-30 21:10 /  | epoch 4 cls HOTA/DetA/AssA `40.559/29.273/59.330`，det `46.017/34.057/64.397`，pair mAP/AP50 `0.187248/0.342315`，both-independent mAP/AP50 `0.213887/0.366343`；HOTA、DetA、pair mAP 与 both AP50 均高于父配置。motion-trust 与 shared-evidence 三层权重均有限非零，结构门控通过，继续到 epoch 8。 |
| RUNNING | 197 GPU 4,5 | `0730_15_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedevidence_sharedattention_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 22:58 /  | 组合 shared-evidence 与 shared-attention，以区分 `0730_12` 强增益是否需要 motion-trust。63 项 decoder 单测、配置/launcher 审计和双卡真实数据 4-iter DDP smoke 通过；smoke checkpoint 三层 evidence adapter 均非零，6 组 attention 误差为零，18 组独立参数最大差异 `7.8762e-4`。正式训练五项门槛通过，22:59 到 epoch 1 iter 50，约 `1.0668 s/iter`、loss `21.4097`、grad norm `120.1933`，双卡各约 `19.2 GiB`，无异常。首判 epoch 4。 |

当前四路正式训练为 99 `0730_14`、197 `0730_15`、252 `0730_13`、178 `0730_12`。
canonical 代码提交为 `0782826`；本次状态提交后四机统一快进，保留所有既有 artifacts 和
未跟踪目录。

## 2026-07-30 23:45 CST 178 门控失败与结构接替

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| STOPPED | 178 GPU 0 | `0730_12_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_motiontrust_sharedevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` | 2026-07-30 21:10 / 2026-07-30 23:30 | epoch 8 checkpoint、检测、2/2 TrackEval 与结构审计完整。cls HOTA/DetA/AssA `44.387/33.262/61.885`，det `49.824/40.708/63.485`，pair mAP/AP50 `0.2144/0.3809`，both-independent mAP/AP50 `0.2457/0.4098`。相对父配置 cls/det HOTA `-0.882/-0.369`、DetA `-4.401/-6.353`，pair mAP `-0.02333`、both AP50 `-0.05615`；AssA 增益不足以抵消检测损失，按固定保护线停止并释放 GPU。 |
| PREPARED | 178 GPU 0 | `0730_16_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_antisymmetric_detail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` |  /  | 接替结构只在逐帧 cls/reg head 前注入由真实双帧 cross-attention 证据生成的有界 `-detail/+detail`，保持 recurrent shared query、下一层路径和两帧特征中点严格等于父模型；不直接改框或共享 query。69 项 decoder 单测、配置深拷贝、launcher 审计和代码差异检查通过，待同步至 178 后执行真实数据 smoke 和正式启动五项门槛。 |

实时资源状态为三路正式训练：252 `0730_13`、99 `0730_14`、197 `0730_15`；
178 在 `0730_12` 停止后处于结构接替窗口。该空窗不是资源限制，完成 `0730_16` smoke
和启动门槛后立即恢复四机并行；AutoDL 继续保持全部关机。
