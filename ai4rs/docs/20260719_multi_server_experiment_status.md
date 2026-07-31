# PairMOT 多服务器实验状态总表

更新时间：2026-07-31 15:14 CST。

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
| 99 本机 | `0731_19 terminal classification common evidence` | RUNNING；14:38 到 epoch 1 iter 400，loss/grad norm `17.2928/11.1189`，无训练异常；首判 epoch 4 | 无 | `/data4/litianhao/PairMmot/workdir_99` |
| 197 | `0731_20 shared-attention + terminal classification common evidence` | RUNNING；14:38 到 epoch 1 iter 250，loss/grad norm `18.3156/27.1948`，无训练异常；首判 epoch 4 | 无 | `/data4/litianhao/PairMmot/workdir_197` |
| 252 | `0731_18 shared-attention + terminal orthogonal factorized evidence` | RUNNING；14:38 到 epoch 2 iter 50，loss/grad norm `12.7175/23.6045`，无训练异常；首判 epoch 4 | 无 | `/data4/litianhao/PairMmot/workdir_252` |
| 178 | `0731_21 independent-attention + terminal orthogonal factorized evidence` | RUNNING；真实单卡 smoke 与结构 checkpoint 检查通过；15:14 到 epoch 1 iter 50，loss/grad norm `20.9349/90.7400` | 无 | `/data4/litianhao/PairMmot/workdir_178` |
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
| RUNNING | `0731_16_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_terminalcommonevidencebypass_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` | 2026-07-31 12:42 |  | 继承`0727_01`及其共享 decoder 路径，只在最后一层最终双帧预测头前注入 swap-invariant、零起点且有界的共同 cross-attention 证据；不改 recurrent query、所有 auxiliary output 及任何供后续层消费的 reference，避免`0731_03`全层注入的递归污染 | 提交`8a663bd`已同步四机。4-iter真实数据smoke总、DN、encoder loss及grad norm有限，唯一 terminal gate 最大权重`3.98724e-4`且结构检查通过。正式训练12:43达到epoch 1 iter 50，`0.9353 s/iter`、loss`21.4237`、grad norm`94.8035`，GPU 0约31.4 GiB，无Traceback/OOM/NaN；首判epoch 4 |
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

1. 新实验编号在所有服务器之间全局递增；当前最后分配编号为`0731_21`，下一编号为`0731_22`。
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
| RUNNING | 197 GPU 4,5 | `0730_15_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedevidence_sharedattention_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 22:58 /  | epoch 4 checkpoint、检测、完整 TrackEval 与结构审计通过。cls HOTA/DetA/AssA `36.732/27.680/51.849`，det `41.818/33.239/53.766`；pair mAP/AP50 `0.157715/0.320595`，both-independent mAP/AP50 `0.186071/0.345886`。相对父配置 cls/det HOTA `+0.523/+3.065`、DetA `+0.612/+0.785`，全部固定保护线通过，继续到 epoch 8；但同点低于 `0730_13`，shared-evidence 暂未显示正交增益。 |

当前四路正式训练为 99 `0730_14`、197 `0730_15`、252 `0730_13`、178 `0730_12`。
canonical 代码提交为 `0782826`；本次状态提交后四机统一快进，保留所有既有 artifacts 和
未跟踪目录。

## 2026-07-30 23:45 CST 178 门控失败与结构接替

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| STOPPED | 178 GPU 0 | `0730_12_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_motiontrust_sharedevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` | 2026-07-30 21:10 / 2026-07-30 23:30 | epoch 8 checkpoint、检测、2/2 TrackEval 与结构审计完整。cls HOTA/DetA/AssA `44.387/33.262/61.885`，det `49.824/40.708/63.485`，pair mAP/AP50 `0.2144/0.3809`，both-independent mAP/AP50 `0.2457/0.4098`。相对父配置 cls/det HOTA `-0.882/-0.369`、DetA `-4.401/-6.353`，pair mAP `-0.02333`、both AP50 `-0.05615`；AssA 增益不足以抵消检测损失，按固定保护线停止并释放 GPU。 |
| RUNNING | 178 GPU 0 | `0730_16_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_antisymmetricdetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` | 2026-07-30 23:48 /  | 接替结构只在逐帧 cls/reg head 前注入由真实双帧 cross-attention 证据生成的有界 `-detail/+detail`，保持 recurrent shared query、下一层路径和两帧特征中点严格等于父模型；不直接改框或共享 query。69 项 decoder 单测、配置深拷贝、launcher 审计和代码差异检查通过；真实数据 4-iter smoke 的总、DN、encoder loss 和 grad norm 均有限，checkpoint 三层 adapter 均非零且结构检查通过。正式训练五项门槛通过，23:49 到 epoch 1 iter 50，约 `0.9458 s/iter`、loss `21.1606`、grad norm `107.4478`，GPU 0 约 `31.4 GiB`，无异常。首判 epoch 4。 |

实时资源状态已恢复四路正式训练：252 `0730_13`、99 `0730_14`、197 `0730_15`、
178 `0730_16`。四台服务器结构基线为 `1112a56`，状态记录已同步至 `770eae7`；
各服务器既有未跟踪文件均保留，活动训练未因同步重启；AutoDL 继续保持全部关机。

## 2026-07-31 00:02 CST 252 epoch-4 门控

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 252 GPU 0,1 | `0730_13_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 22:32 /  | epoch 4 checkpoint、检测、完整 TrackEval 与结构审计通过。cls HOTA/DetA/AssA `37.559/27.119/55.846`，det `43.257/33.530/56.895`；pair mAP/AP50 `0.1547/0.3182`，both-independent mAP/AP50 `0.1839/0.3467`。相对父配置 cls/det HOTA `+1.350/+4.504`、DetA `+0.051/+1.076`；pair mAP 仅下降约 `0.00255`，未超过 `0.003` 保护线，both AP50 提高约 `0.0235`。6 组共享 attention 权重误差为零，18 组独立参数最大差异 `0.03846`。全部固定门槛通过，继续到 epoch 8。 |

99 `0730_14` 的 epoch 4 完整 TrackEval 与结构审计也已通过：cls
HOTA/DetA/AssA `37.075/27.355/53.989`，det `42.159/32.966/55.263`，
pair mAP/AP50 `0.1625/0.3105`，both-independent mAP/AP50 `0.1887/0.3369`；
全部保护线通过，继续到 epoch 8。但其 cls/det HOTA 同点低于 `0730_13`
`0.484/1.098`，motion-trust 组合暂未形成正交增益。197 `0730_15` 的 epoch 4
完整门控随后也通过并继续到 epoch 8；178 `0730_16` 正常训练。

## 2026-07-31 00:24 CST 197 epoch-4 门控

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 197 GPU 4,5 | `0730_15_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedevidence_sharedattention_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-30 22:58 /  | epoch 4 checkpoint、检测、完整 TrackEval 与联合结构审计已完成。cls HOTA/DetA/AssA `36.732/27.680/51.849`，det `41.818/33.239/53.766`；pair mAP/AP50 `0.157715/0.320595`，both-independent mAP/AP50 `0.186071/0.345886`。相对父配置 HOTA、DetA 与 AP 保护线全部通过，继续到 epoch 8。三层 shared-evidence adapter 均非零，6 组共享 attention 误差为零，18 组独立参数最大差异 `0.036013`。同点 cls/det HOTA 低于 `0730_13` `0.827/1.439`，因此 shared-attention 仍是当前主要早期增益来源。 |

当前四路并行保持：252 `0730_13`、99 `0730_14`、197 `0730_15` 均继续到
epoch 8；178 `0730_16` 首判 epoch 4。178 的 warmup grad norm 曾短时升至约
`543`，但到 epoch 2 iter 350/400 已回落至 `28.2/25.3`，loss 同步下降且无
NaN/OOM/Traceback，确认不是持续性训练异常。

## 2026-07-31 01:13 CST 178 接替与四路并行恢复

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| STOPPED | 178 GPU 0 | `0730_16_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_antisymmetricdetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` | 2026-07-30 23:48 / 2026-07-31 00:59 | epoch 4 cls HOTA/DetA/AssA `36.684/27.590/52.398`，det `39.221/31.788/49.436`；pair mAP/AP50 `0.1700/0.3110`，both-independent mAP/AP50 `0.1992/0.3428`。HOTA/AP 提高，但 det DetA 相对父配置下降 `0.666`，超过 `0.5` 门槛 `0.166`，因此精确停止。 |
| RUNNING | 178 GPU 0 | `0731_01_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_antisymmetricdetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` | 2026-07-31 01:11 /  | epoch 4 checkpoint、检测、完整 TrackEval 与两项结构审计均完成。cls HOTA/DetA/AssA `37.590/28.607/52.920`，det `40.313/33.923/49.759`；相对 `0727_01` 同点 HOTA `+1.381/+1.560`、DetA `+1.539/+1.469`。pair mAP `0.173430`、both-independent AP50 `0.356102`，分别提高 `0.016177/0.032953`；全门槛通过，继续到 epoch 8。 |

四路结构实验保持并行：99 `0731_02` 已到 epoch 3，252 `0731_03` 与 197
`0731_04` 均接近 epoch 3，178 `0731_01` 已通过 epoch 4 全门槛并进入 epoch 5。
四台代码和记录统一到 `7e184b1`，同步未重启任何在途训练。

## 2026-07-31 03:08 CST 99 epoch-4 全门槛

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 99 GPU 0,1 | `0731_02_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_envelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 01:44 /  | epoch 4 checkpoint、检测、完整 TrackEval 与结构审计均完成。cls HOTA/DetA/AssA `37.859/28.112/54.688`，det `42.873/34.173/55.071`；相对 `0727_01` 同点 HOTA `+1.650/+4.120`、DetA `+1.044/+1.719`。pair mAP `0.167896`、both-independent AP50 `0.349436`，分别提高 `0.010643/0.026287`。三层 enveloped-detail 权重均有限非零，全门槛通过，继续到 epoch 8。 |

当前仍为四路正式结构实验并行：99 `0731_02` 与 178 `0731_01` 已通过 epoch 4
全门槛并继续到 epoch 8；252 `0731_03`、197 `0731_04` 正等待 epoch 4 完整评估。

## 2026-07-31 03:12 CST 197 epoch-4 全门槛

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 197 GPU 4,5 | `0731_04_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_orthogonalevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 01:52 /  | epoch 4 checkpoint、检测、完整 TrackEval 与结构审计均完成。cls HOTA/DetA/AssA `36.831/27.861/52.246`，det `43.581/34.573/56.147`；相对 `0727_01` 同点 HOTA `+0.622/+4.828`、DetA `+0.793/+2.119`。pair mAP `0.161207`、both-independent AP50 `0.346363`，分别提高 `0.003954/0.023214`。两类三层门控均有限非零，全门槛通过，继续到 epoch 8。 |

99、197、178 三路已通过 epoch 4 全门槛并继续到 epoch 8；252 `0731_03`
仍在等待完整 epoch 4 评估，四台训练进程均保持运行。

## 2026-07-31 03:16 CST 252 epoch-4 门控与接替

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| STOPPED | 252 GPU 0,1 | `0731_03_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_commonevidencebypass_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 01:44 / 2026-07-31 03:15 | epoch 4 checkpoint、检测、完整 TrackEval 与结构审计均完成。cls HOTA/DetA/AssA `36.564/27.324/52.415`，det `43.279/34.694/55.415`；HOTA/DetA 均提高，但 pair mAP `0.153565` 相对父配置下降 `0.003688`，超过固定 `0.003` 保护线 `0.000688`。三层门控均有限非零，按门槛精确停止，GPU 0/1 已释放。 |
| PREPARED | 252 GPU 0,1 | `0731_05_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_envelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` |  /  | 接替结构组合 shared-attention 与受真实帧差包络约束的 head-only 细节，不含 common-evidence bypass、loss 调权、类别 reweight 或 residual-scale；等待单测、配置审计和真实数据 smoke。 |

当前 99、197、178 三路正式训练继续；252 已按门槛停止失败候选并进入结构接替验证。

## 2026-07-31 03:25 CST 252 接替实验启动

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 252 GPU 0,1 | `0731_05_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_envelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 03:23 /  | 79 项 decoder 单测、配置深拷贝、完整模型构建、双卡真实数据 4-iter smoke 与组合 checkpoint 审计通过。正式训练 03:25 到 epoch 1 iter 50：`1.1589 s/iter`、loss `21.4509`、grad norm `104.0930`，总/DN/encoder loss 有限；GPU0/1 各约 `19.2 GiB`、100% 利用，无 Traceback/OOM/NaN/NCCL。五项启动门槛通过，首判 epoch 4。 |

四路正式结构实验再次并行：252 `0731_05` 首判 epoch 4；99 `0731_02`、
197 `0731_04`、178 `0731_01` 继续各自的 epoch 8 持续性检查。

## 2026-07-31 03:40 CST 178 Epoch-8 Gate 与结构接替

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| STOPPED | 178 GPU 0 | `0731_01_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_antisymmetricdetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` | 2026-07-31 01:11 / 2026-07-31 03:37 | epoch 8 完整 artifacts 与结构审计完成。cls HOTA/DetA/AssA `45.152/37.611/57.666`，det `50.817/46.745/57.206`；相对父配置 cls HOTA `-0.117`、det HOTA `+0.624`，pair mAP `+0.008558`、both-independent AP50 `+0.013936`。因 cls HOTA 唯一低于父配置，按固定门槛精确停止。 |
| PREPARED | 178 GPU 0 | `0731_06_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_regressionenvelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` |  /  | 保留 shared-attention 分类共享路径，将零起点、受观测帧差包络约束的 swap-odd 细节只注入框回归/reference 更新；待单测、完整模型构建与真数据 4-iter smoke 后启动。 |

当前 99、197、252 三路正式训练不受影响；178 GPU 0 已释放并用于上述结构性接替，
不是 residual-scale、loss 权重或类别 reweight 调参。

## 2026-07-31 03:47 CST 178 接替实验正式启动

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 178 GPU 0 | `0731_06_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_regressionenvelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` | 2026-07-31 03:45 /  | 80 项单测、完整模型构建、单卡真数据 4-iter smoke 与组合结构审计通过；smoke 三层门控均获非零更新。03:46 到 epoch 1 iter 50：`0.9610 s/iter`、loss `20.9556`、grad norm `111.7257`，总/DN/encoder loss 有限，无训练异常，五项启动门槛通过。 |

四路正式结构实验恢复并行：99 `0731_02`、197 `0731_04` 继续 epoch 8，
252 `0731_05` 和 178 `0731_06` 首判 epoch 4。四台代码统一到 `62f1028`，
同步没有重启任何在途训练。

## 2026-07-31 04:35 CST epoch-8 淘汰与四机并行恢复

- 99 `0731_02 enveloped-detail` 的 epoch 8 完整结果为：cls HOTA/DetA/AssA
  `45.128/37.117/57.200`，det `50.223/44.663/58.189`，pair mAP/AP50
  `0.236938/0.434026`，both-independent mAP/AP50 `0.275284/0.473473`。
  相对 `0727_01` 同点，cls HOTA `-0.141`、cls DetA `-0.546`、det DetA
  `-2.398`，仅 det AssA `+3.044`；HOTA/DetA 门槛失败，完成全部 artifacts 后停止。
- 197 `0731_04 orthogonal-evidence` 的 epoch 8 完整结果为：cls HOTA/DetA/AssA
  `44.760/36.328/57.530`，det `49.732/44.183/57.992`，pair mAP/AP50
  `0.226459/0.424496`，both-independent mAP/AP50 `0.263142/0.463058`。
  相对父配置 cls/det HOTA 为 `-0.509/-0.461`，cls/det DetA 为
  `-1.335/-2.878`，pair mAP `-0.011275`；共同证据旁路进一步把检测覆盖搬运到
  AssA，故淘汰。评估完成后旧 launcher 曾继续到 epoch 9 iter 400 左右，04:22
  审计发现后精确停止；epoch 9 未形成新判断点，也不用于任何结论。
- 两台暂时空闲不是资源不足，而是两个候选刚完成固定 epoch-8 淘汰。接替结构采用同一
  `classification_enveloped_detail_decoder` 开关：框回归和迭代 reference 严格保持
  共享 decoder 路径，仅分类状态接收零起点、swap-odd、逐元素受观测帧差包络约束的细节。
  99 `0731_07` 不共享 attention，用于测量分类专用细节的主效应；197 `0731_08`
  叠加 shared-attention，用于检验能否保留其 det 增益并补回 cls。它们与 252 的
  full-path `0731_05`、178 的 regression-only `0731_06` 构成可归因的结构路径拆分，
  不含 scale、loss 权重或类别重权扫描。
- 82 项 decoder 单测全部通过；两个正式配置均通过深拷贝和完整模型构建。99/197
  双卡真数据 4-iter smoke 均生成 `iter_4.pth`，总、DN、encoder loss 和 grad norm
  有限。99 三层门控最大权重为 `0.000304/0.000270/0.000287`；197 的 6 组共享
  attention 误差为零、18 组独立参数最大差异 `0.000794`，三层门控为
  `0.000283/0.000295/0.000290`。
- 正式 fresh 训练均基于提交 `7dee533`：99 GPU 0/1 的 `0731_07` 于 04:34 到
  iter 50，`0.9703 s/iter`、loss `21.4241`、grad norm `112.0013`；197 仅使用
  GPU 4/5 的 `0731_08` 已越过 iter 100，iter 100 为 `0.8708 s/iter`、loss
  `20.5811`、grad norm `114.1662`。两项均无 Traceback、OOM、NaN、NCCL 或
  unused-parameter 错误。99、197、252、178 四台现均为 RUNNING，首判 epoch 4。

## 2026-07-31 05:44 CST 197 epoch-4 淘汰

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| STOPPED | 197 GPU 4,5 | `0731_08_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_classificationenvelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 04:33 / 2026-07-31 05:44 | epoch 4 checkpoint、检测、完整 TrackEval、原始 CSV 和结构检查全部完成。cls HOTA/DetA/AssA `36.312/27.192/52.399`，det `42.336/33.389/54.669`；pair mAP/AP50 `0.153285/0.312909`，both-independent mAP/AP50 `0.182831/0.343821`。双HOTA与DetA通过，但pair mAP相对父配置下降 `0.003968`，超过固定保护线，故精确停止。共享attention误差为零、独立参数及三层分类门控均已学习。 |

197 GPU 4/5 已释放。因99 `0731_07` 是同一分类专用细节的无 shared-attention
主效应对照，先等待其完整epoch 4结果，再决定197接替结构，避免在因果证据缺失时
启动新的参数或模块拼接实验。

## 2026-07-31 06:02 CST HOTA 优先门槛与四机并行恢复

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 252 GPU 0,1 | `0731_05_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_envelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 03:23 /  | epoch 4 双 HOTA 通过，继续到 epoch 8。 |
| RUNNING | 178 GPU 0 | `0731_06_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_regressionenvelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` | 2026-07-31 03:45 /  | epoch 4 双 HOTA 通过，继续到 epoch 8。 |
| RUNNING | 197 GPU 4,5 | `0731_08_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_classificationenvelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 04:33；06:00 resumed /  | 将 AP 降为诊断项后，从原 epoch 4 checkpoint 原位恢复；已确认 epoch 5 iter 100，GPU 4/5 正常。 |
| RUNNING | 99 GPU 0,1 | `0731_09_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_regressionenvelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 06:02 /  | 178 `0731_06` 的 2xb4 复现；84 项单测、模型构建、真实数据 smoke 与结构 checkpoint 检查通过，正式 fresh 启动。 |
| STOPPED | 99 GPU 0,1 | `0731_07_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_classificationenvelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 04:33 / 2026-07-31 05:50 | epoch 4 cls HOTA/DetA `35.533/26.341` 均低于父配置，pair mAP 下降 `0.006227`；即使 det HOTA 提高，仍不满足双 HOTA 目标，完成全部 artifacts 后停止。 |

后续筛选以同 epoch cls HOTA 与 det HOTA 为一级标准，DetA/AssA解释来源，AP仅诊断
明显检测崩塌。最终 decoder 只有同时超过 encoder `0727_01` 的
cls/det HOTA `54.437/62.393`，才进入论文性能递进主线。

## 2026-07-31 06:08 CST 178 epoch-8 结果与接替

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| STOPPED | 178 GPU 0 | `0731_06_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_regressionenvelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` | 2026-07-31 03:45 / 2026-07-31 06:08 | epoch 8 cls/det HOTA `44.398/48.552`，相对父配置同点下降 `0.871/1.641`；pair mAP `0.221531`、both-independent AP50 `0.430058`。完整 TrackEval、原始 CSV、检测 metrics 与结构审计齐全；双 HOTA 主门槛失败后停止。异步评估期间误入 epoch 9，其迭代不参与结论。 |
| RUNNING | 178 GPU 0 | `0731_11_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_midpointregressionenvelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_1xb8_fresh` | 2026-07-31 06:11 /  | 分类保持共享；帧细节在 5D box-logit residual 空间严格反对称，新增修正的 pair midpoint 为零。配置构建、代码测试、真数据 smoke 与结构 checkpoint 检查通过；06:12 达到 iter 50，`0.9405 s/iter`、loss `20.9608`，无异常。 |

## 2026-07-31 06:17 CST 252 epoch-8 阶段结果

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 252 GPU 0,1 | `0731_05_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_envelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 03:23 /  | epoch 8 cls HOTA/DetA/AssA `45.341/37.690/56.836`，det `51.589/45.176/60.817`；相对父配置双 HOTA `+0.072/+1.396`。pair mAP `0.2380`、both-independent AP50 `0.4738`。完整证据与结构审计通过，按 HOTA 主门槛保留；det 增益主要由 AssA 驱动、DetA 低 `1.885`，后续需持续观察。 |

## 2026-07-31 06:35 CST 四路进度与 terminal-only 候选

- 四路训练在代码同步前后均连续运行：99 `0731_09` 到 epoch 2 iter 900，197
  `0731_08` 到 epoch 7 iter 200，252 `0731_05` 到 epoch 10 iter 250，178
  `0731_11` 到 epoch 2 iter 450；总、DN、encoder loss 与 grad norm 均有限。
- 下一决策点依次为 197 epoch 8、99/178 epoch 4、252 epoch 12。197 若双 HOTA
  失败，优先使用已完成配置准备的 `0731_10 midpoint-regression` 双卡复现。
- 新增 `terminal_enveloped_detail_decoder`：前两层 decoder、辅助输出及 iterative
  references 保持共享父路径，只在最后一层的逐帧分类和回归输出前注入零起点、
  swap-odd、受真实帧差包络约束的细节。该结构直接隔离全路径 detail 的逐层
  reference 递归污染，不是 scale、loss 权重或类别重权扫描。
- 86 项 decoder 单测、配置深拷贝、完整模型构建及 detector-level 初始化后的零门控
  均已通过。此候选尚未分配实验编号、正式配置或 GPU，仅在现有评估提供支持且资源
  释放后进入 smoke。四台代码已无重启地快进到提交 `764ff7d`，未跟踪文件均保留。

## 2026-07-31 07:25 CST 197 淘汰与 midpoint 双卡接替

- 197 `0731_08 shared-attention + classification-only enveloped-detail` 的 epoch 8
  checkpoint、检测 metrics、完整 TrackEval、原始 CSV 和结构审计均已完成。
  cls HOTA/DetA/AssA 为 `43.801/35.436/56.745`，det 为
  `49.318/43.792/57.563`；相对 `0727_01` 同点的双 HOTA 分别下降
  `1.468/0.875`，cls/det DetA 分别下降 `2.227/3.269`。pair mAP
  `0.213045`、both-independent AP50 `0.436989` 也明显低于父配置。
- checkpoint 中 6 组共享 attention 最大误差为零，18 组独立参数最大差异
  `0.067752`，三层分类细节门最大权重
  `0.075650/0.106004/0.114755`。结构已充分学习，失败原因是分类专用细节使
  检测覆盖下降，而不是模块未生效。07:14 精确停止，GPU 4/5 释放。
- 接替实验 `0731_10` 将帧细节放在 5D box-logit residual 空间形成严格
  `-detail/+detail` 修正，使新增回归细节不移动 pair midpoint；分类路径继续共享。
  正式配置深拷贝、完整模型构建、launcher 语法和目标资源均通过复核。
- 197 双卡真实数据 4-iter smoke 的四步总、DN、encoder loss 与 grad norm
  均有限；6 组 attention 误差为零、18 组独立参数最大差异 `0.000788`，
  三层 midpoint 细节门最大权重 `0.000394/0.000391/0.000394`。07:18 fresh
  启动正式训练，07:20 达到 epoch 1 iter 100：`0.8707 s/iter`、loss
  `20.5465`、grad norm `101.3330`，GPU 4/5 正常，无训练异常。

## 2026-07-31 07:25 CST 99/178 epoch-4 双 HOTA 通过

- 99 `0731_09 regression-only` 的 epoch 4 cls HOTA/DetA/AssA 为
  `37.813/27.802/55.047`，det 为 `44.030/33.349/59.176`；相对父配置
  HOTA 提高 `1.604/5.277`、DetA 提高 `0.734/0.895`。pair mAP
  `0.170013`、both-independent AP50 `0.353940`，完整检测、TrackEval、原始
  CSV 和结构检查均通过，保留到 epoch 8。
- 178 `0731_11 midpoint-regression` 的 epoch 4 cls HOTA/DetA/AssA 为
  `38.668/30.574/53.235`，det 为 `43.586/38.232/51.254`；相对父配置
  HOTA 提高 `2.459/4.833`，cls DetA/AssA 提高 `3.506/1.141`，
  det DetA/AssA 提高 `5.778/3.788`。pair mAP `0.184185`、
  both-independent AP50 `0.391559`，结构审计通过，保留到 epoch 8。
- 99 回归分支与 178 midpoint 分支均在 HOTA、DetA、AssA 和 AP 上同步提高；
  当前不做参数扫描，继续用 epoch 8 判断早期覆盖增益能否保持。

## 2026-07-31 07:45 CST 0731_05 epoch-12 决策

- 252 `0731_05 shared-attention + full-path enveloped-detail` 的 epoch 12
  checkpoint、检测 metrics、完整 TrackEval、原始 CSV 与结构审计均已完成。
  cls HOTA/DetA/AssA 为 `50.171/42.372/61.238`，det 为
  `56.430/49.540/66.445`。
- 相对 `0727_01` 同点 `49.680/56.541`，cls HOTA 提高 `0.491`，det HOTA
  低 `0.111`，因此该点不构成严格双 HOTA 通过。分解上 cls DetA/AssA 为
  `+1.012/-0.572`，det DetA/AssA 为 `-0.805/+0.866`。
- det DetA 差距已由 epoch 8 的 `-1.885` 收窄至 `-0.805`，且 pair mAP
  `0.276140`、both-independent AP50 `0.534695` 分别高于父配置同点
  `0.273170/0.511752`。AP 只作覆盖诊断，不用于改写 HOTA 结论。
- 鉴于 det HOTA 仅低 `0.111`、检测覆盖差距正在收窄，且这是当前最成熟候选，
  保留到 epoch 16 作最后一次中期确认；若仍不能双超越父 encoder，则完成全量
  评估后停止并释放 252 给 terminal-only 结构。

## 2026-07-31 07:58 CST 0731_12 terminal-only 静态就绪

- 已为 252 预留 `0731_12 shared-attention + terminal-only enveloped-detail`
  的正式 2xb4 配置、4-iter 真数据 smoke 配置、正式 launcher 与 smoke launcher。
  它只在最终 decoder 输出前注入帧细节；前两层、辅助输出与 iterative references
  均保持 shared-attention 父路径。
- 两份配置均通过 `copy.deepcopy`；正式配置完整构建为
  `MultispecPairRotatedRTDETR`，且仅启用 `shared_attention_decoder=True` 与
  `terminal_enveloped_detail_decoder=True`。调用 detector `init_weights()` 后唯一
  terminal gate 的参数最大绝对值为零。
- 两份 launcher 均通过 `bash -n`，目标正式与 smoke workdir 在 252 上均确认不存在。
  当前状态仅为 `PREPARED`：未运行 smoke、未创建目录、未占 GPU，也未进入队列。
  只有 `0731_05` epoch 16 或其他完整 HOTA 证据触发接替时，才执行真实 DDP smoke。

## 2026-07-31 09:20 CST epoch-16 决策与 terminal 结构接替

- 252 `0731_05 full-path enveloped-detail` 的 epoch 16 完整结果为 cls
  HOTA/DetA/AssA `51.007/42.964/62.130`，det
  `57.940/50.762/68.403`。相对 encoder 同点 HOTA 为
  `-0.084/-0.380`；虽然 pair mAP `0.2853` 与 both-independent AP50
  `0.5439` 高于父配置，但一级双 HOTA 门槛失败。09:07 精确终止目标进程组，
  epoch 16 及此前全部 checkpoint/评测产物保留，GPU 0/1 确认释放。
- 99 `0731_09 regression-only` 的 epoch 8 cls/det HOTA 为
  `44.043/49.271`，相对 encoder 同点 `-1.226/-0.922`，完成结构审计后停止。
  99 已于 08:46 接替运行 `0731_12 terminal-only`；正式 iter 50 总、DN、
  encoder loss 与 grad norm 均有限，09:20 已进入 epoch 2。
- 197 `0731_10 midpoint-regression` 的 epoch 4 cls/det HOTA 为
  `38.794/44.142`，相对父配置 `+2.585/+5.389`，双 DetA/AssA 也全部提高，
  因而保留到 epoch 8；09:06 到 epoch 7 iter 650。
- 178 `0731_11 midpoint-regression` 的 epoch 8 cls/det HOTA 为
  `45.173/51.263`，相对父配置 `-0.096/+1.070`。cls 只窄幅落后、det 增益明确，
  尚不足以淘汰；保留到 epoch 12，09:05 到 epoch 11 iter 350。
- 新增 `0731_13 terminal-midpoint enveloped-detail`：分类只在最终层接收
  bounded swap-odd 帧细节；回归也只在最终层接收帧细节，并把新增 5D box-logit
  residual 显式构造成严格 `-detail/+detail`，使 pair midpoint 不发生漂移。
  前两层、辅助输出和递归 reference 与 shared-attention 父路径逐元素一致。
- `0731_13` 的针对性单测、配置深拷贝、完整模型构建、launcher 语法、252 真数据
  双卡 4-iter smoke 与 checkpoint 结构验收均通过。smoke 中 6 组 attention
  最大误差为零，18 组独立参数最大差异 `0.000794`，唯一 terminal gate 最大权重
  `0.000393`。09:18 fresh 启动正式训练，09:19 达到 iter 50：
  loss `22.1957`、grad norm `115.9534`，GPU 0/1 各约 `19.2 GiB`，无
  Traceback/OOM/NaN/NCCL。代码提交为 `d9c97e0`。

## 2026-07-31 09:50 CST midpoint 决策与 HOTA 优先恢复

- 197 `0731_10 midpoint-regression` epoch 8 的 cls/det HOTA 为
  `45.254/50.846`，相对 encoder 同点 `-0.015/+0.653`；178 同结构
  `0731_11` epoch 12 为 `49.478/56.451`，相对同点 `-0.202/-0.090`。
  两项 checkpoint、检测、完整 TrackEval 与原始 CSV 均已落盘，09:47 精确停止，
  GPU 释放后未残留目标训练进程。
- 按用户最新的 cls/det HOTA 主指标规则，重新恢复此前因 DetA/mAP 次要保护线
  提前停止、但 HOTA 当时双过的两个结构方向。197 从 `0730_09 motion-trust`
  epoch 8 恢复；该点相对父配置 HOTA `+0.229/+0.967`，09:49 已到 epoch 9
  iter 50，继续到 epoch 12。178 从 `0730_16 antisymmetric-detail` epoch 4
  恢复；该点相对父配置 HOTA `+0.475/+0.468`，09:50 已到 epoch 5 iter 50，
  继续到 epoch 8。
- 两个恢复均加载原 optimizer、EMA、epoch 与 iter，不是 fresh 重跑。178 当前
  PyTorch 2.6 对旧可信 MMEngine checkpoint 的 weights-only 默认值不兼容，只在
  恢复脚本中设置 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`，未修改模型、loss 或配置。
  两项正式 iter 50 的总/DN/encoder loss 与 grad norm 均有限，无
  Traceback/OOM/NaN；197 仍严格只使用 GPU 4/5。

## 2026-07-31 10:04 CST 99 terminal-only epoch-4 结果

- `0731_12` epoch 4 cls HOTA/DetA/AssA 为 `37.799/28.644/53.288`，
  det 为 `43.483/34.398/56.345`；相对 encoder 同点 HOTA
  `+1.590/+4.730`，且 cls/det 的 DetA、AssA 四项全部提高。
- pair mAP `0.165773`、both-independent AP50 `0.355169`，未显示检测覆盖崩塌。
  结构检查确认 6 组 shared attention 严格相同、18 组独立参数已分化、唯一
  terminal gate 有限非零。checkpoint、检测、TrackEval 与原始 CSV 均完整，
  因而继续到 epoch 8；10:03 已进入 epoch 5。

## 2026-07-31 10:49 CST 252 terminal-midpoint epoch-4 结果

- `0731_13` epoch 4 cls HOTA/DetA/AssA 为 `36.148/27.020/51.933`，
  det 为 `43.247/33.958/56.389`；相对 encoder 同点分别为
  `-0.061/-0.048/-0.161` 与 `+4.494/+1.504/+8.923`。因此当前不记为
  双 HOTA 提升，但 cls 差距极窄，det 的检测与关联组成量均明显提高。
- pair mAP `0.153978`，相对父配置约 `-0.003275`；both-independent AP50
  `0.335487`，相对约 `+0.012338`。按当前规则 AP 仅作诊断，未显示与 det HOTA
  同向的检测崩塌。
- checkpoint、检测 metrics、完整 TrackEval 和原始 CSV 均已落盘。结构检查确认
  6 组 shared attention 最大误差为零、18 组独立参数最大差异 `0.037019`，
  唯一 terminal-midpoint gate 最大权重 `0.081010`，排除模块未学习。
- 同 epoch 它低于 99 `0731_12 terminal-only` `1.651/0.236`，说明新增 midpoint
  约束尚未形成早期正交增益；但其 cls 仅低父配置 `0.061`，且该结构的核心假设是避免
  中期递归污染，因此允许继续到 epoch 8 做一次稳定性判定。epoch 8 若仍非双 HOTA
  通过，则停止，不再延长。

## 2026-07-31 11:12 CST 178/197 恢复实验收口

- 197 `0730_09 motion-trust` 恢复后的 epoch 12 完整结果为：cls
  HOTA/DetA/AssA `48.877/40.354/61.309`，det
  `55.819/48.907/65.923`。相对 encoder 同点 HOTA `-0.803/-0.722`，
  DetA `-1.006/-1.438`，仅 det AssA `+0.344`。pair mAP/AP50
  `0.263196/0.479269`，both-independent mAP/AP50
  `0.300922/0.513250`。三层 motion-trust 权重最大值
  `0.453352/0.294524/0.256910`，结构已充分学习但双 HOTA 失败，已停止并释放
  GPU 4/5。
- 178 `0730_16 antisymmetric-detail` 恢复后的 epoch 8 完整结果为：cls
  HOTA/DetA/AssA `44.591/37.157/56.398`，det
  `49.685/46.466/54.883`。相对 encoder 同点 HOTA `-0.678/-0.508`，
  DetA `-0.506/-0.595`，AssA `-0.903/-0.262`。pair mAP/AP50
  `0.241645/0.435966`，both-independent mAP/AP50
  `0.279185/0.474553`。三层结构权重最大值
  `0.035530/0.036084/0.034844`，同样是方向失败而非结构未学习，已停止并释放
  GPU 0。
- 两项恢复评估复用了旧 `val_track_0001` 目录，已分别复制保留为
  `val_track_epoch12_resume_20260731` 与
  `val_track_epoch08_resume_20260731`。共享存储元数据时钟与登录节点约有 25 分钟
  偏移，结果通过 checkpoint、检测 metrics、TrackEval metrics 与原始 CSV 的
  写入顺序和内容交叉确认。99 `0731_12` 的 epoch 8 训练已完成并等待完整评估；
  252 `0731_13` 在 epoch 6 稳定运行。178/197 暂时保留，待 99 同点结果后部署
  下一项有因果依据的结构实验。

## 2026-07-31 11:33 CST 99 epoch-8 决策与新结构准备

- 99 `0731_12 terminal-only` epoch 8 的完整结果为 cls
  HOTA/DetA/AssA `43.178/35.246/55.355`，det
  `50.010/43.700/59.276`。相对 encoder 同点 HOTA `-2.091/-0.183`，
  DetA `-2.417/-3.361`，cls AssA `-1.946`，仅 det AssA `+4.131`。
  pair mAP/AP50 `0.217536/0.410164`，both-independent AP50 `0.448156`，
  也未显示检测覆盖保护。完整 checkpoint、检测、TrackEval、原始 CSV 与结构审计
  后停止，GPU 0/1 释放。
- checkpoint 中 6 组 shared attention 误差为零、18 组独立参数最大差异
  `0.063149`、唯一 terminal gate 最大权重 `0.117809`，排除模块未学习。
  结论是：去除递归污染仍不够，最终分类分数专门化本身会改变匹配与筛选并损伤
  DetA；不能继续用缩小 residual scale 掩盖该结构问题。
- 新增两项因果对照。`0731_14` 只在最终框输出使用 bounded frame detail，分类
  hidden state 在所有层严格等于 shared parent；`0731_15` 进一步把最终新增框
  residual 构造成严格反对称，使 pair midpoint 不变。两项均不改 encoder、
  proposal、PairDN、head、loss 或训练协议。
- 89 项 decoder 单测、两份正式和两份 smoke 配置深拷贝、两个完整模型构建及四个
  launcher 语法检查通过。下一步在 99/197 各执行真实双卡 4-iter smoke；只有
  checkpoint 结构、有限 loss 与正式 iter 50 五项门槛全部通过才记为 `RUNNING`。

## 2026-07-31 11:45 CST terminal regression 两项正式启动

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| RUNNING | 197 GPU 4,5 | `0731_15_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_terminalmidpointregressionenvelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 11:43 /  | 4-iter 真实数据 DDP smoke 的总、DN、encoder loss 与 grad norm 全部有限；checkpoint 中 6 组 attention 严格共享、18 组独立参数最大差异 `7.79018e-4`，terminal midpoint gate 最大值 `3.92011e-4`。正式 fresh 训练于 11:45 到 epoch 1 iter 50，`0.8932 s/iter`、loss `22.1811`、grad norm `124.8561`，GPU 4/5 各约 `19.2 GiB`，无 Traceback/OOM/NaN/NCCL/DDP 异常。 |
| RUNNING | 99 GPU 0,1 | `0731_14_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_terminalregressionenvelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 11:43 /  | 4-iter 真实数据 DDP smoke 的总、DN、encoder loss 与 grad norm 全部有限；checkpoint 中 6 组 attention 严格共享、18 组独立参数最大差异 `7.85518e-4`，terminal gate 最大值 `3.92929e-4`。正式 fresh 训练于 11:45 到 epoch 1 iter 50，`0.9528 s/iter`、loss `22.1461`、grad norm `126.9356`，GPU 0/1 各约 `19.2 GiB`，无 Traceback/OOM/NaN/NCCL/DDP 异常；GPU 2 上外部任务未触碰。 |

- 两项实验均基于提交 `1575d96`，99、197、252、178 已验证为同一 commit；
  252 的既有 `0731_13` 进程保持运行且未重启。首个统一科学决策点为 epoch 4，
  主判据仍是相对 `0727_01` 同 epoch 的 cls/det HOTA，DetA/AssA 用于归因，
  mAP 仅作检测崩塌诊断。

## 2026-07-31 12:59 CST 0731_14/15 epoch-4 双提升

- 两项实验的 epoch 4 checkpoint、检测 metrics、完整 TrackEval、54 个原始
  TrackEval 输出及结构检查均已形成。固定父配置 `0727_01` 同点为：cls
  HOTA/DetA/AssA `36.209/27.068/52.094`，det
  `38.753/32.454/47.466`；pair mAP/AP50 `0.157253/0.296134`，
  both-independent mAP/AP50 `0.184465/0.323149`。
- 99 `0731_14 terminal regression-only` 的 cls 为
  `37.209/27.864/53.684`，det 为 `43.646/34.679/56.188`。相对父配置
  cls/det HOTA `+1.000/+4.893`，DetA `+0.796/+2.225`，AssA
  `+1.590/+8.722`。pair mAP/AP50 `0.163485/0.323237`，
  both-independent mAP/AP50 `0.192363/0.351268`，均无检测退化。
  checkpoint 中 6 组 shared attention 误差为零、18 组独立参数最大差异
  `0.031446`、terminal gate 最大权重 `0.070805`。
- 197 `0731_15 terminal midpoint-regression` 的 cls 为
  `38.153/27.923/55.246`，det 为 `44.506/34.409/59.052`。相对父配置
  cls/det HOTA `+1.944/+5.753`，DetA `+0.855/+1.955`，AssA
  `+3.152/+11.586`。pair mAP/AP50 `0.161120/0.318966`，
  both-independent mAP/AP50 `0.189558/0.346678`，也均提高。
  checkpoint 中 shared attention 误差为零、独立参数最大差异 `0.029741`、
  terminal midpoint gate 最大权重 `0.068100`。
- 两者均通过 HOTA 主门槛并继续到 epoch 8。197 的早期 HOTA 更高，但更依赖
  AssA；99 的 det DetA 更高、AssA 搬运较少，因此 epoch 8 重点检查哪条路径能
  避免历史上的 DetA→AssA 中期退化。

## 2026-07-31 14:38 CST epoch-8 淘汰与严格正交结构接替

| Status | 服务器/资源 | 实验 | 开始/结束 | 进度或说明 |
| --- | --- | --- | --- | --- |
| STOPPED | 252 GPU 0,1 | `0731_03_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_commonevidencebypass_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 01:44 / 2026-07-31 14:08 前 | epoch 8 cls/det HOTA `44.798/50.415`，相对 encoder 同点 `-0.471/+0.222`；cls DetA/AssA `-1.030/-0.209`，det `-2.602/+3.680`。完整产物和结构审计齐全后停止。 |
| STOPPED | 99 GPU 0,1 | `0731_14_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_terminalregressionenvelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 11:43 / 2026-07-31 14:08 前 | epoch 8 cls/det HOTA `44.321/49.059`，相对父配置 `-0.948/-1.134`；DetA 分别下降 `1.901/3.480`。模块已学习，停止并保留 epoch 4/8 全部产物。 |
| STOPPED | 197 GPU 4,5 | `0731_15_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_terminalmidpointregressionenvelopeddetail_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 11:43 / 2026-07-31 14:08 前 | epoch 8 cls/det HOTA `43.918/49.071`，相对父配置 `-1.351/-1.122`；DetA 分别下降 `2.925/4.146`。与`0731_14`形成独立重复，排除偶然波动。 |
| STOPPED_INVALID | 252 GPU 0,1 | `0731_17_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_terminalfactorizedevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 14:05 / 2026-07-31 14:14 前 | 正式 epoch 1 iter 300 前后审计发现 boxes 仍以`common_output`为基底，不满足严格正交设计；首个正式 epoch 前停止，不进入实验结果。 |
| RUNNING | 252 GPU 0,1 | `0731_18_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_terminalorthogonalfactorizedevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 14:15 /  | 分类只接收共同证据，boxes 只接收严格反对称 detail；共同证据不改变任一 box reference，detail midpoint 为零。98 项单测、完整构建、真实 2 卡 smoke 和正式五项门槛通过；14:38 到 epoch 2 iter 50。 |
| RUNNING | 99 GPU 0,1 | `0731_19_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_terminalclassificationcommonevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 14:30 /  | 不共享 decoder attention，只有末层 classification 接收共同证据；boxes、aux 与 recurrent references 严格保持父模型。100 项单测、完整构建、真实 2 卡 smoke 和正式五项门槛通过；14:38 到 epoch 1 iter 400。 |
| RUNNING | 197 GPU 4,5 | `0731_20_paper_base_liquid_encoder_p5temporal_dualevidence_decoder_sharedattention_terminalclassificationcommonevidence_pairdn_paircoherent_le180_r18_coco_full_1200x900_bf16_2xb4_fresh` | 2026-07-31 14:33 /  | 与`0731_19`唯一核心结构差异是共享 decoder attention；同样只改变末层 classification。100 项单测、完整构建、真实 2 卡 smoke 和正式五项门槛通过；14:38 到 epoch 1 iter 250。 |

- 178 `0731_16` epoch 4 为 cls/det HOTA `37.750/43.723`，相对父配置
  `+1.541/+4.970`，但 DetA `-0.565/-0.841`、AssA `+5.091/+14.900`，
  显示明显 DetA→AssA 搬运；14:38 已进入 epoch 8，等待完整结果后再决定是否停止。
- 当前四台服务器 tracked HEAD 均为干净的 `ac9d629`。178 运行时由
  `d78500d` 启动，252 运行时由 `ad99b0d` 启动；仓库后续快进不改变已加载进程。

## 2026-07-31 15:03 CST 0731_16 epoch-8 淘汰

- 178 `0731_16 terminal common-evidence bypass` 的 epoch 8 checkpoint、检测
  metrics、完整 TrackEval、54 个原始输出文件和结构审计均已形成。cls
  HOTA/DetA/AssA 为 `43.972/32.412/62.419`，det 为
  `49.378/39.704/63.985`。
- 相对 `0727_01` 同点，cls HOTA/DetA/AssA 为
  `-1.297/-5.251/+5.118`，det 为 `-0.815/-7.357/+8.840`。epoch 4 的
  早期双 HOTA 增益完全没有保持，且两路都出现强烈的 DetA→AssA 搬运。
- pair mAP/AP50 为 `0.205329/0.366688`，相对父配置下降
  `0.032405/0.064222`；both-independent mAP/AP50 为
  `0.236540/0.395712`，下降 `0.039642/0.070239`。检测诊断与 DetA 结论一致。
- checkpoint 中唯一 terminal-common gate 最大权重为 `0.032061`，epoch/iter
  为 `8/8304`，排除结构未学习。15:03 精确关闭唯一 screen，训练进程退出，
  GPU0 为 `0%/1 MiB`；epoch 4/8 全部产物保留，GPU1 未触碰。

## 2026-07-31 15:14 CST 0731_21 正式启动

- 当前三项严格隔离实验构成了 `shared attention × antisymmetric box detail` 的
  2×2 结构设计，但缺少“独立 attention + 分类共同证据 + 反对称 box detail”。
  `0731_21` 补齐该单元：分类只接收末层共同证据，boxes 只接收严格反对称
  5D detail residual；auxiliary output 与 recurrent references 保持父路径，
  两帧 attention-weight 投影保持独立。
- 原实现人为要求 factorized evidence 必须开启 shared attention。审计确认其
  common/antisymmetric 代数、box midpoint 守恒和父路径隔离并不依赖参数共享后，
  提交 `a77e135` 移除该不必要绑定并增加独立 attention 专项测试。完整 100 项
  decoder 单测、正式/smoke 配置深拷贝、完整模型构建、launcher 语法及目标环境
  数据/GMC/预训练权重检查全部通过。
- 178 GPU0 的真实 4-iter smoke 最终 loss/grad norm 为
  `19.8328/67.2125`，总、DN、encoder loss 有限。checkpoint 中 6 组独立
  attention-weight 最大差异 `7.67466e-4`，common/detail terminal gate 最大值
  分别为 `3.97146e-4/3.92367e-4`；结构已更新且 midpoint 单测通过。
- 15:13 fresh 正式启动，15:14 到 epoch 1 iter 50：学习率
  `2.5488e-6`、time `0.9401 s/iter`、loss `20.9349`、grad norm
  `90.7400`；GPU0 约 31.4 GiB，总、DN、encoder loss 有限，无
  Traceback/OOM/NaN/NCCL。GPU1 未触碰，首判 epoch 4。
