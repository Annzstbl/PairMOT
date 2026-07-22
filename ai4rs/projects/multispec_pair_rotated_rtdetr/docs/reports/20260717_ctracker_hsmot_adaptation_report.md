# CTracker适配HSMOT报告

为将原版CTracker作为HSMOT对比方法，在尽量不改变其核心方法的前提下完成了必要适配：数据侧复用主线`HSMOTPairDataset`读取相邻帧8波段图像及track ID，并统一采用`900×1200`输入尺度（按32倍数padding后为`928×1216`）；模型侧将ResNet首层替换为最基础的谱维`Conv3d(3×7×7)`加像素级SE加权结构，把原CTracker RGB首层权重无损映射为Conv3d权重，其余可兼容预训练参数继续从原版`model_final.pt`加载，同时将分类分支扩展为HSMOT八类、将成对水平框回归改为成对`le90`旋转框回归，并以旋转框delta Smooth L1、解码后KLD及原有关联损失联合训练；训练协议保持原版CTracker的ResNet-50、Adam、100 epochs、有效batch size 8、梯度裁剪和ReduceLROnPlateau设置，仅对新增3D Stem参数使用10倍学习率。推理跟踪仍保留原版CTracker的相邻帧pair框关联、跨漏帧历史速度外推、Hungarian匹配、0.4置信度阈值、0.5 IoU阈值和10帧保留策略，仅将IoU/NMS替换为类别约束下的旋转IoU/旋转Gaussian Soft-NMS，并按主线格式分别保存`val_det` pair检测、13列旋转框跟踪结果以及TrackEval的HOTA/CLEAR/Identity评估结果。

## 178服务器存储与恢复

178训练数据实际位于本地NVMe `/data1/users/litianhao01/data/hsmot`，训练图像使用`npy2jpg`下的三组JPG表示，标注使用`mot`目录。原训练将逐iteration无缓冲日志和每epoch约485 MB checkpoint直接写入NFS挂载`/data4`，在NFS服务端间歇性阻塞时会使训练主进程进入`wait_on_page_bit_common`并停止使用GPU。现已将活动工作目录切换到本地`/data1/users/litianhao01/PairMOT/workdir_178`：checkpoint先写入临时文件并以`os.replace`原子发布，独立归档进程再使用`rsync --remove-source-files`异步传输完整的具名epoch checkpoint到`/data4`，只有传输成功后才删除本地源文件；NFS阻塞只会暂停归档，不再阻塞训练。活动`checkpoint_latest.pt`在训练期间保留并覆盖为单个本地文件，训练结束后与`model_final.pt`和`train.log`一并归档并从本地移除。

## 178服务器训练结果（100 epochs）

正式实验`ctracker_hsmot_r50_3dse_rotated_1200x900_bs4_acc2`已完成全部100 epochs。训练集包含8,297个pair，单卡micro batch为4、梯度累积为2，有效batch size为8；每轮2,075个micro-batch iteration，共207,500 iterations，约103,800次参数更新。100个epoch checkpoint、`checkpoint_latest.pt`、`model_final.pt`及日志均已归档至`/data4/litianhao/PairMmot/workdir_178/ctracker_hsmot_r50_3dse_rotated_1200x900_bs4_acc2`，训练产物总占用约47 GB。

旧的前91轮逐iteration日志在NFS故障恢复时被覆盖，但各epoch checkpoint完整保留了`ReduceLROnPlateau`状态，可据此还原全程的epoch最优损失：epoch 1/10/20/40/60/80/90/100分别为9.4110、3.9701、3.3759、2.8219、2.5505、2.3592、2.2909和2.2228，首末下降约76.4%。所有抽查checkpoint的`num_bad_epochs`均为0，基础参数和3D Stem的学习率始终分别为`5e-5`和`5e-4`，表明每个阶段仍持续刷新最优值，没有触发学习率衰减。恢复后可用的epoch 92--100完整日志中未出现NaN/Inf；四项均值从首轮到末轮分别为分类0.1179→0.1145、旋转框delta回归1.8845→1.8383、KLD 0.1702→0.1644、关联0.1087→0.1056，running loss为2.2759→2.2265，后段仍保持缓慢下降，训练数值稳定。

从checkpoint时间戳看，正常阶段中位耗时约26.1分钟/epoch；epoch 88--92受NFS阻塞影响，单轮间隔一度达到60--169分钟。切换到本地NVMe活动目录后，epoch 93--100稳定在约20.9分钟/epoch。由首个epoch checkpoint至最终checkpoint为51.33小时，加上首轮约24分钟后完整墙钟时间约51.7小时；额外耗时来自存储阻塞而非计算或数值异常，恢复时模型、优化器、调度器和iteration均连续加载。

## 最终推理与TrackEval结果

使用`model_final.pt`、`score_threshold=0.4`、旋转IoU阈值0.5和10帧保留策略在全部50个测试序列上完成推理。为缩短等待时间，推理仅在序列粒度进行多进程/双卡切分，不拆分序列内部时序，也不改变模型、阈值或关联逻辑；合并后已核验50个跟踪文件和50个`val_det` pair检测文件与GT序列名一一对应。最终跟踪输出为115,879条，相比epoch 9的59,239条增加95.6%。TrackEval打印的`Tracking data leaks ... timesteps`表示相应帧没有置信度超过阈值的输出行，评估时按空预测处理，并非越界时间步或末帧重复。

| 统计口径 | HOTA e9→e100 | DetA e100 | AssA e100 | MOTA e9→e100 | IDF1 e9→e100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 八类宏平均 | 10.32→34.15 | 31.28 | 40.11 | 5.17→24.57 | 8.95→34.28 |
| 检测数加权 | 23.64→42.30 | 39.61 | 47.34 | 14.60→35.57 | 18.47→39.74 |
| HUMAN | 7.39→13.99 | 24.34 | 8.53 | 2.86→13.00 | 7.03→14.49 |
| VEHICLE | 34.41→61.69 | 61.74 | 62.04 | 29.49→64.32 | 30.63→64.09 |
| BIKE | 9.50→22.53 | 19.27 | 27.08 | 1.44→9.63 | 7.22→20.44 |

| 类别 | HOTA e9→e100 | MOTA e9→e100 | IDF1 e9→e100 |
| --- | ---: | ---: | ---: |
| car | 36.85→64.02 | 33.96→69.38 | 33.65→66.57 |
| pedestrian | 7.39→13.99 | 2.86→13.00 | 7.03→14.49 |
| bike | 12.03→24.35 | 0.48→7.77 | 9.74→21.80 |
| van | 4.73→45.56 | 0.20→33.35 | 2.84→45.57 |
| truck | 1.17→29.26 | -2.52→4.53 | 0.67→24.42 |
| bus | 12.88→59.15 | 3.49→58.01 | 11.74→68.09 |
| tricycle | 1.62→14.48 | -1.06→-10.19 | 0.52→11.43 |
| awning-bike | 5.92→22.40 | 3.96→20.73 | 5.39→21.90 |

最终模型相对epoch 9在所有类别的HOTA和IDF1上均有提升，且加权检测召回`DetRe`从21.73%升至43.38%、检测精度`DetPr`从71.97%升至73.45%，说明输出量接近翻倍主要转化为有效召回，并未以整体精度下降为代价。车辆已经形成较强基线，car/bus/van的HOTA分别达到64.02/59.15/45.56；总体仍由检测限制（加权DetA 39.61低于AssA 47.34）。但pedestrian表现相反，DetA为24.34而AssA仅8.53，低帧率、密集遮挡下的pair关联是HUMAN的首要瓶颈。tricycle虽然HOTA和IDF1提高，但MOTA下降到-10.19，表明新增召回尚不足以抵消FP和ID切换；truck的HOTA已到29.26而MOTA仅4.53，也仍受检测错误影响。绝对IDSW从12,713增至16,483、Frag从12,553增至16,567，部分来自覆盖目标数大幅增加，但也说明行人和小类的轨迹碎片化仍需作为后续分析重点。

最终完整结果位于`/data4/litianhao/PairMmot/workdir_178/ctracker_hsmot_r50_3dse_rotated_1200x900_bs4_acc2/final_inference`，其中`metrics.json`为汇总指标，`trackeval_stdout.log`为完整评估日志，`trackers/ctracker_epoch100/preds`为旋转框跟踪结果，`val_det`为pair检测结果。
