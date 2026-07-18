# CTracker适配HSMOT报告

为将原版CTracker作为HSMOT对比方法，在尽量不改变其核心方法的前提下完成了必要适配：数据侧复用主线`HSMOTPairDataset`读取相邻帧8波段图像及track ID，并统一采用`900×1200`输入尺度（按32倍数padding后为`928×1216`）；模型侧将ResNet首层替换为最基础的谱维`Conv3d(3×7×7)`加像素级SE加权结构，把原CTracker RGB首层权重无损映射为Conv3d权重，其余可兼容预训练参数继续从原版`model_final.pt`加载，同时将分类分支扩展为HSMOT八类、将成对水平框回归改为成对`le90`旋转框回归，并以旋转框delta Smooth L1、解码后KLD及原有关联损失联合训练；训练协议保持原版CTracker的ResNet-50、Adam、100 epochs、有效batch size 8、梯度裁剪和ReduceLROnPlateau设置，仅对新增3D Stem参数使用10倍学习率。推理跟踪仍保留原版CTracker的相邻帧pair框关联、跨漏帧历史速度外推、Hungarian匹配、0.4置信度阈值、0.5 IoU阈值和10帧保留策略，仅将IoU/NMS替换为类别约束下的旋转IoU/旋转Gaussian Soft-NMS，并按主线格式分别保存`val_det` pair检测、13列旋转框跟踪结果以及TrackEval的HOTA/CLEAR/Identity评估结果。

## 178服务器存储与恢复

178训练数据实际位于本地NVMe `/data1/users/litianhao01/data/hsmot`，训练图像使用`npy2jpg`下的三组JPG表示，标注使用`mot`目录。原训练将逐iteration无缓冲日志和每epoch约485 MB checkpoint直接写入NFS挂载`/data4`，在NFS服务端间歇性阻塞时会使训练主进程进入`wait_on_page_bit_common`并停止使用GPU。现已将活动工作目录切换到本地`/data1/users/litianhao01/PairMOT/workdir_178`：checkpoint先写入临时文件并以`os.replace`原子发布，独立归档进程再使用`rsync --remove-source-files`异步传输完整的具名epoch checkpoint到`/data4`，只有传输成功后才删除本地源文件；NFS阻塞只会暂停归档，不再阻塞训练。活动`checkpoint_latest.pt`在训练期间保留并覆盖为单个本地文件，训练结束后与`model_final.pt`和`train.log`一并归档并从本地移除。
