# DiffusionTrack适配HSMOT修改与测试报告

## 1. 目标与适配边界

本次工作将原版DiffusionTrack接入HSMOT八波段多光谱旋转目标跟踪任务。适配遵循“保留原论文方法，只修改输入、框表示及数据集相关接口”的原则，具体边界如下：

- 保留原版Pair Diffusion检测与关联流程、Dynamic-K匹配、Pair Cluster-NMS和两阶段训练方式。
- 保留原版默认的`diffusion_tracker_kl.py` Kalman跟踪链路，包括活动轨迹的Pair Diffusion传播、丢失轨迹的Kalman预测、检测恢复及`max_time_lost=30`设置。
- 不采用Linxu后续版本中定期将历史帧与当前帧重新组成pair并再次diffusion的推理改动。
- 不引入新的RPN实验。
- 仅进行HSMOT所必需的八波段输入、YOLO11L骨干、MMOT官方ConvMSI stem、旋转框、多类别、旋转匹配/损失及输出评估适配。

## 2. 网络结构修改

### 2.1 YOLO11L骨干

新增`yolox/models/yolo11_backbone.py`，采用与Linxu版本一致的Ultralytics YOLO11L OBB网络结构，但不复用其八通道2D stem。适配器跳过YOLO自身的OBB检测头，提取P3、P4、P5特征供原DiffusionTrack扩散头使用，输出通道分别为`256/512/512`。

HSMOT输入尺寸为`900×1200`，其中900不能被YOLO最大stride 32整除。为保持对外输入和旋转框坐标仍为`900×1200`，仅在YOLO骨干内部对右侧和下侧补齐至`928×1216`，不改变数据集、损失和输出坐标空间。

### 2.2 MMOT官方ConvMSI stem

根据最终要求，删除此前Base Conv3D+SE及其SE分支，将YOLO11首层改为与MMOT官方checkpoint完全相同的ConvMSI结构：

- 输入格式为`B×8×H×W`，将八个波段作为Conv3D的谱维。
- Conv3D参数为`1→64`、kernel=`(3,3,3)`、stride=`(1,2,2)`、padding=`(1,1,1)`。
- Conv3D后依次执行BN3d和SiLU。
- 使用`64→64`、kernel=`(8,1,1)`、groups=`64`的depthwise Conv3D将八个谱响应融合为深度1。
- squeeze谱维后执行BN2d和SiLU，输出`B×64×H/2×W/2`。
- 不再包含`se_conv1/se_conv2`或任何SE分支。
- Det阶段默认使用MMOT官方`yolo11L-8ch-3dstem.pt`作为起点，ConvMSI的`conv3d/bn3d/fuse/bn2d`全部严格继承，YOLO11其余骨干、neck和原OBB head参数也全部载入checkpoint；DiffusionTrack实际跳过原OBB head并使用自己的旋转扩散头。

MMOT官方权重位于`/data4/litianhao/PairMmot/pretrained_weights/mmot_official/yolo11L-8ch-3dstem.pt`，Det配置默认读取该文件，也可通过环境变量`YOLO11_WEIGHTS`覆盖。checkpoint包含仓库自定义的`ConvMSI`类型，当前pip版Ultralytics不能直接反序列化，因此在`yolo11_backbone.py`中实现同构ConvMSI并将其临时注册到checkpoint要求的Python模块位置；该操作只发生在内存中，不修改已安装的Ultralytics文件。Det阶段强制要求该MMOT权重存在，缺失时直接报错，不再静默回退到RGB或随机初始化。

### 2.3 扩散头旋转框化

修改`diffusion/models/diffusion_head.py`、`diffusion_models.py`和`diffusionnet.py`：

- 扩散变量由水平框4维扩展为旋转框5维`(cx,cy,w,h,theta)`。
- 角度采用`le90`、弧度制，对应扩散归一化空间中的`[0,1]`。
- 回归分支由4维delta扩展为5维delta；`dx/dy/dw/dh/dtheta`是相对proposal的残差，因此输出层和回归MLP的bias均初始化为0，使初始解码为恒等变换。分类先验bias仍保持原版设置。
- 解码后的角度统一回绕至`[-pi/2, pi/2)`。
- 使用MMCV `roi_align_rotated`替换Detectron2水平ROI Pooler，保留原FPN层级分配和ROI顺序。
- 移除当前训练/推理链对Detectron2和FVCore focal loss的依赖，focal loss改为等价的Torch实现。
- 修正扩散schedule默认生成float64、导致旋转ROIAlign类型不一致的问题，统一为float32。

## 3. HSMOT数据适配

新增`yolox.data.datasets.HSMOTDataset`，直接读取HSMOT原生八波段NPY和13列MOT标注：

- 图像输入为`H×W×8` NPY。
- 原始目标格式为`qbox8 + class_id + track_id`。
- 训练张量格式为`class_id + qbox8 + track_id`，最多保留500个目标。
- 类别为car、bike、pedestrian、van、truck、bus、tricycle、awning-bike，共8类。
- 枚举序列中的所有图像，包括没有标注的帧，以保证在线推理帧序连续。
- 截断标记大于0的目标按现有HSMOT主线处理方式忽略。

数据增强相关修改位于`data_augment.py`和`mosaicdetection.py`：

- resize、mosaic、mixup、随机透视及翻转均支持四点旋转框。
- HSMOT路径不执行RGB/BGR通道翻转。
- HSMOT路径关闭HSV/颜色扰动，保留原版几何增强和mosaic/mixup策略。
- Pair训练按track ID对齐前后帧目标。
- Pair采样范围由原来的2帧范围改为低帧率HSMOT使用的`[-1,1]`。
- 修复原训练器将所有标签硬截断为5列的问题；HSMOT配置通过`target_dim=9`传入完整的`class+qbox8`。

## 4. 旋转匹配、损失与NMS

新增`yolox/utils/rotated_boxes.py`，提供qbox/rbox转换、旋转IoU、类别感知旋转NMS、成对旋转IoU和旋转Pair Cluster-NMS。

`diffusion_losses.py`中的修改包括：

- Dynamic-K分类代价和动态匹配数量逻辑保持不变。
- 水平框L1代价替换为归一化`cxcywh` L1加周期角度L1；角度距离采用`min(|d|, 1-|d|)`处理`pi`周期边界。
- 水平Pair GIoU代价替换为前后帧联合计算的Pair Rotated IoU代价。
- 回归损失同时监督前后帧旋转框，并保留原配置键`loss_bbox`和`loss_giou`以兼容原训练配置；其中`loss_giou`现在实际表示Pair Rotated IoU loss。
- SimOTA中心先验仍按原方法使用中心和宽高确定候选范围，最终匹配代价使用精确旋转IoU。
- 对扩散早期可能产生的退化或极端随机框进行NaN保护，退化框按零重叠处理，避免污染匹配矩阵。
- Pair Cluster-NMS保留原迭代传播逻辑，仅将水平Pair GIoU替换为旋转Pair IoU；多类别场景按类别分别执行。

## 5. 推理、跟踪和输出

`yolox/tracker/diffusion_tracker_kl.py`按旋转框重新适配，但状态机保持原版KL实现：

1. 活动轨迹以上一帧旋转框作为reference proposal，经Pair Diffusion得到reference/current成对预测。
2. 轨迹与reference预测使用类别约束的旋转IoU和原Hungarian阈值完成关联。
3. current成对预测使用当前帧独立检测结果进行位置校正。
4. 未匹配活动轨迹进入Lost状态。
5. 旧Lost轨迹由原版Kalman filter预测`cx,cy,aspect,height`，角度保留最近可靠观测，再与当前帧旋转检测进行匹配恢复。
6. 超过30帧未恢复的轨迹移除。

Kalman滤波没有被删除，也没有扩展成新的旋转Kalman模型；角度不进入Kalman状态，这属于在尽量不修改原方法前提下对旋转框的最小适配。

`diffusion_mot_evaluator_kl.py`和`tools/track.py`支持输出：

- `trackers/<tracker_name>/preds/<sequence>.txt`：13列HSMOT旋转跟踪结果，格式为`frame,id,qbox8,score,class,truncation`。
- `trackers/<tracker_name>/pair_detections/<sequence>.txt`：主线canonical pair-detection缓存，包含前后帧qbox、类别和分数。
- 可选调用`TrackEval/scripts/run_hsmot_8ch.py`，计算HOTA、CLEAR和Identity指标。

## 6. 两阶段训练配置

新增两个实验配置：

- `exps/example/mot/yolo11l_diffusion_det_hsmot.py`：Stage 1 detection，默认加载MMOT官方八波段YOLO11L OBB权重，完整继承ConvMSI和其余YOLO参数，YOLO11骨干和ConvMSI共同参与训练。
- `exps/example/mot/yolo11l_diffusion_track_hsmot.py`：Stage 2 tracking，从Stage 1完整checkpoint初始化，冻结骨干并训练原Pair Diffusion跟踪头。

两个阶段均使用8类、`900×1200`基准输入、原版AdamW设置`lr=2.5e-5`和`weight_decay=1e-4`，默认训练20 epochs。Stage 1与Stage 2模型的state dict键集合已比较，结果完全一致，Stage 1完整checkpoint可直接注入Stage 2。

启动方式：

```bash
cd /data/users/wangying01/lth/PairMOT/DiffusionTrack
conda activate py310

# Stage 1：双卡每卡2张，累积4个micro-batch，有效全局BS=16
CUDA_VISIBLE_DEVICES=2,3 python tools/train.py \
  -f exps/example/mot/yolo11l_diffusion_det_hsmot.py \
  -d 2 -b 4 --accumulate 4 --fp16

# Stage 2
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  -f exps/example/mot/yolo11l_diffusion_track_hsmot.py \
  -d 1 -b 1 \
  -c DiffusionTrack_outputs/yolo11l_diffusion_det_hsmot/latest_ckpt.pth.tar
```

推理与评估方式：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/track.py \
  -f exps/example/mot/yolo11l_diffusion_track_hsmot.py \
  -c /path/to/tracking_checkpoint.pth.tar \
  -d 1 -b 1 \
  --output-dir /path/to/output \
  --tracker-name diffusiontrack \
  --tracker-sub-folder preds \
  --evaluate
```

## 7. py310依赖处理

遵守“不修改py310已有库”的要求，只补装原环境缺失的库，并将其写入`requirements.txt`：

- `ultralytics==8.3.23`
- `einops==0.8.1`
- `einops-exts==0.0.4`
- `tensorboardX==2.6.4`
- `protobuf==5.29.5`
- `lap==0.5.12`
- `cython-bbox==0.1.5`
- `Cython==3.0.12`
- 以及Ultralytics缺失的psutil、py-cpuinfo、seaborn和ultralytics-thop。

未升级、降级或覆盖原有Torch、TorchVision、MMCV、MMRotate、NumPy和OpenCV。`pip check`仅报告环境原有的`opencv-python 5.0.0.93`声明需要NumPy 2，而当前为NumPy 1.26.4；本次未按要求修改这两个既有包，实际数据读取和训练测试正常。

## 8. 已执行测试

### 8.1 数据和几何单元测试

- HSMOT OnePic单帧预处理输出：图像`8×256×320`，标签`100×10`。
- Pair非mosaic、Pair mosaic、Detection mosaic/mixup均成功产生八通道图像和对齐目标。
- qbox到rbox再回qbox转换通过；重复旋转pair框能被Pair Cluster-NMS正确抑制。
- 旋转Dynamic-K、Pair Rotated IoU loss完成CUDA前向和反向，分类和框梯度均为有限值。

### 8.2 模型结构和前后向测试

- YOLO11L+ConvMSI在`128×160`输入下输出P3/P4/P5：`(256,16,20)`、`(512,8,10)`、`(512,4,5)`。
- MMOT官方YOLO11L OBB权重加载后，在`900×1200`输入下输出：`(256,116,152)`、`(512,58,76)`、`(512,29,38)`；内部实际补齐为`928×1216`。
- 对MMOT起点进行了逐元素核验：ConvMSI的`conv3d`、`bn3d`、`fuse`、`bn2d`所有parameter和buffer均与checkpoint完全一致；YOLO第1层之后的全部state dict也逐元素一致；模型中不存在SE参数，checkpoint的八类名称和HSMOT配置顺序完全一致。
- Stage 2在`256×320`执行完整前向、全部6层辅助损失和反向成功，峰值显存约6.28 GiB。
- Stage 2在`900×1200`执行完整前向和反向成功，直接模型测试峰值显存约6.42 GiB。
- Stage 1从MMOT官方权重起点、骨干解冻状态下，在`900×1200`执行完整前向和反向成功：总loss及全部子loss为有限值，ConvMSI中`conv3d/bn3d/fuse/bn2d`所有可训练参数梯度均为有限值，峰值显存约10.64 GiB。

### 8.3 训练器烟雾测试

在本机两张RTX 3090上进一步使用真实HSMOT数据测试了原版等效全局BS：

- FP32、`-d 2 -b 4`（每卡BS=2）在第一步前向阶段OOM；当时单卡已分配约22.60 GiB，尚未进入反向，因此FP32不可用。
- FP16、`-d 2 -b 4`连续完成45个DDP前向、反向和optimizer step；900×1200阶段训练器日志峰值约19.49 GiB。
- 训练器新增`--accumulate`，包含loss归一化、DDP中间micro-batch的`no_sync`、仅在累积边界执行optimizer/GradScaler/EMA更新，以及按optimizer更新次数推进学习率。
- `-d 2 -b 4 --accumulate 4 --fp16`已真实完成三个完整累积周期；日志确认物理全局BS=4、累积4步、有效全局BS=16，900×1200阶段峰值约17.81 GiB，loss均为有限值。

在RTX 3090 24GB、py310环境上，使用HSMOT OnePic完成Stage 2单卡完整1 epoch、20 iterations训练器测试：

- 输入基准尺寸：`900×1200`。
- batch size：1。
- 20个iteration均完成forward、loss、backward、optimizer step和EMA更新。
- loss包含主输出及5组aux输出的`loss_ce/loss_bbox/loss_giou`，训练过程中均为有限值。
- 训练器峰值显存约10376 MB。
- epoch结束后成功保存checkpoint并进入在线跟踪验证。

本测试用于验证可训练性和链路完整性，不代表模型已经收敛，也不用于报告精度。

### 8.4 223服务器部署与正式训练

- 代码目录：`/data/users/linxu/code/DiffusionTrack-PairMOT`，未覆盖原`DiffusionTrack-lx`。
- 独立环境：`/data/users/linxu/.conda/envs/py310_pairmot`，核心版本为Python 3.10、Torch 2.0.1+cu118、TorchVision 0.15.2+cu118、NumPy 1.26.4、MMCV 2.2.0和Ultralytics 8.3.23。
- 223驱动525.60.13可运行cu118预编译Torch；MMCV使用cu118/torch2.0预编译wheel，不调用服务器上的旧CUDA Toolkit编译。
- 原数据位于`/data4/linxu/MMOT/data`。其COCO-video标注含完整track ID和旋转四点框；通过`tools/convert_hsmot_coco_to_mot.py`转换后，在`/data4/linxu/PairMOT_DiffusionTrack/data/hsmot`建立只保存标注并软链接原NPY的独立视图。
- DDP启动顺序已修正为先`torch.cuda.set_device(local_rank)`再初始化NCCL；验证rank 0只占GPU 1、rank 1只占GPU 2，不再由rank 1在GPU 1遗留约806 MiB上下文。
- I/O profile中，NFS冷随机读取曾测得约19.4 MiB/s；真实mosaic/mixup DataLoader在0/2/4/8 workers下稳态均值分别约5.23/2.53/1.48/0.88秒。1.7 GB本地小样本对照在0/2/4/8 workers下约1.04/0.74/0.23/0.03秒，但后两项受页缓存影响，是理想上限。
- 正式Stage 1使用GPU 1、2，`-d 2 -b 4 --accumulate 4 --fp16`、每rank 8 workers，有效全局BS=16，输出目录为`/data4/linxu/PairMOT_DiffusionTrack/work_dirs/yolo11l_diffusion_det_hsmot_b4_d2_acc4_fp16_w8`。
- 重启后iteration 20/40/60的真实训练`data_time`分别为0.003/0.148/0.260秒，迭代时间约1.15--1.51秒，loss及全部子loss有限。8 workers已能用预取隐藏大部分NFS延迟，暂不需要迁移约72.3 GB训练NPY到仅余约98 GB的本地盘。
- Stage 1已改为每5 epoch保存具名checkpoint并执行完整验证；验证指标为类别感知旋转IoU下的COCO 101点插值`mAP50`和`mAP50:95`，阈值范围为0.50:0.05:0.95。验证在DDP rank 0顺序执行，其余rank同步等待，并使用AMP降低显存；`best_ckpt`按`mAP50:95`更新。单张真实HSMOT验证样本已完成旋转推理、NMS、IoU匹配和AP汇总烟雾测试。
- 加入上述验证后已从epoch 1重新启动，旧训练输出保留为后缀`_before_val5_20260717`。新任务rank PID为1668/1669，分别只占GPU 1/2；epoch 1 iteration 20的loss为45.019、`data_time=0.002s`，训练正常。
- 首次训练在epoch 10约iteration 1680后由一个极端diffusion proposal触发AMP数值溢出：未约束的角度delta成为`inf`，旋转框解码中的`remainder(inf, pi)`产生`NaN`并污染Dynamic-K cost；rank 0退出后旧启动器未清理rank 1，导致GPU 2看似仍在单卡运行，实际DDP已经停止。检查epoch 9 `latest_ckpt`确认约2.94亿模型parameter/buffer以及全部AdamW状态均为有限值。
- 数值修复只作用于无效极端proposal：旋转框delta解码强制使用FP32，宽高delta采用对称clamp，解码结果限制在FP16 ROIAlign可表示范围；匹配器排除非有限proposal而保留全部有效proposal及其原cost。启动器同时改为任一rank失败时自动终止其余rank。真实HSMOT batch从epoch 9 checkpoint执行AMP前向及scale=1反向通过，19项loss和全部梯度有限，峰值显存约8.26 GiB。
- 进一步回查epoch 1--10日志发现，主输出`loss_giou`长期精确停在理论上限2.000，epoch 5验证mAP为0。根因是原水平框代码误将分类正样本先验`-4.595`用于旋转框残差输出bias；解码时`exp(dw/dh)≈0.01`，每个refinement head都会将宽高缩小约100倍。真实增强目标抽查30个样本、2045个框未发现退化框，确认问题不在标注或几何增强。修复仅改初始化，不在forward、匹配或损失热路径增加额外检查。
- 修复后单元测试确认零delta严格保持`(cx,cy,w,h,theta)`不变；真实HSMOT batch从MMOT官方骨干起点执行AMP前向和反向，全部loss与梯度有限，主Pair Rotated IoU loss为1.941而非固定2.000。
- 远端第一次重启时发现同步目标误落入顶层仓库下的同名`DiffusionTrack/`子目录，实际训练仍加载顶层旧代码；该次无效输出已归档为`yolo11l_diffusion_det_hsmot_b4_d2_acc4_fp16_w8_wrong_sync_20260718_0127`。核对进程`cwd`及源码SHA256后，补丁已同步到实际执行的顶层目录并从epoch 0重新启动。
- 当前有效训练日志为`/data4/linxu/PairMOT_DiffusionTrack/logs/stage1_det_bboxinitfix_actual_epoch0_20260718.log`，主进程PID 47013，rank PID 47156/47157分别使用GPU 1/2。epoch 1 iteration 20/40的主`loss_giou`为1.911/1.750，六层旋转IoU损失均脱离固定2.000状态，全部子loss有限，`data_time`为0.001--0.002秒。
- 该任务运行到epoch 7 iteration约1181时再次在Dynamic-K cost中遇到NaN。核验SHA256后确认此前只把旋转残差初始化文件同步到了远端实际执行的顶层仓库，AMP框解码、非有限proposal隔离和DDP失败清理仍误留在未执行的嵌套`DiffusionTrack/`目录。实际顶层旧启动器还会顺序等待两个rank，而`@logger.catch`会吞掉rank异常并返回0，导致另一个rank继续占用GPU。此次已将全部关键文件同步到实际顶层，并把训练入口改为`@logger.catch(reraise=True)`，确保异常以非零状态传给父进程并触发其余rank清理。
- 清理残留进程后已从`latest_ckpt`的epoch 6状态恢复，当前日志为`/data4/linxu/PairMOT_DiffusionTrack/logs/stage1_det_resume_fullfix_20260718.log`。rank PID 51579/51580分别使用GPU 1/2，epoch 7 iteration 20的`total_loss=26.414`、`data_time=0.006s`，全部子loss有限。

### 8.5 checkpoint、推理和输出测试

- Stage 1和Stage 2 checkpoint schema比较结果：键集合完全一致，`det_only=0`、`track_only=0`。
- `tools/track.py`成功加载随机初始化烟雾checkpoint，在OnePic 20帧上完成逐帧推理，测试速度约7.12 FPS。
- 成功建立HSMOT track结果文件和pair-detection文件；由于烟雾checkpoint未训练且使用正式阈值，结果正文为空，这是预期现象。
- 使用人工旋转框单独验证输出格式：跟踪结果每行13列，pair缓存每行28列，与主线读取接口一致。

### 8.5 TrackEval接口测试

使用OnePic标注复制为预测的方式进行了评估接口烟雾测试，`run_hsmot_8ch.py`成功读取13列旋转结果并完成HOTA、CLEAR、Identity和Count计算。该测试只证明目录结构、格式和评估命令正确；由于预测直接复制自GT，得到的100分结果不是模型性能，不能作为实验指标使用。

### 8.6 静态检查

- 所有修改Python文件通过`py_compile/compileall`。
- `git diff --check`通过，无空白符错误。
- 正式训练、推理和TrackEval产生的临时烟雾测试文件均已清理。

## 9. 当前状态与注意事项

代码当前已经具备从MMOT官方YOLO11L起点开始Stage 1正式训练的条件。正式训练前需要确认目标机器上存在`/data4/litianhao/PairMmot/pretrained_weights/mmot_official/yolo11L-8ch-3dstem.pt`；如果路径不同，应复制该文件或设置`YOLO11_WEIGHTS`。Stage 2必须使用Stage 1完整checkpoint启动，否则冻结的YOLO11骨干将保持随机初始化。当前测试只验证工程链路和数值稳定性，尚未产生可用于比较的正式训练精度。
