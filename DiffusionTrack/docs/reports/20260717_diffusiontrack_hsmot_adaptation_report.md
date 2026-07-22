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
  -d 2 -b 4 --accumulate 4 --amp-dtype bf16

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

## 10. AMP NaN根因与BF16修复（2026-07-18）

epoch 8 checkpoint的约2.94亿模型parameter/buffer及全部AdamW状态经逐张量检查均为有限值，但原FP16任务在epoch 9从iteration 460开始间歇出现主输出`loss_ce=NaN`，日志至少记录15次，最终在iteration 980之后出现整个YOLO11输出特征、proposal logits和旋转框均为NaN并退出。根因不是单一旋转框异常，而是两条AMP数值链：其一，原版分类融合使用`p=sqrt(sigmoid(logit)*score)`后执行`clamp(1e-6,1-1e-6)`和手写inverse-logit；FP16/BF16均会把`1-1e-6`舍入成1，导致正样本`BCEWithLogits(+inf,1)=NaN`，GPU复现同时确认反向梯度为NaN。其二，最终异常栈证明进入DiffusionHead前的YOLO11特征已经全NaN，说明FP16 backbone仍存在激活溢出路径；matcher中的非有限proposal屏蔽只能处理症状，不能修复这两条上游链。

训练AMP现改为显式`--amp-dtype {fp32,fp16,bf16}`，3090正式配置使用BF16以获得FP32同级指数范围，旧`--fp16`仅保留为兼容别名。分类几何均值、`torch.logit(eps=1e-6)`、focal/BCE、Dynamic-K matcher概率与cost、旋转框delta解码均固定为FP32，保持原方法公式和匹配逻辑不变。当前Torch 2.0.1/cu118缺少BF16 depthwise Conv3D和nearest-upsample kernel，MMCV 2.2也缺少BF16 rotated ROIAlign kernel，因此仅ConvMSI的谱融合Conv3D、YOLO11无参数上采样和rotated ROIAlign局部使用FP32，输出随即转回BF16；网络结构、参数和梯度路径均未改变。

checkpoint同时补充raw model、EMA model、optimizer成功更新次数、AMP dtype、GradScaler及skip计数。`model`键仍保存EMA以兼容原推理入口，resume优先恢复`raw_model`，避免此前EMA参数配raw optimizer state的不严格组合。FP16下只有GradScaler真正执行optimizer step后才更新EMA和学习率；BF16不使用GradScaler。旧checkpoint没有optimizer-step字段时按`已完成epoch数×每epoch累积后step数`恢复。

本机RTX 3090验证结果如下：饱和BF16分类概率的loss和logits/score梯度全部有限；BF16特征通过FP32 rotated ROIAlign桥接后前反向有限；YOLO11L+ConvMSI+六层DiffusionHead在`256×320`及原生`900×1200`完成完整前向、19项loss和反向，所有loss与参数梯度有限，`900×1200`峰值显存约8525 MiB。223使用相同py310环境的BF16完整模型前反向同样通过。正式任务使用GPU 2、3、`-d 2 -b 4 --accumulate 4 --amp-dtype bf16`，有效全局BS=16，从已确认有限的epoch 8 checkpoint恢复到独立实验目录`yolo11l_diffusion_det_hsmot_b4_d2_acc4_bf16_w8`；日志为`/data4/linxu/PairMOT_DiffusionTrack/logs/stage1_det_resume_bf16_nanfix_20260718.log`。

正式resume已进入epoch 9并连续运行至至少iteration 220，日志中的`loss_ce: nan`计数为0，全部主输出和五层辅助输出loss保持有限；iteration 20/100/220的`total_loss`分别为24.815/24.913/24.281。两个rank分别且只使用GPU 2、3。iteration 20冷启动阶段`data_time=8.036s`，预取稳定后一度降至0.388--0.945s；随后同机另外两个NPY训练任务竞争`/data4`，16个worker曾同时进入NFS I/O等待并使`data_time`升至4.452s，但任务能够自动恢复，确认该现象是共享存储吞吐波动而非BF16、DDP或NaN故障。

## 11. 单图过拟合闭环与进一步修复（2026-07-20）

为把“能够运行”进一步收紧为可验证的学习闭环，在223服务器GPU 2构造了单张`900×1200`八波段图像重复20次的Stage 1检测集。最终选用`data28-6`第4帧中一个未截断truck，标注为一个LE135旋转框；训练、验证均关闭mosaic、mixup、随机尺度和翻转，使用BF16、BS=1、完整可训练YOLO11L/ConvMSI骨干、stem十倍学习率及主输出加五层辅助输出的分类/L1/Pair Rotated IoU损失。

诊断中发现MMCV的`box_iou_rotated`对极端但有限的随机diffusion proposal偶尔返回远大于1的有限值。原代码只处理NaN/Inf，异常有限值仍会进入Dynamic-K并造成错误匹配。`rotated_iou`现统一把非有限值置零并将最终结果限制到数学定义域`[0,1]`；该修复不改变任何正常旋转IoU。独立检查确认原始训练目标、类别、900×1200等比缩放、LE135角度、骨干梯度及推理解码均一致。

第一阶段严格执行1 epoch warmup加40 epoch无增强训练，共820个optimizer step。最终raw权重达到`AP50=0.4272、AP50:95=0.1493`，EMA为`0.3466、0.1157`；单次raw proposal的最高旋转IoU达到0.9048。这排除了EMA加载或推理未使用训练权重的问题，也证明随机时间、随机proposal下已经能够学习，但800 step不足以稳定覆盖一步推理固定使用的`t=999`。继续用2.5e-5或5e-6恒定学习率会破坏精细定位，使用末期实际学习率1.25e-6则稳定改善；随机时间续训至epoch 60时曾达到`AP50=0.5147、AP50:95=0.1952`，随后固定`t=999`微调至epoch 100，最佳验证为`AP50=0.6403、AP50:95=0.3666`。

严格意义上的单样本记忆还必须消除diffusion noise这一随机数据源，否则验证实际测量的是“同一图像对未见随机proposal的泛化”。因此只在`yolo11l_diffusion_det_hsmot_overfit.py`中增加默认关闭的`fixed_training_t`和`fixed_noise_seed`诊断开关；DiffusionHead仅在显式设置seed时让训练与一步推理使用同一固定高斯proposal，正式Stage 1/Stage 2配置既不定义这些值，也继续使用原论文的均匀随机训练时间和独立随机proposal。固定`t=999、noise seed=8823`后从epoch 100低学习率微调，epoch 105即达到`AP50=1.0000`，epoch 110和120达到`AP50=1.0000、AP50:95=0.5000`。最终epoch 120 checkpoint独立重载复测中，raw和EMA结果完全一致，均为`AP50=1.0000、AP50:95=0.5000`；raw/EMA最佳proposal旋转IoU分别为0.7472/0.7070，类别均正确解码为truck。

最终诊断checkpoint绝对路径为`/data4/linxu/PairMOT_DiffusionTrack/work_dirs/yolo11l_diffusion_det_hsmot_overfit_one_large_object_t999_fixednoise_noaug_l1_gpu2_v10/epoch_120_ckpt.pth.tar`，训练日志位于同目录`train_log.txt`。该结果证明NPY输入、ConvMSI/YOLO11骨干、pair diffusion、LE135旋转匹配与L1/IoU损失、反向更新、EMA/checkpoint、旋转解码、类别感知旋转NMS和mAP评估能够完整闭环；随机noise结果用于衡量diffusion泛化，固定noise结果用于隔离并证明代码链路，两者不混作正式数据集精度。

## 12. Batched pair-detection缓存与KL离线跟踪（2026-07-20）

原验证器将loader写死为BS=1，并为每张图重新构造tracker；后处理也只读取batch中的第一个内部pair，因此简单增大loader batch会静默漏掉其余样本。现新增相邻帧验证视图：每个序列首帧与自身组成pair，其他帧与前一有效帧组成`prev->curr`；一个外部batch同时运行原方法已有的`prev->curr`和`curr->curr`两条分支，前者产生pair关联框，后者产生当前帧独立检测。后处理按batch逐项解码，不再丢弃样本。Stage 1旋转AP直接使用本批独立检测结果，同时把pair框、独立检测、帧号、序列和缩放信息写入`val_det/<epoch>/`。其中`pair_detections/*.txt`保持主线28列canonical格式，`frame_detections/*.txt`保存独立检测，`manifest.json`保存包括空帧在内的完整顺序和阈值元数据；坐标统一还原到原图空间。

KL/Kalman状态机新增纯缓存入口。缓存以低阈值导出以支持AP和后续阈值扫描，回放时再按命令行`association_thresh/det_thresh`过滤；轨迹传播、类别感知旋转IoU匹配、Kalman丢失目标预测与找回、新生轨迹和去重逻辑均与在线版本一致。`tools/track.py`默认先执行batched pair-detection并保存缓存，再从缓存生成TrackEval结果；传入`--detection-cache`可跳过网络并重复执行不同跟踪阈值。由于Torch 2.0/MMCV 2.2没有BF16 RotatedROIAlign推理kernel，验证固定使用FP32。初版直接走`DiffusionNet.forward`时会重复计算当前帧骨干；最终实现与原在线tracker一致，只提取一次prev/curr骨干特征，再在特征层组成`prev->curr`和`curr->curr`。复用后223的24 GiB GPU上BS=4峰值14.97 GiB，BS=6峰值19.92 GiB，正式配置使用BS=6。BS=6的真实checkpoint测试成功生成缓存；低阈值缓存回放用2.879秒生成2938行轨迹结果，证明网络检测、缓存序列化/反序列化和KL跟踪链路闭环。

GT加噪参数也与原版及DiffusionTrack-lx逐项核对：`T=1000`、cosine schedule的`s=0.008`、训练`t~Uniform{0,...,999}`、`scale=2`、`eps~N(0,1)`、GT不足时placeholder为`N(0.5,(1/6)^2)`、训练后裁剪到`[-2,2]`均一致；旋转适配只把噪声从4维扩为`(cx,cy,w,h,theta)`五维，并以等价LE135映射`(theta+pi/4)/pi`处理角度。我们的schedule buffer为FP32以避免BF16训练中由FP64 buffer引起整条proposal链类型提升，lx/原版用FP64生成schedule；两者数学参数完全相同但不逐bit相同。实测累计alpha最大绝对差为`4.24e-7`，noise标准差系数最大差约`2.94e-6`（发生在`t=0`），不构成加噪幅度改变。正式配置不启用固定`t`、固定noise或lx的`randn_proposals`实验。

## 13. Stage 2 `inter=2`与双阶段验证（2026-07-20）

Stage 2新增独立配置`yolo11l_diffusion_track_hsmot_inter2.py`。这里的`inter=2`不仅写入推理tracker参数，还把训练数据的pair采样范围真实改为同序列`[-2,2]`；保持原版随机正负方向与包含零偏移的采样逻辑。YOLO11/ConvMSI骨干全部解冻，stem维持十倍学习率；从Stage 1完整checkpoint初始化。223启动配置为两卡、物理全局BS=2（每卡1）、梯度累积8次、BF16，等效全局BS=16。每卡BS=2的首次测试在反向阶段超过24 GiB，因此没有继续使用物理全局BS=4。

BS=1/rank测试进一步暴露并修复了pair标签边界问题：Dynamic-K曾把单侧padding误识别为500个当前帧目标。现在目标准备阶段以pair两侧旋转框有效掩码的交集生成共享目标数，严格对应“只监督两帧共同ID”的原方法语义，不增加GPU同步或逐目标Python检查。随后还移除了YOLO adapter中`task_model/layers/head`对同一模块的三重注册；`layers/head`改为只读属性，旧Stage 1 checkpoint仍由规范的`backbone.task_model.*`键完整继承，同时消除重复state dict和潜在CUDA迁移。

真实启动已确认配置为8类、`pair_interval=2`、`interval=2`、骨干可训练，Stage 1 checkpoint加载无缺失/形状告警；首次前向得到有限`total_loss=31.1585`。后续验证期间其他用户的两个任务进入GPU 2/3，各占约9.8 GiB，与本任务单rank约12.5 GiB叠加后导致OOM；当时GPU 0/1也已被其他任务占用。该失败属于外部显存竞争而非NaN或模型链路错误。新增显存门控队列脚本，只在GPU 2/3同时低于2 GiB后自动启动，不清理其他用户进程。

## 14. 本地3-JPG与双卡验证（2026-07-20）

223本地`/data/users/qinhaolin01/data/hsmot`包含完整75/50个训练/测试序列，对应8372/5466帧。每帧由三张900×1200 RGB JPEG组成，按`3+3+2`通道拼回八波段输入。DiffusionTrack的HSMOT dataset现原生支持`img_format={npy,3jpg}`，3-JPG通道顺序与ai4rs主线loader一致；配置通过`HSMOT_IMG_SUBDIR/HSMOT_IMG_FORMAT`切换。按当前实验协议，标注末列仅作为元数据，不用于过滤目标。

验证loader不再只让rank 0处理全量数据。全局验证BS=6经两卡分片后，每rank BS=3；本测试集5466帧可被两卡整除，因此各处理2733帧且没有sampler补齐样本。两个rank分别完成pair detection、旋转IoU和局部记录，末尾只汇总CPU/Numpy格式的AP及canonical detection cache记录，再由rank 0统一计算指标、排序并写缓存。训练内验证前释放CUDA缓存，但不移动或修改模型、EMA和optimizer状态。独立`tools/eval_hsmot_det.py`可直接对任意checkpoint执行同一双卡验证。

epoch 5 checkpoint的完整复测已在GPU 0/1上通过：5466帧全部处理，pair-detection及AP汇总总耗659.3秒，输出`mAP50:95=0.0014、mAP50=0.0014`。缓存manifest包含50个序列和5466个帧记录，说明双rank合并无遗漏。随后曾从该epoch 5检测checkpoint启动Pair refinement（按原版训练定义属于Stage 2），配置中的`task=tracking`会启用成对帧与`inter=2`采样；KL tracker仍然只是训练后的推理状态机，不构成额外可训练阶段。GPU 0/1均参与该历史任务，有效全局BS=16，前80个iteration的全部loss有限，稳态`data_time`约0.001秒。

对照DiffusionTrack-lx的实际scheduler后，20 epoch正式配置改为按有效全局batch线性缩放的`yoloxwarmcos`：`base_lr=0.001/64×effective_batch`。当物理全局BS=2、累积8步时，普通backbone和DiffusionHead在1 epoch二次warmup后达到`2.5e-4`，ConvMSI stem保持10倍即`2.5e-3`；随后cosine下降，最后5 epoch分别固定为`1.25e-5/1.25e-4`。scheduler按optimizer update而非micro-batch推进。同时将LR设置移到`optimizer.step()`之前，消除新任务第一步先使用峰值、随后再跳回warmup起点的异常。

## 15. Pair共享几何增强修复（2026-07-21）

对训练张量进行GT可视化后发现，Pair mosaic虽然共享四个tile的位置和mosaic中心，但REF与CUR曾分别调用一次`random_perspective`，从而独立采样旋转、缩放、剪切和平移。该实现会凭空引入与真实帧间运动无关的几何差异；极端样本中REF只占画面很小区域而CUR被大幅放大，违反Pair Detection要求两侧使用同一预处理变换的约束。

现将两侧未仿射的mosaic同时保留，只采样一套几何增强参数，并通过重放Python和NumPy随机状态把完全相同的变换应用到REF和CUR。HSV仍保持关闭，3-JPG到八通道拼接、`/255`、多尺度、mosaic取样以及`[-2,2]`帧偏移逻辑均未改变；无增强阶段原本就是确定性等比resize，不受影响。

完整数据链数值测试把`track_range`临时设为0，使Pair两侧原始图像和GT一致；经过Mosaic和随机仿射后得到`image_equal=True`、`label_equal=True`，证明输出图像及旋转GT逐元素一致。真实`track_range=2`样本在`576×768`、`704×928`和`896×1184`三个训练尺度完成可视化，REF/CUR的mosaic边界和仿射几何一致、共同track ID列表一致。可视化保存在`DiffusionTrack/visualizations/gt_preprocess_shared_aug_20260721/`，远端副本位于`/data4/linxu/PairMOT_DiffusionTrack/gt_preprocess_shared_aug_20260721/`。

使用独立增强的旧任务在epoch 11 iteration 1220停止，目录`yolo11l_diffusion_track_hsmot_inter2_b2_d2_acc8_bf16_w8_boxfix`及其checkpoint只保留用于问题追溯，不再用于resume或评估。修复后从同一Stage-1检测checkpoint重新开始，实验名为`yolo11l_diffusion_track_hsmot_inter2_b2_d2_acc8_bf16_w8_sharedaug`。GPU 0/1双卡、物理全局BS=2、累积8、BF16及有效全局BS=16保持不变；epoch 1前60 iteration已正常完成，显存约16.23 GiB，全部loss有限。

## 16. Stage-1语义校正与缩放范围（2026-07-21）

再次逐行核对原版入口后明确两阶段含义：Stage 1使用`task=detection`和单图`MosaicDetection`，loader只返回一张图；`DiffusionNet`在内部复用同一份特征并复制同一份GT形成self-pair，因此不存在训练帧间隔，配置也不定义`pair_interval`。Stage 2才使用`task=tracking`和`DiffusionMosaicDetection`读取两个真实帧，并令`pair_interval`控制同序列偏移。KL/Kalman是推理状态机，不是第二个可训练网络阶段。此前把`task=tracking`的inter=2任务称为“Stage-1 Pair Detection”属于命名错误；相关Stage-2任务均已停止，不再自动重启。

HSMOT两个训练阶段继承的几何缩放范围已从`(0.1,2.0)`收紧到`(0.5,1.5)`。问题样本的实际随机缩放为0.162717，确实会把整个mosaic缩到画面小区域；新范围排除了此类近乎空画面，同时保留尺度增强。Stage-1因为self-pair只执行一次增强，内部两侧天然完全一致；Stage-2继续使用上一节实现的一套参数同步增强两侧。

当前真正启动的是Stage-1 detection：实验`yolo11l_diffusion_det_hsmot_b4_d2_acc4_bf16_w8_scale05_15`从MMOT官方八通道YOLO11L权重开始，不加载旧Stage-2 checkpoint；数据为223本地完整3-JPG，GPU 0/1、物理全局BS=4、累积4、BF16、有效全局BS=16。配置日志确认`task=detection`、`scale=(0.5,1.5)`且无`pair_interval`；epoch 1 iteration 20已完成，全部loss有限。

## 17. 训练与验证可视化（2026-07-21）

Stage-1现由rank 0在全局iteration 1及其后每500 iteration保存一张送入模型前的真实训练图，直接从GPU训练张量恢复前三波段可视图并叠加增强后的旋转GT。文件写入实验目录`train_visualizations/`，文件名同时记录epoch、epoch内iteration和全局iteration；该路径只在保存点发生一次GPU到CPU拷贝，不进入常规iteration热路径。

验证频率从每5 epoch改为每3 epoch，即epoch 3/6/9/12/15/18执行双卡完整旋转mAP验证。每次验证利用每个序列首帧的`prev_frame_id==frame_id`唯一标记，在负责该样本的DDP rank保存一张图，因此完整50序列每次各产生一张且不会因双卡重复。验证图绿色显示GT，红色显示score不低于0.05的最多30个预测，位于`val_det/epoch_NNN/visualizations/<sequence>_frame_XXXXXX.jpg`；pair detection cache和原mAP统计保持不变。

GPU训练张量写图和验证GT/预测写图均已通过独立烟雾测试。当前有效任务重启为`yolo11l_diffusion_det_hsmot_b4_d2_acc4_bf16_w8_scale05_15_val3_vis500_v2`；iteration 1真实训练图已成功写入，iteration 20全部loss有限。首张图已同步到本仓库`DiffusionTrack/visualizations/stage1_val3_vis500_v2/`供检查。

## 18. data43-2多目标过拟合诊断与匹配修复（2026-07-21）

为验证多类别、密集目标和小目标条件下的完整监督链路，另以`data43-2`第一帧构造单图重复20次的数据集。该帧包含15个GT，覆盖car、bike、pedestrian、truck、bus和tricycle六类；训练使用固定`896×1184`、BS=1、accumulate=1、BF16，并关闭mosaic、mixup、多尺度、翻转及EMA。可视化同时保存原始GT、高斯proposal、六层refine输出和每层SimOTA分配；固定`t=999/noise seed=0`仅用于严格记忆诊断，正式配置仍保留随机时间和随机噪声。

诊断首先发现原版Dynamic-K的缺失GT修复存在真实冲突：多个缺失GT可在同一轮选择同一个query，随后冲突清理又复用了修复前的旧mask。匹配矩阵因此可能同时满足“每列非空”，却让同一个query对应多个GT；转为训练索引时每行只保留一个GT，其他GT实际没有监督。现先解析当前冲突，再逐个给缺失GT分配尚未占用的query；没有空闲query时只允许从拥有多个正样本的GT转移一个query。最终强制验证每个query至多匹配一个GT、每个GT至少一个正样本。200组随机压力测试全部通过，真实六层快照均覆盖15/15 GT且query无重复。

第二个问题来自六层级联框回归初始化。通用Xavier初始化会使同一个随机`(dx,dy,dw,dh,dtheta)`残差连续作用六次；早期层已经接近GT的框会被后续层逐级破坏。各层最终`bboxes_delta`投影现以全零权重和bias初始化，使级联在训练开始时为恒等映射，同时保留每层独立可学习残差、结构、损失和前向流程。修复前固定条件下epoch 10的stage 0最佳pair IoU均值为0.5505，至stage 5降到0.0191；修复后epoch 5六层加权L1由3.434、2.520、1.999、1.510、1.350单调降到1.021，证明refine方向已恢复。

第三个问题是退化旋转框。高斯proposal裁剪后约4%的query会出现接近零的宽或高；MMCV旋转IoU CUDA算子曾对两个相距数百像素的此类框返回1，单纯把输出限制到`[0,1]`无法识别该假值。现在旋转IoU显式将非有限或边长不大于`1e-4`的框标为无效，pairwise IoU置零；可微IoU先以有限dummy几何避开CUDA未定义行为，再将对应结果和梯度置零；解码在NMS前过滤无效框，matcher也不允许其成为正样本。GPU测试确认退化框IoU为0且反向梯度有限，正常框IoU与梯度不变。最终重跑的六层初始快照中38个退化proposal均未被选中，所有层仍保持15/15 GT覆盖。

此外，短程诊断关闭EMA，避免decay约0.9998的权重仍被初始化主导；固定seed推理改为把同一个proposal模板广播到batch和`curr->curr`分支，与训练逐样本重建同seed generator的语义一致。这两项均只在overfit配置显式启用时生效。正式训练需要从MMOT官方YOLO11L骨干重新开始，才能使用新的级联初始化；恢复旧DiffusionHead checkpoint会覆盖初始化，不应与本次修复混用。

单卡诊断还复现了训练内验证后的allocator碎片化：FP32旋转验证峰值约18.9 GiB，返回BF16训练后虽然实际存活张量只有约10.9 GiB，但验证临时块仍被PyTorch保留；第二次验证后ConvMSI尝试申请约6.07 GiB工作区时因连续空闲显存不足而OOM。trainer现于验证返回、局部验证张量销毁后调用`torch.cuda.empty_cache()`，只释放allocator缓存，不迁移或重建模型、optimizer与EMA。最终重跑每5 epoch先保存checkpoint再验证，既验证释放逻辑，也保证外部故障时可从最近完整epoch恢复。

## 19. 扩散初始框最小尺寸修复（2026-07-21）

在保留上述IoU输入合法性保护的基础上，又从扩散几何转换源头处理零面积框。前向扩散latent仍严格按原公式、原cosine schedule和五维标准高斯生成，不改变其分布；只有在归一化框恢复为像素旋转框时，才把宽和高分别限制为至少1像素。中心坐标和LE135角度完整保留原取值范围，正常宽高不做线性重映射。训练和推理共用`unit_rboxes_to_absolute`，从而不会出现一侧有下限、另一侧仍产生零面积初始框的问题。下限在像素坐标转换后执行，避免BF16中先除以图像尺寸再乘回时把1像素舍入为约0.998像素；现有finite/边长检查、无效IoU置零和NMS过滤继续作为后续回归异常的第二道保护。

CPU FP32/BF16及223 GPU BF16边界测试均确认零宽高输入精确输出为1像素，正常框、中心与角度不变。`data43-2`固定`t=999/noise seed=0`的首个真实训练快照中，原本会退化的40个初始框均被提升到1像素，500个框中低于1像素和非有限框数量均为0；六个refinement stage仍全部覆盖15/15 GT。随后从MMOT官方YOLO11L骨干完整训练41 epoch，全程无NaN、OOM或漏分配，总损失从40.512降至8.311，最佳验证出现在epoch 35：`mAP50=0.5842、mAP50:95=0.3782`。bike仍未在该短程严格诊断中拟合，说明尺寸下限解决的是退化几何污染而不是普通IoU对不相交小框缺少几何梯度的问题。损失继续使用可微普通旋转IoU，未改为ProbIoU或GIoU。

223实验目录为`/data4/linxu/PairMOT_DiffusionTrack/work_dirs/yolo11l_diffusion_det_hsmot_overfit_data43_2_minside1px_exact_plainriou_fixedseed_gpu3_v30`，日志为`/data4/linxu/PairMOT_DiffusionTrack/logs/stage1_overfit_data43_2_minside1px_exact_plainriou_fixedseed_gpu3_v30_20260721.log`。
