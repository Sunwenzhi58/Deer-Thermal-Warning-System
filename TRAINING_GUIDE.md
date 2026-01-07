# 分割模型训练指南
## Segmentation Model Training Guide

## 一、训练前准备

### 1. 检查数据集结构

确保数据集结构正确：
```
deer_thermal/datasets/
├── images/
│   ├── train/     # 训练图像（489张）
│   ├── val/       # 验证图像（61张）
│   └── test/      # 测试图像（62张）
├── labels/
│   ├── train/     # 训练标注（分割格式，多边形坐标）
│   ├── val/       # 验证标注
│   └── test/      # 测试标注
└── data.yaml      # 数据集配置文件
```

### 2. 验证标注格式

标注文件应该是分割格式（多边形坐标）：
```
0 0.550366 0.721194 0.577839 0.703014 0.596154 0.696954 ...
```

### 3. 检查data.yaml配置

确保`datasets/data.yaml`配置正确：
```yaml
train: images/train
val: images/val
test: images/test

nc: 1
names: ['deer']
```

## 二、训练步骤

### 方法1：直接运行训练脚本（推荐）

```bash
cd deer_thermal
python train.py
```

### 方法2：使用命令行训练

```bash
cd deer_thermal
yolo segment train data=datasets/data.yaml model=yolov8n-seg.pt epochs=250 batch=4 imgsz=640 device=0
```

### 方法3：自定义训练参数

编辑`train.py`文件，修改训练参数后运行：
```python
python train.py
```

## 三、训练参数说明

### 基础参数

| 参数 | 说明 | 推荐值 | 备注 |
|------|------|--------|------|
| `epochs` | 训练轮数 | 250 | 可根据训练曲线调整 |
| `batch` | 批次大小 | 4 | 根据GPU内存调整：4GB→4, 8GB→8, 16GB→16 |
| `imgsz` | 图像尺寸 | 640 | 可选：320, 640, 1280 |
| `device` | 训练设备 | 0 | 0=GPU, 'cpu'=CPU |

### 学习率参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `lr0` | 初始学习率 | 0.01 |
| `lrf` | 最终学习率比例 | 0.01 |
| `momentum` | SGD动量 | 0.937 |
| `weight_decay` | 权重衰减 | 0.0005 |

### 分割任务特定参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `overlap_mask` | mask是否重叠 | True |
| `mask_ratio` | mask下采样比例 | 4 |

## 四、训练过程监控

### 1. 实时查看训练进度

训练过程中会显示：
- 当前epoch和总epochs
- 训练损失和验证损失
- mAP指标（mAP50, mAP50-95）
- 学习率变化

### 2. 查看训练结果

训练完成后，结果保存在：
```
runs/segment/train/
├── weights/
│   ├── best.pt      # 最佳模型（验证集上表现最好）
│   └── last.pt      # 最新模型（最后一个epoch）
├── results.png      # 训练曲线图
├── confusion_matrix.png  # 混淆矩阵
├── F1_curve.png     # F1曲线
├── PR_curve.png     # PR曲线
└── ...
```

### 3. 关键指标说明

- **mAP50**: IoU阈值为0.5时的平均精度
- **mAP50-95**: IoU阈值从0.5到0.95的平均精度
- **box_loss**: 边界框损失
- **seg_loss**: 分割损失（关键指标）
- **cls_loss**: 分类损失

## 五、模型选择

### 不同大小的模型

| 模型 | 参数量 | 速度 | 精度 | 推荐场景 |
|------|--------|------|------|----------|
| yolov8n-seg.pt | 最小 | 最快 | 较低 | 快速原型、资源受限 |
| yolov8s-seg.pt | 小 | 快 | 中等 | **推荐用于生产** |
| yolov8m-seg.pt | 中 | 中等 | 较高 | 精度要求高 |
| yolov8l-seg.pt | 大 | 慢 | 高 | 高精度需求 |
| yolov8x-seg.pt | 最大 | 最慢 | 最高 | 研究、竞赛 |

### 修改模型大小

在`train.py`中修改：
```python
model = YOLO("yolov8s-seg.pt")  # 改为s, m, l, x等
```

## 六、常见问题解决

### Q1: 内存不足（CUDA out of memory）

**解决方案**：
1. 减小batch大小：`batch=2` 或 `batch=1`
2. 减小图像尺寸：`imgsz=320`
3. 使用更小的模型：`yolov8n-seg.pt`

### Q2: 训练速度慢

**解决方案**：
1. 使用GPU训练：`device=0`
2. 增加workers：`workers=8`（根据CPU核心数）
3. 启用混合精度：`amp=True`（默认已启用）

### Q3: 验证损失不下降

**解决方案**：
1. 检查数据集质量（标注是否正确）
2. 增加训练轮数：`epochs=300`或更多
3. 调整学习率：`lr0=0.001`（降低初始学习率）
4. 使用数据增强（默认已启用）

### Q4: 过拟合（训练损失下降但验证损失上升）

**解决方案**：
1. 增加数据增强
2. 使用dropout（如果支持）
3. 早停（Early Stopping）
4. 增加验证集比例

### Q5: 分割精度低

**解决方案**：
1. 检查标注质量（多边形是否精确）
2. 使用更大的模型：`yolov8s-seg.pt`或更大
3. 增加训练轮数
4. 调整学习率
5. 使用更大的图像尺寸：`imgsz=1280`

## 七、训练后操作

### 1. 复制最佳模型

训练完成后，最佳模型会自动复制到`models/best.pt`，也可以手动复制：

```bash
# Windows
copy runs\segment\train\weights\best.pt models\best.pt

# Linux/Mac
cp runs/segment/train/weights/best.pt models/best.pt
```

### 2. 验证模型

```bash
python test_detection.py
```

### 3. 导出模型（可选）

导出为ONNX格式用于部署：
```python
from ultralytics import YOLO
model = YOLO('models/best.pt')
model.export(format='onnx')
```

## 八、训练建议

### 1. 首次训练

- 使用`yolov8n-seg.pt`快速验证流程
- 训练50-100个epochs查看效果
- 确认数据集和配置正确

### 2. 正式训练

- 使用`yolov8s-seg.pt`或`yolov8m-seg.pt`
- 训练250-300个epochs
- 监控训练曲线，适时调整

### 3. 优化训练

- 根据验证集表现调整超参数
- 尝试不同的学习率策略
- 使用更大的图像尺寸提高精度

## 九、训练时间估算

基于您的数据集（约612张图像）：

| 模型 | GPU | 250 epochs | 预计时间 |
|------|-----|------------|----------|
| yolov8n-seg | RTX 3060 | 250 | 2-3小时 |
| yolov8s-seg | RTX 3060 | 250 | 4-6小时 |
| yolov8m-seg | RTX 3060 | 250 | 8-12小时 |

*实际时间取决于GPU性能、批次大小等因素*

## 十、下一步

训练完成后：
1. ✅ 复制最佳模型到`models/best.pt`
2. ✅ 运行`python test_detection.py`验证模型
3. ✅ 运行`python MainProgram.py`测试GUI
4. ✅ 集成温度分析功能（使用`deer_thermal_monitor.py`）

