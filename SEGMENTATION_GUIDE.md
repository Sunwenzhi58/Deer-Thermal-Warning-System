# 分割模式使用指南（已更新）
## Segmentation Mode Usage Guide (Updated)

## ✅ 数据集格式确认

您的数据集是**YOLO分割格式**（多边形点坐标），格式如下：

```
0 0.550366 0.721194 0.577839 0.703014 0.596154 0.696954 0.619048 0.686854 ...
0 0.213340 0.518169 0.273902 0.510687 0.327196 0.543818 0.340924 0.588706 ...
```

**格式说明**：
- 第一列：`class_id`（类别ID，0表示鹿只）
- 后面：归一化的多边形顶点坐标对 `x1 y1 x2 y2 x3 y3 ...`
- 每行代表一个分割实例（一只鹿）

## 推荐使用方案

### ✅ 方案：分割模型 + 真实Mask（推荐，最精确）

**适用场景**：您的数据集是分割格式，直接使用分割模型

**优点**：
- 最精确的温度提取
- 只包含鹿只区域，无背景干扰
- 完全匹配您的数据集格式

**步骤1：训练分割模型**

```python
# train.py 已配置为分割模式
python train.py
```

训练脚本会自动：
- 使用 `yolov8n-seg.pt` 分割预训练模型
- 读取分割格式的标注文件
- 训练分割模型

**步骤2：使用分割模型进行监控**

```python
from deer_thermal_monitor import DeerThermalMonitor
import Config

# 初始化监控系统（使用分割模式）
monitor = DeerThermalMonitor(
    model_path=Config.model_path,  # 训练好的分割模型
    task='segment',  # 分割任务
    use_segmentation=True  # 使用分割模式
)

# 运行监控
monitor.run()
```

## 温度提取流程

使用分割模型时，温度提取流程：

1. **模型推理**：分割模型输出mask
2. **Mask处理**：将mask缩放到温度矩阵尺寸
3. **温度提取**：使用mask从温度矩阵中提取鹿只区域温度
4. **温度分析**：计算最大温度、平均温度等统计信息
5. **报警判断**：根据温度阈值判断是否需要报警

```python
# 伪代码示例
mask = results.masks[i]  # 分割模型输出的mask
mask_resized = resize(mask, temp_matrix.shape)  # 缩放到温度矩阵尺寸
deer_temps = temp_matrix[mask_resized]  # 提取鹿只区域温度
temp_stats = analyze_temperature(deer_temps)  # 分析温度
if should_alert(temp_stats):
    trigger_alert(...)  # 触发报警
```

## 数据集结构

您的数据集结构应该是：

```
datasets/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/  # 分割格式标注文件
    ├── val/
    └── test/
```

每个标注文件（.txt）包含：
- 每行一个分割实例
- 格式：`class_id x1 y1 x2 y2 x3 y3 ...`（多边形顶点）

## 训练参数建议

```python
# train.py 中的推荐参数
model.train(
    data='datasets/data.yaml',
    epochs=250,
    batch=4,  # 根据GPU内存调整
    imgsz=640,
    device=0,  # GPU
    task='segment'  # 分割任务
)
```

## 注意事项

1. **模型选择**：必须使用分割预训练模型（如 `yolov8n-seg.pt`）
2. **数据格式**：确保标注文件是分割格式（多边形坐标）
3. **Mask精度**：分割mask比bbox更精确，温度提取更准确
4. **性能**：分割模型计算量稍大，但精度更高

## 验证数据集格式

可以使用以下代码验证标注格式：

```python
import os
import numpy as np

def verify_segmentation_format(label_file):
    """验证分割标注格式"""
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 3:
            print(f"警告: 第{i+1}行数据不足（至少需要class_id和2个坐标点）")
            continue
        
        class_id = parts[0]
        coords = [float(x) for x in parts[1:]]
        
        # 检查坐标数量（应该是偶数，因为是成对的x,y）
        if len(coords) % 2 != 0:
            print(f"错误: 第{i+1}行坐标数量不是偶数")
            continue
        
        # 检查坐标范围（应该在0-1之间）
        if any(c < 0 or c > 1 for c in coords):
            print(f"警告: 第{i+1}行坐标超出0-1范围")
        
        print(f"实例 {i+1}: 类别={class_id}, 顶点数={len(coords)//2}")

# 测试
verify_segmentation_format('datasets/labels/train/frame_001.txt')
```

## 常见问题

### Q: 为什么必须使用分割模型？
A: 因为您的数据集是分割格式（多边形坐标），分割模型可以直接使用这些标注进行训练。

### Q: 可以使用检测模型吗？
A: 可以，但需要将分割标注转换为检测标注（会丢失精度），不推荐。

### Q: 分割模型训练时间会更长吗？
A: 是的，分割模型计算量更大，训练时间会稍长，但精度更高。

### Q: 如何提高分割精度？
A: 
- 增加训练轮数（epochs）
- 使用更大的模型（yolov8s-seg.pt, yolov8m-seg.pt等）
- 增加数据增强
- 调整学习率
