# 快速开始指南

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 准备模型

### 选项A：训练新模型（推荐）

```bash
python train.py
```

训练完成后，模型会保存在 `runs/segment/train/weights/best.pt`，请将其复制到 `models/best.pt`

### 选项B：使用预训练模型

如果有预训练模型，请将其放在 `models/best.pt`

## 3. 运行程序

### GUI界面（推荐）

```bash
python MainProgram.py
```

### 命令行测试

```bash
python test_detection.py
```

## 4. 配置说明

编辑 `Config.py` 文件以修改配置：

- `model_path`: 模型文件路径（默认: 'models/best.pt'）
- `save_path`: 检测结果保存路径（默认: 'save_data'）
- `CH_names`: 类别中文名称（默认: ['鹿只']）

## 5. 数据集配置

编辑 `datasets/data.yaml` 文件：

- `nc`: 类别数量
- `names`: 类别名称列表
- `train/val/test`: 数据集路径

## 常见问题

### Q: 运行MainProgram.py时提示找不到UiMain.py
A: UI文件应该已经存在，如果仍有问题，请检查UIProgram目录下是否有UiMain.py文件

### Q: 模型加载失败
A: 请确保models/best.pt文件存在，或先运行train.py训练模型

### Q: 训练时内存不足
A: 在train.py中减小batch参数（如改为batch=2或batch=1）

### Q: 检测结果不准确
A: 增加训练轮数（epochs），或使用更大的分割模型（如yolov8s-seg.pt, yolov8m-seg.pt）

### Q: 为什么使用分割模型？
A: 数据集是分割格式（多边形坐标），分割模型可以精确检测鹿只轮廓，提高温度提取准确性

## 下一步

1. 训练分割模型：运行 `python train.py`（使用yolov8n-seg.pt预训练模型）
2. 配置温度阈值：在`Config.py`中调整`TEMPERATURE_CONFIG`参数
3. 集成温度分析：使用`deer_thermal_monitor.py`进行完整的温度监控
4. 优化报警策略：根据实际需求调整报警阈值和方式

