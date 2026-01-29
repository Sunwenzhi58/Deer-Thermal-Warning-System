# 鹿只高温预警系统
## Deer Thermal Warning System

基于YOLOv8分割模型的鹿只高温预警系统，用于红外图像中鹿只的检测和温度分析。

## 目录结构

```
deer_thermal/
├── datasets/              # 数据集目录
│   ├── images/           # 图像文件（train/val/test）
│   ├── labels/           # 标注文件（train/val/test）
│   └── data.yaml         # 数据集配置文件
├── models/               # 模型保存目录
│   └── best.pt          # 训练好的模型（需要训练后生成）
├── save_data/           # 检测结果保存目录
├── Font/                # 字体文件目录
├── TestFiles/           # 测试文件目录
├── UIProgram/           # UI程序目录
│   ├── UiMain.py        # UI主界面（已配置为鹿只高温预警系统）
│   ├── precess_bar.py   # 进度条组件
│   ├── QssLoader.py     # CSS加载器
│   └── style.css        # 样式表
├── Config.py            # 配置文件
├── train.py             # 训练脚本
├── MainProgram.py       # 主程序（GUI）
├── detect_tools.py      # 检测工具函数
└── requirements.txt     # 依赖包列表
```

## 安装步骤

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备UI文件

由于UI文件需要Qt Designer生成，您有两种选择：

**方法一：使用现有UI文件（推荐）**
```bash
# UI文件已配置为鹿只高温预警系统，可直接使用
# 如果UI文件不存在，可以从GarbageDetection复制后修改标题
```

**方法二：使用Qt Designer创建**
1. 安装Qt Designer：`pip install pyqt5-tools`
2. 打开Qt Designer：`designer`
3. 创建主窗口，添加以下控件：
   - `label_show`: QLabel（显示图像，770x480）
   - `tableWidget`: QTableWidget（显示检测结果，5列）
   - `PicBtn`: QPushButton（打开图片按钮）
   - `VideoBtn`: QPushButton（打开视频按钮）
   - `CapBtn`: QPushButton（摄像头按钮）
   - `FilesBtn`: QPushButton（批量检测按钮）
   - `SaveBtn`: QPushButton（保存按钮）
   - `ExitBtn`: QPushButton（退出按钮）
   - `comboBox`: QComboBox（目标选择下拉框）
   - `PiclineEdit`: QLineEdit（图片路径显示）
   - `VideolineEdit`: QLineEdit（视频路径显示）
   - `CaplineEdit`: QLineEdit（摄像头状态显示）
   - `label_nums`: QLabel（目标数量）
   - `label_conf`: QLabel（置信度）
   - `label_xmin`, `label_ymin`, `label_xmax`, `label_ymax`: QLabel（坐标显示）
   - `time_lb`: QLabel（检测时间）
4. 保存为`UIProgram/UiMain.ui`
5. 转换为Python代码：`pyuic5 UIProgram/UiMain.ui -o UIProgram/UiMain.py`

### 3. 准备模型文件

训练模型：
```bash
python train.py
```

或者下载预训练模型到`models/best.pt`

### 4. 配置Config.py

根据实际情况修改`Config.py`中的配置：
- `model_path`: 模型文件路径
- `save_path`: 结果保存路径
- `CH_names`: 类别中文名称

## 使用方法

### 训练模型

```bash
python train.py
```

训练参数可在`train.py`中修改：
- `epochs`: 训练轮数（默认250）
- `batch`: 批次大小（默认4，根据GPU内存调整）
- `imgsz`: 输入图像尺寸（默认640）
- `device`: 训练设备（0为GPU，'cpu'为CPU）

### 运行GUI程序

```bash
python MainProgram.py
```

### 功能说明

1. **单张图片检测**：点击"打开图片"按钮，选择图片进行检测
2. **批量图片检测**：点击"批量检测"按钮，选择文件夹批量处理
3. **视频检测**：点击"打开视频"按钮，选择视频文件进行检测
4. **摄像头检测**：点击"摄像头"按钮，使用摄像头实时检测
5. **保存结果**：点击"保存"按钮，保存检测结果到`save_data`目录

## 数据集格式

数据集应按照YOLO分割格式组织：
- 图像文件：`datasets/images/train/`, `datasets/images/val/`, `datasets/images/test/`
- 标注文件：`datasets/labels/train/`, `datasets/labels/val/`, `datasets/labels/test/`
- 标注格式：每行一个目标，格式为 `class_id x1 y1 x2 y2 x3 y3 ...`（归一化的多边形顶点坐标对）
- 示例：`0 0.550366 0.721194 0.577839 0.703014 0.596154 0.696954 ...`

## 注意事项

1. 确保已安装PyTorch和CUDA（如果使用GPU训练）
2. 模型文件路径需正确配置
3. UI文件必须存在才能运行GUI程序
4. 字体文件（如需要）放在`Font/`目录下

## 系统特点

- **分割检测**：使用YOLOv8分割模型，精确检测鹿只轮廓
- **温度分析**：从红外图像提取温度矩阵，分析鹿只区域温度
- **高温预警**：根据温度阈值自动触发报警
- **实时监控**：支持图片、视频、摄像头多种输入方式


## 许可证

请参考项目根目录的LICENSE文件。

