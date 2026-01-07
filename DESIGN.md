# 鹿只高温预警系统设计方案
## Deer Thermal Warning System Design Document

## 一、方案可行性分析

### ✅ 方案总体可行，但需要注意以下关键点：

1. **数据集格式说明**
   - **当前数据集格式**：✅ **YOLO分割格式**（多边形点坐标）
   - 标签格式：`class_id x1 y1 x2 y2 x3 y3 ...`（第一列是类别ID，后面是归一化的多边形顶点坐标对）
   - 示例：`0 0.550366 0.721194 0.577839 0.703014 0.596154 0.696954 ...`
   - **这是标准的分割标注格式**，可以直接用于YOLOv8分割模型训练
   - **推荐方案**：使用YOLOv8分割模型（yolov8n-seg.pt）进行训练和推理

2. **温度矩阵获取**
   - 需要明确红外图像的数据格式和来源
   - 常见方式：
     - 红外相机SDK直接获取温度矩阵
     - 从热成像文件（如FLIR .seq, .jpg with metadata）读取
     - 从温度数据文件（.csv, .npy）读取

3. **模型选择**
   - **推荐方案**：使用YOLOv8分割模型（yolov8n-seg.pt）
   - 数据集是分割格式，直接使用分割模型进行训练和推理
   - 分割模型输出mask，可以精确提取鹿只区域温度，避免背景干扰

## 二、系统架构设计

```
┌─────────────────────────────────────────────────────────┐
│              鹿只高温预警系统架构                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────┐  │
│  │  数据获取层   │───▶│  模型推理层   │───▶│ 温度分析层 │  │
│  │              │    │              │    │          │  │
│  │ - 红外图像   │    │ - YOLO分割   │    │ - 温度提取│  │
│  │ - 温度矩阵   │    │ - Mask生成   │    │ - 统计分析│  │
│  └──────────────┘    └──────────────┘    └──────────┘  │
│         │                    │                  │        │
│         └────────────────────┴──────────────────┘        │
│                            │                             │
│                   ┌─────────▼──────────┐                │
│                   │   报警判断层        │                │
│                   │  - 阈值判断        │                │
│                   │  - 报警触发        │                │
│                   └─────────┬──────────┘                │
│                             │                            │
│                   ┌─────────▼──────────┐                │
│                   │   可视化显示层     │                │
│                   │  - 检测结果        │                │
│                   │  - 温度标注        │                │
│                   │  - 报警提示        │                │
│                   └────────────────────┘                │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## 三、模块设计

### 3.1 数据获取模块 (`thermal_data_loader.py`)

**功能**：获取红外图像和对应的温度矩阵

**接口设计**：
```python
class ThermalDataLoader:
    def get_thermal_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取红外图像和温度矩阵
        
        Returns:
            thermal_img: 红外图像 (H, W, 3) RGB格式
            temp_matrix: 温度矩阵 (H, W) 单位：摄氏度
        """
        pass
    
    def load_from_file(self, img_path: str, temp_path: str = None):
        """
        从文件加载数据
        
        Args:
            img_path: 红外图像路径
            temp_path: 温度数据文件路径（可选，如果图像包含温度信息则不需要）
        """
        pass
    
    def load_from_camera(self, camera_id: int = 0):
        """
        从红外相机实时获取数据
        
        Args:
            camera_id: 相机ID
        """
        pass
```

**实现方案**：
- **方案1**：从文件读取（开发/测试阶段）
  - 图像：PNG/JPG格式
  - 温度：单独的.npy/.csv文件，或从图像元数据读取
  
- **方案2**：从相机SDK获取（生产环境）
  - 使用FLIR/海康等红外相机SDK
  - 直接获取温度矩阵

### 3.2 模型推理模块 (`deer_detector.py`)

**功能**：使用YOLOv8分割模型进行鹿只分割

**接口设计**：
```python
class DeerDetector:
    def __init__(self, model_path: str, task: str = 'segment'):
        """
        Args:
            model_path: 模型路径（分割模型，如yolov8n-seg.pt）
            task: 'segment'（分割任务）
        """
        self.model = YOLO(model_path, task=task)
    
    def predict(self, img: np.ndarray) -> Dict:
        """
        预测图像中的鹿只（分割模式）
        
        Returns:
            {
                'masks': [mask_array, ...],         # 分割掩码（主要输出）
                'boxes': [[x1, y1, x2, y2], ...],  # 检测框（从mask计算，用于可视化）
                'confidences': [0.95, ...],         # 置信度
                'classes': [0, ...]                  # 类别ID
            }
        """
        results = self.model(img)[0]
        
        # 分割模式：获取mask和相关信息
        masks = results.masks.data.cpu().numpy() if results.masks is not None else []
        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
        confidences = results.boxes.conf.cpu().numpy() if results.boxes is not None else []
        classes = results.boxes.cls.cpu().numpy().astype(int) if results.boxes is not None else []
        
        return {
            'masks': masks,
            'boxes': boxes,
            'confidences': confidences,
            'classes': classes
        }
```

### 3.3 温度分析模块 (`temperature_analyzer.py`)

**功能**：从温度矩阵中提取目标区域温度并分析

**接口设计**：
```python
class TemperatureAnalyzer:
    def __init__(self, 
                 max_temp_threshold: float = 40.0,
                 mean_temp_threshold: float = 38.5,
                 alert_percentile: float = 99.5):
        """
        Args:
            max_temp_threshold: 最大温度报警阈值（摄氏度）
            mean_temp_threshold: 平均温度报警阈值（摄氏度）
            alert_percentile: 用于计算报警温度的百分位数
        """
        self.max_temp_threshold = max_temp_threshold
        self.mean_temp_threshold = mean_temp_threshold
        self.alert_percentile = alert_percentile
    
    def extract_temperature(self, 
                            temp_matrix: np.ndarray,
                            mask: np.ndarray) -> np.ndarray:
        """
        提取目标区域的温度（使用分割mask）
        
        Args:
            temp_matrix: 温度矩阵 (H, W)
            mask: 分割掩码 (H, W) 布尔数组，True表示鹿只区域
        
        Returns:
            该区域的温度数组（1D）
        """
        # 确保mask和temp_matrix尺寸匹配
        if mask.shape != temp_matrix.shape:
            mask = cv2.resize(mask.astype(np.uint8), 
                             (temp_matrix.shape[1], temp_matrix.shape[0]),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
        
        # 使用mask提取温度
        return temp_matrix[mask].flatten()
    
    def analyze_temperature(self, temps: np.ndarray) -> Dict:
        """
        分析温度统计信息
        
        Returns:
            {
                'max': 最大温度,
                'mean': 平均温度,
                'min': 最小温度,
                'std': 标准差,
                'percentile_99_5': 99.5百分位温度,
                'count': 像素数量
            }
        """
        if len(temps) == 0:
            return None
        
        return {
            'max': np.max(temps),
            'mean': np.mean(temps),
            'min': np.min(temps),
            'std': np.std(temps),
            'percentile_99_5': np.percentile(temps, self.alert_percentile),
            'count': len(temps)
        }
    
    def should_alert(self, stats: Dict) -> bool:
        """
        判断是否需要报警
        
        Args:
            stats: 温度统计信息
        
        Returns:
            True if should alert
        """
        if stats is None:
            return False
        
        return (stats['percentile_99_5'] > self.max_temp_threshold or 
                stats['mean'] > self.mean_temp_threshold)
```

### 3.4 报警模块 (`alert_manager.py`)

**功能**：管理报警逻辑和通知

**接口设计**：
```python
class AlertManager:
    def __init__(self, 
                 alert_methods: List[str] = ['log', 'visual'],
                 alert_cooldown: int = 5):
        """
        Args:
            alert_methods: 报警方式列表 ['log', 'visual', 'sound', 'email', 'sms']
            alert_cooldown: 报警冷却时间（秒），避免频繁报警
        """
        self.alert_methods = alert_methods
        self.alert_cooldown = alert_cooldown
        self.last_alert_time = {}
    
    def trigger_alert(self, 
                     deer_id: int,
                     stats: Dict,
                     position: Tuple[int, int, int, int],
                     timestamp: float = None):
        """
        触发报警
        
        Args:
            deer_id: 鹿只ID
            stats: 温度统计信息
            position: 位置信息 (x1, y1, x2, y2) 或 mask
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = time.time()
        
        # 检查冷却时间
        if deer_id in self.last_alert_time:
            if timestamp - self.last_alert_time[deer_id] < self.alert_cooldown:
                return
        
        self.last_alert_time[deer_id] = timestamp
        
        # 执行报警
        for method in self.alert_methods:
            if method == 'log':
                self._log_alert(deer_id, stats, position)
            elif method == 'visual':
                self._visual_alert(deer_id, stats, position)
            elif method == 'sound':
                self._sound_alert()
            elif method == 'email':
                self._email_alert(deer_id, stats)
            # ... 其他报警方式
```

### 3.5 可视化模块 (`visualizer.py`)

**功能**：可视化检测结果和温度信息

**接口设计**：
```python
class ThermalVisualizer:
    def __init__(self, colormap: str = 'jet'):
        """
        Args:
            colormap: 温度热力图颜色映射
        """
        self.colormap = colormap
    
    def draw_results(self,
                    img: np.ndarray,
                    detections: List[Dict],
                    temp_matrix: np.ndarray = None) -> np.ndarray:
        """
        绘制检测结果和温度信息
        
        Args:
            img: 原始图像
            detections: 检测结果列表，每个元素包含：
                {
                    'id': 0,                    # 鹿只ID
                    'box': [x1, y1, x2, y2],     # 边界框（从mask计算）
                    'mask': mask_array,          # 分割掩码（主要）
                    'confidence': 0.95,          # 置信度
                    'class': 0,                  # 类别ID
                    'temp_stats': {              # 温度统计信息
                        'max': 42.5,
                        'mean': 40.2,
                        'percentile_99_5': 42.0,
                        ...
                    },
                    'alert': True/False          # 是否报警
                }
            temp_matrix: 温度矩阵（可选，用于绘制热力图）
        
        Returns:
            绘制后的图像
        """
        result_img = img.copy()
        
        for det in detections:
            # 确定颜色（报警为红色，正常为绿色）
            alert = det.get('alert', False)
            color = (0, 0, 255) if alert else (0, 255, 0)
            thickness = 3 if alert else 2
            
            # 绘制分割mask（如果有）
            mask = det.get('mask')
            if mask is not None:
                # 创建半透明覆盖层
                overlay = result_img.copy()
                mask_colored = np.zeros_like(result_img)
                mask_colored[mask > 0] = color
                cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0, result_img)
            
            # 绘制边界框（从mask计算或直接提供）
            box = det.get('box')
            if box is not None:
                cv2.rectangle(result_img, (int(box[0]), int(box[1])), 
                             (int(box[2]), int(box[3])), color, thickness)
            
            # 绘制温度信息
            if 'temp_stats' in det:
                stats = det['temp_stats']
                label = f"Max:{stats['max']:.1f}°C Mean:{stats['mean']:.1f}°C"
                if alert:
                    label += " [ALERT]"
                
                # 确定标签位置
                label_x = int(box[0]) if box is not None else 10
                label_y = int(box[1]) - 10 if box is not None else 20
                
                # 绘制文本背景
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(result_img,
                             (label_x, label_y - text_height - baseline - 5),
                             (label_x + text_width, label_y),
                             color, -1)
                
                # 绘制文本
                cv2.putText(result_img, label,
                           (label_x, label_y - baseline - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_img
```

### 3.6 主监控类 (`deer_thermal_monitor.py`)

**功能**：整合所有模块，实现完整的监控流程

**完整实现**：见下方代码

## 四、实现方案

### ✅ 基于分割Mask的温度提取（当前方案）

**数据集格式**：YOLO分割格式（多边形点坐标）

**优点**：
- ✅ 温度提取最精确，只包含鹿只区域
- ✅ 避免背景温度干扰，提高报警准确性
- ✅ 完全匹配数据集格式，无需转换
- ✅ 分割模型输出mask，直接可用

**实现流程**：
```python
# 1. 使用分割模型进行推理
results = model(thermal_img)[0]

# 2. 获取分割mask
masks = results.masks.data.cpu().numpy()  # 分割模型输出的mask

# 3. 将mask缩放到温度矩阵尺寸
mask_resized = cv2.resize(mask.astype(np.uint8), 
                         (temp_matrix.shape[1], temp_matrix.shape[0]),
                         interpolation=cv2.INTER_NEAREST).astype(bool)

# 4. 使用mask提取鹿只区域温度
deer_temps = temp_matrix[mask_resized].flatten()

# 5. 温度分析
temp_stats = analyzer.analyze_temperature(deer_temps)

# 6. 报警判断
if analyzer.should_alert(temp_stats):
    alert_manager.trigger_alert(...)
```

**关键优势**：
- **精确性**：mask精确勾勒鹿只轮廓，温度提取只包含鹿只区域
- **准确性**：避免背景温度干扰，提高高温检测准确性
- **一致性**：数据集和模型都是分割格式，完全匹配

## 五、数据流设计

```
红外图像/温度数据
    ↓
[数据获取模块] → thermal_img, temp_matrix
    ↓
[模型推理模块] → detections (masks + boxes)
    ↓
[温度分析模块] → temp_stats (max, mean, percentile...)
    ↓
[报警判断模块] → alert_flag
    ↓
[可视化模块] → result_img (带标注的图像)
    ↓
显示/保存/报警通知
```

## 六、配置文件设计

在`Config.py`中添加：

```python
# 温度分析配置
TEMPERATURE_CONFIG = {
    'max_temp_threshold': 40.0,      # 最大温度阈值（摄氏度）
    'mean_temp_threshold': 38.5,     # 平均温度阈值（摄氏度）
    'alert_percentile': 99.5,         # 报警温度百分位数
    'min_pixels': 100,                # 最小有效像素数
}

# 报警配置
ALERT_CONFIG = {
    'methods': ['log', 'visual'],     # 报警方式
    'cooldown': 5,                    # 报警冷却时间（秒）
    'email_enabled': False,            # 是否启用邮件报警
    'sound_enabled': False,            # 是否启用声音报警
}

# 数据源配置
DATA_SOURCE_CONFIG = {
    'type': 'file',                   # 'file' 或 'camera'
    'camera_id': 0,                   # 相机ID
    'temp_file_suffix': '.npy',       # 温度文件后缀
}
```

## 七、集成到现有框架

### 7.1 修改MainProgram.py

在GUI中添加：
- 温度阈值设置界面
- 实时温度显示
- 报警状态显示
- 温度统计图表

### 7.2 新增功能模块

在`deer_thermal`目录下已创建：
- ✅ `thermal_data_loader.py` - 数据获取（支持文件和相机）
- ✅ `temperature_analyzer.py` - 温度分析（支持mask提取）
- ✅ `alert_manager.py` - 报警管理（多种报警方式）
- ✅ `visualizer.py` - 可视化（支持mask绘制和温度热力图）
- ✅ `deer_thermal_monitor.py` - 主监控类（整合所有模块，支持分割模式）

**注意**：模型推理功能已集成在`deer_thermal_monitor.py`中，使用YOLOv8分割模型。

## 八、测试方案

1. **单元测试**：测试各个模块功能
2. **集成测试**：测试完整流程
3. **性能测试**：测试实时性能
4. **准确性测试**：验证温度提取准确性

## 九、部署建议

1. **开发环境**：使用文件数据源，便于调试
2. **生产环境**：集成红外相机SDK，实时监控
3. **数据记录**：保存检测结果和温度数据，便于后续分析

## 十、后续优化方向

1. **多目标跟踪**：跟踪同一只鹿的温度变化趋势
2. **温度趋势分析**：分析温度变化规律
3. **机器学习优化**：使用ML模型优化温度阈值
4. **分布式部署**：支持多相机监控

