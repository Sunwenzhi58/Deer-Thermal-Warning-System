# -*- coding: utf-8 -*-
"""
鹿只高温预警系统 - 主监控类
Deer Thermal Warning System - Main Monitor Class
"""
import time
import numpy as np
from typing import Tuple, List, Dict, Union
from ultralytics import YOLO
import cv2
import Config

# 导入自定义模块（将在后续创建）
try:
    from thermal_data_loader import ThermalDataLoader
    from temperature_analyzer import TemperatureAnalyzer
    from alert_manager import AlertManager
    from visualizer import ThermalVisualizer
except ImportError:
    print("警告: 部分模块未找到，请确保所有模块文件已创建")
    # 临时定义，避免导入错误
    ThermalDataLoader = None
    TemperatureAnalyzer = None
    AlertManager = None
    ThermalVisualizer = None


class DeerThermalMonitor:
    """
    鹿只高温预警监控主类
    """
    
    def __init__(self, 
                 model_path: str = None,
                 task: str = 'segment',  # 'detect' 或 'segment' - 默认'segment'因为数据集是分割格式
                 use_segmentation: bool = True):
        """
        初始化监控系统
        
        Args:
            model_path: 模型路径，如果为None则使用Config中的路径
            task: 任务类型 'detect' 或 'segment'（当前数据集是分割格式，推荐'segment'）
            use_segmentation: 是否使用分割模式（默认True，因为数据集是分割格式）
        """
        # 加载模型
        if model_path is None:
            model_path = Config.model_path
        
        # 当前数据集是分割格式，默认使用分割模式
        self.task = task
        self.use_segmentation = use_segmentation or (task == 'segment')
        
        try:
            self.model = YOLO(model_path, task=self.task)
            print(f"模型加载成功: {model_path}")
        except Exception as e:
            raise Exception(f"模型加载失败: {e}")
        
        # 初始化各模块
        self.data_loader = ThermalDataLoader() if ThermalDataLoader else None
        self.temp_analyzer = TemperatureAnalyzer(
            max_temp_threshold=getattr(Config, 'TEMPERATURE_CONFIG', {}).get('max_temp_threshold', 40.0),
            mean_temp_threshold=getattr(Config, 'TEMPERATURE_CONFIG', {}).get('mean_temp_threshold', 38.5),
            alert_percentile=getattr(Config, 'TEMPERATURE_CONFIG', {}).get('alert_percentile', 99.5)
        ) if TemperatureAnalyzer else None
        self.alert_manager = AlertManager(
            alert_methods=getattr(Config, 'ALERT_CONFIG', {}).get('methods', ['log', 'visual']),
            alert_cooldown=getattr(Config, 'ALERT_CONFIG', {}).get('cooldown', 5)
        ) if AlertManager else None
        self.visualizer = ThermalVisualizer() if ThermalVisualizer else None
        
        # 运行状态
        self.is_running = False
        self.frame_count = 0
        
    def get_thermal_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取红外图像和温度矩阵
        
        Returns:
            thermal_img: 红外图像 (H, W, 3)
            temp_matrix: 温度矩阵 (H, W) 单位：摄氏度
        
        注意：这是一个接口方法，实际实现需要根据数据源类型
        """
        if self.data_loader:
            return self.data_loader.get_thermal_data()
        else:
            # 临时实现：从文件读取（需要根据实际情况修改）
            raise NotImplementedError("请实现数据获取模块或使用ThermalDataLoader")
    
    def predict(self, thermal_img: np.ndarray) -> Dict:
        """
        模型推理
        
        Args:
            thermal_img: 红外图像
        
        Returns:
            {
                'boxes': [[x1, y1, x2, y2], ...] 或 None,
                'masks': [mask_array, ...] 或 None,
                'confidences': [0.95, ...],
                'classes': [0, ...]
            }
        """
        results = self.model(thermal_img)[0]
        
        # 获取基础信息
        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
        confidences = results.boxes.conf.cpu().numpy() if results.boxes is not None else []
        classes = results.boxes.cls.cpu().numpy().astype(int) if results.boxes is not None else []
        
        # 分割模式：获取mask
        if self.task == 'segment' and results.masks is not None:
            # 分割模式：获取分割mask
            masks = results.masks.data.cpu().numpy()
            # 将mask转换为图像尺寸
            masks_resized = []
            for mask in masks:
                # mask是原始尺寸的，需要缩放到图像尺寸
                mask_resized = cv2.resize(mask.astype(np.uint8), 
                                         (thermal_img.shape[1], thermal_img.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
                masks_resized.append(mask_resized > 0.5)
            
            return {
                'masks': masks_resized,
                'boxes': boxes,
                'confidences': confidences,
                'classes': classes
            }
        else:
            # 检测模式：只有bbox，没有mask
            return {
                'boxes': boxes,
                'masks': None,
                'confidences': confidences,
                'classes': classes
            }
    
    def should_alert(self, max_temp: float, mean_temp: float) -> bool:
        """
        判断是否需要报警
        
        Args:
            max_temp: 最大温度
            mean_temp: 平均温度
        
        Returns:
            True if should alert
        """
        if self.temp_analyzer:
            stats = {
                'max': max_temp,
                'mean': mean_temp,
                'percentile_99_5': max_temp  # 简化处理
            }
            return self.temp_analyzer.should_alert(stats)
        else:
            # 简单阈值判断
            max_threshold = getattr(Config, 'TEMPERATURE_CONFIG', {}).get('max_temp_threshold', 40.0)
            mean_threshold = getattr(Config, 'TEMPERATURE_CONFIG', {}).get('mean_temp_threshold', 38.5)
            return max_temp > max_threshold or mean_temp > mean_threshold
    
    def trigger_alert(self, deer_id: int, mask_or_box: Union[np.ndarray, Tuple], 
                     max_temp: float, mean_temp: float):
        """
        触发报警
        
        Args:
            deer_id: 鹿只ID
            mask_or_box: 掩码或边界框
            max_temp: 最大温度
            mean_temp: 平均温度
        """
        if self.alert_manager:
            stats = {
                'max': max_temp,
                'mean': mean_temp
            }
            if isinstance(mask_or_box, tuple):
                position = mask_or_box
            else:
                # 从mask计算bbox
                y_coords, x_coords = np.where(mask_or_box)
                if len(x_coords) > 0:
                    position = (int(np.min(x_coords)), int(np.min(y_coords)),
                               int(np.max(x_coords)), int(np.max(y_coords)))
                else:
                    position = (0, 0, 0, 0)
            
            self.alert_manager.trigger_alert(deer_id, stats, position)
        else:
            # 简单日志输出
            print(f"[ALERT] 鹿只 #{deer_id}: 最大温度={max_temp:.2f}°C, 平均温度={mean_temp:.2f}°C")
    
    def display_results(self, thermal_img: np.ndarray, detections: List[Dict], 
                       temp_matrix: np.ndarray = None):
        """
        可视化显示结果
        
        Args:
            thermal_img: 原始红外图像
            detections: 检测结果列表
            temp_matrix: 温度矩阵（可选）
        """
        if self.visualizer:
            result_img = self.visualizer.draw_results(thermal_img, detections, temp_matrix)
        else:
            # 简单可视化
            result_img = thermal_img.copy()
            for det in detections:
                box = det['box']
                color = (0, 0, 255) if det.get('alert', False) else (0, 255, 0)
                cv2.rectangle(result_img, 
                             (int(box[0]), int(box[1])), 
                             (int(box[2]), int(box[3])), 
                             color, 2)
                
                if 'temp_stats' in det:
                    stats = det['temp_stats']
                    label = f"Max:{stats['max']:.1f}°C Mean:{stats['mean']:.1f}°C"
                    if det.get('alert', False):
                        label += " [ALERT]"
                    cv2.putText(result_img, label,
                               (int(box[0]), int(box[1]) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return result_img
    
    def process_frame(self, thermal_img: np.ndarray, temp_matrix: np.ndarray) -> Dict:
        """
        处理单帧数据
        
        Args:
            thermal_img: 红外图像
            temp_matrix: 温度矩阵
        
        Returns:
            处理结果字典
        """
        # 1. 模型推理
        predictions = self.predict(thermal_img)
        
        # 2. 温度分析
        detections = []
        masks = predictions.get('masks')
        boxes = predictions.get('boxes', [])
        confidences = predictions.get('confidences', [])
        classes = predictions.get('classes', [])
        
        for i in range(len(boxes)):
            deer_id = i
            box = boxes[i] if len(boxes) > i else None
            h, w = temp_matrix.shape
            
            # 提取温度 - 优先使用分割mask（因为数据集是分割格式）
            if self.use_segmentation and masks is not None and i < len(masks):
                # 使用真实分割mask（从分割模型输出）- 这是推荐方式
                mask = masks[i]
                # 将mask缩放到图像尺寸
                if mask.shape != temp_matrix.shape[:2]:
                    mask = cv2.resize(mask.astype(np.uint8), 
                                     (temp_matrix.shape[1], temp_matrix.shape[0]),
                                     interpolation=cv2.INTER_NEAREST).astype(bool)
                else:
                    mask = mask.astype(bool)
                deer_temps = temp_matrix[mask].flatten()
                region = mask
            elif box is not None:
                # 如果没有mask，使用bbox区域（降级方案）
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                deer_temps = temp_matrix[y1:y2, x1:x2].flatten()
                region = (x1, y1, x2, y2)
            else:
                continue  # 跳过无效检测
            
            # 过滤无效数据
            deer_temps = deer_temps[~np.isnan(deer_temps)]
            deer_temps = deer_temps[deer_temps > -50]  # 过滤异常低温
            
            if len(deer_temps) == 0:
                continue
            
            # 温度分析
            if self.temp_analyzer:
                temp_stats = self.temp_analyzer.analyze_temperature(deer_temps)
            else:
                # 简单统计
                temp_stats = {
                    'max': np.max(deer_temps),
                    'mean': np.mean(deer_temps),
                    'min': np.min(deer_temps),
                    'percentile_99_5': np.percentile(deer_temps, 99.5)
                }
            
            # 报警判断
            should_alert = self.should_alert(temp_stats['percentile_99_5'], temp_stats['mean'])
            
            if should_alert:
                self.trigger_alert(deer_id, region, 
                                 temp_stats['percentile_99_5'], temp_stats['mean'])
            
            # 保存检测结果
            detections.append({
                'id': deer_id,
                'box': boxes[i],
                'mask': region if self.use_segmentation and masks is not None else None,
                'confidence': confidences[i],
                'class': classes[i],
                'temp_stats': temp_stats,
                'alert': should_alert
            })
        
        return {
            'detections': detections,
            'predictions': predictions
        }
    
    def run(self, max_frames: int = None, display: bool = True):
        """
        运行监控循环
        
        Args:
            max_frames: 最大处理帧数，None表示无限循环
            display: 是否显示结果
        """
        self.is_running = True
        self.frame_count = 0
        
        print("开始监控...")
        
        try:
            while self.is_running:
                # 1. 数据获取
                try:
                    thermal_img, temp_matrix = self.get_thermal_data()
                except Exception as e:
                    print(f"数据获取失败: {e}")
                    time.sleep(1)
                    continue
                
                # 2. 处理帧
                result = self.process_frame(thermal_img, temp_matrix)
                detections = result['detections']
                
                # 3. 可视化显示
                if display:
                    result_img = self.display_results(thermal_img, detections, temp_matrix)
                    cv2.imshow('Deer Thermal Monitor', result_img)
                    
                    # 按'q'退出
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # 4. 统计信息
                self.frame_count += 1
                alert_count = sum(1 for d in detections if d.get('alert', False))
                print(f"帧 #{self.frame_count}: 检测到 {len(detections)} 只鹿, "
                      f"报警 {alert_count} 次")
                
                # 5. 限制帧数
                if max_frames and self.frame_count >= max_frames:
                    break
                
                time.sleep(0.1)  # 控制帧率
                
        except KeyboardInterrupt:
            print("\n监控已停止")
        finally:
            self.is_running = False
            cv2.destroyAllWindows()
            print(f"总共处理 {self.frame_count} 帧")
    
    def stop(self):
        """停止监控"""
        self.is_running = False


if __name__ == "__main__":
    # 示例使用
    monitor = DeerThermalMonitor(
        model_path=Config.model_path,
        task='detect',
        use_segmentation=False
    )
    
    # 运行监控（需要实现数据获取模块）
    # monitor.run(max_frames=100, display=True)

