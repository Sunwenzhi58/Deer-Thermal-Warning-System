# -*- coding: utf-8 -*-
"""
可视化模块
Visualizer Module
"""
import cv2
import numpy as np
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class ThermalVisualizer:
    """
    热成像可视化器：可视化检测结果和温度信息
    """
    
    def __init__(self, colormap: str = 'jet', font_scale: float = 0.6):
        """
        初始化可视化器
        
        Args:
            colormap: 温度热力图颜色映射（'jet', 'hot', 'cool', 'viridis'等）
            font_scale: 字体缩放比例
        """
        self.colormap = colormap
        self.font_scale = font_scale
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def draw_results(self,
                    img: np.ndarray,
                    detections: List[Dict],
                    temp_matrix: Optional[np.ndarray] = None,
                    show_temp_heatmap: bool = False) -> np.ndarray:
        """
        绘制检测结果和温度信息
        
        Args:
            img: 原始图像 (H, W, 3)
            detections: 检测结果列表，每个元素包含：
                {
                    'id': 鹿只ID,
                    'box': [x1, y1, x2, y2],
                    'mask': mask_array (可选),
                    'confidence': 0.95,
                    'class': 0,
                    'temp_stats': {
                        'max': 42.5,
                        'mean': 40.2,
                        ...
                    },
                    'alert': True/False
                }
            temp_matrix: 温度矩阵 (H, W)（可选，用于绘制热力图）
            show_temp_heatmap: 是否显示温度热力图
        
        Returns:
            绘制后的图像
        """
        result_img = img.copy()
        
        # 如果需要显示温度热力图
        if show_temp_heatmap and temp_matrix is not None:
            result_img = self._overlay_temp_heatmap(result_img, temp_matrix)
        
        # 绘制每个检测结果
        for det in detections:
            box = det.get('box')
            if box is None:
                continue
            
            # 确定颜色
            alert = det.get('alert', False)
            if alert:
                color = (0, 0, 255)  # 红色：报警
                thickness = 3
            else:
                color = (0, 255, 0)  # 绿色：正常
                thickness = 2
            
            x1, y1, x2, y2 = map(int, box)
            
            # 绘制检测框
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, thickness)
            
            # 绘制mask（如果有）
            mask = det.get('mask')
            if mask is not None and isinstance(mask, np.ndarray):
                if mask.dtype == bool or mask.dtype == np.uint8:
                    # 调整mask尺寸
                    if mask.shape != img.shape[:2]:
                        mask = cv2.resize(mask.astype(np.uint8),
                                         (img.shape[1], img.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
                    
                    # 创建半透明覆盖层
                    overlay = result_img.copy()
                    mask_colored = np.zeros_like(result_img)
                    mask_colored[mask > 0] = color
                    cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0, result_img)
            
            # 绘制标签
            label_parts = []
            
            # 置信度
            confidence = det.get('confidence', 0)
            label_parts.append(f"Conf:{confidence:.2f}")
            
            # 温度信息
            temp_stats = det.get('temp_stats')
            if temp_stats:
                max_temp = temp_stats.get('max', 0)
                mean_temp = temp_stats.get('mean', 0)
                label_parts.append(f"Max:{max_temp:.1f}°C")
                label_parts.append(f"Mean:{mean_temp:.1f}°C")
            
            # 报警标记
            if alert:
                label_parts.append("[ALERT]")
            
            label = " | ".join(label_parts)
            
            # 计算文本尺寸
            (text_width, text_height), baseline = cv2.getTextSize(
                label, self.font, self.font_scale, 1)
            
            # 绘制文本背景
            cv2.rectangle(result_img,
                         (x1, y1 - text_height - baseline - 5),
                         (x1 + text_width, y1),
                         color, -1)
            
            # 绘制文本
            cv2.putText(result_img, label,
                       (x1, y1 - baseline - 5),
                       self.font, self.font_scale,
                       (255, 255, 255), 1, cv2.LINE_AA)
            
            # 绘制温度条（如果温度统计可用）
            if temp_stats and y2 - y1 > 30:
                self._draw_temp_bar(result_img, x2 + 5, y1, y2, temp_stats, alert)
        
        return result_img
    
    def _overlay_temp_heatmap(self, img: np.ndarray, temp_matrix: np.ndarray) -> np.ndarray:
        """
        在图像上叠加温度热力图
        
        Args:
            img: 原始图像
            temp_matrix: 温度矩阵
        
        Returns:
            叠加后的图像
        """
        # 归一化温度矩阵到0-255
        temp_min, temp_max = np.min(temp_matrix), np.max(temp_matrix)
        if temp_max > temp_min:
            temp_norm = ((temp_matrix - temp_min) / (temp_max - temp_min) * 255).astype(np.uint8)
        else:
            temp_norm = np.zeros_like(temp_matrix, dtype=np.uint8)
        
        # 应用颜色映射
        colormap_func = cm.get_cmap(self.colormap)
        temp_colored = colormap_func(temp_norm)[:, :, :3]  # 只取RGB
        temp_colored = (temp_colored * 255).astype(np.uint8)
        
        # 转换为BGR
        temp_colored_bgr = cv2.cvtColor(temp_colored, cv2.COLOR_RGB2BGR)
        
        # 叠加（半透明）
        result = cv2.addWeighted(img, 0.6, temp_colored_bgr, 0.4, 0)
        
        return result
    
    def _draw_temp_bar(self, img: np.ndarray, x: int, y1: int, y2: int,
                       temp_stats: Dict, is_alert: bool):
        """
        绘制温度条
        
        Args:
            img: 图像
            x: x坐标
            y1, y2: y坐标范围
            temp_stats: 温度统计信息
            is_alert: 是否报警
        """
        bar_width = 10
        bar_height = y2 - y1
        
        # 温度范围（假设）
        temp_min = temp_stats.get('min', 30)
        temp_max = temp_stats.get('max', 45)
        mean_temp = temp_stats.get('mean', 35)
        
        # 计算温度条位置
        temp_range = temp_max - temp_min
        if temp_range > 0:
            mean_pos = int((mean_temp - temp_min) / temp_range * bar_height)
        else:
            mean_pos = bar_height // 2
        
        # 绘制背景
        cv2.rectangle(img, (x, y1), (x + bar_width, y2), (100, 100, 100), -1)
        
        # 绘制温度条
        color = (0, 0, 255) if is_alert else (0, 255, 0)
        cv2.rectangle(img, (x, y2 - mean_pos), (x + bar_width, y2), color, -1)
        
        # 绘制刻度
        for i in range(3):
            y = y1 + i * bar_height // 2
            temp_val = temp_max - i * temp_range / 2
            cv2.line(img, (x + bar_width, y), (x + bar_width + 3, y), (255, 255, 255), 1)
            cv2.putText(img, f"{temp_val:.1f}",
                       (x + bar_width + 5, y + 5),
                       self.font, 0.4, (255, 255, 255), 1)
    
    def create_summary_image(self, img: np.ndarray, detections: List[Dict],
                            temp_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        创建汇总图像（包含检测结果和统计信息）
        
        Args:
            img: 原始图像
            detections: 检测结果
            temp_matrix: 温度矩阵
        
        Returns:
            汇总图像
        """
        # 绘制检测结果
        result_img = self.draw_results(img, detections, temp_matrix, show_temp_heatmap=True)
        
        # 添加统计信息文本
        alert_count = sum(1 for d in detections if d.get('alert', False))
        total_count = len(detections)
        
        stats_text = [
            f"检测数量: {total_count}",
            f"报警数量: {alert_count}",
        ]
        
        if detections:
            all_temps = []
            for det in detections:
                if 'temp_stats' in det:
                    all_temps.append(det['temp_stats'].get('max', 0))
            
            if all_temps:
                stats_text.append(f"最高温度: {max(all_temps):.1f}°C")
                stats_text.append(f"平均温度: {np.mean(all_temps):.1f}°C")
        
        # 在图像左上角绘制统计信息
        y_offset = 20
        for i, text in enumerate(stats_text):
            cv2.putText(result_img, text,
                       (10, y_offset + i * 25),
                       self.font, 0.7, (255, 255, 255), 2)
            cv2.putText(result_img, text,
                       (10, y_offset + i * 25),
                       self.font, 0.7, (0, 0, 0), 1)
        
        return result_img


if __name__ == "__main__":
    # 测试代码
    visualizer = ThermalVisualizer()
    
    # 创建测试图像
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    test_img.fill(128)
    
    # 创建测试检测结果
    test_detections = [
        {
            'id': 0,
            'box': [100, 100, 200, 200],
            'confidence': 0.95,
            'class': 0,
            'temp_stats': {
                'max': 42.5,
                'mean': 40.2,
                'min': 38.0
            },
            'alert': True
        },
        {
            'id': 1,
            'box': [300, 150, 400, 250],
            'confidence': 0.88,
            'class': 0,
            'temp_stats': {
                'max': 37.5,
                'mean': 36.8,
                'min': 35.5
            },
            'alert': False
        }
    ]
    
    # 创建测试温度矩阵
    test_temp_matrix = np.random.rand(480, 640) * 10 + 35
    
    # 绘制结果
    result = visualizer.create_summary_image(test_img, test_detections, test_temp_matrix)
    
    # 保存结果
    cv2.imwrite('test_visualization.jpg', result)
    print("可视化测试完成，结果已保存到 test_visualization.jpg")

