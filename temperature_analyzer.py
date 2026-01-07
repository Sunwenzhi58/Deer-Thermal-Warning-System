# -*- coding: utf-8 -*-
"""
温度分析模块
Temperature Analyzer Module
"""
import numpy as np
from typing import Dict, Union, Tuple


class TemperatureAnalyzer:
    """
    温度分析器：从温度矩阵中提取目标区域温度并分析
    """
    
    def __init__(self, 
                 max_temp_threshold: float = 40.0,
                 mean_temp_threshold: float = 38.5,
                 alert_percentile: float = 99.5,
                 min_pixels: int = 100):
        """
        初始化温度分析器
        
        Args:
            max_temp_threshold: 最大温度报警阈值（摄氏度）
            mean_temp_threshold: 平均温度报警阈值（摄氏度）
            alert_percentile: 用于计算报警温度的百分位数（0-100）
            min_pixels: 最小有效像素数，少于该数量认为无效
        """
        self.max_temp_threshold = max_temp_threshold
        self.mean_temp_threshold = mean_temp_threshold
        self.alert_percentile = alert_percentile
        self.min_pixels = min_pixels
    
    def extract_temperature(self, 
                            temp_matrix: np.ndarray,
                            region: Union[np.ndarray, Tuple[int, int, int, int]]) -> np.ndarray:
        """
        提取目标区域的温度
        
        Args:
            temp_matrix: 温度矩阵 (H, W)，单位：摄氏度
            region: 目标区域
                - 如果是mask: (H, W) 布尔数组或0/1数组
                - 如果是bbox: (x1, y1, x2, y2) 坐标元组
        
        Returns:
            该区域的温度数组（1D）
        """
        if isinstance(region, tuple):
            # bbox模式
            x1, y1, x2, y2 = map(int, region)
            h, w = temp_matrix.shape
            # 确保坐标在范围内
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return np.array([])
            
            return temp_matrix[y1:y2, x1:x2].flatten()
        else:
            # mask模式
            mask = region
            # 确保mask是布尔类型
            if mask.dtype != bool:
                mask = mask.astype(bool)
            
            # 确保mask和temp_matrix尺寸匹配
            if mask.shape != temp_matrix.shape:
                # 尝试调整mask尺寸
                from scipy import ndimage
                mask = ndimage.zoom(mask.astype(float), 
                                   (temp_matrix.shape[0] / mask.shape[0],
                                    temp_matrix.shape[1] / mask.shape[1]),
                                   order=0) > 0.5
            
            return temp_matrix[mask].flatten()
    
    def analyze_temperature(self, temps: np.ndarray) -> Dict:
        """
        分析温度统计信息
        
        Args:
            temps: 温度数组（1D）
        
        Returns:
            温度统计信息字典，如果数据无效返回None
            {
                'max': 最大温度,
                'mean': 平均温度,
                'min': 最小温度,
                'std': 标准差,
                'median': 中位数,
                'percentile_99_5': 99.5百分位温度,
                'percentile_95': 95百分位温度,
                'count': 像素数量,
                'valid_count': 有效像素数量（非NaN）
            }
        """
        if len(temps) == 0:
            return None
        
        # 过滤无效值
        valid_temps = temps[~np.isnan(temps)]
        valid_temps = valid_temps[valid_temps > -50]  # 过滤异常低温
        valid_temps = valid_temps[valid_temps < 100]   # 过滤异常高温
        
        if len(valid_temps) < self.min_pixels:
            return None
        
        return {
            'max': float(np.max(valid_temps)),
            'mean': float(np.mean(valid_temps)),
            'min': float(np.min(valid_temps)),
            'std': float(np.std(valid_temps)),
            'median': float(np.median(valid_temps)),
            'percentile_99_5': float(np.percentile(valid_temps, self.alert_percentile)),
            'percentile_95': float(np.percentile(valid_temps, 95)),
            'count': len(temps),
            'valid_count': len(valid_temps)
        }
    
    def should_alert(self, stats: Dict) -> bool:
        """
        判断是否需要报警
        
        Args:
            stats: 温度统计信息字典（analyze_temperature的返回值）
        
        Returns:
            True if should alert, False otherwise
        """
        if stats is None:
            return False
        
        # 检查是否有足够的有效数据
        if stats.get('valid_count', 0) < self.min_pixels:
            return False
        
        # 判断条件：99.5百分位温度超过阈值 或 平均温度超过阈值
        percentile_temp = stats.get('percentile_99_5', stats.get('max', 0))
        mean_temp = stats.get('mean', 0)
        
        return (percentile_temp > self.max_temp_threshold or 
                mean_temp > self.mean_temp_threshold)
    
    def get_alert_level(self, stats: Dict) -> str:
        """
        获取报警级别
        
        Args:
            stats: 温度统计信息
        
        Returns:
            'none', 'warning', 'alert', 'critical'
        """
        if stats is None:
            return 'none'
        
        percentile_temp = stats.get('percentile_99_5', stats.get('max', 0))
        mean_temp = stats.get('mean', 0)
        
        # 严重报警：超过最大阈值1.5倍
        if percentile_temp > self.max_temp_threshold * 1.5:
            return 'critical'
        # 报警：超过阈值
        elif percentile_temp > self.max_temp_threshold or mean_temp > self.mean_temp_threshold:
            return 'alert'
        # 警告：接近阈值（80%）
        elif percentile_temp > self.max_temp_threshold * 0.8 or mean_temp > self.mean_temp_threshold * 0.8:
            return 'warning'
        else:
            return 'none'


if __name__ == "__main__":
    # 测试代码
    analyzer = TemperatureAnalyzer(
        max_temp_threshold=40.0,
        mean_temp_threshold=38.5
    )
    
    # 模拟温度数据
    temp_matrix = np.random.rand(480, 640) * 10 + 35  # 35-45度
    bbox = (100, 100, 200, 200)
    
    # 提取温度
    temps = analyzer.extract_temperature(temp_matrix, bbox)
    print(f"提取到 {len(temps)} 个温度值")
    
    # 分析温度
    stats = analyzer.analyze_temperature(temps)
    print(f"温度统计: {stats}")
    
    # 判断报警
    should_alert = analyzer.should_alert(stats)
    print(f"是否需要报警: {should_alert}")
    
    # 获取报警级别
    alert_level = analyzer.get_alert_level(stats)
    print(f"报警级别: {alert_level}")

