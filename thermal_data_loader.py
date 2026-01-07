# -*- coding: utf-8 -*-
"""
热成像数据加载模块
Thermal Data Loader Module
"""
import numpy as np
import cv2
import os
from typing import Tuple, Optional
import Config


class ThermalDataLoader:
    """
    热成像数据加载器：获取红外图像和对应的温度矩阵
    """
    
    def __init__(self, data_source_type: str = None):
        """
        初始化数据加载器
        
        Args:
            data_source_type: 数据源类型 'file' 或 'camera'
        """
        config = getattr(Config, 'DATA_SOURCE_CONFIG', {})
        self.data_source_type = data_source_type or config.get('type', 'file')
        self.camera_id = config.get('camera_id', 0)
        self.temp_file_suffix = config.get('temp_file_suffix', '.npy')
        self.camera = None
        
        if self.data_source_type == 'camera':
            self._init_camera()
    
    def _init_camera(self):
        """初始化相机连接"""
        # 这里需要根据实际使用的红外相机SDK进行实现
        # 例如：FLIR相机使用PySpin，海康相机使用hikvision SDK等
        try:
            # 示例：使用OpenCV打开相机（实际需要替换为红外相机SDK）
            self.camera = cv2.VideoCapture(self.camera_id)
            if not self.camera.isOpened():
                raise Exception(f"无法打开相机 {self.camera_id}")
            print(f"相机初始化成功: {self.camera_id}")
        except Exception as e:
            print(f"相机初始化失败: {e}")
            self.camera = None
    
    def get_thermal_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取红外图像和温度矩阵
        
        Returns:
            thermal_img: 红外图像 (H, W, 3) RGB格式
            temp_matrix: 温度矩阵 (H, W) 单位：摄氏度
        
        Raises:
            NotImplementedError: 如果数据源类型不支持
        """
        if self.data_source_type == 'file':
            raise NotImplementedError("请使用load_from_file方法加载文件数据")
        elif self.data_source_type == 'camera':
            return self.load_from_camera()
        else:
            raise ValueError(f"不支持的数据源类型: {self.data_source_type}")
    
    def load_from_file(self, img_path: str, temp_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        从文件加载数据
        
        Args:
            img_path: 红外图像路径
            temp_path: 温度数据文件路径（可选）
                - 如果为None，尝试从图像文件名推断（如image.png -> image.npy）
                - 如果图像包含温度元数据，可以直接从图像读取
        
        Returns:
            thermal_img: 红外图像 (H, W, 3)
            temp_matrix: 温度矩阵 (H, W)
        """
        # 加载图像
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件不存在: {img_path}")
        
        thermal_img = cv2.imread(img_path)
        if thermal_img is None:
            # 尝试使用中文路径读取
            thermal_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        if thermal_img is None:
            raise ValueError(f"无法读取图像: {img_path}")
        
        # 加载温度矩阵
        if temp_path is None:
            # 尝试从文件名推断
            base_name = os.path.splitext(img_path)[0]
            temp_path = base_name + self.temp_file_suffix
        
        if os.path.exists(temp_path):
            # 从文件加载温度矩阵
            if temp_path.endswith('.npy'):
                temp_matrix = np.load(temp_path)
            elif temp_path.endswith('.csv'):
                temp_matrix = np.loadtxt(temp_path, delimiter=',')
            elif temp_path.endswith('.txt'):
                temp_matrix = np.loadtxt(temp_path)
            else:
                raise ValueError(f"不支持的温度文件格式: {temp_path}")
        else:
            # 如果没有温度文件，尝试从图像生成模拟温度矩阵
            # 实际应用中，这需要根据红外相机的特性来实现
            print(f"警告: 温度文件不存在 {temp_path}，使用模拟温度矩阵")
            temp_matrix = self._generate_simulated_temp_matrix(thermal_img)
        
        # 确保尺寸匹配
        if thermal_img.shape[:2] != temp_matrix.shape:
            # 调整温度矩阵尺寸
            temp_matrix = cv2.resize(temp_matrix, 
                                   (thermal_img.shape[1], thermal_img.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)
        
        return thermal_img, temp_matrix
    
    def _generate_simulated_temp_matrix(self, img: np.ndarray) -> np.ndarray:
        """
        生成模拟温度矩阵（仅用于测试）
        
        实际应用中，应该从红外相机或温度数据文件获取真实温度
        
        Args:
            img: 红外图像
        
        Returns:
            模拟温度矩阵
        """
        h, w = img.shape[:2]
        # 将图像转换为灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # 归一化到0-1
        gray_norm = gray.astype(np.float32) / 255.0
        
        # 映射到温度范围（例如：30-45摄氏度）
        temp_matrix = gray_norm * 15 + 30
        
        return temp_matrix
    
    def load_from_camera(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        从红外相机实时获取数据
        
        Returns:
            thermal_img: 红外图像
            temp_matrix: 温度矩阵
        
        Raises:
            Exception: 如果相机未初始化或读取失败
        """
        if self.camera is None:
            raise Exception("相机未初始化")
        
        ret, frame = self.camera.read()
        if not ret:
            raise Exception("无法从相机读取数据")
        
        thermal_img = frame
        
        # 从相机获取温度矩阵
        # 实际应用中，需要使用红外相机SDK获取真实温度数据
        # 这里使用模拟数据
        temp_matrix = self._generate_simulated_temp_matrix(thermal_img)
        
        return thermal_img, temp_matrix
    
    def load_batch_from_directory(self, img_dir: str, temp_dir: Optional[str] = None):
        """
        批量从目录加载数据
        
        Args:
            img_dir: 图像目录
            temp_dir: 温度数据目录（可选）
        
        Yields:
            (img_path, thermal_img, temp_matrix) 元组
        """
        img_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        
        for filename in os.listdir(img_dir):
            if any(filename.lower().endswith(ext) for ext in img_extensions):
                img_path = os.path.join(img_dir, filename)
                
                if temp_dir:
                    temp_filename = os.path.splitext(filename)[0] + self.temp_file_suffix
                    temp_path = os.path.join(temp_dir, temp_filename)
                else:
                    temp_path = None
                
                try:
                    thermal_img, temp_matrix = self.load_from_file(img_path, temp_path)
                    yield img_path, thermal_img, temp_matrix
                except Exception as e:
                    print(f"加载失败 {img_path}: {e}")
                    continue
    
    def close(self):
        """关闭资源"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None


if __name__ == "__main__":
    # 测试代码
    loader = ThermalDataLoader(data_source_type='file')
    
    # 测试从文件加载
    test_img_dir = "datasets/images/test"
    if os.path.exists(test_img_dir):
        for img_path, thermal_img, temp_matrix in loader.load_batch_from_directory(test_img_dir):
            print(f"加载成功: {img_path}")
            print(f"  图像尺寸: {thermal_img.shape}")
            print(f"  温度矩阵尺寸: {temp_matrix.shape}")
            print(f"  温度范围: {np.min(temp_matrix):.2f} - {np.max(temp_matrix):.2f} °C")
            break
    else:
        print(f"测试目录不存在: {test_img_dir}")

