# -*- coding: utf-8 -*-
"""
鹿只高温预警系统 - 简单测试脚本
Deer Thermal Warning System - Simple Test Script
"""
import cv2
from ultralytics import YOLO
import Config
import detect_tools as tools
import os

def test_detection():
    """测试鹿只分割检测功能"""
    # 检查模型文件是否存在
    if not os.path.exists(Config.model_path):
        print(f"错误: 模型文件不存在: {Config.model_path}")
        print("请先训练分割模型或下载预训练模型到models/best.pt")
        return
    
    # 加载分割模型
    print("正在加载分割模型...")
    try:
        model = YOLO(Config.model_path, task='segment')  # 使用分割模式
        print("分割模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 查找测试图片
    test_image_dir = "datasets/images/test"
    if os.path.exists(test_image_dir):
        test_images = [f for f in os.listdir(test_image_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if test_images:
            test_image_path = os.path.join(test_image_dir, test_images[0])
            print(f"\n正在分割检测图片: {test_image_path}")
            
            # 执行分割检测
            results = model(test_image_path)[0]
            
            # 显示结果
            num_targets = len(results.boxes) if results.boxes is not None else 0
            print(f"检测到 {num_targets} 只鹿")
            
            if results.boxes is not None:
                for i in range(num_targets):
                    cls = int(results.boxes.cls[i])
                    conf = float(results.boxes.conf[i])
                    print(f"  鹿只 {i+1}: {Config.CH_names[cls]} - 置信度: {conf:.2%}")
            
            # 检查是否有分割mask
            if results.masks is not None:
                print(f"  分割mask数量: {len(results.masks)}")
            
            # 保存结果图像（自动绘制分割mask）
            result_img = results.plot()
            save_path = "save_data/test_result.jpg"
            os.makedirs("save_data", exist_ok=True)
            cv2.imwrite(save_path, result_img)
            print(f"\n结果已保存到: {save_path}")
        else:
            print(f"测试目录 {test_image_dir} 中没有找到图片文件")
    else:
        print(f"测试目录 {test_image_dir} 不存在")
        print("请确保数据集已正确放置在datasets目录下")

if __name__ == "__main__":
    test_detection()

