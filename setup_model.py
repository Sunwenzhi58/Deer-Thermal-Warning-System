# -*- coding: utf-8 -*-
"""
将训练好的模型复制到models目录
Copy trained model to models directory
"""
import os
import shutil

def setup_model():
    """设置模型文件"""
    # 训练好的模型路径
    trained_model = 'runs/segment/train/weights/best.pt'
    # 目标路径
    target_model = 'models/best.pt'
    
    # 检查训练好的模型是否存在
    if not os.path.exists(trained_model):
        print(f"错误: 训练好的模型不存在: {trained_model}")
        print("请先运行 train.py 训练模型")
        return False
    
    # 创建models目录（如果不存在）
    os.makedirs('models', exist_ok=True)
    
    # 复制模型文件
    try:
        shutil.copy2(trained_model, target_model)
        print(f"✓ 模型已成功复制到: {target_model}")
        print(f"  源文件: {trained_model}")
        
        # 检查文件大小
        size = os.path.getsize(target_model) / (1024 * 1024)  # MB
        print(f"  模型大小: {size:.2f} MB")
        return True
    except Exception as e:
        print(f"错误: 复制模型失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("设置模型文件")
    print("=" * 60)
    if setup_model():
        print("\n✓ 模型设置完成！现在可以运行 MainProgram.py 了")
    else:
        print("\n✗ 模型设置失败，请检查错误信息")

