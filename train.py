#coding:utf-8
"""
鹿只高温预警系统 - 分割模型训练脚本
Deer Thermal Warning System - Segmentation Model Training Script
"""
from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 检查数据集配置文件是否存在
    data_yaml = 'datasets/data.yaml'
    if not os.path.exists(data_yaml):
        print(f"错误: 数据集配置文件不存在: {data_yaml}")
        print("请确保datasets/data.yaml文件存在")
        exit(1)
    
    # 加载分割预训练模型
    print("=" * 60)
    print("鹿只高温预警系统 - 分割模型训练")
    print("=" * 60)
    print("正在加载预训练模型: yolov8n-seg.pt")
    print("数据集配置: datasets/data.yaml")
    print("=" * 60)
    
    model = YOLO("yolov8n-seg.pt")  # 加载分割预训练模型
    
    # 训练参数配置
    # 可以根据实际情况调整以下参数
    train_args = {
        'data': data_yaml,           # 数据集配置文件路径
        'epochs': 250,                # 训练轮数（可根据需要调整）
        'batch': 4,                   # 批次大小（根据GPU内存调整：4, 8, 16等）
        'imgsz': 640,                 # 输入图像尺寸
        'device': 0,                  # 训练设备：0为GPU，'cpu'为CPU，或指定GPU编号如0,1
        'workers': 8,                 # 数据加载线程数
        'project': 'runs/segment',    # 项目保存目录
        'name': 'train',              # 训练运行名称
        'exist_ok': True,             # 允许覆盖已存在的训练结果
        'pretrained': True,           # 使用预训练权重
        'optimizer': 'auto',          # 优化器：'SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'auto'
        'verbose': True,              # 显示详细训练信息
        'seed': 0,                    # 随机种子（用于可复现性）
        'deterministic': True,        # 确定性训练
        'single_cls': False,          # 单类别模式（False表示多类别）
        'rect': False,                # 矩形训练（False表示方形训练）
        'cos_lr': False,              # 使用余弦学习率调度
        'close_mosaic': 10,           # 最后N个epoch关闭mosaic增强
        'resume': False,              # 是否从上次训练继续
        'amp': True,                  # 自动混合精度训练（加速训练）
        'fraction': 1.0,              # 使用数据集的比例（1.0表示全部使用）
        'profile': False,             # 性能分析
        'freeze': None,               # 冻结前N层（None表示不冻结）
        'lr0': 0.01,                  # 初始学习率
        'lrf': 0.01,                  # 最终学习率（lr0 * lrf）
        'momentum': 0.937,            # SGD动量
        'weight_decay': 0.0005,       # 权重衰减
        'warmup_epochs': 3.0,         # 预热轮数
        'warmup_momentum': 0.8,       # 预热动量
        'warmup_bias_lr': 0.1,        # 预热偏置学习率
        'box': 7.5,                   # 边界框损失权重
        'cls': 0.5,                   # 分类损失权重
        'dfl': 1.5,                   # DFL损失权重
        'pose': 12.0,                 # 姿态损失权重（分割任务不使用）
        'kobj': 1.0,                 # 关键点对象损失权重（分割任务不使用）
        'label_smoothing': 0.0,       # 标签平滑
        'nbs': 64,                    # 标称批次大小
        'overlap_mask': True,         # 训练时mask是否重叠（分割任务）
        'mask_ratio': 4,              # mask下采样比例（分割任务）
        'dropout': 0.0,               # Dropout（仅分类任务）
    }
    
    print("\n训练参数:")
    print(f"  训练轮数: {train_args['epochs']}")
    print(f"  批次大小: {train_args['batch']}")
    print(f"  图像尺寸: {train_args['imgsz']}")
    print(f"  训练设备: {'GPU' if train_args['device'] != 'cpu' else 'CPU'}")
    print(f"  优化器: {train_args['optimizer']}")
    print(f"  初始学习率: {train_args['lr0']}")
    print("\n开始训练...\n")
    
    # 开始训练
    try:
        results = model.train(**train_args)
        
        print("\n" + "=" * 60)
        print("训练完成！")
        print("=" * 60)
        print(f"最佳模型保存在: {results.save_dir}/weights/best.pt")
        print(f"最新模型保存在: {results.save_dir}/weights/last.pt")
        print("\n请将最佳模型复制到 models/best.pt:")
        print(f"  copy {results.save_dir}/weights/best.pt models/best.pt")
        print("=" * 60)
        
        # 可选：自动复制最佳模型到models目录
        import shutil
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        best_model_src = os.path.join(results.save_dir, 'weights', 'best.pt')
        best_model_dst = os.path.join(models_dir, 'best.pt')
        if os.path.exists(best_model_src):
            shutil.copy2(best_model_src, best_model_dst)
            print(f"\n✓ 已自动复制最佳模型到: {best_model_dst}")
        
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # 训练完成后，可以将模型转为onnx格式用于部署
    # print("\n正在导出ONNX格式...")
    # success = model.export(format='onnx')
    # if success:
    #     print("ONNX模型导出成功！")