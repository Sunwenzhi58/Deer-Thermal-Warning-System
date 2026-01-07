#coding:utf-8
# 鹿只高温预警系统配置文件
# Configuration file for Deer Thermal Warning System

# 图片及视频检测结果保存路径
save_path = 'save_data'

# 使用的模型路径
model_path = 'models/best.pt'

# 类别名称（英文）
names = {0: 'deer'}

# 类别名称（中文）
CH_names = ['鹿只']

# 高温预警阈值配置
TEMPERATURE_CONFIG = {
    'max_temp_threshold': 40.0,      # 最大温度阈值（摄氏度）
    'mean_temp_threshold': 38.5,     # 平均温度阈值（摄氏度）
    'alert_percentile': 99.5,        # 报警温度百分位数
    'min_pixels': 100,               # 最小有效像素数
}

# 报警配置
ALERT_CONFIG = {
    'methods': ['log', 'visual'],     # 报警方式: 'log', 'visual', 'sound', 'email', 'sms'
    'cooldown': 5,                   # 报警冷却时间（秒）
    'email_enabled': False,          # 是否启用邮件报警
    'sms_enabled': False,            # 是否启用短信报警
    # 邮件配置（如果启用）
    'smtp_server': 'smtp.example.com',
    'smtp_port': 587,
    'email_sender': '',
    'email_receiver': '',
    'email_password': '',
}

# 数据源配置
DATA_SOURCE_CONFIG = {
    'type': 'file',                  # 'file' 或 'camera'
    'camera_id': 0,                  # 相机ID
    'temp_file_suffix': '.npy',      # 温度文件后缀
}

