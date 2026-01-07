# -*- coding: utf-8 -*-
"""
报警管理模块
Alert Manager Module
"""
import time
import logging
from typing import List, Dict, Tuple
import Config


class AlertManager:
    """
    报警管理器：管理报警逻辑和通知
    """
    
    def __init__(self, 
                 alert_methods: List[str] = None,
                 alert_cooldown: int = 5,
                 log_file: str = 'alerts.log'):
        """
        初始化报警管理器
        
        Args:
            alert_methods: 报警方式列表
                - 'log': 日志记录
                - 'visual': 视觉提示（在图像上标注）
                - 'sound': 声音报警
                - 'email': 邮件通知
                - 'sms': 短信通知
            alert_cooldown: 报警冷却时间（秒），避免频繁报警
            log_file: 日志文件路径
        """
        if alert_methods is None:
            config = getattr(Config, 'ALERT_CONFIG', {})
            alert_methods = config.get('methods', ['log', 'visual'])
        
        self.alert_methods = alert_methods
        self.alert_cooldown = alert_cooldown
        self.last_alert_time = {}  # {deer_id: timestamp}
        self.alert_history = []  # 报警历史记录
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def trigger_alert(self, 
                     deer_id: int,
                     stats: Dict,
                     position: Tuple[int, int, int, int],
                     timestamp: float = None):
        """
        触发报警
        
        Args:
            deer_id: 鹿只ID
            stats: 温度统计信息字典
            position: 位置信息 (x1, y1, x2, y2)
            timestamp: 时间戳（可选）
        """
        if timestamp is None:
            timestamp = time.time()
        
        # 检查冷却时间
        if deer_id in self.last_alert_time:
            time_since_last = timestamp - self.last_alert_time[deer_id]
            if time_since_last < self.alert_cooldown:
                return  # 仍在冷却期内
        
        # 更新最后报警时间
        self.last_alert_time[deer_id] = timestamp
        
        # 记录报警信息
        alert_info = {
            'deer_id': deer_id,
            'timestamp': timestamp,
            'stats': stats,
            'position': position
        }
        self.alert_history.append(alert_info)
        
        # 执行各种报警方式
        for method in self.alert_methods:
            try:
                if method == 'log':
                    self._log_alert(deer_id, stats, position)
                elif method == 'visual':
                    # 视觉报警由可视化模块处理
                    pass
                elif method == 'sound':
                    self._sound_alert()
                elif method == 'email':
                    self._email_alert(deer_id, stats)
                elif method == 'sms':
                    self._sms_alert(deer_id, stats)
            except Exception as e:
                self.logger.error(f"报警方式 {method} 执行失败: {e}")
    
    def _log_alert(self, deer_id: int, stats: Dict, position: Tuple):
        """日志报警"""
        max_temp = stats.get('max', 0)
        mean_temp = stats.get('mean', 0)
        x1, y1, x2, y2 = position
        
        message = (f"[高温报警] 鹿只 #{deer_id} | "
                  f"位置: ({x1}, {y1}, {x2}, {y2}) | "
                  f"最大温度: {max_temp:.2f}°C | "
                  f"平均温度: {mean_temp:.2f}°C")
        
        self.logger.warning(message)
    
    def _sound_alert(self):
        """声音报警"""
        try:
            import winsound  # Windows
            winsound.Beep(1000, 500)  # 频率1000Hz，持续500ms
        except ImportError:
            try:
                import os
                # Linux/Mac
                os.system('aplay /usr/share/sounds/alsa/Front_Left.wav 2>/dev/null || '
                         'afplay /System/Library/Sounds/Glass.aiff 2>/dev/null')
            except:
                print("\a")  # 系统提示音
    
    def _email_alert(self, deer_id: int, stats: Dict):
        """邮件报警（需要配置SMTP）"""
        # 这里需要实现邮件发送功能
        # 可以使用smtplib或第三方库如sendgrid
        config = getattr(Config, 'ALERT_CONFIG', {})
        if not config.get('email_enabled', False):
            return
        
        # 示例实现（需要配置SMTP服务器）
        try:
            import smtplib
            from email.mime.text import MIMEText
            
            # 配置SMTP（需要根据实际情况修改）
            smtp_server = config.get('smtp_server', 'smtp.example.com')
            smtp_port = config.get('smtp_port', 587)
            sender = config.get('email_sender', '')
            receiver = config.get('email_receiver', '')
            password = config.get('email_password', '')
            
            if not all([smtp_server, sender, receiver]):
                return
            
            msg = MIMEText(f"鹿只 #{deer_id} 高温报警\n"
                          f"最大温度: {stats.get('max', 0):.2f}°C\n"
                          f"平均温度: {stats.get('mean', 0):.2f}°C")
            msg['Subject'] = f'鹿只高温报警 - 鹿只 #{deer_id}'
            msg['From'] = sender
            msg['To'] = receiver
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"邮件报警已发送: 鹿只 #{deer_id}")
        except Exception as e:
            self.logger.error(f"邮件发送失败: {e}")
    
    def _sms_alert(self, deer_id: int, stats: Dict):
        """短信报警（需要配置短信服务）"""
        # 这里需要实现短信发送功能
        # 可以使用第三方服务如阿里云、腾讯云等
        config = getattr(Config, 'ALERT_CONFIG', {})
        if not config.get('sms_enabled', False):
            return
        
        # 示例实现（需要根据实际短信服务API实现）
        self.logger.info(f"短信报警功能需要配置短信服务API")
    
    def get_alert_statistics(self, time_window: int = 3600) -> Dict:
        """
        获取报警统计信息
        
        Args:
            time_window: 时间窗口（秒），统计最近多少秒的报警
        
        Returns:
            统计信息字典
        """
        current_time = time.time()
        recent_alerts = [
            alert for alert in self.alert_history
            if current_time - alert['timestamp'] <= time_window
        ]
        
        return {
            'total_alerts': len(self.alert_history),
            'recent_alerts': len(recent_alerts),
            'unique_deers': len(set(alert['deer_id'] for alert in recent_alerts)),
            'time_window': time_window
        }
    
    def clear_history(self):
        """清空报警历史"""
        self.alert_history = []
        self.last_alert_time = {}


if __name__ == "__main__":
    # 测试代码
    manager = AlertManager(
        alert_methods=['log', 'sound'],
        alert_cooldown=3
    )
    
    # 模拟报警
    stats = {'max': 42.5, 'mean': 40.2}
    position = (100, 100, 200, 200)
    
    print("触发报警1...")
    manager.trigger_alert(1, stats, position)
    
    time.sleep(1)
    
    print("触发报警2（应该被冷却期阻止）...")
    manager.trigger_alert(1, stats, position)
    
    time.sleep(4)
    
    print("触发报警3（冷却期已过）...")
    manager.trigger_alert(1, stats, position)
    
    # 获取统计信息
    stats_info = manager.get_alert_statistics()
    print(f"报警统计: {stats_info}")

