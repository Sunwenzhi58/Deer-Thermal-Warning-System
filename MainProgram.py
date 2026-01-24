# -*- coding: utf-8 -*-
"""
鹿只高温预警系统 - 主程序
Deer Thermal Warning System - Main Program
"""
import os
import sys

# 解决 PyQt5 和 OpenCV Qt 插件冲突问题
# Fix PyQt5 and OpenCV Qt plugin conflict
# 必须在导入 PyQt5 和 cv2 之前设置
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
if hasattr(sys, 'frozen'):
    pass
else:
    # 对于 Linux 系统，确保使用系统的 Qt 插件
    if sys.platform.startswith('linux'):
        os.environ["QT_QPA_PLATFORM"] = "xcb"

import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, \
    QMessageBox, QWidget, QHeaderView, QTableWidgetItem, QAbstractItemView
from PyQt5.QtGui import QColor, QGuiApplication
from PIL import ImageFont, ImageDraw, Image
from ultralytics import YOLO
sys.path.append('UIProgram')
try:
    from UIProgram.UiMain import Ui_MainWindow
except ImportError:
    print("警告: UIProgram/UiMain.py 不存在，请使用Qt Designer创建UI文件")
    print("Warning: UIProgram/UiMain.py not found, please create UI file using Qt Designer")
    sys.exit(1)

from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QCoreApplication
import detect_tools as tools
import cv2
import Config
from UIProgram.QssLoader import QSSLoader
from UIProgram.precess_bar import ProgressBar
import numpy as np
from temperature_analyzer import TemperatureAnalyzer
from thermal_data_loader import ThermalDataLoader
from alert_manager import AlertManager


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(QMainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.initMain()
        self.signalconnect()

        # 加载css渲染效果
        style_file = 'UIProgram/style.css'
        if os.path.exists(style_file):
            qssStyleSheet = QSSLoader.read_qss_file(style_file)
            self.setStyleSheet(qssStyleSheet)
        
        # 优化图像显示区域背景（样式已在CSS中定义，这里可以添加额外效果）
        # 图像显示区域的样式已在style.css中统一管理

    def signalconnect(self):
        """连接信号和槽函数"""
        self.ui.PicBtn.clicked.connect(self.open_img)
        self.ui.comboBox.activated.connect(self.combox_change)
        self.ui.VideoBtn.clicked.connect(self.vedio_show)
        self.ui.CapBtn.clicked.connect(self.camera_show)
        self.ui.SaveBtn.clicked.connect(self.save_detect_video)
        self.ui.ExitBtn.clicked.connect(QCoreApplication.quit)
        self.ui.FilesBtn.clicked.connect(self.detact_batch_imgs)

    def initMain(self):
        """初始化主窗口"""
        self.show_width = 770
        self.show_height = 480

        self.org_path = None

        self.is_camera_open = False
        self.cap = None

        # 加载分割模型（数据集是分割格式）
        try:
            self.model = YOLO(Config.model_path, task='segment')  # 使用分割模式
            self.model(np.zeros((48, 48, 3)))  # 预先加载推理模型
        except Exception as e:
            QMessageBox.warning(self, '警告', f'模型加载失败: {str(e)}\n请确保models/best.pt文件存在（分割模型）')
            print(f"模型加载错误: {e}")

        # 加载字体（如果存在）
        try:
            self.fontC = ImageFont.truetype("Font/platech.ttf", 25, 0)
        except:
            try:
                # 尝试使用系统默认字体
                self.fontC = ImageFont.truetype("simsun.ttc", 25, 0)
            except:
                self.fontC = ImageFont.load_default()

        # 用于绘制不同颜色矩形框
        self.colors = tools.Colors()

        # 初始化温度分析器
        temp_config = getattr(Config, 'TEMPERATURE_CONFIG', {})
        self.temp_analyzer = TemperatureAnalyzer(
            max_temp_threshold=temp_config.get('max_temp_threshold', 40.0),
            mean_temp_threshold=temp_config.get('mean_temp_threshold', 38.5),
            alert_percentile=temp_config.get('alert_percentile', 99.5),
            min_pixels=temp_config.get('min_pixels', 100)
        )
        
        # 初始化热成像数据加载器
        self.thermal_loader = ThermalDataLoader(data_source_type='file')
        
        # 初始化报警管理器
        self.alert_manager = AlertManager()
        
        # 存储温度统计信息
        self.temp_stats_list = []  # 每个检测目标的温度统计

        # 更新视频图像
        self.timer_camera = QTimer()

        # 表格设置 - 简洁风格
        self.ui.tableWidget.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.ui.tableWidget.verticalHeader().setDefaultSectionSize(38)  # 适中的行高
        self.ui.tableWidget.setColumnCount(6)  # 增加一列用于显示温度
        self.ui.tableWidget.setColumnWidth(0, 80)  # 设置列宽
        self.ui.tableWidget.setColumnWidth(1, 200)
        self.ui.tableWidget.setColumnWidth(2, 150)
        self.ui.tableWidget.setColumnWidth(3, 90)
        self.ui.tableWidget.setColumnWidth(4, 230)
        self.ui.tableWidget.setColumnWidth(5, 150)  # 温度列
        self.ui.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)  # 设置表格整行选中
        self.ui.tableWidget.verticalHeader().setVisible(False)  # 隐藏列标题
        self.ui.tableWidget.setAlternatingRowColors(True)  # 表格背景交替
        self.ui.tableWidget.setShowGrid(True)  # 显示网格线
        self.ui.tableWidget.setGridStyle(Qt.SolidLine)  # 设置网格线样式
        
        # 设置表头（包括温度列）
        self.ui.tableWidget.setHorizontalHeaderItem(0, QTableWidgetItem("序号"))
        self.ui.tableWidget.setHorizontalHeaderItem(1, QTableWidgetItem("文件路径"))
        self.ui.tableWidget.setHorizontalHeaderItem(2, QTableWidgetItem("类别"))
        self.ui.tableWidget.setHorizontalHeaderItem(3, QTableWidgetItem("置信度"))
        self.ui.tableWidget.setHorizontalHeaderItem(4, QTableWidgetItem("坐标位置"))
        self.ui.tableWidget.setHorizontalHeaderItem(5, QTableWidgetItem("温度信息"))

    def open_img(self):
        """打开单张图片进行鹿只分割检测"""
        if self.cap:
            # 打开图片前关闭摄像头
            self.video_stop()
            self.is_camera_open = False
            self.ui.CaplineEdit.setText('摄像头未开启')
            self.cap = None

        file_path, _ = QFileDialog.getOpenFileName(None, '打开图片', './', "Image files (*.jpg *.jpeg *.png)")
        if not file_path:
            return

        self.ui.comboBox.setDisabled(False)
        self.org_path = file_path
        self.org_img = tools.img_cvread(self.org_path)

        # 加载温度数据
        try:
            thermal_img, temp_matrix = self.thermal_loader.load_from_file(self.org_path)
        except Exception as e:
            print(f"警告: 无法加载温度数据: {e}，使用模拟温度矩阵")
            # 如果无法加载温度数据，生成模拟温度矩阵
            h, w = self.org_img.shape[:2]
            gray = cv2.cvtColor(self.org_img, cv2.COLOR_BGR2GRAY) if len(self.org_img.shape) == 3 else self.org_img
            gray_norm = gray.astype(np.float32) / 255.0
            temp_matrix = gray_norm * 15 + 30  # 映射到30-45摄氏度范围
            thermal_img = self.org_img

        # 鹿只分割检测
        t1 = time.time()
        try:
            self.results = self.model(self.org_path)[0]
        except Exception as e:
            QMessageBox.warning(self, '错误', f'分割检测失败: {str(e)}')
            return
        t2 = time.time()
        take_time_str = '{:.3f} s'.format(t2 - t1)
        self.ui.time_lb.setText(take_time_str)

        location_list = self.results.boxes.xyxy.tolist()
        self.location_list = [list(map(int, e)) for e in location_list]
        cls_list = self.results.boxes.cls.tolist()
        self.cls_list = [int(i) for i in cls_list]
        self.conf_list = self.results.boxes.conf.tolist()
        self.conf_list = ['%.2f %%' % (each*100) for each in self.conf_list]

        # 提取分割mask并分析温度
        self.temp_stats_list = []
        masks = self.results.masks
        if masks is not None:
            for i, (box, mask) in enumerate(zip(self.location_list, masks.data)):
                # 获取mask并调整尺寸
                mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask
                if mask_np.shape != temp_matrix.shape:
                    mask_resized = cv2.resize(mask_np.astype(np.uint8), 
                                             (temp_matrix.shape[1], temp_matrix.shape[0]),
                                             interpolation=cv2.INTER_NEAREST) > 0.5
                else:
                    mask_resized = mask_np.astype(bool)
                
                # 提取温度
                temps = self.temp_analyzer.extract_temperature(temp_matrix, mask_resized)
                # 分析温度
                temp_stats = self.temp_analyzer.analyze_temperature(temps)
                self.temp_stats_list.append(temp_stats)
                
                # 检查是否需要报警
                if temp_stats and self.temp_analyzer.should_alert(temp_stats):
                    self.alert_manager.trigger_alert(
                        deer_id=i,
                        stats=temp_stats,
                        position=tuple(box)
                    )
        else:
            # 如果没有mask，使用bbox区域
            for i, box in enumerate(self.location_list):
                temps = self.temp_analyzer.extract_temperature(temp_matrix, tuple(box))
                temp_stats = self.temp_analyzer.analyze_temperature(temps)
                self.temp_stats_list.append(temp_stats)
                
                if temp_stats and self.temp_analyzer.should_alert(temp_stats):
                    self.alert_manager.trigger_alert(
                        deer_id=i,
                        stats=temp_stats,
                        position=tuple(box)
                    )

        # 绘制检测结果和温度信息
        now_img = self.results.plot()
        now_img = self._draw_temperature_info(now_img, self.location_list, self.temp_stats_list)
        self.draw_img = now_img
        # 获取缩放后的图片尺寸
        self.img_width, self.img_height = self.get_resize_size(now_img)
        resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show.setPixmap(pix_img)
        self.ui.label_show.setAlignment(Qt.AlignCenter)
        # 设置路径显示
        self.ui.PiclineEdit.setText(self.org_path)

        # 目标数目
        target_nums = len(self.cls_list)
        self.ui.label_nums.setText(str(target_nums))

        # 设置目标选择下拉框
        choose_list = ['全部']
        target_names = [Config.CH_names[id] + '_' + str(index) for index, id in enumerate(self.cls_list)]
        choose_list = choose_list + target_names

        self.ui.comboBox.clear()
        self.ui.comboBox.addItems(choose_list)

        if target_nums >= 1:
            self.ui.label_conf.setText(str(self.conf_list[0]))
            # 默认显示第一个目标框坐标
            self.ui.label_xmin.setText(str(self.location_list[0][0]))
            self.ui.label_ymin.setText(str(self.location_list[0][1]))
            self.ui.label_xmax.setText(str(self.location_list[0][2]))
            self.ui.label_ymax.setText(str(self.location_list[0][3]))
            
            # 显示第一个目标的温度信息
            if len(self.temp_stats_list) > 0 and self.temp_stats_list[0] is not None:
                stats = self.temp_stats_list[0]
                max_temp = stats.get('max', 0)
                mean_temp = stats.get('mean', 0)
                temp_text = f"最大:{max_temp:.1f}°C 平均:{mean_temp:.1f}°C"
                # 如果UI中有温度标签，显示温度
                # 这里可以添加一个温度显示标签，暂时使用置信度标签下方显示
        else:
            self.ui.label_conf.setText('')
            self.ui.label_xmin.setText('')
            self.ui.label_ymin.setText('')
            self.ui.label_xmax.setText('')
            self.ui.label_ymax.setText('')

        # 删除表格所有行
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()
        self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, path=self.org_path)

    def detact_batch_imgs(self):
        """批量检测图片（鹿只分割）"""
        if self.cap:
            # 打开图片前关闭摄像头
            self.video_stop()
            self.is_camera_open = False
            self.ui.CaplineEdit.setText('摄像头未开启')
            self.cap = None
        directory = QFileDialog.getExistingDirectory(self,
                                                      "选取文件夹",
                                                      "./")  # 起始路径
        if not directory:
            return
        self.org_path = directory
        img_suffix = ['jpg', 'png', 'jpeg', 'bmp']
        for file_name in os.listdir(directory):
            full_path = os.path.join(directory, file_name)
            if os.path.isfile(full_path) and file_name.split('.')[-1].lower() in img_suffix:
                img_path = full_path
                self.org_img = tools.img_cvread(img_path)
                
                # 加载温度数据
                try:
                    thermal_img, temp_matrix = self.thermal_loader.load_from_file(img_path)
                except Exception as e:
                    print(f"警告: 无法加载温度数据: {e}，使用模拟温度矩阵")
                    h, w = self.org_img.shape[:2]
                    gray = cv2.cvtColor(self.org_img, cv2.COLOR_BGR2GRAY) if len(self.org_img.shape) == 3 else self.org_img
                    gray_norm = gray.astype(np.float32) / 255.0
                    temp_matrix = gray_norm * 15 + 30
                
                # 鹿只分割检测
                t1 = time.time()
                try:
                    self.results = self.model(img_path)[0]
                except Exception as e:
                    print(f"分割检测失败 {img_path}: {e}")
                    continue
                t2 = time.time()
                take_time_str = '{:.3f} s'.format(t2 - t1)
                self.ui.time_lb.setText(take_time_str)

                location_list = self.results.boxes.xyxy.tolist()
                self.location_list = [list(map(int, e)) for e in location_list]
                cls_list = self.results.boxes.cls.tolist()
                self.cls_list = [int(i) for i in cls_list]
                self.conf_list = self.results.boxes.conf.tolist()
                self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]

                # 提取温度信息
                self.temp_stats_list = []
                masks = self.results.masks
                if masks is not None:
                    for i, (box, mask) in enumerate(zip(self.location_list, masks.data)):
                        mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask
                        if mask_np.shape != temp_matrix.shape:
                            mask_resized = cv2.resize(mask_np.astype(np.uint8), 
                                                     (temp_matrix.shape[1], temp_matrix.shape[0]),
                                                     interpolation=cv2.INTER_NEAREST) > 0.5
                        else:
                            mask_resized = mask_np.astype(bool)
                        temps = self.temp_analyzer.extract_temperature(temp_matrix, mask_resized)
                        temp_stats = self.temp_analyzer.analyze_temperature(temps)
                        self.temp_stats_list.append(temp_stats)
                        if temp_stats and self.temp_analyzer.should_alert(temp_stats):
                            self.alert_manager.trigger_alert(i, temp_stats, tuple(box))
                else:
                    for i, box in enumerate(self.location_list):
                        temps = self.temp_analyzer.extract_temperature(temp_matrix, tuple(box))
                        temp_stats = self.temp_analyzer.analyze_temperature(temps)
                        self.temp_stats_list.append(temp_stats)
                        if temp_stats and self.temp_analyzer.should_alert(temp_stats):
                            self.alert_manager.trigger_alert(i, temp_stats, tuple(box))

                now_img = self.results.plot()
                now_img = self._draw_temperature_info(now_img, self.location_list, self.temp_stats_list)

                self.draw_img = now_img
                # 获取缩放后的图片尺寸
                self.img_width, self.img_height = self.get_resize_size(now_img)
                resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
                pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
                self.ui.label_show.setPixmap(pix_img)
                self.ui.label_show.setAlignment(Qt.AlignCenter)
                # 设置路径显示
                self.ui.PiclineEdit.setText(img_path)

                # 目标数目
                target_nums = len(self.cls_list)
                self.ui.label_nums.setText(str(target_nums))

                # 设置目标选择下拉框
                choose_list = ['全部']
                target_names = [Config.CH_names[id] + '_' + str(index) for index, id in enumerate(self.cls_list)]
                choose_list = choose_list + target_names

                self.ui.comboBox.clear()
                self.ui.comboBox.addItems(choose_list)

                if target_nums >= 1:
                    self.ui.label_conf.setText(str(self.conf_list[0]))
                    self.ui.label_xmin.setText(str(self.location_list[0][0]))
                    self.ui.label_ymin.setText(str(self.location_list[0][1]))
                    self.ui.label_xmax.setText(str(self.location_list[0][2]))
                    self.ui.label_ymax.setText(str(self.location_list[0][3]))
                else:
                    self.ui.label_conf.setText('')
                    self.ui.label_xmin.setText('')
                    self.ui.label_ymin.setText('')
                    self.ui.label_xmax.setText('')
                    self.ui.label_ymax.setText('')

                self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, path=img_path)
                self.ui.tableWidget.scrollToBottom()
                QApplication.processEvents()  # 刷新页面

    def combox_change(self):
        """下拉框选择改变时的处理"""
        com_text = self.ui.comboBox.currentText()
        print(com_text)
        if com_text == '全部':
            cur_box = self.location_list
            cur_img = self.results.plot()
            cur_img = self._draw_temperature_info(cur_img, self.location_list, self.temp_stats_list)
            if len(self.conf_list) > 0:
                self.ui.label_conf.setText(str(self.conf_list[0]))
        else:
            index = int(com_text.split('_')[-1])
            cur_box = [self.location_list[index]]
            cur_img = self.results[index].plot()
            # 绘制单个目标的温度信息
            if index < len(self.temp_stats_list):
                cur_temp_stats = [self.temp_stats_list[index]]
                cur_img = self._draw_temperature_info(cur_img, cur_box, cur_temp_stats)
            self.ui.label_conf.setText(str(self.conf_list[index]))

        # 设置坐标位置值
        if len(cur_box) > 0:
            self.ui.label_xmin.setText(str(cur_box[0][0]))
            self.ui.label_ymin.setText(str(cur_box[0][1]))
            self.ui.label_xmax.setText(str(cur_box[0][2]))
            self.ui.label_ymax.setText(str(cur_box[0][3]))

        resize_cvimg = cv2.resize(cur_img, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.ui.label_show.clear()
        self.ui.label_show.setPixmap(pix_img)
        self.ui.label_show.setAlignment(Qt.AlignCenter)

    def get_video_path(self):
        """获取视频路径"""
        file_path, _ = QFileDialog.getOpenFileName(None, '打开视频', './', "Video files (*.avi *.mp4)")
        if not file_path:
            return None
        self.org_path = file_path
        self.ui.VideolineEdit.setText(file_path)
        return file_path

    def video_start(self):
        """开始视频分割检测"""
        # 删除表格所有行
        self.ui.tableWidget.setRowCount(0)
        self.ui.tableWidget.clearContents()

        # 清空下拉框
        self.ui.comboBox.clear()

        # 定时器开启，每隔一段时间，读取一帧
        self.timer_camera.start(1)
        self.timer_camera.timeout.connect(self.open_frame)

    def tabel_info_show(self, locations, clses, confs, path=None):
        """在表格中显示分割检测信息和温度"""
        path = path
        for idx, (location, cls, conf) in enumerate(zip(locations, clses, confs)):
            row_count = self.ui.tableWidget.rowCount()  # 返回当前行数(尾部)
            self.ui.tableWidget.insertRow(row_count)  # 尾部插入一行
            item_id = QTableWidgetItem(str(row_count+1))  # 序号
            item_id.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中
            item_path = QTableWidgetItem(str(path))  # 路径
            item_path.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)  # 路径左对齐，便于阅读长路径

            item_cls = QTableWidgetItem(str(Config.CH_names[cls]))
            item_cls.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中

            item_conf = QTableWidgetItem(str(conf))
            item_conf.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 设置文本居中

            item_location = QTableWidgetItem(str(location))  # 目标框位置
            item_location.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 坐标居中

            # 温度信息
            temp_info = "N/A"
            if idx < len(self.temp_stats_list) and self.temp_stats_list[idx] is not None:
                stats = self.temp_stats_list[idx]
                max_temp = stats.get('max', 0)
                mean_temp = stats.get('mean', 0)
                alert_level = self.temp_analyzer.get_alert_level(stats)
                if alert_level != 'none':
                    temp_info = f"最大:{max_temp:.1f}°C 平均:{mean_temp:.1f}°C [预警]"
                else:
                    temp_info = f"最大:{max_temp:.1f}°C 平均:{mean_temp:.1f}°C"
            
            item_temp = QTableWidgetItem(temp_info)
            item_temp.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            # 如果是高温预警，设置红色字体和加粗
            if "[预警]" in temp_info:
                item_temp.setForeground(QColor(204, 0, 0))  # 使用红色
                font = item_temp.font()
                font.setBold(True)
                item_temp.setFont(font)

            self.ui.tableWidget.setItem(row_count, 0, item_id)
            self.ui.tableWidget.setItem(row_count, 1, item_path)
            self.ui.tableWidget.setItem(row_count, 2, item_cls)
            self.ui.tableWidget.setItem(row_count, 3, item_conf)
            self.ui.tableWidget.setItem(row_count, 4, item_location)
            self.ui.tableWidget.setItem(row_count, 5, item_temp)
        self.ui.tableWidget.scrollToBottom()

    def video_stop(self):
        """停止视频分割检测"""
        if self.cap:
            self.cap.release()
        self.timer_camera.stop()

    def open_frame(self):
        """打开视频帧进行鹿只分割检测"""
        ret, now_img = self.cap.read()
        if ret:
            # 生成模拟温度矩阵（视频模式下）
            h, w = now_img.shape[:2]
            gray = cv2.cvtColor(now_img, cv2.COLOR_BGR2GRAY) if len(now_img.shape) == 3 else now_img
            gray_norm = gray.astype(np.float32) / 255.0
            temp_matrix = gray_norm * 15 + 30
            
            # 鹿只分割检测
            t1 = time.time()
            try:
                results = self.model(now_img)[0]
            except Exception as e:
                print(f"分割检测失败: {e}")
                return
            t2 = time.time()
            take_time_str = '{:.3f} s'.format(t2 - t1)
            self.ui.time_lb.setText(take_time_str)

            location_list = results.boxes.xyxy.tolist()
            self.location_list = [list(map(int, e)) for e in location_list]
            cls_list = results.boxes.cls.tolist()
            self.cls_list = [int(i) for i in cls_list]
            self.conf_list = results.boxes.conf.tolist()
            self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]

            # 提取温度信息
            self.temp_stats_list = []
            masks = results.masks
            if masks is not None:
                for i, (box, mask) in enumerate(zip(self.location_list, masks.data)):
                    mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask
                    if mask_np.shape != temp_matrix.shape:
                        mask_resized = cv2.resize(mask_np.astype(np.uint8), 
                                                 (temp_matrix.shape[1], temp_matrix.shape[0]),
                                                 interpolation=cv2.INTER_NEAREST) > 0.5
                    else:
                        mask_resized = mask_np.astype(bool)
                    temps = self.temp_analyzer.extract_temperature(temp_matrix, mask_resized)
                    temp_stats = self.temp_analyzer.analyze_temperature(temps)
                    self.temp_stats_list.append(temp_stats)
                    if temp_stats and self.temp_analyzer.should_alert(temp_stats):
                        self.alert_manager.trigger_alert(i, temp_stats, tuple(box))
            else:
                for i, box in enumerate(self.location_list):
                    temps = self.temp_analyzer.extract_temperature(temp_matrix, tuple(box))
                    temp_stats = self.temp_analyzer.analyze_temperature(temps)
                    self.temp_stats_list.append(temp_stats)
                    if temp_stats and self.temp_analyzer.should_alert(temp_stats):
                        self.alert_manager.trigger_alert(i, temp_stats, tuple(box))

            now_img = results.plot()
            now_img = self._draw_temperature_info(now_img, self.location_list, self.temp_stats_list)

            # 获取缩放后的图片尺寸
            self.img_width, self.img_height = self.get_resize_size(now_img)
            resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
            pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
            self.ui.label_show.setPixmap(pix_img)
            self.ui.label_show.setAlignment(Qt.AlignCenter)

            # 目标数目
            target_nums = len(self.cls_list)
            self.ui.label_nums.setText(str(target_nums))

            # 设置目标选择下拉框
            choose_list = ['全部']
            target_names = [Config.CH_names[id] + '_' + str(index) for index, id in enumerate(self.cls_list)]
            choose_list = choose_list + target_names

            self.ui.comboBox.clear()
            self.ui.comboBox.addItems(choose_list)

            if target_nums >= 1:
                self.ui.label_conf.setText(str(self.conf_list[0]))
                self.ui.label_xmin.setText(str(self.location_list[0][0]))
                self.ui.label_ymin.setText(str(self.location_list[0][1]))
                self.ui.label_xmax.setText(str(self.location_list[0][2]))
                self.ui.label_ymax.setText(str(self.location_list[0][3]))
            else:
                self.ui.label_conf.setText('')
                self.ui.label_xmin.setText('')
                self.ui.label_ymin.setText('')
                self.ui.label_xmax.setText('')
                self.ui.label_ymax.setText('')

            self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, path=self.org_path)

        else:
            self.cap.release()
            self.timer_camera.stop()

    def vedio_show(self):
        """显示视频分割检测"""
        if self.is_camera_open:
            self.is_camera_open = False
            self.ui.CaplineEdit.setText('摄像头未开启')

        video_path = self.get_video_path()
        if not video_path:
            return None
        self.cap = cv2.VideoCapture(video_path)
        self.video_start()
        self.ui.comboBox.setDisabled(True)

    def camera_show(self):
        """摄像头实时分割检测"""
        self.is_camera_open = not self.is_camera_open
        if self.is_camera_open:
            self.ui.CaplineEdit.setText('摄像头开启')
            self.cap = cv2.VideoCapture(0)
            self.video_start()
            self.ui.comboBox.setDisabled(True)
        else:
            self.ui.CaplineEdit.setText('摄像头未开启')
            self.ui.label_show.setText('')
            if self.cap:
                self.cap.release()
                cv2.destroyAllWindows()
            self.ui.label_show.clear()

    def _draw_temperature_info(self, img, locations, temp_stats_list):
        """在图像上绘制温度信息（仅标注超过50度的温度）"""
        result_img = img.copy()
        
        for idx, (location, temp_stats) in enumerate(zip(locations, temp_stats_list)):
            if temp_stats is None:
                continue
                
            x1, y1, x2, y2 = location
            max_temp = temp_stats.get('max', 0)
            mean_temp = temp_stats.get('mean', 0)
            
            # 只有超过50度阈值的才需要标注在图上
            if max_temp <= 50:
                continue
            
            alert_level = self.temp_analyzer.get_alert_level(temp_stats)
            
            # 根据报警级别选择颜色
            if alert_level == 'critical':
                color = (0, 0, 255)  # 红色 - 严重
            elif alert_level == 'alert':
                color = (0, 165, 255)  # 橙色 - 报警
            elif alert_level == 'warning':
                color = (0, 255, 255)  # 黄色 - 警告
            else:
                color = (0, 255, 0)  # 绿色 - 正常
            
            # 绘制温度文本
            temp_text = f"Max:{max_temp:.1f}°C Mean:{mean_temp:.1f}°C"
            if alert_level != 'none':
                temp_text += " [高温预警]"
            
            # 使用PIL绘制中文
            img_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # 计算文本尺寸
            try:
                # PIL 10.0.0+ 使用textbbox
                text_bbox = draw.textbbox((x1, y2 + 5), temp_text, font=self.fontC)
                bg_x1, bg_y1, bg_x2, bg_y2 = text_bbox
            except AttributeError:
                # 旧版本PIL使用textsize
                text_size = draw.textsize(temp_text, font=self.fontC)
                bg_x1, bg_y1 = x1, y2 + 5
                bg_x2, bg_y2 = x1 + text_size[0], y2 + 5 + text_size[1]
            
            # 绘制文本背景（注意PIL使用RGB颜色）
            color_rgb = (color[2], color[1], color[0])  # BGR转RGB
            draw.rectangle([bg_x1-2, bg_y1-2, bg_x2+2, bg_y2+2], fill=color_rgb)
            
            # 绘制文本
            draw.text((x1, y2 + 5), temp_text, fill=(255, 255, 255), font=self.fontC)
            
            result_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        return result_img

    def get_resize_size(self, img):
        """获取缩放后的图片尺寸"""
        _img = img.copy()
        img_height, img_width, depth = _img.shape
        ratio = img_width / img_height
        if ratio >= self.show_width / self.show_height:
            self.img_width = self.show_width
            self.img_height = int(self.img_width / ratio)
        else:
            self.img_height = self.show_height
            self.img_width = int(self.img_height * ratio)
        return self.img_width, self.img_height

    def save_detect_video(self):
        """保存分割检测结果"""
        if self.cap is None and not self.org_path:
            QMessageBox.about(self, '提示', '当前没有可保存信息，请先打开图片或视频！')
            return

        if self.is_camera_open:
            QMessageBox.about(self, '提示', '摄像头视频无法保存!')
            return

        if self.cap:
            res = QMessageBox.information(self, '提示', '保存视频分割检测结果可能需要较长时间，请确认是否继续保存？',
                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if res == QMessageBox.Yes:
                self.video_stop()
                com_text = self.ui.comboBox.currentText()
                self.btn2Thread_object = btn2Thread(
                    self.org_path, 
                    self.model, 
                    com_text,
                    temp_analyzer=self.temp_analyzer,
                    draw_temp_func=self._draw_temperature_info
                )
                self.btn2Thread_object.start()
                self.btn2Thread_object.update_ui_signal.connect(self.update_process_bar)
            else:
                return
        else:
            if os.path.isfile(self.org_path):
                fileName = os.path.basename(self.org_path)
                name, end_name = fileName.split('.')
                save_name = name + '_detect_result.' + end_name
                save_img_path = os.path.join(Config.save_path, save_name)
                # 确保保存目录存在
                os.makedirs(Config.save_path, exist_ok=True)
                # 保存图片
                cv2.imwrite(save_img_path, self.draw_img)
                QMessageBox.about(self, '提示', '图片保存成功!\n文件路径:{}'.format(save_img_path))
            else:
                img_suffix = ['jpg', 'png', 'jpeg', 'bmp']
                os.makedirs(Config.save_path, exist_ok=True)
                for file_name in os.listdir(self.org_path):
                    full_path = os.path.join(self.org_path, file_name)
                    if os.path.isfile(full_path) and file_name.split('.')[-1].lower() in img_suffix:
                        name, end_name = file_name.split('.')
                        save_name = name + '_detect_result.' + end_name
                        save_img_path = os.path.join(Config.save_path, save_name)
                        results = self.model(full_path)[0]
                        now_img = results.plot()
                        # 保存图片
                        cv2.imwrite(save_img_path, now_img)

                QMessageBox.about(self, '提示', '图片保存成功!\n文件路径:{}'.format(Config.save_path))

    def update_process_bar(self, cur_num, total):
        """更新进度条"""
        if cur_num == 1:
            self.progress_bar = ProgressBar(self)
            self.progress_bar.show()
        if cur_num >= total:
            self.progress_bar.close()
            QMessageBox.about(self, '提示', '视频保存成功!\n文件在{}目录下'.format(Config.save_path))
            return
        if self.progress_bar.isVisible() is False:
            # 点击取消保存时，终止进程
            self.btn2Thread_object.stop()
            return
        value = int(cur_num / total * 100)
        self.progress_bar.setValue(cur_num, total, value)
        QApplication.processEvents()


class btn2Thread(QThread):
    """
    进行分割检测后的视频保存（多线程处理）
    """
    # 声明一个信号
    update_ui_signal = pyqtSignal(int, int)

    def __init__(self, path, model, com_text, temp_analyzer=None, draw_temp_func=None):
        super(btn2Thread, self).__init__()
        self.org_path = path
        self.model = model
        self.com_text = com_text
        self.temp_analyzer = temp_analyzer
        self.draw_temp_func = draw_temp_func
        # 用于绘制不同颜色矩形框
        self.colors = tools.Colors()
        self.is_running = True  # 标志位，表示线程是否正在运行

    def run(self):
        # VideoCapture方法是cv2库提供的读取视频方法
        cap = cv2.VideoCapture(self.org_path)
        # 设置需要保存视频的格式"xvid"
        # 该参数是MPEG-4编码类型，文件名后缀为.avi
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # 设置视频帧频
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 设置视频大小
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # VideoWriter方法是cv2库提供的保存视频方法
        # 按照设置的格式来out输出
        fileName = os.path.basename(self.org_path)
        name, end_name = fileName.split('.')
        save_name = name + '_detect_result.avi'
        save_video_path = os.path.join(Config.save_path, save_name)
        os.makedirs(Config.save_path, exist_ok=True)
        out = cv2.VideoWriter(save_video_path, fourcc, fps, size)

        prop = cv2.CAP_PROP_FRAME_COUNT
        total = int(cap.get(prop))
        print("[INFO] 视频总帧数：{}".format(total))
        cur_num = 0

        # 确定视频打开并循环读取
        while (cap.isOpened() and self.is_running):
            cur_num += 1
            print('当前第{}帧，总帧数{}'.format(cur_num, total))
            # 逐帧读取，ret返回布尔值
            # 参数ret为True 或者False,代表有没有读取到图片
            # frame表示截取到一帧的图片
            ret, frame = cap.read()
            if ret == True:
                # 分割检测
                results = self.model(frame)[0]
                frame = results.plot()  # 自动绘制分割mask
                
                # 如果温度分析器可用，提取并绘制温度信息
                if self.temp_analyzer is not None and self.draw_temp_func is not None:
                    h, w = frame.shape[:2]
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                    gray_norm = gray.astype(np.float32) / 255.0
                    temp_matrix = gray_norm * 15 + 30
                    
                    location_list = results.boxes.xyxy.tolist()
                    location_list = [list(map(int, e)) for e in location_list]
                    temp_stats_list = []
                    
                    masks = results.masks
                    if masks is not None:
                        for box, mask in zip(location_list, masks.data):
                            mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask
                            if mask_np.shape != temp_matrix.shape:
                                mask_resized = cv2.resize(mask_np.astype(np.uint8), 
                                                         (temp_matrix.shape[1], temp_matrix.shape[0]),
                                                         interpolation=cv2.INTER_NEAREST) > 0.5
                            else:
                                mask_resized = mask_np.astype(bool)
                            temps = self.temp_analyzer.extract_temperature(temp_matrix, mask_resized)
                            temp_stats = self.temp_analyzer.analyze_temperature(temps)
                            temp_stats_list.append(temp_stats)
                    else:
                        for box in location_list:
                            temps = self.temp_analyzer.extract_temperature(temp_matrix, tuple(box))
                            temp_stats = self.temp_analyzer.analyze_temperature(temps)
                            temp_stats_list.append(temp_stats)
                    
                    frame = self.draw_temp_func(frame, location_list, temp_stats_list)
                
                out.write(frame)
                self.update_ui_signal.emit(cur_num, total)
            else:
                break
        # 释放资源
        cap.release()
        out.release()

    def stop(self):
        self.is_running = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

