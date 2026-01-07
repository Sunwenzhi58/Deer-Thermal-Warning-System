# 编码格式声明：指定文件编码为UTF-8，避免中文注释/字符串出现乱码
import cv2  # OpenCV库：核心计算机视觉工具（图像读取、绘制、视频处理等）
from PyQt5.QtGui import QPixmap, QImage  # PyQt5图形组件：用于将OpenCV图像转换为Qt显示格式（适用于GUI界面开发）
import numpy as np  # NumPy库：处理图像数组（像素矩阵运算、数据格式转换）
from PIL import Image, ImageDraw, ImageFont  # PIL(Pillow)库：解决OpenCV中文显示问题（OpenCV原生不支持中文文本绘制）
import csv  # CSV库：读写标注数据、实验结果等结构化数据（适用于鹿只高温预警系统中的数据记录）
import os  # OS库：文件路径操作、判断文件是否存在（用于数据集管理）


# 绘图展示函数：快速显示单张图像（适用于调试阶段验证标注框、图像处理结果）
def cv_show(name, img):
    cv2.imshow(name, img)  # 打开图像显示窗口，name为窗口名称，img为OpenCV格式图像（numpy数组）
    cv2.waitKey(0)  # 等待键盘输入：0表示无限等待（直到按下任意键关闭窗口），适用于静态图像查看
    cv2.destroyAllWindows()  # 关闭所有OpenCV创建的窗口，避免占用内存


def drawRectBox(image, rect, addText, fontC, color):
    """
    绘制带中文标签的矩形框（核心标注可视化函数，适用于目标检测结果展示）
    :param image: 原始图像（OpenCV格式，numpy数组，BGR通道顺序）
    :param rect: 矩形框坐标（int类型列表/元组，格式：[x1, y1, x2, y2]，x1y1为左上角，x2y2为右下角）
    :param addText: 标注文本（如"鹿只"、"高温预警"等类别名称）
    :param fontC: PIL字体对象（指定字体文件、字号，解决中文显示问题）
    :param color: 矩形框和文本背景色（BGR格式元组，如(0,255,0)为绿色）
    :return: imagex: 绘制后的图像（OpenCV格式，numpy数组）
    """
    # 1. 绘制目标矩形框：参数依次为图像、左上角坐标、右下角坐标、颜色、线条粗细（2像素）
    cv2.rectangle(image, (rect[0], rect[1]),
                  (rect[2], rect[3]),
                  color, 2)

    # 2. 绘制文本背景框：在矩形框上方绘制实心矩形（避免文本与图像背景混淆）
    # 背景框位置：左上角(x1-1, y1-25)，右下角(x1+60, y1)，-1表示实心填充，cv2.LINE_AA表示抗锯齿线条
    cv2.rectangle(image, (rect[0] - 1, rect[1] - 25), (rect[0] + 60, rect[1]), color, -1, cv2.LINE_AA)

    # 3. 转换图像格式：OpenCV图像（BGR）→ PIL图像（RGB），用于中文文本绘制
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)  # 创建PIL绘图对象
    # 绘制中文文本：位置(rect[0]+2, rect[1]-27)（文本背景框内居中），文本内容，白色字体(255,255,255)，指定字体
    draw.text((rect[0] + 2, rect[1] - 27), addText, (255, 255, 255), font=fontC)
    imagex = np.array(img)  # PIL图像 → OpenCV图像（numpy数组）
    return imagex


def img_cvread(path):
    """
    读取含中文路径的图像（解决OpenCV.imread()不支持中文路径的问题）
    :param path: 图像文件路径（可含中文，如"鹿只数据集/热成像_001.png"）
    :return: img: OpenCV格式图像（numpy数组，BGR通道）
    """
    # 改进读取方式：np.fromfile()以字节流读取文件，cv2.imdecode()解码为图像（cv2.IMREAD_COLOR表示读取彩色图像）
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img


def draw_boxes(img, boxes):
    """
    批量绘制矩形框（适用于多目标检测结果快速可视化，无文本标签）
    :param img: 原始图像（OpenCV格式）
    :param boxes: 矩形框列表，每个元素为[x1, y1, x2, y2]（int类型）
    :return: img: 绘制后的图像
    """
    for each in boxes:  # 遍历所有矩形框
        x1 = each[0]
        y1 = each[1]
        x2 = each[2]
        y2 = each[3]
        # 绘制绿色矩形框（(0,255,0)），线条粗细2像素
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img


def cvimg_to_qpiximg(cvimg):
    """
    OpenCV图像转换为PyQt5的QPixmap格式（适用于Qt GUI界面显示图像）
    :param cvimg: OpenCV格式图像（numpy数组，BGR通道）
    :return: qpix_img: QPixmap格式图像（Qt支持的显示格式）
    """
    height, width, depth = cvimg.shape  # 获取图像尺寸（高、宽、通道数，depth=3为彩色图）
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)  # 通道转换：BGR（OpenCV）→ RGB（Qt）
    # 创建QImage对象：参数依次为图像数据、宽度、高度、每行字节数（width*depth）、图像格式（RGB888=24位彩色）
    qimg = QImage(cvimg.data, width, height, width * depth, QImage.Format_RGB888)
    qpix_img = QPixmap(qimg)  # 转换为QPixmap（Qt中更适合显示的图像格式）
    return qpix_img


def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=50):
    """
    封装函数：在OpenCV图像上显示中文文本（独立于矩形框，适用于单独添加标题、备注等）
    :param img: 原始图像（OpenCV格式或PIL格式）
    :param text: 中文文本内容（如"鹿只高温预警系统"）
    :param position: 文本绘制位置（元组，(x, y)，x/y为文本左上角坐标）
    :param textColor: 文本颜色（BGR格式，默认绿色(0,255,0)）
    :param textSize: 字号（默认50）
    :return: 绘制中文后的图像（OpenCV格式）
    """
    if (isinstance(img, np.ndarray)):  # 判断图像是否为OpenCV格式（numpy数组）
        # 转换通道：BGR（OpenCV）→ RGB（PIL）
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)  # 创建PIL绘图对象
    # 加载中文字体：simsun.ttc（宋体，Windows系统默认自带，Linux/Mac需自行配置字体路径）
    try:
        fontStyle = ImageFont.truetype("simsun.ttc", textSize, encoding="utf-8")
    except:
        # 如果找不到字体，使用默认字体
        fontStyle = ImageFont.load_default()
    draw.text(position, text, textColor, font=fontStyle)  # 绘制中文文本
    # 转换回OpenCV格式：RGB（PIL）→ BGR（OpenCV）
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def insert_rows(path, lines, header):
    """
    批量写入数据到CSV文件（支持自动添加表头和序号，适用于保存实验结果、标注数据）
    :param path: CSV文件路径（如"鹿只高温预警结果.csv"）
    :param lines: 待写入数据列表，每个元素为一行数据（列表格式）
    :param header: CSV表头列表（如["序号", "日期", "检测数量", "高温预警数量"]）
    :return: 无返回值，直接写入文件
    """
    no_header = False  # 标记是否需要添加表头（文件不存在时需添加）
    if not os.path.exists(path):  # 判断文件是否存在
        no_header = True  # 文件不存在，需要添加表头
        start_num = 1  # 序号从1开始
    else:
        # 文件存在，序号从已有行数+1开始（避免序号重复）
        start_num = len(open(path, encoding='utf-8').readlines())

    csv_head = header  # 表头数据
    # 打开CSV文件：'a'表示追加模式（避免覆盖已有数据），newline=''避免空行
    with open(path, 'a', newline='', encoding='utf-8') as f:
        csv_write = csv.writer(f)  # 创建CSV写入对象
        if no_header:
            csv_write.writerow(csv_head)  # 写入表头

        for each_list in lines:  # 遍历每行数据
            each_list = [start_num] + each_list  # 给每行数据添加序号
            csv_write.writerow(each_list)  # 写入一行数据
            start_num += 1  # 序号自增


class Colors:
    """
    颜色调色板类（用于目标检测中不同类别绘制不同颜色，提升可视化区分度）
    参考YOLO系列的颜色配置，支持20种基础颜色和20种姿态估计颜色
    """

    def __init__(self):
        """初始化颜色：从16进制颜色码转换为RGB格式"""
        # 基础颜色16进制码（20种，涵盖鲜明对比色，适用于不同目标类别）
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        # 转换为RGB格式（PIL支持的顺序），存储为颜色列表
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)  # 基础颜色数量（20）
        # 姿态估计专用颜色（20种，适用于关键点绘制）
        self.pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                                      [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                                      [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                                      [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
                                     dtype=np.uint8)

    def __call__(self, i, bgr=False):
        """
        获取指定索引的颜色（支持BGR/RGB格式切换）
        :param i: 颜色索引（int，超出范围时自动循环）
        :param bgr: 是否返回BGR格式（默认False=RGB，True=BGR用于OpenCV绘制）
        :return: 颜色元组（如RGB格式(255,38,38)，BGR格式(38,38,255)）
        """
        c = self.palette[int(i) % self.n]  # 循环取色（避免索引越界）
        return (c[2], c[1], c[0]) if bgr else c  # BGR格式需反转RGB顺序

    @staticmethod
    def hex2rgb(h):  # 静态方法：16进制颜色码转换为RGB元组（PIL支持的顺序）
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


def yolo_to_location(w, h, yolo_data):
    """
    YOLO格式标注转换为像素坐标（两点式：x1,y1左上角 → x2,y2右下角）
    YOLO标注格式：[x_center, y_center, width, height]（均为图像宽高的归一化值，范围0~1）
    :param w: 图像宽度（像素值）
    :param h: 图像高度（像素值）
    :param yolo_data: YOLO格式标注列表，如[0.5, 0.5, 0.4, 0.6]（x_center=0.5*w，y_center=0.5*h等）
    :return: [x1, y1, x2, y2]：像素坐标的矩形框（int类型）
    """
    x_, y_, w_, h_ = yolo_data  # 解包YOLO归一化数据
    # 计算左上角x1：x_center*w - 0.5*width*w（width为归一化宽度）
    x1 = int(w * x_ - 0.5 * w * w_)
    # 计算右下角x2：x_center*w + 0.5*width*w
    x2 = int(w * x_ + 0.5 * w * w_)
    # 计算左上角y1：y_center*h - 0.5*height*h（height为归一化高度）
    y1 = int(h * y_ - 0.5 * h * h_)
    # 计算右下角y2：y_center*h + 0.5*height*h
    y2 = int(h * y_ + 0.5 * h * h_)
    return [x1, y1, x2, y2]  # 返回像素坐标矩形框


def location_to_yolo(w, h, locations):
    """
    像素坐标（两点式）转换为YOLO格式标注（归一化值）
    :param w: 图像宽度（像素值）
    :param h: 图像高度（像素值）
    :param locations: 像素坐标矩形框，如[x1, y1, x2, y2]
    :return: [x_center, y_center, width, height]：YOLO格式归一化值（保留5位小数）
    """
    x1, y1, x2, y2 = locations  # 解包像素坐标
    # 计算归一化中心x：(x1+x2)/2 / w（中心点x坐标除以图像宽度）
    x_ = (x1 + x2) / 2 / w
    x_ = float('%.5f' % x_)  # 保留5位小数
    # 计算归一化中心y：(y1+y2)/2 / h
    y_ = (y1 + y2) / 2 / h
    y_ = float('%.5f' % y_)  # 保留5位小数
    # 计算归一化宽度：(x2-x1)/w（矩形框宽度除以图像宽度）
    w_ = (x2 - x1) / w
    w_ = float('%.5f' % w_)  # 保留5位小数
    # 计算归一化高度：(y2-y1)/h
    h_ = (y2 - y1) / h
    h_ = float('%.5f' % h_)  # 保留5位小数
    return [x_, y_, w_, h_]  # 返回YOLO格式标注


def draw_yolo_data(img_path, yolo_file_path):
    """
    读取YOLO格式标注文件并在图像上绘制矩形框（验证标注是否正确）
    :param img_path: 图像文件路径
    :param yolo_file_path: YOLO标注文件路径（.txt格式，每行对应一个目标）
    """
    img = img_cvread(img_path)  # 读取图像（OpenCV格式，支持中文路径）
    h, w, _ = img.shape  # 获取图像高度、宽度（用于坐标转换）
    print(img.shape)  # 打印图像尺寸（调试用）

    # 读取YOLO标注文件（每行格式：class_id x_center y_center width height，空格分隔）
    with open(yolo_file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()  # 按行读取所有标注数据
        for each in data:  # 遍历每个目标的标注
            if each.strip():  # 跳过空行
                temp = each.split()  # 按空格分割字符串
                if len(temp) >= 5:
                    # 提取YOLO归一化坐标（跳过第一个元素class_id），转换为浮点数
                    x_, y_, w_, h_ = float(temp[1]), float(temp[2]), float(temp[3]), float(temp[4])
                    # 转换YOLO格式到像素坐标
                    x1, y1, x2, y2 = yolo_to_location(w, h, [x_, y_, w_, h_])
                    # 绘制红色矩形框（BGR格式(0,0,255)）验证标注
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 显示绘制后的图像
    cv2.imshow('windows', img)
    cv2.waitKey(0)  # 等待按键关闭窗口
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 测试代码
    img_path = 'datasets/images/train/frame_002.png'  # 测试图像路径
    yolo_file_path = 'datasets/labels/train/frame_002.txt'  # 对应YOLO标注文件路径
    draw_yolo_data(img_path, yolo_file_path)  # 调用函数绘制标注框

