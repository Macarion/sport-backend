from datetime import datetime
from io import BytesIO
from math import pi
import random
import matplotlib
matplotlib.use('Agg')              # ★ 必须放在 import pyplot 之前
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PyPDF4 import PdfFileWriter, PdfFileReader
from kivy.core.text import LabelBase
from matplotlib.font_manager import FontProperties
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.legends import Legend
from .student_bulk import get_student_info_for_report

LabelBase.register(name='Font_HanZi', fn_regular='./msyh.TTF')
LabelBase.register(name='Font_HanZi_bd', fn_regular='./msyhbd.ttf')
Font_HanZi_bd = FontProperties(fname="./msyhbd.ttf")

def generate_id(existing_codes, length=12):
    while True:
        # 生成一个12位随机数字序号
        code = ''.join(random.choices('0123456789', k=length))
        # 确保序号不重复
        if code not in existing_codes:
            existing_codes.add(code)
            return code

class PDFGenerator:
    def __init__(self):
        # 注册宋体字体
        pdfmetrics.registerFont(TTFont('Font_HanZi', 'msyh.TTF'))
        pdfmetrics.registerFont(TTFont('Font_HanZi_bd', 'msyhbd.TTF'))
        pdfmetrics.registerFontFamily('STHanZi', normal='Font_HanZi', italic=None, boldItalic='Font_HanZi_bd')

        # 创建一个新的PDF writer对象
        self.pdf_writer = PdfFileWriter()

        # 创建一个新的内存中的PDF文件对象
        self.pdf_output = BytesIO()

        self.page_width, self.page_height = A4

        # 创建一个新的画布对象以在PDF上绘制
        self.pdf_canvas = canvas.Canvas(self.pdf_output, pagesize=A4)

        # 初始化当前坐标
        self.x = 50
        self.y = self.page_height - 50

    def add_new_page(self):
        self.pdf_canvas.showPage()
        self.x = 50
        self.y = self.page_height - 50

    def check_page_space(self, height):
        if self.y - height < 50:
            self.add_new_page()

    def add_layout(self, layout, items, with_border=True):
        start_x = self.x
        start_y = self.y

        for item in items:
            item_type = item["type"]
            content = item["content"]
            width = item["width"]
            height = item["height"]
            font_size = item.get("font_size", 14)
            bold = item.get("bold", False)
            padding = item.get("padding", 10)

            if layout == "vertical":  # 上下排布
                # 检查当前页面剩余高度
                self.check_page_space(height + 10)
                x_start = self.x
                y_start = self.y - height

                if with_border:  # 绘制边框
                    self.pdf_canvas.rect(x_start, y_start, width, height, stroke=1, fill=0)

                if item_type == "text":
                    # 绘制文字
                    font_name = "Font_HanZi_bd" if bold else "Font_HanZi"
                    self.pdf_canvas.setFont(font_name, font_size)

                    lines = content.split('\n')
                    y = y_start + height - padding - font_size
                    for line in lines:
                        self.pdf_canvas.drawString(x_start + padding, y, line)
                        y -= font_size + 2

                elif item_type == "image":
                    # 绘制图片
                    self.pdf_canvas.drawImage(content, x_start + padding, y_start + padding,
                                              width=width - 2 * padding, height=height - 2 * padding)

                # 更新 `y`
                self.y = y_start - 10

            elif layout == "horizontal":  # 左右排布
                if self.x + width + 10 > self.page_width - 50:  # 如果超出页面宽度，则换行
                    # 换行到下一行，重置 x 位置，y 向下调整
                    self.x = start_x
                    self.y -= height + 10  # 调整y位置，使得在新行中排布

                    # 如果 y 位置不足，换页
                    if self.y < 50:
                        self.pdf_canvas.showPage()
                        self.x = start_x
                        self.y = self.page_height - 50

                # 确定当前绘制的 x, y 起始位置
                x_start = self.x
                y_start = self.y - height

                if with_border:  # 绘制边框
                    self.pdf_canvas.rect(x_start, y_start, width, height, stroke=1, fill=0)

                if item_type == "text":
                    # 绘制文字
                    font_name = "Font_HanZi_bd" if bold else "Font_HanZi"
                    self.pdf_canvas.setFont(font_name, font_size)

                    lines = content.split('\n')
                    y = y_start + height - padding - font_size
                    for line in lines:
                        self.pdf_canvas.drawString(x_start + padding, y, line)
                        y -= font_size + 2

                elif item_type == "image":
                    # 绘制图片
                    self.pdf_canvas.drawImage(content, x_start + padding, y_start + padding,
                                              width=width - 2 * padding, height=height - 2 * padding)

                # 更新 `x`
                self.x += width + 10  # 当前绘制后更新 x 位置，准备下一个元素

            else:
                raise ValueError("Invalid layout type: Choose 'vertical' or 'horizontal'")

        # 如果是上下排布，重置 `x`
        if layout == "vertical":
            self.x = start_x

        if layout == "horizontal":
            self.x = 50

    def _add_text(self, text, alignment="left", font_size=14, bold=False, line_spacing=18, char_spacing=0):
        # 设置字体为宋体
        font_name = "Font_HanZi_bd" if bold else "Font_HanZi"
        self.pdf_canvas.setFont(font_name, font_size)

        text_width = sum(
            self.pdf_canvas.stringWidth(char, font_name, font_size) + char_spacing
            for char in text
        ) - char_spacing
        if alignment == "center":
            self.x = (self.page_width - text_width) / 2
        elif alignment == "right":
            self.x = self.page_width - text_width - 50
        else:  # 默认靠左对齐
            self.x = 50

        # 计算文本高度并检查是否适合页面
        if self.y - line_spacing < 50:  # 到达页面底部
            self.pdf_canvas.showPage()
            self.pdf_canvas.setFont(font_name, font_size)  # 重新设置字体
            self.y = self.page_height - 50

        for char in text:
            self.pdf_canvas.drawString(self.x, self.y, char)
            self.x += self.pdf_canvas.stringWidth(char, font_name, font_size) + char_spacing
        self.y -= line_spacing

    def add_text(self, text, alignment="left", font_size=14, bold=False, line_spacing=18, char_spacing=0):
        # 将文本分割成行并逐行添加
        for line in text.split('\n'):
            self._add_text(line, alignment, font_size, bold, line_spacing, char_spacing)

    def _add_image(self, image_path, img_width=400, img_height=300, align="left"):
        try:
            if self.y - img_height < 50:
                self.pdf_canvas.showPage()
                self.y = 750

            if align == "right":
                x_position = self.x
            elif align == "center":
                page_width = self.pdf_canvas._pagesize[0]
                x_position = (page_width - img_width) / 2
            elif align == "left":
                page_width = self.pdf_canvas._pagesize[0]
                x_position = page_width - img_width - self.x + 200
            else:
                raise ValueError(f"Invalid alignment option: {align}")

            self.pdf_canvas.drawImage(image_path, x_position, self.y - img_height - 50, width=img_width, height=img_height)
            self.y -= img_height
        except Exception as e:
            print(f"Error adding image {image_path}: {e}")

    def add_image(self, image_path, img_width=400, img_height=300, align="left"):
        self._add_image(image_path, img_width, img_height, align)

    def save_pdf(self, filename):
        # 将画布保存到PDF文件
        self.pdf_canvas.save()

        # 从输出文件中读取PDF
        pdf_reader = PdfFileReader(self.pdf_output)

        # 将报告页面添加到PDF writer对象中
        for page in pdf_reader.pages:
            self.pdf_writer.addPage(page)

        # 将更新后的PDF保存到文件中
        with open(filename, "wb") as output_file:
            self.pdf_writer.write(output_file)

    def add_table(self, data, col_widths=None, row_height=20, font_size=14, bold=False, table_bg_color="#87CEEB",
                  font_color="#000000", alpha=0.5):
        font_name = "Font_HanZi_bd" if bold else "Font_HanZi"
        self.pdf_canvas.setFont(font_name, font_size)

        # 如果未提供列宽，则动态调整
        if not col_widths:
            col_widths = [max(self.pdf_canvas.stringWidth(str(cell), font_name, font_size) for cell in col) + 10 for col
                          in zip(*data)]

        # 计算表格总宽度
        table_width = sum(col_widths)
        table_height = len(data) * row_height

        # 检查是否需要分页
        if self.y - table_height < 50:
            self.pdf_canvas.showPage()
            self.y = 750

        # 表格起始坐标
        x_start = (A4[0] - table_width) / 2
        y_start = self.y - row_height

        # 将背景色转换为 RGB 格式，并设置透明度
        r_bg, g_bg, b_bg = self.hex_to_rgb(table_bg_color)

        # 绘制表格单元格和内容
        for i, row in enumerate(data):
            for j, cell in enumerate(row):
                x = x_start + sum(col_widths[:j])
                y = y_start - (i * row_height)

                # 绘制背景色（带透明度）
                self.pdf_canvas.setFillColorRGB(r_bg, g_bg, b_bg, alpha)  # 设置背景色填充，透明度为 alpha
                self.pdf_canvas.rect(x, y, col_widths[j], row_height, fill=1)  # 填充背景色，fill=1 表示填充

                # 绘制单元格边框（边框使用字体颜色，不受透明度影响）
                self.pdf_canvas.setStrokeColor(font_color)  # 设置边框颜色为字体颜色
                self.pdf_canvas.setLineWidth(0.5)  # 设置边框线宽
                self.pdf_canvas.rect(x, y, col_widths[j], row_height, fill=0)  # 只绘制边框，不填充
                self.pdf_canvas.setFillColor(font_color)  # 设置字体颜色为黑色

                # 绘制单元格内容（垂直居中）
                text_x = x + col_widths[j] / 2
                text_y = y + row_height / 2 - font_size / 2
                self.pdf_canvas.drawCentredString(text_x, text_y, str(cell))

        # 更新 y 坐标
        self.pdf_canvas.setFillColor(font_color)  # 设置字体颜色为黑色
        self.y -= table_height + 30
        self.x = 50

    def hex_to_rgb(self, hex_color):
        """
        将十六进制颜色转换为 RGB 格式
        """
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        return r / 255.0, g / 255.0, b / 255.0

    def add_spider(self, data, code):
        # number of variable
        categories = list(data)[1:]
        N = len(categories)
        values = data.loc[0].drop('group').values.flatten().tolist()
        values += values[:1]
        print(values)

        # 设置每个点的角度值
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        # Initialise the spider plot
        plt.clf()
        ax = plt.subplot(111, polar=True)
        plt.xticks(angles[:-1], categories, fontproperties=Font_HanZi_bd, color='black', size=20)
        ax.set_rlabel_position(0)
        plt.yticks([30, 60, 90, 100], ["30", "60", "90", ''], color="black", size=10)
        plt.ylim(0, 100)
        ax.plot(angles, values, linewidth=1, linestyle='solid')
        ax.fill(angles, values, facecolor='#F08080', edgecolor='#000000')

        plt.savefig(str(code)+".png", dpi=300, bbox_inches="tight", transparent=True)

    def add_text_remark(self, text, box_width, box_height, font_size=14, bold=False, padding=10):
        # 选择字体
        font_name = "Font_HanZi_bd" if bold else "Font_HanZi"
        self.pdf_canvas.setFont(font_name, font_size)

        # 检查是否需要换页
        if self.y - box_height < 50:
            self.pdf_canvas.showPage()
            self.y = 750

        # 计算方框的起始位置
        x_start = self.x
        y_start = self.y - box_height - 250

        # 绘制方框
        self.pdf_canvas.rect(x_start, y_start, box_width, box_height, stroke=1, fill=0)

        # 分割文字为多行
        lines = text.split('\n')

        # 每行文字的最大宽度
        max_line_width = box_width - 2 * padding

        # 逐行绘制文字
        y = y_start + box_height - padding - font_size  # 初始 y 坐标（顶部留白）
        for line in lines:
            # 处理行宽超出方框的情况，自动换行
            words = line.split(' ')
            current_line = ""
            for word in words:
                if self.pdf_canvas.stringWidth(current_line + word + " ", font_name, font_size) > max_line_width:
                    # 绘制当前行文字
                    self.pdf_canvas.drawString(x_start + padding, y, current_line)
                    y -= font_size + 2  # 下一行
                    current_line = word + " "
                else:
                    current_line += word + " "
            # 绘制最后的文字
            if current_line:
                self.pdf_canvas.drawString(x_start + padding, y, current_line)
                y -= font_size + 2  # 下一行

        # 更新 y 坐标
        self.y = y_start - 10  # 留点额外空间

    def remark_v(self, percentage):
        if percentage is None:
            remark = "未检测"
        elif percentage < 50:
            remark = "需要更多努力!"
        elif 50 <= percentage < 60:
            remark = "通过了，但仍需改进。"
        elif 60 <= percentage < 70:
            remark = "表现不错，有一定基础。"
        elif 70 <= percentage < 90:
            remark = "较为优秀，保持稳定状态。"
        else:
            remark = "非常优秀，表现突出！"
        return remark

    def draw_text_once(self, text, x=None, y=None, alignment="left", font_size=12, bold=False, char_spacing=0):
        font_name = "Font_HanZi_bd" if bold else "Font_HanZi"
        self.pdf_canvas.setFont(font_name, font_size)

        if x is None or y is None:
            return  # 你必须明确指定坐标位置

        text_width = sum(
            self.pdf_canvas.stringWidth(char, font_name, font_size) + char_spacing
            for char in text
        ) - char_spacing

        if alignment == "center":
            x -= text_width / 2
        elif alignment == "right":
            x -= text_width

        for char in text:
            self.pdf_canvas.drawString(x, y, char)
            x += self.pdf_canvas.stringWidth(char, font_name, font_size) + char_spacing

    def pdf_report_gen(self, usr, situp_numf):
        # 从 student_bulk 表获取用户真实信息
        student_info = get_student_info_for_report(usr)
        name = student_info['name']
        sex = student_info['gender']
        st_num = student_info['student_id']
        collage = student_info['university']
        st_class = student_info['class_name']
        height, weight, v_cap, sit_up, pull_up, long_jump, s_and_r, long_dis_race, fif_m = None, None, None, situp_numf, None, None, None, None, None
        remark_none = '未检测'
        BMI = ''
        sit_up_remark = ''
        a = [['' for _ in range(5)] for _ in range(8)]
        for i in range(0, 8):
            for j in range(0, 1):
                a[i][j] = ''

        if height is None or weight is None:
            BMI = "未检测"
        if v_cap is None:
            v_cap = "未检测"
        if long_jump is None:
            long_jump = "未检测"
        if s_and_r is None:
            s_and_r = "未检测"
        if long_dis_race is None:
            long_dis_race = "未检测"
        if fif_m is None:
            fif_m = "未检测"

        if sex == '女':
            num = pull_up
        else:
            num = sit_up
        # count = len([x for x in grade if x is not None]) - 2
        count=1

        if sit_up is not None:
            sit_up = int(sit_up)
            percentage = (sit_up / 60) * 100
            if percentage < 50:
                sit_up_score = percentage
                sit_up_remark = "不合格"
            elif 50 <= percentage < 60:
                sit_up_score = percentage
                sit_up_remark = "合格"
            elif 60 <= percentage < 70:
                sit_up_score = percentage
                sit_up_remark = "良"
            elif 70 <= percentage < 90:
                sit_up_score = percentage
                sit_up_remark = "良好"
            else:
                sit_up_score = min(100, percentage)
                sit_up_remark = "优秀"
        elif pull_up is not None:
            pull_up = int(pull_up)
            percentage = (pull_up / 60) * 100
            if percentage < 50:
                sit_up_score = percentage
                sit_up_remark = "不合格"
            elif 50 <= percentage < 60:
                sit_up_score = percentage
                sit_up_remark = "合格"
            elif 60 <= percentage < 70:
                sit_up_score = percentage
                sit_up_remark = "良"
            elif 70 <= percentage < 90:
                sit_up_score = percentage
                sit_up_remark = "良好"
            else:
                sit_up_score = min(100, percentage)
                sit_up_remark = "优秀"
        else:
            sit_up_score = 0
            num = "未检测"
        remark0 = PDFGenerator.remark_v(self, sit_up_score)

        if count == 0:
            total_score = 0
        else:
            total_score = (sit_up_score)/count
        if total_score < 50:
            total_remark = "不合格"
        elif 50 <= total_score < 60:
            total_remark = "合格"
        elif 60 <= total_score < 70:
            total_remark = "良"
        elif 70 <= total_score < 90:
            total_remark = "良好"
        else:
            total_score = min(100, total_score)
            total_remark = "优秀"
        remark_f = PDFGenerator.remark_v(self, total_score)

        if sit_up_remark == " 优秀":
            a[4][1] = a[4][2] = a[4][3] = a[4][0] = ''
            a[4][4] = '√'
        elif sit_up_remark == "良好":
            a[4][1] = a[4][2] = a[4][0] = a[4][4] = ''
            a[4][3] = '√'
        elif sit_up_remark == "良":
            a[4][1] = a[4][0] = a[4][3] = a[4][4] = ''
            a[4][2] = '√'
        elif sit_up_remark == "合格":
            a[4][0] = a[4][2] = a[4][3] = a[4][4] = ''
            a[4][1] = '√'
        elif sit_up_remark == "不合格":
            a[4][1] = a[4][2] = a[4][3] = a[4][4] = ''
            a[4][0] = '√'

        if total_remark == " 优秀":
            a[7][1] = a[7][2] = a[7][3] = a[7][0] = ''
            a[7][4] = '√'
        elif total_remark == "良好":
            a[7][1] = a[7][2] = a[7][0] = a[7][4] = ''
            a[7][3] = '√'
        elif total_remark == "良":
            a[7][1] = a[7][0] = a[7][3] = a[7][4] = ''
            a[7][2] = '√'
        elif total_remark == "合格":
            a[7][0] = a[7][2] = a[7][3] = a[7][4] = ''
            a[7][1] = '√'
        elif total_remark == "不合格":
            a[7][0] = '√'
            a[7][1] = a[7][2] = a[7][3] = a[7][4] = ''

        existing_id = set()
        id = generate_id(existing_id)
        pdf_generator = PDFGenerator()

        # 添加文本
        pdf_generator.add_text("体 测 成 绩 分 析 报 告", alignment="center", font_size=28, bold=True, line_spacing=24)
        pdf_generator.add_text("编号:"+str(id), alignment="right", line_spacing=24)
        b="   "
        pdf_generator.add_text("姓 名:"+name+b+"   性 别:"+sex+b+"      学 号:"+st_num+"\n"+"学 校:"+collage+b+"班 级:"+st_class, char_spacing=5, line_spacing=24)

        # 添加表格
        data = [["分类", "测试项目", "测试结果", "不合格", " 合格 ", "   良   ", " 良好 ", " 优秀 "],
                ["形态", "身高(厘米)/体重(千克)", str(BMI), str(a[0][0]), str(a[0][1]), str(a[0][2]), str(a[0][3]), str(a[0][4])],
                ["机能", "肺活量(毫升)", str(v_cap), str(a[1][0]), str(a[1][1]), str(a[1][2]), str(a[1][3]), str(a[1][4])],
                ["素质", "立定跳远(厘米)", str(long_jump), str(a[2][0]), str(a[2][1]), str(a[2][2]), str(a[2][3]), str(a[2][4])],
                ["素质", "坐位体前屈(厘米)", str(s_and_r), str(a[3][0]), str(a[3][1]), str(a[3][2]), str(a[3][3]), str(a[3][4])],
                ["素质", "仰卧起坐/引体向上(个)", str(situp_numf), str(a[4][0]), str(a[4][1]), str(a[4][2]), str(a[4][3]), str(a[4][4])],
                ["素质", "1000/800米(秒)",str(long_dis_race), str(a[5][0]), str(a[5][1]), str(a[5][2]), str(a[5][3]), str(a[5][4])],
                ["素质", "50米(秒)", str(fif_m), str(a[6][0]), str(a[6][1]), str(a[6][2]), str(a[6][3]), str(a[6][4])],
                ["综合评价", "测试项目(个)", str(count), "", "", "", "", ""],
                ["综合评价", "总分", int(total_score), str(a[7][0]), str(a[7][1]), str(a[7][2]), str(a[7][3]), str(a[7][4])],
                ]
        pdf_generator.add_table(data, row_height=25, font_size=14, bold=False)

        sit_up_score = int((int(situp_numf) / 60) * 100)
        data = pd.DataFrame({
            'group': ['成绩（百分）'],
            'BMI': [5],
            '肺活量': [5],
            '仰卧/引体': [max(5, sit_up_score)],
            '立定跳远': [5],
            '坐位体前屈': [5],
            '50米': [5],
            '1000/800米': [5]
        })

        pdf_generator.add_spider(data, id)
        plt_path = Path(str(id)+'.png')
        item_remark = "BMI："+remark_none+"\n"+"\n"+\
                      "肺活量："+remark_none+"\n"+"\n"+\
                      "仰卧起坐/引体向上："+ remark0 +"\n"+"\n"\
                      +"立定跳远："+remark_none+"\n"+"\n"\
                      +"坐位体前屈："+remark_none+"\n"+"\n"\
                      +"50米："+remark_none+"\n"+"\n"+\
                      "1000/800米："+remark_none+"\n"
        h_items = [
            {"type": "image", "content": plt_path, "width": 264, "height": 242},
            {"type": "text", "content": item_remark, "width": 207, "height": 242},
        ]
        pdf_generator.add_layout("horizontal", h_items, with_border=False)

        t_remark = "综合评价：\n" + remark_f
        pdf_generator.add_text_remark(t_remark, box_width=500, box_height=100, font_size=14, bold=False, padding=10)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf_generator.draw_text_once("报告生成时间：" + timestamp, x=310, y=self.y -750, alignment="center",
                            font_size=10)

        pdf_generator.save_pdf("report_pdf/my_report.pdf")
        plt_path.unlink()


