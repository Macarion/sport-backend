"""
引体向上 单项成绩报告 PDF 生成模块
美化版 - 包含仪表盘、等级对照表、训练建议
"""
from datetime import datetime
from io import BytesIO
from math import pi, cos, sin
import random
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor, white, black
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PyPDF4 import PdfFileWriter, PdfFileReader
from kivy.core.text import LabelBase
from matplotlib.font_manager import FontProperties
from .student_bulk import get_student_info_for_report

LabelBase.register(name='Font_HanZi', fn_regular='./msyh.TTF')
LabelBase.register(name='Font_HanZi_bd', fn_regular='./msyhbd.ttf')
Font_HanZi_bd = FontProperties(fname="./msyhbd.ttf")


def generate_id(existing_codes, length=12):
    """生成唯一的报告编号"""
    while True:
        code = ''.join(random.choices('0123456789', k=length))
        if code not in existing_codes:
            existing_codes.add(code)
            return code


class PullupPDFGenerator:
    """引体向上单项报告生成器 - 美化版"""
    
    def __init__(self):
        # 注册字体
        pdfmetrics.registerFont(TTFont('Font_HanZi', 'msyh.TTF'))
        pdfmetrics.registerFont(TTFont('Font_HanZi_bd', 'msyhbd.TTF'))
        pdfmetrics.registerFontFamily('STHanZi', normal='Font_HanZi', italic=None, boldItalic='Font_HanZi_bd')

        self.pdf_writer = PdfFileWriter()
        self.pdf_output = BytesIO()
        self.page_width, self.page_height = A4
        self.pdf_canvas = canvas.Canvas(self.pdf_output, pagesize=A4)

        # 初始化坐标
        self.x = 50
        self.y = self.page_height - 50
        
        # 配色方案 - 蓝色系
        self.primary_color = HexColor("#1565C0")      # 深蓝
        self.secondary_color = HexColor("#2196F3")    # 中蓝
        self.light_color = HexColor("#BBDEFB")        # 浅蓝
        self.accent_color = HexColor("#FF5722")       # 橙红强调
        self.bg_color = HexColor("#E3F2FD")           # 背景蓝

    def add_new_page(self):
        self.pdf_canvas.showPage()
        self.x = 50
        self.y = self.page_height - 50

    def check_page_space(self, height):
        if self.y - height < 50:
            self.add_new_page()

    def draw_header_banner(self):
        """绘制顶部装饰横幅"""
        # 蓝色渐变横幅
        self.pdf_canvas.setFillColor(self.primary_color)
        self.pdf_canvas.rect(0, self.page_height - 60, self.page_width, 60, fill=1, stroke=0)
        
        # 标题
        self.pdf_canvas.setFillColor(white)
        self.pdf_canvas.setFont("Font_HanZi_bd", 24)
        self.pdf_canvas.drawCentredString(self.page_width / 2, self.page_height - 40, "引体向上 单项成绩报告")
        
        self.y = self.page_height - 75

    def draw_info_card(self, name, sex, st_num, collage, st_class, report_id):
        """绘制用户信息卡片"""
        card_x = 40
        card_y = self.y - 60
        card_width = self.page_width - 80
        card_height = 55
        
        # 卡片背景
        self.pdf_canvas.setFillColor(self.bg_color)
        self.pdf_canvas.roundRect(card_x, card_y, card_width, card_height, 10, fill=1, stroke=0)
        
        # 左侧蓝色装饰条
        self.pdf_canvas.setFillColor(self.secondary_color)
        self.pdf_canvas.rect(card_x, card_y, 5, card_height, fill=1, stroke=0)
        
        # 信息文字
        self.pdf_canvas.setFillColor(black)
        self.pdf_canvas.setFont("Font_HanZi", 12)
        
        left_x = card_x + 20
        right_x = card_x + card_width / 2 + 20
        
        self.pdf_canvas.drawString(left_x, card_y + 38, f"姓名：{name}")
        self.pdf_canvas.drawString(left_x, card_y + 20, f"学校：{collage}")
        self.pdf_canvas.drawString(left_x, card_y + 2, f"班级：{st_class}")
        
        self.pdf_canvas.drawString(right_x, card_y + 38, f"性别：{sex}")
        self.pdf_canvas.drawString(right_x, card_y + 20, f"学号：{st_num}")
        self.pdf_canvas.drawString(right_x, card_y + 2, f"报告编号：{report_id}")
        
        self.y = card_y - 10

    def draw_score_gauge(self, score, level):
        """绘制分数仪表盘"""
        # 生成仪表盘图片
        fig, ax = plt.subplots(figsize=(3.5, 2.2), subplot_kw={'projection': 'polar'})
        
        # 设置为半圆
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        
        # 背景弧
        theta_bg = np.linspace(0, np.pi, 100)
        ax.fill_between(theta_bg, 0.6, 1.0, color='#E0E0E0', alpha=0.5)
        
        # 分数弧 - 根据分数计算角度
        score_angle = (score / 100) * np.pi
        theta_score = np.linspace(0, score_angle, 100)
        
        # 根据分数选择颜色
        if score >= 90:
            color = '#2196F3'  # 蓝色 - 优秀
        elif score >= 70:
            color = '#03A9F4'  # 浅蓝 - 良好
        elif score >= 60:
            color = '#FFC107'  # 黄色 - 良
        elif score >= 50:
            color = '#FF9800'  # 橙色 - 合格
        else:
            color = '#F44336'  # 红色 - 不合格
        
        ax.fill_between(theta_score, 0.6, 1.0, color=color, alpha=0.8)
        
        # 隐藏刻度
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.spines['polar'].set_visible(False)
        ax.grid(False)
        
        # 中间显示分数
        ax.text(np.pi/2, 0.2, f'{int(score)}', ha='center', va='center', 
                fontsize=36, fontweight='bold', color=color, fontproperties=Font_HanZi_bd)
        ax.text(np.pi/2, -0.15, '分', ha='center', va='center', 
                fontsize=14, color='#666666', fontproperties=Font_HanZi_bd)
        
        # 保存图片
        gauge_path = 'temp_gauge_pullup.png'
        plt.savefig(gauge_path, dpi=150, bbox_inches='tight', transparent=True)
        plt.close()
        
        # 绘制到PDF
        gauge_x = (self.page_width - 180) / 2
        gauge_y = self.y - 110
        self.pdf_canvas.drawImage(gauge_path, gauge_x, gauge_y, width=180, height=110)
        
        # 等级标签
        self.pdf_canvas.setFont("Font_HanZi_bd", 16)
        self.pdf_canvas.setFillColor(HexColor(color))
        self.pdf_canvas.drawCentredString(self.page_width / 2, gauge_y - 5, f"等级：{level}")
        
        self.y = gauge_y - 25
        
        # 删除临时文件
        Path(gauge_path).unlink(missing_ok=True)

    def draw_result_card(self, count, score, level):
        """绘制成绩结果卡片"""
        card_x = 40
        card_y = self.y - 70
        card_width = self.page_width - 80
        card_height = 60
        
        # 卡片背景
        self.pdf_canvas.setFillColor(white)
        self.pdf_canvas.setStrokeColor(self.light_color)
        self.pdf_canvas.setLineWidth(2)
        self.pdf_canvas.roundRect(card_x, card_y, card_width, card_height, 10, fill=1, stroke=1)
        
        # 三列布局
        col_width = card_width / 3
        
        items = [
            ("测试次数", f"{count} 个" if count else "未检测"),
            ("得分", f"{int(score)} 分"),
            ("等级", level)
        ]
        
        for i, (label, value) in enumerate(items):
            col_x = card_x + col_width * i + col_width / 2
            
            # 分隔线
            if i > 0:
                self.pdf_canvas.setStrokeColor(self.light_color)
                self.pdf_canvas.line(card_x + col_width * i, card_y + 15, 
                                    card_x + col_width * i, card_y + card_height - 15)
            
            # 标签
            self.pdf_canvas.setFillColor(HexColor("#666666"))
            self.pdf_canvas.setFont("Font_HanZi", 11)
            self.pdf_canvas.drawCentredString(col_x, card_y + 42, label)
            
            # 数值
            self.pdf_canvas.setFillColor(self.primary_color)
            self.pdf_canvas.setFont("Font_HanZi_bd", 16)
            self.pdf_canvas.drawCentredString(col_x, card_y + 18, value)
        
        self.y = card_y - 15

    def draw_standard_table(self):
        """绘制等级标准对照表"""
        # 标题
        self.pdf_canvas.setFillColor(self.primary_color)
        self.pdf_canvas.setFont("Font_HanZi_bd", 14)
        self.pdf_canvas.drawString(50, self.y, "📊 评分标准（男生）")
        
        self.y -= 25
        
        # 表格数据 - 引体向上男生标准
        headers = ["等级", "次数范围", "得分范围", "说明"]
        data = [
            ["优秀", "15次及以上", "90-100分", "上肢力量出色"],
            ["良好", "12-14次", "70-89分", "表现较好"],
            ["良", "10-11次", "60-69分", "达到中等水平"],
            ["合格", "8-9次", "50-59分", "达到基本要求"],
            ["不合格", "8次以下", "50分以下", "需要加强锻炼"],
        ]
        
        table_x = 50
        table_y = self.y
        col_widths = [70, 110, 90, 190]
        row_height = 22
        
        # 表头
        self.pdf_canvas.setFillColor(self.secondary_color)
        self.pdf_canvas.rect(table_x, table_y - row_height, sum(col_widths), row_height, fill=1, stroke=0)
        
        self.pdf_canvas.setFillColor(white)
        self.pdf_canvas.setFont("Font_HanZi_bd", 11)
        x_pos = table_x
        for i, header in enumerate(headers):
            self.pdf_canvas.drawCentredString(x_pos + col_widths[i]/2, table_y - 18, header)
            x_pos += col_widths[i]
        
        # 数据行
        for row_idx, row in enumerate(data):
            y_pos = table_y - row_height * (row_idx + 2)
            
            # 交替背景色
            if row_idx % 2 == 0:
                self.pdf_canvas.setFillColor(self.bg_color)
            else:
                self.pdf_canvas.setFillColor(white)
            self.pdf_canvas.rect(table_x, y_pos, sum(col_widths), row_height, fill=1, stroke=0)
            
            # 绘制文字
            self.pdf_canvas.setFillColor(black)
            self.pdf_canvas.setFont("Font_HanZi", 10)
            x_pos = table_x
            for i, cell in enumerate(row):
                self.pdf_canvas.drawCentredString(x_pos + col_widths[i]/2, y_pos + 8, cell)
                x_pos += col_widths[i]
        
        # 表格边框
        self.pdf_canvas.setStrokeColor(self.light_color)
        self.pdf_canvas.rect(table_x, table_y - row_height * 6, sum(col_widths), row_height * 6, fill=0, stroke=1)
        
        self.y = table_y - row_height * 6 - 15

    def draw_suggestion_box(self, remark, score, level):
        """绘制训练建议区域"""
        # 标题
        self.pdf_canvas.setFillColor(self.primary_color)
        self.pdf_canvas.setFont("Font_HanZi_bd", 14)
        self.pdf_canvas.drawString(50, self.y, "💪 训练建议")
        
        self.y -= 15
        
        box_x = 50
        box_width = self.page_width - 100
        box_height = 120  # 固定高度
        box_y = self.y - box_height  # 从当前位置往下
        
        # 建议框背景
        self.pdf_canvas.setFillColor(HexColor("#FFF3E0"))  # 淡橙色背景
        self.pdf_canvas.setStrokeColor(self.accent_color)
        self.pdf_canvas.setLineWidth(2)
        self.pdf_canvas.roundRect(box_x, box_y, box_width, box_height, 8, fill=1, stroke=1)
        
        # 左侧图标装饰
        self.pdf_canvas.setFillColor(self.accent_color)
        self.pdf_canvas.circle(box_x + 25, box_y + box_height/2, 15, fill=1, stroke=0)
        self.pdf_canvas.setFillColor(white)
        self.pdf_canvas.setFont("Font_HanZi_bd", 16)
        self.pdf_canvas.drawCentredString(box_x + 25, box_y + box_height/2 - 6, "!")
        
        # 建议文字
        self.pdf_canvas.setFillColor(black)
        self.pdf_canvas.setFont("Font_HanZi", 11)
        
        # 根据等级给出不同建议（五个等级）
        if level == "优秀":
            tips = [
                f"当前评价：{remark}",
                "• 继续保持，可尝试负重引体向上挑战更高难度",
                "• 建议每周训练3-4次，尝试不同握距全面发展",
                "• 上肢力量出色，可作为同学的训练榜样"
            ]
        elif level == "良好":
            tips = [
                f"当前评价：{remark}",
                "• 距离优秀仅差几个，冲刺15个以上达到满分",
                "• 建议每天练习2-3组，每组做到力竭",
                "• 配合俯卧撑、划船等动作综合提升上肢力量"
            ]
        elif level == "良":
            tips = [
                f"当前评价：{remark}",
                "• 已有较好基础，坚持训练可快速提升到良好",
                "• 建议每天练习2组，每组8-10个，逐步增加",
                "• 注意动作标准：下巴过杠，手臂完全伸直"
            ]
        elif level == "合格":
            tips = [
                f"当前评价：{remark}",
                "• 达到基本要求，继续努力向更高等级冲刺",
                "• 建议每天练习，可用弹力带辅助逐步减少依赖",
                "• 同时加强握力训练，如悬挂、握力器等"
            ]
        else:  # 不合格
            tips = [
                f"当前评价：{remark}",
                "• 需要加强上肢力量，从基础动作开始练习",
                "• 建议先做斜板引体或跳跃引体向上入门",
                "• 每天坚持悬挂30秒×3组，增强握力和背部力量"
            ]
        
        y_text = box_y + box_height - 25
        for tip in tips:
            self.pdf_canvas.drawString(box_x + 55, y_text, tip)
            y_text -= 22
        
        self.y = box_y - 20

    def draw_footer(self, timestamp):
        """绘制页脚"""
        # 底部装饰线
        self.pdf_canvas.setStrokeColor(self.light_color)
        self.pdf_canvas.setLineWidth(1)
        self.pdf_canvas.line(50, 45, self.page_width - 50, 45)
        
        # 时间戳
        self.pdf_canvas.setFillColor(HexColor("#999999"))
        self.pdf_canvas.setFont("Font_HanZi", 9)
        self.pdf_canvas.drawCentredString(self.page_width / 2, 28, f"报告生成时间：{timestamp}")
        self.pdf_canvas.drawCentredString(self.page_width / 2, 14, "智慧体测系统 · 北京交通大学")

    def get_score_and_remark(self, count):
        """根据引体向上次数计算分数和评价"""
        if count is None:
            return 0, "未检测", "暂无数据，请完成测试"
        
        count = int(count)
        # 引体向上按15次为满分（男生标准）
        percentage = (count / 15) * 100
        
        if percentage < 50:
            score = percentage
            level = "不合格"
            remark = "上肢力量需要加强，建议从辅助引体向上开始练习"
        elif 50 <= percentage < 60:
            score = percentage
            level = "合格"
            remark = "达到基本要求，可以逐步增加训练次数"
        elif 60 <= percentage < 70:
            score = percentage
            level = "良"
            remark = "表现不错，上肢力量有一定基础"
        elif 70 <= percentage < 90:
            score = percentage
            level = "良好"
            remark = "较为优秀，上肢力量较强，保持训练"
        else:
            score = min(100, percentage)
            level = "优秀"
            remark = "非常优秀！上肢力量出色，继续保持"
        
        return score, level, remark

    def save_pdf(self, filename):
        self.pdf_canvas.save()
        pdf_reader = PdfFileReader(self.pdf_output)
        for page in pdf_reader.pages:
            self.pdf_writer.addPage(page)
        with open(filename, "wb") as output_file:
            self.pdf_writer.write(output_file)

    def generate_report(self, username, pullup_count):
        """
        生成引体向上单项报告（美化版）
        
        Args:
            username: 用户名
            pullup_count: 引体向上次数
        """
        # 从 student_bulk 表获取用户真实信息
        student_info = get_student_info_for_report(username)
        name = student_info['name']
        sex = student_info['gender']
        st_num = student_info['student_id']
        collage = student_info['university']
        st_class = student_info['class_name']
        
        # 计算成绩
        score, level, remark = self.get_score_and_remark(pullup_count)
        
        # 生成报告编号
        existing_id = set()
        report_id = generate_id(existing_id)
        
        # === 开始生成 PDF ===
        
        # 1. 顶部横幅
        self.draw_header_banner()
        
        # 2. 用户信息卡片
        self.draw_info_card(name, sex, st_num, collage, st_class, report_id)
        
        # 3. 分数仪表盘
        self.draw_score_gauge(score, level)
        
        # 4. 成绩结果卡片
        self.draw_result_card(pullup_count, score, level)
        
        # 5. 等级标准表
        self.draw_standard_table()
        
        # 6. 训练建议
        self.draw_suggestion_box(remark, score, level)
        
        # 7. 页脚
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.draw_footer(timestamp)
        
        # 保存 PDF
        self.save_pdf("report_pdf/pullup_report.pdf")
        
        return "report_pdf/pullup_report.pdf"
