# sports/base.py
from abc import ABC, abstractmethod


class BaseSport(ABC):
    """体测项目处理单元：一个类封装一个项目的完整处理链（推理 + 评测），一处注册。"""

    @abstractmethod
    def start(self):
        """加载模型与初始化状态（在用户进程内调用，耗时秒级）。"""

    @abstractmethod
    def process(self, frame, config, frame_id, uid=0):
        """处理一帧（在共享内存帧上就地绘制），返回评测数据 dict；本帧无有效数据时返回 None。"""

    @abstractmethod
    def stop(self):
        """释放模型与状态。"""
