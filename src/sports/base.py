from abc import ABC, abstractmethod
from core.task import SportTask


class BaseSport(ABC):
    """体测项目处理单元：一个类封装一个项目的完整处理链（推理 + 评测），一处注册。

    四段生命周期（见 docs/adr/0003-process-reuse-pool.md）：
      load  — 进程级：加载模型与初始化，每进程一次（耗时秒级）
      reset — 会话级：换用户复位评测状态，模型保留，每个会话一次（含首次）
      process — 帧级：处理一帧
      close — 进程级：释放，每进程一次
    """

    @abstractmethod
    def load(self):
        """加载模型与初始化状态（在用户进程内调用，耗时秒级，每进程仅一次）。"""

    @abstractmethod
    def reset(self):
        """换会话复位评测状态（保留已加载模型，供下一个用户复用）。"""

    @abstractmethod
    def process(self, task: SportTask):
        """处理一帧（在共享内存帧上就地绘制），返回评测数据 dict；本帧无有效数据时返回 None。"""

    @abstractmethod
    def close(self):
        """释放模型与状态。"""

    def set_face_client(self, face_client):
        """注入人脸识别客户端（worker 启动时调用；不接人脸的项目自然无感）。"""
        self._face_client = face_client
