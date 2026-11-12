# sports/situp.py
"""仰卧起坐：YOLO Pose 推理 + 状态机评测的链式装配（细节见 situp_infer / situp_eval）。"""
from .base import BaseSport
from .situp_infer import Inferencer
from .situp_eval import Evaluator


class SportSitup(BaseSport):
    """仰卧起坐：YOLO Pose 推理 + 状态机评测的链式封装。"""

    def __init__(self):
        self._inferencer = None
        self._evaluator = None
        self._face_client = None     # 人脸识别客户端（set_face_client 注入）

    def load(self):
        """加载模型与评测状态机（用户进程内调用，耗时秒级，每进程一次）。"""
        self._inferencer = Inferencer()
        self._evaluator = Evaluator(face_client=self._face_client)

    def reset(self):
        """换会话复位：重建评测状态机，推理模型保留（进程复用，ADR-0003）。"""
        self._evaluator = Evaluator(face_client=self._face_client)

    def process(self, task):
        """处理一帧（就地绘制），返回评测数据 dict；本帧无有效数据返回 None。"""
        if self._inferencer is None or self._evaluator is None:
            raise RuntimeError('SportSitup 未调用 load()')
        task.set_data(self._inferencer.process(task))
        return self._evaluator.process(task)

    def close(self):
        self._inferencer = None
        self._evaluator = None
