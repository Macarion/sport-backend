"""
已弃用
"""

import json
import queue
import threading

from api.base_sport import BaseSport
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

from multiprocessing.connection import PipeConnection

from api.new_pullup import PULL
from api.new_situp import SITUP
from api.new_jump import JUMP


class SportManager:
    """
    体育项目管理类，负责管理体育项目的初始化、启动、停止以及视频流处理等功能。
    """

    sport_dict = {
        "situp": SITUP,
        "pullup": PULL,
        "jump": JUMP,
        # "sitreach": SITREACH,   # 此处并未用到，使用的是单独的接口和前端页面，800/1000米跑也是
    }

    def __init__(self, uid, sport_type):

        self.uid = uid

        if not SportManager.check_sport_type(sport_type):
            raise ValueError("不支持的体育项目")

        self.sport: BaseSport = SportManager.sport_dict[sport_type](uid)

        self.frame_id = 0

        self.frame_thread = None
        self.frame_queue = queue.Queue(maxsize=30)

        self.result_thread = None
        self.result_queue = queue.Queue(maxsize=100)

        self.output_track = None

        self.running = False

        self.time = [0] * 10

        self.pipe_in: PipeConnection = None
        self.pipe_out: PipeConnection = None
    
    @staticmethod
    def check_sport_type(sport_type):
        return sport_type in SportManager.sport_dict

    def start(self):
        print("开始测试")
        self.running = True
        # self.pipe_in, self.pipe_out = self.sport.start()
        # if self.pipe_in is None or self.pipe_out is None:
        #     raise ValueError("启动失败: start() 函数返回值未包含 pipe_in 或 pipe_out")

        # print(self.pipe_in, self.pipe_out)

        # # self.frame_thread = threading.Thread(target=self.process_frames_loop, daemon=True)
        # # self.frame_thread.start()
        # self.result_thread = threading.Thread(target=self.handle_result, daemon=True)
        # self.result_thread.start()
        print("开始测试-finish")

    def stop(self):
        print("结束测试")
        self.running = False
        # self.result_queue.put(None)
        # self.frame_queue.put(None)

        # # self.frame_thread.join()
        # self.result_thread.join()

        # self.sport.stop()
        self.close_ws()
        print("结束测试-finish")

    def set_output_track(self, output_track):
        self.output_track = output_track

    def push_frame(self, frame):
        if self.frame_queue.full():
            self.frame_queue.get_nowait()
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass
    
    # 多进程版本
    def push_frame_multiprocess_version(self, frame):
        # 处理帧数据
        try:
            self.pipe_in.send({
                "frame": frame,
                "frame_id": self.frame_id,
                "drawing": True
            })
        except EOFError:
            pass
    
    # 多进程版本
    def handle_result(self, result):

        channel_layer = get_channel_layer()

        try:
            while self.running:
                result = self.pipe_out.recv()
                if result is None:
                    break

                frame = result["frame"]
                del result["frame"]

                if frame is not None and self.output_track:
                    self.output_track.push_frame(frame)

                async_to_sync(channel_layer.group_send)(
                    f"webrtc_{self.uid}", {"type": "send_data", "text": json.dumps(result)}
                )
        except EOFError:
            pass

    def process_frames_loop(self):
        """
        处理RTC视频帧
        """
        while self.running:
            frame = self.frame_queue.get()
            if frame is None:
                break

            result, painting = self.sport.update(frame, self.frame_id)
            self.frame_id += 1

            if result is not None:
                try:
                    self.result_queue.put_nowait(result)
                except queue.Full:
                    pass

            if painting is not None and self.output_track:
                self.output_track.push_frame(painting)

    def result_sender_loop(self):

        channel_layer = get_channel_layer()

        while self.running:

            result = self.result_queue.get()

            if result is None:
                break

            async_to_sync(channel_layer.group_send)(
                f"webrtc_{self.uid}", {"type": "send_data", "text": json.dumps(result)}
            )

    def close_ws(self):
        """
        关闭WebSocket连接的方法
        该方法会向指定用户组广播关闭事件，通知所有连接断开
        """
        channel_layer = get_channel_layer()  # 获取通道层实例，用于跨进程通信

        group_name = f"webrtc_{self.uid}"  # 构建组名，基于用户ID，用于将用户分组
        async_to_sync(channel_layer.group_send)(  # 将异步函数同步调用
            group_name,  # 目标组名，即特定用户的组
            {
                "type": "close_client",  # 对应consumer里的方法
            },
        )
