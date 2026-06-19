import json
import queue
import threading

import cv2

from api.base_sport import BaseSport
from api.ffmpeg_utils import FFmpegVideoHandler
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

from PIL import Image
import base64
import io

from api.new_pullup import PULL
from api.new_sitreach import SITREACH
from api.new_situp import SITUP
from api.new_jump import JUMP

sport_dict = {
    "situp": SITUP,
    "pullup": PULL,
    "sitreach": SITREACH,   # 此处并未用到，使用的是单独的接口和前端页面，800/1000米跑也是
    "jump": JUMP
}

def encode_pil_b64(painting):
    img = Image.fromarray(painting)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

class SportManager:

    """
    体育项目管理类，负责管理体育项目的初始化、启动、停止以及视频流处理等功能。
    """
    def __init__(self, uid, sport_type):

        self.uid = uid

        if sport_type not in sport_dict:
            raise ValueError("不支持的体育项目")

        self.sport: BaseSport = sport_dict[sport_type](uid)

        self.frame_id = 0

        self.thread = None

        self.data_queue = queue.Queue()

        self.ffmpeg_handler = FFmpegVideoHandler()

    def start(self):
        print("开始测试")
        self.sport.start()
        self.ffmpeg_handler.start()
        self.thread = threading.Thread(target=self.process_frames_loop)
        self.thread.start()
        print("开始测试-finish")

    def stop(self):
        print("结束测试")
        self.ffmpeg_handler.stop()
        self.thread.join()
        self.sport.stop()
        self.close_ws()
        print("结束测试-finish")

    def handle_video_stream(self, video_stream):

        """
        使用ffmpeg处理视频流
        """
        self.ffmpeg_handler.handle_video_stream(video_stream)  # 调用FFmpeg处理器处理视频流

    def process_frames_loop(self):
        """
        处理视频帧的循环函数
        持续从FFmpeg处理队列中获取帧，进行处理并通过WebSocket推送结果
        """
        while True:
            # 从FFmpeg处理队列中获取一帧
            frame = self.ffmpeg_handler.frame_queue.get()

            # 如果获取到None，表示处理结束，退出循环
            if frame is None:
                break

            # 处理帧数据
            result, painting = self.sport.update(frame, self.frame_id)
            self.frame_id += 1

            if result is not None and painting is not None:
                # 将BGR转换为RGB
                painting = cv2.cvtColor(painting, cv2.COLOR_BGR2RGB)

                self.push_to_ws(result, painting)


    def push_to_ws(self, result, painting):

        """
        通过WebSocket发送结果和绘画图像到指定用户
        :param result: 要发送的结果数据
        :param painting: PIL图像对象，将被编码为base64格式
        """
        channel_layer = get_channel_layer()  # 获取Django Channels的通道层
        img_b64_text = encode_pil_b64(painting)
        message = json.dumps({"result": result, "painting": img_b64_text})

        async_to_sync(channel_layer.group_send)(
            f"data_{self.uid}", {"type": "send_data", "text": message}
        )
    
    def close_ws(self):

        """
        关闭WebSocket连接的方法
        该方法会向指定用户组广播关闭事件，通知所有连接断开
        """
        channel_layer = get_channel_layer()  # 获取通道层实例，用于跨进程通信
        group_name = f"data_{self.uid}"  # 构建组名，基于用户ID，用于将用户分组
        # 向整个组广播踢人事件
        async_to_sync(channel_layer.group_send)(  # 将异步函数同步调用
            group_name,  # 目标组名，即特定用户的组
            {
                "type": "close_client",  # 对应consumer里的方法
            }
        )
