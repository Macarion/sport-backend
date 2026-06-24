import asyncio
import json
from aioice import Candidate
from av import VideoFrame
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.layers import get_channel_layer
import numpy as np
from api.user_manager import UserManager
from api.sport_manager import sport_dict
from aiortc.rtcicetransport import candidate_from_aioice
from aiortc import MediaStreamError, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack


class DataConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        kwargs = self.scope["url_route"]["kwargs"]
        self.sport_type = kwargs.get("sport_type")
        self.uid = kwargs.get("uid")

        self.group_name = f"data_{self.uid}"

        if self.sport_type not in sport_dict or self.uid is None:
            await self.close()
            return

        await self.channel_layer.group_add(self.group_name, self.channel_name)

        # 连接时的操作
        await self.accept()

    async def disconnect(self, close_code):
        # 离开消息组
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def receive(self, text_data=None, bytes_data=None):
        # 处理接收到的消息
        pass

    async def send_data(self, event):
        """组消息回调，推送数据到前端"""
        text_data = event["text"]
        # 直接发送文本，不需要外层type包装
        await self.send(text_data=str(text_data))

    async def close_client(self, text_data=None, bytes_data=None):
        await self.close()


# 自定义输出轨道：持续向后端发送处理后的图像帧给前端
class ProcessVideoTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.processed_frame = None  # 存储处理完成的画面

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        if self.processed_frame is None:
            # 无画面时返回黑色底
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            frame = self.processed_frame
        # 构造 aiortc VideoFrame
        video_frame = self.video_frame_from_ndarray(frame, time_base, pts)
        return video_frame

    def video_frame_from_ndarray(self, img_bgr, time_base, pts):
        # vf = VideoFrame(img_bgr.shape[0], img_bgr.shape[1], format="bgr24")
        vf = VideoFrame.from_ndarray(img_bgr, format="bgr24")
        vf.pts = pts
        vf.time_base = time_base
        return vf

    def push_frame(self, img_bgr):
        self.processed_frame = img_bgr


class WebRTCConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        self.uid = self.scope["url_route"]["kwargs"]["uid"]
        self.group_name = f"webrtc_{self.uid}"

        if self.uid is None:
            await self.close()
            return

        await self.channel_layer.group_add(self.group_name, self.channel_name)

        self.pc = RTCPeerConnection()

        # 全局输出轨道实例，用于向后推送处理后的画面
        output_track = ProcessVideoTrack()

        UserManager().set_output_track(self.uid, output_track)

        # 后端主动添加处理后的视频轨道，推送给前端
        self.pc.addTrack(output_track)

        # 收到视频轨道
        @self.pc.on("track")
        def on_track(track):

            if track.kind == "video":
                asyncio.create_task(self.handle_video(track))
        
        await self.accept()
    
    async def disconnect(self, code):
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def receive(self, text_data=None, bytes_data=None):

        data = json.loads(text_data)

        if data["type"] == "offer":

            offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])

            await self.pc.setRemoteDescription(offer)

            answer = await self.pc.createAnswer()
            await self.pc.setLocalDescription(answer)

            await self.send(
                text_data=json.dumps(
                    {"type": "answer", "sdp": self.pc.localDescription.sdp}
                )
            )

        elif data["type"] == "ice":

            cand = data.get("candidate")

            if not cand:
                return

            candidate = Candidate.from_sdp(cand.get("candidate"))
            icecandidate = candidate_from_aioice(candidate)
            icecandidate.sdpMid = cand.get("sdpMid")
            icecandidate.sdpMLineIndex = cand.get("sdpMLineIndex")
            await self.pc.addIceCandidate(icecandidate)

    async def handle_video(self, track):

        while True:

            try:
                frame = await track.recv()
            except MediaStreamError:
                break

            img = frame.to_ndarray(format="bgr24")

            # 直接进入AI逻辑
            UserManager().push_frame(self.uid, img)

    async def close_client(self, text_data=None, bytes_data=None):
        await self.pc.close()
        await self.close()