import asyncio
import json
import numpy as np
from aioice import Candidate
from av import VideoFrame
from channels.generic.websocket import AsyncWebsocketConsumer
from aiortc.rtcicetransport import candidate_from_aioice
from aiortc import MediaStreamError, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack

from config import FRAME_WIDTH, FRAME_HEIGHT, FRAME_CHANNELS, get_logger
from user_manager import user_manager

log = get_logger('WebSocket')

SHAPE = (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS)


class ProcessVideoTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.processed_frame = None

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        if self.processed_frame is None:
            frame = np.zeros(SHAPE, dtype=np.uint8)
        else:
            frame = self.processed_frame
        vf = VideoFrame.from_ndarray(frame, format="bgr24")
        vf.pts = pts
        vf.time_base = time_base
        return vf

    def push_frame(self, img_bgr):
        self.processed_frame = img_bgr


class WebRTCConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        self.uid = self.scope["url_route"]["kwargs"]["uid"]
        self.project = self.scope["url_route"]["kwargs"].get("project", "situp")
        self.group_name = f"webrtc_{self.uid}"

        if self.uid is None:
            await self.close()
            return

        await self.channel_layer.group_add(self.group_name, self.channel_name)

        self.pc = RTCPeerConnection()
        self.output_track = ProcessVideoTrack()
        self.pc.addTrack(self.output_track)

        @self.pc.on("track")
        def on_track(track):
            if track.kind == "video":
                asyncio.create_task(self.handle_video(track))

        await user_manager.register_consumer(self)

        await self.accept()

    async def disconnect(self, code):
        user_manager.unregister_consumer(self.uid)

        if hasattr(self, 'pc') and self.pc:
            await self.pc.close()

        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def eval_result(self, event):
        await self.send(text_data=json.dumps({
            "type": "eval_result",
            "data": event["data"],
        }))

    async def test_end(self, event):
        await self.send(text_data=json.dumps({
            "type": "test_end",
        }))
        await self.close()

    async def receive(self, text_data=None, bytes_data=None):
        data = json.loads(text_data)

        if data["type"] == "offer":
            offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
            await self.pc.setRemoteDescription(offer)
            answer = await self.pc.createAnswer()
            await self.pc.setLocalDescription(answer)
            await self.send(text_data=json.dumps({
                "type": "answer",
                "sdp": self.pc.localDescription.sdp
            }))

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
        loop = asyncio.get_event_loop()

        while True:
            try:
                frame = await track.recv()
            except MediaStreamError:
                break

            frame_nd = frame.to_ndarray(format="bgr24")
            frame_bytes = frame_nd.tobytes()

            await loop.run_in_executor(
                None, lambda: user_manager.push_frame(self.uid, frame_bytes)
            )

